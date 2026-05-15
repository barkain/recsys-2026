#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 3 full 5-fold training (CUDA/CPU, single-fold or 5-fold).

Usage:
  # CPU local, all folds
  uv run python scripts/expR54_phase3_full5fold_train.py --device cpu

  # Single fold on CUDA (Colab)
  uv run python scripts/expR54_phase3_full5fold_train.py --fold 1 --device cuda

  # All folds CUDA
  uv run python scripts/expR54_phase3_full5fold_train.py --device cuda

  # Custom output dir
  uv run python scripts/expR54_phase3_full5fold_train.py --fold 2 --device cuda \
      --output-dir /content/drive/MyDrive/r54_phase3

Each fold:
- Train on dev folds {0..4}\{f} (~6,400 pairs) + 20,000 train-split sample
  (seed=0, session-balanced max 2/session)
- Structured query, R21 track text, BGE-base, same Phase 2 hyperparams, 1 epoch
- Save fold model + retrieval lists with cosine scores
- Delete fold-local checkpoints after fold completes

--skip-completed (default ON) avoids redoing folds whose oof_lists.json exists.

Fold-0 fallback: if --use-smoke-fold0 and SMOKE_FOLD0 exists locally, reuse it
(applies only when --fold 0 or --fold not specified, in local CPU runs).

Aggregated output (5-fold mode): {output_dir}/oof_r54_lists.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
SMOKE_FOLD0 = REPO / "cache" / "r54" / "phase3_smoke" / "fold_0"
DEFAULT_OUTPUT_DIR = REPO / "cache" / "r54" / "phase3_full"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
TAU = 0.05
MAX_SEQ_LEN = 256
TOPK = 300

TRAIN_SPLIT_SAMPLE_PAIRS = 20000
MAX_PAIRS_PER_SESSION = 2
SAMPLING_SEED = 0
TRAINING_SEED = 0

CKPT_EVERY = 200
LOG_EVERY = 50
KEEP_CKPTS = 2


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def build_track_text(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def build_short_track_ref(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = artists[0] if isinstance(artists, list) and artists else str(artists)
    return f"{name} by {artist}"


def build_query_structured_from_dev(case, meta):
    user_utterances = []
    played_tracks = []
    for h in case["history"]:
        role = h.get("role", "")
        content = str(h.get("content", ""))
        if role == "user":
            user_utterances.append(content)
        elif role == "music":
            tid = content.strip()
            if tid in meta:
                played_tracks.append(build_short_track_ref(tid, meta))
    current_query = case["user_query"]
    history = user_utterances[-3:]
    context_tracks = played_tracks[-5:]
    parts = [f"[QUERY] {current_query}"]
    if history:
        parts.append(f"[HISTORY] {' '.join(history)}")
    if context_tracks:
        parts.append(f"[CONTEXT] {'; '.join(context_tracks)}")
    return " ".join(parts)


def build_query_structured_from_session(user_msgs_so_far, played_so_far, current_user_msg, meta):
    history = user_msgs_so_far[-3:] if len(user_msgs_so_far) > 3 else user_msgs_so_far
    context_tracks = []
    for tid in played_so_far[-5:]:
        if tid in meta:
            context_tracks.append(build_short_track_ref(tid, meta))
    parts = [f"[QUERY] {current_user_msg}"]
    older_history = history[:-1] if history else []
    if older_history:
        parts.append(f"[HISTORY] {' '.join(older_history)}")
    if context_tracks:
        parts.append(f"[CONTEXT] {'; '.join(context_tracks)}")
    return " ".join(parts)


def load_catalog():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if matches:
        ds = Dataset.from_file(str(matches[-1]))
    else:
        from datasets import load_dataset  # type: ignore[reportMissingImports]
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
    cols = ds.to_dict()
    meta = {}
    track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta, track_ids


def build_train_split_sample(meta, dev_session_ids):
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    try:
        train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                                download_config=DownloadConfig(local_files_only=True))["train"]
    except Exception:
        train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]

    session_pairs = defaultdict(list)
    n_dev_overlap = 0
    for item in train_ds:
        sid = item["session_id"]
        if sid in dev_session_ids:
            n_dev_overlap += 1
            continue
        convs = item["conversations"]
        user_msgs_so_far = []
        played_so_far = []
        most_recent_user_msg = ""
        for conv in convs:
            role = conv["role"]
            content = str(conv["content"])
            if role == "user":
                user_msgs_so_far.append(content)
                most_recent_user_msg = content
            elif role == "music":
                tid = content.strip()
                if tid not in meta:
                    played_so_far.append(tid)
                    continue
                if most_recent_user_msg:
                    q = build_query_structured_from_session(user_msgs_so_far, played_so_far,
                                                             most_recent_user_msg, meta)
                    t = build_track_text(tid, meta)
                    session_pairs[sid].append((q, t))
                played_so_far.append(tid)

    rng = np.random.RandomState(SAMPLING_SEED)
    capped_pool = []
    for sid in sorted(session_pairs.keys()):
        pairs = session_pairs[sid]
        if len(pairs) <= MAX_PAIRS_PER_SESSION:
            capped_pool.extend((sid, q, t) for q, t in pairs)
        else:
            idx = rng.choice(len(pairs), MAX_PAIRS_PER_SESSION, replace=False)
            capped_pool.extend((sid, pairs[j][0], pairs[j][1]) for j in idx)

    if len(capped_pool) <= TRAIN_SPLIT_SAMPLE_PAIRS:
        sampled = capped_pool
    else:
        idx = rng.choice(len(capped_pool), TRAIN_SPLIT_SAMPLE_PAIRS, replace=False)
        sampled = [capped_pool[i] for i in idx]
    sampled_sessions = {s for s, _, _ in sampled}
    print(f"  train_split: {n_dev_overlap} dev overlap, {len(sampled)} pairs "
          f"from {len(sampled_sessions)} sessions", flush=True)
    return [(q, t) for _, q, t in sampled]


def list_ckpts(ckpt_dir):
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("ckpt_batch_*.pkl"),
                  key=lambda p: int(p.stem.split("_")[-1]))


def save_checkpoint(ckpt_dir, model, optimizer, batch_idx, perm, total_batches,
                    epoch_loss, n_batches_seen, device):
    import torch  # type: ignore[reportMissingImports]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_batch_{batch_idx:06d}.pkl"
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "batch_idx": batch_idx, "total_batches": total_batches,
        "perm": perm.tolist(),
        "model_state_dict": cpu_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "device": device,
        "epoch_loss": epoch_loss, "n_batches_seen": n_batches_seen,
        "created_at": datetime.now().isoformat(),
    }
    tmp_path = ckpt_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)  # noqa: S301
    tmp_path.rename(ckpt_path)
    all_ckpts = list_ckpts(ckpt_dir)
    for old in all_ckpts[:-KEEP_CKPTS]:
        old.unlink()


def train_fold(fold_idx, all_pairs, model_dir, ckpt_dir, device):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  fold {fold_idx}: {len(all_pairs)} pairs, {total_batches} batches, device={device}",
          flush=True)

    np.random.seed(TRAINING_SEED + fold_idx)
    torch.manual_seed(TRAINING_SEED + fold_idx)
    if device == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED + fold_idx)

    ckpts = list_ckpts(ckpt_dir)
    start_batch = 0
    perm = None
    epoch_loss = 0.0
    n_batches_seen = 0
    model_state = None
    optimizer_state = None

    if ckpts:
        latest = ckpts[-1]
        print(f"  resuming from {latest.name}", flush=True)
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        if ckpt["total_batches"] == total_batches:
            start_batch = ckpt["batch_idx"]
            perm = np.array(ckpt["perm"], dtype=np.int64)
            epoch_loss = ckpt["epoch_loss"]
            n_batches_seen = ckpt["n_batches_seen"]
            model_state = ckpt["model_state_dict"]
            optimizer_state = ckpt["optimizer_state_dict"]
            np.random.set_state(ckpt["numpy_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            print(f"  resumed at batch {start_batch}/{total_batches}", flush=True)

    if perm is None:
        perm = np.random.permutation(len(all_pairs)).astype(np.int64)

    model = SentenceTransformer(MODEL_NAME, device=device)
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    model.train()
    tokenizer = model.tokenizer

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
                            return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        out = model.forward(encoded)
        emb = out["sentence_embedding"]
        return F_t.normalize(emb, dim=-1)

    t_epoch = time.time()
    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * BATCH_SIZE
        batch_indices = perm[start:start + BATCH_SIZE]
        queries = [all_pairs[int(i)][0] for i in batch_indices]
        positives = [all_pairs[int(i)][1] for i in batch_indices]
        if len(queries) == 0:
            continue

        q_emb = encode_with_grad(queries)
        p_emb = encode_with_grad(positives)
        sim = q_emb @ p_emb.T / TAU
        labels = torch.arange(len(queries), device=sim.device)
        loss = F_t.cross_entropy(sim, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss.item())
        epoch_loss += loss_val
        n_batches_seen += 1
        completed = batch_idx + 1

        if completed % LOG_EVERY == 0:
            elapsed = time.time() - t_epoch
            sec_per_batch = elapsed / max(completed - start_batch, 1)
            eta = (total_batches - completed) * sec_per_batch
            avg_loss = epoch_loss / max(n_batches_seen, 1)
            print(f"      f{fold_idx} batch {completed}/{total_batches}  loss={loss_val:.4f}  "
                  f"avg={avg_loss:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s ({eta/3600:.1f}h)",
                  flush=True)

        if completed % CKPT_EVERY == 0:
            save_checkpoint(ckpt_dir, model, optimizer, completed, perm, total_batches,
                            epoch_loss, n_batches_seen, device)
            print(f"      f{fold_idx} checkpoint saved at batch {completed}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def encode_catalog(model, meta, all_track_ids, device):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks on {device}...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    bs = 256 if device == "cuda" else 128
    embs = model.encode(track_texts, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def retrieve(model, queries, track_embs, track_ids, played_lists, device, topk=TOPK):
    print(f"{ts()} Encoding {len(queries)} queries on {device}...", flush=True)
    bs = 256 if device == "cuda" else 64
    q_embs = model.encode(queries, batch_size=bs, show_progress_bar=False,
                          normalize_embeddings=True).astype(np.float32)
    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(queries)):
        played_set = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        sims = q_embs[i] @ track_embs.T
        ranked = np.argsort(-sims)
        top = []
        for j in ranked:
            tid = track_ids[j]
            if tid not in played_set:
                top.append((tid, float(sims[j])))
                if len(top) >= topk:
                    break
        results.append(top)
    return results


def run_one_fold(fold_i, folds, cases, meta, all_track_ids, train_split_pairs,
                 output_dir, device, use_smoke_fold0):
    fold_t0 = time.time()
    print(f"\n{ts()} === FOLD {fold_i} (device={device}) ===")
    fold_dir = output_dir / f"fold_{fold_i}"
    model_dir = fold_dir / "model"
    embs_path = fold_dir / "track_embs.npy"
    lists_path = fold_dir / "oof_lists.json"
    ckpt_dir = fold_dir / "checkpoints"

    if lists_path.exists():
        print(f"  fold {fold_i}: lists already exist — skipping")
        return

    # Optional fold-0 reuse from local smoke
    if use_smoke_fold0 and fold_i == 0 and SMOKE_FOLD0.exists():
        print(f"  fold-0: reusing local phase3_smoke artifacts")
        fold_dir.mkdir(parents=True, exist_ok=True)
        if (SMOKE_FOLD0 / "model").exists() and not model_dir.exists():
            shutil.copytree(str(SMOKE_FOLD0 / "model"), str(model_dir))
        if (SMOKE_FOLD0 / "track_embs.npy").exists() and not embs_path.exists():
            shutil.copy(str(SMOKE_FOLD0 / "track_embs.npy"), str(embs_path))
        if (SMOKE_FOLD0 / "oof_lists.json").exists():
            shutil.copy(str(SMOKE_FOLD0 / "oof_lists.json"), str(lists_path))
            return

    n = len(cases)
    val_idx = folds[fold_i].tolist()
    train_idx = [j for j in range(n) if j not in set(val_idx)]
    val_cases = [cases[j] for j in val_idx]
    train_dev_cases = [cases[j] for j in train_idx]

    dev_pairs = []
    for c in train_dev_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        dev_pairs.append((build_query_structured_from_dev(c, meta), build_track_text(gt, meta)))
    all_pairs = dev_pairs + train_split_pairs
    print(f"  fold {fold_i}: train_dev={len(dev_pairs)}  train_split={len(train_split_pairs)}  "
          f"total={len(all_pairs)}", flush=True)

    if model_dir.exists() and (model_dir / "config.json").exists():
        print(f"  loading existing model from {model_dir}", flush=True)
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = train_fold(fold_i, all_pairs, model_dir, ckpt_dir, device)

    if embs_path.exists():
        print(f"  loading existing embeddings", flush=True)
        track_embs = np.load(embs_path)
    else:
        track_embs = encode_catalog(model, meta, all_track_ids, device)
        np.save(embs_path, track_embs)

    val_queries = [build_query_structured_from_dev(c, meta) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    fold_lists = retrieve(model, val_queries, track_embs, all_track_ids, val_played, device)

    manifest = {
        "fold": fold_i,
        "n_train_pairs": len(all_pairs),
        "n_dev_pairs": len(dev_pairs),
        "n_train_split_pairs": len(train_split_pairs),
        "n_val_cases": len(val_cases),
        "model_dir": str(model_dir),
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
        "max_seq_len": MAX_SEQ_LEN,
        "training_seed": TRAINING_SEED + fold_i,
        "sampling_seed": SAMPLING_SEED,
        "query_format": "structured",
        "positive_format": "r21_exact",
        "device": device,
        "elapsed_s": time.time() - fold_t0,
        "created_at": datetime.now().isoformat(),
    }
    with open(lists_path, "w") as f:
        json.dump({"lists": fold_lists, "manifest": manifest, "val_idx": val_idx}, f)
    print(f"  fold {fold_i} saved. Elapsed: {time.time() - fold_t0:.0f}s", flush=True)

    hits200 = sum(1 for k, c in enumerate(val_cases)
                   if c["gt"] in [t for t, _ in fold_lists[k][:200]])
    print(f"  fold {fold_i} val hit@200: {hits200}/{len(val_cases)} "
          f"({hits200/len(val_cases):.4f})", flush=True)

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"  fold {fold_i} checkpoints removed", flush=True)


def aggregate_oof_lists(output_dir, n):
    """Combine per-fold oof_lists.json into a global indexed oof_r54_lists.json."""
    oof_results = [None] * n
    fold_manifests = {}
    all_present = True
    for fi in range(5):
        lists_path = output_dir / f"fold_{fi}" / "oof_lists.json"
        if not lists_path.exists():
            print(f"  fold {fi}: missing, skipping aggregation")
            all_present = False
            continue
        fold_data = json.load(open(lists_path))
        val_idx = fold_data.get("val_idx") or fold_data["manifest"].get("val_idx")
        if val_idx is None:
            raise RuntimeError(f"fold {fi}: missing val_idx in {lists_path}")
        for k, v in enumerate(val_idx):
            oof_results[v] = fold_data["lists"][k]
        fold_manifests[fi] = fold_data["manifest"]

    if not all_present:
        print(f"  Aggregation incomplete — some folds missing")
        return

    agg_path = output_dir / "oof_r54_lists.json"
    with open(agg_path, "w") as f:
        json.dump({
            "lists": oof_results,
            "fold_manifests": fold_manifests,
            "n_cases": n,
            "experiment": "R54 Phase 3 full (structured query + train-split 20k)",
            "created_at": datetime.now().isoformat(),
        }, f)
    print(f"  Aggregated: {agg_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="Run a single fold (0-4). If unset, run all 5.")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="Force device. Default: cuda if available else cpu.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Where to write fold artifacts.")
    ap.add_argument("--use-smoke-fold0", action="store_true",
                    help="For local CPU runs only: reuse cache/r54/phase3_smoke/fold_0.")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip aggregating oof_r54_lists.json at the end.")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch  # type: ignore[reportMissingImports]
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"{ts()} Device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"{ts()} R54 Phase 3 full training (output_dir={output_dir})")

    print(f"{ts()} Loading payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)
    n = len(cases)

    print(f"{ts()} Loading catalog...", flush=True)
    meta, all_track_ids = load_catalog()

    print(f"{ts()} Building folds...", flush=True)
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)

    print(f"\n{ts()} Sampling train-split (seed={SAMPLING_SEED})...", flush=True)
    train_split_pairs = build_train_split_sample(meta, dev_session_ids)

    if args.fold is not None:
        assert 0 <= args.fold < 5, "fold must be 0-4"
        run_one_fold(args.fold, folds, cases, meta, all_track_ids, train_split_pairs,
                     output_dir, args.device, args.use_smoke_fold0)
    else:
        for fold_i in range(5):
            run_one_fold(fold_i, folds, cases, meta, all_track_ids, train_split_pairs,
                         output_dir, args.device, args.use_smoke_fold0)

    if not args.no_aggregate:
        print(f"\n{ts()} Aggregating OOF lists...")
        aggregate_oof_lists(output_dir, n)

    print(f"\n{ts()} Done. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
