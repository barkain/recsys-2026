#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R55: production-grade R54 retriever — single all-data training + blind retrieval.

Trains BGE-base on ALL 8,000 dev cases + 20K session-balanced train-split
sample (no held-out fold), encodes the full track catalog, retrieves top-300
blind-A candidates per query with cosine scores. Output is a drop-in
replacement for the 5-fold ensemble used by R54b.

Differences from R54 Phase 3 (5-fold ensemble):
- One training run on all dev (vs 5-fold split). No OOF dev eval is possible —
  R55 is an inference-time production probe. Validate by blind churn vs R54
  ensemble before submission.
- Same structured-query format, BGE-base, hyperparams, sampling seed as Phase 3.
- Resumable checkpoints (CKPT_EVERY batches) so an interrupted run can resume.

Designed to run on an ephemeral GPU (RunPod A100 / L40S). Expected total
runtime on A100: ~25-40 min. See `docs/r55_runpod_runbook.md`.

Usage:
  uv run python scripts/expR55_production_train.py --device cuda
  uv run python scripts/expR55_production_train.py --device cuda --skip-training
      (when model dir already exists; re-runs only catalog encode + blind retrieve)

Outputs (under cache/r55_production/):
  model/                 — SentenceTransformer
  track_embeddings.npy   — catalog embeddings (47k × 768)
  track_ids.json         — catalog track order
  blind_r55_lists.json   — { lists: { sid: [(tid, score), ...] }, manifest: {...} }
  checkpoints/           — resumable training state (auto-pruned, removed on completion)
  manifest.json          — training summary
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
OUT_DIR = REPO / "cache" / "r55_production"
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


def build_query_structured_from_session(user_msgs_so_far, played_so_far,
                                          current_user_msg, meta):
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


def build_query_structured_blind(item, meta):
    """Same structured-query format as dev side, for blind-A test items."""
    import pandas as pd  # type: ignore[reportMissingImports]
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    last_user = user_rows.iloc[-1]
    turn_num = int(last_user["turn_number"])
    user_query = str(last_user["content"])
    prior = df[df["turn_number"] < turn_num]
    user_utterances = [str(r["content"]) for _, r in prior.iterrows() if r["role"] == "user"]
    played = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
    history = user_utterances[-3:]
    context_tracks = []
    for tid in played[-5:]:
        if tid in meta:
            context_tracks.append(build_short_track_ref(tid, meta))
    parts = [f"[QUERY] {user_query}"]
    if history:
        parts.append(f"[HISTORY] {' '.join(history)}")
    if context_tracks:
        parts.append(f"[CONTEXT] {'; '.join(context_tracks)}")
    return " ".join(parts), played


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
    """Same session-balanced sampling as Phase 3 (seed=0, max 2/session, 20k pairs)."""
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
    print(f"  train_split: {n_dev_overlap} dev overlap (excluded), "
          f"{len(sampled)} pairs from {len(sampled_sessions)} sessions", flush=True)
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


def train_production(all_pairs, model_dir, ckpt_dir, device):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  {len(all_pairs)} pairs, {total_batches} batches, device={device}", flush=True)

    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED)

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
        else:
            print(f"  WARN: checkpoint total_batches={ckpt['total_batches']} != "
                  f"current {total_batches}; starting fresh", flush=True)

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
            print(f"      batch {completed}/{total_batches}  loss={loss_val:.4f}  "
                  f"avg={avg_loss:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

        if completed % CKPT_EVERY == 0:
            save_checkpoint(ckpt_dir, model, optimizer, completed, perm, total_batches,
                            epoch_loss, n_batches_seen, device)
            print(f"      checkpoint saved at batch {completed}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    print(f"  training complete: avg_loss={epoch_loss / max(n_batches_seen, 1):.4f}",
          flush=True)
    return model


def encode_catalog(model, meta, all_track_ids, device):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks on {device}...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    bs = 256 if device == "cuda" else 128
    embs = model.encode(track_texts, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def retrieve_blind(model, blind_items, meta, track_embs, track_ids, device, topk=TOPK):
    print(f"{ts()} Building blind queries...", flush=True)
    queries = []
    sids = []
    played_lists = []
    for item in blind_items:
        sid = str(item["session_id"])
        q, played = build_query_structured_blind(item, meta)
        queries.append(q)
        sids.append(sid)
        played_lists.append(played)
    print(f"  {len(queries)} blind queries")

    print(f"{ts()} Encoding blind queries on {device}...", flush=True)
    bs = 256 if device == "cuda" else 64
    q_embs = model.encode(queries, batch_size=bs, show_progress_bar=False,
                          normalize_embeddings=True).astype(np.float32)

    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = {}
    for i, sid in enumerate(sids):
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
        results[sid] = top
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--skip-training", action="store_true",
                    help="If model already exists at OUT_DIR/model, skip training and "
                         "re-run only catalog encoding + blind retrieval.")
    ap.add_argument("--output-dir", default=str(OUT_DIR),
                    help="Where to write artifacts.")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch  # type: ignore[reportMissingImports]
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"{ts()} Device: {args.device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    model_dir = out_dir / "model"
    embs_path = out_dir / "track_embeddings.npy"
    ids_path = out_dir / "track_ids.json"
    blind_lists_path = out_dir / "blind_r55_lists.json"
    ckpt_dir = out_dir / "checkpoints"
    manifest_path = out_dir / "manifest.json"

    print(f"{ts()} Loading dev payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    dev_session_ids = set(c["session_id"] for c in cases)
    print(f"  {len(cases)} dev cases ({len(dev_session_ids)} unique sessions)")

    print(f"{ts()} Loading catalog...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  {len(all_track_ids)} tracks")

    print(f"\n{ts()} Building dev pairs (all {len(cases)})...", flush=True)
    dev_pairs = []
    for c in cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        dev_pairs.append((build_query_structured_from_dev(c, meta), build_track_text(gt, meta)))
    print(f"  Dev pairs: {len(dev_pairs)}")

    print(f"\n{ts()} Sampling train-split (seed={SAMPLING_SEED})...", flush=True)
    train_split_pairs = build_train_split_sample(meta, dev_session_ids)
    print(f"  Train-split sample: {len(train_split_pairs)} pairs")

    all_pairs = dev_pairs + train_split_pairs
    print(f"  TOTAL training pairs: {len(all_pairs)}")

    if args.skip_training and model_dir.exists() and (model_dir / "config.json").exists():
        print(f"\n{ts()} Loading existing R55 model from {model_dir}")
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
        model = SentenceTransformer(str(model_dir), device=args.device)
    else:
        print(f"\n{ts()} Training R55 (production-grade R54)...")
        model = train_production(all_pairs, model_dir, ckpt_dir, args.device)
        print(f"  Saved to {model_dir}")

    if embs_path.exists() and ids_path.exists():
        print(f"\n{ts()} Loading cached catalog embeddings")
        track_embs = np.load(embs_path)
        cached_ids = json.loads(ids_path.read_text())
        if cached_ids != all_track_ids:
            print(f"  WARNING: cached track_ids order differs — re-encoding")
            track_embs = encode_catalog(model, meta, all_track_ids, args.device)
            np.save(embs_path, track_embs)
            ids_path.write_text(json.dumps(all_track_ids))
    else:
        track_embs = encode_catalog(model, meta, all_track_ids, args.device)
        np.save(embs_path, track_embs)
        ids_path.write_text(json.dumps(all_track_ids))
        print(f"  Embeddings saved to {embs_path}")

    print(f"\n{ts()} Loading blind-A test set...", flush=True)
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    try:
        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                          download_config=DownloadConfig(local_files_only=True))
    except Exception:
        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test")
    blind_items = list(db)
    print(f"  Blind-A: {len(blind_items)} sessions")

    blind_lists = retrieve_blind(model, blind_items, meta, track_embs, all_track_ids,
                                  args.device)
    print(f"  Retrieved top-{TOPK} for {len(blind_lists)} blind queries")

    manifest = {
        "experiment": "R55 production-grade R54 (single all-data training)",
        "model_dir": str(model_dir),
        "n_blind_sessions": len(blind_lists),
        "n_train_pairs": len(all_pairs),
        "n_dev_pairs": len(dev_pairs),
        "n_train_split_pairs": len(train_split_pairs),
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
        "max_seq_len": MAX_SEQ_LEN, "topk": TOPK,
        "training_seed": TRAINING_SEED, "sampling_seed": SAMPLING_SEED,
        "query_format": "structured",
        "positive_format": "r21_exact",
        "device": args.device,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    with open(blind_lists_path, "w") as f:
        json.dump({"lists": blind_lists, "manifest": manifest}, f)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{ts()} Saved:")
    print(f"  {blind_lists_path}")
    print(f"  {manifest_path}")

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        print(f"  Checkpoints removed (training complete)")

    print(f"\n{ts()} Done. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
