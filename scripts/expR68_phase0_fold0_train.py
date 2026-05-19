#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R68 Phase 0 — BGE-large-en-v1.5 fold-0 smoke training.

Identical recipe to R54 Phase 3 (structured query, R21 track text, 1 epoch,
in-batch InfoNCE, tau=0.05, lr=2e-5, batch_size=32, max_seq_len=256), except:
  * MODEL_NAME = BAAI/bge-large-en-v1.5 (was BGE-base, 768-d -> 1024-d)
  * Fold 0 ONLY (Phase 1 covers 5 folds; this is the smoke).
  * Fold split MUST match R54: derived via
    `grouped_session_folds(sessions, seed=0, k=5)` and cross-checked
    against `cache/r54/phase2_full/oof_manifest.json` val_indices_sample.

Outputs (under --output_dir, default cache/r68/phase0_fold0/):
  model/                          SentenceTransformer checkpoint
  query_embeddings_dev.npy        fold-0 held-out dev query embs (N x 1024)
  track_embeddings.npy            full catalog track embs (N_tracks x 1024)
  track_ids.json                  ordered track ids aligned to track_embeddings
  oof_r68_lists_fold0.json        per held-out case top-300 (tid, cosine)
  train_log.json                  loss curve, hyperparams, device, elapsed

This script is designed to be RUN ON A GPU BOX. The Mac wave just authors
it. Wave 1 commits this file; Wave 2 (post-sync) runs the eval.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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
R54_OOF_MANIFEST = REPO / "cache" / "r54" / "phase2_full" / "oof_manifest.json"
DEFAULT_OUTPUT_DIR = REPO / "cache" / "r68" / "phase0_fold0"
DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Same as R54 Phase 3
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
FOLD = 0
LOG_EVERY = 50


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# --- text builders (copied verbatim from R54 phase3 train; do not edit) ---

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
                    q = build_query_structured_from_session(
                        user_msgs_so_far, played_so_far, most_recent_user_msg, meta)
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


def derive_fold0_split(cases):
    """Build session-grouped 5-fold split. Cross-check against R54 manifest."""
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    if not R54_OOF_MANIFEST.exists():
        raise RuntimeError(
            f"Missing R54 OOF manifest at {R54_OOF_MANIFEST}. Cannot guarantee "
            f"fold-0 alignment with R54. Aborting.")
    manifest = json.load(open(R54_OOF_MANIFEST))
    seed = int(manifest["fold_split"]["seed"]) if "fold_split" in manifest else 0
    k = int(manifest["fold_split"]["k"]) if "fold_split" in manifest else 5
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=seed, k=k)
    fold0 = folds[FOLD].tolist()
    # Cross-check sample
    sample = manifest["folds"][str(FOLD)]["val_indices_sample"]
    if fold0[:len(sample)] != list(sample):
        raise RuntimeError(
            f"Fold-0 mismatch vs R54 manifest: derived={fold0[:5]} "
            f"manifest={sample}")
    print(f"  fold-0 split cross-checked OK ({len(fold0)} held-out cases)", flush=True)
    return fold0, folds


def train(all_pairs, model_name, device, output_dir, log_path):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  fold-0: {len(all_pairs)} pairs, {total_batches} batches, device={device}",
          flush=True)

    np.random.seed(TRAINING_SEED + FOLD)
    torch.manual_seed(TRAINING_SEED + FOLD)
    if device == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED + FOLD)

    perm = np.random.permutation(len(all_pairs)).astype(np.int64)

    model = SentenceTransformer(model_name, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
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
    loss_curve: list[dict[str, float]] = []
    epoch_loss = 0.0
    n_batches_seen = 0

    for batch_idx in range(total_batches):
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
            sec_per_batch = elapsed / max(completed, 1)
            eta = (total_batches - completed) * sec_per_batch
            avg_loss = epoch_loss / max(n_batches_seen, 1)
            print(f"{ts()} fold=0 step={completed}/{total_batches} "
                  f"loss={loss_val:.4f} avg={avg_loss:.4f} lr={LR:.2e} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s ({eta/3600:.1f}h)",
                  flush=True)
            loss_curve.append({
                "step": completed, "loss": loss_val, "avg_loss": avg_loss,
                "elapsed_s": elapsed,
            })

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))

    # Write partial train log (final log written by main with full metadata)
    with open(log_path, "w") as f:
        json.dump({"loss_curve": loss_curve,
                   "n_batches_seen": n_batches_seen,
                   "epoch_loss": epoch_loss,
                   "avg_loss_final": epoch_loss / max(n_batches_seen, 1),
                   "train_elapsed_s": time.time() - t_epoch}, f, indent=2)
    return model


def encode_catalog(model, meta, all_track_ids, device):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks on {device}...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    bs = 256 if device == "cuda" else 128
    embs = model.encode(track_texts, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def encode_queries(model, queries, device):
    print(f"{ts()} Encoding {len(queries)} queries on {device}...", flush=True)
    bs = 256 if device == "cuda" else 64
    return model.encode(queries, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)


def retrieve(q_embs, track_embs, track_ids, played_lists, topk=TOPK):
    print(f"{ts()} Retrieving top-{topk} via cosine (normalized dot product)...",
          flush=True)
    results = []
    for i in range(q_embs.shape[0]):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME,
                    help="HF model name (default: BAAI/bge-large-en-v1.5)")
    ap.add_argument("--fold", type=int, default=FOLD,
                    help="Fold index to train (Phase 0 expects 0)")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="Output directory for fold-0 artifacts")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="Force device. Default: cuda if available else cpu.")
    args = ap.parse_args()

    if args.fold != FOLD:
        raise SystemExit(
            f"Phase 0 supports fold=0 only (got --fold {args.fold}). "
            f"Use scripts/expR68_phase1_full5fold_train.py for 5-fold.")

    if args.device is None:
        try:
            import torch  # type: ignore[reportMissingImports]
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"{ts()} Device: {args.device}", flush=True)
    if args.device != "cuda":
        print(f"{ts()} WARNING: device is not cuda. R68 Phase 0 is designed "
              f"for GPU. CPU run will take 24h+.", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"{ts()} R68 Phase 0 fold-0 training (model={args.model})", flush=True)
    print(f"{ts()} Output dir: {output_dir}", flush=True)

    # Load payload
    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)
    n = len(cases)

    # Load catalog
    print(f"{ts()} Loading catalog...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  catalog: {len(all_track_ids)} tracks, meta entries: {len(meta)}",
          flush=True)

    # Derive fold-0 split (validate against R54 manifest)
    print(f"{ts()} Deriving fold-0 split (must match R54 manifest)...", flush=True)
    fold0_val_idx, _all_folds = derive_fold0_split(cases)

    n_val = len(fold0_val_idx)
    train_idx = [j for j in range(n) if j not in set(fold0_val_idx)]
    val_cases = [cases[j] for j in fold0_val_idx]
    train_dev_cases = [cases[j] for j in train_idx]

    # Build training pairs
    dev_pairs = []
    for c in train_dev_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        dev_pairs.append((build_query_structured_from_dev(c, meta),
                          build_track_text(gt, meta)))

    print(f"{ts()} Sampling train-split (seed={SAMPLING_SEED})...", flush=True)
    train_split_pairs = build_train_split_sample(meta, dev_session_ids)
    all_pairs = dev_pairs + train_split_pairs
    print(f"  fold-0: train_dev={len(dev_pairs)}  train_split={len(train_split_pairs)}  "
          f"total={len(all_pairs)}  val={n_val}", flush=True)

    # Train
    log_path = output_dir / "train_log.json"
    model = train(all_pairs, args.model, args.device, output_dir, log_path)

    # Encode catalog
    track_embs = encode_catalog(model, meta, all_track_ids, args.device)
    track_embs_path = output_dir / "track_embeddings.npy"
    np.save(track_embs_path, track_embs)
    print(f"  saved {track_embs_path} (shape={track_embs.shape}, dtype={track_embs.dtype})",
          flush=True)

    track_ids_path = output_dir / "track_ids.json"
    with open(track_ids_path, "w") as f:
        json.dump(all_track_ids, f)
    print(f"  saved {track_ids_path} ({len(all_track_ids)} ids)", flush=True)

    # Encode held-out dev queries
    val_queries = [build_query_structured_from_dev(c, meta) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    q_embs = encode_queries(model, val_queries, args.device)
    q_embs_path = output_dir / "query_embeddings_dev.npy"
    np.save(q_embs_path, q_embs)
    print(f"  saved {q_embs_path} (shape={q_embs.shape}, dtype={q_embs.dtype})",
          flush=True)

    # Retrieve top-300 per held-out case (drop played)
    fold_lists = retrieve(q_embs, track_embs, all_track_ids, val_played)

    # Sanity: hit@200 on val
    hits200 = sum(1 for k, c in enumerate(val_cases)
                  if c["gt"] in [t for t, _ in fold_lists[k][:200]])
    hits300 = sum(1 for k, c in enumerate(val_cases)
                  if c["gt"] in [t for t, _ in fold_lists[k][:300]])
    print(f"  fold-0 val hit@200: {hits200}/{n_val} ({hits200/n_val:.4f})", flush=True)
    print(f"  fold-0 val hit@300: {hits300}/{n_val} ({hits300/n_val:.4f})", flush=True)

    # Persist OOF lists
    lists_path = output_dir / "oof_r68_lists_fold0.json"
    manifest = {
        "experiment": "R68 Phase 0 fold-0",
        "model": args.model,
        "fold": FOLD,
        "n_train_pairs": len(all_pairs),
        "n_dev_pairs": len(dev_pairs),
        "n_train_split_pairs": len(train_split_pairs),
        "n_val_cases": n_val,
        "model_dir": str(output_dir / "model"),
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
        "max_seq_len": MAX_SEQ_LEN, "topk": TOPK,
        "training_seed": TRAINING_SEED + FOLD,
        "sampling_seed": SAMPLING_SEED,
        "query_format": "structured",
        "positive_format": "r21_exact",
        "device": args.device,
        "val_hit_at_200": hits200,
        "val_hit_at_300": hits300,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    with open(lists_path, "w") as f:
        json.dump({"lists": fold_lists, "manifest": manifest,
                   "val_idx": fold0_val_idx}, f)
    print(f"{ts()} Saved OOF lists: {lists_path}", flush=True)

    # Final train log (merge with partial)
    if log_path.exists():
        partial = json.load(open(log_path))
    else:
        partial = {}
    partial.update({
        "model_name": args.model,
        "fold": FOLD,
        "hyperparams": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
            "max_seq_len": MAX_SEQ_LEN,
            "train_split_sample_pairs": TRAIN_SPLIT_SAMPLE_PAIRS,
            "max_pairs_per_session": MAX_PAIRS_PER_SESSION,
            "training_seed": TRAINING_SEED + FOLD,
            "sampling_seed": SAMPLING_SEED,
        },
        "device": args.device,
        "total_elapsed_s": time.time() - t0,
        "n_val_cases": n_val,
        "val_hit_at_200": hits200,
        "val_hit_at_300": hits300,
        "created_at": datetime.now().isoformat(),
    })
    with open(log_path, "w") as f:
        json.dump(partial, f, indent=2)
    print(f"{ts()} Saved train log: {log_path}", flush=True)

    print(f"\n{ts()} Phase 0 fold-0 complete. Total elapsed: {time.time() - t0:.0f}s",
          flush=True)
    print(f"Sync trigger artifact: {lists_path}", flush=True)


if __name__ == "__main__":
    main()
