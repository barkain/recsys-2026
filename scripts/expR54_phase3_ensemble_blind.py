#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 3 blind retrieval — 5-fold ensemble (no Colab needed).

Each Phase 3 fold model was trained on 80% of dev + 20K train-split. The union
of all 5 folds' training data covers 100% of dev (each dev case is in train
for 4 of 5 fold models). Ensembling them produces a retriever functionally
close to "production trained on all dev", while also dampening the per-fold
cosine variance that destabilized the CV5 LR on fold 3.

For each blind query:
  1. Encode with each fold's R54 model (5 query embeddings)
  2. Compute cosine vs each fold's cached catalog embeddings
  3. Average the 5 cosines per track
  4. Top-300 (played tracks excluded)

Output: cache/r54_production/blind_r54_lists.json
        format: { lists: { sid: [(tid, score), ...] }, manifest: {...} }
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path(__file__).resolve().parent.parent

FOLD_DIRS = [
    REPO / "cache" / "r54" / "phase3_smoke" / "fold_0",
    REPO / "cache" / "r54" / "phase3_full" / "fold_1",
    REPO / "cache" / "r54" / "phase3_full" / "fold_2",
    REPO / "cache" / "r54" / "phase3_full" / "fold_3",
    REPO / "cache" / "r54" / "phase3_full" / "fold_4",
]
OUT_DIR = REPO / "cache" / "r54_production"
TOPK = 300
MAX_SEQ_LEN = 256


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def build_short_track_ref(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = artists[0] if isinstance(artists, list) and artists else str(artists)
    return f"{name} by {artist}"


def build_query_structured_blind(item, meta):
    import pandas as pd
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
    return " ".join(parts), played, turn_num


def load_catalog():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta, track_ids


def encode_queries(model_dir, queries):
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
    model = SentenceTransformer(str(model_dir), device="cpu")
    embs = model.encode(queries, batch_size=32, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 3 ensemble blind retrieval (5-fold local)")
    print("=" * 70)

    # Verify all fold artifacts present
    print(f"{ts()} Checking fold artifacts...")
    for d in FOLD_DIRS:
        model_cfg = d / "model" / "config.json"
        embs_path = d / "track_embs.npy"
        if not model_cfg.exists() or not embs_path.exists():
            print(f"  MISSING: {d}")
            return
        print(f"  OK: {d.relative_to(REPO)}")

    # Load catalog (needed for query construction)
    print(f"{ts()} Loading catalog metadata...")
    meta, all_track_ids = load_catalog()
    n_tracks = len(all_track_ids)
    print(f"  {n_tracks} tracks")

    # Load blind-A
    print(f"{ts()} Loading blind-A test set...")
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))
    blind_items = list(db)
    print(f"  {len(blind_items)} blind sessions")

    # Build queries
    print(f"{ts()} Building blind queries (structured format)...")
    queries = []
    sids = []
    played_lists = []
    for item in blind_items:
        sid = str(item["session_id"])
        q, played, _ = build_query_structured_blind(item, meta)
        queries.append(q)
        sids.append(sid)
        played_lists.append(played)

    # Per-fold scoring — accumulate into avg_scores
    avg_scores = np.zeros((len(queries), n_tracks), dtype=np.float64)
    for fi, fold_dir in enumerate(FOLD_DIRS):
        print(f"\n{ts()} Fold {fi}: loading embeddings + encoding queries...")
        track_embs = np.load(fold_dir / "track_embs.npy")  # (n_tracks, dim) normalized
        assert track_embs.shape[0] == n_tracks, \
            f"fold {fi} track_embs row count {track_embs.shape[0]} != {n_tracks}"
        q_embs = encode_queries(fold_dir / "model", queries)  # (n_queries, dim)
        # Cosine = q_emb @ track_embs.T (both normalized)
        fold_scores = q_embs @ track_embs.T  # (n_queries, n_tracks)
        avg_scores += fold_scores.astype(np.float64)
        del track_embs, q_embs, fold_scores
        print(f"  fold {fi} done. elapsed: {time.time() - t0:.0f}s")

    avg_scores /= len(FOLD_DIRS)
    print(f"\n{ts()} Computing top-{TOPK} per query (played tracks excluded)...")

    blind_lists = {}
    for i, sid in enumerate(sids):
        played_set = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        # Mask played tracks
        scores = avg_scores[i].copy()
        for ti, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[ti] = -np.inf
        # Top-K (argpartition is faster than full sort)
        top_idx = np.argpartition(-scores, TOPK)[:TOPK]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        blind_lists[sid] = [(all_track_ids[j], float(scores[j])) for j in top_idx]

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "blind_r54_lists.json"
    out_payload = {
        "lists": blind_lists,
        "manifest": {
            "experiment": "R54 Phase 3 ensemble blind retrieval (5-fold)",
            "fold_dirs": [str(d.relative_to(REPO)) for d in FOLD_DIRS],
            "n_blind_sessions": len(blind_lists),
            "topk": TOPK,
            "method": "average cosine across 5 fold R54 models",
            "elapsed_s": time.time() - t0,
            "created_at": datetime.now().isoformat(),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f)
    print(f"  Saved blind lists -> {out_path}")
    print(f"  Size: {out_path.stat().st_size:,} bytes")

    # Quick sanity stats
    score_min = min(min(s for _, s in lst[:300]) for lst in blind_lists.values())
    score_max = max(max(s for _, s in lst[:300]) for lst in blind_lists.values())
    list_lens = [len(lst) for lst in blind_lists.values()]
    print(f"  sessions: {len(blind_lists)}")
    print(f"  list lengths: min={min(list_lens)} median={int(np.median(list_lens))} max={max(list_lens)}")
    print(f"  cosine scores: min={score_min:.4f} max={score_max:.4f}")
    print(f"\n{ts()} Done. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
