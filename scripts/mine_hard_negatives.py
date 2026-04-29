#!/usr/bin/env python3
# ruff: noqa: T201
"""B3.2: Mine BM25 hard negatives for sequence model training.

For each (session, turn) training example, retrieve BM25 top-50 candidates
using the last_music_meta query, remove the GT track, and store as hard negatives.

Output: cache/seq_model/hard_negatives.pkl
"""
import json
import pickle
import sys
import time
from pathlib import Path

from datasets import DownloadConfig, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "cache" / "seq_model"
HN_PATH = CACHE_DIR / "hard_negatives.pkl"

from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts


def main():
    if HN_PATH.exists():
        with open(HN_PATH, "rb") as f:
            hn = pickle.load(f)
        print(f"Cache already exists: {len(hn)} entries in {HN_PATH}")
        return

    t0 = time.time()
    print("Loading BM25 and metadata...")
    bm25 = CachedBM25()
    metadata = load_track_metadata()

    print("Loading train dataset...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]

    hard_negatives = {}  # (session_id, turn_number) -> [track_id, ...]
    queries = []
    keys = []

    # Build queries for all trainable turns (t >= 2, i.e., at least 1 prior music)
    for item in train:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        for conv in convs:
            if conv["role"] != "user":
                continue
            turn = int(conv["turn_number"])
            user_query = str(conv["content"])
            history = [c for c in convs if int(c["turn_number"]) < turn]
            music_turns = [str(c["content"]).strip() for c in history if c["role"] == "music"]
            if not music_turns:
                continue  # skip t=1 (no history)

            # Build BM25 query using last_music_meta (same as source B)
            q = " ".join(query_parts(history, user_query, metadata, "last_music_meta"))
            if not q:
                q = user_query
            queries.append(q)

            # Find GT for this turn
            gt = None
            for c in convs:
                if c["role"] == "music" and int(c["turn_number"]) == turn:
                    gt = str(c["content"]).strip()
                    break

            keys.append((sid, turn, gt))

    print(f"  {len(queries)} trainable turns")

    # Batch BM25 retrieval
    print("BM25 batch retrieval (top-50)...")
    results = bm25.retrieve_batch(queries, topk=50)

    # Build hard negatives (remove GT)
    for (sid, turn, gt), candidates in zip(keys, results):
        negatives = [tid for tid in candidates if tid != gt][:16]
        hard_negatives[(sid, turn)] = negatives

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HN_PATH, "wb") as f:
        pickle.dump(hard_negatives, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    print(f"Saved {len(hard_negatives)} entries to {HN_PATH}")
    print(f"Avg negatives per example: {sum(len(v) for v in hard_negatives.values()) / len(hard_negatives):.1f}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
