"""Analyze hit rank distribution across inference results.

Usage:
    python3 analyze_hits.py --tids echo_bm25_nlq_llm10_k100_devset echo_bm25_nlq_llm20_k100_devset
    python3 analyze_hits.py --tid echo_bm25_nlq_llm10_k100_devset
"""
import argparse
import json
import math
import os
from collections import Counter

from datasets import load_dataset


CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "./cache")
DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def load_ground_truth():
    ds = load_dataset(DATASET_NAME, split="test", cache_dir=CACHE_DIR)
    gt = {}
    for row in ds:
        sid = row["session_id"]
        if row["role"] == "music" and row["turn_number"] == 8:
            content = row["content"]
            if isinstance(content, dict):
                gt[sid] = content.get("track_id")
            elif isinstance(content, str):
                try:
                    gt[sid] = json.loads(content).get("track_id")
                except Exception:
                    pass
    return gt


def analyze(tid: str, gt: dict):
    path = f"exp/inference/devset/{tid}.json"
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    hit_ranks = []
    misses = 0
    for item in data:
        sid = item["session_id"]
        gt_id = gt.get(sid)
        if not gt_id:
            continue
        predicted = item["predicted_track_ids"]
        if isinstance(predicted, str):
            predicted = json.loads(predicted.replace("'", '"'))

        found = False
        for i, tid_pred in enumerate(predicted[:100]):
            if tid_pred == gt_id:
                hit_ranks.append(i + 1)  # 1-indexed
                found = True
                break
        if not found:
            misses += 1

    n = len(hit_ranks) + misses
    hits = len(hit_ranks)

    ndcg = sum(1/math.log2(r+1) for r in hit_ranks) / n if n else 0

    rank_buckets = Counter()
    for r in hit_ranks:
        if r == 1:
            rank_buckets["rank1"] += 1
        elif r <= 5:
            rank_buckets["rank2-5"] += 1
        elif r <= 10:
            rank_buckets["rank6-10"] += 1
        elif r <= 20:
            rank_buckets["rank11-20"] += 1
        else:
            rank_buckets["rank21-100"] += 1

    print(f"\n{'='*55}")
    print(f"Config: {tid}")
    print(f"{'='*55}")
    print(f"  Sessions:  {n}")
    print(f"  Hits@100:  {hits} ({100*hits/n:.0f}%)")
    print(f"  Misses:    {misses} ({100*misses/n:.0f}%)")
    print(f"  nDCG@20:   {ndcg:.4f}")
    print(f"  Rank distribution:")
    for bucket in ["rank1", "rank2-5", "rank6-10", "rank11-20", "rank21-100"]:
        c = rank_buckets.get(bucket, 0)
        pct = 100*c/hits if hits else 0
        ndcg_contrib = 0
        if bucket == "rank1":
            ndcg_contrib = c * 1.0 / n
        elif bucket == "rank2-5":
            ndcg_contrib = sum(1/math.log2(r+1) for r in hit_ranks if 2 <= r <= 5) / n
        elif bucket == "rank6-10":
            ndcg_contrib = sum(1/math.log2(r+1) for r in hit_ranks if 6 <= r <= 10) / n
        elif bucket == "rank11-20":
            ndcg_contrib = sum(1/math.log2(r+1) for r in hit_ranks if 11 <= r <= 20) / n
        print(f"    {bucket:12s}: {c:3d} ({pct:.0f}%)  +{ndcg_contrib:.4f} nDCG")
    print(f"  Avg hit rank: {sum(hit_ranks)/hits:.1f}" if hits else "  No hits")

    return {"ndcg": ndcg, "hits": hits, "misses": misses, "n": n, "hit_ranks": hit_ranks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", help="Single TID to analyze")
    parser.add_argument("--tids", nargs="+", help="Multiple TIDs to compare")
    args = parser.parse_args()

    tids = args.tids or ([args.tid] if args.tid else [])
    if not tids:
        parser.print_help()
        return

    print("Loading ground truth...")
    gt = load_ground_truth()
    print(f"Ground truth sessions: {len(gt)}")

    results = {}
    for tid in tids:
        results[tid] = analyze(tid, gt)

    if len(tids) > 1:
        print(f"\n{'='*55}")
        print("COMPARISON")
        print(f"{'='*55}")
        print(f"{'Config':<40} {'nDCG':>8} {'Hits':>6} {'AvgRk':>7}")
        print("-" * 65)
        for tid, r in results.items():
            if r:
                avg_rank = sum(r["hit_ranks"])/len(r["hit_ranks"]) if r["hit_ranks"] else 0
                short = tid.replace("echo_bm25_", "").replace("_devset", "")
                print(f"{short:<40} {r['ndcg']:>8.4f} {r['hits']:>6} {avg_rank:>7.1f}")


if __name__ == "__main__":
    main()
