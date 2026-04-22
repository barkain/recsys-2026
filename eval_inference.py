"""Generic offline evaluator — works with any run_inference_devset.py output.

Loads ground truth from talkpl-ai/TalkPlayData-Challenge-Dataset (test split,
already cached) and computes nDCG@20 + LexDiv against pre-computed inference
results from exp/inference/devset/{tid}.json.

Usage:
    python3 eval_inference.py --tid echo_hybrid_qr_llm_reranker_devset
    python3 eval_inference.py --tid echo_multi_query_devset
"""
import json
import math
import os
import random
import argparse
from collections import defaultdict
from datasets import load_dataset


CACHE_DIR = os.environ.get("HF_DATASETS_CACHE", "./cache")
DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"


def ndcg_at_k(predicted, gt_id, k=20):
    """Binary nDCG@k — single relevant item (the ground-truth music turn)."""
    for i, tid in enumerate(predicted[:k]):
        if tid == gt_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def lexical_diversity(responses, seed=42):
    """Mean pairwise Jaccard distance on unigrams."""
    tokenized = [set(r.lower().split()) for r in responses if r]
    if len(tokenized) < 2:
        return 0.0
    random.seed(seed)
    pairs = [(i, j) for i in range(len(tokenized)) for j in range(i + 1, len(tokenized))]
    sample = random.sample(pairs, min(200, len(pairs)))
    total, count = 0.0, 0
    for i, j in sample:
        a, b = tokenized[i], tokenized[j]
        union = a | b
        if union:
            total += 1 - len(a & b) / len(union)
            count += 1
    return total / count if count else 0.0


def build_ground_truth(dataset):
    """Build {session_id: {turn_number: gt_track_id}} from music turns."""
    gt = {}
    for item in dataset:
        sid = item["session_id"]
        gt[sid] = {}
        for conv in item["conversations"]:
            if conv["role"] == "music":
                gt[sid][int(conv["turn_number"])] = conv["content"].strip()
    return gt


def main(args):
    inf_path = f"exp/inference/devset/{args.tid}.json"
    if not os.path.exists(inf_path):
        print(f"ERROR: inference results not found at {inf_path}")
        print(f"Run first: python3 run_inference_devset.py --tid {args.tid}")
        return

    print(f"Loading inference results: {inf_path}")
    with open(inf_path) as f:
        results = json.load(f)

    print(f"Loading ground truth from {DATASET_NAME} (test split, cached)...")
    ds = load_dataset(DATASET_NAME, split="test", cache_dir=CACHE_DIR)
    gt = build_ground_truth(ds)

    ndcg_scores = []
    responses = []
    skipped = 0

    # Evaluate last turn per session (mirrors Codabench blind scoring)
    # Build per-session max turn for last-turn eval
    session_max_turn = defaultdict(int)
    for r in results:
        t = int(r["turn_number"])
        if t > session_max_turn[r["session_id"]]:
            session_max_turn[r["session_id"]] = t

    for r in results:
        sid = r["session_id"]
        turn = int(r["turn_number"])

        # Last-turn-only (default) — mirrors Codabench
        if not args.all_turns and turn != session_max_turn[sid]:
            continue

        gt_id = gt.get(sid, {}).get(turn)
        if gt_id is None:
            skipped += 1
            continue

        score = ndcg_at_k(r["predicted_track_ids"], gt_id)
        ndcg_scores.append(score)
        if r.get("predicted_response"):
            responses.append(r["predicted_response"])

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    lex_div = lexical_diversity(responses)
    mode = "all-turns" if args.all_turns else "last-turn-only"

    print(f"\n{'='*55}")
    print(f"Config             : {args.tid}")
    print(f"Mode               : {mode}")
    print(f"Turns evaluated    : {len(ndcg_scores)} (skipped {skipped} — no GT)")
    print(f"nDCG@20            : {mean_ndcg:.4f}")
    print(f"LexDiv (proxy)     : {lex_div:.4f}")
    print(f"{'='*55}\n")

    os.makedirs("exp/eval", exist_ok=True)
    suffix = "_all_turns" if args.all_turns else "_last_turn"
    out = f"exp/eval/{args.tid}{suffix}.json"
    with open(out, "w") as f:
        json.dump({"ndcg20": mean_ndcg, "lexdiv": lex_div, "n": len(ndcg_scores)}, f, indent=2)
    print(f"Results saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", required=True)
    parser.add_argument("--all-turns", action="store_true")
    args = parser.parse_args()
    main(args)
