"""Generic offline evaluator — works with any run_inference_devset.py output.

Loads ground truth from talkpl-ai/TalkPlayData-Challenge-Dataset (test split,
already cached) and computes nDCG@20 + LexDiv against pre-computed inference
results from exp/inference/devset/{tid}.json.

Usage:
    python3 eval_inference.py --tid echo_hybrid_qr_llm_reranker_devset
    python3 eval_inference.py --tid echo_multi_query_devset
"""
# ruff: noqa: T201, S311
import json
import math
import os
import random
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from datasets import Dataset, DownloadConfig, load_dataset


CACHE_DIR = os.environ.get("HF_DATASETS_CACHE")
DATASET_NAME = "talkpl-ai/TalkPlayData-Challenge-Dataset"
DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


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
    """Build ground-truth maps from music turns.

    Some devset rows reuse session_id, so the canonical key must include user_id
    when inference outputs provide it. The session-only map is retained as a
    fallback for older artifacts.
    """
    gt = {}
    gt_by_user = {}
    for item in dataset:
        sid = item["session_id"]
        uid = item.get("user_id")
        gt[sid] = {}
        if uid is not None:
            gt_by_user[(sid, uid)] = {}
        for conv in item["conversations"]:
            if conv["role"] == "music":
                turn = int(conv["turn_number"])
                track_id = conv["content"].strip()
                gt[sid][turn] = track_id
                if uid is not None:
                    gt_by_user[(sid, uid)][turn] = track_id
    return {"session": gt, "session_user": gt_by_user}


def lookup_ground_truth(gt, sid: str, user_id: str | None, turn: int) -> str | None:
    if isinstance(gt, dict) and "session_user" in gt:
        if user_id is not None:
            found = gt["session_user"].get((sid, user_id), {}).get(turn)
            if found is not None:
                return found
        return gt["session"].get(sid, {}).get(turn)
    return gt.get(sid, {}).get(turn)


def load_case_filter(path):
    session_ids = set()
    case_ids = set()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                case_ids.add((parts[0], parts[1]))
            else:
                session_ids.add(line)
    return session_ids, case_ids


def _case_key(row):
    return str(row["session_id"]), "" if row.get("user_id") is None else str(row.get("user_id"))


def write_case_ids(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sid, uid in rows:
            f.write(f"{sid}\t{uid}\n")


def filter_results(results, args):
    if args.session_ids:
        keep_sessions, keep_cases = load_case_filter(args.session_ids)
        results = [
            r for r in results
            if str(r["session_id"]) in keep_sessions or _case_key(r) in keep_cases
        ]
        print(f"Filtered to {len(results)} rows from {args.session_ids}")

    if args.sample_size:
        cases = sorted({_case_key(r) for r in results})
        n = min(args.sample_size, len(cases))
        sampled = set(random.Random(args.sample_seed).sample(cases, n))
        results = [r for r in results if _case_key(r) in sampled]
        print(f"Filtered to deterministic sample: {n} cases (seed={args.sample_seed})")

    if args.write_session_ids:
        cases = sorted({_case_key(r) for r in results})
        write_case_ids(args.write_session_ids, cases)
        print(f"Wrote evaluated case IDs to {args.write_session_ids}")

    return results


def mean_and_ci95(scores):
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(scores) / len(scores)
    if len(scores) < 2:
        return mean, 0.0, mean, mean
    variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
    stderr = math.sqrt(variance / len(scores))
    delta = 1.96 * stderr
    return mean, stderr, max(0.0, mean - delta), min(1.0, mean + delta)


def cached_test_arrow_path() -> str | None:
    pattern = (
        DEFAULT_HF_CACHE
        / "talkpl-ai___talk_play_data-challenge-dataset"
        / "default"
        / "*"
        / "*"
        / "*-test.arrow"
    )
    matches = sorted(pattern.parent.parent.glob(f"*/{pattern.name}"))
    if matches:
        return str(matches[-1])
    matches = sorted(DEFAULT_HF_CACHE.glob(
        "talkpl-ai___talk_play_data-challenge-dataset/default/*/*/*-test.arrow"
    ))
    return str(matches[-1]) if matches else None


def load_ground_truth_dataset(args):
    if args.gt_arrow:
        print(f"Loading ground truth arrow: {args.gt_arrow}")
        return Dataset.from_file(args.gt_arrow)

    arrow_path = cached_test_arrow_path()
    if arrow_path and not args.allow_network:
        print(f"Loading ground truth arrow from HF cache: {arrow_path}")
        return Dataset.from_file(arrow_path)

    print(f"Loading ground truth from {DATASET_NAME} (test split, local cache)...")
    load_kwargs = {
        "split": "test",
        "download_config": DownloadConfig(local_files_only=not args.allow_network),
    }
    if CACHE_DIR:
        load_kwargs["cache_dir"] = CACHE_DIR
    return load_dataset(DATASET_NAME, **load_kwargs)


def main(args):
    inf_path = f"exp/inference/devset/{args.tid}.json"
    if not os.path.exists(inf_path):
        print(f"ERROR: inference results not found at {inf_path}")
        print(f"Run first: python3 run_inference_devset.py --tid {args.tid}")
        return

    print(f"Loading inference results: {inf_path}")
    with open(inf_path) as f:
        results = json.load(f)
    results = filter_results(results, args)

    ds = load_ground_truth_dataset(args)
    gt = build_ground_truth(ds)

    ndcg_scores = []
    responses = []
    skipped = 0

    # Evaluate last turn per session (mirrors Codabench blind scoring)
    # Build per-session max turn for last-turn eval
    session_max_turn = defaultdict(int)
    for r in results:
        t = int(r["turn_number"])
        case = _case_key(r)
        if t > session_max_turn[case]:
            session_max_turn[case] = t

    # --use_candidate_pool: when set, compute nDCG/hit-rate against the wider
    # candidate pool (up to args.k) instead of the 20-long predicted_track_ids.
    # Default behavior is preserved (pool unused) so existing runs remain
    # identical.
    use_pool = getattr(args, "use_candidate_pool", False)
    eval_k = getattr(args, "k", 20) if use_pool else 20

    pool_missing = 0
    for r in results:
        sid = r["session_id"]
        turn = int(r["turn_number"])

        # Last-turn-only (default) — mirrors Codabench
        if not args.all_turns and turn != session_max_turn[_case_key(r)]:
            continue

        gt_id = lookup_ground_truth(gt, sid, r.get("user_id"), turn)
        if gt_id is None:
            skipped += 1
            continue

        if use_pool:
            pool = r.get("candidate_pool_track_ids")
            if pool:
                predicted = pool[:eval_k]
            else:
                # Honest fallback: no pool in this row, use top-20 predictions.
                pool_missing += 1
                predicted = r["predicted_track_ids"]
        else:
            predicted = r["predicted_track_ids"]

        score = ndcg_at_k(predicted, gt_id, k=eval_k)
        ndcg_scores.append(score)
        if r.get("predicted_response"):
            responses.append(r["predicted_response"])

    if use_pool and pool_missing:
        print(
            f"WARN: --use_candidate_pool set but {pool_missing} rows lacked "
            f"candidate_pool_track_ids; fell back to predicted_track_ids for those"
        )

    mean_ndcg, ndcg_stderr, ndcg_ci_low, ndcg_ci_high = mean_and_ci95(ndcg_scores)
    hits = sum(1 for x in ndcg_scores if x > 0)
    hit_rate = hits / len(ndcg_scores) if ndcg_scores else 0.0
    lex_div = lexical_diversity(responses)
    mode = "all-turns" if args.all_turns else "last-turn-only"

    # Label honestly reflects the cutoff used. Default eval_k=20 keeps the
    # pre-existing "@20" output verbatim for all non-pool runs.
    pool_tag = " (from candidate_pool)" if use_pool else ""
    print(f"\n{'='*55}")
    print(f"Config             : {args.tid}")
    print(f"Mode               : {mode}")
    print(f"Turns evaluated    : {len(ndcg_scores)} (skipped {skipped} — no GT)")
    print(f"nDCG@{eval_k:<14}: {mean_ndcg:.4f}{pool_tag}")
    print(f"nDCG stderr        : {ndcg_stderr:.4f}")
    print(f"nDCG 95% CI        : [{ndcg_ci_low:.4f}, {ndcg_ci_high:.4f}]")
    print(f"Hit rate @{eval_k:<9}: {hit_rate:.4f} ({hits}/{len(ndcg_scores)})")
    print(f"LexDiv (proxy)     : {lex_div:.4f}")
    print(f"{'='*55}\n")

    os.makedirs("exp/eval", exist_ok=True)
    suffix = "_all_turns" if args.all_turns else "_last_turn"
    out = f"exp/eval/{args.tid}{suffix}.json"
    with open(out, "w") as f:
        json.dump({
            "ndcg20": mean_ndcg,
            "ndcg20_stderr": ndcg_stderr,
            "ndcg20_ci95": [ndcg_ci_low, ndcg_ci_high],
            "hit_rate20": hit_rate,
            "hits20": hits,
            "lexdiv": lex_div,
            "n": len(ndcg_scores),
        }, f, indent=2)
    print(f"Results saved: {out}")

    if args.min_ndcg is not None and mean_ndcg < args.min_ndcg:
        print(f"FAIL: nDCG@20 {mean_ndcg:.4f} < required {args.min_ndcg:.4f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", required=True)
    parser.add_argument("--all-turns", action="store_true")
    parser.add_argument("--session_ids", type=str, default=None,
                        help="Path to newline-delimited session IDs or tab-delimited session_id/user_id cases")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Evaluate a deterministic random sample of cases from this inference file")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--write_session_ids", type=str, default=None,
                        help="Write evaluated tab-delimited session_id/user_id case IDs to this file")
    parser.add_argument("--min_ndcg", type=float, default=None,
                        help="Exit non-zero unless nDCG@20 is at least this value")
    parser.add_argument("--allow_network", action="store_true",
                        help="Allow HuggingFace network access instead of requiring local cache")
    parser.add_argument("--gt_arrow", type=str, default=None,
                        help="Direct path to cached *-test.arrow ground-truth file")
    parser.add_argument("--use_candidate_pool", action="store_true",
                        help="Score against candidate_pool_track_ids[:k] (additive pool field) instead of predicted_track_ids. Falls back per-row when the field is missing.")
    parser.add_argument("--k", type=int, default=20,
                        help="Top-K cutoff for --use_candidate_pool (default 20, matches the emitted slate)")
    args = parser.parse_args()
    main(args)
