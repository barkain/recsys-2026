"""Paired comparison for two inference outputs on the same devset sessions."""
import argparse
import json
import math
import os
import random
import sys

from eval_inference import build_ground_truth, load_ground_truth_dataset, lookup_ground_truth, ndcg_at_k


def load_results(tid: str) -> dict[tuple[str, str | None, int], list[str]]:
    path = f"exp/inference/devset/{tid}.json"
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {
        (str(r["session_id"]), r.get("user_id"), int(r["turn_number"])): list(r["predicted_track_ids"])
        for r in rows
    }


def load_case_filter(path: str | None) -> tuple[set[str], set[tuple[str, str]]] | None:
    if not path:
        return None
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ci95(values: list[float]) -> tuple[float, float, float]:
    avg = mean(values)
    if len(values) < 2:
        return avg, avg, avg
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    delta = 1.96 * math.sqrt(variance / len(values))
    return avg, avg - delta, avg + delta


def bootstrap_ci(values: list[float], seed: int, rounds: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(rounds):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--session_ids", default=None)
    parser.add_argument("--min_diff", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_network", action="store_true")
    parser.add_argument("--gt_arrow", default=None)
    args = parser.parse_args()

    base = load_results(args.base)
    cand = load_results(args.candidate)
    keep_filter = load_case_filter(args.session_ids)

    gt = build_ground_truth(load_ground_truth_dataset(args))
    common = sorted(set(base) & set(cand))
    if keep_filter:
        keep_sessions, keep_cases = keep_filter
        common = [
            k for k in common
            if k[0] in keep_sessions or (k[0], "" if k[1] is None else str(k[1])) in keep_cases
        ]

    base_scores, cand_scores, diffs = [], [], []
    wins = losses = ties = 0
    skipped = 0
    for sid, user_id, turn in common:
        gt_id = lookup_ground_truth(gt, sid, user_id, turn)
        if gt_id is None:
            skipped += 1
            continue
        b = ndcg_at_k(base[(sid, user_id, turn)], gt_id)
        c = ndcg_at_k(cand[(sid, user_id, turn)], gt_id)
        base_scores.append(b)
        cand_scores.append(c)
        diffs.append(c - b)
        if c > b:
            wins += 1
        elif c < b:
            losses += 1
        else:
            ties += 1

    base_mean, base_low, base_high = ci95(base_scores)
    cand_mean, cand_low, cand_high = ci95(cand_scores)
    diff_mean = mean(diffs)
    diff_low, diff_high = bootstrap_ci(diffs, args.seed)

    print(f"Base       : {args.base}")
    print(f"Candidate  : {args.candidate}")
    print(f"Pairs      : {len(diffs)} (skipped {skipped})")
    print(f"Base nDCG  : {base_mean:.4f} [{base_low:.4f}, {base_high:.4f}]")
    print(f"Cand nDCG  : {cand_mean:.4f} [{cand_low:.4f}, {cand_high:.4f}]")
    print(f"Delta      : {diff_mean:+.4f} bootstrap95 [{diff_low:+.4f}, {diff_high:+.4f}]")
    print(f"W/L/T      : {wins}/{losses}/{ties}")

    os.makedirs("exp/eval", exist_ok=True)
    out = f"exp/eval/compare_{args.candidate}_vs_{args.base}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base": args.base,
                "candidate": args.candidate,
                "n": len(diffs),
                "base_ndcg20": base_mean,
                "candidate_ndcg20": cand_mean,
                "delta_ndcg20": diff_mean,
                "delta_bootstrap_ci95": [diff_low, diff_high],
                "wins": wins,
                "losses": losses,
                "ties": ties,
            },
            f,
            indent=2,
        )
    print(f"Saved: {out}")

    if args.min_diff is not None and diff_mean < args.min_diff:
        print(f"FAIL: delta {diff_mean:+.4f} < required +{args.min_diff:.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
