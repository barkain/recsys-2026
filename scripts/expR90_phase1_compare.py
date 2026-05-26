"""R90 Phase 1 Variant A — head-to-head fold-0 comparison vs R84 OOF baseline.

Loads:
- R90 Variant A fold-0 lists from cache/r90/phase1_fold0_varA/oof_r84_lists.json
- R84 fold-0 OOF lists from cache/r84/phase0b_fold0/oof_r84_lists.json
- R12 dev payload + supporting maps

Computes the four R90 Phase 1 gates:
- G1 same-artist canary: h7 same-artist nDCG@20 Δ >= -0.005
- G2 h7 aggregate: nDCG@20 Δ >= +0.003 OR (recovered - lost) >= 5 (top-30)
- G3 history buckets: per-n_prior Δ >= -0.005 for n_prior in {2,3,4,5,6}
- G4 per-fold sanity (fold-0 only since we retrained only fold-0):
    fold-0 h7 nDCG@20 Δ >= -0.005 vs R84 fold-0

Outputs:
- exp/eval/expR90_phase1_compare.json
- docs/r90_phase1_compare.md

Dry-run (--dry-run): loads existing R84 fold-0 OOF + R12 + maps, validates
the gate-computation pipeline with R84 vs R84 (so all deltas should be 0),
exits before any R90-specific path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R84_FOLD0_LISTS = REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
DEFAULT_R90_LISTS = REPO / "cache" / "r90" / "phase1_fold0_varA" / "oof_r84_lists.json"

OUT_JSON = REPO / "exp" / "eval" / "expR90_phase1_compare.json"
OUT_MD = REPO / "docs" / "r90_phase1_compare.md"

GATE_SAME_ARTIST_MIN = -0.005
GATE_H7_NDCG_MIN = 0.003
GATE_NET_RECOVERY_MIN = 5
GATE_HISTORY_BUCKET_MIN = -0.005
GATE_PER_FOLD_MIN = -0.005


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(rank: int, k: int) -> float:
    if 1 <= rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def gt_rank(top_list: list[tuple[str, float]], gt: str) -> int:
    for i, (t, _s) in enumerate(top_list, 1):
        if t == gt:
            return i
    return 0


def load_oof_lists(path: Path) -> dict[int, list[tuple[str, float]]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): [(t, float(s)) for t, s in v] for k, v in raw.items()}


def per_case_table(cases: list[dict], a_lists: dict, b_lists: dict,
                    track_artist: dict, label_a: str, label_b: str) -> list[dict]:
    """For each case in BOTH a_lists and b_lists, compute the comparison row."""
    rows = []
    for ci, case in enumerate(cases):
        if ci not in a_lists or ci not in b_lists:
            continue
        gt = case["gt"]
        a_rank = gt_rank(a_lists[ci], gt)
        b_rank = gt_rank(b_lists[ci], gt)
        rows.append({
            "case_idx": ci,
            "n_prior": int(case.get("n_prior_music", 0)),
            "is_h7": case.get("n_prior_music") == 7,
            "same_art": bool(same_artist_case(case, track_artist)),
            "a_rank": a_rank,
            "b_rank": b_rank,
            "a_ndcg20": ndcg_at_k(a_rank, 20),
            "b_ndcg20": ndcg_at_k(b_rank, 20),
            "a_in_top30": 0 < a_rank <= 30,
            "b_in_top30": 0 < b_rank <= 30,
        })
    return rows


def compute_gates(rows: list[dict]) -> dict:
    """Compute G1-G4 from per-case rows. `a` is the BASELINE (R84), `b` is the CANDIDATE (R90)."""
    h7 = [r for r in rows if r["is_h7"]]
    same_h7 = [r for r in h7 if r["same_art"]]
    diff_h7 = [r for r in h7 if not r["same_art"]]

    # G1: same-artist canary
    same_a = float(np.mean([r["a_ndcg20"] for r in same_h7])) if same_h7 else 0.0
    same_b = float(np.mean([r["b_ndcg20"] for r in same_h7])) if same_h7 else 0.0
    g1_delta = same_b - same_a
    g1_pass = g1_delta >= GATE_SAME_ARTIST_MIN

    # G2: h7 aggregate
    h7_a = float(np.mean([r["a_ndcg20"] for r in h7])) if h7 else 0.0
    h7_b = float(np.mean([r["b_ndcg20"] for r in h7])) if h7 else 0.0
    g2_delta = h7_b - h7_a
    recovered = sum(1 for r in h7 if r["b_in_top30"] and not r["a_in_top30"])
    lost = sum(1 for r in h7 if r["a_in_top30"] and not r["b_in_top30"])
    g2_net = recovered - lost
    g2_pass = (g2_delta >= GATE_H7_NDCG_MIN) or (g2_net >= GATE_NET_RECOVERY_MIN)

    # G3: history buckets {2,3,4,5,6}
    history_buckets = {}
    g3_pass = True
    for n_prior in [2, 3, 4, 5, 6]:
        sub = [r for r in rows if r["n_prior"] == n_prior]
        if not sub:
            history_buckets[str(n_prior)] = {"n": 0, "skipped": True}
            continue
        a_n = float(np.mean([r["a_ndcg20"] for r in sub]))
        b_n = float(np.mean([r["b_ndcg20"] for r in sub]))
        delta = b_n - a_n
        bucket_pass = delta >= GATE_HISTORY_BUCKET_MIN
        history_buckets[str(n_prior)] = {
            "n": len(sub), "a_ndcg20": a_n, "b_ndcg20": b_n, "delta": delta,
            "pass": bucket_pass,
        }
        if not bucket_pass:
            g3_pass = False

    # G4: per-fold sanity — fold-0 only since R90 only retrained fold-0
    # (When fold sets are identical, this duplicates G2 for h7.)
    g4_delta = g2_delta
    g4_pass = g4_delta >= GATE_PER_FOLD_MIN

    # Diff-artist diagnostic (not a gate; for the report)
    diff_a = float(np.mean([r["a_ndcg20"] for r in diff_h7])) if diff_h7 else 0.0
    diff_b = float(np.mean([r["b_ndcg20"] for r in diff_h7])) if diff_h7 else 0.0

    overall_pass = g1_pass and g2_pass and g3_pass and g4_pass
    verdict = "PROCEED_TO_5FOLD" if overall_pass else (
        "INVESTIGATE" if (g1_pass and g2_pass and not g3_pass) else "ARCHIVE_VARIANT_A"
    )

    return {
        "n_cases": len(rows),
        "n_h7": len(h7),
        "n_same_artist_h7": len(same_h7),
        "n_diff_artist_h7": len(diff_h7),
        "gate_thresholds": {
            "G1_same_artist_min": GATE_SAME_ARTIST_MIN,
            "G2_h7_ndcg_min": GATE_H7_NDCG_MIN,
            "G2_net_recovery_min": GATE_NET_RECOVERY_MIN,
            "G3_history_bucket_min": GATE_HISTORY_BUCKET_MIN,
            "G4_per_fold_min": GATE_PER_FOLD_MIN,
        },
        "G1_same_artist": {
            "a_ndcg20": same_a, "b_ndcg20": same_b,
            "delta": g1_delta, "pass": g1_pass,
        },
        "G2_h7_aggregate": {
            "a_ndcg20": h7_a, "b_ndcg20": h7_b,
            "delta": g2_delta,
            "recovered": recovered, "lost": lost, "net": g2_net,
            "pass_by_delta": g2_delta >= GATE_H7_NDCG_MIN,
            "pass_by_recovery": g2_net >= GATE_NET_RECOVERY_MIN,
            "pass": g2_pass,
        },
        "G3_history_buckets": {
            "buckets": history_buckets,
            "pass": g3_pass,
        },
        "G4_per_fold_sanity": {
            "fold0_h7_delta": g4_delta, "pass": g4_pass,
        },
        "diff_artist_h7": {
            "a_ndcg20": diff_a, "b_ndcg20": diff_b, "delta": diff_b - diff_a,
        },
        "overall_pass": overall_pass,
        "verdict": verdict,
    }


def write_markdown(report: dict, label_a: str, label_b: str) -> None:
    """Render comparison report as markdown."""
    lines = []
    lines.append(f"# R90 Phase 1 Variant A compare")
    lines.append("")
    lines.append(f"Date: {report['created_at']}  ")
    lines.append(f"Baseline: **{label_a}** (cache/r84/phase0b_fold0/)  ")
    lines.append(f"Candidate: **{label_b}** (cache/r90/phase1_fold0_varA/)")
    lines.append("")
    lines.append(f"## Verdict: **{report['gates']['verdict']}**")
    lines.append("")
    gates = report["gates"]
    g1 = gates["G1_same_artist"]
    g2 = gates["G2_h7_aggregate"]
    g3 = gates["G3_history_buckets"]
    g4 = gates["G4_per_fold_sanity"]
    da = gates["diff_artist_h7"]
    lines.append(f"### Gate summary")
    lines.append(f"| gate | result | detail |")
    lines.append(f"|---|---|---|")
    lines.append(f"| G1 same-artist canary | {'PASS' if g1['pass'] else 'FAIL'} | "
                 f"h7-same Δ = {g1['delta']:+.4f} (threshold ≥ {gates['gate_thresholds']['G1_same_artist_min']:+.3f}) |")
    lines.append(f"| G2 h7 aggregate | {'PASS' if g2['pass'] else 'FAIL'} | "
                 f"h7 Δ = {g2['delta']:+.4f}, rec/lost = {g2['recovered']}/{g2['lost']} (net {g2['net']:+d}) |")
    lines.append(f"| G3 history buckets (n_prior 2-6) | {'PASS' if g3['pass'] else 'FAIL'} | "
                 f"see breakdown below |")
    lines.append(f"| G4 fold-0 sanity | {'PASS' if g4['pass'] else 'FAIL'} | "
                 f"fold-0 h7 Δ = {g4['fold0_h7_delta']:+.4f} (threshold ≥ {gates['gate_thresholds']['G4_per_fold_min']:+.3f}) |")
    lines.append("")
    lines.append(f"### h7 nDCG@20 detail")
    lines.append(f"| segment | n | {label_a} | {label_b} | Δ |")
    lines.append(f"|---|---:|---:|---:|---:|")
    lines.append(f"| h7 (all) | {gates['n_h7']} | {g2['a_ndcg20']:.4f} | {g2['b_ndcg20']:.4f} | {g2['delta']:+.4f} |")
    lines.append(f"| h7 same-artist | {gates['n_same_artist_h7']} | {g1['a_ndcg20']:.4f} | {g1['b_ndcg20']:.4f} | {g1['delta']:+.4f} |")
    lines.append(f"| h7 diff-artist | {gates['n_diff_artist_h7']} | {da['a_ndcg20']:.4f} | {da['b_ndcg20']:.4f} | {da['delta']:+.4f} |")
    lines.append("")
    lines.append(f"### History buckets (n_prior 2-6)")
    lines.append(f"| n_prior | n | {label_a} | {label_b} | Δ | gate |")
    lines.append(f"|---:|---:|---:|---:|---:|---|")
    for k in ["2", "3", "4", "5", "6"]:
        b = g3["buckets"].get(k, {"n": 0})
        if b.get("skipped") or b.get("n", 0) == 0:
            lines.append(f"| {k} | 0 | — | — | — | skipped |")
        else:
            lines.append(f"| {k} | {b['n']} | {b['a_ndcg20']:.4f} | {b['b_ndcg20']:.4f} | "
                         f"{b['delta']:+.4f} | {'PASS' if b['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("Files: `exp/eval/expR90_phase1_compare.json` (this JSON), "
                 "`docs/r90_phase1_compare.md` (this report).")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r90-lists", type=Path, default=DEFAULT_R90_LISTS,
                    help=f"R90 fold-0 OOF lists. Default: {DEFAULT_R90_LISTS}")
    ap.add_argument("--r84-lists", type=Path, default=R84_FOLD0_LISTS,
                    help=f"R84 fold-0 OOF baseline. Default: {R84_FOLD0_LISTS}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run with R84 vs R84 to verify gate plumbing; all deltas should be 0.")
    args = ap.parse_args()

    t0 = time.time()
    print(f"{ts()} R90 Phase 1 compare. dry_run={args.dry_run}")
    print(f"  baseline: {args.r84_lists}")
    print(f"  candidate: {args.r90_lists}")

    if not args.r84_lists.exists():
        print(f"FATAL: R84 baseline missing: {args.r84_lists}")
        sys.exit(1)

    if not args.dry_run and not args.r90_lists.exists():
        print(f"FATAL: R90 candidate missing: {args.r90_lists}")
        print("       Run scripts/expR90_phase1_eval.py first, OR use --dry-run.")
        sys.exit(1)

    print(f"{ts()} Loading R12 payload + maps...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    maps, _track_pop, _track_album = load_supporting_maps()
    track_artist = maps["track_artist"]

    print(f"{ts()} Loading R84 fold-0 OOF lists...")
    r84_lists = load_oof_lists(args.r84_lists)
    print(f"  R84 cases: {len(r84_lists)}")

    if args.dry_run:
        print(f"{ts()} DRY-RUN: comparing R84 vs R84 (expected: all deltas = 0)...")
        rows = per_case_table(cases, r84_lists, r84_lists, track_artist, "R84", "R84")
        gates = compute_gates(rows)
        # Sanity: dry-run must produce all-zero deltas
        assert abs(gates["G1_same_artist"]["delta"]) < 1e-9, gates["G1_same_artist"]
        assert abs(gates["G2_h7_aggregate"]["delta"]) < 1e-9, gates["G2_h7_aggregate"]
        for k, b in gates["G3_history_buckets"]["buckets"].items():
            if not b.get("skipped") and b.get("n", 0) > 0:
                assert abs(b["delta"]) < 1e-9, (k, b)
        assert gates["G2_h7_aggregate"]["recovered"] == 0
        assert gates["G2_h7_aggregate"]["lost"] == 0
        print(f"  PASS: all deltas zero, no recovery/loss, gates report symmetric outcome.")
        print(f"  G1 PASS={gates['G1_same_artist']['pass']}, "
              f"G2 PASS={gates['G2_h7_aggregate']['pass']}, "
              f"G3 PASS={gates['G3_history_buckets']['pass']}, "
              f"G4 PASS={gates['G4_per_fold_sanity']['pass']}")
        # Note: with zero delta, G2 passes by recovery-net=0 < 5 and delta=0 < 0.003,
        # so it should FAIL — let's confirm the gate plumbing is correct
        # (this is what we want: identical models should NOT pass the +0.003 gate)
        dry_summary = {
            "experiment": "R90 Phase 1 compare dry-run (R84 vs R84)",
            "created_at": datetime.now().isoformat(),
            "n_cases": len(rows),
            "n_h7": gates["n_h7"],
            "all_deltas_zero": True,
            "expected_g2_outcome": "FAIL (identical models can't pass +0.003 delta gate)",
            "actual_g2_pass": gates["G2_h7_aggregate"]["pass"],
            "expected_g1_outcome": "PASS (zero delta passes -0.005 threshold)",
            "actual_g1_pass": gates["G1_same_artist"]["pass"],
            "elapsed_s": time.time() - t0,
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(dry_summary, f, indent=2)
        print(f"  wrote dry-run summary -> {OUT_JSON.relative_to(REPO)}")
        print(f"{ts()} DRY-RUN PASS — gate plumbing OK. Elapsed: {time.time() - t0:.1f}s")
        return

    print(f"{ts()} Loading R90 candidate lists...")
    r90_lists = load_oof_lists(args.r90_lists)
    print(f"  R90 cases: {len(r90_lists)}")

    print(f"{ts()} Computing per-case table + gates...")
    rows = per_case_table(cases, r84_lists, r90_lists, track_artist, "R84", "R90_varA")
    gates = compute_gates(rows)

    report = {
        "experiment": "R90 Phase 1 Variant A vs R84 fold-0 compare",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "baseline": str(args.r84_lists.relative_to(REPO)),
        "candidate": str(args.r90_lists.relative_to(REPO)),
        "gates": gates,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"{ts()} Wrote {OUT_JSON.relative_to(REPO)}")

    write_markdown(report, "R84_fold0", "R90_varA_fold0")
    print(f"{ts()} Wrote {OUT_MD.relative_to(REPO)}")
    print(f"\n=== VERDICT: {gates['verdict']} ===")
    print(f"G1 same-artist: {'PASS' if gates['G1_same_artist']['pass'] else 'FAIL'}  "
          f"Δ={gates['G1_same_artist']['delta']:+.4f}")
    print(f"G2 h7 aggregate: {'PASS' if gates['G2_h7_aggregate']['pass'] else 'FAIL'}  "
          f"Δ={gates['G2_h7_aggregate']['delta']:+.4f}  "
          f"rec/lost={gates['G2_h7_aggregate']['recovered']}/{gates['G2_h7_aggregate']['lost']}")
    print(f"G3 history buckets: {'PASS' if gates['G3_history_buckets']['pass'] else 'FAIL'}")
    print(f"G4 fold-0 sanity: {'PASS' if gates['G4_per_fold_sanity']['pass'] else 'FAIL'}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
