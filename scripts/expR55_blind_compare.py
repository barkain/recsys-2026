#!/usr/bin/env python3
# ruff: noqa: T201
"""R55 vs R54-ensemble blind-list comparison + churn gates.

After `expR55_production_train.py` produces `cache/r55_production/blind_r55_lists.json`,
this script compares against the R54 ensemble baseline (and optionally the R54b
production top-20s) and enforces the pre-submission churn gates.

Gates (from feedback memory `retriever-swap-churn-gates`):
  - top-1 changed vs R54b: <25/80 preferred, hard stop >35/80
  - top-20 overlap median: >=14/20

Usage:
  uv run python scripts/expR55_blind_compare.py

Outputs:
  exp/eval/expR55_blind_compare.json

Exit codes:
  0 — all gates pass (safe to proceed with submission)
  1 — soft gate fail (review before submission)
  2 — hard gate fail (do not submit)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R54_ENSEMBLE = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R55_LISTS = REPO / "cache" / "r55_production" / "blind_r55_lists.json"
R54B_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r54b_aligned_submission.json"
REPORT_OUT = REPO / "exp" / "eval" / "expR55_blind_compare.json"

# Gates (see feedback memory retriever-swap-churn-gates)
SOFT_TOP1_CHURN = 25  # > triggers soft warning, < OK
HARD_TOP1_CHURN = 35  # > aborts submission
TOP20_OVERLAP_MIN = 14  # median must be >= this


def load_lists_file(path):
    """Return dict sid -> [(tid, score), ...]."""
    with open(path) as f:
        data = json.load(f)
    return data["lists"]


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def main():
    if not R55_LISTS.exists():
        print(f"FAIL: {R55_LISTS} not found. Run expR55_production_train.py first.")
        sys.exit(3)
    if not R54_ENSEMBLE.exists():
        print(f"FAIL: {R54_ENSEMBLE} not found.")
        sys.exit(3)
    if not R54B_SUBMISSION.exists():
        print(f"FAIL: {R54B_SUBMISSION} not found.")
        sys.exit(3)

    print("R55 vs R54 ensemble — blind-list comparison")
    print("=" * 70)
    r54_lists = load_lists_file(R54_ENSEMBLE)
    r55_lists = load_lists_file(R55_LISTS)
    with open(R54B_SUBMISSION) as f:
        r54b_rows = json.load(f)
    r54b_top1 = {r["session_id"]: r["predicted_track_ids"][0] for r in r54b_rows}

    sids_r54 = set(r54_lists.keys())
    sids_r55 = set(r55_lists.keys())
    common = sids_r54 & sids_r55
    print(f"  R54 ensemble sessions: {len(sids_r54)}")
    print(f"  R55 sessions:          {len(sids_r55)}")
    print(f"  Common:                {len(common)}")
    if sids_r54 != sids_r55:
        missing = (sids_r54 | sids_r55) - common
        print(f"  WARN: {len(missing)} sessions differ. First 3: {sorted(missing)[:3]}")

    # ===== Overlap stats =====
    top1_overlap = []   # 0/1 per session (same top-1 tid)
    top20_overlap = []  # int 0..20 per session
    top300_overlap = []  # int 0..300 per session
    r54_cos_top1 = []
    r55_cos_top1 = []
    r55_cos_min = []
    r55_cos_p50 = []

    # Top-1 churn vs R54b production
    top1_changed_vs_r54b = 0
    changed_sids = []

    for sid in sorted(common):
        r54 = r54_lists[sid]
        r55 = r55_lists[sid]
        r54_top1 = r54[0][0] if r54 else None
        r55_top1 = r55[0][0] if r55 else None
        top1_overlap.append(int(r54_top1 == r55_top1))

        r54_top20 = {t for t, _ in r54[:20]}
        r55_top20 = {t for t, _ in r55[:20]}
        top20_overlap.append(len(r54_top20 & r55_top20))

        r54_top300 = {t for t, _ in r54[:300]}
        r55_top300 = {t for t, _ in r55[:300]}
        top300_overlap.append(len(r54_top300 & r55_top300))

        if r54 and r55:
            r54_cos_top1.append(r54[0][1])
            r55_cos_top1.append(r55[0][1])
            r55_scores = [s for _, s in r55]
            r55_cos_min.append(min(r55_scores))
            r55_cos_p50.append(percentile(r55_scores, 0.5))

        # Churn vs R54b production top-1
        if sid in r54b_top1 and r55_top1 != r54b_top1[sid]:
            top1_changed_vs_r54b += 1
            changed_sids.append(sid)

    n = len(common)

    def summary(xs, label, max_val=None):
        if not xs:
            return f"  {label}: no data"
        mn, mx = min(xs), max(xs)
        avg = sum(xs) / len(xs)
        p50 = percentile(xs, 0.5)
        return (f"  {label}: mean={avg:.3f} median={p50:.3f} min={mn} max={mx}"
                + (f" / {max_val}" if max_val else ""))

    print(f"\n=== Overlap vs R54 ensemble (n={n} sessions) ===")
    print(f"  top-1 same: {sum(top1_overlap)}/{n} ({sum(top1_overlap) / n:.1%})")
    print(summary(top20_overlap, "top-20 overlap", 20))
    print(summary(top300_overlap, "top-300 overlap", 300))

    print(f"\n=== Cosine score distribution ===")
    if r54_cos_top1 and r55_cos_top1:
        print(f"  R54 ens top-1 cos: mean={sum(r54_cos_top1) / len(r54_cos_top1):.4f}  "
              f"median={percentile(r54_cos_top1, 0.5):.4f}")
        print(f"  R55     top-1 cos: mean={sum(r55_cos_top1) / len(r55_cos_top1):.4f}  "
              f"median={percentile(r55_cos_top1, 0.5):.4f}")
        print(f"  R55 top-300 min cos: median={percentile(r55_cos_min, 0.5):.4f}")
        print(f"  R55 top-300 p50 cos: median={percentile(r55_cos_p50, 0.5):.4f}")

    # ===== Churn vs R54b production =====
    print(f"\n=== Top-1 churn vs R54b production ===")
    print(f"  R55 top-1 differs from R54b top-1: {top1_changed_vs_r54b}/{n}")
    print(f"  Soft threshold: < {SOFT_TOP1_CHURN}")
    print(f"  Hard stop:      > {HARD_TOP1_CHURN}")

    # Note: this is raw retrieval top-1; the final submission top-1 will be set
    # by LR re-ranking. The retrieval churn here is a leading indicator.

    # ===== Gates =====
    top20_median = percentile(top20_overlap, 0.5) if top20_overlap else 0
    print(f"\n=== Submission gates ===")
    print(f"  top-1 churn (retrieval): {top1_changed_vs_r54b}/{n}")
    print(f"  top-20 overlap median:   {top20_median:.1f}/20  (require >= {TOP20_OVERLAP_MIN})")

    gate_status = "PASS"
    fail_reasons = []
    if top1_changed_vs_r54b > HARD_TOP1_CHURN:
        gate_status = "HARD_FAIL"
        fail_reasons.append(
            f"top-1 churn {top1_changed_vs_r54b} > hard stop {HARD_TOP1_CHURN}")
    elif top1_changed_vs_r54b >= SOFT_TOP1_CHURN:
        gate_status = "SOFT_FAIL"
        fail_reasons.append(
            f"top-1 churn {top1_changed_vs_r54b} >= soft threshold {SOFT_TOP1_CHURN}")

    if top20_median < TOP20_OVERLAP_MIN:
        if gate_status == "PASS":
            gate_status = "SOFT_FAIL"
        fail_reasons.append(
            f"top-20 overlap median {top20_median:.1f} < {TOP20_OVERLAP_MIN}")

    print(f"\n  GATE STATUS: {gate_status}")
    for reason in fail_reasons:
        print(f"    - {reason}")

    # ===== Save report =====
    report = {
        "n_sessions": n,
        "overlap": {
            "top1_same": sum(top1_overlap),
            "top20_overlap_mean": sum(top20_overlap) / n if n else 0,
            "top20_overlap_median": top20_median,
            "top300_overlap_mean": sum(top300_overlap) / n if n else 0,
            "top300_overlap_median": percentile(top300_overlap, 0.5),
        },
        "churn_vs_r54b": {
            "top1_changed": top1_changed_vs_r54b,
            "changed_sids": changed_sids,
            "soft_threshold": SOFT_TOP1_CHURN,
            "hard_threshold": HARD_TOP1_CHURN,
        },
        "cosine_stats": {
            "r54_ensemble_top1_mean": sum(r54_cos_top1) / len(r54_cos_top1) if r54_cos_top1 else None,
            "r55_top1_mean": sum(r55_cos_top1) / len(r55_cos_top1) if r55_cos_top1 else None,
            "r55_top300_min_median": percentile(r55_cos_min, 0.5) if r55_cos_min else None,
            "r55_top300_p50_median": percentile(r55_cos_p50, 0.5) if r55_cos_p50 else None,
        },
        "gates": {
            "status": gate_status,
            "fail_reasons": fail_reasons,
            "top1_overlap_min": TOP20_OVERLAP_MIN,
        },
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {REPORT_OUT}")

    if gate_status == "HARD_FAIL":
        sys.exit(2)
    if gate_status == "SOFT_FAIL":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
