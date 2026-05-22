#!/usr/bin/env python3
"""R72 — R68 pool-coverage scan (per Codex consult).

For each fold-0 case, count:
  - GT in R54-stacked RRF top-300 (baseline pool)
  - GT in R68 single-source top-300
  - GT in 9-source RRF (8 R54 sources + R68)
  - GT in R68-substituted RRF (R68 replaces R54, R68-stacked variant)

Identifies where R68 actually adds coverage:
  - "POOL_INSIDE": cases where R68 contains GT and baseline already does → no headroom.
  - "POOL_UNIQUE": cases where R68 contains GT and baseline does NOT → recoverable
    if we admit R68 as 9th RRF source.
  - "POOL_LOST": cases where baseline contains GT but R68 does NOT → R68 would
    drop these if we substituted.

For POOL_UNIQUE cases, check: does the 9-source RRF (baseline + R68) actually
surface GT into top-300? Top-30? Top-20? That tells us if late fusion via
admission is viable.

No model training. Just counting. ~30 seconds.
"""
from __future__ import annotations
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
SW_R68_AUGMENTED = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
                    "ALS": 1.0, "R21": 1.0, "R54": 1.0, "R68": 1.0}  # 9 sources
SW_R68_SUBSTITUTED = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
                      "ALS": 1.0, "R21": 1.0, "R68": 1.0}  # R68 replaces R54
POOL_K = 300
RRF_K = 20
FOLD = 0

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R68_LISTS = REPO / "cache" / "r68" / "phase0_fold0" / "oof_r68_lists_fold0.json"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR72_pool_coverage.json"
OUT_MD = REPO / "docs" / "r72_pool_coverage_result.md"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def gt_rank_in(pool: list, gt: str) -> int:
    try:
        return pool.index(gt) + 1
    except ValueError:
        return -1


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R72 — R68 pool-coverage scan (fold-0)")
    print("=" * 70)

    print(f"{ts()} Loading payloads ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()

    print(f"{ts()} Loading W0 fold map ...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold0_idx = [i for i in range(n) if case_fold[i] == FOLD]
    h7_idx = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    print(f"  fold-0: {len(fold0_idx)}  h7: {len(h7_idx)}", flush=True)

    print(f"{ts()} Loading R68 lists ...", flush=True)
    with open(R68_LISTS) as f:
        r68_data = json.load(f)
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"]["val_idx"]
    r68_by_case = {}
    for k_pos, case_idx in enumerate(r68_val_idx):
        r68_by_case[int(case_idx)] = [
            (str(t), float(s)) for t, s in r68_data["lists"][k_pos]]
    print(f"  R68 lists: {len(r68_by_case)}", flush=True)

    print(f"{ts()} Building case index (ALS) ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    print(f"{ts()} Scanning ...", flush=True)
    rows = []
    for i in fold0_idx:
        gt = cases[i]["gt"]
        src_lists_base = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        # R68 single-source list
        r68_list = r68_by_case.get(i, [])
        r68_source = [t for t, _ in r68_list[:POOL_K]]

        # Baseline R54-stacked RRF top-300
        pool_base = weighted_rrf(src_lists_base, SW_BASELINE,
                                 topk=POOL_K, k=RRF_K)
        rank_base = gt_rank_in(pool_base, gt)

        # R68 single-source rank
        r68_rank = -1
        if gt in r68_source:
            r68_rank = r68_source.index(gt) + 1

        # 9-source RRF (admit R68)
        src_lists_aug = dict(src_lists_base)
        src_lists_aug["R68"] = r68_source
        pool_aug = weighted_rrf(src_lists_aug, SW_R68_AUGMENTED,
                                topk=POOL_K, k=RRF_K)
        rank_aug = gt_rank_in(pool_aug, gt)

        # R68 substitution: R54 dropped, R68 in
        src_lists_sub = dict(src_lists_base)
        del src_lists_sub["R54"]
        src_lists_sub["R68"] = r68_source
        pool_sub = weighted_rrf(src_lists_sub, SW_R68_SUBSTITUTED,
                                topk=POOL_K, k=RRF_K)
        rank_sub = gt_rank_in(pool_sub, gt)

        rows.append({
            "case_idx": i,
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "in_base_pool": rank_base > 0,
            "in_r68_source": r68_rank > 0,
            "in_aug_pool": rank_aug > 0,
            "in_sub_pool": rank_sub > 0,
            "rank_base": rank_base,
            "r68_source_rank": r68_rank,
            "rank_aug": rank_aug,
            "rank_sub": rank_sub,
        })

    # Categorize fold-0 cases
    def categorize(r):
        bp = r["in_base_pool"]
        rp = r["in_r68_source"]
        if bp and rp:
            return "POOL_BOTH"
        if bp and not rp:
            return "POOL_BASE_ONLY"
        if rp and not bp:
            return "POOL_R68_UNIQUE"  # R68 catches GT that baseline pool misses
        return "POOL_NEITHER"

    cats = Counter(categorize(r) for r in rows)
    h7_cats = Counter(categorize(r) for r in rows if r["n_prior_music"] == 7)

    unique_to_r68 = [r for r in rows if r["in_r68_source"] and not r["in_base_pool"]]
    lost_to_r68_sub = [r for r in rows if r["in_base_pool"] and not r["in_sub_pool"]]
    aug_recovered = [r for r in unique_to_r68 if r["in_aug_pool"]]
    aug_recovered_top30 = [r for r in unique_to_r68
                           if r["in_aug_pool"] and r["rank_aug"] <= 30]
    aug_recovered_top20 = [r for r in unique_to_r68
                           if r["in_aug_pool"] and r["rank_aug"] <= 20]

    h7_unique = [r for r in unique_to_r68 if r["n_prior_music"] == 7]
    h7_aug_recovered = [r for r in aug_recovered if r["n_prior_music"] == 7]
    h7_aug_top30 = [r for r in aug_recovered_top30 if r["n_prior_music"] == 7]
    h7_aug_top20 = [r for r in aug_recovered_top20 if r["n_prior_music"] == 7]

    print(f"\n{ts()} === fold-0 ({len(fold0_idx)} cases) ===", flush=True)
    for k, v in cats.items():
        print(f"  {k:20}  {v:5d}  ({100 * v / len(fold0_idx):.1f}%)", flush=True)
    print(f"\n  R68-unique-in-pool (R68 catches GT, baseline misses):", flush=True)
    print(f"    fold-0 total: {len(unique_to_r68)}", flush=True)
    print(f"    h7: {len(h7_unique)}", flush=True)
    print(f"\n  Augmented (9-source) RRF rescues:", flush=True)
    print(f"    fold-0 total in aug pool top-300: {len(aug_recovered)} "
          f"of {len(unique_to_r68)} unique", flush=True)
    print(f"    fold-0 aug rank ≤ 30: {len(aug_recovered_top30)}", flush=True)
    print(f"    fold-0 aug rank ≤ 20: {len(aug_recovered_top20)}", flush=True)
    print(f"    h7 in aug pool top-300: {len(h7_aug_recovered)} of {len(h7_unique)}",
          flush=True)
    print(f"    h7 aug rank ≤ 30: {len(h7_aug_top30)}", flush=True)
    print(f"    h7 aug rank ≤ 20: {len(h7_aug_top20)}", flush=True)

    print(f"\n  Substitution loss (baseline catches GT, R68 misses):", flush=True)
    print(f"    cases lost if we drop R54 source: {len(lost_to_r68_sub)} "
          f"in fold-0", flush=True)

    # If POOL_R68_UNIQUE > 0 AND aug rank distribution puts most into top-30,
    # we have a viable late-fusion path (no LR retrain needed, just admit
    # R68 to RRF and let LR rank as-is, possibly with a small boost).
    if h7_aug_top30 and len(h7_aug_top30) >= 5:
        verdict = "POOL_HEADROOM_DETECTED"
        meaning = (f"R68 admission as 9th RRF source surfaces "
                   f"{len(h7_aug_top30)} h7 GT recoveries into top-30 of "
                   f"augmented pool. Worth building source-aware late-fusion "
                   f"or boost rule (Codex path 2). Test: rerank augmented "
                   f"top-30 with frozen R54c (or OOF sibling), measure h7.")
    elif h7_aug_recovered and len(h7_aug_recovered) >= 5:
        verdict = "POOL_HEADROOM_MARGINAL"
        meaning = (f"R68 admission surfaces GT into top-300 of augmented "
                   f"pool but rank is deep (>30 mostly). Needs ranker awareness "
                   f"to actually convert. Less promising for late-fusion only.")
    else:
        verdict = "POOL_SATURATED"
        meaning = ("R68's unique GT recoveries don't survive RRF admission. "
                   "Retrieval-side path is genuinely saturated. Sprint should "
                   "pivot away from retrieval to response/composite.")

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  {meaning}", flush=True)

    # Distribution of aug ranks for unique-to-R68 cases
    aug_rank_dist = Counter()
    for r in unique_to_r68:
        if r["in_aug_pool"]:
            if r["rank_aug"] <= 10:
                aug_rank_dist["top-10"] += 1
            elif r["rank_aug"] <= 20:
                aug_rank_dist["top-20"] += 1
            elif r["rank_aug"] <= 30:
                aug_rank_dist["top-30"] += 1
            elif r["rank_aug"] <= 100:
                aug_rank_dist["top-100"] += 1
            else:
                aug_rank_dist["top-300"] += 1

    out = {
        "experiment": "R72 pool-coverage scan (fold-0)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "meaning": meaning,
        "n_fold0": len(fold0_idx),
        "n_h7": len(h7_idx),
        "categories_fold0": dict(cats),
        "categories_h7": dict(h7_cats),
        "unique_to_r68": {
            "fold0_total": len(unique_to_r68),
            "h7_total": len(h7_unique),
        },
        "aug_admission_recovery": {
            "fold0_in_aug_pool": len(aug_recovered),
            "fold0_top30": len(aug_recovered_top30),
            "fold0_top20": len(aug_recovered_top20),
            "h7_in_aug_pool": len(h7_aug_recovered),
            "h7_top30": len(h7_aug_top30),
            "h7_top20": len(h7_aug_top20),
            "aug_rank_distribution": dict(aug_rank_dist),
        },
        "substitution_loss": {
            "fold0_dropped": len(lost_to_r68_sub),
        },
        "per_case_unique": [
            {"case_idx": r["case_idx"], "h7": r["n_prior_music"] == 7,
             "r68_rank": r["r68_source_rank"], "aug_rank": r["rank_aug"],
             "sub_rank": r["rank_sub"]}
            for r in unique_to_r68
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}", flush=True)

    md = [
        "# R72 — R68 pool-coverage scan (fold-0)",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        meaning,
        "",
        "## Pool coverage categories (fold-0, n=1600)",
        "",
        "| Category | Count | % |",
        "|---|---:|---:|",
    ]
    for k, v in cats.items():
        md.append(f"| {k} | {v} | {100 * v / len(fold0_idx):.1f}% |")
    md += [
        "",
        "## R68-unique-in-pool (R68 catches GT, baseline pool misses)",
        "",
        f"- fold-0 total: **{len(unique_to_r68)}**",
        f"- h7: **{len(h7_unique)}**",
        "",
        "## Augmented (9-source) RRF — does admitting R68 surface GT?",
        "",
        f"- fold-0 GT in aug pool top-300: **{len(aug_recovered)} / {len(unique_to_r68)}**",
        f"- fold-0 GT in aug top-30: **{len(aug_recovered_top30)}**",
        f"- fold-0 GT in aug top-20: **{len(aug_recovered_top20)}**",
        f"- h7 GT in aug pool top-300: **{len(h7_aug_recovered)} / {len(h7_unique)}**",
        f"- h7 GT in aug top-30: **{len(h7_aug_top30)}**",
        f"- h7 GT in aug top-20: **{len(h7_aug_top20)}**",
        "",
        "### Aug-rank distribution for R68-unique cases",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for k, v in aug_rank_dist.items():
        md.append(f"| {k} | {v} |")
    md += [
        "",
        "## Substitution loss (lose if we drop R54)",
        "",
        f"- fold-0 cases lost if R68 substitutes R54: **{len(lost_to_r68_sub)}**",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
