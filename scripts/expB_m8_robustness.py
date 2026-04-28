#!/usr/bin/env python3
# ruff: noqa: E402,T201
# pyright: reportMissingImports=false
"""M8 robustness + final reporting — ZERO API.

M8 = base + G3_surv + G4 + G5_surv + G6_surv = 16 features.
Earlier 3-seed eval: mean_cv5=0.1637 std=0.0017 (above MARGINAL threshold of 0.163).
Run 5 extra seeds (5..9) and compose the final report.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expB_feature_expansion import (
    GROUP4_FEATURES,
    cv5_evaluate_subset,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES as BASE_FEATURE_NAMES

OUT_FEATURES = REPO_ROOT / "exp" / "eval" / "_expB_features.pkl"
OUT_FINAL = REPO_ROOT / "exp" / "eval" / "expB_final_report.json"


def main() -> None:
    t0 = time.time()
    if not OUT_FEATURES.exists():
        raise FileNotFoundError(f"need cached features at {OUT_FEATURES}")
    with open(OUT_FEATURES, "rb") as f:
        cache = pickle.load(f)  # noqa: S301
    X = cache["X"]
    gt_idx = cache["gt_idx"]
    sizes = cache["sizes"]
    feat_names = cache["feat_names"]
    sessions = cache["sessions"]
    print(f"loaded features {X.shape}; {len(feat_names)} feats", flush=True)

    g3_surv_names = ["already_artist_played_count", "candidate_artist_recent_frequency"]
    g4_names = [f for f in GROUP4_FEATURES if f in feat_names]
    g5_surv_names = ["tag_frequency_score"]
    g6_surv_names = ["release_year_distance_to_history_median", "decade_match_flag"]

    m8_names = list(BASE_FEATURE_NAMES) + g3_surv_names + g4_names + g5_surv_names + g6_surv_names
    m8_idx = sorted({feat_names.index(n) for n in m8_names if n in feat_names})
    print(f"M8 features ({len(m8_idx)}): {[feat_names[i] for i in m8_idx]}", flush=True)

    # 3 seeds (already done) + 5 new seeds = 8 total
    seeds_a = [0, 1, 2]
    seeds_b = [5, 6, 7, 8, 9]
    print("\n[M8 robustness] running seeds 0..2 (refresh) + 5..9 (new)", flush=True)
    res_a = cv5_evaluate_subset(X, gt_idx, sizes, sessions, m8_idx, feat_names, seeds_a)
    print(
        f"  seeds 0..2 mean: {res_a['aggregate']['mean_cv5']:.4f}  "
        f"std: {res_a['aggregate']['std_of_cv5_means']:.4f}",
        flush=True,
    )
    res_b = cv5_evaluate_subset(X, gt_idx, sizes, sessions, m8_idx, feat_names, seeds_b)
    print(
        f"  seeds 5..9 mean: {res_b['aggregate']['mean_cv5']:.4f}  "
        f"std: {res_b['aggregate']['std_of_cv5_means']:.4f}",
        flush=True,
    )

    all_means = (
        [r["cv5_mean"] for r in res_a["per_seed"]]
        + [r["cv5_mean"] for r in res_b["per_seed"]]
    )
    mean_8 = float(np.mean(all_means))
    std_8 = float(np.std(all_means, ddof=1))
    range_8 = (float(min(all_means)), float(max(all_means)))
    print(
        f"\n[Robustness] 8-seed mean={mean_8:.4f}  std={std_8:.4f}  "
        f"range={range_8}",
        flush=True,
    )

    if mean_8 >= 0.167 and std_8 <= 0.003:
        verdict = "SUBMISSION-READY"
    elif mean_8 >= 0.163:
        verdict = "MARGINAL"
    else:
        verdict = "REJECT"
    print(
        f"\n=== VERDICT: {verdict} (mean_cv5={mean_8:.4f}, std={std_8:.4f}) ===",
        flush=True,
    )

    weights_a = res_a["avg_weights"]
    weights_b = res_b["avg_weights"]
    avg_weights = {k: (weights_a[k] + weights_b[k]) / 2 for k in weights_a}
    top5 = sorted(avg_weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop-5 features by |avg weight|:", flush=True)
    for name, w in top5:
        print(f"  {name:<42}  w={w:+.4f}", flush=True)

    final = {
        "elapsed_sec": time.time() - t0,
        "no_llm_calls": True,
        "best_model": "M8_base_g3_g4_g5_g6",
        "feature_names": [feat_names[i] for i in m8_idx],
        "n_features": len(m8_idx),
        "mean_cv5_8seeds": mean_8,
        "std_of_means_8seeds": std_8,
        "min_8seeds": range_8[0],
        "max_8seeds": range_8[1],
        "all_seeds_means": all_means,
        "verdict": verdict,
        "avg_weights": avg_weights,
        "top5_by_abs_weight": top5,
        "seeds_a_per_seed": res_a["per_seed"],
        "seeds_b_per_seed": res_b["per_seed"],
    }
    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FINAL, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=float)
    print(f"\nWrote {OUT_FINAL}", flush=True)


if __name__ == "__main__":
    main()
