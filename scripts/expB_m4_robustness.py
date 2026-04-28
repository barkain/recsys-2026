#!/usr/bin/env python3
# ruff: noqa: E402,T201
# pyright: reportMissingImports=false
"""M4 robustness — confirm whether M4 is also unstable across 8 seeds.

M4 = base + G3_surv + G5_surv + G6_surv = 13 features.
3-seed eval: mean=0.1622 std=0.0033.
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

from scripts.expB_feature_expansion import cv5_evaluate_subset
from scripts.tune_postrank_v23 import FEATURE_NAMES as BASE_FEATURE_NAMES

OUT_FEATURES = REPO_ROOT / "exp" / "eval" / "_expB_features.pkl"
OUT = REPO_ROOT / "exp" / "eval" / "expB_m4_robustness.json"


def main() -> None:
    t0 = time.time()
    with open(OUT_FEATURES, "rb") as f:
        cache = pickle.load(f)  # noqa: S301
    X = cache["X"]
    gt_idx = cache["gt_idx"]
    sizes = cache["sizes"]
    feat_names = cache["feat_names"]
    sessions = cache["sessions"]

    m4_names = list(BASE_FEATURE_NAMES) + [
        "already_artist_played_count",
        "candidate_artist_recent_frequency",
        "tag_frequency_score",
        "release_year_distance_to_history_median",
        "decade_match_flag",
    ]
    m4_idx = sorted({feat_names.index(n) for n in m4_names if n in feat_names})

    m6_names = list(BASE_FEATURE_NAMES) + [
        "already_artist_played_count",
        "candidate_artist_recent_frequency",
        "tag_token_overlap_query",
        "album_token_overlap_query",
        "artist_or_title_exact_from_query",
    ]
    m6_idx = sorted({feat_names.index(n) for n in m6_names if n in feat_names})

    seeds_a = [0, 1, 2]
    seeds_b = [5, 6, 7, 8, 9]
    out: dict = {}
    for label, idxs in [("M4", m4_idx), ("M6_base_g3_g4", m6_idx)]:
        print(f"\n[{label}] features={len(idxs)}", flush=True)
        res_a = cv5_evaluate_subset(X, gt_idx, sizes, sessions, idxs, feat_names, seeds_a)
        print(
            f"  seeds 0..2 mean: {res_a['aggregate']['mean_cv5']:.4f}  "
            f"std: {res_a['aggregate']['std_of_cv5_means']:.4f}",
            flush=True,
        )
        res_b = cv5_evaluate_subset(X, gt_idx, sizes, sessions, idxs, feat_names, seeds_b)
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
        print(
            f"  8-seed mean={mean_8:.4f}  std={std_8:.4f}  "
            f"range=[{min(all_means):.4f}, {max(all_means):.4f}]",
            flush=True,
        )
        out[label] = {
            "feature_names": [feat_names[i] for i in idxs],
            "all_seeds_means": all_means,
            "mean_8seeds": mean_8,
            "std_of_means_8seeds": std_8,
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {OUT}; elapsed {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
