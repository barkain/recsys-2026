#!/usr/bin/env python3
# ruff: noqa: E402,T201
# pyright: reportMissingImports=false, reportOptionalSubscript=false
"""Experiment B follow-up — curated feature combinations + forward-stepwise.

Loads the cached _expB_features.pkl (so no recompute of the 34-feat matrix)
and evaluates targeted combinations the ablation suggests are best:

  M4 (already best, base + G3_surv + G5_surv + G6_surv) = 0.1622
  M6 = base + G3_surv + G4
  M7 = base + G3_surv + G4 + G5_surv  (drop G6_temporal — ablation said it hurts in big model)
  M8 = base + G3_surv + G4 + G5_surv + G6_surv (M4 + G4)
  M9 = M8 + G2_strong (mean_sim_recent5)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expB_feature_expansion import (
    GROUP2_FEATURES,
    GROUP3_FEATURES,
    GROUP4_FEATURES,
    GROUP5_FEATURES,
    GROUP6_FEATURES,
    cv5_evaluate_subset,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES as BASE_FEATURE_NAMES

OUT_FEATURES = REPO_ROOT / "exp" / "eval" / "_expB_features.pkl"
OUT_CURATED = REPO_ROOT / "exp" / "eval" / "expB_curated_models.json"


def main() -> None:
    t0 = time.time()
    if not OUT_FEATURES.exists():
        raise FileNotFoundError(f"need cached features at {OUT_FEATURES} (run expB_feature_expansion.py first)")
    with open(OUT_FEATURES, "rb") as f:
        cache = pickle.load(f)  # noqa: S301
    X = cache["X"]
    gt_idx = cache["gt_idx"]
    sizes = cache["sizes"]
    feat_names = cache["feat_names"]
    sessions = cache["sessions"]
    print(f"loaded features {X.shape}; {len(feat_names)} feats", flush=True)

    seeds = [0, 1, 2]
    base_idx = [feat_names.index(f) for f in BASE_FEATURE_NAMES]

    g3_surv = [
        "already_artist_played_count",
        "candidate_artist_recent_frequency",
    ]
    g4_all = [f for f in GROUP4_FEATURES if f in feat_names]
    g5_surv = ["tag_frequency_score"]
    g6_surv = [
        "release_year_distance_to_history_median",
        "decade_match_flag",
    ]
    g2_strong = ["mean_sim_recent5"]

    def idx_of(names: list[str]) -> list[int]:
        return [feat_names.index(n) for n in names if n in feat_names]

    m4_feats = base_idx + idx_of(g3_surv + g5_surv + g6_surv)
    models = {
        "M4_ref": m4_feats,
        "M6_base_g3_g4": base_idx + idx_of(g3_surv + g4_all),
        "M7_base_g3_g4_g5": base_idx + idx_of(g3_surv + g4_all + g5_surv),
        "M8_base_g3_g4_g5_g6": base_idx + idx_of(g3_surv + g4_all + g5_surv + g6_surv),
        "M9_base_g3_g4_g5_g6_g2strong": base_idx + idx_of(g3_surv + g4_all + g5_surv + g6_surv + g2_strong),
    }
    seen_sigs: dict[tuple[int, ...], str] = {}
    deduped: dict[str, list[int]] = {}
    for name, idxs in models.items():
        sig = tuple(sorted(set(idxs)))
        if sig in seen_sigs:
            continue
        seen_sigs[sig] = name
        deduped[name] = idxs

    results: dict[str, Any] = {}
    print("\n[Curated] feature-group combinations", flush=True)
    for mname, idxs in deduped.items():
        idxs_sorted = sorted(set(idxs))
        t_m = time.time()
        res = cv5_evaluate_subset(X, gt_idx, sizes, sessions, idxs_sorted, feat_names, seeds)
        agg = res["aggregate"]
        delta = agg["mean_cv5"] - 0.1602
        print(
            f"  [{mname}] feats={len(idxs_sorted)}  mean_cv5={agg['mean_cv5']:.4f}  "
            f"std={agg['std_of_cv5_means']:.4f}  Δ vs M0={delta:+.4f}  ({time.time() - t_m:.1f}s)",
            flush=True,
        )
        results[mname] = res

    print("\n[Forward-stepwise] from base, excluding G1_source", flush=True)
    candidates_pool = [
        f for f in (
            GROUP2_FEATURES + GROUP3_FEATURES + GROUP4_FEATURES
            + GROUP5_FEATURES + GROUP6_FEATURES
        ) if f in feat_names
    ]
    current = list(base_idx)
    history = []
    base_res = cv5_evaluate_subset(X, gt_idx, sizes, sessions, current, feat_names, seeds)
    current_mean = base_res["aggregate"]["mean_cv5"]
    history.append({"step": 0, "added": None, "feats": len(current), "mean_cv5": current_mean})
    print(f"  step 0: base only  mean_cv5={current_mean:.4f}", flush=True)

    used = {feat_names[i] for i in current}
    best_res_so_far: dict[str, Any] = base_res
    for step in range(1, 9):
        best_name: str | None = None
        best_delta = -1e9
        best_step_res: dict[str, Any] | None = None
        for cand in candidates_pool:
            if cand in used:
                continue
            cand_idx = feat_names.index(cand)
            trial = sorted(set(current + [cand_idx]))
            res = cv5_evaluate_subset(X, gt_idx, sizes, sessions, trial, feat_names, seeds)
            d = res["aggregate"]["mean_cv5"] - current_mean
            if d > best_delta:
                best_delta = d
                best_name = cand
                best_step_res = res
        if best_name is None or best_step_res is None or best_delta < 1e-4:
            print(
                f"  step {step}: no improvement (best_delta={best_delta:+.4f}) — stopping",
                flush=True,
            )
            break
        current.append(feat_names.index(best_name))
        current = sorted(set(current))
        used.add(best_name)
        current_mean = best_step_res["aggregate"]["mean_cv5"]
        std_now = best_step_res["aggregate"]["std_of_cv5_means"]
        best_res_so_far = best_step_res
        history.append({
            "step": step,
            "added": best_name,
            "feats": len(current),
            "mean_cv5": current_mean,
            "std": std_now,
        })
        print(
            f"  step {step}: +{best_name}  feats={len(current)}  "
            f"mean_cv5={current_mean:.4f}  std={std_now:.4f}  (Δ={best_delta:+.4f})",
            flush=True,
        )

    fwd_res = best_res_so_far
    print(
        f"\n[Forward-stepwise] final feats={len(current)} "
        f"mean_cv5={fwd_res['aggregate']['mean_cv5']:.4f} "
        f"std={fwd_res['aggregate']['std_of_cv5_means']:.4f}",
        flush=True,
    )

    # Pick best across all candidates including forward-stepwise
    candidate_results = dict(results)
    candidate_results["M_forward"] = fwd_res
    best_key = max(
        candidate_results,
        key=lambda k: candidate_results[k]["aggregate"]["mean_cv5"],
    )
    if best_key == "M_forward":
        best_idxs = current
        best_label = "forward_stepwise"
    else:
        best_idxs = sorted(set(deduped[best_key]))
        best_label = best_key

    print(
        f"\n[Robustness] best curated = {best_label}; running 5 extra seeds 5..9",
        flush=True,
    )
    extra_seeds = [5, 6, 7, 8, 9]
    extra_res = cv5_evaluate_subset(
        X, gt_idx, sizes, sessions, best_idxs, feat_names, extra_seeds,
    )
    base_res_for_best = candidate_results[best_key]
    all_seeds_means = (
        [r["cv5_mean"] for r in base_res_for_best["per_seed"]]
        + [r["cv5_mean"] for r in extra_res["per_seed"]]
    )
    print(
        f"  best across 8 seeds: mean={np.mean(all_seeds_means):.4f}  "
        f"std={np.std(all_seeds_means, ddof=1):.4f}  "
        f"range=[{min(all_seeds_means):.4f}, {max(all_seeds_means):.4f}]",
        flush=True,
    )

    final = {
        "elapsed_sec": time.time() - t0,
        "no_llm_calls": True,
        "results": dict(results),
        "forward_stepwise": {
            "history": history,
            "final_features": [feat_names[i] for i in current],
            "final_n_features": len(current),
            "final_per_seed": fwd_res["per_seed"],
            "final_aggregate": fwd_res["aggregate"],
            "final_avg_weights": fwd_res["avg_weights"],
        },
        "robustness_best": {
            "model_label": best_label,
            "feature_indices": best_idxs,
            "feature_names": [feat_names[i] for i in best_idxs],
            "all_seeds_means": all_seeds_means,
            "mean_8seeds": float(np.mean(all_seeds_means)),
            "std_8seeds": float(np.std(all_seeds_means, ddof=1)),
            "min_8seeds": float(np.min(all_seeds_means)),
            "max_8seeds": float(np.max(all_seeds_means)),
        },
    }
    OUT_CURATED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CURATED, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=float)
    print(f"\nWrote {OUT_CURATED}", flush=True)


if __name__ == "__main__":
    main()
