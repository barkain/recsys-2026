#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R58 Phase 2 — Option A: second-stage LightGBM specialist over LR top-50.

Dev-only evaluator. NO blind code, NO submission packaging, NO retrieval.

Implements two adjustments to the architecture-choice doc:

  1. Exclude all-negative groups (cases where GT is NOT in LR top-50) from
     training by default. Run both configs for comparison:
       reachable_only_train : train only on groups with GT in top-50 (4116)
       all_groups_train     : include all 8000 groups

  2. Meta-only ablation before full 51 features:
       meta_only : LR signal (5) + per-source ranks/cosine (9) = 14 features
       full_51   : meta + 37 baseline LR features = 51 features

  Total: 4 configs × 7 betas = 28 evaluations.

For each config, predicts stage-2 scores OOF (CV5 with the same fold
assignment as LR). Blends with LR via:

  final_score(c) = z(lr_score(c)) + beta * z(stage2_score(c))

z(.) is per-case z-normalisation.

Gates and metrics mirror R56 / R57b strictly. Same-artist canary is
enforced.

Inputs (must exist from R58 inventory):
  cache/r58/top50_dev.pkl              — top-50 table
  exp/eval/expR55_post_refresh_decomp.json — baseline reference

Output:
  exp/eval/expR58_stage2_results.json  — full diagnostics per config × beta
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOP50_PATH = REPO / "cache" / "r58" / "top50_dev.pkl"
DECOMP_JSON = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
OUT = REPO / "exp" / "eval" / "expR58_stage2_results.json"

# Feature column names
# NOTE: r54_cosine is dropped here because it duplicates the same column in
# the 37 baseline features (BASE_FEATURE_NAMES). full_51 would otherwise have
# a duplicate feature name which LightGBM rejects. The cosine itself is still
# present via BASE_FEATURE_NAMES.
META_FEATURE_COLS = [
    "lr_score", "lr_score_minus_top", "lr_score_minus_at20",
    "margin_to_20_case", "candidate_rank",
    "r21_rank", "r54_rank",
    "a_rank", "b_rank", "c_rank", "d_rank", "f_rank", "als_rank",
]  # 13 features (meta_only); full_51 becomes 13 + 37 = 50 in practice

# 37 baseline LR feature names (in their stored order in the top50 row "features" list)
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2  # noqa: E402

BASE_FEATURE_NAMES = (
    FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]  # 28 + 2 = wait, FEATURE_NAMES_V2 already has 28
    + ["same_album_last1", "same_album_last3", "same_album_any",
       "album_history_count", "pool_same_album_count"]
    + ["r54_rank_inv", "r54_presence", "r54_cosine"]
)
assert len(BASE_FEATURE_NAMES) == 37, f"expected 37 base features, got {len(BASE_FEATURE_NAMES)}"

BETA_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Gates (mirror R56/R57b)
GATE_PROD_NDCG_DELTA = 0.010
GATE_EXP_NDCG_DELTA = 0.005
GATE_PROD_CHURN_FRAC = 0.030
GATE_EXP_CHURN_FRAC = 0.015
GATE_SAME_ARTIST_REGRESS_EPS = 0.002

NDCG_EPS = 0.0005  # tolerance for baseline reproduction


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_20(rank):
    if rank > 0 and rank <= 20:
        return 1.0 / np.log2(rank + 1)
    return 0.0


def load_top50_table():
    print(f"{ts()} Loading top-50 table from {TOP50_PATH}...")
    with open(TOP50_PATH, "rb") as f:
        rows = pickle.load(f)
    print(f"  {len(rows)} rows")
    return rows


def load_payload_for_artist():
    print(f"{ts()} Loading payload (artist map for canary metric)...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    return payload


def organize_by_case(rows):
    """Group rows by case_idx. Returns case_idx -> list of rows (sorted by candidate_rank)."""
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_idx"]].append(r)
    for c in by_case:
        by_case[c].sort(key=lambda r: r["candidate_rank"])
    return by_case


def build_feature_matrix(rows, feature_set):
    """Return X (n_rows, n_feat) per the feature_set: 'meta_only' or 'full_51'."""
    n = len(rows)
    if feature_set == "meta_only":
        feat_names = META_FEATURE_COLS
        X = np.zeros((n, len(feat_names)), dtype=np.float64)
        for i, r in enumerate(rows):
            for j, col in enumerate(feat_names):
                X[i, j] = r[col]
    elif feature_set == "full_51":
        feat_names = META_FEATURE_COLS + BASE_FEATURE_NAMES
        X = np.zeros((n, len(feat_names)), dtype=np.float64)
        for i, r in enumerate(rows):
            for j, col in enumerate(META_FEATURE_COLS):
                X[i, j] = r[col]
            X[i, len(META_FEATURE_COLS):] = r["features"]  # 37 floats list
    else:
        raise ValueError(f"unknown feature_set {feature_set}")
    return X, feat_names


def run_cv5_stage2(rows, feature_set, training_filter):
    """Train CV5 stage-2 LightGBM. Returns per-row stage2_score (OOF) + feature importance."""
    X, feat_names = build_feature_matrix(rows, feature_set)
    n_rows = len(rows)
    case_of_row = np.array([r["case_idx"] for r in rows], dtype=np.int64)
    fold_of_row = np.array([r["fold_id"] for r in rows], dtype=np.int64)
    label = np.array([r["gt_flag"] for r in rows], dtype=np.float64)
    # gt_in_top50 per case
    case_gt_in_top50 = {}
    for r in rows:
        if r["gt_flag"]:
            case_gt_in_top50[r["case_idx"]] = True
    for r in rows:
        case_gt_in_top50.setdefault(r["case_idx"], False)

    # Group sizes per case (in order of case_idx values present)
    unique_cases = sorted({r["case_idx"] for r in rows})

    stage2_scores = np.zeros(n_rows, dtype=np.float64)
    importance_sum = np.zeros(len(feat_names), dtype=np.float64)
    n_folds = 0

    for fi in range(5):
        # Train: rows in cases whose fold != fi
        # Eval: rows in cases whose fold == fi
        train_mask = fold_of_row != fi
        eval_mask = fold_of_row == fi

        # Optionally drop training cases whose GT is not in top-50
        if training_filter == "reachable_only":
            train_cases_keep = {c for c in unique_cases
                                if case_gt_in_top50.get(c, False)
                                and (fold_of_row[case_of_row == c][0] != fi)}
            train_mask_idx = np.array([r["case_idx"] in train_cases_keep for r in rows])
            train_mask = train_mask & train_mask_idx
        elif training_filter == "all_groups":
            pass
        else:
            raise ValueError(f"unknown training_filter {training_filter}")

        X_tr = X[train_mask]
        y_tr = label[train_mask]
        X_va = X[eval_mask]
        y_va = label[eval_mask]

        # Group sizes — for train and eval, contiguous case_idx blocks
        # Rows are NOT pre-sorted by case_idx + candidate_rank; ensure groups are correct
        # by sorting within mask.
        def _grouped_indices(mask):
            sel_rows = [(rows[i]["case_idx"], rows[i]["candidate_rank"], i)
                        for i in range(n_rows) if mask[i]]
            sel_rows.sort()
            ordered_idx = [tup[2] for tup in sel_rows]
            group_sizes = []
            last_case = None
            cur = 0
            for tup in sel_rows:
                if tup[0] != last_case:
                    if cur > 0:
                        group_sizes.append(cur)
                    cur = 1
                    last_case = tup[0]
                else:
                    cur += 1
            if cur > 0:
                group_sizes.append(cur)
            return ordered_idx, group_sizes

        tr_idx, tr_groups = _grouped_indices(train_mask)
        va_idx, va_groups = _grouped_indices(eval_mask)
        X_tr_ordered = X[tr_idx]
        y_tr_ordered = label[tr_idx]
        X_va_ordered = X[va_idx]
        y_va_ordered = label[va_idx]

        ds_tr = lgb.Dataset(X_tr_ordered, label=y_tr_ordered,
                            group=tr_groups, feature_name=list(feat_names))
        ds_va = lgb.Dataset(X_va_ordered, label=y_va_ordered,
                            group=va_groups, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(X_va_ordered)
        for k, idx in enumerate(va_idx):
            stage2_scores[idx] = preds[k]
        imp = model.feature_importance(importance_type="gain")
        importance_sum += imp
        n_folds += 1

    avg_importance = (importance_sum / max(n_folds, 1)).tolist()
    return stage2_scores, dict(zip(feat_names, avg_importance))


def z_per_case(scores, case_of_row):
    """Per-case z-normalisation. Returns z-scored array same shape as scores."""
    z = np.zeros_like(scores)
    cases = defaultdict(list)
    for i, c in enumerate(case_of_row):
        cases[int(c)].append(i)
    for _c, idxs in cases.items():
        s = scores[idxs]
        m = s.mean()
        std = s.std(ddof=0)
        if std < 1e-9:
            z[idxs] = s - m
        else:
            z[idxs] = (s - m) / std
    return z


def compute_metrics(rows, final_scores, case_of_row, ta, baseline_gt_idx,
                     baseline_src_union_has_gt):
    """For each case, take top-1 of final_scores within its top-50; also derive HIT/DEMOTED.

    Returns dict with h7, all, same/diff, hist_depth, buckets, etc.
    """
    by_case = defaultdict(list)
    for i, r in enumerate(rows):
        by_case[r["case_idx"]].append((i, r))
    case_indices = sorted(by_case.keys())

    case_top1 = {}
    case_gt_rank_in_top50 = {}
    case_top20_set = {}

    for c in case_indices:
        cur = by_case[c]
        # Build (final_score, row) tuples
        cur_scored = [(final_scores[i], r) for i, r in cur]
        cur_scored.sort(key=lambda t: -t[0])
        case_top1[c] = cur_scored[0][1]["candidate_track_id"]
        # GT rank in this top-50 (1..50 or -1)
        gt_rank = -1
        for rank, (sc, r) in enumerate(cur_scored, start=1):
            if r["gt_flag"]:
                gt_rank = rank
                break
        case_gt_rank_in_top50[c] = gt_rank
        case_top20_set[c] = [r["candidate_track_id"] for _sc, r in cur_scored[:20]]

    n = len(case_indices)
    # We need to know the actual case-level GT and n_prior_music etc. Get it from the
    # original rows.
    case_meta = {}
    for c in case_indices:
        first_row = by_case[c][0][1]
        case_meta[c] = {
            "session_id": first_row["session_id"],
            "gt_in_pool": bool(first_row["gt_in_pool"]),
            "gt_lr_rank": first_row["gt_lr_rank"],
        }

    # Compute final case_lr_rank — what we need is GT's rank in the FINAL ranking.
    # For cases where GT is in our top-50 at some rank R (1..50), final rank = R.
    # For cases where GT is not in top-50 but is in pool@300, we KEEP the LR rank if
    # LR placed GT outside top-20 (rank > 20). If LR placed GT inside top-50 but stage-2
    # demoted it past 50, we set rank > 50 (which contributes 0 to nDCG@20).
    # For nDCG@20 calculation: rank > 20 → 0.
    final_rank = []
    for c in case_indices:
        if case_gt_rank_in_top50[c] > 0:
            final_rank.append(case_gt_rank_in_top50[c])
        else:
            # GT not in our top-50 final ranking. Was it in LR top-50? If LR placed
            # GT in top-50 but stage-2 displaced it past rank 50, this is a regression.
            # For nDCG@20 purposes, rank > 50 is the same as rank > 20 → 0.
            # If GT was not in LR top-50 either, rank stays as LR-rank in pool (which
            # is also > 50 so > 20 → 0).
            lr_rank = case_meta[c]["gt_lr_rank"]
            final_rank.append(lr_rank if lr_rank > 0 else -1)

    # Need n_prior_music — fetch from payload (loaded externally and passed in)
    # We'll require caller to provide this map. Hack: load it here once.
    # Better: precompute it once outside this function.
    return case_indices, final_rank, case_top1, case_top20_set


def compute_full_metrics(rows, final_scores, case_of_row, payload):
    cases = payload["cases"]
    ta = payload["track_artist"]
    case_indices, final_rank, case_top1, case_top20_set = compute_metrics(
        rows, final_scores, case_of_row, ta, None, None,
    )
    n = len(case_indices)
    ndcg = np.array([ndcg_at_20(r) for r in final_rank])

    case_to_pos = {c: i for i, c in enumerate(case_indices)}
    h7_idxs = [case_to_pos[c] for c in case_indices if cases[c]["n_prior_music"] == 7]
    same_art_idxs, diff_art_idxs = [], []
    for c in case_indices:
        gt_a = ta.get(cases[c]["gt"], "")
        played = {ta.get(t, "") for t in cases[c]["music_turns"]} - {""}
        if gt_a and gt_a in played:
            same_art_idxs.append(case_to_pos[c])
        else:
            diff_art_idxs.append(case_to_pos[c])

    # hist depth
    by_depth = defaultdict(list)
    for c in case_indices:
        d = cases[c]["n_prior_music"]
        label = f"h{d}" if d < 7 else "h7+"
        by_depth[label].append(case_to_pos[c])

    return {
        "all_ndcg": float(ndcg.mean()),
        "h7_ndcg": float(ndcg[h7_idxs].mean()) if h7_idxs else 0.0,
        "h7_n": len(h7_idxs),
        "same_artist_ndcg": float(ndcg[same_art_idxs].mean()) if same_art_idxs else 0.0,
        "diff_artist_ndcg": float(ndcg[diff_art_idxs].mean()) if diff_art_idxs else 0.0,
        "same_artist_n": len(same_art_idxs),
        "diff_artist_n": len(diff_art_idxs),
        "by_depth_ndcg": {k: float(ndcg[idxs].mean()) for k, idxs in by_depth.items()},
        "case_top1": case_top1,
        "case_top20": case_top20_set,
        "final_rank_per_case": dict(zip(case_indices, final_rank)),
    }


def verify_baseline(metrics_beta0):
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)
    failures = []
    if abs(metrics_beta0["all_ndcg"] - decomp["overall_ndcg"]) > NDCG_EPS:
        failures.append(f"all_ndcg: refresh={decomp['overall_ndcg']:.5f}  here={metrics_beta0['all_ndcg']:.5f}")
    if abs(metrics_beta0["h7_ndcg"] - decomp["h7_ndcg"]) > NDCG_EPS:
        failures.append(f"h7_ndcg: refresh={decomp['h7_ndcg']:.5f}  here={metrics_beta0['h7_ndcg']:.5f}")
    return failures


def gate_verdict(base_m, var_m, churn_all, churn_h7, n, n_h7, recovered, lost):
    net = recovered - lost
    h7_d = var_m["h7_ndcg"] - base_m["h7_ndcg"]
    sa_d = var_m["same_artist_ndcg"] - base_m["same_artist_ndcg"]
    da_d = var_m["diff_artist_ndcg"] - base_m["diff_artist_ndcg"]
    all_d = var_m["all_ndcg"] - base_m["all_ndcg"]
    if net <= 0:
        return "FAIL_REGRESS", f"net {net} <= 0"
    if sa_d < -GATE_SAME_ARTIST_REGRESS_EPS:
        return "FAIL_REGRESS", f"same_artist Δ={sa_d:+.5f} regresses"
    if all_d < 0:
        return "FAIL_REGRESS", f"all Δ={all_d:+.5f} worse than baseline"
    ca = churn_all / n
    ch = churn_h7 / n_h7
    if (h7_d >= GATE_PROD_NDCG_DELTA and sa_d >= 0
            and ca <= GATE_PROD_CHURN_FRAC and ch <= GATE_PROD_CHURN_FRAC):
        return "PASS_PROD", f"h7 Δ={h7_d:+.5f}"
    if (h7_d >= GATE_EXP_NDCG_DELTA and sa_d >= 0
            and ca <= GATE_EXP_CHURN_FRAC and ch <= GATE_EXP_CHURN_FRAC):
        return "PASS_EXP", f"h7 Δ={h7_d:+.5f}"
    return "FAIL_GATE", f"h7 Δ={h7_d:+.5f}"


def main():
    t0 = time.time()
    print("R58 Phase 2 — Option A stage-2 LightGBM (dev-only)")
    print("=" * 70)

    rows = load_top50_table()
    payload = load_payload_for_artist()
    case_of_row = np.array([r["case_idx"] for r in rows], dtype=np.int64)
    lr_scores_per_row = np.array([r["lr_score"] for r in rows], dtype=np.float64)
    z_lr = z_per_case(lr_scores_per_row, case_of_row)

    # First: beta=0 baseline reproduction (uses any stage-2 since beta=0 zeros it)
    print(f"\n{ts()} Reproducing baseline (beta=0 with zero stage-2)...")
    dummy_stage2 = np.zeros_like(lr_scores_per_row)
    final_b0 = z_lr + 0.0 * dummy_stage2
    base_metrics = compute_full_metrics(rows, final_b0, case_of_row, payload)
    failures = verify_baseline(base_metrics)
    if failures:
        print(f"  BASELINE REPRODUCTION FAILED:")
        for f in failures:
            print(f"    {f}")
        sys.exit(2)
    print(f"  PASS within ε:")
    print(f"    all-dev nDCG: {base_metrics['all_ndcg']:.5f}")
    print(f"    h7 nDCG:      {base_metrics['h7_ndcg']:.5f}")
    print(f"    same_artist:  {base_metrics['same_artist_ndcg']:.5f}")
    print(f"    diff_artist:  {base_metrics['diff_artist_ndcg']:.5f}")

    # Baseline top-1 per case (for churn calc)
    base_top1 = base_metrics["case_top1"]
    base_top20 = base_metrics["case_top20"]
    cases = payload["cases"]
    n_h7 = sum(1 for c in cases if c["n_prior_music"] == 7)
    n = len(cases)
    # Baseline bucket assignment (HIT vs DEMOTED based on LR within top-20)
    base_final_rank = base_metrics["final_rank_per_case"]
    base_bucket = {}
    for c, r in base_final_rank.items():
        if r > 0 and r <= 20:
            base_bucket[c] = "HIT"
        elif r > 0:
            base_bucket[c] = "DEMOTED"
        else:
            base_bucket[c] = "OOT"  # out-of-top-50 in LR; carry-over

    # Sweep configs
    configs = []
    for tf in ["reachable_only", "all_groups"]:
        for fs in ["meta_only", "full_51"]:
            configs.append((tf, fs))

    all_results = {}
    for cfg_idx, (training_filter, feature_set) in enumerate(configs):
        cfg_name = f"{training_filter}+{feature_set}"
        print(f"\n{ts()} Config {cfg_idx + 1}/4: {cfg_name}")
        stage2_scores, importance = run_cv5_stage2(rows, feature_set, training_filter)
        z_s2 = z_per_case(stage2_scores, case_of_row)

        cfg_results = {
            "feature_importance": importance,
            "betas": {},
        }
        for beta in BETA_SWEEP:
            final = z_lr + beta * z_s2
            m = compute_full_metrics(rows, final, case_of_row, payload)
            # Churn vs baseline
            churn_all = sum(1 for c in base_top1 if m["case_top1"][c] != base_top1[c])
            churn_h7 = sum(1 for c in base_top1
                            if cases[c]["n_prior_music"] == 7
                            and m["case_top1"][c] != base_top1[c])
            # Recovered/lost
            recovered = 0
            lost = 0
            for c, rank in m["final_rank_per_case"].items():
                bb = base_bucket.get(c, "OOT")
                if rank > 0 and rank <= 20:
                    nb = "HIT"
                elif rank > 0:
                    nb = "DEMOTED"
                else:
                    nb = "OOT"
                if bb == "DEMOTED" and nb == "HIT":
                    recovered += 1
                elif bb == "HIT" and nb == "DEMOTED":
                    lost += 1
            net = recovered - lost
            verdict, reason = gate_verdict(
                base_metrics, m, churn_all, churn_h7, n, n_h7, recovered, lost,
            )
            cfg_results["betas"][f"{beta:.2f}"] = {
                "beta": beta,
                "h7_ndcg": m["h7_ndcg"],
                "h7_delta": m["h7_ndcg"] - base_metrics["h7_ndcg"],
                "all_ndcg": m["all_ndcg"],
                "all_delta": m["all_ndcg"] - base_metrics["all_ndcg"],
                "same_artist_ndcg": m["same_artist_ndcg"],
                "same_artist_delta": m["same_artist_ndcg"] - base_metrics["same_artist_ndcg"],
                "diff_artist_ndcg": m["diff_artist_ndcg"],
                "diff_artist_delta": m["diff_artist_ndcg"] - base_metrics["diff_artist_ndcg"],
                "by_depth_ndcg": m["by_depth_ndcg"],
                "churn_all": churn_all,
                "churn_h7": churn_h7,
                "churn_all_frac": churn_all / n,
                "churn_h7_frac": churn_h7 / n_h7,
                "recovered": recovered,
                "lost": lost,
                "net_recovery": net,
                "gate_verdict": verdict,
                "gate_reason": reason,
            }
        all_results[cfg_name] = cfg_results

    # Report
    print(f"\n{'=' * 110}")
    print(f"{'config':<32s} | {'β':>5s} | {'h7_Δ':>9s} | {'all_Δ':>9s} | {'sa_Δ':>9s} | {'da_Δ':>9s} | "
          f"{'rec':>4s} | {'lost':>4s} | {'net':>4s} | {'churn%':>6s} | {'h7ch%':>6s} | verdict")
    print("-" * 130)
    best_per_cfg = {}
    for cfg_name, cfg_results in all_results.items():
        for bstr, r in cfg_results["betas"].items():
            print(f"{cfg_name:<32s} | {bstr:>5s} | {r['h7_delta']:+.5f} | {r['all_delta']:+.5f} | "
                  f"{r['same_artist_delta']:+.5f} | {r['diff_artist_delta']:+.5f} | "
                  f"{r['recovered']:>4d} | {r['lost']:>4d} | {r['net_recovery']:>+4d} | "
                  f"{r['churn_all_frac']:>5.2%} | {r['churn_h7_frac']:>5.2%} | {r['gate_verdict']}")
        # Track best beta per config (max h7 among PASS_*, else max h7 overall)
        passing = [r for r in cfg_results["betas"].values()
                    if r["gate_verdict"] in ("PASS_PROD", "PASS_EXP")]
        if passing:
            best = max(passing, key=lambda r: r["h7_delta"])
        else:
            best = max(cfg_results["betas"].values(), key=lambda r: r["h7_delta"])
        best_per_cfg[cfg_name] = best

    print(f"\n{ts()} Best beta per config:")
    for cfg, b in best_per_cfg.items():
        print(f"  {cfg}: β={b['beta']:.2f}  h7 Δ={b['h7_delta']:+.5f}  verdict={b['gate_verdict']}")

    # Final decision
    passing_configs = {cfg: b for cfg, b in best_per_cfg.items()
                       if b["gate_verdict"] in ("PASS_PROD", "PASS_EXP")}
    if passing_configs:
        # Pick globally best by h7
        best_cfg = max(passing_configs.items(), key=lambda kv: kv[1]["h7_delta"])
        print(f"\n  BEST PASSING CONFIG: {best_cfg[0]}  β={best_cfg[1]['beta']:.2f}  "
              f"verdict={best_cfg[1]['gate_verdict']}  h7 Δ={best_cfg[1]['h7_delta']:+.5f}")
    else:
        print(f"\n  NO CONFIG PASSED GATES.")
        # Best by h7 alone, for diagnostic
        best_diag = max(best_per_cfg.items(), key=lambda kv: kv[1]["h7_delta"])
        print(f"  Best by h7 alone (diagnostic): {best_diag[0]}  β={best_diag[1]['beta']:.2f}  "
              f"h7 Δ={best_diag[1]['h7_delta']:+.5f}  verdict={best_diag[1]['gate_verdict']}  "
              f"reason: {best_diag[1]['gate_reason']}")

    out_data = {
        "baseline": {
            "h7_ndcg": base_metrics["h7_ndcg"],
            "all_ndcg": base_metrics["all_ndcg"],
            "same_artist_ndcg": base_metrics["same_artist_ndcg"],
            "diff_artist_ndcg": base_metrics["diff_artist_ndcg"],
        },
        "configs": {cfg: {"feature_importance": cfg_results["feature_importance"],
                           "betas": cfg_results["betas"]}
                     for cfg, cfg_results in all_results.items()},
        "best_per_config": best_per_cfg,
        "gates": {
            "production_h7_delta": GATE_PROD_NDCG_DELTA,
            "exploratory_h7_delta": GATE_EXP_NDCG_DELTA,
            "production_churn_frac": GATE_PROD_CHURN_FRAC,
            "exploratory_churn_frac": GATE_EXP_CHURN_FRAC,
            "same_artist_regress_eps": GATE_SAME_ARTIST_REGRESS_EPS,
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\n{ts()} Saved: {OUT}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
