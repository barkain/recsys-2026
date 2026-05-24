"""R84c margin transferability check.

Compare OOF sibling-R54-LR top-1 margins (used by R84c routing) to frozen R54c
production LR top-1 margins (what blind will use). If distributions match,
the raw thresholds (0.5, 2.0) transfer cleanly. If not, compute quantile-mapped
thresholds to deploy on blind.

Output: exp/eval/expR84c_margin_transfer.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_blind_submission import FEAT_ALL  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402

N_FOLDS = 5
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR84c_margin_transfer.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def train_sibling_lr(case_features, train_idx, feat_key, feat_names):
    X, y, gt = [], [], []
    for i in train_idx:
        cf = case_features[i]
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf[feat_key][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=feat_names)
    return lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def margin_of(scores):
    s = np.sort(scores)[::-1]
    return float(s[0] - s[1]) if len(s) >= 2 else 0.0


def main():
    print(f"{ts()} R84c margin transferability check")
    print("=" * 70)

    # Load case features cache
    print(f"\n{ts()} Loading case_features cache...", flush=True)
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)
    n = len(case_features)
    print(f"  {n} cases loaded")

    # Load fold map
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])

    # Pre-compute OOF sibling-R54-LR margins per case
    print(f"\n{ts()} Pre-computing OOF sibling-R54-LR margins (5-fold)...", flush=True)
    oof_margins = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = [i for i in range(n) if case_fold[i] == fold_k]
        t = time.time()
        lr = train_sibling_lr(case_features, train_idx, "feats_r54", list(FEAT_ALL))
        for i in eval_idx:
            scores = lr.predict(case_features[i]["feats_r54"])
            oof_margins[i] = margin_of(scores)
        print(f"  fold {fold_k}: {time.time() - t:.0f}s", flush=True)

    # Frozen R54c LR margins on dev (in-sample, what blind will use)
    print(f"\n{ts()} Computing frozen R54c LR margins on dev (in-sample)...", flush=True)
    frozen_lr = lgb.Booster(model_file=str(R54_LR))
    frozen_margins = {}
    t = time.time()
    for i in range(n):
        scores = frozen_lr.predict(case_features[i]["feats_r54"])
        frozen_margins[i] = margin_of(scores)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t:.0f}s)", flush=True)

    # Compare distributions
    oof_arr = np.array([oof_margins[i] for i in range(n)])
    fro_arr = np.array([frozen_margins[i] for i in range(n)])

    def summary(arr):
        return {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    s_oof = summary(oof_arr)
    s_fro = summary(fro_arr)

    print(f"\n  {'stat':6}  {'OOF sibling':>12}  {'frozen R54c':>12}")
    for k in ["mean", "std", "p10", "p25", "p50", "p75", "p90", "p95", "max"]:
        print(f"  {k:6}  {s_oof[k]:12.4f}  {s_fro[k]:12.4f}")

    # Raw threshold route rate
    LOW_RAW, HIGH_RAW = 0.5, 2.0
    oof_routed_r84 = sum(1 for v in oof_arr if v < LOW_RAW or v >= HIGH_RAW)
    fro_routed_r84 = sum(1 for v in fro_arr if v < LOW_RAW or v >= HIGH_RAW)
    print(f"\n  Raw threshold ({LOW_RAW}, {HIGH_RAW}) route rate:")
    print(f"    OOF sibling:  {oof_routed_r84}/{n} ({oof_routed_r84/n:.1%})")
    print(f"    frozen R54c:  {fro_routed_r84}/{n} ({fro_routed_r84/n:.1%})")

    # Jaccard between OOF-routed cases and frozen-margin-routed cases
    oof_set = {i for i in range(n) if oof_arr[i] < LOW_RAW or oof_arr[i] >= HIGH_RAW}
    fro_set = {i for i in range(n) if fro_arr[i] < LOW_RAW or fro_arr[i] >= HIGH_RAW}
    inter = len(oof_set & fro_set)
    union = len(oof_set | fro_set)
    jaccard = inter / union if union else 0.0
    print(f"  Jaccard(OOF routed, frozen routed): {jaccard:.3f}  "
          f"(intersection={inter}, union={union})")

    # Quantile-mapped thresholds
    pct_low = float((oof_arr < LOW_RAW).mean())
    pct_high = float((oof_arr >= HIGH_RAW).mean())
    LOW_QM = float(np.percentile(fro_arr, pct_low * 100))
    HIGH_QM = float(np.percentile(fro_arr, 100 - pct_high * 100))
    fro_routed_qm = sum(1 for v in fro_arr if v < LOW_QM or v >= HIGH_QM)
    print(f"\n  Quantile-mapped thresholds (matching OOF route rates):")
    print(f"    pct(margin<{LOW_RAW}) in OOF = {pct_low:.3f}  → "
          f"frozen p{pct_low*100:.1f} = {LOW_QM:.4f}")
    print(f"    pct(margin>={HIGH_RAW}) in OOF = {pct_high:.3f}  → "
          f"frozen p{100-pct_high*100:.1f} = {HIGH_QM:.4f}")
    print(f"    frozen routed under QM: {fro_routed_qm}/{n} "
          f"({fro_routed_qm/n:.1%})")

    qm_set = {i for i in range(n) if fro_arr[i] < LOW_QM or fro_arr[i] >= HIGH_QM}
    jaccard_qm = len(oof_set & qm_set) / len(oof_set | qm_set) if (oof_set | qm_set) else 0
    print(f"  Jaccard(OOF routed, frozen QM routed): {jaccard_qm:.3f}")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    if jaccard >= 0.8:
        rec = "USE_RAW_THRESHOLDS"
        why = f"Raw thresholds transfer cleanly (Jaccard={jaccard:.3f} ≥ 0.8)"
    elif jaccard_qm > jaccard + 0.05:
        rec = "USE_QUANTILE_MAPPED"
        why = f"QM materially improves transfer (Jaccard {jaccard:.3f} → {jaccard_qm:.3f})"
    else:
        rec = "USE_RAW_THRESHOLDS"
        why = "Raw transfer is acceptable; QM doesn't help much"
    print(f"    {rec}: {why}")

    out = {
        "experiment": "R84c margin transferability check",
        "created_at": datetime.now().isoformat(),
        "n_cases": n,
        "oof_sibling_margin_stats": s_oof,
        "frozen_r54c_margin_stats": s_fro,
        "raw_thresholds": {"low": LOW_RAW, "high": HIGH_RAW},
        "raw_route_rates": {
            "oof_sibling": oof_routed_r84 / n,
            "frozen_r54c": fro_routed_r84 / n,
            "jaccard": jaccard,
        },
        "quantile_mapped_thresholds": {
            "low": LOW_QM, "high": HIGH_QM,
            "frozen_route_rate": fro_routed_qm / n,
            "jaccard_with_oof": jaccard_qm,
        },
        "recommendation": {
            "rule": rec,
            "rationale": why,
            "thresholds_to_use": (
                {"low": LOW_RAW, "high": HIGH_RAW}
                if rec == "USE_RAW_THRESHOLDS"
                else {"low": LOW_QM, "high": HIGH_QM}
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
