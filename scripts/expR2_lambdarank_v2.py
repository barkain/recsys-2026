# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Experiment R2 v2 — conservative LambdaRank hyperparameters.

Same structure as expR2_lambdarank.py, but with a much more aggressive
regularization to test if LightGBM can match Powell at all. Also adds Spearman
rank-correlation constraint between LightGBM and Powell on val set as a
diagnostic.

We test 3 hp configs:
  hp_tiny:  num_leaves=7, n_estimators=80, min_child=80, lr=0.03 (heavy reg)
  hp_small: num_leaves=15, n_estimators=120, min_child=40, lr=0.04
  hp_orig:  num_leaves=31, n_estimators=200, min_child=20, lr=0.05 (matches v1)

Keeps 5 seeds × 5 folds. Skips B0/B2/B4 (we already have them from v1).
Reports B1 (Powell, refit per fold for fresh comparison) + B3_<hp> for each
HP config.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXPR2_FEATURES = REPO_ROOT / "exp" / "eval" / "_expR2_features.pkl"
OUT_RESULTS = REPO_ROOT / "exp" / "eval" / "expR2_lambdarank_v2_results.json"

NDCG_K = 20
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

POWELL_INIT = {
    "orig_rank": 3.0,
    "same_artist_last_played": 1.0,
    "tag_overlap_last_played": 0.3,
    "artist_token_overlap_query": 0.5,
    "title_token_overlap_query": 0.5,
    "all_history_user_token_overlap": 0.1,
    "already_played": -2.0,
    "recency_weighted_metadata": 0.5,
}


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def split_order(sessions: list[str], seed: int) -> list[int]:
    return sorted(range(len(sessions)),
                  key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))


def cv_folds(sessions: list[str], seed: int, k: int = N_FOLDS) -> list[np.ndarray]:
    order = split_order(sessions, seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(idx)
    return [np.asarray(f, dtype=np.int64) for f in folds]


def vec_ndcg(X_sub: np.ndarray, gt_sub: np.ndarray, sizes_sub: np.ndarray, weights: np.ndarray) -> float:
    pool_axis = np.arange(X_sub.shape[1])[None, :]
    valid_pool = pool_axis < sizes_sub[:, None]
    scores = X_sub @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_sub >= 0
    safe_gt = np.where(has_gt, gt_sub, 0)
    gt_scores = scores[np.arange(X_sub.shape[0]), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def vec_ndcg_per_session(X_sub: np.ndarray, gt_sub: np.ndarray, sizes_sub: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pool_axis = np.arange(X_sub.shape[1])[None, :]
    valid_pool = pool_axis < sizes_sub[:, None]
    scores = X_sub @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_sub >= 0
    safe_gt = np.where(has_gt, gt_sub, 0)
    gt_scores = scores[np.arange(X_sub.shape[0]), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    return np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0).astype(np.float64)


def eval_per_session_from_scores(scores: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    pool_axis = np.arange(scores.shape[1])[None, :]
    valid_pool = pool_axis < sizes[:, None]
    s = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_idx >= 0
    safe_gt = np.where(has_gt, gt_idx, 0)
    gt_scores = s[np.arange(scores.shape[0]), safe_gt]
    strict_gt = (s > gt_scores[:, None]).sum(axis=1)
    tie_before = ((s == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    return np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0).astype(np.float64)


def fit_powell(X_train: np.ndarray, gt_train: np.ndarray, sizes_train: np.ndarray, feat_names: list[str]) -> np.ndarray:
    from scipy.optimize import minimize  # type: ignore
    init = np.asarray([POWELL_INIT.get(fn, 0.0) for fn in feat_names], dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        return -vec_ndcg(X_train, gt_train, sizes_train, w)

    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return np.asarray(res.x, dtype=np.float64)


def reshape_for_lgbm(X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray, idx: np.ndarray):
    K = X.shape[1]
    F = X.shape[2]
    Xs = X[idx].reshape(-1, F)
    rows = len(idx)
    labels = np.zeros(rows * K, dtype=np.float64)
    for k_, i in enumerate(idx):
        gp = gt_idx[i]
        if gp >= 0:
            labels[k_ * K + gp] = 1.0
    groups = np.full(rows, K, dtype=np.int64)
    return Xs, labels, groups


def lgbm_train_lambdarank(X_tr, y_tr, group_tr, X_val, y_val, group_val, seed, params):
    import lightgbm as lgb  # type: ignore
    train_set = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
    valid_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
    p = dict(params)
    p["objective"] = "lambdarank"
    p["metric"] = "ndcg"
    p["ndcg_eval_at"] = [NDCG_K]
    p["lambdarank_truncation_level"] = NDCG_K
    p["seed"] = seed
    p["verbosity"] = -1
    p["num_threads"] = 0
    n_est = p.pop("n_estimators", 200)
    booster = lgb.train(
        params=p,
        train_set=train_set,
        num_boost_round=n_est,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return booster


def predict_lgbm_per_session(booster, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    K = X.shape[1]
    F = X.shape[2]
    flat = X[idx].reshape(-1, F)
    preds = booster.predict(flat)
    return preds.reshape(len(idx), K)


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT R2 v2 — conservative LambdaRank hyperparameters", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    print("[Setup] loading expR2 features", flush=True)
    with open(EXPR2_FEATURES, "rb") as f:
        features = pickle.load(f)  # noqa: S301
    X = features["X"]
    gt_idx = features["gt_idx"]
    sizes = features["sizes"]
    feat_names = list(features["feat_names"])
    sessions = features["sessions"]
    n_sess = X.shape[0]
    print(f"  X shape={X.shape}, sessions={n_sess}", flush=True)

    HP_CONFIGS = [
        ("hp_tiny", {"num_leaves": 7, "learning_rate": 0.03,
                     "min_child_samples": 80, "n_estimators": 100,
                     "lambda_l2": 1.0,
                     "feature_fraction": 0.8, "bagging_fraction": 0.8,
                     "bagging_freq": 1, "min_gain_to_split": 1e-3}),
        ("hp_small", {"num_leaves": 15, "learning_rate": 0.04,
                      "min_child_samples": 40, "n_estimators": 150,
                      "lambda_l2": 0.5,
                      "feature_fraction": 0.9, "bagging_fraction": 0.9,
                      "bagging_freq": 1, "min_gain_to_split": 1e-4}),
    ]

    results: dict[str, Any] = {"B1_powell": {"per_seed": []}}
    for name, _ in HP_CONFIGS:
        results[f"B3_{name}"] = {"per_seed": [], "hp": _}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        folds = cv_folds(sessions, seed)
        b1_perfold = []
        b3_perfold = {name: [] for name, _ in HP_CONFIGS}

        for fi, fold in enumerate(folds):
            held = set(fold.tolist())
            train = np.asarray([i for i in range(n_sess) if i not in held], dtype=np.int64)
            X_tr_full = X[train]
            X_held_full = X[fold]

            # B1
            wts = fit_powell(X_tr_full, gt_idx[train], sizes[train], feat_names)
            persess_b1 = vec_ndcg_per_session(X_held_full, gt_idx[fold], sizes[fold], wts)
            b1_perfold.append(float(persess_b1.mean()))

            # 80/20 inner split for early stopping
            rng = np.random.default_rng(seed * 100 + fi)
            shuffled = rng.permutation(train)
            split = int(0.85 * len(shuffled))
            inner_tr = shuffled[:split]
            inner_va = shuffled[split:]
            Xtr_flat, ytr_flat, gtr = reshape_for_lgbm(X, gt_idx, sizes, inner_tr)
            Xva_flat, yva_flat, gva = reshape_for_lgbm(X, gt_idx, sizes, inner_va)

            for name, hp in HP_CONFIGS:
                booster = lgbm_train_lambdarank(
                    Xtr_flat, ytr_flat, gtr, Xva_flat, yva_flat, gva, seed, dict(hp)
                )
                scores = predict_lgbm_per_session(booster, X, fold)
                persess = eval_per_session_from_scores(scores, gt_idx[fold], sizes[fold])
                b3_perfold[name].append(float(persess.mean()))

            line = f"  seed{seed} fold{fi}: B1={b1_perfold[-1]:.4f}"
            for name, _ in HP_CONFIGS:
                line += f" {name}={b3_perfold[name][-1]:.4f}"
            print(line, flush=True)

        results["B1_powell"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b1_perfold)),
            "cv5_std": float(np.std(b1_perfold, ddof=1)),
            "per_fold": [float(x) for x in b1_perfold],
        })
        for name, _ in HP_CONFIGS:
            results[f"B3_{name}"]["per_seed"].append({
                "seed": seed,
                "cv5_mean": float(np.mean(b3_perfold[name])),
                "cv5_std": float(np.std(b3_perfold[name], ddof=1)),
                "per_fold": [float(x) for x in b3_perfold[name]],
            })

    for key in results:
        means = [s["cv5_mean"] for s in results[key]["per_seed"]]
        results[key]["aggregate"] = {
            "mean_cv5": float(np.mean(means)),
            "std_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "min": float(np.min(means)),
            "max": float(np.max(means)),
        }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w") as f:
        json.dump({
            "meta": {"total_time_sec": time.time() - t_start, "seeds": SEEDS,
                     "n_folds": N_FOLDS, "ndcg_k": NDCG_K},
            "results": results,
        }, f, indent=2)
    print(f"\n  wrote {OUT_RESULTS}", flush=True)

    print("\nFinal aggregates:")
    for key in results:
        a = results[key]["aggregate"]
        print(f"  {key}: mean={a['mean_cv5']:.4f} ± {a['std_of_means']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
