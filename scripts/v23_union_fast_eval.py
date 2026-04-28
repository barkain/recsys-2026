#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Fast no-LLM evaluation for v23 deep-retrieval union rankers.

This intentionally skips the expensive postrank-style Powell scorer and uses
vectorized source-rank features for controlled R0/R1/R2/R4 checks.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_v23_union_retrieval import (
    ARTIFACT,
    OUT_RANKERS,
    OUT_RECALL,
    build_cases,
    cv_folds,
    split_indices,
)

RANK_KEYS = {
    "A": "A_v23_pool50",
    "B": "B_last_music_meta_500",
    "C": "C_full_history_500",
    "D": "D_track_neighbors_200",
}


def ndcg_from_rank(rank: int | None) -> float:
    if rank is None or rank > 20:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def std(vals: list[float]) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def source_rank_score(cases: list[dict[str, Any]], source: str, idxs: list[int]) -> float:
    return mean([ndcg_from_rank(cases[i]["source_ranks"][RANK_KEYS[source]]) for i in idxs])


def build_rank_matrices(cases: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[int | None]]:
    mats: list[np.ndarray] = []
    gt_idxs: list[int | None] = []
    for row in cases:
        union = row["union"]
        rank_mat = np.zeros((len(union), 4), dtype=np.float64)
        for col, source in enumerate("ABCD"):
            rank_map = {tid: rank for rank, tid in enumerate(row["sources"][source], start=1)}
            for j, tid in enumerate(union):
                rank = rank_map.get(tid)
                if rank is not None:
                    rank_mat[j, col] = 1.0 / (60.0 + rank)
        mats.append(rank_mat)
        gt_idxs.append(union.index(row["gt"]) if row["gt"] in union else None)
    return mats, gt_idxs


def rank_from_scores(scores: np.ndarray, gt_idx: int | None) -> int | None:
    if gt_idx is None:
        return None
    gt_score = scores[gt_idx]
    if gt_score <= 0:
        return None
    # Stable tie break by union order.
    higher = np.count_nonzero(scores > gt_score)
    tied_before = np.count_nonzero(scores[:gt_idx] == gt_score)
    return int(higher + tied_before + 1)


def weighted_rrf_score(mats: list[np.ndarray], gt_idxs: list[int | None], weights: tuple[float, ...], idxs: list[int]) -> float:
    w = np.asarray(weights, dtype=np.float64)
    vals = []
    for i in idxs:
        scores = mats[i] @ w
        vals.append(ndcg_from_rank(rank_from_scores(scores, gt_idxs[i])))
    return mean(vals)


def r1_prefix_score(
    cases: list[dict[str, Any]],
    mats: list[np.ndarray],
    gt_idxs: list[int | None],
    prefix: int,
    idxs: list[int],
) -> float:
    vals = []
    w_bcd = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    for i in idxs:
        rank_a = cases[i]["source_ranks"][RANK_KEYS["A"]]
        if rank_a is not None and rank_a <= prefix:
            vals.append(ndcg_from_rank(rank_a))
            continue
        gt_idx = gt_idxs[i]
        if gt_idx is None:
            vals.append(0.0)
            continue
        scores = mats[i] @ w_bcd
        gt_score = scores[gt_idx]
        if gt_score <= 0:
            vals.append(0.0)
            continue
        prefix_set = set(cases[i]["sources"]["A"][:prefix])
        rank_after_prefix = 1
        union = cases[i]["union"]
        for j, score in enumerate(scores):
            if union[j] in prefix_set:
                continue
            if score > gt_score or (score == gt_score and j < gt_idx):
                rank_after_prefix += 1
        vals.append(ndcg_from_rank(prefix + rank_after_prefix))
    return mean(vals)


def cv_eval(fn, folds: list[list[int]]) -> tuple[float, float]:
    vals = [fn(fold) for fold in folds]
    return mean(vals), std(vals)


def tune_grid(fn, grid: list[Any], train_idx: list[int]) -> tuple[Any, float]:
    best_param = None
    best_score = -1.0
    for param in grid:
        score = fn(param, train_idx)
        if score > best_score:
            best_param = param
            best_score = score
    return best_param, best_score


def main() -> None:
    os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"
    print("building cases", flush=True)
    cases, recall = build_cases()
    mats, gt_idxs = build_rank_matrices(cases)
    sessions = [row["session_id"] for row in cases]
    train_idx, hold_idx = split_indices(sessions)
    folds = cv_folds(sessions)

    print("evaluating fixed rankers", flush=True)
    rankers: dict[str, Any] = {}
    source_names = {
        "R0_v23_only_in_union": "A",
        "R4a_last_music_meta_500_alone": "B",
        "R4b_full_history_500_alone": "C",
        "R4c_track_neighbors_200_alone": "D",
    }
    for name, source in source_names.items():
        cv_mean, cv_std = cv_eval(lambda idxs, s=source: source_rank_score(cases, s, idxs), folds)
        rankers[name] = {
            "holdout_ndcg": source_rank_score(cases, source, hold_idx),
            "cv5_mean": cv_mean,
            "cv5_std": cv_std,
            "params": {"source": source},
        }

    print("tuning R1 prefix", flush=True)
    prefix_grid = [{"prefix": p} for p in [10, 15, 20, 30]]

    def r1_fn(p: dict[str, int], idxs: list[int]) -> float:
        return r1_prefix_score(cases, mats, gt_idxs, p["prefix"], idxs)

    best_p, train_score = tune_grid(r1_fn, prefix_grid, train_idx)
    cv_scores = []
    cv_params = []
    for fold in folds:
        tr = [i for i in range(len(cases)) if i not in set(fold)]
        p, _ = tune_grid(r1_fn, prefix_grid, tr)
        cv_params.append(p)
        cv_scores.append(r1_fn(p, fold))
    rankers["R1_v23_prefix_plus_fused"] = {
        "train_ndcg": train_score,
        "holdout_ndcg": r1_fn(best_p, hold_idx),
        "cv5_mean": mean(cv_scores),
        "cv5_std": std(cv_scores),
        "params": best_p,
        "cv_params": cv_params,
    }

    print("tuning R2 weighted RRF", flush=True)
    weight_grid = [
        {"weights": tuple(float(x) for x in weights)}
        for weights in __import__("itertools").product(
            [1.0, 2.0, 4.0, 8.0],
            [0.0, 0.5, 1.0, 2.0],
            [0.0, 0.5, 1.0, 2.0],
            [0.0, 0.5, 1.0, 2.0],
        )
    ]
    def r2_fn(p: dict[str, tuple[float, ...]], idxs: list[int]) -> float:
        return weighted_rrf_score(mats, gt_idxs, p["weights"], idxs)

    best_w, train_score = tune_grid(r2_fn, weight_grid, train_idx)
    cv_scores = []
    cv_params = []
    for fold in folds:
        tr = [i for i in range(len(cases)) if i not in set(fold)]
        p, _ = tune_grid(r2_fn, weight_grid, tr)
        cv_params.append(p)
        cv_scores.append(r2_fn(p, fold))
    rankers["R2_weighted_rrf"] = {
        "train_ndcg": train_score,
        "holdout_ndcg": r2_fn(best_w, hold_idx),
        "cv5_mean": mean(cv_scores),
        "cv5_std": std(cv_scores),
        "params": best_w,
        "cv_params": cv_params,
    }

    for row in rankers.values():
        row["delta_vs_v23_raw_0.0892"] = row["holdout_ndcg"] - 0.0892
        row["delta_vs_tuned_pool_0.0912"] = row["holdout_ndcg"] - 0.0912

    result = {
        "artifact": ARTIFACT,
        "n": len(cases),
        "split": "100_100_seed0",
        "train_n": len(train_idx),
        "holdout_n": len(hold_idx),
        "skipped": ["R3_linear_union_scorer"],
        "skip_reason": "Vectorized fast pass; R3 Powell scorer intentionally skipped pending cheap-ranker signal.",
        "rankers": rankers,
    }

    Path("exp/eval").mkdir(parents=True, exist_ok=True)
    with open(OUT_RECALL, "w", encoding="utf-8") as f:
        json.dump(recall, f, indent=2)
    with open(OUT_RANKERS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "union_hit_rate": recall["union_hit_rate"],
        "source_hit_rates": recall["source_hit_rates"],
        "rankers": {
            key: {
                "holdout_ndcg": value["holdout_ndcg"],
                "cv5_mean": value["cv5_mean"],
                "cv5_std": value["cv5_std"],
                "params": value["params"],
            }
            for key, value in rankers.items()
        },
        "artifacts": [OUT_RECALL, OUT_RANKERS],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
