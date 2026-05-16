#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R60 Variant A fold-0 diagnostic.

Approved scope only:
  - dev-only fold 0;
  - learned-admission@300 pools;
  - 37-feature R54c-compatible LambdaRank retraining;
  - no blind access and no submission packaging.

Outputs:
  exp/eval/expR60_variant_a_fold0.json
  docs/r60_variant_a_fold0_result.md
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR54_phase3_blind_submission import FEAT_ALL  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    featurize_for_frozen_lr,
    load_supporting_maps,
    ndcg_from_rank,
    percentile_summary,
    same_artist_case,
    score_pool,
)
from scripts.expS2_lambdarank_grouped import grouped_session_folds  # noqa: E402

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
R59_PHASE2_JSON = REPO / "exp" / "eval" / "expR59_c3_phase2_frozen_lr.json"
C3_ADMISSION_JSON = REPO / "exp" / "eval" / "expR59_c3_pool_admission.json"
OUT_JSON = REPO / "exp" / "eval" / "expR60_variant_a_fold0.json"
OUT_MD = REPO / "docs" / "r60_variant_a_fold0_result.md"

POOL_K = 300
TOP_K = 20
BASELINE_EPS = 0.0005
EXPECTED_LR_FEATURES = 37
EXPECTED_ADMISSION_FEATURES = 99

SCOREBOARD_TARGETS = {
    "all": 0.31588,
    "h7": 0.34838,
    "same_artist": 0.62821,
    "diff_artist": 0.14237,
}
EXPECTED_RRF_POOL_HIT = 0.62200


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def mean_for_indices(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(values[np.asarray(indices, dtype=np.int64)].mean())


def split_indices(case_ids: list[int], cases: list[dict[str, Any]], track_artist: dict[str, str]) -> dict[str, list[int]]:
    same = [idx for idx in case_ids if same_artist_case(cases[idx], track_artist)]
    same_set = set(same)
    return {
        "all": list(case_ids),
        "h7": [idx for idx in case_ids if int(cases[idx]["n_prior_music"]) == 7],
        "same_artist": same,
        "diff_artist": [idx for idx in case_ids if idx not in same_set],
    }


def metrics_for_splits(
    values: np.ndarray,
    splits: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split_name, indices in splits.items():
        out[split_name] = {
            "n": len(indices),
            "ndcg_at_20": mean_for_indices(values, indices),
        }
    return out


def delta_metrics(
    baseline: dict[str, dict[str, Any]],
    variant: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split_name in ["all", "h7", "same_artist", "diff_artist"]:
        out[split_name] = {
            "n": variant[split_name]["n"],
            "delta_ndcg_at_20": variant[split_name]["ndcg_at_20"] - baseline[split_name]["ndcg_at_20"],
        }
    return out


def top_feature_importance(model: lgb.Booster, limit: int = 15) -> list[dict[str, Any]]:
    gains = model.feature_importance(importance_type="gain")
    rows = [
        {"feature": name, "gain": float(gain)}
        for name, gain in zip(FEAT_ALL, gains, strict=True)
    ]
    return sorted(rows, key=lambda row: row["gain"], reverse=True)[:limit]


def validate_contracts(ranker: lgb.Booster, admission_artifact: dict[str, Any]) -> dict[str, Any]:
    lr_feature_count = len(FEAT_ALL)
    admission_feature_count = len(c3.FEATURE_NAMES)
    artifact_admission_feature_count = len(admission_artifact.get("feature_names", []))
    if lr_feature_count != EXPECTED_LR_FEATURES:
        raise RuntimeError(f"Variant A feature count mismatch: {lr_feature_count} != {EXPECTED_LR_FEATURES}")
    if ranker.num_feature() != EXPECTED_LR_FEATURES:
        raise RuntimeError(f"Frozen LR model feature count mismatch: {ranker.num_feature()} != {EXPECTED_LR_FEATURES}")
    if admission_feature_count != EXPECTED_ADMISSION_FEATURES:
        raise RuntimeError(
            f"C3 admission feature count mismatch: {admission_feature_count} != {EXPECTED_ADMISSION_FEATURES}"
        )
    if artifact_admission_feature_count != EXPECTED_ADMISSION_FEATURES:
        raise RuntimeError(
            "C3 admission artifact feature count mismatch: "
            f"{artifact_admission_feature_count} != {EXPECTED_ADMISSION_FEATURES}"
        )
    return {
        "variant_a_feature_count": lr_feature_count,
        "variant_a_feature_names": list(FEAT_ALL),
        "c3_admission_feature_count": admission_feature_count,
        "c3_admission_artifact_feature_count": artifact_admission_feature_count,
        "c3_admission_feature_names": list(c3.FEATURE_NAMES),
    }


def score_rrf_baseline(
    ranker: lgb.Booster,
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
) -> dict[str, Any]:
    print(f"{ts()} Recomputing RRF-pool frozen-LR scoreboard baseline...", flush=True)
    cases = payload["cases"]
    n_cases = len(cases)
    ndcg20 = np.zeros(n_cases, dtype=np.float64)
    gt_rank = np.full(n_cases, -1, dtype=np.int32)
    gt_in_pool = np.zeros(n_cases, dtype=bool)
    top1: list[str] = [""] * n_cases
    top20: list[list[str]] = [[] for _ in range(n_cases)]

    start_time = time.time()
    for case_idx, case in enumerate(cases):
        source_lists = c3.make_source_lists(
            payload,
            r21_source,
            r54_source,
            case_index["als_source"],
            case_idx,
        )
        result = score_pool(
            ranker,
            case_index["baseline_pools"][case_idx],
            source_lists,
            case,
            maps,
            als_factors,
            als_to_idx,
            case_index["als_session_vecs"][case_idx],
            track_pop,
            max_pop,
            track_album,
            case_index["r54_scores"][case_idx],
            rrf_full_rank_map=None,
        )
        ndcg20[case_idx] = float(result["ndcg_at_20"])
        gt_rank[case_idx] = int(result["gt_rank"])
        gt_in_pool[case_idx] = bool(result["gt_in_pool"])
        top1[case_idx] = str(result["top1"])
        top20[case_idx] = list(result["top20"])
        if (case_idx + 1) % 1000 == 0:
            print(f"  baseline scored {case_idx + 1}/{n_cases} ({time.time() - start_time:.0f}s)", flush=True)

    return {
        "ndcg_at_20": ndcg20,
        "gt_rank": gt_rank,
        "gt_in_pool": gt_in_pool,
        "top1": top1,
        "top20": top20,
    }


def baseline_reproduction_report(
    baseline_scores: dict[str, Any],
    cases: list[dict[str, Any]],
    maps: dict[str, Any],
    r59_phase2: dict[str, Any],
) -> dict[str, Any]:
    all_splits = split_indices(list(range(len(cases))), cases, maps["track_artist"])
    computed = metrics_for_splits(baseline_scores["ndcg_at_20"], all_splits)
    loaded_rrf = r59_phase2["metrics"]["rrf"]
    metric_rows = {}
    pass_all = True
    for split_name in ["all", "h7", "same_artist", "diff_artist"]:
        expected_from_artifact = float(loaded_rrf[split_name]["ndcg_at_20"])
        target = SCOREBOARD_TARGETS[split_name]
        computed_value = computed[split_name]["ndcg_at_20"]
        abs_delta = abs(computed_value - expected_from_artifact)
        target_abs_delta = abs(computed_value - target)
        passed = abs_delta <= BASELINE_EPS and target_abs_delta <= BASELINE_EPS
        pass_all = pass_all and passed
        metric_rows[split_name] = {
            "n": computed[split_name]["n"],
            "target": target,
            "artifact_value": expected_from_artifact,
            "computed": computed_value,
            "abs_delta_vs_artifact": abs_delta,
            "abs_delta_vs_target": target_abs_delta,
            "epsilon": BASELINE_EPS,
            "pass": passed,
        }
    pool_hit = float(np.mean(baseline_scores["gt_in_pool"]))
    pool_hit_abs_delta = abs(pool_hit - EXPECTED_RRF_POOL_HIT)
    pool_hit_pass = pool_hit_abs_delta <= BASELINE_EPS
    pass_all = pass_all and pool_hit_pass
    return {
        "source_artifact": str(R59_PHASE2_JSON),
        "metrics": metric_rows,
        "rrf_pool_hit_at_300": {
            "target": EXPECTED_RRF_POOL_HIT,
            "computed": pool_hit,
            "abs_delta": pool_hit_abs_delta,
            "epsilon": BASELINE_EPS,
            "pass": pool_hit_pass,
        },
        "pass": pass_all,
    }


def learned_pools_from_oof_scores(
    payload: dict[str, Any],
    case_index: dict[str, Any],
    oof_scores: list[np.ndarray | None],
    buckets: list[str],
) -> dict[str, Any]:
    cases = payload["cases"]
    n_cases = len(cases)
    learned_pools: list[list[str]] = [[] for _ in range(n_cases)]
    learned_hit = np.zeros(n_cases, dtype=bool)
    for case_idx, scores in enumerate(oof_scores):
        if scores is None:
            pool: list[str] = []
        else:
            pool = c3.top_k_ids(case_index["candidate_ids_by_case"][case_idx], scores, POOL_K)
        learned_pools[case_idx] = pool
        learned_hit[case_idx] = cases[case_idx]["gt"] in set(pool)
    pool_miss_admitted = [
        case_idx
        for case_idx in range(n_cases)
        if buckets[case_idx] == "POOL_MISS" and learned_hit[case_idx]
    ]
    return {
        "learned_pools": learned_pools,
        "learned_hit": learned_hit,
        "global_pool_hit": float(np.mean(learned_hit)),
        "global_pool_hit_count": int(np.sum(learned_hit)),
        "global_pool_miss_admitted": len(pool_miss_admitted),
    }


def featurize_case_pool(
    case_idx: int,
    pool: list[str],
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
) -> np.ndarray:
    source_lists = c3.make_source_lists(
        payload,
        r21_source,
        r54_source,
        case_index["als_source"],
        case_idx,
    )
    full_rrf_scores = c3.rrf_scores(source_lists)
    full_rrf_rank = {
        track_id: rank + 1
        for rank, track_id in enumerate(c3.sorted_by_score(full_rrf_scores))
    }
    r21_rank_map = {track_id: rank + 1 for rank, track_id in enumerate(source_lists["R21"][:POOL_K])}
    r54_rank_map = {track_id: rank + 1 for rank, track_id in enumerate(source_lists["R54"][:POOL_K])}
    return featurize_for_frozen_lr(
        pool,
        source_lists,
        r21_rank_map,
        r54_rank_map,
        case_index["r54_scores"][case_idx],
        payload["cases"][case_idx],
        maps,
        als_factors,
        als_to_idx,
        case_index["als_session_vecs"][case_idx],
        track_pop,
        max_pop,
        track_album,
        rrf_full_rank_map=full_rrf_rank,
    )


def build_lr_matrices(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    learned_pools: list[list[str]],
    train_cases: list[int],
    val_cases: list[int],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
) -> dict[str, Any]:
    feature_count = len(FEAT_ALL)
    if feature_count != EXPECTED_LR_FEATURES:
        raise RuntimeError(f"Refusing to build Variant A matrix with {feature_count} features")

    cases = payload["cases"]
    train_groups = [len(learned_pools[case_idx]) for case_idx in train_cases]
    val_groups = [len(learned_pools[case_idx]) for case_idx in val_cases]
    if any(group <= 0 for group in train_groups + val_groups):
        raise RuntimeError("Encountered an empty learned-admission LR group")

    train_rows = int(sum(train_groups))
    val_rows = int(sum(val_groups))
    X_train = np.zeros((train_rows, feature_count), dtype=np.float32)
    y_train = np.zeros(train_rows, dtype=np.float32)
    X_val = np.zeros((val_rows, feature_count), dtype=np.float32)
    y_val = np.zeros(val_rows, dtype=np.float32)
    val_offsets: dict[int, tuple[int, int]] = {}

    print(
        f"{ts()} Building Variant A LR matrices: train={train_rows:,} rows val={val_rows:,} rows "
        f"features={feature_count}",
        flush=True,
    )
    start_time = time.time()

    offset = 0
    for n_done, case_idx in enumerate(train_cases, start=1):
        pool = learned_pools[case_idx]
        feats = featurize_case_pool(
            case_idx,
            pool,
            payload,
            r21_source,
            r54_source,
            case_index,
            maps,
            als_factors,
            als_to_idx,
            track_pop,
            max_pop,
            track_album,
        )
        size = len(pool)
        X_train[offset:offset + size] = feats.astype(np.float32, copy=False)
        gt = cases[case_idx]["gt"]
        for local_idx, track_id in enumerate(pool):
            if track_id == gt:
                y_train[offset + local_idx] = 1.0
                break
        offset += size
        if n_done % 1000 == 0:
            print(f"  train features {n_done}/{len(train_cases)} ({time.time() - start_time:.0f}s)", flush=True)

    offset = 0
    for n_done, case_idx in enumerate(val_cases, start=1):
        pool = learned_pools[case_idx]
        feats = featurize_case_pool(
            case_idx,
            pool,
            payload,
            r21_source,
            r54_source,
            case_index,
            maps,
            als_factors,
            als_to_idx,
            track_pop,
            max_pop,
            track_album,
        )
        size = len(pool)
        X_val[offset:offset + size] = feats.astype(np.float32, copy=False)
        gt = cases[case_idx]["gt"]
        for local_idx, track_id in enumerate(pool):
            if track_id == gt:
                y_val[offset + local_idx] = 1.0
                break
        val_offsets[case_idx] = (offset, size)
        offset += size
        if n_done % 500 == 0:
            print(f"  val features {n_done}/{len(val_cases)} ({time.time() - start_time:.0f}s)", flush=True)

    positives_train = int(np.sum(y_train))
    positives_val = int(np.sum(y_val))
    print(
        f"  LR positives: train={positives_train}/{len(train_cases)} val={positives_val}/{len(val_cases)}",
        flush=True,
    )
    return {
        "X_train": X_train,
        "y_train": y_train,
        "group_train": train_groups,
        "X_val": X_val,
        "y_val": y_val,
        "group_val": val_groups,
        "val_offsets": val_offsets,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_positives": positives_train,
        "val_positives": positives_val,
    }


def train_variant_a_lr(matrices: dict[str, Any]) -> lgb.Booster:
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 10,
        "verbose": -1,
        "seed": 0,
    }
    print(f"{ts()} Training fold-0 Variant A LambdaRank model...", flush=True)
    dtrain = lgb.Dataset(
        matrices["X_train"],
        label=matrices["y_train"],
        group=matrices["group_train"],
        feature_name=list(FEAT_ALL),
        free_raw_data=True,
    )
    dval = lgb.Dataset(
        matrices["X_val"],
        label=matrices["y_val"],
        group=matrices["group_val"],
        reference=dtrain,
        feature_name=list(FEAT_ALL),
        free_raw_data=True,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def score_fold0_variant(
    model: lgb.Booster,
    matrices: dict[str, Any],
    val_cases: list[int],
    learned_pools: list[list[str]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    print(f"{ts()} Scoring fold-0 learned-admission pools with Variant A LR...", flush=True)
    preds = model.predict(matrices["X_val"])
    n_cases = len(cases)
    ndcg20 = np.zeros(n_cases, dtype=np.float64)
    gt_rank = np.full(n_cases, -1, dtype=np.int32)
    top1: list[str] = [""] * n_cases
    top20: list[list[str]] = [[] for _ in range(n_cases)]

    for case_idx in val_cases:
        offset, size = matrices["val_offsets"][case_idx]
        scores = preds[offset:offset + size]
        pool = learned_pools[case_idx]
        order = np.argsort(-scores, kind="mergesort")
        ranked_pool = [pool[int(local_idx)] for local_idx in order]
        top20[case_idx] = ranked_pool[:TOP_K]
        top1[case_idx] = ranked_pool[0] if ranked_pool else ""
        gt = cases[case_idx]["gt"]
        if gt in pool:
            gt_pool_idx = pool.index(gt)
            pos = np.where(order == gt_pool_idx)[0]
            if len(pos):
                gt_rank[case_idx] = int(pos[0]) + 1
                ndcg20[case_idx] = ndcg_from_rank(int(pos[0]) + 1, TOP_K)

    return {
        "ndcg_at_20": ndcg20,
        "gt_rank": gt_rank,
        "top1": top1,
        "top20": top20,
    }


def churn_by_split(
    val_cases: list[int],
    splits: dict[str, list[int]],
    baseline_scores: dict[str, Any],
    variant_scores: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    val_set = set(val_cases)
    for split_name, split_case_ids in splits.items():
        case_ids = [idx for idx in split_case_ids if idx in val_set]
        top1_changed = sum(1 for idx in case_ids if baseline_scores["top1"][idx] != variant_scores["top1"][idx])
        overlaps = [
            float(len(set(baseline_scores["top20"][idx]) & set(variant_scores["top20"][idx])))
            for idx in case_ids
        ]
        n = len(case_ids)
        out[split_name] = {
            "n": n,
            "top1_changed": top1_changed,
            "top1_churn_rate": top1_changed / max(n, 1),
            "top1_blind_equiv_per_80": top1_changed / max(n, 1) * 80.0,
            "top20_overlap": percentile_summary(overlaps),
        }
    return out


def conversion_and_recovery(
    val_cases: list[int],
    buckets: list[str],
    learned_hit: np.ndarray,
    baseline_scores: dict[str, Any],
    variant_scores: dict[str, Any],
) -> dict[str, Any]:
    recovered = [
        idx for idx in val_cases
        if baseline_scores["ndcg_at_20"][idx] == 0.0 and variant_scores["ndcg_at_20"][idx] > 0.0
    ]
    lost = [
        idx for idx in val_cases
        if baseline_scores["ndcg_at_20"][idx] > 0.0 and variant_scores["ndcg_at_20"][idx] == 0.0
    ]
    pool_miss_admitted = [
        idx for idx in val_cases
        if buckets[idx] == "POOL_MISS" and bool(learned_hit[idx])
    ]
    pool_miss_top20 = [
        idx for idx in pool_miss_admitted
        if 1 <= int(variant_scores["gt_rank"][idx]) <= TOP_K
    ]
    pool_miss_buried = [
        idx for idx in pool_miss_admitted
        if int(variant_scores["gt_rank"][idx]) > TOP_K
    ]
    return {
        "top20_recovered_lost": {
            "recovered": len(recovered),
            "lost": len(lost),
            "net": len(recovered) - len(lost),
        },
        "pool_miss_conversion": {
            "admitted": len(pool_miss_admitted),
            "top20": len(pool_miss_top20),
            "buried_21_300": len(pool_miss_buried),
            "convert_rate": len(pool_miss_top20) / max(len(pool_miss_admitted), 1),
        },
    }


def verdict_from_gates(
    baseline_pass: bool,
    deltas: dict[str, dict[str, Any]],
    conversion: dict[str, Any],
    churn: dict[str, Any],
) -> tuple[str, dict[str, bool], list[str]]:
    h7_delta = deltas["h7"]["delta_ndcg_at_20"]
    same_delta = deltas["same_artist"]["delta_ndcg_at_20"]
    top20_net = conversion["top20_recovered_lost"]["net"]
    convert_rate = conversion["pool_miss_conversion"]["convert_rate"]
    max_top1_blind_eq = max(churn[split]["top1_blind_equiv_per_80"] for split in churn)
    min_top20_overlap_mean = min(churn[split]["top20_overlap"]["mean"] for split in churn)

    checks = {
        "baseline_reproduction_pass": baseline_pass,
        "h7_delta_nonnegative": h7_delta >= 0.0,
        "h7_delta_ge_003": h7_delta >= 0.003,
        "pool_miss_conversion_gt_10pct": convert_rate > 0.10,
        "pool_miss_conversion_ge_15pct": convert_rate >= 0.15,
        "top20_net_nonnegative": top20_net >= 0,
        "top20_net_positive": top20_net > 0,
        "same_artist_canary_ok": same_delta >= -0.002,
        "top1_hard_stop_ok_all_tracked": max_top1_blind_eq <= 35.0,
        "top20_overlap_hard_stop_ok_all_tracked": min_top20_overlap_mean >= 14.0,
    }
    archive_reasons = []
    if not checks["baseline_reproduction_pass"]:
        archive_reasons.append("baseline reproduction failed")
    if not checks["h7_delta_nonnegative"]:
        archive_reasons.append("fold-0 h7 delta < 0")
    if not checks["pool_miss_conversion_gt_10pct"]:
        archive_reasons.append("POOL_MISS conversion <= 10%")
    if not checks["top20_net_nonnegative"]:
        archive_reasons.append("top-20 net < 0")
    if not checks["same_artist_canary_ok"]:
        archive_reasons.append("same-artist delta < -0.002")
    if not checks["top1_hard_stop_ok_all_tracked"]:
        archive_reasons.append("top1 blind-equivalent churn > 35/80")
    if not checks["top20_overlap_hard_stop_ok_all_tracked"]:
        archive_reasons.append("top-20 overlap mean < 14/20")

    if archive_reasons:
        return "ARCHIVE", checks, archive_reasons
    if (
        checks["h7_delta_ge_003"]
        and checks["pool_miss_conversion_ge_15pct"]
        and checks["top20_net_positive"]
        and checks["same_artist_canary_ok"]
        and checks["top1_hard_stop_ok_all_tracked"]
        and checks["top20_overlap_hard_stop_ok_all_tracked"]
    ):
        return "PROCEED_TO_CV5_REVIEW", checks, []
    return "BORDERLINE", checks, []


def write_markdown(report: dict[str, Any]) -> None:
    baseline = report["baseline_reproduction"]
    fold = report["fold0"]
    delta = fold["delta"]
    conversion = fold["conversion"]
    churn = fold["churn"]
    top20 = conversion["top20_recovered_lost"]
    pool_conv = conversion["pool_miss_conversion"]
    verdict = report["verdict"]

    lines = [
        "| Kill-shot metric | Fold-0 value | Target / stop | Status |",
        "|---|---:|---:|---:|",
        (
            f"| Baseline reproduction | {baseline['pass']} | epsilon <= {BASELINE_EPS:.4f} "
            "on all 4 metrics | "
            f"{'PASS' if baseline['pass'] else 'FAIL'} |"
        ),
        (
            f"| h7 nDCG@20 delta | {delta['h7']['delta_ndcg_at_20']:+.5f} | "
            "archive if < 0; proceed if >= +0.003 | "
            f"{'PASS' if report['gate_checks']['h7_delta_nonnegative'] else 'FAIL'} |"
        ),
        (
            f"| POOL_MISS conversion | {pool_conv['top20']} / {pool_conv['admitted']} "
            f"({pool_conv['convert_rate']:.2%}) | archive if <= 10%; proceed if >= 15% | "
            f"{'PASS' if report['gate_checks']['pool_miss_conversion_gt_10pct'] else 'FAIL'} |"
        ),
        (
            f"| Top-20 recovered / lost / net | {top20['recovered']} / {top20['lost']} / {top20['net']} | "
            "archive if net < 0; proceed if net > 0 | "
            f"{'PASS' if report['gate_checks']['top20_net_nonnegative'] else 'FAIL'} |"
        ),
        (
            f"| Same-artist canary delta | {delta['same_artist']['delta_ndcg_at_20']:+.5f} | "
            "archive if < -0.002 | "
            f"{'PASS' if report['gate_checks']['same_artist_canary_ok'] else 'FAIL'} |"
        ),
        (
            f"| Top-1 churn blind-eq | {churn['all']['top1_blind_equiv_per_80']:.2f}/80 | "
            "archive if any tracked split > 35/80 | "
            f"{'PASS' if report['gate_checks']['top1_hard_stop_ok_all_tracked'] else 'FAIL'} |"
        ),
        (
            f"| Top-20 overlap mean | {churn['all']['top20_overlap']['mean']:.2f}/20 | "
            "archive if any tracked split < 14/20 | "
            f"{'PASS' if report['gate_checks']['top20_overlap_hard_stop_ok_all_tracked'] else 'FAIL'} |"
        ),
        f"| Verdict | **{verdict}** | see §8/§9 gates | {verdict} |",
        "",
        "# R60 Variant A Fold-0 Result",
        "",
        f"Created: {report['created_at']}",
        "",
        f"Next step recommendation: **{report['next_step_recommendation']}**",
        "",
        "## Baseline Reproduction",
        "",
        "| Split | n | Target | R59 artifact | Recomputed | Abs delta vs artifact | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ["all", "h7", "same_artist", "diff_artist"]:
        row = baseline["metrics"][split_name]
        lines.append(
            f"| {split_name} | {row['n']} | {row['target']:.5f} | {row['artifact_value']:.5f} | "
            f"{row['computed']:.5f} | {row['abs_delta_vs_artifact']:.6f} | {row['pass']} |"
        )
    pool_hit = baseline["rrf_pool_hit_at_300"]
    lines.extend(
        [
            (
                f"| pool_hit@300 | {report['cases']['n']} | {pool_hit['target']:.5f} | "
                f"{report['r59_phase2_pool_hit_rrf']:.5f} | {pool_hit['computed']:.5f} | "
                f"{pool_hit['abs_delta']:.6f} | {pool_hit['pass']} |"
            ),
            "",
            "## Fold-0 Metrics",
            "",
            "| Split | n | RRF frozen-LR baseline | Variant A | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split_name in ["all", "h7", "same_artist", "diff_artist"]:
        b = fold["baseline_metrics"][split_name]
        v = fold["variant_metrics"][split_name]
        d = fold["delta"][split_name]
        lines.append(
            f"| {split_name} | {v['n']} | {b['ndcg_at_20']:.5f} | {v['ndcg_at_20']:.5f} | "
            f"{d['delta_ndcg_at_20']:+.5f} |"
        )

    lines.extend(
        [
            "",
            "## Conversion",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Fold-0 cases | {fold['n_cases']} |",
            f"| Fold-0 h7 cases | {fold['split_sizes']['h7']} |",
            f"| Fold-0 same-artist cases | {fold['split_sizes']['same_artist']} |",
            f"| Fold-0 diff-artist cases | {fold['split_sizes']['diff_artist']} |",
            f"| Global learned-pool POOL_MISS admitted | {report['learned_pool_summary']['global_pool_miss_admitted']} |",
            f"| Fold-0 POOL_MISS admitted | {pool_conv['admitted']} |",
            f"| Fold-0 POOL_MISS converted top-20 | {pool_conv['top20']} |",
            f"| Fold-0 POOL_MISS buried 21-300 | {pool_conv['buried_21_300']} |",
            f"| Fold-0 conversion rate | {pool_conv['convert_rate']:.2%} |",
            f"| Top-20 recovered | {top20['recovered']} |",
            f"| Top-20 lost | {top20['lost']} |",
            f"| Top-20 net | {top20['net']} |",
            "",
            "## Churn",
            "",
            "| Split | n | Top-1 changed | Top-1 blind-eq | Top-20 overlap mean | Top-20 overlap median |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ["all", "h7", "same_artist", "diff_artist"]:
        row = churn[split_name]
        lines.append(
            f"| {split_name} | {row['n']} | {row['top1_changed']} | "
            f"{row['top1_blind_equiv_per_80']:.2f}/80 | "
            f"{row['top20_overlap']['mean']:.2f}/20 | {row['top20_overlap']['median']:.2f}/20 |"
        )

    lines.extend(
        [
            "",
            "## Feature Contract",
            "",
            f"- Variant A LR features: {report['feature_contract']['variant_a_feature_count']}",
            f"- C3 admission features: {report['feature_contract']['c3_admission_feature_count']}",
            f"- C3 admission artifact features: {report['feature_contract']['c3_admission_artifact_feature_count']}",
            "- `admission_score` and `admission_rank_inv` are not used.",
            "- Learned-pool `rrf_rank_inv` is pinned to full weighted-RRF source-union rank.",
            "",
            "## Top Feature Importances",
            "",
            "| Rank | Feature | Gain |",
            "|---:|---|---:|",
        ]
    )
    for rank, row in enumerate(report["feature_importance_top15"], start=1):
        lines.append(f"| {rank} | {row['feature']} | {row['gain']:.3f} |")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- `dev_only`: {report['dev_only']}",
            f"- `blind_access`: {report['blind_access']}",
            f"- Fold split: `grouped_session_folds(seed=0)`, fold 0 n={fold['n_cases']}",
            "- LR training cases: folds 1-4 only; no production initialization.",
            f"- Admission artifact loaded: `{C3_ADMISSION_JSON}`",
            f"- Admission scores source: {report['admission_scores_source']}",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.time()
    print("R60 Variant A fold-0 diagnostic")
    print("=" * 70)

    with open(R59_PHASE2_JSON) as f:
        r59_phase2 = json.load(f)
    with open(C3_ADMISSION_JSON) as f:
        c3_admission_artifact = json.load(f)

    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    ranker = lgb.Booster(model_file=str(LR_MODEL))
    feature_contract = validate_contracts(ranker, c3_admission_artifact)
    print(
        f"  feature contract: LR={feature_contract['variant_a_feature_count']} "
        f"C3 admission={feature_contract['c3_admission_feature_count']}",
        flush=True,
    )

    sessions = [case["session_id"] for case in cases]
    folds = grouped_session_folds(sessions, seed=0)
    fold0_cases = sorted(int(idx) for idx in folds[0].tolist())
    fold0_set = set(fold0_cases)
    train_cases = [idx for idx in range(len(cases)) if idx not in fold0_set]
    all_splits = split_indices(list(range(len(cases))), cases, maps["track_artist"])
    fold0_splits = split_indices(fold0_cases, cases, maps["track_artist"])
    print(
        f"  fold 0 cases={len(fold0_cases)} h7={len(fold0_splits['h7'])} "
        f"same={len(fold0_splits['same_artist'])} diff={len(fold0_splits['diff_artist'])}",
        flush=True,
    )

    case_index = c3.build_case_index(
        payload,
        r21_source,
        r54_source,
        r54_scores,
        als_factors,
        als_track_ids,
        als_to_idx,
    )
    buckets, bucket_details = c3.load_bucket_labels(
        len(cases),
        case_index["baseline_hit"],
        case_index["source_union_has_gt"],
    )
    print(f"  bucket counts: {dict(Counter(buckets))}", flush=True)

    baseline_scores = score_rrf_baseline(
        ranker,
        payload,
        r21_source,
        r54_source,
        case_index,
        maps,
        als_factors,
        als_to_idx,
        track_pop,
        max_pop,
        track_album,
    )
    baseline_repro = baseline_reproduction_report(baseline_scores, cases, maps, r59_phase2)
    for split_name, row in baseline_repro["metrics"].items():
        print(
            f"  baseline {split_name}: computed={row['computed']:.5f} "
            f"artifact={row['artifact_value']:.5f} abs_delta={row['abs_delta_vs_artifact']:.6f}",
            flush=True,
        )
    print(
        f"  baseline pool_hit@300={baseline_repro['rrf_pool_hit_at_300']['computed']:.5f} "
        f"pass={baseline_repro['pass']}",
        flush=True,
    )
    if not baseline_repro["pass"]:
        raise RuntimeError("Baseline reproduction failed; aborting before Variant A training")

    print(f"{ts()} Building C3 admission OOF scores for learned-admission@300 pools...", flush=True)
    admission_scores_source = "rebuilt with c3.train_cv5; expR59_c3_pool_admission.json has aggregate metrics only"
    X_admit, y_admit = c3.build_feature_matrix(
        payload,
        r21_source,
        r54_source,
        case_index,
        als_factors,
        als_to_idx,
    )
    oof_scores, admission_train_report, admission_fold_reports = c3.train_cv5(
        X_admit,
        y_admit,
        payload,
        case_index,
    )
    del X_admit, y_admit
    gc.collect()

    learned_pool_info = learned_pools_from_oof_scores(payload, case_index, oof_scores, buckets)
    learned_pools = learned_pool_info["learned_pools"]
    del oof_scores
    gc.collect()
    print(
        f"  learned pool_hit@300={learned_pool_info['global_pool_hit']:.5f} "
        f"POOL_MISS admitted={learned_pool_info['global_pool_miss_admitted']}",
        flush=True,
    )

    matrices = build_lr_matrices(
        payload,
        r21_source,
        r54_source,
        case_index,
        learned_pools,
        train_cases,
        fold0_cases,
        maps,
        als_factors,
        als_to_idx,
        track_pop,
        max_pop,
        track_album,
    )
    model = train_variant_a_lr(matrices)
    variant_scores = score_fold0_variant(model, matrices, fold0_cases, learned_pools, cases)

    baseline_fold0_metrics = metrics_for_splits(baseline_scores["ndcg_at_20"], fold0_splits)
    variant_fold0_metrics = metrics_for_splits(variant_scores["ndcg_at_20"], fold0_splits)
    deltas = delta_metrics(baseline_fold0_metrics, variant_fold0_metrics)
    conversion = conversion_and_recovery(
        fold0_cases,
        buckets,
        learned_pool_info["learned_hit"],
        baseline_scores,
        variant_scores,
    )
    churn = churn_by_split(fold0_cases, fold0_splits, baseline_scores, variant_scores)
    verdict, gate_checks, archive_reasons = verdict_from_gates(
        baseline_repro["pass"],
        deltas,
        conversion,
        churn,
    )
    if verdict == "PROCEED_TO_CV5_REVIEW":
        next_step = "request separate user approval for full CV5 Variant A review"
    elif verdict == "BORDERLINE":
        next_step = "review diagnostics before deciding whether to archive or request CV5 approval"
    else:
        next_step = "archive R60 Variant A matched-pool fold-0 path"

    per_case_fold0 = []
    for case_idx in fold0_cases:
        per_case_fold0.append(
            {
                "case_idx": case_idx,
                "session_id": cases[case_idx]["session_id"],
                "n_prior_music": int(cases[case_idx]["n_prior_music"]),
                "bucket": buckets[case_idx],
                "same_artist": same_artist_case(cases[case_idx], maps["track_artist"]),
                "baseline_gt_rank": int(baseline_scores["gt_rank"][case_idx]),
                "variant_gt_rank": int(variant_scores["gt_rank"][case_idx]),
                "baseline_ndcg_at_20": float(baseline_scores["ndcg_at_20"][case_idx]),
                "variant_ndcg_at_20": float(variant_scores["ndcg_at_20"][case_idx]),
                "learned_gt_in_pool": bool(learned_pool_info["learned_hit"][case_idx]),
                "top1_changed": baseline_scores["top1"][case_idx] != variant_scores["top1"][case_idx],
                "top20_overlap": int(
                    len(set(baseline_scores["top20"][case_idx]) & set(variant_scores["top20"][case_idx]))
                ),
            }
        )

    report = {
        "experiment": "R60 Variant A fold-0 diagnostic",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start_time,
        "dev_only": True,
        "blind_access": False,
        "lr_retraining": True,
        "lr_training_scope": "folds_1_4_only",
        "admission_score_as_lr_feature": False,
        "admission_rank_inv_as_lr_feature": False,
        "admission_scores_source": admission_scores_source,
        "feature_contract": feature_contract,
        "baseline_reproduction": baseline_repro,
        "r59_phase2_pool_hit_rrf": float(r59_phase2["pool_hit"]["rrf"]),
        "cases": {"n": len(cases)},
        "fold0": {
            "fold": 0,
            "n_cases": len(fold0_cases),
            "train_cases": len(train_cases),
            "case_indices": fold0_cases,
            "split_sizes": {name: len(indices) for name, indices in fold0_splits.items()},
            "baseline_metrics": baseline_fold0_metrics,
            "variant_metrics": variant_fold0_metrics,
            "delta": deltas,
            "conversion": conversion,
            "churn": churn,
            "per_case": per_case_fold0,
        },
        "all_dev_split_sizes": {name: len(indices) for name, indices in all_splits.items()},
        "bucket_details": bucket_details,
        "learned_pool_summary": {
            "global_pool_hit": learned_pool_info["global_pool_hit"],
            "global_pool_hit_count": learned_pool_info["global_pool_hit_count"],
            "global_pool_miss_admitted": learned_pool_info["global_pool_miss_admitted"],
            "artifact_learned_pool_hit": c3_admission_artifact["evaluation"]["learned_pool_hit"],
            "artifact_pool_miss_recovered": c3_admission_artifact["evaluation"]["pool_miss_recovered"],
        },
        "admission_training": {
            "params": admission_train_report["params"],
            "num_boost_round": admission_train_report["num_boost_round"],
            "elapsed_s": admission_train_report["elapsed_s"],
            "folds": admission_fold_reports,
        },
        "lr_training": {
            "params": {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": [20],
                "num_leaves": 31,
                "learning_rate": 0.05,
                "min_data_in_leaf": 10,
                "seed": 0,
                "num_boost_round": 300,
            },
            "train_rows": matrices["train_rows"],
            "val_rows": matrices["val_rows"],
            "train_positives": matrices["train_positives"],
            "val_positives": matrices["val_positives"],
        },
        "feature_importance_top15": top_feature_importance(model, 15),
        "gate_checks": gate_checks,
        "archive_reasons": archive_reasons,
        "verdict": verdict,
        "next_step_recommendation": next_step,
        "notes": (
            "Variant A uses learned-admission@300 pools and the existing 37-feature "
            "R54c/R55 LR feature contract. Learned-pool rrf_rank_inv is computed "
            "from full weighted-RRF source-union rank, not admission rank."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report)

    h7_delta = deltas["h7"]["delta_ndcg_at_20"]
    same_delta = deltas["same_artist"]["delta_ndcg_at_20"]
    pool_conv = conversion["pool_miss_conversion"]
    top20 = conversion["top20_recovered_lost"]
    top1_eq = churn["all"]["top1_blind_equiv_per_80"]
    overlap = churn["all"]["top20_overlap"]["mean"]
    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(
        f"{ts()} Verdict: {verdict} h7_delta={h7_delta:+.5f} "
        f"conversion={pool_conv['top20']}/{pool_conv['admitted']} "
        f"recovered/lost/net={top20['recovered']}/{top20['lost']}/{top20['net']} "
        f"same_delta={same_delta:+.5f} top1_churn_blind_eq={top1_eq:.2f}/80 "
        f"top20_overlap={overlap:.2f}/20",
        flush=True,
    )
    print(f"Elapsed: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
