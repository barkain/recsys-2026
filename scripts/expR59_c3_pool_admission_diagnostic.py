#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R59 Phase B / C3 learned pool-admission diagnostic.

Dev-only, OOF-clean, no Blind-A access, no LR retraining.

The diagnostic tests whether a fixed LightGBM LambdaRank admission model can
replace weighted RRF for pool@300 admission over the existing source-union
candidate universe. It scores candidates OOF by grouped-session CV5 and
compares learned admission@300 against weighted_rrf@300.

Outputs:
  exp/eval/expR59_c3_pool_admission.json
  docs/r59_candidates/c3_diagnostic_result.md
"""
from __future__ import annotations

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

from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds  # noqa: E402

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
DECOMP_JSON = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"
R58_TOP50 = REPO / "cache" / "r58" / "top50_dev.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR59_c3_pool_admission.json"
OUT_MD = REPO / "docs" / "r59_candidates" / "c3_diagnostic_result.md"

SOURCE_NAMES = ["A", "B", "C", "D", "F", "ALS", "R21", "R54"]
STRONG_SOURCES = {"B", "C", "R21", "R54"}
DENSE_SOURCES = {"R21", "R54"}
LEXICAL_SOURCES = {"B", "C"}
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}
BASELINE_EPS = 0.0005
NUM_BOOST_ROUND = 120


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def source_key(source_name: str) -> str:
    return f"src_{source_name.lower()}"


def build_feature_names() -> list[str]:
    names = []
    for source_name in SOURCE_NAMES:
        prefix = source_name.lower()
        names.extend(
            [
                f"{prefix}_rank",
                f"{prefix}_rank_inv",
                f"{prefix}_log_rank_bin",
                f"{prefix}_top10",
                f"{prefix}_top20",
                f"{prefix}_top50",
                f"{prefix}_top100",
                f"{prefix}_top300",
                f"{prefix}_missing",
            ]
        )
    names.extend(
        [
            "n_sources_present",
            "n_strong_sources_present",
            "min_rank",
            "best_dense_rank",
            "best_lexical_rank",
            "rank_dispersion",
            "any_shallow_source",
            "any_deep_source",
            "all_present_deep",
            "single_source_only",
            "single_source_deep",
            "dense_shallow",
            "lexical_shallow",
            "strong_shallow",
            "shallow_and_deep",
            "r54_cosine",
            "als_score",
            "weighted_rrf_score",
            "weighted_rrf_rank_inv",
            "n_prior_music",
            "n_unique_artists",
            "last_artist_repeat",
            "source_overlap_mean_jaccard",
            "source_overlap_max_jaccard",
            "strong_source_overlap_mean",
            "dense_source_overlap_jaccard",
            "lexical_source_overlap_jaccard",
        ]
    )
    return names


FEATURE_NAMES = build_feature_names()


def first_rank_map(ranked: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rank, track_id in enumerate(ranked, start=1):
        if track_id not in out:
            out[track_id] = rank
    return out


def rrf_scores(source_lists: dict[str, list[str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for source_name in SOURCE_NAMES:
        weight = SW[source_name]
        if weight == 0:
            continue
        for rank, track_id in enumerate(source_lists[source_name], start=1):
            scores[track_id] = scores.get(track_id, 0.0) + weight / (RRF_K + rank)
    return scores


def sorted_by_score(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.__getitem__, reverse=True)


def jaccard(a_set: set[str], b_set: set[str]) -> float:
    union = a_set | b_set
    if not union:
        return 0.0
    return len(a_set & b_set) / len(union)


def source_overlap_features(source_lists: dict[str, list[str]]) -> tuple[float, float, float, float, float]:
    source_sets = {source_name: set(source_lists[source_name]) for source_name in SOURCE_NAMES}
    all_vals = []
    strong_vals = []
    for idx, left in enumerate(SOURCE_NAMES):
        for right in SOURCE_NAMES[idx + 1:]:
            val = jaccard(source_sets[left], source_sets[right])
            all_vals.append(val)
            if left in STRONG_SOURCES and right in STRONG_SOURCES:
                strong_vals.append(val)
    mean_all = float(np.mean(all_vals)) if all_vals else 0.0
    max_all = float(np.max(all_vals)) if all_vals else 0.0
    mean_strong = float(np.mean(strong_vals)) if strong_vals else 0.0
    dense = jaccard(source_sets["R21"], source_sets["R54"])
    lexical = jaccard(source_sets["B"], source_sets["C"])
    return mean_all, max_all, mean_strong, dense, lexical


def load_payloads() -> tuple[dict[str, Any], list[list[str]], list[list[str]], list[dict[str, float]]]:
    print(f"{ts()} Loading dev payload and OOF source lists...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_source = []
    r54_scores = []
    for case_lists in r54_data["lists"]:
        r54_source.append([track_id for track_id, _score in case_lists])
        r54_scores.append({track_id: float(score) for track_id, score in case_lists})
    return payload, r21_source, r54_source, r54_scores


def load_als_cache() -> tuple[np.ndarray, list[str], dict[str, int]]:
    print(f"{ts()} Loading cached ALS factors...", flush=True)
    als_data = np.load(ALS_CACHE, allow_pickle=True)
    factors = np.asarray(als_data["factors"], dtype=np.float32)
    track_ids = [str(track_id) for track_id in als_data["track_ids"].tolist()]
    track_to_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}
    print(f"  ALS factors: {factors.shape[0]} tracks x {factors.shape[1]} dims", flush=True)
    return factors, track_ids, track_to_idx


def build_als_source_for_case(
    case: dict[str, Any],
    factors: np.ndarray,
    track_ids: list[str],
    track_to_idx: dict[str, int],
) -> tuple[list[str], np.ndarray | None]:
    played = case["music_turns"]
    session_vec = als_session_vector(played, track_to_idx, factors)
    if session_vec is None:
        return [], None
    scores = factors @ session_vec
    for track_id in played:
        idx = track_to_idx.get(track_id)
        if idx is not None:
            scores[idx] = -np.inf
    top_count = min(200, len(scores))
    top_idx = np.argpartition(-scores, top_count - 1)[:top_count]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [track_ids[int(idx)] for idx in top_idx], session_vec


def make_source_lists(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    als_source: list[list[str]],
    case_idx: int,
) -> dict[str, list[str]]:
    return {
        "A": payload["src_a"][case_idx],
        "B": payload["src_b"][case_idx],
        "C": payload["src_c"][case_idx],
        "D": payload["src_d"][case_idx],
        "F": payload["src_f"][case_idx],
        "ALS": als_source[case_idx],
        "R21": r21_source[case_idx],
        "R54": r54_source[case_idx],
    }


def unique_source_union(source_lists: dict[str, list[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for source_name in SOURCE_NAMES:
        for track_id in source_lists[source_name]:
            if track_id not in seen:
                seen[track_id] = None
    return list(seen)


def build_case_index(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    r54_scores: list[dict[str, float]],
    factors: np.ndarray,
    track_ids: list[str],
    track_to_idx: dict[str, int],
) -> dict[str, Any]:
    cases = payload["cases"]
    n_cases = len(cases)
    als_source: list[list[str]] = []
    als_session_vecs: list[np.ndarray | None] = []
    candidate_ids_by_case: list[list[str]] = []
    baseline_pools: list[list[str]] = []
    baseline_hit = np.zeros(n_cases, dtype=bool)
    source_union_has_gt = np.zeros(n_cases, dtype=bool)
    gt_sources: list[list[str]] = []
    row_starts = np.zeros(n_cases, dtype=np.int64)
    row_counts = np.zeros(n_cases, dtype=np.int32)
    total_rows = 0

    print(f"{ts()} Building ALS lists, source unions, and RRF baseline...", flush=True)
    start_time = time.time()
    for case_idx, case in enumerate(cases):
        als_case_source, session_vec = build_als_source_for_case(case, factors, track_ids, track_to_idx)
        als_source.append(als_case_source)
        als_session_vecs.append(session_vec)

        source_lists = make_source_lists(payload, r21_source, r54_source, als_source, case_idx)
        candidates = unique_source_union(source_lists)
        scores = rrf_scores(source_lists)
        ranked_rrf = sorted_by_score(scores)
        pool = ranked_rrf[:POOL_K]
        gt = case["gt"]

        row_starts[case_idx] = total_rows
        row_counts[case_idx] = len(candidates)
        total_rows += len(candidates)
        candidate_ids_by_case.append(candidates)
        baseline_pools.append(pool)
        baseline_hit[case_idx] = gt in pool
        source_union_has_gt[case_idx] = gt in scores
        gt_sources.append([source_name for source_name in SOURCE_NAMES if gt in source_lists[source_name]])

        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  indexed {case_idx + 1}/{n_cases} cases ({elapsed:.0f}s)", flush=True)

    baseline_pool_hit = float(np.mean(baseline_hit))
    source_union_hit = float(np.mean(source_union_has_gt))
    print(f"  candidate rows: {total_rows:,}", flush=True)
    print(f"  weighted_rrf pool_hit@300: {baseline_pool_hit:.4f}", flush=True)
    print(f"  source_union_hit: {source_union_hit:.4f}", flush=True)

    return {
        "als_source": als_source,
        "als_session_vecs": als_session_vecs,
        "candidate_ids_by_case": candidate_ids_by_case,
        "baseline_pools": baseline_pools,
        "baseline_hit": baseline_hit,
        "source_union_has_gt": source_union_has_gt,
        "gt_sources": gt_sources,
        "row_starts": row_starts,
        "row_counts": row_counts,
        "total_rows": total_rows,
        "r54_scores": r54_scores,
    }


def load_bucket_labels(
    n_cases: int,
    baseline_hit: np.ndarray,
    source_union_has_gt: np.ndarray,
) -> tuple[list[str], dict[str, Any]]:
    details: dict[str, Any] = {"source": "baseline_pool_and_source_union"}
    buckets = [""] * n_cases
    top50_by_case: dict[int, dict[str, Any]] = {}
    if R58_TOP50.exists():
        print(f"{ts()} Loading R58 top-50 cache for exact HIT/DEMOTED bucket labels...", flush=True)
        with open(R58_TOP50, "rb") as f:
            rows = pickle.load(f)
        for row in rows:
            case_idx = int(row["case_idx"])
            if case_idx not in top50_by_case:
                top50_by_case[case_idx] = row
        details["source"] = str(R58_TOP50)
        details["rows"] = len(rows)
    else:
        print("  R58 top-50 cache missing; HIT/DEMOTED labels will be collapsed.", flush=True)

    for case_idx in range(n_cases):
        if baseline_hit[case_idx]:
            row = top50_by_case.get(case_idx)
            gt_lr_rank = int(row.get("gt_lr_rank", -1)) if row else -1
            if 1 <= gt_lr_rank <= 20:
                buckets[case_idx] = "HIT"
            else:
                buckets[case_idx] = "DEMOTED"
        elif source_union_has_gt[case_idx]:
            buckets[case_idx] = "POOL_MISS"
        else:
            buckets[case_idx] = "UNREACHABLE"
    details["bucket_counts"] = dict(Counter(buckets))
    return buckets, details


def fill_source_rank_features(row: np.ndarray, ranks: list[int], offset: int) -> None:
    pos = offset
    for rank in ranks:
        if rank > 0:
            row[pos] = float(rank)
            row[pos + 1] = 1.0 / rank
            row[pos + 2] = float(np.ceil(np.log2(rank + 1)))
            row[pos + 3] = 1.0 if rank <= 10 else 0.0
            row[pos + 4] = 1.0 if rank <= 20 else 0.0
            row[pos + 5] = 1.0 if rank <= 50 else 0.0
            row[pos + 6] = 1.0 if rank <= 100 else 0.0
            row[pos + 7] = 1.0 if rank <= 300 else 0.0
            row[pos + 8] = 0.0
        else:
            row[pos + 8] = 1.0
        pos += 9


def candidate_als_scores(
    candidates: list[str],
    session_vec: np.ndarray | None,
    factors: np.ndarray,
    track_to_idx: dict[str, int],
) -> np.ndarray:
    out = np.zeros(len(candidates), dtype=np.float32)
    if session_vec is None or not candidates:
        return out
    idxs = np.array([track_to_idx.get(track_id, -1) for track_id in candidates], dtype=np.int64)
    valid = idxs >= 0
    if np.any(valid):
        out[valid] = factors[idxs[valid]] @ session_vec
    return out


def build_feature_matrix(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    factors: np.ndarray,
    track_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    n_features = len(FEATURE_NAMES)
    total_rows = int(case_index["total_rows"])
    X = np.zeros((total_rows, n_features), dtype=np.float32)
    y = np.zeros(total_rows, dtype=np.float32)
    als_source = case_index["als_source"]
    als_session_vecs = case_index["als_session_vecs"]
    candidate_ids_by_case = case_index["candidate_ids_by_case"]
    r54_scores = case_index["r54_scores"]

    print(f"{ts()} Building feature matrix: {total_rows:,} rows x {n_features} features...", flush=True)
    start_time = time.time()
    agreement_offset = len(SOURCE_NAMES) * 9
    for case_idx, case in enumerate(cases):
        source_lists = make_source_lists(payload, r21_source, r54_source, als_source, case_idx)
        rank_maps = {source_name: first_rank_map(source_lists[source_name]) for source_name in SOURCE_NAMES}
        scores = rrf_scores(source_lists)
        rrf_rank = {track_id: rank + 1 for rank, track_id in enumerate(sorted_by_score(scores))}
        overlap_vals = source_overlap_features(source_lists)
        candidates = candidate_ids_by_case[case_idx]
        start = int(case_index["row_starts"][case_idx])
        gt = case["gt"]
        played = case["music_turns"]
        n_prior_music = float(case["n_prior_music"])
        played_artists = [track_artist.get(track_id, "") for track_id in played]
        unique_artists = {artist for artist in played_artists if artist}
        last_artist = track_artist.get(played[-1], "") if played else ""
        als_scores = candidate_als_scores(candidates, als_session_vecs[case_idx], factors, track_to_idx)

        for local_idx, track_id in enumerate(candidates):
            row_idx = start + local_idx
            row = X[row_idx]
            ranks = [rank_maps[source_name].get(track_id, 0) for source_name in SOURCE_NAMES]
            present_ranks = [rank for rank in ranks if rank > 0]
            fill_source_rank_features(row, ranks, 0)

            n_sources = len(present_ranks)
            strong_count = sum(1 for source_name in STRONG_SOURCES if rank_maps[source_name].get(track_id, 0) > 0)
            min_rank = min(present_ranks) if present_ranks else 0
            max_rank = max(present_ranks) if present_ranks else 0
            dense_ranks = [
                rank_maps[source_name].get(track_id, 0)
                for source_name in DENSE_SOURCES
                if rank_maps[source_name].get(track_id, 0) > 0
            ]
            lexical_ranks = [
                rank_maps[source_name].get(track_id, 0)
                for source_name in LEXICAL_SOURCES
                if rank_maps[source_name].get(track_id, 0) > 0
            ]
            shallow = any(rank <= 20 for rank in present_ranks)
            deep = any(rank > 100 for rank in present_ranks)
            best_dense = min(dense_ranks) if dense_ranks else 0
            best_lexical = min(lexical_ranks) if lexical_ranks else 0
            strong_shallow = any(
                0 < rank_maps[source_name].get(track_id, 0) <= 20 for source_name in STRONG_SOURCES
            )

            pos = agreement_offset
            row[pos] = float(n_sources)
            row[pos + 1] = float(strong_count)
            row[pos + 2] = float(min_rank)
            row[pos + 3] = float(best_dense)
            row[pos + 4] = float(best_lexical)
            row[pos + 5] = float(max_rank - min_rank) if present_ranks else 0.0
            row[pos + 6] = 1.0 if shallow else 0.0
            row[pos + 7] = 1.0 if deep else 0.0
            row[pos + 8] = 1.0 if present_ranks and all(rank > 100 for rank in present_ranks) else 0.0
            row[pos + 9] = 1.0 if n_sources == 1 else 0.0
            row[pos + 10] = 1.0 if n_sources == 1 and deep else 0.0
            row[pos + 11] = 1.0 if 0 < best_dense <= 20 else 0.0
            row[pos + 12] = 1.0 if 0 < best_lexical <= 20 else 0.0
            row[pos + 13] = 1.0 if strong_shallow else 0.0
            row[pos + 14] = 1.0 if shallow and deep else 0.0
            row[pos + 15] = r54_scores[case_idx].get(track_id, 0.0)
            row[pos + 16] = als_scores[local_idx]
            row[pos + 17] = scores.get(track_id, 0.0)
            base_rank = rrf_rank.get(track_id, 0)
            row[pos + 18] = 1.0 / base_rank if base_rank > 0 else 0.0
            row[pos + 19] = n_prior_music
            row[pos + 20] = float(len(unique_artists))
            candidate_artist = track_artist.get(track_id, "")
            row[pos + 21] = 1.0 if candidate_artist and candidate_artist == last_artist else 0.0
            row[pos + 22] = overlap_vals[0]
            row[pos + 23] = overlap_vals[1]
            row[pos + 24] = overlap_vals[2]
            row[pos + 25] = overlap_vals[3]
            row[pos + 26] = overlap_vals[4]
            if track_id == gt:
                y[row_idx] = 1.0

        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  features {case_idx + 1}/{len(cases)} ({elapsed:.0f}s)", flush=True)

    return X, y


def row_indices_for_cases(row_starts: np.ndarray, row_counts: np.ndarray, case_ids: list[int]) -> np.ndarray:
    total = int(sum(int(row_counts[case_idx]) for case_idx in case_ids))
    out = np.empty(total, dtype=np.int64)
    offset = 0
    for case_idx in case_ids:
        start = int(row_starts[case_idx])
        count = int(row_counts[case_idx])
        out[offset:offset + count] = np.arange(start, start + count, dtype=np.int64)
        offset += count
    return out


def train_cv5(
    X: np.ndarray,
    y: np.ndarray,
    payload: dict[str, Any],
    case_index: dict[str, Any],
) -> tuple[list[np.ndarray | None], dict[str, Any], list[dict[str, Any]]]:
    cases = payload["cases"]
    sessions = [case["session_id"] for case in cases]
    folds = grouped_session_folds(sessions, seed=0)
    row_starts = case_index["row_starts"]
    row_counts = case_index["row_counts"]
    n_cases = len(cases)
    oof_scores: list[np.ndarray | None] = [None] * n_cases
    importances = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    fold_reports = []
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [POOL_K],
        "label_gain": [0, 1],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 0,
        "num_threads": 4,
        "force_col_wise": True,
        "deterministic": True,
    }

    print(f"{ts()} Training grouped-session CV5 LambdaRank admission model...", flush=True)
    start_time = time.time()
    all_cases = set(range(n_cases))
    for fold_idx, fold in enumerate(folds):
        val_cases = sorted(int(idx) for idx in fold.tolist())
        val_set = set(val_cases)
        train_cases = sorted(all_cases - val_set)
        train_rows = row_indices_for_cases(row_starts, row_counts, train_cases)
        val_rows = row_indices_for_cases(row_starts, row_counts, val_cases)
        group_train = [int(row_counts[case_idx]) for case_idx in train_cases]
        group_val = [int(row_counts[case_idx]) for case_idx in val_cases]
        positives_train = int(np.sum(y[train_rows]))
        positives_val = int(np.sum(y[val_rows]))

        print(
            f"  fold {fold_idx}: train_rows={len(train_rows):,} val_rows={len(val_rows):,} "
            f"train_pos={positives_train} val_pos={positives_val}",
            flush=True,
        )

        dtrain = lgb.Dataset(
            X[train_rows],
            label=y[train_rows],
            group=group_train,
            feature_name=FEATURE_NAMES,
            free_raw_data=True,
        )
        dval = lgb.Dataset(
            X[val_rows],
            label=y[val_rows],
            group=group_val,
            reference=dtrain,
            feature_name=FEATURE_NAMES,
            free_raw_data=True,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[lgb.log_evaluation(0)],
        )
        preds = model.predict(X[val_rows]).astype(np.float32)
        importances += model.feature_importance(importance_type="gain")

        offset = 0
        fold_hits = []
        for case_idx in val_cases:
            count = int(row_counts[case_idx])
            case_scores = preds[offset:offset + count]
            offset += count
            oof_scores[case_idx] = case_scores.copy()
            candidates = case_index["candidate_ids_by_case"][case_idx]
            top_ids = top_k_ids(candidates, case_scores, POOL_K)
            fold_hits.append(payload["cases"][case_idx]["gt"] in set(top_ids))
        fold_pool_hit = float(np.mean(fold_hits)) if fold_hits else 0.0
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_positives": positives_train,
                "val_positives": positives_val,
                "learned_pool_hit": fold_pool_hit,
            }
        )
        print(f"    fold {fold_idx} learned admission pool_hit@300={fold_pool_hit:.4f}", flush=True)
        del dtrain, dval, model, preds, train_rows, val_rows

    elapsed = time.time() - start_time
    feature_importance = {
        name: float(score / len(folds))
        for name, score in zip(FEATURE_NAMES, importances, strict=True)
    }
    train_report = {
        "params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "elapsed_s": elapsed,
        "feature_importance": feature_importance,
    }
    return oof_scores, train_report, fold_reports


def top_k_ids(candidates: list[str], scores: np.ndarray, k: int) -> list[str]:
    if not candidates:
        return []
    count = min(k, len(candidates))
    order = np.argsort(-scores, kind="mergesort")[:count]
    return [candidates[int(idx)] for idx in order]


def case_splits(payload: dict[str, Any]) -> dict[str, list[int]]:
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    h7 = []
    same_artist = []
    diff_artist = []
    for case_idx, case in enumerate(cases):
        if int(case["n_prior_music"]) == 7:
            h7.append(case_idx)
        gt_artist = track_artist.get(case["gt"], "")
        played_artists = {track_artist.get(track_id, "") for track_id in case["music_turns"]} - {""}
        if gt_artist and gt_artist in played_artists:
            same_artist.append(case_idx)
        else:
            diff_artist.append(case_idx)
    return {
        "all_dev": list(range(len(cases))),
        "h7": h7,
        "same_artist": same_artist,
        "diff_artist": diff_artist,
    }


def summarize_subset(
    name: str,
    case_ids: list[int],
    baseline_hit: np.ndarray,
    learned_hit: np.ndarray,
    buckets: list[str],
) -> dict[str, Any]:
    if not case_ids:
        return {
            "name": name,
            "n": 0,
            "baseline_pool_hit": 0,
            "learned_pool_hit": 0,
            "baseline_pool_hit_rate": 0.0,
            "learned_pool_hit_rate": 0.0,
            "delta_pool_hit_rate": 0.0,
            "recovered_pool_miss": 0,
            "lost_previously_covered": 0,
            "net_pool_recovery": 0,
        }
    idx = np.asarray(case_ids, dtype=np.int64)
    base_count = int(np.sum(baseline_hit[idx]))
    learned_count = int(np.sum(learned_hit[idx]))
    recovered = sum(1 for case_idx in case_ids if buckets[case_idx] == "POOL_MISS" and learned_hit[case_idx])
    lost = int(np.sum(baseline_hit[idx] & ~learned_hit[idx]))
    return {
        "name": name,
        "n": len(case_ids),
        "baseline_pool_hit": base_count,
        "learned_pool_hit": learned_count,
        "baseline_pool_hit_rate": base_count / len(case_ids),
        "learned_pool_hit_rate": learned_count / len(case_ids),
        "delta_pool_hit_rate": (learned_count - base_count) / len(case_ids),
        "recovered_pool_miss": recovered,
        "lost_previously_covered": lost,
        "net_pool_recovery": recovered - lost,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p00": float(np.percentile(arr, 0)) if len(arr) else 0.0,
        "p10": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "p25": float(np.percentile(arr, 25)) if len(arr) else 0.0,
        "p75": float(np.percentile(arr, 75)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "p100": float(np.percentile(arr, 100)) if len(arr) else 0.0,
    }


def evaluate(
    payload: dict[str, Any],
    case_index: dict[str, Any],
    oof_scores: list[np.ndarray | None],
    buckets: list[str],
    decomp: dict[str, Any],
) -> dict[str, Any]:
    cases = payload["cases"]
    n_cases = len(cases)
    baseline_hit = case_index["baseline_hit"]
    source_union_has_gt = case_index["source_union_has_gt"]
    candidate_ids_by_case = case_index["candidate_ids_by_case"]
    baseline_pools = case_index["baseline_pools"]
    learned_hit = np.zeros(n_cases, dtype=bool)
    overlap_fracs = []
    learned_top300_by_case: list[list[str]] = []

    print(f"{ts()} Evaluating learned admission@300 vs weighted_rrf@300...", flush=True)
    for case_idx, case in enumerate(cases):
        scores = oof_scores[case_idx]
        candidates = candidate_ids_by_case[case_idx]
        if scores is None:
            top_ids: list[str] = []
        else:
            top_ids = top_k_ids(candidates, scores, POOL_K)
        learned_top300_by_case.append(top_ids)
        top_set = set(top_ids)
        learned_hit[case_idx] = case["gt"] in top_set
        baseline_set = set(baseline_pools[case_idx])
        overlap_fracs.append(len(top_set & baseline_set) / POOL_K)

    split_metrics = {
        split_name: summarize_subset(split_name, case_ids, baseline_hit, learned_hit, buckets)
        for split_name, case_ids in case_splits(payload).items()
    }

    bucket_breakdown = {}
    for bucket_name in ["HIT", "DEMOTED", "POOL_MISS", "UNREACHABLE"]:
        case_ids = [idx for idx, bucket in enumerate(buckets) if bucket == bucket_name]
        bucket_breakdown[bucket_name] = summarize_subset(bucket_name, case_ids, baseline_hit, learned_hit, buckets)

    recovered_pool_miss = [
        case_idx
        for case_idx in range(n_cases)
        if buckets[case_idx] == "POOL_MISS" and learned_hit[case_idx]
    ]
    lost_previously = [
        case_idx
        for case_idx in range(n_cases)
        if baseline_hit[case_idx] and not learned_hit[case_idx]
    ]
    gained_pool_miss = len(recovered_pool_miss)
    lost_previously_count = len(lost_previously)
    net_pool_recovery = gained_pool_miss - lost_previously_count

    source_counts = Counter()
    pattern_counts = Counter()
    single_source_counts = Counter()
    for case_idx in recovered_pool_miss:
        sources = case_index["gt_sources"][case_idx]
        source_counts.update(sources)
        pattern = "+".join(sources) if sources else "NONE"
        pattern_counts[pattern] += 1
        if len(sources) == 1:
            single_source_counts[sources[0]] += 1

    top_pattern, top_pattern_count = ("NONE", 0)
    if pattern_counts:
        top_pattern, top_pattern_count = pattern_counts.most_common(1)[0]
    single_source_recovered = int(sum(single_source_counts.values()))
    top_single_source_count = single_source_counts.most_common(1)[0][1] if single_source_counts else 0
    recovered_denom = max(gained_pool_miss, 1)
    gains_concentrated = (
        gained_pool_miss >= 100
        and single_source_recovered / recovered_denom > 0.50
        and top_single_source_count / recovered_denom > 0.35
    )

    baseline_pool_hit = float(np.mean(baseline_hit))
    learned_pool_hit = float(np.mean(learned_hit))
    decomp_pool_hit = float(decomp.get("pool_hit", 0.0))
    baseline_reproduction = {
        "computed_pool_hit": baseline_pool_hit,
        "r55_pool_hit": decomp_pool_hit,
        "abs_delta": abs(baseline_pool_hit - decomp_pool_hit),
        "epsilon": BASELINE_EPS,
        "pass": abs(baseline_pool_hit - decomp_pool_hit) <= BASELINE_EPS,
    }
    previously_covered = int(np.sum(baseline_hit))
    pool_miss_count = int(Counter(buckets)["POOL_MISS"])

    gate_checks = {
        "learned_pool_hit_beats_weighted_rrf": learned_pool_hit > baseline_pool_hit,
        "net_pool_recovery_positive": net_pool_recovery > 0,
        "recovered_pool_miss_at_least_100": gained_pool_miss >= 100,
        "loss_lte_25pct_gained": lost_previously_count <= 0.25 * gained_pool_miss,
        "gains_not_concentrated_single_source": not gains_concentrated,
    }
    if all(gate_checks.values()):
        verdict = "PROCEED"
    else:
        verdict = "ARCHIVE"

    return {
        "baseline_reproduction": baseline_reproduction,
        "n_cases": n_cases,
        "baseline_pool_hit": baseline_pool_hit,
        "learned_pool_hit": learned_pool_hit,
        "delta_pool_hit": learned_pool_hit - baseline_pool_hit,
        "source_union_hit": float(np.mean(source_union_has_gt)),
        "pool_miss_cases": pool_miss_count,
        "previously_covered_cases": previously_covered,
        "pool_miss_recovered": gained_pool_miss,
        "previously_covered_lost": lost_previously_count,
        "net_pool_recovery": net_pool_recovery,
        "split_metrics": split_metrics,
        "bucket_breakdown": bucket_breakdown,
        "top300_overlap_with_weighted_rrf": percentile_summary(overlap_fracs),
        "source_coverage_recovered_cases": {
            "by_source": dict(source_counts),
            "by_pattern_top20": dict(pattern_counts.most_common(20)),
            "single_source_only": dict(single_source_counts),
            "top_pattern": top_pattern,
            "top_pattern_share": top_pattern_count / recovered_denom,
            "single_source_recovered": single_source_recovered,
            "single_source_share": single_source_recovered / recovered_denom,
            "gains_concentrated": gains_concentrated,
        },
        "gate_checks": gate_checks,
        "verdict": verdict,
    }


def load_decomp() -> dict[str, Any]:
    with open(DECOMP_JSON) as f:
        return json.load(f)


def top_feature_importance(train_report: dict[str, Any], n: int = 10) -> list[tuple[str, float]]:
    imp = train_report["feature_importance"]
    return sorted(((name, float(score)) for name, score in imp.items()), key=lambda item: -item[1])[:n]


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def fmt_rate(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: dict[str, Any]) -> None:
    evaluation = report["evaluation"]
    train_report = report["training"]
    lines = [
        "# R59 C3 Pool-Admission Diagnostic Result",
        "",
        f"Created: {report['created_at']}",
        "",
        "## Verdict",
        "",
        f"**{evaluation['verdict']}**",
        "",
        "Recommendation: "
        + (
            "archive C3; learned admission did not clear the Phase B pool gate."
            if evaluation["verdict"] == "ARCHIVE"
            else "proceed only to a later dev-only frozen-LR conversion check."
        ),
        "",
        "## Baseline Reproduction",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| R55 weighted_rrf pool_hit@300 | {evaluation['baseline_reproduction']['r55_pool_hit']:.4f} |",
        f"| This script weighted_rrf pool_hit@300 | {evaluation['baseline_reproduction']['computed_pool_hit']:.4f} |",
        f"| Abs delta | {evaluation['baseline_reproduction']['abs_delta']:.6f} |",
        f"| Pass epsilon {BASELINE_EPS:.4f} | {evaluation['baseline_reproduction']['pass']} |",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| weighted_rrf pool_hit@300 | {evaluation['baseline_pool_hit']:.4f} |",
        f"| learned admission pool_hit@300 | {evaluation['learned_pool_hit']:.4f} |",
        f"| delta pool_hit@300 | {evaluation['delta_pool_hit']:+.4f} |",
        f"| POOL_MISS recovered | {evaluation['pool_miss_recovered']} / {evaluation['pool_miss_cases']} |",
        f"| Previously-covered GT lost | {evaluation['previously_covered_lost']} / {evaluation['previously_covered_cases']} |",
        f"| Net pool recovery | {evaluation['net_pool_recovery']} |",
        "",
        "## Breakdowns",
        "",
        "| Split | n | RRF hit | Learned hit | Delta | Recovered POOL_MISS | Lost covered | Net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in ["all_dev", "h7", "same_artist", "diff_artist"]:
        metrics = evaluation["split_metrics"][split_name]
        lines.append(
            f"| {split_name} | {metrics['n']} | {fmt_rate(metrics['baseline_pool_hit_rate'])} | "
            f"{fmt_rate(metrics['learned_pool_hit_rate'])} | {metrics['delta_pool_hit_rate']:+.4f} | "
            f"{metrics['recovered_pool_miss']} | {metrics['lost_previously_covered']} | "
            f"{metrics['net_pool_recovery']} |"
        )
    lines.extend(
        [
            "",
            "## Bucket Of Origin",
            "",
            "| Bucket | n | RRF hit | Learned hit | Delta | Recovered POOL_MISS | Lost covered |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket_name in ["HIT", "DEMOTED", "POOL_MISS", "UNREACHABLE"]:
        metrics = evaluation["bucket_breakdown"][bucket_name]
        lines.append(
            f"| {bucket_name} | {metrics['n']} | {fmt_rate(metrics['baseline_pool_hit_rate'])} | "
            f"{fmt_rate(metrics['learned_pool_hit_rate'])} | {metrics['delta_pool_hit_rate']:+.4f} | "
            f"{metrics['recovered_pool_miss']} | {metrics['lost_previously_covered']} |"
        )
    overlap = evaluation["top300_overlap_with_weighted_rrf"]
    lines.extend(
        [
            "",
            "## Top-300 Overlap With Weighted RRF",
            "",
            "| Statistic | Overlap fraction |",
            "|---|---:|",
            f"| mean | {overlap['mean']:.4f} |",
            f"| median | {overlap['median']:.4f} |",
            f"| p10 | {overlap['p10']:.4f} |",
            f"| p25 | {overlap['p25']:.4f} |",
            f"| p75 | {overlap['p75']:.4f} |",
            f"| p90 | {overlap['p90']:.4f} |",
            "",
            "## Source Coverage Of Recovered POOL_MISS Cases",
            "",
            "| Source | Recovered GT count |",
            "|---|---:|",
        ]
    )
    for source_name, count in sorted(evaluation["source_coverage_recovered_cases"]["by_source"].items()):
        lines.append(f"| {source_name} | {count} |")
    lines.extend(
        [
            "",
            "Top source patterns:",
            "",
            "| Pattern | Count |",
            "|---|---:|",
        ]
    )
    for pattern, count in evaluation["source_coverage_recovered_cases"]["by_pattern_top20"].items():
        lines.append(f"| {pattern} | {count} |")
    lines.extend(
        [
            "",
            "## Gate Checks",
            "",
            "| Gate | Pass |",
            "|---|---:|",
        ]
    )
    for gate_name, passed in evaluation["gate_checks"].items():
        lines.append(f"| {gate_name} | {passed} |")
    lines.extend(
        [
            "",
            "## LightGBM Feature Importances",
            "",
            "| Rank | Feature | Mean gain |",
            "|---:|---|---:|",
        ]
    )
    for rank, (feature_name, score) in enumerate(top_feature_importance(train_report), start=1):
        lines.append(f"| {rank} | {feature_name} | {score:.2f} |")
    lines.extend(
        [
            "",
            "## Recommendation For Next Phase",
            "",
            (
                "Do not run a frozen-LR conversion phase. The diagnostic is a pool-level "
                "falsifier for C3 unless all gate checks pass."
                if evaluation["verdict"] == "ARCHIVE"
                else "Run a separate dev-only frozen-LR conversion phase. Keep LR frozen and do not add admission_score as an LR feature."
            ),
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.time()
    print("R59 Phase B / C3 learned pool-admission diagnostic")
    print("=" * 70)
    payload, r21_source, r54_source, r54_scores = load_payloads()
    factors, track_ids, track_to_idx = load_als_cache()
    decomp = load_decomp()

    case_index = build_case_index(
        payload,
        r21_source,
        r54_source,
        r54_scores,
        factors,
        track_ids,
        track_to_idx,
    )
    buckets, bucket_details = load_bucket_labels(
        len(payload["cases"]),
        case_index["baseline_hit"],
        case_index["source_union_has_gt"],
    )
    print(f"  bucket counts: {dict(Counter(buckets))}", flush=True)

    X, y = build_feature_matrix(payload, r21_source, r54_source, case_index, factors, track_to_idx)
    oof_scores, train_report, fold_reports = train_cv5(X, y, payload, case_index)
    evaluation = evaluate(payload, case_index, oof_scores, buckets, decomp)

    report = {
        "experiment": "R59 Phase B C3 pool-admission diagnostic",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start_time,
        "dev_only": True,
        "blind_access": False,
        "lr_retraining": False,
        "candidate_universe": "A/B/C/D/F/ALS/R21 OOF/R54 Phase 2 OOF source union",
        "model": "LightGBM LambdaRank admission model",
        "feature_names": FEATURE_NAMES,
        "row_count": int(case_index["total_rows"]),
        "bucket_details": bucket_details,
        "folds": fold_reports,
        "training": train_report,
        "evaluation": evaluation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=json_default)
    write_markdown(report)

    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"{ts()} Verdict: {evaluation['verdict']}", flush=True)
    print(f"Elapsed: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
