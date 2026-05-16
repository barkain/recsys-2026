#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R59 C3 Phase 2: frozen-LR conversion on learned admission pools.

Dev-only conversion check. This script:
  - rebuilds the Phase 1 OOF learned-admission@300 pool;
  - scores both weighted_rrf@300 and learned-admission@300 pools with the
    frozen production R54c LightGBM LambdaRank model;
  - does not retrain LR and does not add admission_score as an LR feature.

Outputs:
  exp/eval/expR59_c3_phase2_frozen_lr.json
  docs/r59_candidates/c3_phase2_frozen_lr_result.md
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
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_ALL,
    FEAT_BASE,
    FEAT_R39_ALL,
    load_track_albums,
)
from scripts.tune_postrank_v23 import tokens  # noqa: E402

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
MAPS_CACHE = REPO / "cache" / "r54_phase3_payload_maps.pkl"
POP_CACHE = REPO / "cache" / "r54_phase3_track_pop.json"
OUT_JSON = REPO / "exp" / "eval" / "expR59_c3_phase2_frozen_lr.json"
OUT_MD = REPO / "docs" / "r59_candidates" / "c3_phase2_frozen_lr_result.md"

POOL_K = 300
TOP_K = 20
BASELINE_EPS = 0.0005
GATE_PROD_H7_DELTA = 0.010
GATE_EXP_H7_DELTA = 0.005
GATE_TOP1_CHURN_PER_80 = 25
GATE_TOP20_OVERLAP = 14.0


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def load_supporting_maps() -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    with open(MAPS_CACHE, "rb") as f:
        maps = pickle.load(f)
    with open(POP_CACHE) as f:
        track_pop = json.load(f)
    track_album = load_track_albums()
    return maps, track_pop, track_album


def ndcg_from_rank(rank: int, cutoff: int) -> float:
    if 1 <= rank <= cutoff:
        return float(1.0 / np.log2(rank + 1))
    return 0.0


def verify_oof_baseline_reproduction(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the OOF R39+R54 baseline before any learned-pool comparison."""
    print(f"{ts()} Baseline reproduction check from R58 OOF top-50 cache...", flush=True)
    with open(c3.DECOMP_JSON) as f:
        decomp = json.load(f)
    if not c3.R58_TOP50.exists():
        raise FileNotFoundError(f"Required OOF baseline cache missing: {c3.R58_TOP50}")
    with open(c3.R58_TOP50, "rb") as f:
        rows = pickle.load(f)
    gt_rank_by_case: dict[int, int] = {}
    for row in rows:
        case_idx = int(row["case_idx"])
        if case_idx not in gt_rank_by_case:
            gt_rank_by_case[case_idx] = int(row["gt_lr_rank"])
    ndcg = np.asarray(
        [ndcg_from_rank(gt_rank_by_case.get(case_idx, -1), 20) for case_idx in range(len(cases))],
        dtype=np.float64,
    )
    h7_idx = [case_idx for case_idx, case in enumerate(cases) if case["n_prior_music"] == 7]
    all_ndcg = float(ndcg.mean())
    h7_ndcg = mean_for_indices(ndcg, h7_idx)
    all_delta = all_ndcg - float(decomp["overall_ndcg"])
    h7_delta = h7_ndcg - float(decomp["h7_ndcg"])
    passed = abs(all_delta) <= BASELINE_EPS and abs(h7_delta) <= BASELINE_EPS
    report = {
        "source": str(c3.R58_TOP50),
        "r55_overall_ndcg": float(decomp["overall_ndcg"]),
        "r55_h7_ndcg": float(decomp["h7_ndcg"]),
        "computed_overall_ndcg": all_ndcg,
        "computed_h7_ndcg": h7_ndcg,
        "overall_abs_delta": abs(all_delta),
        "h7_abs_delta": abs(h7_delta),
        "epsilon": BASELINE_EPS,
        "pass": passed,
    }
    print(
        f"  all-dev nDCG={all_ndcg:.5f} vs R55 {float(decomp['overall_ndcg']):.5f} "
        f"abs_delta={abs(all_delta):.6f}",
        flush=True,
    )
    print(
        f"  h7 nDCG={h7_ndcg:.5f} vs R55 {float(decomp['h7_ndcg']):.5f} "
        f"abs_delta={abs(h7_delta):.6f}",
        flush=True,
    )
    if not passed:
        raise RuntimeError("Baseline reproduction failed; aborting learned-pool comparison")
    print("  baseline reproduction PASS", flush=True)
    return report


def featurize_for_frozen_lr(
    pool: list[str],
    source_lists: dict[str, list[str]],
    r21_rank_map: dict[str, int],
    r54_rank_map: dict[str, int],
    r54_score_map: dict[str, float],
    case: dict[str, Any],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    als_vec: np.ndarray | None,
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
    rrf_full_rank_map: dict[str, int] | None = None,
) -> np.ndarray:
    """Build the exact frozen 37-feature LR rows, with optional RRF rank pinning.

    For learned-admission pools, C3 requires keeping the LR base rank feature
    tied to weighted-RRF rank over the full source union, not admission rank.
    """
    n_base = len(FEAT_BASE)
    n_r39 = len(FEAT_R39_ALL)
    feats = np.zeros((len(pool), len(FEAT_ALL)), dtype=np.float64)

    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]

    history = case["history"]
    case_query = case["user_query"]
    played = case["music_turns"]
    played_set = set(played)
    user_msgs = [str(row["content"]) for row in history if row["role"] == "user"] + [case_query]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    last_artist = ta.get(played[-1], "") if played else ""
    last_tags = tt.get(played[-1], set()) if played else set()
    prior = [
        (1.0 / (idx + 1), ta.get(track_id, ""), tt.get(track_id, set()))
        for idx, track_id in enumerate(reversed(played))
    ]
    pool_artists = [ta.get(track_id, "") for track_id in pool[:POOL_K]]
    artist_counts = Counter(artist for artist in pool_artists if artist)
    source_rank = {
        source_name: {track_id: rank + 1 for rank, track_id in enumerate(source_lists[source_name])}
        for source_name in c3.SOURCE_NAMES
    }

    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(track_id, "") for track_id in played[-3:]} - {""}
    all_albums = [track_album.get(track_id, "") for track_id in played]
    album_hist_counts = Counter(album for album in all_albums if album)

    for rank, track_id in enumerate(pool, start=1):
        candidate_artist = ta.get(track_id, "")
        candidate_tags = tt.get(track_id, set())
        row = feats[rank - 1]
        rank_for_feature = rrf_full_rank_map.get(track_id, 0) if rrf_full_rank_map else rank
        row[0] = 1.0 / rank_for_feature if rank_for_feature > 0 else 0.0
        row[1] = 1.0 if candidate_artist and candidate_artist == last_artist else 0.0
        if candidate_tags or last_tags:
            row[2] = len(candidate_tags & last_tags) / len(candidate_tags | last_tags)
        row[3] = float(len(tat.get(track_id, set()) & now_tok))
        row[4] = float(len(ttl.get(track_id, set()) & now_tok))
        row[5] = float(len(tmt.get(track_id, set()) & all_tok))
        row[6] = 1.0 if track_id in played_set else 0.0
        recency_score = 0.0
        for weight, prior_artist, prior_tags in prior:
            artist_match = 1.0 if candidate_artist and candidate_artist == prior_artist else 0.0
            tag_match = (
                len(candidate_tags & prior_tags) / len(candidate_tags | prior_tags)
                if (candidate_tags or prior_tags)
                else 0.0
            )
            recency_score += weight * (artist_match + tag_match)
        row[7] = recency_score
        for feat_idx, source_name in enumerate(["A", "B", "C", "D", "F", "ALS"]):
            source_rank_value = source_rank[source_name].get(track_id)
            row[8 + feat_idx] = 1.0 / source_rank_value if source_rank_value else 0.0
        for feat_idx, source_name in enumerate(["A", "B", "C", "D", "F", "ALS"]):
            row[14 + feat_idx] = 1.0 if track_id in source_rank[source_name] else 0.0
        row[20] = sum(
            1 for source_name in ["A", "B", "C", "D", "F", "ALS"]
            if track_id in source_rank[source_name]
        )
        if als_vec is not None:
            als_idx = als_to_idx.get(track_id)
            if als_idx is not None:
                row[21] = float(np.dot(als_vec, als_factors[als_idx]))
        row[22] = float(n_hist)
        row[23] = track_pop.get(track_id, 0) / max_pop
        row[24] = artist_counts.get(candidate_artist, 0) / max(len(pool), 1) if candidate_artist else 0.0
        row[25] = float(artist_counts.get(candidate_artist, 0)) if candidate_artist else 0.0
        row[26] = row[20]
        row[27] = 1.0 / r21_rank_map[track_id] if track_id in r21_rank_map else 0.0
        row[28] = 1.0 if track_id in r21_rank_map else 0.0

        candidate_album = track_album.get(track_id, "")
        row[n_base + 0] = 1.0 if candidate_album and candidate_album == last1_album else 0.0
        row[n_base + 1] = 1.0 if candidate_album and candidate_album in last3_albums else 0.0
        row[n_base + 2] = 1.0 if candidate_album and candidate_album in album_hist_counts else 0.0
        row[n_base + 3] = (
            float(album_hist_counts.get(candidate_album, 0)) / max(n_hist, 1)
            if candidate_album
            else 0.0
        )
        pool_album_count = (
            sum(1 for other_id in pool if track_album.get(other_id, "") == candidate_album)
            if candidate_album
            else 0
        )
        row[n_base + 4] = pool_album_count / max(len(pool), 1)

        row[n_r39 + 0] = 1.0 / r54_rank_map[track_id] if track_id in r54_rank_map else 0.0
        row[n_r39 + 1] = 1.0 if track_id in r54_rank_map else 0.0
        row[n_r39 + 2] = r54_score_map.get(track_id, 0.0)

    return feats


def score_pool(
    ranker: lgb.Booster,
    pool: list[str],
    source_lists: dict[str, list[str]],
    case: dict[str, Any],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    als_vec: np.ndarray | None,
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
    r54_score_map: dict[str, float],
    rrf_full_rank_map: dict[str, int] | None,
) -> dict[str, Any]:
    r21_rank_map = {track_id: rank + 1 for rank, track_id in enumerate(source_lists["R21"][:POOL_K])}
    r54_rank_map = {track_id: rank + 1 for rank, track_id in enumerate(source_lists["R54"][:POOL_K])}
    feats = featurize_for_frozen_lr(
        pool,
        source_lists,
        r21_rank_map,
        r54_rank_map,
        r54_score_map,
        case,
        maps,
        als_factors,
        als_to_idx,
        als_vec,
        track_pop,
        max_pop,
        track_album,
        rrf_full_rank_map=rrf_full_rank_map,
    )
    scores = ranker.predict(feats)
    order = np.argsort(-scores, kind="mergesort")
    top20 = [pool[int(idx)] for idx in order[:TOP_K]]
    gt = case["gt"]
    gt_rank = -1
    if gt in pool:
        gt_pool_idx = pool.index(gt)
        pos = np.where(order == gt_pool_idx)[0]
        if len(pos):
            gt_rank = int(pos[0]) + 1
    return {
        "top20": top20,
        "top1": top20[0] if top20 else "",
        "gt_rank": gt_rank,
        "ndcg_at_20": ndcg_from_rank(gt_rank, 20),
        "ndcg_at_10": ndcg_from_rank(gt_rank, 10),
        "ndcg_at_7": ndcg_from_rank(gt_rank, 7),
        "gt_in_pool": gt in pool,
    }


def same_artist_case(case: dict[str, Any], track_artist: dict[str, str]) -> bool:
    gt_artist = track_artist.get(case["gt"], "")
    played_artists = {track_artist.get(track_id, "") for track_id in case["music_turns"]} - {""}
    return bool(gt_artist and gt_artist in played_artists)


def mean_for_indices(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(values[np.asarray(indices, dtype=np.int64)].mean())


def metric_block(prefix: str, case_rows: list[dict[str, Any]], cases: list[dict[str, Any]], maps: dict[str, Any]) -> dict[str, Any]:
    ndcg20 = np.asarray([row[f"{prefix}_ndcg_at_20"] for row in case_rows], dtype=np.float64)
    ndcg10 = np.asarray([row[f"{prefix}_ndcg_at_10"] for row in case_rows], dtype=np.float64)
    ndcg7 = np.asarray([row[f"{prefix}_ndcg_at_7"] for row in case_rows], dtype=np.float64)
    h7_idx = [idx for idx, case in enumerate(cases) if case["n_prior_music"] == 7]
    same_idx = [idx for idx, case in enumerate(cases) if same_artist_case(case, maps["track_artist"])]
    diff_idx = [idx for idx in range(len(cases)) if idx not in set(same_idx)]
    h7_same_idx = [idx for idx in h7_idx if idx in set(same_idx)]
    h7_diff_idx = [idx for idx in h7_idx if idx not in set(same_idx)]
    by_depth = {}
    for depth in range(8):
        label = f"h{depth}"
        idxs = [
            idx for idx, case in enumerate(cases)
            if min(int(case["n_prior_music"]), 7) == depth
        ]
        by_depth[label] = {
            "n": len(idxs),
            "ndcg_at_20": mean_for_indices(ndcg20, idxs),
            "ndcg_at_10": mean_for_indices(ndcg10, idxs),
            "ndcg_at_7": mean_for_indices(ndcg7, idxs),
        }
    return {
        "all": {
            "n": len(cases),
            "ndcg_at_20": float(ndcg20.mean()),
            "ndcg_at_10": float(ndcg10.mean()),
            "ndcg_at_7": float(ndcg7.mean()),
        },
        "h7": {
            "n": len(h7_idx),
            "ndcg_at_20": mean_for_indices(ndcg20, h7_idx),
            "ndcg_at_10": mean_for_indices(ndcg10, h7_idx),
            "ndcg_at_7": mean_for_indices(ndcg7, h7_idx),
        },
        "same_artist": {
            "n": len(same_idx),
            "ndcg_at_20": mean_for_indices(ndcg20, same_idx),
            "ndcg_at_10": mean_for_indices(ndcg10, same_idx),
            "ndcg_at_7": mean_for_indices(ndcg7, same_idx),
        },
        "diff_artist": {
            "n": len(diff_idx),
            "ndcg_at_20": mean_for_indices(ndcg20, diff_idx),
            "ndcg_at_10": mean_for_indices(ndcg10, diff_idx),
            "ndcg_at_7": mean_for_indices(ndcg7, diff_idx),
        },
        "h7_same_artist": {
            "n": len(h7_same_idx),
            "ndcg_at_20": mean_for_indices(ndcg20, h7_same_idx),
            "ndcg_at_10": mean_for_indices(ndcg10, h7_same_idx),
            "ndcg_at_7": mean_for_indices(ndcg7, h7_same_idx),
        },
        "h7_diff_artist": {
            "n": len(h7_diff_idx),
            "ndcg_at_20": mean_for_indices(ndcg20, h7_diff_idx),
            "ndcg_at_10": mean_for_indices(ndcg10, h7_diff_idx),
            "ndcg_at_7": mean_for_indices(ndcg7, h7_diff_idx),
        },
        "by_depth": by_depth,
    }


def delta_metrics(rrf_metrics: dict[str, Any], learned_metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split_name in ["all", "h7", "same_artist", "diff_artist", "h7_same_artist", "h7_diff_artist"]:
        out[split_name] = {
            "n": learned_metrics[split_name]["n"],
            "delta_ndcg_at_20": learned_metrics[split_name]["ndcg_at_20"] - rrf_metrics[split_name]["ndcg_at_20"],
            "delta_ndcg_at_10": learned_metrics[split_name]["ndcg_at_10"] - rrf_metrics[split_name]["ndcg_at_10"],
            "delta_ndcg_at_7": learned_metrics[split_name]["ndcg_at_7"] - rrf_metrics[split_name]["ndcg_at_7"],
        }
    return out


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def evaluate_churn(case_rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    h7_indices = [idx for idx, case in enumerate(cases) if case["n_prior_music"] == 7]
    top1_changed_all = sum(1 for row in case_rows if row["top1_changed"])
    top1_changed_h7 = sum(1 for idx in h7_indices if case_rows[idx]["top1_changed"])
    overlap_all = [float(row["top20_overlap"]) for row in case_rows]
    overlap_h7 = [float(case_rows[idx]["top20_overlap"]) for idx in h7_indices]
    h7_churn_equiv_80 = top1_changed_h7 / max(len(h7_indices), 1) * 80.0
    return {
        "all": {
            "n": len(case_rows),
            "top1_changed": top1_changed_all,
            "top1_churn_rate": top1_changed_all / max(len(case_rows), 1),
            "top20_overlap": percentile_summary(overlap_all),
        },
        "h7": {
            "n": len(h7_indices),
            "top1_changed": top1_changed_h7,
            "top1_churn_rate": top1_changed_h7 / max(len(h7_indices), 1),
            "top1_churn_equiv_per_80": h7_churn_equiv_80,
            "top20_overlap": percentile_summary(overlap_h7),
        },
    }


def conversion_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [
        row for row in case_rows
        if row["learned_ndcg_at_20"] > 0.0 and row["rrf_ndcg_at_20"] == 0.0
    ]
    lost = [
        row for row in case_rows
        if row["rrf_ndcg_at_20"] > 0.0 and row["learned_ndcg_at_20"] == 0.0
    ]
    pool_miss_admitted = [
        row for row in case_rows
        if row["bucket"] == "POOL_MISS" and row["learned_gt_in_pool"]
    ]
    pool_miss_top20 = [
        row for row in pool_miss_admitted
        if 1 <= int(row["learned_gt_rank"]) <= 20
    ]
    pool_miss_buried = [
        row for row in pool_miss_admitted
        if int(row["learned_gt_rank"]) > 20
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
            "gt_admitted_but_buried": len(pool_miss_buried),
            "convert_rate": len(pool_miss_top20) / max(len(pool_miss_admitted), 1),
        },
    }


def gate_verdict(delta: dict[str, Any], churn: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    h7_delta = delta["h7"]["delta_ndcg_at_20"]
    same_delta = delta["same_artist"]["delta_ndcg_at_20"]
    diff_delta = delta["diff_artist"]["delta_ndcg_at_20"]
    top1_ok = churn["h7"]["top1_churn_equiv_per_80"] <= GATE_TOP1_CHURN_PER_80
    top20_ok = churn["h7"]["top20_overlap"]["median"] >= GATE_TOP20_OVERLAP
    recovered_gt_top20 = delta["top20_recovered_lost"]["recovered"]
    lost_gt_top20 = delta["top20_recovered_lost"]["lost"]
    churn_controlled = top1_ok and top20_ok
    no_same_diff_regression = same_delta >= 0.0 and diff_delta >= 0.0
    recovered_gt_exceeds_lost = recovered_gt_top20 > lost_gt_top20
    checks = {
        "h7_delta_ge_010": h7_delta >= GATE_PROD_H7_DELTA,
        "h7_delta_ge_005": h7_delta >= GATE_EXP_H7_DELTA,
        "same_artist_nonnegative": same_delta >= 0.0,
        "diff_artist_nonnegative": diff_delta >= 0.0,
        "top1_churn_ok": top1_ok,
        "top20_overlap_ok": top20_ok,
        "churn_controlled": churn_controlled,
        "recovered_gt_exceeds_lost": recovered_gt_exceeds_lost,
    }
    if checks["h7_delta_ge_010"] and no_same_diff_regression:
        return "PRODUCTION_CANDIDATE", checks
    if checks["h7_delta_ge_005"] and churn_controlled and recovered_gt_exceeds_lost:
        return "EXPLORATORY", checks
    return "ARCHIVE", checks


def write_markdown(report: dict[str, Any]) -> None:
    baseline = report["baseline_reproduction"]
    rrf = report["metrics"]["rrf"]
    learned = report["metrics"]["learned"]
    delta = report["metrics"]["delta"]
    churn = report["churn"]
    pool_hit = report["pool_hit"]
    conversion = delta["pool_miss_conversion"]
    top20_delta = delta["top20_recovered_lost"]
    lines = [
        "# R59 C3 Phase 2 Frozen-LR Conversion",
        "",
        f"Created: {report['created_at']}",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Baseline Reproduction",
        "",
        "| Metric | R55 decomp | Recomputed OOF | Abs delta | Pass |",
        "|---|---:|---:|---:|---:|",
        (
            f"| all-dev nDCG@20 | {baseline['r55_overall_ndcg']:.5f} | "
            f"{baseline['computed_overall_ndcg']:.5f} | {baseline['overall_abs_delta']:.6f} | "
            f"{baseline['pass']} |"
        ),
        (
            f"| h7 nDCG@20 | {baseline['r55_h7_ndcg']:.5f} | "
            f"{baseline['computed_h7_ndcg']:.5f} | {baseline['h7_abs_delta']:.6f} | "
            f"{baseline['pass']} |"
        ),
        "",
        "## Headline",
        "",
        "| Metric | RRF pool | Learned pool | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| h7 nDCG@20 | {rrf['h7']['ndcg_at_20']:.5f} | "
            f"{learned['h7']['ndcg_at_20']:.5f} | {delta['h7']['delta_ndcg_at_20']:+.5f} |"
        ),
        (
            f"| all-dev nDCG@20 | {rrf['all']['ndcg_at_20']:.5f} | "
            f"{learned['all']['ndcg_at_20']:.5f} | {delta['all']['delta_ndcg_at_20']:+.5f} |"
        ),
        (
            f"| all-dev nDCG@10 | {rrf['all']['ndcg_at_10']:.5f} | "
            f"{learned['all']['ndcg_at_10']:.5f} | {delta['all']['delta_ndcg_at_10']:+.5f} |"
        ),
        (
            f"| all-dev nDCG@7 | {rrf['all']['ndcg_at_7']:.5f} | "
            f"{learned['all']['ndcg_at_7']:.5f} | {delta['all']['delta_ndcg_at_7']:+.5f} |"
        ),
        (
            f"| same-artist nDCG@20 | {rrf['same_artist']['ndcg_at_20']:.5f} | "
            f"{learned['same_artist']['ndcg_at_20']:.5f} | "
            f"{delta['same_artist']['delta_ndcg_at_20']:+.5f} |"
        ),
        (
            f"| diff-artist nDCG@20 | {rrf['diff_artist']['ndcg_at_20']:.5f} | "
            f"{learned['diff_artist']['ndcg_at_20']:.5f} | "
            f"{delta['diff_artist']['delta_ndcg_at_20']:+.5f} |"
        ),
        (
            f"| pool_hit@300 | {pool_hit['rrf']:.5f} | "
            f"{pool_hit['learned']:.5f} | {pool_hit['delta']:+.5f} |"
        ),
        "",
        "## Conversion",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| top-20 recovered | {top20_delta['recovered']} |",
        f"| top-20 lost | {top20_delta['lost']} |",
        f"| top-20 net | {top20_delta['net']} |",
        f"| POOL_MISS admitted to learned pool | {conversion['admitted']} |",
        f"| POOL_MISS converted to frozen-LR top-20 | {conversion['top20']} |",
        f"| POOL_MISS admitted but buried 21-300 | {conversion['buried_21_300']} |",
        f"| convert rate | {conversion['top20']} / {conversion['admitted']} ({conversion['convert_rate']:.2%}) |",
        "",
        "## History Depth",
        "",
        "| Bucket | n | RRF nDCG@20 | Learned nDCG@20 | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for depth_name in sorted(rrf["by_depth"]):
        rrf_depth = rrf["by_depth"][depth_name]
        learned_depth = learned["by_depth"][depth_name]
        lines.append(
            f"| {depth_name} | {rrf_depth['n']} | {rrf_depth['ndcg_at_20']:.5f} | "
            f"{learned_depth['ndcg_at_20']:.5f} | "
            f"{learned_depth['ndcg_at_20'] - rrf_depth['ndcg_at_20']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Churn",
            "",
            "| Split | top-1 changed | top-1 equiv /80 | top-20 overlap mean | top-20 overlap median |",
            "|---|---:|---:|---:|---:|",
            (
                f"| all-dev | {churn['all']['top1_changed']}/{churn['all']['n']} | "
                f"{churn['all']['top1_churn_rate'] * 80.0:.1f}/80 | "
                f"{churn['all']['top20_overlap']['mean']:.2f}/20 | "
                f"{churn['all']['top20_overlap']['median']:.2f}/20 |"
            ),
            (
                f"| h7 | {churn['h7']['top1_changed']}/{churn['h7']['n']} | "
                f"{churn['h7']['top1_churn_equiv_per_80']:.1f}/80 | "
                f"{churn['h7']['top20_overlap']['mean']:.2f}/20 | "
                f"{churn['h7']['top20_overlap']['median']:.2f}/20 |"
            ),
            "",
            "## Gate Checks",
            "",
            "| Gate | Pass |",
            "|---|---:|",
        ]
    )
    for name, passed in report["gate_checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- LR model is loaded from `cache/r54_phase3_lr_model.txt` and is never retrained.",
            "- Learned admission is rebuilt OOF using the same fixed Phase 1 LambdaRank admission setup.",
            "- `admission_score` is not used as an LR feature.",
            "- Learned-pool `rrf_rank_inv` is pinned to full weighted-RRF source-union rank, not admission rank.",
            "- Dev R54 source/cosine features use the preserved Phase 2 OOF proxy.",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.time()
    print("R59 C3 Phase 2 frozen-LR conversion")
    print("=" * 70)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    baseline_reproduction = verify_oof_baseline_reproduction(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    ranker = lgb.Booster(model_file=str(LR_MODEL))
    if ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError(f"Frozen LR feature count mismatch: model={ranker.num_feature()} expected={len(FEAT_ALL)}")

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

    X_admit, y_admit = c3.build_feature_matrix(
        payload,
        r21_source,
        r54_source,
        case_index,
        als_factors,
        als_to_idx,
    )
    oof_scores, admission_train_report, fold_reports = c3.train_cv5(
        X_admit,
        y_admit,
        payload,
        case_index,
    )
    del X_admit, y_admit
    gc.collect()

    print(f"{ts()} Scoring RRF and learned pools with frozen LR...", flush=True)
    case_rows = []
    learned_pool_hit = 0
    rrf_pool_hit = 0
    score_start = time.time()
    for case_idx, case in enumerate(cases):
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
        rrf_pool = case_index["baseline_pools"][case_idx]
        learned_scores = oof_scores[case_idx]
        if learned_scores is None:
            learned_pool = []
        else:
            learned_pool = c3.top_k_ids(
                case_index["candidate_ids_by_case"][case_idx],
                learned_scores,
                POOL_K,
            )
        if case["gt"] in rrf_pool:
            rrf_pool_hit += 1
        if case["gt"] in learned_pool:
            learned_pool_hit += 1

        als_vec = case_index["als_session_vecs"][case_idx]
        r54_score_map = case_index["r54_scores"][case_idx]
        rrf_result = score_pool(
            ranker,
            rrf_pool,
            source_lists,
            case,
            maps,
            als_factors,
            als_to_idx,
            als_vec,
            track_pop,
            max_pop,
            track_album,
            r54_score_map,
            rrf_full_rank_map=None,
        )
        learned_result = score_pool(
            ranker,
            learned_pool,
            source_lists,
            case,
            maps,
            als_factors,
            als_to_idx,
            als_vec,
            track_pop,
            max_pop,
            track_album,
            r54_score_map,
            rrf_full_rank_map=full_rrf_rank,
        )
        top20_overlap = len(set(rrf_result["top20"]) & set(learned_result["top20"]))
        case_rows.append(
            {
                "case_idx": case_idx,
                "session_id": case["session_id"],
                "n_prior_music": int(case["n_prior_music"]),
                "bucket": buckets[case_idx],
                "same_artist": same_artist_case(case, maps["track_artist"]),
                "rrf_gt_in_pool": bool(rrf_result["gt_in_pool"]),
                "learned_gt_in_pool": bool(learned_result["gt_in_pool"]),
                "rrf_gt_rank": int(rrf_result["gt_rank"]),
                "learned_gt_rank": int(learned_result["gt_rank"]),
                "rrf_ndcg_at_20": float(rrf_result["ndcg_at_20"]),
                "learned_ndcg_at_20": float(learned_result["ndcg_at_20"]),
                "rrf_ndcg_at_10": float(rrf_result["ndcg_at_10"]),
                "learned_ndcg_at_10": float(learned_result["ndcg_at_10"]),
                "rrf_ndcg_at_7": float(rrf_result["ndcg_at_7"]),
                "learned_ndcg_at_7": float(learned_result["ndcg_at_7"]),
                "top1_changed": rrf_result["top1"] != learned_result["top1"],
                "top20_overlap": int(top20_overlap),
            }
        )
        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - score_start
            print(f"  scored {case_idx + 1}/{len(cases)} cases ({elapsed:.0f}s)", flush=True)

    rrf_metrics = metric_block("rrf", case_rows, cases, maps)
    learned_metrics = metric_block("learned", case_rows, cases, maps)
    deltas = delta_metrics(rrf_metrics, learned_metrics)
    conversion = conversion_summary(case_rows)
    deltas.update(conversion)
    churn = evaluate_churn(case_rows, cases)
    verdict, gate_checks = gate_verdict(deltas, churn)

    report = {
        "experiment": "R59 C3 Phase 2 frozen-LR conversion",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start_time,
        "dev_only": True,
        "blind_access": False,
        "lr_model": str(LR_MODEL),
        "lr_retraining": False,
        "admission_score_as_lr_feature": False,
        "baseline_reproduction": baseline_reproduction,
        "r54_dev_lists": str(c3.R54_PHASE2_OOF),
        "feature_names": FEAT_ALL,
        "pool_hit": {
            "rrf": rrf_pool_hit / len(cases),
            "learned": learned_pool_hit / len(cases),
            "delta": (learned_pool_hit - rrf_pool_hit) / len(cases),
            "rrf_count": rrf_pool_hit,
            "learned_count": learned_pool_hit,
        },
        "bucket_details": bucket_details,
        "admission_folds": fold_reports,
        "admission_training": {
            "params": admission_train_report["params"],
            "num_boost_round": admission_train_report["num_boost_round"],
            "elapsed_s": admission_train_report["elapsed_s"],
        },
        "metrics": {
            "rrf": rrf_metrics,
            "learned": learned_metrics,
            "delta": deltas,
        },
        "churn": churn,
        "gate_checks": gate_checks,
        "verdict": verdict,
        "per_case": case_rows,
        "notes": (
            "Frozen LR conversion uses cached production LR model and preserves OOF "
            "source discipline for dev R21/R54 lists. Learned-pool rrf_rank_inv is "
            "computed from full weighted-RRF source-union rank, not admission rank."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report)

    h7_delta = deltas["h7"]["delta_ndcg_at_20"]
    same_delta = deltas["same_artist"]["delta_ndcg_at_20"]
    top1_equiv = churn["h7"]["top1_churn_equiv_per_80"]
    top20_overlap = churn["h7"]["top20_overlap"]["median"]
    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(
        f"{ts()} Verdict: {verdict}  h7_delta={h7_delta:+.5f}  "
        f"same_delta={same_delta:+.5f}  top1_churn={top1_equiv:.1f}/80  "
        f"top20_overlap={top20_overlap:.2f}/20",
        flush=True,
    )
    print(f"Elapsed: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
