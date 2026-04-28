#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Offline candidate-union analysis for clean v23.

This script is intentionally deterministic and spend-safe: it uses only cached
BM25/track-sim indexes plus the existing v23 pool artifact. It does not import
LLM modules or call inference APIs.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth, ndcg_at_k
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from datasets import Dataset
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever


ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
OUT_RECALL = "exp/eval/v23_union_recall_diagnostic.json"
OUT_RANKERS = "exp/eval/v23_union_ranker_eval.json"
TRACK_SIM_K = 200
BM25_K = 500
TARGET_LOCAL = 0.146
CUSHION = 0.40


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def rrf(lists: list[list[str]], weights: list[float], topk: int = 20, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked, weight in zip(lists, weights):
        if weight == 0:
            continue
        for rank, tid in enumerate(ranked):
            scores[tid] = scores.get(tid, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]


def split_indices(sessions: list[str], seed: int = 0, train_n: int = 100) -> tuple[list[int], list[int]]:
    order = sorted(range(len(sessions)), key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))
    return order[:train_n], order[train_n:]


def cv_folds(sessions: list[str], k: int = 5, seed: int = 0) -> list[list[int]]:
    order = sorted(range(len(sessions)), key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))
    folds = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(idx)
    return folds


def mean_ndcg_for_rows(rows: list[dict[str, Any]], preds: list[list[str]], idxs: list[int]) -> float:
    if not idxs:
        return 0.0
    return float(np.mean([ndcg_at_k(preds[i], rows[i]["gt"]) for i in idxs]))


def hit_count(rows: list[dict[str, Any]], source: str) -> int:
    return sum(1 for r in rows if r["gt"] in r["sources"][source])


def rank1(seq: list[str], gt: str) -> int | None:
    try:
        return seq.index(gt) + 1
    except ValueError:
        return None


def build_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(ARTIFACT, encoding="utf-8") as f:
        artifact_rows = json.load(f)

    ds = Dataset.from_file(cached_test_arrow_path())
    ds_by_key = {(str(x["session_id"]), str(x.get("user_id", ""))): x for x in ds}
    gt_map = build_ground_truth(ds)
    metadata = load_track_metadata()

    if not os.path.exists("cache/bm25/track_name_artist_name_album_name_tag_list"):
        raise FileNotFoundError("Missing cached BM25 index")
    if not os.path.exists("cache/track_sim/metadata-qwen3_embedding_0.6b/vectors.npy"):
        raise FileNotFoundError("Missing cached track-sim vectors")

    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir="./cache")

    bm25_queries: list[str] = []
    query_counts: list[tuple[int, int]] = []
    shell_cases: list[dict[str, Any]] = []
    for r in artifact_rows:
        sid = str(r["session_id"])
        uid = str(r.get("user_id", ""))
        turn = int(r["turn_number"])
        item = ds_by_key[(sid, uid)]
        rows = sorted(item["conversations"], key=lambda x: int(x["turn_number"]))
        history = [x for x in rows if int(x["turn_number"]) < turn]
        user_query = [x["content"] for x in rows if x["role"] == "user" and int(x["turn_number"]) == turn][0]
        gt = lookup_ground_truth(gt_map, sid, uid, turn)
        music_turns = [h["content"].strip() for h in history if h["role"] == "music"]
        last_q = " ".join(query_parts(history, user_query, metadata, "last_music_meta"))
        full_q = " ".join(query_parts(history, user_query, metadata, "full"))
        bm25_queries.extend([last_q, full_q])
        query_counts.append((len(bm25_queries) - 2, len(bm25_queries) - 1))
        shell_cases.append({
            "session_id": sid,
            "user_id": uid,
            "turn_number": turn,
            "gt": gt,
            "artifact_row": r,
            "history": history,
            "music_turns": music_turns,
            "conversations": rows,
        })

    bm25_results = bm25.retrieve_batch(bm25_queries, topk=BM25_K)
    cases: list[dict[str, Any]] = []
    for shell, (last_i, full_i) in zip(shell_cases, query_counts):
        a = dedupe(shell["artifact_row"]["candidate_pool_track_ids"])
        b = dedupe(bm25_results[last_i])
        c = dedupe(bm25_results[full_i])
        last_tid = shell["music_turns"][-1] if shell["music_turns"] else None
        d = dedupe(track_sim.track_id_to_neighbors(last_tid, topk=TRACK_SIM_K)) if last_tid else []
        union = dedupe(a + b + c + d)
        source_ranks = {
            "A_v23_pool50": rank1(a, shell["gt"]),
            "B_last_music_meta_500": rank1(b, shell["gt"]),
            "C_full_history_500": rank1(c, shell["gt"]),
            f"D_track_neighbors_{TRACK_SIM_K}": rank1(d, shell["gt"]),
        }
        cases.append({
            "session_id": shell["session_id"],
            "user_id": shell["user_id"],
            "turn_number": shell["turn_number"],
            "gt": shell["gt"],
            "sources": {"A": a, "B": b, "C": c, "D": d},
            "source_ranks": source_ranks,
            "union": union,
            "conversations": shell["conversations"],
        })

    recall_summary = make_recall_summary(cases)
    return cases, recall_summary


def make_recall_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(cases)
    membership_counts: Counter[str] = Counter()
    for r in cases:
        membership = "".join(k for k in "ABCD" if r["gt"] in r["sources"][k])
        membership_counts[membership or "none"] += 1

    missed = [r for r in cases if r["gt"] not in r["sources"]["A"] and r["gt"] in r["union"]]
    missed_ranks: dict[str, list[int]] = defaultdict(list)
    for r in missed:
        for source, rank in r["source_ranks"].items():
            if rank is not None:
                missed_ranks[source].append(rank)

    per_session = []
    for r in cases:
        per_session.append({
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": r["turn_number"],
            "gt": r["gt"],
            "union_size": len(r["union"]),
            "union_hit": r["gt"] in r["union"],
            "source_ranks": r["source_ranks"],
            "membership": "".join(k for k in "ABCD" if r["gt"] in r["sources"][k]) or "none",
        })

    return {
        "artifact": ARTIFACT,
        "n": n,
        "sources": {
            "A": "v23 candidate_pool_track_ids top50",
            "B": f"last_music_meta_bm25@{BM25_K}",
            "C": f"full_history_bm25@{BM25_K}",
            "D": f"track_neighbors@{TRACK_SIM_K}",
        },
        "average_union_size": float(np.mean([len(r["union"]) for r in cases])),
        "empty_source_sessions": {
            k: sum(1 for r in cases if not r["sources"][k])
            for k in "ABCD"
        },
        "source_hits": {
            "A_v23_pool50": hit_count(cases, "A"),
            f"B_last_music_meta_{BM25_K}": hit_count(cases, "B"),
            f"C_full_history_{BM25_K}": hit_count(cases, "C"),
            f"D_track_neighbors_{TRACK_SIM_K}": hit_count(cases, "D"),
        },
        "source_hit_rates": {
            "A_v23_pool50": hit_count(cases, "A") / n,
            f"B_last_music_meta_{BM25_K}": hit_count(cases, "B") / n,
            f"C_full_history_{BM25_K}": hit_count(cases, "C") / n,
            f"D_track_neighbors_{TRACK_SIM_K}": hit_count(cases, "D") / n,
        },
        "union_hits": sum(1 for r in cases if r["gt"] in r["union"]),
        "union_hit_rate": sum(1 for r in cases if r["gt"] in r["union"]) / n,
        "membership_counts": dict(sorted(membership_counts.items())),
        "thresholds": {
            "target_0.146_possible": (sum(1 for r in cases if r["gt"] in r["union"]) / n) >= TARGET_LOCAL,
            "cushion_0.40_possible": (sum(1 for r in cases if r["gt"] in r["union"]) / n) >= CUSHION,
        },
        "missed_by_A_but_in_union": len(missed),
        "missed_gt_rank_stats": {
            k: {
                "n": len(v),
                "median": float(np.median(v)) if v else None,
                "p75": float(np.quantile(v, 0.75)) if v else None,
            }
            for k, v in missed_ranks.items()
        },
        "per_session": per_session,
    }


def predictions_for_strategy(cases: list[dict[str, Any]], strategy: str, params: Any = None) -> list[list[str]]:
    preds: list[list[str]] = []
    for r in cases:
        src = r["sources"]
        if strategy == "R0_v23":
            pred = src["A"][:20]
        elif strategy == "R4_last_music":
            pred = src["B"][:20]
        elif strategy == "R4_full_history":
            pred = src["C"][:20]
        elif strategy == "R4_track_neighbors":
            pred = src["D"][:20]
        elif strategy == "R1_prefix":
            prefix = src["A"][: params["prefix"]]
            rest = [x for x in rrf([src["B"], src["C"], src["D"]], [1, 1, 1], topk=100) if x not in set(prefix)]
            pred = (prefix + rest)[:20]
        elif strategy == "R2_weighted_rrf":
            pred = rrf([src["A"], src["B"], src["C"], src["D"]], list(params["weights"]), topk=20)
        else:
            raise ValueError(strategy)
        preds.append(pred)
    return preds


def tune_discrete(
    cases: list[dict[str, Any]],
    strategy: str,
    grid: list[Any],
    train_idx: list[int],
) -> tuple[Any, float]:
    best_param = None
    best_score = -1.0
    for param in grid:
        preds = predictions_for_strategy(cases, strategy, param)
        score = mean_ndcg_for_rows(cases, preds, train_idx)
        if score > best_score:
            best_score = score
            best_param = param
    return best_param, best_score


def source_feature_matrix(cases: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[int | None], list[str]]:
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from scripts.tune_postrank_v23 import FEATURE_NAMES, build_row_features, reconstruct_context

    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )
    matrices: list[np.ndarray] = []
    gt_idxs: list[int | None] = []
    for r in cases:
        union = r["union"]
        ctx = reconstruct_context(r["conversations"], r["turn_number"])
        base = build_row_features(union, ctx, item_db)
        extra = np.zeros((len(union), 10), dtype=np.float64)
        for i, tid in enumerate(union):
            for j, key in enumerate("ABCD"):
                source = r["sources"][key]
                rr = 0.0
                if tid in source:
                    rank = source.index(tid) + 1
                    rr = 1.0 / rank
                    extra[i, j] = 1.0
                extra[i, 4 + j] = rr
            extra[i, 8] = 1.0 if i < 50 else 0.0
            extra[i, 9] = 1.0 / (i + 1)
        matrices.append(np.concatenate([base, extra], axis=1))
        gt_idxs.append(union.index(r["gt"]) if r["gt"] in union else None)
    feature_names = FEATURE_NAMES + [
        "src_A", "src_B", "src_C", "src_D",
        "rr_A", "rr_B", "rr_C", "rr_D",
        "union_top50", "union_orig_rank",
    ]
    return matrices, gt_idxs, feature_names


def score_linear(mats: list[np.ndarray], gt_idxs: list[int | None], weights: np.ndarray, idxs: list[int]) -> float:
    vals = []
    for i in idxs:
        gt = gt_idxs[i]
        if gt is None:
            vals.append(0.0)
            continue
        scores = mats[i] @ weights
        order = np.argsort(-scores, kind="stable")[:20]
        if gt in order:
            vals.append(1.0 / math.log2(int(np.where(order == gt)[0][0]) + 2))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def tune_linear(
    mats: list[np.ndarray],
    gt_idxs: list[int | None],
    feature_names: list[str],
    train_idx: list[int],
) -> tuple[np.ndarray, float]:
    init = np.zeros(mats[0].shape[1], dtype=np.float64)
    base_count = len(feature_names) - 10
    init[feature_names.index("orig_rank")] = 2.0
    init[base_count + 0] = 2.0  # A indicator
    init[base_count + 8] = 1.0  # union rank prior

    def objective(w):
        return -score_linear(mats, gt_idxs, w, train_idx)

    res = minimize(objective, init, method="Powell", options={"maxiter": 1200, "xtol": 1e-4, "ftol": 1e-4})
    return res.x, -float(res.fun)


def eval_rankers(cases: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = [r["session_id"] for r in cases]
    train_idx, hold_idx = split_indices(sessions)
    folds = cv_folds(sessions)

    rows: dict[str, Any] = {}

    def add_fixed(name: str, strategy: str, params=None):
        preds = predictions_for_strategy(cases, strategy, params)
        cv_scores = [mean_ndcg_for_rows(cases, preds, fold) for fold in folds]
        rows[name] = {
            "holdout_ndcg": mean_ndcg_for_rows(cases, preds, hold_idx),
            "cv5_mean": float(np.mean(cv_scores)),
            "cv5_std": float(np.std(cv_scores, ddof=1)),
            "params": params,
        }

    add_fixed("R0_v23_only_in_union", "R0_v23")
    add_fixed(f"R4a_last_music_meta_{BM25_K}_alone", "R4_last_music")
    add_fixed(f"R4b_full_history_{BM25_K}_alone", "R4_full_history")
    add_fixed(f"R4c_track_neighbors_{TRACK_SIM_K}_alone", "R4_track_neighbors")

    prefix_grid = [{"prefix": p} for p in [10, 15, 20, 30]]
    best_p, train_score = tune_discrete(cases, "R1_prefix", prefix_grid, train_idx)
    preds = predictions_for_strategy(cases, "R1_prefix", best_p)
    cv_scores = []
    cv_params = []
    for fold in folds:
        tr = [i for i in range(len(cases)) if i not in set(fold)]
        p, _ = tune_discrete(cases, "R1_prefix", prefix_grid, tr)
        cv_params.append(p)
        cv_scores.append(mean_ndcg_for_rows(cases, predictions_for_strategy(cases, "R1_prefix", p), fold))
    rows["R1_v23_prefix_plus_fused"] = {
        "train_ndcg": train_score,
        "holdout_ndcg": mean_ndcg_for_rows(cases, preds, hold_idx),
        "cv5_mean": float(np.mean(cv_scores)),
        "cv5_std": float(np.std(cv_scores, ddof=1)),
        "params": best_p,
        "cv_params": cv_params,
    }

    weight_grid = [
        {"weights": w}
        for w in itertools.product([1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 2.0])
    ]
    best_w, train_score = tune_discrete(cases, "R2_weighted_rrf", weight_grid, train_idx)
    preds = predictions_for_strategy(cases, "R2_weighted_rrf", best_w)
    cv_scores = []
    cv_params = []
    for fold in folds:
        tr = [i for i in range(len(cases)) if i not in set(fold)]
        p, _ = tune_discrete(cases, "R2_weighted_rrf", weight_grid, tr)
        cv_params.append(p)
        cv_scores.append(mean_ndcg_for_rows(cases, predictions_for_strategy(cases, "R2_weighted_rrf", p), fold))
    rows["R2_weighted_rrf"] = {
        "train_ndcg": train_score,
        "holdout_ndcg": mean_ndcg_for_rows(cases, preds, hold_idx),
        "cv5_mean": float(np.mean(cv_scores)),
        "cv5_std": float(np.std(cv_scores, ddof=1)),
        "params": best_w,
        "cv_params": cv_params,
    }

    if os.environ.get("MCRS_SKIP_UNION_R3") == "1":
        for row in rows.values():
            row["delta_vs_v23_raw_0.0892"] = row["holdout_ndcg"] - 0.0892
            row["delta_vs_tuned_pool_0.0912"] = row["holdout_ndcg"] - 0.0912

        return {
            "artifact": ARTIFACT,
            "n": len(cases),
            "split": "100_100_seed0",
            "train_n": len(train_idx),
            "holdout_n": len(hold_idx),
            "skipped": ["R3_linear_union_scorer"],
            "skip_reason": "MCRS_SKIP_UNION_R3=1; cheap rankers are evaluated before expensive Powell optimization.",
            "rankers": rows,
        }

    mats, gt_idxs, feature_names = source_feature_matrix(cases)
    weights, train_score = tune_linear(mats, gt_idxs, feature_names, train_idx)
    cv_scores = []
    for fold in folds:
        tr = [i for i in range(len(cases)) if i not in set(fold)]
        w, _ = tune_linear(mats, gt_idxs, feature_names, tr)
        cv_scores.append(score_linear(mats, gt_idxs, w, fold))
    rows["R3_linear_union_scorer"] = {
        "train_ndcg": train_score,
        "holdout_ndcg": score_linear(mats, gt_idxs, weights, hold_idx),
        "cv5_mean": float(np.mean(cv_scores)),
        "cv5_std": float(np.std(cv_scores, ddof=1)),
        "weights": weights.tolist(),
        "feature_names": feature_names,
    }

    for row in rows.values():
        row["delta_vs_v23_raw_0.0892"] = row["holdout_ndcg"] - 0.0892
        row["delta_vs_tuned_pool_0.0912"] = row["holdout_ndcg"] - 0.0912

    return {
        "artifact": ARTIFACT,
        "n": len(cases),
        "split": "100_100_seed0",
        "train_n": len(train_idx),
        "holdout_n": len(hold_idx),
        "rankers": rows,
    }


def main() -> None:
    os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"
    cases, recall = build_cases()
    rankers = eval_rankers(cases)

    os.makedirs("exp/eval", exist_ok=True)
    with open(OUT_RECALL, "w", encoding="utf-8") as f:
        json.dump(recall, f, indent=2)
    with open(OUT_RANKERS, "w", encoding="utf-8") as f:
        json.dump(rankers, f, indent=2)

    print(json.dumps({
        "union_hit_rate": recall["union_hit_rate"],
        "source_hit_rates": recall["source_hit_rates"],
        "average_union_size": recall["average_union_size"],
        "rankers": {
            k: {
                "holdout_ndcg": v["holdout_ndcg"],
                "cv5_mean": v["cv5_mean"],
                "cv5_std": v["cv5_std"],
            }
            for k, v in rankers["rankers"].items()
        },
        "artifacts": [OUT_RECALL, OUT_RANKERS],
    }, indent=2))


if __name__ == "__main__":
    main()
