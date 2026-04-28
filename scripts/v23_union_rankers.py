# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Ranker-only follow-up to v23_union_analysis: caches retrievals to disk on
first run, then evaluates R0..R4 quickly.

READ-ONLY + COMPUTE-ONLY. No API/LLM calls.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from eval_inference import build_ground_truth, cached_test_arrow_path
from offline_retrieval_sweep import CachedBM25, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from scripts.tune_postrank_v23 import (
    FEATURE_NAMES,
    INIT_WEIGHTS,
    build_row_features,
    reconstruct_context,
    _stable_hash,
)
from mcrs.db_item.music_catalog import MusicCatalogDB

ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
BM25_K = 500
TRACK_NEIGHBORS_K = 200
SEED = 0
CACHE_FILE = "exp/eval/v23_union_retrievals_cache.json"


def get_retrievals(rows, ds_by_case, ds_by_sid):
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached retrievals from {CACHE_FILE}", flush=True)
        with open(CACHE_FILE) as f:
            return json.load(f)

    print("Loading BM25 cached index...", flush=True)
    bm25 = CachedBM25()
    print("Loading track-sim cached vectors...", flush=True)
    track_sim = TrackSimilarityRetriever(cache_dir="./cache")
    print("Loading metadata catalog...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )

    # Build queries per row
    queries_meta = []
    queries_full = []
    cases = []
    metadata: dict[str, dict] = {}

    def get_meta(tid: str) -> dict:
        if tid in metadata:
            return metadata[tid]
        try:
            m = item_db.id_to_full_metadata(tid)
        except KeyError:
            m = {}
        metadata[tid] = m
        return m

    for r in rows:
        sid = str(r["session_id"])
        uid = r.get("user_id")
        turn = int(r["turn_number"])
        item = ds_by_case.get((sid, uid)) or ds_by_sid.get(sid)
        if item is None:
            continue
        rows_sorted = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        history = [c for c in rows_sorted if int(c["turn_number"]) < turn]
        user_query = ""
        for c in rows_sorted:
            if int(c["turn_number"]) == turn and c["role"] == "user":
                user_query = str(c["content"])
                break
        if not user_query:
            user_msgs = [c for c in history if c["role"] == "user"]
            if user_msgs:
                user_query = str(user_msgs[-1]["content"])
        music_turns = [str(c["content"]).strip() for c in history if c["role"] == "music"]
        for tid in music_turns:
            get_meta(tid)
        q_meta = " ".join(query_parts(history, user_query, metadata, "last_music_meta"))
        q_full = " ".join(query_parts(history, user_query, metadata, "full"))
        cases.append({
            "row_idx": len(cases),
            "session_id": sid,
            "user_id": uid,
            "turn_number": turn,
            "v23_pool": list(r["candidate_pool_track_ids"]),
            "music_turns": music_turns,
        })
        queries_meta.append(q_meta if q_meta else user_query)
        queries_full.append(q_full if q_full else user_query)

    print(f"BM25 last_music_meta @ K={BM25_K}", flush=True)
    bm25_meta = bm25.retrieve_batch(queries_meta, topk=BM25_K)
    print(f"BM25 full_history @ K={BM25_K}", flush=True)
    bm25_full = bm25.retrieve_batch(queries_full, topk=BM25_K)
    print(f"track_neighbors @ K={TRACK_NEIGHBORS_K}", flush=True)
    nbrs = []
    for i, c in enumerate(cases):
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        if anchor:
            nbrs.append(track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBORS_K))
        else:
            nbrs.append([])
        if (i + 1) % 50 == 0:
            print(f"  neighbors {i+1}/200", flush=True)

    payload = {
        "cases": cases,
        "bm25_meta": bm25_meta,
        "bm25_full": bm25_full,
        "neighbors": nbrs,
    }
    os.makedirs("exp/eval", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(payload, f)
    print(f"Cached retrievals to {CACHE_FILE}", flush=True)
    return payload


def main():
    print("Loading artifact...", flush=True)
    with open(ARTIFACT) as f:
        rows = json.load(f)
    arrow = cached_test_arrow_path()
    if not arrow:
        sys.exit("ERROR: devset arrow not in HF cache")
    ds = Dataset.from_file(arrow)
    gt_maps = build_ground_truth(ds)
    ds_by_case: dict[tuple[str, str | None], dict] = {}
    ds_by_sid: dict[str, dict] = {}
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        ds_by_case[(sid, uid)] = item
        ds_by_sid.setdefault(sid, item)

    payload = get_retrievals(rows, ds_by_case, ds_by_sid)
    cases = payload["cases"]
    bm25_meta = payload["bm25_meta"]
    bm25_full = payload["bm25_full"]
    neighbors_lists = payload["neighbors"]

    n = len(cases)
    print(f"Cases: {n}", flush=True)

    # Re-derive GTs and v23 pool / pred from artifact rows aligned by index
    v23_pred = {i: list(rows[i]["predicted_track_ids"]) for i in range(n)}
    v23_pool = {i: list(rows[i]["candidate_pool_track_ids"]) for i in range(n)}
    gts: list[str | None] = []
    for i, c in enumerate(cases):
        sid = str(c["session_id"])
        uid = c["user_id"]
        turn = int(c["turn_number"])
        gt_id = None
        if uid is not None:
            gt_id = gt_maps["session_user"].get((sid, str(uid)), {}).get(turn)
        if gt_id is None:
            gt_id = gt_maps["session"].get(sid, {}).get(turn)
        gts.append(gt_id)

    def ndcg(predicted: list[str], gt: str | None, k: int = 20) -> float:
        if gt is None:
            return 0.0
        for i, tid in enumerate(predicted[:k]):
            if tid == gt:
                return 1.0 / math.log2(i + 2)
        return 0.0

    # Splits (mirrors postrank tuner with seed=0)
    row_sessions = [c["session_id"] for c in cases]
    order = sorted(range(n), key=lambda i: _stable_hash(f"{row_sessions[i]}:{SEED}"))
    train_idx = order[:100]
    holdout_idx = order[100:200]

    def cv_folds(k: int = 5):
        folds = [[] for _ in range(k)]
        for pos, idx in enumerate(order):
            folds[pos % k].append(idx)
        return folds
    folds = cv_folds(5)

    def eval_indices(pred_fn, idx_list):
        total = 0.0
        ct = 0
        for i in idx_list:
            total += ndcg(pred_fn(i), gts[i])
            ct += 1
        return total / ct if ct else 0.0

    # ----- R0: v23 predicted_track_ids (= v23 own top-20 ranking) -----
    def r0_pred(i):
        return v23_pred[i][:20]
    r0_holdout = eval_indices(r0_pred, holdout_idx)
    r0_cv = [eval_indices(r0_pred, f) for f in folds]

    # ----- R1: v23 prefix + RRF over B,C,D -----
    def rrf_merge(lists, rrf_k=60, topk=20):
        scores: dict[str, float] = {}
        for ranked in lists:
            for rank, tid in enumerate(ranked):
                scores[tid] = scores.get(tid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

    def r1_pred(i, kprime):
        prefix = v23_pred[i][:kprime]
        fused = rrf_merge([bm25_meta[i], bm25_full[i], neighbors_lists[i]])
        out = list(prefix)
        for tid in fused:
            if len(out) >= 20:
                break
            if tid not in out:
                out.append(tid)
        for tid in v23_pool[i] + v23_pred[i]:
            if len(out) >= 20:
                break
            if tid not in out:
                out.append(tid)
        return out[:20]

    r1_results = {}
    for kp in [10, 15, 20, 30]:
        pred = lambda i, kp=kp: r1_pred(i, kp)
        ho = eval_indices(pred, holdout_idx)
        cv = [eval_indices(pred, f) for f in folds]
        r1_results[f"K_prime_{kp}"] = {
            "holdout_ndcg": ho,
            "cv5_mean": float(np.mean(cv)),
            "cv5_std": float(np.std(cv, ddof=1)),
        }

    # ----- R2: weighted RRF tuned on train -----
    def weighted_rrf(lists_with_w, rrf_k=60, topk=20):
        scores: dict[str, float] = {}
        for ranked, w in lists_with_w:
            if w == 0:
                continue
            for rank, tid in enumerate(ranked):
                scores[tid] = scores.get(tid, 0.0) + w / (rrf_k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

    def r2_pred(i, w):
        wA, wB, wC, wD = w
        return weighted_rrf([
            (v23_pool[i], wA),
            (bm25_meta[i], wB),
            (bm25_full[i], wC),
            (neighbors_lists[i], wD),
        ])

    print("Running R2 weighted RRF grid (4*4*4*4 = 256 combos on 100 train)...", flush=True)
    grid_A = [1, 2, 4, 8]
    grid_B = [0, 0.5, 1, 2]
    grid_C = [0, 0.5, 1, 2]
    grid_D = [0, 0.5, 1, 2]
    best_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    best_train: float = -1.0
    for wA in grid_A:
        for wB in grid_B:
            for wC in grid_C:
                for wD in grid_D:
                    w = (float(wA), float(wB), float(wC), float(wD))
                    s = eval_indices(lambda i, w=w: r2_pred(i, w), train_idx)
                    if s > best_train:
                        best_train, best_w = s, w
    r2_holdout = eval_indices(lambda i, w=best_w: r2_pred(i, w), holdout_idx)
    r2_cv = [eval_indices(lambda i, w=best_w: r2_pred(i, w), f) for f in folds]

    # ----- R4: per-source restricted -----
    def r4a_pred(i):
        return bm25_meta[i][:20]
    def r4b_pred(i):
        return bm25_full[i][:20]
    r4a_ho = eval_indices(r4a_pred, holdout_idx)
    r4b_ho = eval_indices(r4b_pred, holdout_idx)
    r4a_cv = [eval_indices(r4a_pred, f) for f in folds]
    r4b_cv = [eval_indices(r4b_pred, f) for f in folds]

    # ----- R3: linear scorer on union — TRAIN ONCE on TRAIN split, evaluate on holdout/CV
    print("R3: building union pools and union features...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )
    conv_by_sid: dict[str, list[dict]] = {item["session_id"]: item["conversations"] for item in ds}

    union_lists = []
    rank_in_src_per_row: list[dict[str, dict[str, int]]] = []
    for i, c in enumerate(cases):
        order_lst, seen = [], set()
        rs: dict[str, dict[str, int]] = {"A": {}, "B": {}, "C": {}, "D": {}}
        for label, lst in (
            ("A", v23_pool[i]),
            ("B", bm25_meta[i]),
            ("C", bm25_full[i]),
            ("D", neighbors_lists[i]),
        ):
            for r_pos, tid in enumerate(lst):
                if tid not in seen:
                    seen.add(tid)
                    order_lst.append(tid)
                if tid not in rs[label]:
                    rs[label][tid] = r_pos + 1
        union_lists.append(order_lst)
        rank_in_src_per_row.append(rs)

    UNION_FEATURE_NAMES = list(FEATURE_NAMES) + [
        "in_A", "in_B", "in_C", "in_D",
        "recip_rank_A", "recip_rank_B", "recip_rank_C", "recip_rank_D",
    ]
    UNION_INIT = list([INIT_WEIGHTS[n] for n in FEATURE_NAMES]) + [
        0.5, 0.3, 0.3, 0.2, 2.0, 1.0, 1.0, 0.5,
    ]

    feat_matrices: list[np.ndarray] = []
    gt_indices: list[int | None] = []
    print("Building per-row union feature matrices (this is the slow step)...", flush=True)
    for i, c in enumerate(cases):
        cands = union_lists[i]
        ctx = reconstruct_context(conv_by_sid.get(cases[i]["session_id"], []), cases[i]["turn_number"])
        X_base = build_row_features(cands, ctx, item_db)
        K = len(cands)
        X_extra = np.zeros((K, 8), dtype=np.float64)
        rs = rank_in_src_per_row[i]
        for j, tid in enumerate(cands):
            for col, label in enumerate(["A", "B", "C", "D"]):
                rk = rs[label].get(tid)
                if rk is not None:
                    X_extra[j, col] = 1.0
                    X_extra[j, col + 4] = 1.0 / rk
        X = np.hstack([X_base, X_extra])
        feat_matrices.append(X)
        gt_indices.append(cands.index(gts[i]) if gts[i] in cands else None)
        if (i + 1) % 25 == 0:
            print(f"  union features {i+1}/{n}", flush=True)

    def mean_ndcg_idx(idx_list, weights):
        total = 0.0
        ct = 0
        for i in idx_list:
            X = feat_matrices[i]
            gt_idx = gt_indices[i]
            if gt_idx is None:
                ct += 1
                continue
            scores = X @ weights
            order2 = np.argsort(-scores, kind="stable")
            for r, j in enumerate(order2[:20]):
                if j == gt_idx:
                    total += 1.0 / math.log2(r + 2)
                    break
            ct += 1
        return total / ct if ct else 0.0

    from scipy.optimize import minimize
    init_w = np.array(UNION_INIT, dtype=np.float64)
    print("Tuning R3 on train (Powell)...", flush=True)

    def fit(idx_list):
        def neg(w):
            return -mean_ndcg_idx(idx_list, w)
        res = minimize(neg, init_w, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
        return res.x, -res.fun

    w_opt, train_ndcg_r3 = fit(train_idx)
    r3_holdout = mean_ndcg_idx(holdout_idx, w_opt)
    # Use the SAME w_opt for CV folds (faster than per-fold Powell; document this).
    r3_cv = [mean_ndcg_idx(f, w_opt) for f in folds]
    r3_weights = dict(zip(UNION_FEATURE_NAMES, w_opt.tolist()))

    raw_v23 = 0.0892
    tuned_pool = 0.0912

    def line(name, ho, cv):
        return {
            "holdout_ndcg": ho,
            "cv5_mean": float(np.mean(cv)),
            "cv5_std": float(np.std(cv, ddof=1)),
            "delta_vs_v23_raw": ho - raw_v23,
            "delta_vs_tuned_pool": ho - tuned_pool,
        }

    table = {
        "splits": {
            "seed": SEED,
            "train_n": len(train_idx),
            "holdout_n": len(holdout_idx),
            "cv_k": 5,
        },
        "baseline_refs": {
            "v23_raw_ndcg": raw_v23,
            "tuned_pool_cv5_mean": tuned_pool,
        },
        "R0_v23_baseline": line("R0", r0_holdout, r0_cv),
        "R1_v23_prefix_plus_rrf": r1_results,
        "R2_weighted_rrf": {
            "best_weights": list(best_w),
            "train_ndcg": best_train,
            **line("R2", r2_holdout, r2_cv),
        },
        "R3_linear_union": {
            "weights": r3_weights,
            "train_ndcg": train_ndcg_r3,
            **line("R3", r3_holdout, r3_cv),
        },
        "R4a_last_music_meta_top20": line("R4a", r4a_ho, r4a_cv),
        "R4b_full_history_top20": line("R4b", r4b_ho, r4b_cv),
    }

    out = "exp/eval/v23_union_ranker_eval.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nWrote {out}")
    print(json.dumps({k: v for k, v in table.items() if k.startswith("R")}, indent=2))


if __name__ == "__main__":
    main()
