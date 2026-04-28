# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Light-weight ranker eval on cached retrieval union (no R3 union feature
matrices — those re-run pyarrow per candidate which scales badly).

For R3 (learned-linear), we instead run the existing tune_postrank_v23 features
on the SAME 200 rows but with the union top-50 pool (best 50 by RRF) — that
keeps the postrank machinery's cost the same as the existing v23 50-pool.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from eval_inference import build_ground_truth, cached_test_arrow_path
from scripts.tune_postrank_v23 import (
    FEATURE_NAMES,
    INIT_WEIGHTS,
    build_row_features,
    reconstruct_context,
    _stable_hash,
)
from mcrs.db_item.music_catalog import MusicCatalogDB

ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
CACHE_FILE = "exp/eval/v23_union_retrievals_cache.json"
SEED = 0


def main():
    print("Loading artifact + cached retrievals + GT...", flush=True)
    with open(ARTIFACT) as f:
        rows = json.load(f)
    with open(CACHE_FILE) as f:
        payload = json.load(f)
    cases = payload["cases"]
    bm25_meta = payload["bm25_meta"]
    bm25_full = payload["bm25_full"]
    neighbors_lists = payload["neighbors"]
    arrow = cached_test_arrow_path()
    if not arrow:
        sys.exit("ERROR: devset arrow not in HF cache")
    ds = Dataset.from_file(arrow)
    gt_maps = build_ground_truth(ds)

    n = len(cases)

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

    # ---- R0
    def r0_pred(i):
        return v23_pred[i][:20]
    r0_holdout = eval_indices(r0_pred, holdout_idx)
    r0_cv = [eval_indices(r0_pred, f) for f in folds]

    # ---- R1: v23 prefix + RRF over B,C,D
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

    # ---- R2: weighted RRF, tuned on train -> eval holdout/CV
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

    print("R2 grid...", flush=True)
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

    # ---- R4: per-source restricted
    def r4a_pred(i):
        return bm25_meta[i][:20]
    def r4b_pred(i):
        return bm25_full[i][:20]
    r4a_ho = eval_indices(r4a_pred, holdout_idx)
    r4b_ho = eval_indices(r4b_pred, holdout_idx)
    r4a_cv = [eval_indices(r4a_pred, f) for f in folds]
    r4b_cv = [eval_indices(r4b_pred, f) for f in folds]

    # ---- R3 (alt): tune postrank features on SAME 50-deep RRF top from union, on 100/100 split.
    # This re-uses the postrank machinery without exploding feature compute.
    print("R3-alt: tune postrank features on a 50-pool generated by best R2 weights", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )
    conv_by_sid: dict[str, list[dict]] = {item["session_id"]: item["conversations"] for item in ds}

    # Build a 50-deep RRF pool per row using best R2 weights. This is comparable
    # in cost to v23's 50-pool postrank.
    pool50: list[list[str]] = []
    for i in range(n):
        pool50.append(weighted_rrf([
            (v23_pool[i], best_w[0]),
            (bm25_meta[i], best_w[1]),
            (bm25_full[i], best_w[2]),
            (neighbors_lists[i], best_w[3]),
        ], topk=50))

    feat_matrices: list[np.ndarray] = []
    gt_indices: list[int | None] = []
    print("Building per-row features (50-pool)...", flush=True)
    for i, c in enumerate(cases):
        cands = pool50[i]
        ctx = reconstruct_context(conv_by_sid.get(cases[i]["session_id"], []), cases[i]["turn_number"])
        X = build_row_features(cands, ctx, item_db)
        feat_matrices.append(X)
        gt_i = gts[i]
        gt_indices.append(cands.index(gt_i) if gt_i is not None and gt_i in cands else None)

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
    init_w = np.array([INIT_WEIGHTS[n] for n in FEATURE_NAMES], dtype=np.float64)
    print("Tuning R3 (Powell) on train...", flush=True)

    def fit(idx_list):
        def neg(w):
            return -mean_ndcg_idx(idx_list, w)
        res = minimize(neg, init_w, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
        return res.x, -res.fun

    w_opt, train_ndcg_r3 = fit(train_idx)
    r3_holdout = mean_ndcg_idx(holdout_idx, w_opt)
    r3_cv = [mean_ndcg_idx(f, w_opt) for f in folds]
    r3_weights = dict(zip(FEATURE_NAMES, w_opt.tolist()))

    raw_v23 = 0.0892
    tuned_pool = 0.0912

    def line(ho, cv):
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
        "R0_v23_baseline": line(r0_holdout, r0_cv),
        "R1_v23_prefix_plus_rrf": r1_results,
        "R2_weighted_rrf": {
            "best_weights": list(best_w),
            "train_ndcg": best_train,
            **line(r2_holdout, r2_cv),
        },
        "R3_postrank_on_RRF_pool50": {
            "weights": r3_weights,
            "train_ndcg": train_ndcg_r3,
            **line(r3_holdout, r3_cv),
        },
        "R4a_last_music_meta_top20": line(r4a_ho, r4a_cv),
        "R4b_full_history_top20": line(r4b_ho, r4b_cv),
    }

    out = "exp/eval/v23_union_ranker_eval.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nWrote {out}")
    print(json.dumps({k: v for k, v in table.items() if k.startswith("R")}, indent=2))


if __name__ == "__main__":
    main()
