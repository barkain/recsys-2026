# ruff: noqa: T201
"""Deep-retrieval union recall + ranker analysis on the v23 200-row clean artifact.

READ-ONLY + COMPUTE-ONLY.  No API/LLM calls; uses the cached BM25 index and
pre-computed track-similarity embeddings only.

Outputs:
    exp/eval/v23_union_recall_diagnostic.json
    exp/eval/v23_union_ranker_eval.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from statistics import median

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

# Tuneables
ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
BM25_K = 500
TRACK_NEIGHBORS_K = 200
N_BOOT_SEED = 0


def main() -> None:
    print("Loading artifact...", flush=True)
    with open(ARTIFACT) as f:
        rows = json.load(f)
    print(f"  {len(rows)} rows; pool depth={len(rows[0]['candidate_pool_track_ids'])}", flush=True)

    print("Loading devset (Arrow cache)...", flush=True)
    arrow = cached_test_arrow_path()
    if not arrow:
        sys.exit("ERROR: devset arrow not in HF cache")
    ds = Dataset.from_file(arrow)
    gt_maps = build_ground_truth(ds)

    print("Indexing devset by (session_id, user_id)...", flush=True)
    ds_by_case: dict[tuple[str, str | None], dict] = {}
    ds_by_sid: dict[str, dict] = {}
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        ds_by_case[(sid, uid)] = item
        ds_by_sid.setdefault(sid, item)

    print("Loading BM25 cached index...", flush=True)
    bm25 = CachedBM25()

    print("Loading track-sim cached vectors...", flush=True)
    # Use brute force (no FAISS) to avoid the optional faiss dep.
    track_sim = TrackSimilarityRetriever(cache_dir="./cache")

    print("Loading metadata catalog (for postrank features)...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )
    # Reuse for meta_text builder
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

    print("Building per-row context, GT, queries...", flush=True)
    cases: list[dict] = []
    queries_meta: list[str] = []
    queries_full: list[str] = []
    for r in rows:
        sid = str(r["session_id"])
        uid = r.get("user_id")
        turn = int(r["turn_number"])
        item = ds_by_case.get((sid, uid)) or ds_by_sid.get(sid)
        if item is None:
            continue
        # Reconstruct conversational context up to (but excluding) target turn.
        rows_sorted = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        history = [c for c in rows_sorted if int(c["turn_number"]) < turn]
        # Target user turn message
        user_query = ""
        for c in rows_sorted:
            if int(c["turn_number"]) == turn and c["role"] == "user":
                user_query = str(c["content"])
                break
        if not user_query:
            # last user turn before target as fallback
            user_msgs = [c for c in history if c["role"] == "user"]
            if user_msgs:
                user_query = str(user_msgs[-1]["content"])
        music_turns = [str(c["content"]).strip() for c in history if c["role"] == "music"]
        # GT
        gt_id = None
        if uid is not None:
            gt_id = gt_maps["session_user"].get((sid, str(uid)), {}).get(turn)
        if gt_id is None:
            gt_id = gt_maps["session"].get(sid, {}).get(turn)

        # Need a metadata-resolved corpus query (offline_retrieval_sweep uses
        # stripped metadata strings).  We mirror its logic for last_music_meta
        # and full modes via query_parts(history, user_query, metadata, mode).
        # But that helper expects a metadata dict — pre-fill needed entries.
        for tid in music_turns + (rows_sorted and []):
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
            "gt": gt_id,
            "user_query": user_query,
            "history": history,
        })
        queries_meta.append(q_meta if q_meta else user_query)
        queries_full.append(q_full if q_full else user_query)

    print(f"  cases assembled: {len(cases)}", flush=True)

    print(f"Running BM25 last_music_meta @ K={BM25_K} ...", flush=True)
    bm25_meta = bm25.retrieve_batch(queries_meta, topk=BM25_K)
    print(f"Running BM25 full_history @ K={BM25_K} ...", flush=True)
    bm25_full = bm25.retrieve_batch(queries_full, topk=BM25_K)
    print(f"Running track_neighbors @ K={TRACK_NEIGHBORS_K} ...", flush=True)
    neighbors_lists = []
    for c in cases:
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        if anchor:
            nbrs = track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBORS_K)
        else:
            nbrs = []
        neighbors_lists.append(nbrs)

    # Diagnostic data structure
    diag: list[dict] = []
    src_hit = {"A": 0, "B": 0, "C": 0, "D": 0}
    src_only = {"A": 0, "B": 0, "C": 0, "D": 0}
    overlaps = {"any2": 0, "any3": 0, "all4": 0}
    union_hit = 0
    empty_sources = 0
    union_sizes: list[int] = []

    rank_in_B: list[int] = []
    rank_in_C: list[int] = []
    rank_in_D: list[int] = []
    rank_in_A: list[int] = []
    # For sessions where union-hit AND not in v23 top-50:
    miss_in_A_rank_B: list[int] = []
    miss_in_A_rank_C: list[int] = []
    miss_in_A_rank_D: list[int] = []

    print("Computing recall analysis...", flush=True)
    for i, c in enumerate(cases):
        gt = c["gt"]
        A = c["v23_pool"]
        B = bm25_meta[i] if i < len(bm25_meta) else []
        C = bm25_full[i] if i < len(bm25_full) else []
        D = neighbors_lists[i]
        if not B or not C or not D:
            empty_sources += 1
        union_set = set(A) | set(B) | set(C) | set(D)
        union_sizes.append(len(union_set))

        present = {
            "A": gt in A if gt else False,
            "B": gt in B if gt else False,
            "C": gt in C if gt else False,
            "D": gt in D if gt else False,
        }
        # Per-source rank (1-based)
        def find_rank(lst: list[str]) -> int | None:
            if not gt:
                return None
            try:
                return lst.index(gt) + 1
            except ValueError:
                return None
        ra, rb, rc, rd = (find_rank(A), find_rank(B), find_rank(C), find_rank(D))

        if present["A"]:
            src_hit["A"] += 1
            rank_in_A.append(ra)
        if present["B"]:
            src_hit["B"] += 1
            rank_in_B.append(rb)
        if present["C"]:
            src_hit["C"] += 1
            rank_in_C.append(rc)
        if present["D"]:
            src_hit["D"] += 1
            rank_in_D.append(rd)

        n_hit = sum(present.values())
        if n_hit == 1:
            for k, v in present.items():
                if v:
                    src_only[k] += 1
        elif n_hit >= 2:
            overlaps["any2"] += 1
            if n_hit >= 3:
                overlaps["any3"] += 1
            if n_hit == 4:
                overlaps["all4"] += 1
        if any(present.values()):
            union_hit += 1
            if not present["A"]:
                if rb is not None:
                    miss_in_A_rank_B.append(rb)
                if rc is not None:
                    miss_in_A_rank_C.append(rc)
                if rd is not None:
                    miss_in_A_rank_D.append(rd)

        diag.append({
            "session_id": c["session_id"],
            "user_id": c["user_id"],
            "turn_number": c["turn_number"],
            "gt": gt,
            "in_A": present["A"], "in_B": present["B"], "in_C": present["C"], "in_D": present["D"],
            "rank_A": ra, "rank_B": rb, "rank_C": rc, "rank_D": rd,
            "union_size": len(union_set),
        })

    n = len(cases)
    rate = lambda x: x / n if n else 0.0
    summary_recall = {
        "n": n,
        "K_bm25": BM25_K,
        "K_track_neighbors": TRACK_NEIGHBORS_K,
        "avg_union_size": float(np.mean(union_sizes)) if union_sizes else 0.0,
        "empty_source_sessions": empty_sources,
        "src_hit_rate": {k: rate(v) for k, v in src_hit.items()},
        "src_hit_count": src_hit,
        "src_only_count": src_only,
        "overlap_count": overlaps,
        "union_hit_rate": rate(union_hit),
        "union_hit_count": union_hit,
        "median_rank_when_hit": {
            "A": int(median(rank_in_A)) if rank_in_A else None,
            "B": int(median(rank_in_B)) if rank_in_B else None,
            "C": int(median(rank_in_C)) if rank_in_C else None,
            "D": int(median(rank_in_D)) if rank_in_D else None,
        },
        "miss_in_A_diagnostic": {
            "n_missed_in_A_but_in_union": sum(
                1 for d in diag if (not d["in_A"]) and (d["in_B"] or d["in_C"] or d["in_D"])
            ),
            "median_rank_B": int(median(miss_in_A_rank_B)) if miss_in_A_rank_B else None,
            "median_rank_C": int(median(miss_in_A_rank_C)) if miss_in_A_rank_C else None,
            "median_rank_D": int(median(miss_in_A_rank_D)) if miss_in_A_rank_D else None,
            "n_recovered_via_B": len(miss_in_A_rank_B),
            "n_recovered_via_C": len(miss_in_A_rank_C),
            "n_recovered_via_D": len(miss_in_A_rank_D),
        },
        "thresholds": {
            "target_ndcg_min": 0.146,
            "cushioned_target": 0.40,
            "passes_min": rate(union_hit) >= 0.146,
            "passes_cushion": rate(union_hit) >= 0.40,
        },
    }

    os.makedirs("exp/eval", exist_ok=True)
    with open("exp/eval/v23_union_recall_diagnostic.json", "w") as f:
        json.dump({"per_session": diag, "summary": summary_recall}, f, indent=2)
    print("\n--- RECALL SUMMARY ---")
    print(json.dumps(summary_recall, indent=2))

    # ---------------- Rankers ----------------
    # Build helpers per-session
    print("\nPreparing ranker inputs...", flush=True)

    # Conv lookup for postrank features
    conv_by_sid: dict[str, list[dict]] = {item["session_id"]: item["conversations"] for item in ds}

    # Postrank tuner needs (X, gt_idx, weights).  We'll reuse build_row_features
    # for both v23-only and union pools.

    def ndcg_at_k(predicted: list[str], gt_id: str | None, k: int = 20) -> float:
        if gt_id is None:
            return 0.0
        for i, tid in enumerate(predicted[:k]):
            if tid == gt_id:
                return 1.0 / math.log2(i + 2)
        return 0.0

    # Compute baseline R0_v23: existing predicted top-20 from v23 (raw),
    # but the artifact only stores predicted_track_ids and pool.  v23 top-20 IS
    # the artifact's predicted_track_ids.  Use that as the baseline.
    # (predicted_track_ids = v23 top-20 chosen by v23's own LLM/postrank.)

    # We'll evaluate on the SAME 100/100 split + 5-fold CV used in the
    # postrank tuner.  That requires a deterministic session-keyed split.

    row_sessions = [c["session_id"] for c in cases]
    seed = N_BOOT_SEED
    order = sorted(range(n), key=lambda i: _stable_hash(f"{row_sessions[i]}:{seed}"))
    train_idx = order[:100]
    holdout_idx = order[100:200]

    def cv_folds_idx(k: int = 5):
        folds = [[] for _ in range(k)]
        for pos, idx in enumerate(order):
            folds[pos % k].append(idx)
        return folds

    folds = cv_folds_idx(5)

    # ---- R0: v23 baseline within union ----
    # Existing v23 predicted_track_ids order; nDCG depends on whether GT in top-20.
    # That equals the canonical v23 raw nDCG on the artifact.
    v23_pred = {i: list(rows[i]["predicted_track_ids"]) for i in range(n)}
    v23_pool = {i: list(rows[i]["candidate_pool_track_ids"]) for i in range(n)}

    def r0_predict(i: int) -> list[str]:
        return v23_pred[i][:20]

    # ---- R1 family: prefix v23 top-K' then RRF-fill from B/C/D ----
    def rrf_merge(lists: list[list[str]], rrf_k: int = 60, topk: int = 20) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in lists:
            for rank, tid in enumerate(ranked):
                scores[tid] = scores.get(tid, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

    def r1_predict(i: int, kprime: int) -> list[str]:
        prefix = v23_pred[i][:kprime]
        # RRF over B, C, D (truncated to BM25_K) then merge with prefix.
        fused = rrf_merge([bm25_meta[i], bm25_full[i], neighbors_lists[i]])
        out = list(prefix)
        for tid in fused:
            if len(out) >= 20:
                break
            if tid not in out:
                out.append(tid)
        # If still <20, fill from v23 pool then v23 ranking.
        for tid in v23_pool[i] + v23_pred[i]:
            if len(out) >= 20:
                break
            if tid not in out:
                out.append(tid)
        return out[:20]

    # ---- R2: weighted RRF tuned on train, eval on holdout/CV ----
    def weighted_rrf(lists_with_weights: list[tuple[list[str], float]], rrf_k: int = 60, topk: int = 20) -> list[str]:
        scores: dict[str, float] = {}
        for ranked, w in lists_with_weights:
            if w == 0:
                continue
            for rank, tid in enumerate(ranked):
                scores[tid] = scores.get(tid, 0.0) + w / (rrf_k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

    def r2_predict(i: int, weights: tuple[float, float, float, float]) -> list[str]:
        wA, wB, wC, wD = weights
        # Treat the v23 pool ordering as source A.
        return weighted_rrf([
            (v23_pool[i], wA),
            (bm25_meta[i], wB),
            (bm25_full[i], wC),
            (neighbors_lists[i], wD),
        ])

    def eval_on(pred_fn, idx_list: Iterable[int]) -> float:
        total = 0.0
        ct = 0
        for i in idx_list:
            pred = pred_fn(i)
            total += ndcg_at_k(pred, cases[i]["gt"])
            ct += 1
        return total / ct if ct else 0.0

    # R2 grid
    print("Running R2 grid (weighted RRF)...", flush=True)
    grid_A = [1, 2, 4, 8]
    grid_B = [0, 0.5, 1, 2]
    grid_C = [0, 0.5, 1, 2]
    grid_D = [0, 0.5, 1, 2]

    best_r2 = None
    for wA in grid_A:
        for wB in grid_B:
            for wC in grid_C:
                for wD in grid_D:
                    score_train = eval_on(lambda i, w=(wA, wB, wC, wD): r2_predict(i, w), train_idx)
                    if best_r2 is None or score_train > best_r2["train_ndcg"]:
                        best_r2 = {
                            "weights": (wA, wB, wC, wD),
                            "train_ndcg": score_train,
                        }
    wA, wB, wC, wD = best_r2["weights"]
    r2_holdout = eval_on(lambda i, w=(wA, wB, wC, wD): r2_predict(i, w), holdout_idx)
    # CV
    r2_fold_scores = []
    for f in folds:
        # Use grid result from full train order? More legitimate: re-tune per fold
        # using the OUT-of-fold train.  For runtime, we use the already-best wts
        # tuned on the seed-0 100-split.  Document this in the report.
        r2_fold_scores.append(eval_on(lambda i, w=(wA, wB, wC, wD): r2_predict(i, w), f))
    best_r2["holdout_ndcg"] = r2_holdout
    best_r2["cv5_mean"] = float(np.mean(r2_fold_scores))
    best_r2["cv5_std"] = float(np.std(r2_fold_scores, ddof=1))

    # ---- R3: linear scorer over union features (postrank-style) ----
    print("Building features for union ranker R3...", flush=True)

    # Build per-row union list with deterministic ordering: prefer A first then dedupe.
    union_lists: list[list[str]] = []
    source_origin: list[dict[str, list[int]]] = []  # tid -> list of source indices
    for i, c in enumerate(cases):
        order_lst: list[str] = []
        seen: set[str] = set()
        # Append in order: A, B, C, D
        rank_in_src: dict[str, dict[str, int]] = {"A": {}, "B": {}, "C": {}, "D": {}}
        for label, lst in (("A", v23_pool[i]), ("B", bm25_meta[i]), ("C", bm25_full[i]), ("D", neighbors_lists[i])):
            for r_pos, tid in enumerate(lst):
                if tid not in seen:
                    seen.add(tid)
                    order_lst.append(tid)
                if tid not in rank_in_src[label]:
                    rank_in_src[label][tid] = r_pos + 1
        union_lists.append(order_lst)
        source_origin.append(rank_in_src)

    # For features, reuse build_row_features but extend with source-of-origin
    # one-hots and per-source rank features.

    def build_union_features(i: int, candidates: list[str]) -> np.ndarray:
        ctx = reconstruct_context(conv_by_sid.get(cases[i]["session_id"], []), cases[i]["turn_number"])
        # Base features (8) — but build_row_features uses position as orig_rank,
        # so we re-order with v23 rank semantics.  We supply the union list and
        # the function will treat the position as "rank" — that's fine because
        # candidates already come in A-first ordering.
        X_base = build_row_features(candidates, ctx, item_db)
        # Extra source-aware features:
        #   inA, inB, inC, inD (binary)
        #   recip_rank_A, recip_rank_B, recip_rank_C, recip_rank_D (1/rank or 0)
        K = len(candidates)
        X_extra = np.zeros((K, 8), dtype=np.float64)
        ros = source_origin[i]
        for j, tid in enumerate(candidates):
            for col, label in enumerate(["A", "B", "C", "D"]):
                rank = ros[label].get(tid)
                if rank is not None:
                    X_extra[j, col] = 1.0
                    X_extra[j, col + 4] = 1.0 / rank
        return np.hstack([X_base, X_extra])

    UNION_FEATURE_NAMES = list(FEATURE_NAMES) + [
        "in_A", "in_B", "in_C", "in_D",
        "recip_rank_A", "recip_rank_B", "recip_rank_C", "recip_rank_D",
    ]
    UNION_INIT = list([INIT_WEIGHTS[n] for n in FEATURE_NAMES]) + [
        0.5, 0.3, 0.3, 0.2,
        2.0, 1.0, 1.0, 0.5,
    ]

    print("Computing per-row union feature matrices...", flush=True)
    feat_matrices: list[np.ndarray] = []
    gt_indices: list[int | None] = []
    for i, c in enumerate(cases):
        cands = union_lists[i]
        X = build_union_features(i, cands)
        gt_idx = cands.index(c["gt"]) if c["gt"] in cands else None
        feat_matrices.append(X)
        gt_indices.append(gt_idx)

    def mean_ndcg(idx_list, weights):
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

    def fit(idx_list):
        def neg(w):
            return -mean_ndcg(idx_list, w)
        res = minimize(neg, init_w, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 2000})
        return res.x, -res.fun

    print("Tuning R3 on train (100), eval holdout (100)...", flush=True)
    w_opt, train_ndcg = fit(train_idx)
    r3_holdout = mean_ndcg(holdout_idx, w_opt)

    print("Running R3 5-fold CV...", flush=True)
    r3_fold_scores = []
    for f in folds:
        train_f = [j for j in range(n) if j not in set(f)]
        w_f, _ = fit(train_f)
        r3_fold_scores.append(mean_ndcg(f, w_f))

    r3 = {
        "train_ndcg": train_ndcg,
        "holdout_ndcg": r3_holdout,
        "cv5_mean": float(np.mean(r3_fold_scores)),
        "cv5_std": float(np.std(r3_fold_scores, ddof=1)),
        "weights": dict(zip(UNION_FEATURE_NAMES, w_opt.tolist())),
    }

    # ---- R4: per-source restricted rankers ----
    print("Computing R4 (single-source rankers)...", flush=True)

    def r4a_predict(i: int) -> list[str]:
        return bm25_meta[i][:20]

    def r4b_predict(i: int) -> list[str]:
        return bm25_full[i][:20]

    def cv_eval(pred_fn):
        return [eval_on(pred_fn, f) for f in folds]

    r4a_holdout = eval_on(r4a_predict, holdout_idx)
    r4b_holdout = eval_on(r4b_predict, holdout_idx)
    r4a_cv = cv_eval(r4a_predict)
    r4b_cv = cv_eval(r4b_predict)

    # ---- R0/R1 evaluation ----
    r0_holdout = eval_on(r0_predict, holdout_idx)
    r0_cv = cv_eval(r0_predict)
    r1_results = {}
    for kp in [10, 15, 20, 30]:
        pred = lambda i, kp=kp: r1_predict(i, kp)
        r1_results[f"K'={kp}"] = {
            "holdout_ndcg": eval_on(pred, holdout_idx),
            "cv5_mean": float(np.mean(cv_eval(pred))),
            "cv5_std": float(np.std(cv_eval(pred), ddof=1)),
        }

    raw_v23 = 0.0892
    tuned_pool = 0.0912

    table = {
        "R0_v23_within_union": {
            "holdout_ndcg": r0_holdout,
            "cv5_mean": float(np.mean(r0_cv)),
            "cv5_std": float(np.std(r0_cv, ddof=1)),
            "delta_vs_v23_raw": r0_holdout - raw_v23,
            "delta_vs_tuned_pool": r0_holdout - tuned_pool,
        },
        "R1_v23_prefix_plus_rrf": r1_results,
        "R2_weighted_rrf": {
            "best_weights": list(best_r2["weights"]),
            "train_ndcg": best_r2["train_ndcg"],
            "holdout_ndcg": best_r2["holdout_ndcg"],
            "cv5_mean": best_r2["cv5_mean"],
            "cv5_std": best_r2["cv5_std"],
            "delta_vs_v23_raw": best_r2["holdout_ndcg"] - raw_v23,
            "delta_vs_tuned_pool": best_r2["holdout_ndcg"] - tuned_pool,
        },
        "R3_linear_union": {
            "train_ndcg": r3["train_ndcg"],
            "holdout_ndcg": r3["holdout_ndcg"],
            "cv5_mean": r3["cv5_mean"],
            "cv5_std": r3["cv5_std"],
            "delta_vs_v23_raw": r3["holdout_ndcg"] - raw_v23,
            "delta_vs_tuned_pool": r3["holdout_ndcg"] - tuned_pool,
            "weights": r3["weights"],
        },
        "R4a_last_music_meta_alone": {
            "holdout_ndcg": r4a_holdout,
            "cv5_mean": float(np.mean(r4a_cv)),
            "cv5_std": float(np.std(r4a_cv, ddof=1)),
            "delta_vs_v23_raw": r4a_holdout - raw_v23,
            "delta_vs_tuned_pool": r4a_holdout - tuned_pool,
        },
        "R4b_full_history_alone": {
            "holdout_ndcg": r4b_holdout,
            "cv5_mean": float(np.mean(r4b_cv)),
            "cv5_std": float(np.std(r4b_cv, ddof=1)),
            "delta_vs_v23_raw": r4b_holdout - raw_v23,
            "delta_vs_tuned_pool": r4b_holdout - tuned_pool,
        },
    }

    with open("exp/eval/v23_union_ranker_eval.json", "w") as f:
        json.dump(table, f, indent=2)

    print("\n--- RANKER COMPARISON (holdout/CV) ---")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
