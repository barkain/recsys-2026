#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R12: History-depth-specific ranking policies.

Different Powell weights and source configs per history-depth bucket.
Targets the B1-exposed degradation: hist_1 best, hist_4+ worst.

Uses all_turns (8000 cases) with batched retrieval for efficiency.
No API. No blind.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from scripts.expR5_sequential_retrieval import SessionTransitionGraph
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from datasets import Dataset, DownloadConfig, load_dataset
from scripts.r3_confirm_400_deterministic import cv_folds

POOL_K = 50
RRF_K = 20
CACHE_PATH = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"


def ndcg_at_k(predicted, gt_id, k=20):
    for i, tid in enumerate(predicted[:k]):
        if tid == gt_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def vec_ndcg(X, gt_idx, sizes, weights, idx):
    pool_axis = np.arange(X.shape[1])[None, :]
    valid_pool = pool_axis < sizes[idx, None]
    scores = X[idx] @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    gt = gt_idx[idx]
    has_gt = gt >= 0
    safe_gt = np.where(has_gt, gt, 0)
    gt_scores = scores[np.arange(len(idx)), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_powell(X, gt_idx, sizes, train_idx):
    init = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def load_all_turn_cases():
    path = cached_test_arrow_path()
    ds = Dataset.from_file(path)
    gt_map = build_ground_truth(ds)
    cases = []
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        for ut in [c for c in convs if c["role"] == "user"]:
            turn = int(ut["turn_number"])
            query = str(ut["content"])
            history = [c for c in convs if int(c["turn_number"]) < turn]
            music = [str(c["content"]).strip() for c in history if c["role"] == "music"]
            gt = lookup_ground_truth(gt_map, sid, uid, turn)
            if not gt:
                continue
            cases.append({
                "session_id": sid, "user_id": uid, "turn_number": turn,
                "user_query": query, "history": history, "music_turns": music,
                "gt": str(gt), "n_prior_music": len(music),
            })
    return cases


def build_payload(cases):
    """Build all source lists for all cases with batched operations."""
    t0 = time.time()
    n = len(cases)
    metadata = load_track_metadata()

    # --- BM25 batch ---
    print("  BM25 batch retrieval...", flush=True)
    bm25 = CachedBM25()
    queries_b = []
    queries_c = []
    for c in cases:
        q_b = " ".join(query_parts(c["history"], c["user_query"], metadata, "last_music_meta"))
        q_c = " ".join(query_parts(c["history"], c["user_query"], metadata, "full"))
        queries_b.append(q_b or c["user_query"])
        queries_c.append(q_c or c["user_query"])
    src_b = bm25.retrieve_batch(queries_b, topk=500)
    src_c = bm25.retrieve_batch(queries_c, topk=500)
    print(f"    BM25 done ({time.time()-t0:.1f}s)", flush=True)

    # --- A' batched via deduplication ---
    print("  A' (qwen3 max_recent_5) with anchor dedup...", flush=True)
    t1 = time.time()
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    vectors = qwen_sim.vectors  # (N, D)
    id_to_idx = qwen_sim._id_to_idx
    track_ids_qwen = qwen_sim.track_ids

    # Collect unique anchor sets and compute once
    src_a = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if not a_idxs:
            src_a.append([])
            continue
        anchor_vecs = vectors[a_idxs]
        sims = vectors @ anchor_vecs.T
        scores = sims.max(axis=1)
        exclude = {id_to_idx[t] for t in played if t in id_to_idx}
        cap = min(len(scores), 200 + len(exclude))
        cand = np.argpartition(-scores, cap - 1)[:cap]
        cand = cand[np.argsort(-scores[cand])]
        out = [track_ids_qwen[int(ii)] for ii in cand if int(ii) not in exclude][:200]
        src_a.append(out)
        if (len(src_a)) % 500 == 0:
            print(f"    A' {len(src_a)}/{n} ({time.time()-t1:.1f}s)", flush=True)
    print(f"    A' done ({time.time()-t1:.1f}s)", flush=True)

    # --- D: track neighbors ---
    print("  D track neighbors...", flush=True)
    src_d = []
    for c in cases:
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        src_d.append(qwen_sim.track_id_to_neighbors(anchor, topk=200) if anchor else [])

    # --- F: CF-BPR ---
    print("  F CF-BPR...", flush=True)
    cf_ids, cf_vecs, cf_idx = build_cfbpr_index()
    src_f = []
    for c in cases:
        played = c["music_turns"]
        src_f.append(cfbpr_max_recent(played, cf_vecs, cf_idx, cf_ids, 5, 200) if played else [])

    # --- G: session cooccur ---
    print("  G session cooccur...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    seqs = {}
    for item in ds["train"]:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        seqs[sid] = [str(c["content"]).strip() for c in convs if c["role"] == "music"]
    graph = SessionTransitionGraph(seqs, metadata)
    src_g = []
    for c in cases:
        played = c["music_turns"]
        src_g.append(graph.g_session_cooccur(played, 200) if played else [])

    # --- Metadata maps ---
    print("  Building metadata maps...", flush=True)
    track_artist = {}
    track_tags = {}
    track_title_toks = {}
    track_artist_toks = {}
    track_meta_toks = {}
    all_tids = set()
    for i in range(n):
        all_tids.update(src_b[i][:100])
        all_tids.update(src_c[i][:100])
        all_tids.update(src_a[i][:50])
        all_tids.update(src_d[i][:50])
        all_tids.update(src_f[i][:50])
        all_tids.update(src_g[i][:50])
        all_tids.update(cases[i]["music_turns"])
    for tid in all_tids:
        if tid in track_artist:
            continue
        meta = metadata.get(str(tid), {})
        artist = str(meta.get("artist_name", "")).lower().strip()
        raw_tags = meta.get("tag_list") or []
        tags = {str(t).lower().strip() for t in raw_tags if str(t).strip()} if isinstance(raw_tags, list) else set()
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        meta_parts = [title, str(meta.get("artist_name", "")), album]
        if isinstance(raw_tags, list):
            meta_parts.extend(str(t) for t in raw_tags[:12])
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = tokens(title)
        track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
        track_meta_toks[tid] = tokens(" ".join(meta_parts))

    elapsed = time.time() - t0
    print(f"  Payload built in {elapsed:.1f}s", flush=True)

    return {
        "cases": cases, "src_a": src_a, "src_b": src_b, "src_c": src_c,
        "src_d": src_d, "src_f": src_f, "src_g": src_g,
        "track_artist": track_artist, "track_tags": track_tags,
        "track_title_toks": track_title_toks,
        "track_artist_toks": track_artist_toks,
        "track_meta_toks": track_meta_toks,
    }


def build_features(payload, source_weights, pool_k=POOL_K):
    """Build 8-feature matrices for all cases using given source weights."""
    cases = payload["cases"]
    n = len(cases)
    X = np.zeros((n, pool_k, len(FEATURE_NAMES)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools = []

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        sources = {}
        if source_weights.get("A", 0) > 0: sources["A"] = payload["src_a"][i]
        if source_weights.get("B", 0) > 0: sources["B"] = payload["src_b"][i]
        if source_weights.get("C", 0) > 0: sources["C"] = payload["src_c"][i]
        if source_weights.get("D", 0) > 0: sources["D"] = payload["src_d"][i]
        if source_weights.get("F", 0) > 0: sources["F"] = payload["src_f"][i]
        if source_weights.get("G", 0) > 0: sources["G"] = payload["src_g"][i]

        pool = weighted_rrf(sources, source_weights, topk=pool_k, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j,t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:pool_k], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank-1]
            row[0] = 1.0/rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

    return X, gt_idx, sizes, pools


def assign_bucket(n_prior):
    if n_prior == 0: return "hist_0"
    if n_prior == 1: return "hist_1"
    if n_prior <= 3: return "hist_2_3"
    return "hist_4plus"


def main():
    t0 = time.time()

    # Load or build payload
    if CACHE_PATH.exists():
        print(f"Loading cached payload from {CACHE_PATH}...", flush=True)
        with open(CACHE_PATH, "rb") as f:
            payload = pickle.load(f)
        cases = payload["cases"]
    else:
        print("Loading all-turn cases...", flush=True)
        cases = load_all_turn_cases()
        print(f"  {len(cases)} cases from {len(set(c['session_id'] for c in cases))} sessions")
        print("Building payload...", flush=True)
        payload = build_payload(cases)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached to {CACHE_PATH}")

    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    buckets = [assign_bucket(c["n_prior_music"]) for c in cases]

    # Bucket indices
    bucket_indices = defaultdict(list)
    for i, b in enumerate(buckets):
        bucket_indices[b].append(i)
    print(f"\nBucket sizes: {dict((k, len(v)) for k, v in sorted(bucket_indices.items()))}")

    # Source configs to test
    source_configs = {
        "ABCD": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
        "ABCDF": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
        "ABCDFG": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": 0.5},
    }

    seeds = [0, 1, 2, 3, 4]

    # =====================================================================
    # PHASE 1: Global baseline (all cases, single Powell)
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Global baselines")
    print(f"{'='*70}")

    global_results = {}
    for src_name, src_weights in source_configs.items():
        print(f"\n  Building features for {src_name}...", flush=True)
        X, gt_idx, sizes, _ = build_features(payload, src_weights)

        # Global CV5
        cv5_seeds = []
        for seed in seeds:
            folds = cv_folds(sessions, seed)
            fold_sc = []
            for fold in folds:
                held = set(fold.tolist())
                train = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
                w, _ = fit_powell(X, gt_idx, sizes, train)
                fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
            cv5_seeds.append(float(np.mean(fold_sc)))
        cv5 = float(np.mean(cv5_seeds))
        hit50 = float(np.mean(gt_idx >= 0))

        # Per-bucket nDCG (using global weights fitted on full data)
        all_idx = np.arange(n, dtype=np.int64)
        w_global, _ = fit_powell(X, gt_idx, sizes, all_idx)
        bucket_ndcg = {}
        for bk, bidxs in bucket_indices.items():
            bi = np.array(bidxs, dtype=np.int64)
            bucket_ndcg[bk] = vec_ndcg(X, gt_idx, sizes, w_global, bi)

        global_results[src_name] = {
            "cv5": cv5, "cv5_seeds": cv5_seeds, "hit50": hit50,
            "bucket_ndcg": bucket_ndcg,
            "X": X, "gt_idx": gt_idx, "sizes": sizes,
        }
        bk_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(bucket_ndcg.items()))
        print(f"    CV5={cv5:.4f}  hit@50={hit50:.4f}  buckets: {bk_str}")

    # =====================================================================
    # PHASE 2: Per-bucket Powell (separate weights per bucket)
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Per-bucket Powell")
    print(f"{'='*70}")

    # Use best global source config for per-bucket experiment
    best_global_src = max(global_results, key=lambda k: global_results[k]["cv5"])
    X = global_results[best_global_src]["X"]
    gt_idx = global_results[best_global_src]["gt_idx"]
    sizes = global_results[best_global_src]["sizes"]
    print(f"  Using source config: {best_global_src}")

    # Per-bucket CV5: train Powell separately per bucket
    # Use session-grouped folds, then apply bucket-specific weights
    bucket_cv5_seeds = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        # For each fold, fit per-bucket weights on train, evaluate on held-out
        fold_ndcgs = []
        for fold in folds:
            held = set(fold.tolist())
            # Per-bucket fit
            bucket_weights = {}
            for bk, bidxs in bucket_indices.items():
                train_bk = np.array([i for i in bidxs if i not in held], dtype=np.int64)
                if len(train_bk) < 10:
                    bucket_weights[bk] = np.array([INIT_WEIGHTS[f] for f in FEATURE_NAMES])
                    continue
                w, _ = fit_powell(X, gt_idx, sizes, train_bk)
                bucket_weights[bk] = w

            # Evaluate held-out cases with their bucket's weights
            held_ndcgs = []
            for i in fold:
                i = int(i)
                bk = buckets[i]
                w = bucket_weights[bk]
                pool_size = int(sizes[i])
                if gt_idx[i] < 0:
                    held_ndcgs.append(0.0)
                    continue
                feat = X[i, :pool_size]
                scores = feat @ w
                gt_pos = int(gt_idx[i])
                gt_score = scores[gt_pos]
                rank = int((scores > gt_score).sum())
                ties_before = int(((scores == gt_score) & (np.arange(pool_size) < gt_pos)).sum())
                rank0 = rank + ties_before
                held_ndcgs.append(1.0 / math.log2(rank0 + 2) if rank0 < 20 else 0.0)
            fold_ndcgs.append(float(np.mean(held_ndcgs)))
        bucket_cv5_seeds.append(float(np.mean(fold_ndcgs)))

    bucket_cv5 = float(np.mean(bucket_cv5_seeds))
    bucket_cv5_std = float(np.std(bucket_cv5_seeds, ddof=1))

    # Global baseline CV5 for comparison
    gl_cv5 = global_results[best_global_src]["cv5"]

    print(f"  Global Powell CV5:     {gl_cv5:.4f}")
    print(f"  Per-bucket Powell CV5: {bucket_cv5:.4f} ± {bucket_cv5_std:.4f}")
    print(f"  Δ: {bucket_cv5 - gl_cv5:+.4f}")
    print(f"  Per-seed: {bucket_cv5_seeds}")

    # =====================================================================
    # PHASE 3: Per-bucket source config + Powell
    # =====================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Per-bucket source selection + Powell")
    print(f"{'='*70}")

    # Test: best source config per bucket
    # Build features for each source config, then mix per bucket
    # For each fold: per bucket, pick best source config on train, apply on held-out
    best_src_per_bucket_seeds = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_ndcgs = []
        for fold in folds:
            held = set(fold.tolist())
            # Per-bucket: try each source config, pick best on train
            bucket_weights_and_X = {}
            for bk, bidxs in bucket_indices.items():
                train_bk = [i for i in bidxs if i not in held]
                if len(train_bk) < 10:
                    # Default to best global
                    bucket_weights_and_X[bk] = (
                        np.array([INIT_WEIGHTS[f] for f in FEATURE_NAMES]),
                        global_results[best_global_src]["X"],
                        global_results[best_global_src]["gt_idx"],
                        global_results[best_global_src]["sizes"],
                    )
                    continue
                best_train_ndcg = -1
                best_combo = None
                for src_name in source_configs:
                    Xs = global_results[src_name]["X"]
                    gs = global_results[src_name]["gt_idx"]
                    ss = global_results[src_name]["sizes"]
                    train_idx = np.array(train_bk, dtype=np.int64)
                    w, train_ndcg = fit_powell(Xs, gs, ss, train_idx)
                    if train_ndcg > best_train_ndcg:
                        best_train_ndcg = train_ndcg
                        best_combo = (w, Xs, gs, ss, src_name)
                bucket_weights_and_X[bk] = best_combo[:4]

            # Evaluate
            held_ndcgs = []
            for i in fold:
                i = int(i)
                bk = buckets[i]
                w, Xs, gs, ss = bucket_weights_and_X[bk]
                pool_size = int(ss[i])
                if gs[i] < 0:
                    held_ndcgs.append(0.0)
                    continue
                feat = Xs[i, :pool_size]
                scores = feat @ w
                gt_pos = int(gs[i])
                gt_score = scores[gt_pos]
                rank = int((scores > gt_score).sum())
                ties = int(((scores == gt_score) & (np.arange(pool_size) < gt_pos)).sum())
                rank0 = rank + ties
                held_ndcgs.append(1.0 / math.log2(rank0 + 2) if rank0 < 20 else 0.0)
            fold_ndcgs.append(float(np.mean(held_ndcgs)))
        best_src_per_bucket_seeds.append(float(np.mean(fold_ndcgs)))

    bsb_cv5 = float(np.mean(best_src_per_bucket_seeds))
    bsb_std = float(np.std(best_src_per_bucket_seeds, ddof=1))
    print(f"  Per-bucket source+Powell CV5: {bsb_cv5:.4f} ± {bsb_std:.4f}")
    print(f"  Δ vs global: {bsb_cv5 - gl_cv5:+.4f}")
    print(f"  Per-seed: {best_src_per_bucket_seeds}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    best_cv5 = max(gl_cv5, bucket_cv5, bsb_cv5)
    best_name = (
        "per_bucket_source_powell" if bsb_cv5 == best_cv5 else
        "per_bucket_powell" if bucket_cv5 == best_cv5 else
        f"global_{best_global_src}"
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Global {best_global_src}:         CV5={gl_cv5:.4f}")
    print(f"  Per-bucket Powell:         CV5={bucket_cv5:.4f} ± {bucket_cv5_std:.4f} (Δ={bucket_cv5-gl_cv5:+.4f})")
    print(f"  Per-bucket source+Powell:  CV5={bsb_cv5:.4f} ± {bsb_std:.4f} (Δ={bsb_cv5-gl_cv5:+.4f})")
    print(f"  Best: {best_name} → {best_cv5:.4f}")

    delta = best_cv5 - gl_cv5
    if delta >= 0.010:
        verdict = "PASS"
    elif delta >= 0.005:
        verdict = "PROMISING"
    else:
        verdict = "FAIL"
    print(f"\n  GATE: {verdict}")
    print(f"  Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR12_bucket_policies.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "global_baselines": {k: {"cv5": v["cv5"], "hit50": v["hit50"],
                                  "bucket_ndcg": v["bucket_ndcg"]}
                              for k, v in global_results.items()},
        "per_bucket_powell": {"cv5": bucket_cv5, "cv5_std": bucket_cv5_std,
                              "cv5_seeds": bucket_cv5_seeds},
        "per_bucket_source_powell": {"cv5": bsb_cv5, "cv5_std": bsb_std,
                                     "cv5_seeds": best_src_per_bucket_seeds},
        "best": best_name,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Artifact: {out_path}")


if __name__ == "__main__":
    main()
