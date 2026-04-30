#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Phase 1 (S2 plan): ALS/WRMF collaborative filtering candidate source.

Builds session×track implicit interaction matrix from train data,
trains ALS with a hyperparameter grid, evaluates on the B1 8000-case
all-turn benchmark.

Metrics: standalone hit@20/50/200, unique hits vs ABCDF@200, fusion CV5.
No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import DownloadConfig, load_dataset
from implicit.als import AlternatingLeastSquares

from scripts.expA1_ablation_cv5 import build_features, eval_cv5, vec_ndcg, fit_powell
from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.r3_confirm_400_deterministic import cv_folds
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
POOL_K = 50
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_interaction_matrix():
    """Build session×track CSR matrix from train conversations."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]

    session_ids = []
    track_set = set()
    session_tracks = []

    for item in train:
        sid = str(item["session_id"])
        convs = item["conversations"]
        tracks_in_session = []
        for c in convs:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks_in_session.append(tid)
                track_set.add(tid)
        if tracks_in_session:
            session_ids.append(sid)
            session_tracks.append(tracks_in_session)

    track_ids = sorted(track_set)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    rows, cols, vals = [], [], []
    for si, tracks in enumerate(session_tracks):
        for tid in tracks:
            rows.append(si)
            cols.append(track_to_idx[tid])
            vals.append(1.0)

    matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(session_ids), len(track_ids)),
        dtype=np.float32,
    )

    return matrix, session_ids, track_ids, track_to_idx


def train_als(matrix, factors=128, alpha=40, regularization=0.05, iterations=20):
    """Train ALS model and return it."""
    model = AlternatingLeastSquares(
        factors=factors,
        alpha=alpha,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
        use_gpu=False,
    )
    model.fit(matrix)
    return model


def retrieve_als(model, played_track_ids, track_to_idx, track_ids,
                 item_factors, topk=200, decay=0.8):
    """Retrieve top-k tracks given played history using ALS item factors."""
    anchors = []
    for tid in played_track_ids:
        idx = track_to_idx.get(tid)
        if idx is not None:
            anchors.append(idx)

    if not anchors:
        return []

    n_anchors = len(anchors)
    weights = np.array([decay ** (n_anchors - 1 - j) for j in range(n_anchors)],
                       dtype=np.float32)
    weights /= weights.sum()

    session_vec = np.zeros(item_factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        session_vec += weights[j] * item_factors[idx]

    scores = item_factors @ session_vec
    played_set = set(anchors)
    for idx in played_set:
        scores[idx] = -np.inf

    top_idx = np.argpartition(-scores, min(topk, len(scores) - 1))[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [track_ids[i] for i in top_idx]


def eval_standalone(als_results, cases, abcdf_pools):
    """Compute standalone hit metrics and unique hits."""
    n = len(cases)
    hit20 = hit50 = hit200 = 0
    unique_hits = 0
    unique_ranks = []
    gt_ranks = []

    hist_buckets = defaultdict(lambda: {"n": 0, "hit20": 0, "hit200": 0, "unique": 0})

    for i, c in enumerate(cases):
        gt = c["gt"]
        n_hist = len(c["music_turns"])
        bk = f"hist_{min(n_hist, 7)}"

        hist_buckets[bk]["n"] += 1
        top_list = als_results[i]

        if gt in top_list[:20]:
            hit20 += 1
            hist_buckets[bk]["hit20"] += 1
        if gt in top_list[:50]:
            hit50 += 1
        if gt in set(top_list[:200]):
            hit200 += 1
            hist_buckets[bk]["hit200"] += 1
            rank = top_list.index(gt) + 1
            gt_ranks.append(rank)
            if gt not in abcdf_pools[i]:
                unique_hits += 1
                unique_ranks.append(rank)
                hist_buckets[bk]["unique"] += 1

    return {
        "hit20": hit20, "hit50": hit50, "hit200": hit200,
        "hit20_rate": hit20 / n, "hit50_rate": hit50 / n, "hit200_rate": hit200 / n,
        "unique_hits": unique_hits,
        "unique_median_rank": float(np.median(unique_ranks)) if unique_ranks else None,
        "gt_median_rank": float(np.median(gt_ranks)) if gt_ranks else None,
        "hist_buckets": {k: dict(v) for k, v in sorted(hist_buckets.items())},
    }


def build_features_with_als(payload, als_results, source_weights, pool_k=50, rrf_k=20):
    """Build Powell features with ALS as an additional source."""
    cases = payload["cases"]
    n = len(cases)
    n_features = len(FEATURE_NAMES)
    X = np.zeros((n, pool_k, n_features), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        sources = {}
        if source_weights.get("A", 0) > 0:
            sources["A"] = payload["src_a"][i]
        if source_weights.get("B", 0) > 0:
            sources["B"] = payload["src_b"][i]
        if source_weights.get("C", 0) > 0:
            sources["C"] = payload["src_c"][i]
        if source_weights.get("D", 0) > 0:
            sources["D"] = payload["src_d"][i]
        if source_weights.get("F", 0) > 0:
            sources["F"] = payload["src_f"][i]
        if source_weights.get("ALS", 0) > 0:
            sources["ALS"] = als_results[i]

        pool = weighted_rrf(sources, source_weights, topk=pool_k, k=rrf_k)
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
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set())) for j, t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:pool_k], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
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

    return X, gt_idx, sizes


def main():
    t0 = time.time()

    # --- Load eval payload ---
    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases, {len(set(sessions))} sessions")

    # --- Build ABCDF@200 pools for unique-hit comparison ---
    print(f"{ts()} Building ABCDF@200 pools...", flush=True)
    abcdf_pools = []
    for i in range(n):
        sources = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i],
        }
        weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources, weights, topk=200, k=RRF_K)
        abcdf_pools.append(set(pool))

    # --- Build interaction matrix ---
    print(f"{ts()} Building interaction matrix...", flush=True)
    matrix, train_session_ids, track_ids, track_to_idx = build_interaction_matrix()
    print(f"  Matrix: {matrix.shape[0]} sessions × {matrix.shape[1]} tracks, {matrix.nnz} interactions")

    # --- Hyperparameter grid ---
    grid = [
        {"factors": 64, "alpha": 40, "regularization": 0.05},
        {"factors": 128, "alpha": 40, "regularization": 0.05},
        {"factors": 256, "alpha": 40, "regularization": 0.05},
        {"factors": 128, "alpha": 10, "regularization": 0.05},
        {"factors": 128, "alpha": 100, "regularization": 0.05},
        {"factors": 128, "alpha": 40, "regularization": 0.01},
        {"factors": 128, "alpha": 40, "regularization": 0.1},
    ]

    best_config = None
    best_unique = -1
    all_results = {}

    for gi, params in enumerate(grid):
        config_name = f"f{params['factors']}_a{params['alpha']}_r{params['regularization']}"
        print(f"\n{ts()} Config {gi+1}/{len(grid)}: {config_name}", flush=True)

        model = train_als(matrix, **params, iterations=20)
        item_factors = model.item_factors
        if hasattr(item_factors, 'to_numpy'):
            item_factors = item_factors.to_numpy()
        elif not isinstance(item_factors, np.ndarray):
            item_factors = np.array(item_factors)

        # Retrieve for each eval case
        als_results_list = []
        for i, c in enumerate(cases):
            played = c["music_turns"]
            top200 = retrieve_als(model, played, track_to_idx, track_ids,
                                  item_factors, topk=200, decay=0.8)
            als_results_list.append(top200)

        # Standalone eval
        standalone = eval_standalone(als_results_list, cases, abcdf_pools)
        print(f"  hit@20={standalone['hit20']}/{n} ({standalone['hit20_rate']:.1%})")
        print(f"  hit@50={standalone['hit50']}/{n} ({standalone['hit50_rate']:.1%})")
        print(f"  hit@200={standalone['hit200']}/{n} ({standalone['hit200_rate']:.1%})")
        print(f"  unique_hits vs ABCDF@200={standalone['unique_hits']}")
        if standalone["unique_median_rank"]:
            print(f"  unique median rank={standalone['unique_median_rank']:.0f}")

        # History-depth breakdown
        print(f"  {'bucket':10s} {'n':>5s} {'hit@20':>7s} {'hit@200':>8s} {'unique':>7s}")
        for bk, d in sorted(standalone["hist_buckets"].items()):
            bn = d["n"]
            print(f"  {bk:10s} {bn:5d} {d['hit20']/bn:7.1%} {d['hit200']/bn:8.1%} {d['unique']:7d}")

        all_results[config_name] = {
            "params": params,
            "standalone": standalone,
        }

        if standalone["unique_hits"] > best_unique:
            best_unique = standalone["unique_hits"]
            best_config = config_name
            best_als_results = als_results_list

    # --- Summary ---
    print(f"\n{ts()} {'='*60}")
    print(f"Best config: {best_config} with {best_unique} unique hits vs ABCDF@200")

    # --- Gate check ---
    if best_unique < 200:
        print(f"\n  GATE FAIL: unique_hits ({best_unique}) < 200. Stop Phase 1.")
        elapsed = time.time() - t0
        print(f"\n{ts()} Elapsed: {elapsed:.1f}s")
        out_path = REPO_ROOT / "exp" / "eval" / "expS2_als_candidate_source.json"
        with open(out_path, "w") as f:
            json.dump({"gate": "FAIL", "best_config": best_config,
                       "best_unique_hits": best_unique, "all_results": all_results}, f, indent=2, default=str)
        print(f"Artifact: {out_path}")
        return

    # --- Fusion sweep (only if >= 200 unique hits) ---
    print(f"\n{ts()} Running fusion sweep with best config: {best_config}")
    seeds = [0, 1, 2, 3, 4]

    # Baseline: ABCDF only
    print(f"\n  Baseline ABCDF...", flush=True)
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
    X_base, gt_base, sizes_base = build_features(payload, base_weights)
    cv5_base = eval_cv5(X_base, gt_base, sizes_base, sessions, seeds)
    cv5_base_mean = float(np.mean(cv5_base))
    pool_hit_base = float(np.mean(gt_base >= 0))
    print(f"  ABCDF CV5={cv5_base_mean:.4f}, pool_hit@50={pool_hit_base:.4f}")

    fusion_results = {}
    best_fusion_cv5 = cv5_base_mean
    best_fusion_name = "baseline"

    for w_als in [0.25, 0.5, 1.0, 2.0]:
        fusion_name = f"w_ALS={w_als}"
        print(f"\n  {fusion_name}...", flush=True)
        fusion_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": w_als}

        X_f, gt_f, sizes_f = build_features_with_als(
            payload, best_als_results, fusion_weights, pool_k=POOL_K, rrf_k=RRF_K)

        pool_hit_f = float(np.mean(gt_f >= 0))
        cv5_f = eval_cv5(X_f, gt_f, sizes_f, sessions, seeds)
        cv5_f_mean = float(np.mean(cv5_f))
        cv5_f_std = float(np.std(cv5_f, ddof=1))
        delta = cv5_f_mean - cv5_base_mean

        print(f"    pool_hit@50={pool_hit_f:.4f} (Δ={pool_hit_f - pool_hit_base:+.4f})")
        print(f"    CV5={cv5_f_mean:.4f} ± {cv5_f_std:.4f} (Δ={delta:+.4f})")

        fusion_results[fusion_name] = {
            "w_als": w_als,
            "pool_hit50": pool_hit_f,
            "cv5": cv5_f_mean,
            "cv5_std": cv5_f_std,
            "cv5_seeds": cv5_f,
            "delta_cv5": delta,
            "delta_pool_hit": pool_hit_f - pool_hit_base,
        }

        if cv5_f_mean > best_fusion_cv5:
            best_fusion_cv5 = cv5_f_mean
            best_fusion_name = fusion_name

    # --- Final report ---
    print(f"\n{ts()} {'='*60}")
    print(f"FINAL REPORT")
    print(f"  Best ALS config: {best_config}")
    print(f"  Unique hits vs ABCDF@200: {best_unique}")
    print(f"  Baseline ABCDF CV5: {cv5_base_mean:.4f}")
    print(f"  Best fusion: {best_fusion_name} CV5={best_fusion_cv5:.4f} (Δ={best_fusion_cv5 - cv5_base_mean:+.4f})")

    # Gate verdicts
    print(f"\n  GATE: unique_hits >= 300: {'PASS' if best_unique >= 300 else 'FAIL'} ({best_unique})")
    print(f"  GATE: unique_hits >= 500 (strong): {'PASS' if best_unique >= 500 else 'FAIL'} ({best_unique})")
    cv5_lift = best_fusion_cv5 - cv5_base_mean
    print(f"  GATE: CV5 lift >= +0.010: {'PASS' if cv5_lift >= 0.010 else 'FAIL'} ({cv5_lift:+.4f})")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_als_candidate_source.json"
    out = {
        "best_config": best_config,
        "best_unique_hits": best_unique,
        "baseline_cv5": cv5_base_mean,
        "best_fusion_name": best_fusion_name,
        "best_fusion_cv5": best_fusion_cv5,
        "cv5_lift": cv5_lift,
        "gate_unique_300": best_unique >= 300,
        "gate_unique_500": best_unique >= 500,
        "gate_cv5_lift": cv5_lift >= 0.010,
        "all_configs": all_results,
        "fusion_results": fusion_results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
