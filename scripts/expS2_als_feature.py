#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Experiment: ALS score as ranker feature + larger pool.

Tests whether adding ALS dot-product affinity as a 9th Powell feature
allows the ranker to convert ALS candidates into nDCG.

Also tests pool_k={50, 100} to see if larger pools help.

No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import DownloadConfig, load_dataset
from implicit.als import AlternatingLeastSquares

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.r3_confirm_400_deterministic import cv_folds
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20

FEATURE_NAMES_EXT = FEATURE_NAMES + ["als_score"]
INIT_WEIGHTS_EXT = {**INIT_WEIGHTS, "als_score": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_als():
    """Train ALS (best config from Phase 1) and return factors + track mapping."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]
    track_set = set()
    session_tracks = []
    for item in train:
        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                track_set.add(tid)
        session_tracks.append(tracks)

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
        shape=(len(session_tracks), len(track_ids)),
        dtype=np.float32,
    )

    model = AlternatingLeastSquares(
        factors=128, alpha=100, regularization=0.05,
        iterations=20, random_state=42, use_gpu=False,
    )
    model.fit(matrix)
    item_factors = model.item_factors
    if hasattr(item_factors, "to_numpy"):
        item_factors = item_factors.to_numpy()
    elif not isinstance(item_factors, np.ndarray):
        item_factors = np.array(item_factors)

    return item_factors, track_ids, track_to_idx


def als_session_vector(played_tracks, track_to_idx, item_factors, decay=0.8):
    """Build ALS session vector from played tracks."""
    anchors = []
    for tid in played_tracks:
        idx = track_to_idx.get(tid)
        if idx is not None:
            anchors.append(idx)
    if not anchors:
        return None
    n = len(anchors)
    weights = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    weights /= weights.sum()
    vec = np.zeros(item_factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        vec += weights[j] * item_factors[idx]
    return vec


def als_track_score(session_vec, tid, track_to_idx, item_factors):
    """Compute ALS affinity score for a single track."""
    if session_vec is None:
        return 0.0
    idx = track_to_idx.get(tid)
    if idx is None:
        return 0.0
    return float(np.dot(session_vec, item_factors[idx]))


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
    tie_before = ((scores == gt_scores[:, None]) & valid_pool
                  & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_powell(X, gt_idx, sizes, train_idx, feature_names, init_weights):
    init = np.array([init_weights[name] for name in feature_names], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def eval_cv5(X, gt_idx, sizes, sessions, seeds, feature_names, init_weights):
    n = len(sessions)
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_powell(X, gt_idx, sizes, train, feature_names, init_weights)
            fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
        per_seed.append(float(np.mean(fold_sc)))
    return per_seed


def build_features_ext(payload, source_weights, als_item_factors, als_track_to_idx,
                       pool_k=50, rrf_k=20, use_als_feature=True, als_source=None):
    """Build features with optional ALS score as extra feature."""
    cases = payload["cases"]
    n = len(cases)
    feat_names = FEATURE_NAMES_EXT if use_als_feature else FEATURE_NAMES
    n_feat = len(feat_names)
    X = np.zeros((n, pool_k, n_feat), dtype=np.float64)
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
        if als_source is not None and source_weights.get("ALS", 0) > 0:
            sources["ALS"] = als_source[i]

        pool = weighted_rrf(sources, source_weights, topk=pool_k, k=rrf_k)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        # Precompute ALS session vector
        played = c["music_turns"]
        session_vec = als_session_vector(played, als_track_to_idx, als_item_factors)

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]

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

            if use_als_feature:
                row[8] = als_track_score(session_vec, tid, als_track_to_idx,
                                         als_item_factors)

    return X, gt_idx, sizes


def main():
    t0 = time.time()

    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases, {len(set(sessions))} sessions")

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    print(f"  {len(als_track_ids)} tracks, factors shape: {als_factors.shape}")

    # Build ALS retrieval lists for source fusion
    print(f"{ts()} Building ALS retrieval lists...", flush=True)
    als_source = []
    for i, c in enumerate(cases):
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            played_idx = [als_track_to_idx[t] for t in played if t in als_track_to_idx]
            for idx in played_idx:
                scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    seeds = [0, 1, 2, 3, 4]

    configs = [
        # Baseline: ABCDF, 8 features, pool=50
        ("ABCDF_8f_p50",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
         50, False, False),

        # ABCDF + ALS score feature, pool=50
        ("ABCDF_9f_p50",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
         50, True, False),

        # ABCDF + ALS source + ALS feature, pool=50
        ("ABCDF+ALS_9f_p50",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0},
         50, True, True),

        # ABCDF + ALS source + ALS feature, pool=100
        ("ABCDF+ALS_9f_p100",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0},
         100, True, True),

        # ABCDF, 8 features, pool=100 (control: does larger pool help without ALS?)
        ("ABCDF_8f_p100",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
         100, False, False),

        # ABCDF + ALS source, 8 features, pool=50 (control: ALS source without feature)
        ("ABCDF+ALS_8f_p50",
         {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0},
         50, False, True),
    ]

    results = {}
    for name, weights, pool_k, use_als_feat, use_als_src in configs:
        print(f"\n{ts()} {name}...", flush=True)
        feat_names = FEATURE_NAMES_EXT if use_als_feat else FEATURE_NAMES
        init_w = INIT_WEIGHTS_EXT if use_als_feat else INIT_WEIGHTS

        X, gt_idx, sizes = build_features_ext(
            payload, weights, als_factors, als_track_to_idx,
            pool_k=pool_k, rrf_k=RRF_K,
            use_als_feature=use_als_feat,
            als_source=als_source if use_als_src else None,
        )

        pool_hit = float(np.mean(gt_idx >= 0))
        cv5 = eval_cv5(X, gt_idx, sizes, sessions, seeds, feat_names, init_w)
        cv5_mean = float(np.mean(cv5))
        cv5_std = float(np.std(cv5, ddof=1))

        print(f"  pool_hit@{pool_k}={pool_hit:.4f}  CV5={cv5_mean:.4f} ± {cv5_std:.4f}")
        results[name] = {
            "pool_k": pool_k, "pool_hit": pool_hit,
            "cv5": cv5_mean, "cv5_std": cv5_std, "cv5_seeds": cv5,
            "use_als_feature": use_als_feat, "use_als_source": use_als_src,
        }

    # Summary table
    baseline_cv5 = results["ABCDF_8f_p50"]["cv5"]
    print(f"\n{ts()} {'='*70}")
    print(f"{'Config':30s} {'pool_k':>6s} {'pool_hit':>9s} {'CV5':>7s} {'Δ CV5':>8s}")
    for name, r in results.items():
        delta = r["cv5"] - baseline_cv5
        print(f"  {name:30s} {r['pool_k']:6d} {r['pool_hit']:9.4f} "
              f"{r['cv5']:7.4f} {delta:+8.4f}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_als_feature.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
