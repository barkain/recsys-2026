#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Phase 3: LightGBM LambdaRank over expanded candidate pool.

Replaces 8-feature Powell ranker with a nonlinear pairwise ranker.
Uses ABCDF+ALS sources, pool_k=100, and richer features including
source ranks, presence bits, and ALS scores.

WARNING: This script originally used cv_folds which leaks sessions.
Updated to use grouped_session_folds (zero session overlap).
No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import DownloadConfig, load_dataset
from implicit.als import AlternatingLeastSquares

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import grouped_session_folds
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
POOL_K = 100
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


FEATURE_NAMES_LR = [
    # Original 8
    "rrf_rank_inv",
    "same_artist",
    "tag_jaccard",
    "artist_tok_overlap",
    "title_tok_overlap",
    "meta_tok_overlap",
    "already_played",
    "recency_weighted_meta",
    # Source ranks (reciprocal, 0 if absent)
    "rank_A", "rank_B", "rank_C", "rank_D", "rank_F", "rank_ALS",
    # Source presence bits
    "in_A", "in_B", "in_C", "in_D", "in_F", "in_ALS",
    # Source count
    "n_sources",
    # ALS affinity score
    "als_score",
    # History depth
    "n_hist",
]


def build_als():
    """Train ALS and return factors + track mapping."""
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
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    elif not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    return factors, track_ids, track_to_idx


def als_session_vector(played, track_to_idx, factors, decay=0.8):
    anchors = [track_to_idx[t] for t in played if t in track_to_idx]
    if not anchors:
        return None
    n = len(anchors)
    w = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    w /= w.sum()
    v = np.zeros(factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        v += w[j] * factors[idx]
    return v


def build_als_source(cases, track_to_idx, factors, track_ids):
    """Build ALS retrieval lists for all eval cases."""
    als_source = []
    als_session_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, track_to_idx, factors)
        als_session_vecs.append(sv)
        if sv is not None:
            scores = factors @ sv
            played_idx = [track_to_idx[t] for t in played if t in track_to_idx]
            for idx in played_idx:
                scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([track_ids[j] for j in top_idx])
        else:
            als_source.append([])
    return als_source, als_session_vecs


def build_features(payload, als_source, als_session_vecs, als_factors,
                   als_track_to_idx, pool_k=POOL_K):
    """Build rich feature matrix for LambdaRank."""
    cases = payload["cases"]
    n = len(cases)
    n_feat = len(FEATURE_NAMES_LR)
    X = np.zeros((n, pool_k, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        # Per-source ranked lists
        src_lists = {
            "A": payload["src_a"][i],
            "B": payload["src_b"][i],
            "C": payload["src_c"][i],
            "D": payload["src_d"][i],
            "F": payload["src_f"][i],
            "ALS": als_source[i],
        }

        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=pool_k, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        # Build per-source rank dicts
        src_rank = {}
        for sname, slist in src_lists.items():
            src_rank[sname] = {tid: rank + 1 for rank, tid in enumerate(slist)}

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]

        sv = als_session_vecs[i]

        for rank, tid in enumerate(pool[:pool_k], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Original 8 features
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

            # Per-source reciprocal ranks
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0

            # Source presence bits
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0

            # Source count
            row[20] = sum(1 for sname in src_lists if tid in src_rank[sname])

            # ALS affinity score
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))

            # History depth
            row[22] = float(n_hist)

    return X, gt_idx, sizes


def ndcg_at_k(scores, gt_idx, sizes, k=20):
    """Compute mean nDCG@k from scores."""
    n = len(gt_idx)
    pool_axis = np.arange(scores.shape[1])[None, :]
    valid = pool_axis < sizes[:, None]
    scores = np.where(valid, scores, -np.inf)

    vals = np.zeros(n)
    for i in range(n):
        gt = gt_idx[i]
        if gt < 0:
            continue
        s = scores[i]
        gt_score = s[gt]
        rank0 = int(np.sum(s > gt_score) + np.sum((s == gt_score) & (pool_axis[0] < gt)))
        if rank0 < k:
            vals[i] = 1.0 / np.log2(rank0 + 2)
    return float(vals.mean())


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

    print(f"{ts()} Building ALS source...", flush=True)
    als_source, als_session_vecs = build_als_source(
        cases, als_track_to_idx, als_factors, als_track_ids)

    print(f"{ts()} Building features ({len(FEATURE_NAMES_LR)} features, pool_k={POOL_K})...",
          flush=True)
    X, gt_idx, sizes = build_features(
        payload, als_source, als_session_vecs,
        als_factors, als_track_to_idx, pool_k=POOL_K)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}={pool_hit:.4f}")

    # Flatten for LightGBM: (n * pool_k, n_feat)
    X_flat = X.reshape(-1, len(FEATURE_NAMES_LR))
    # Labels: 1 if GT, 0 otherwise
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0

    # Group sizes for LambdaRank
    group_sizes = np.full(n, POOL_K, dtype=np.int32)
    # Adjust for shorter pools
    for i in range(n):
        if sizes[i] < POOL_K:
            group_sizes[i] = int(sizes[i])

    seeds = [0, 1, 2, 3, 4]

    # --- Powell baseline on same pool for fair comparison ---
    from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS
    from scripts.expS2_als_feature import vec_ndcg, fit_powell

    print(f"\n{ts()} Powell baseline on pool_k={POOL_K}...", flush=True)
    X_powell = X[:, :, :8]  # first 8 features = original Powell features
    powell_cv5_seeds = []
    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_powell(X_powell, gt_idx, sizes, train_idx, FEATURE_NAMES, INIT_WEIGHTS)
            fold_sc.append(vec_ndcg(X_powell, gt_idx, sizes, w, fold))
        powell_cv5_seeds.append(float(np.mean(fold_sc)))
    powell_cv5 = float(np.mean(powell_cv5_seeds))
    print(f"  Powell CV5={powell_cv5:.4f} ± {np.std(powell_cv5_seeds, ddof=1):.4f}")

    # --- LambdaRank CV5 ---
    print(f"\n{ts()} LambdaRank CV5...", flush=True)

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,
    }

    lr_cv5_seeds = []
    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        fold_ndcgs = []
        for fi, fold in enumerate(folds):
            held = set(fold.tolist())
            train_cases = [j for j in range(n) if j not in held]
            val_cases = fold.tolist()

            # Build train/val flat indices
            train_flat = []
            for j in train_cases:
                for k in range(int(sizes[j])):
                    train_flat.append(j * POOL_K + k)
            val_flat = []
            for j in val_cases:
                for k in range(int(sizes[j])):
                    val_flat.append(j * POOL_K + k)

            X_train = X_flat[train_flat]
            y_train = labels[train_flat]
            X_val = X_flat[val_flat]
            y_val = labels[val_flat]

            # Group sizes
            g_train = np.array([int(sizes[j]) for j in train_cases], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_cases], dtype=np.int32)

            dtrain = lgb.Dataset(X_train, y_train, group=g_train,
                                 feature_name=FEATURE_NAMES_LR, free_raw_data=False)
            dval = lgb.Dataset(X_val, y_val, group=g_val,
                               reference=dtrain, free_raw_data=False)

            model = lgb.train(
                lgb_params, dtrain,
                num_boost_round=300,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            # Predict and compute nDCG@20
            val_scores = model.predict(X_val)
            # Reshape back to per-case
            offset = 0
            case_ndcgs = []
            for j in val_cases:
                sz = int(sizes[j])
                if sz == 0:
                    case_ndcgs.append(0.0)
                    offset += sz
                    continue
                sc = val_scores[offset:offset + sz]
                gt = gt_idx[j]
                if gt >= 0:
                    gt_score = sc[gt]
                    rank0 = int(np.sum(sc > gt_score) +
                                np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                    if rank0 < 20:
                        case_ndcgs.append(1.0 / np.log2(rank0 + 2))
                    else:
                        case_ndcgs.append(0.0)
                else:
                    case_ndcgs.append(0.0)
                offset += sz
            fold_ndcgs.append(float(np.mean(case_ndcgs)))

        seed_ndcg = float(np.mean(fold_ndcgs))
        lr_cv5_seeds.append(seed_ndcg)
        print(f"  Seed {seed}: CV5={seed_ndcg:.4f}", flush=True)

    lr_cv5 = float(np.mean(lr_cv5_seeds))
    lr_cv5_std = float(np.std(lr_cv5_seeds, ddof=1))
    lift = lr_cv5 - powell_cv5

    print(f"\n{ts()} {'='*60}")
    print(f"FINAL REPORT")
    print(f"  Pool: ABCDF+ALS, pool_k={POOL_K}, pool_hit={pool_hit:.4f}")
    print(f"  Powell (8f):     CV5={powell_cv5:.4f}")
    print(f"  LambdaRank (23f): CV5={lr_cv5:.4f} ± {lr_cv5_std:.4f}")
    print(f"  Lift: {lift:+.4f}")
    print(f"  GATE CV5 lift >= +0.010: {'PASS' if lift >= 0.010 else 'FAIL'}")

    # Feature importance
    print(f"\n  Top features (last fold):")
    imp = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES_LR, imp), key=lambda x: -x[1])
    for fname, fimp in feat_imp[:10]:
        print(f"    {fname:25s} {fimp:10.1f}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_lambdarank.json"
    out = {
        "pool_k": POOL_K, "pool_hit": pool_hit,
        "powell_cv5": powell_cv5,
        "lambdarank_cv5": lr_cv5, "lambdarank_cv5_std": lr_cv5_std,
        "lift": lift,
        "gate_cv5_010": lift >= 0.010,
        "feature_importance": {f: float(v) for f, v in feat_imp},
        "lgb_params": lgb_params,
        "cv5_seeds": lr_cv5_seeds,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
