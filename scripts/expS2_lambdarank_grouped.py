#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""LambdaRank with GROUPED-SESSION CV5 (no session leakage).

Splits by unique session_id so ALL turns of a session are in the same fold.
Reports both Powell and LambdaRank CV5 under this stricter split.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
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
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expS2_lambdarank import FEATURE_NAMES_LR, build_als  # noqa: E402 — no circular import since lambdarank no longer imports from _grouped

from scipy.optimize import minimize

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
POOL_K = 100
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def grouped_session_folds(sessions, seed, k=5):
    """Split by unique session_id — all turns of a session go to the same fold."""
    unique_sessions = sorted(set(sessions))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_sessions)

    # Assign each unique session to a fold
    session_to_fold = {}
    for i, sid in enumerate(unique_sessions):
        session_to_fold[sid] = i % k

    # Build fold index arrays (case indices)
    folds = [[] for _ in range(k)]
    for case_idx, sid in enumerate(sessions):
        folds[session_to_fold[sid]].append(case_idx)

    return [np.asarray(f, dtype=np.int64) for f in folds]


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


def fit_powell(X, gt_idx, sizes, train_idx):
    init = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def build_features(payload, als_source, als_session_vecs, als_factors,
                   als_track_to_idx, pool_k=POOL_K):
    """Build 23-feature matrix."""
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
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=pool_k, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

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
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sname in src_lists if tid in src_rank[sname])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)

    return X, gt_idx, sizes


def main():
    t0 = time.time()

    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    unique_sessions = sorted(set(sessions))
    print(f"  {n} cases, {len(unique_sessions)} unique sessions")

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()

    print(f"{ts()} Building ALS source...", flush=True)
    als_source = []
    als_session_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_session_vecs.append(sv)
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

    print(f"{ts()} Building features...", flush=True)
    X, gt_idx, sizes = build_features(
        payload, als_source, als_session_vecs,
        als_factors, als_track_to_idx)
    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}={pool_hit:.4f}")

    # Flatten for LightGBM
    n_feat = len(FEATURE_NAMES_LR)
    X_flat = X.reshape(-1, n_feat)
    labels_flat = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels_flat[i * POOL_K + gt_idx[i]] = 1.0

    seeds = [0, 1, 2, 3, 4]

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

    # --- Grouped-session CV5 ---
    print(f"\n{ts()} GROUPED-SESSION CV5 (no session leakage)", flush=True)

    powell_cv5_seeds = []
    lr_cv5_seeds = []

    # Also track per-hist-bucket and last-turn-only
    lr_hist_ndcg = defaultdict(list)
    lr_last_turn_ndcg = []

    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)

        # Verify zero overlap
        for fi, fold in enumerate(folds):
            fold_sids = set(sessions[j] for j in fold)
            other_sids = set(sessions[j] for fj, f2 in enumerate(folds)
                             if fj != fi for j in f2)
            overlap = fold_sids & other_sids
            if seed == 0 and fi == 0:
                print(f"  Fold 0 check: {len(fold)} cases, {len(fold_sids)} sessions, "
                      f"overlap with train: {len(overlap)} sessions")
            assert len(overlap) == 0, f"Session leakage! {len(overlap)} overlapping sessions"

        powell_fold_sc = []
        lr_fold_sc = []

        for fi, fold in enumerate(folds):
            held = set(fold.tolist())
            train_cases = [j for j in range(n) if j not in held]
            val_cases = fold.tolist()

            # Powell (8 features)
            X_powell = X[:, :, :8]
            w, _ = fit_powell(X_powell, gt_idx, sizes, np.asarray(train_cases, dtype=np.int64))
            powell_ndcg = vec_ndcg(X_powell, gt_idx, sizes, w, fold)
            powell_fold_sc.append(powell_ndcg)

            # LambdaRank (23 features)
            train_flat = []
            for j in train_cases:
                for k in range(int(sizes[j])):
                    train_flat.append(j * POOL_K + k)
            val_flat = []
            for j in val_cases:
                for k in range(int(sizes[j])):
                    val_flat.append(j * POOL_K + k)

            X_train = X_flat[train_flat]
            y_train = labels_flat[train_flat]
            X_val = X_flat[val_flat]

            g_train = np.array([int(sizes[j]) for j in train_cases], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_cases], dtype=np.int32)

            dtrain = lgb.Dataset(X_train, y_train, group=g_train,
                                 feature_name=FEATURE_NAMES_LR, free_raw_data=False)
            dval_ds = lgb.Dataset(X_val, labels_flat[val_flat], group=g_val,
                                  reference=dtrain, free_raw_data=False)

            model = lgb.train(
                lgb_params, dtrain, num_boost_round=300,
                valid_sets=[dval_ds],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            val_scores = model.predict(X_val)
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
                    ndcg_val = 1.0 / np.log2(rank0 + 2) if rank0 < 20 else 0.0
                else:
                    ndcg_val = 0.0
                case_ndcgs.append(ndcg_val)

                # Per-hist bucket
                n_hist = len(cases[j]["music_turns"])
                bk = f"hist_{min(n_hist, 7)}"
                lr_hist_ndcg[bk].append(ndcg_val)

                # Last-turn check (turn 8 = hist_7 for 8-turn sessions)
                if n_hist == 7:
                    lr_last_turn_ndcg.append(ndcg_val)

                offset += sz
            lr_fold_sc.append(float(np.mean(case_ndcgs)))

        powell_seed = float(np.mean(powell_fold_sc))
        lr_seed = float(np.mean(lr_fold_sc))
        powell_cv5_seeds.append(powell_seed)
        lr_cv5_seeds.append(lr_seed)
        print(f"  Seed {seed}: Powell={powell_seed:.4f}  LambdaRank={lr_seed:.4f}  "
              f"Δ={lr_seed - powell_seed:+.4f}", flush=True)

    powell_cv5 = float(np.mean(powell_cv5_seeds))
    lr_cv5 = float(np.mean(lr_cv5_seeds))
    lr_std = float(np.std(lr_cv5_seeds, ddof=1))
    lift = lr_cv5 - powell_cv5

    print(f"\n{ts()} {'='*60}")
    print(f"GROUPED-SESSION RESULTS (zero session leakage)")
    print(f"  Pool: ABCDF+ALS, pool_k={POOL_K}, pool_hit={pool_hit:.4f}")
    print(f"  Powell (8f):      CV5={powell_cv5:.4f}")
    print(f"  LambdaRank (23f): CV5={lr_cv5:.4f} ± {lr_std:.4f}")
    print(f"  Lift: {lift:+.4f}")
    print(f"  GATE CV5 lift >= +0.010: {'PASS' if lift >= 0.010 else 'FAIL'}")

    # Per-hist breakdown
    print(f"\n  Per-history-depth nDCG@20:")
    print(f"    {'bucket':10s} {'n':>5s} {'nDCG':>7s}")
    for bk in sorted(lr_hist_ndcg.keys()):
        vals = lr_hist_ndcg[bk]
        print(f"    {bk:10s} {len(vals):5d} {np.mean(vals):7.4f}")

    if lr_last_turn_ndcg:
        print(f"\n  Last-turn-only (hist_7) nDCG: {np.mean(lr_last_turn_ndcg):.4f} "
              f"(n={len(lr_last_turn_ndcg)})")

    # Feature importance from last model
    print(f"\n  Top features (last fold):")
    imp = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES_LR, imp), key=lambda x: -x[1])
    for fname, fimp in feat_imp[:10]:
        print(f"    {fname:25s} {fimp:10.1f}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_lambdarank_grouped.json"
    out = {
        "pool_k": POOL_K, "pool_hit": pool_hit,
        "powell_cv5": powell_cv5,
        "lambdarank_cv5": lr_cv5, "lambdarank_cv5_std": lr_std,
        "lift": lift,
        "gate_cv5_010": lift >= 0.010,
        "powell_seeds": powell_cv5_seeds,
        "lr_seeds": lr_cv5_seeds,
        "feature_importance": {f: float(v) for f, v in feat_imp},
        "hist_ndcg": {k: float(np.mean(v)) for k, v in sorted(lr_hist_ndcg.items())},
        "validation": "grouped-session, zero session overlap between train and val folds",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
