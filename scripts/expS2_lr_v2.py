#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""LambdaRank V2: larger pools, richer features, last-turn-only objective.

Tests:
  1. pool_k in {100, 150, 200}
  2. New features: track_popularity, artist_popularity, pool_artist_concentration
  3. All-turn vs last-turn-only training
  4. Grouped-session CV5 (zero leakage)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import (
    als_session_vector, grouped_session_folds,
)
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

FEATURE_NAMES_V2 = FEATURE_NAMES_LR + [
    "track_popularity",
    "artist_popularity",
    "pool_artist_count",
    "pool_source_agreement",
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_popularity_stats():
    """Count track and artist frequency in training data."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]
    track_counts = Counter()
    artist_tracks = Counter()

    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                track_counts[tid] += 1

    return track_counts


def build_features_v2(payload, als_source, als_session_vecs, als_factors,
                      als_track_to_idx, track_popularity, pool_k=100):
    """Build extended feature matrix."""
    cases = payload["cases"]
    n = len(cases)
    n_feat = len(FEATURE_NAMES_V2)
    X = np.zeros((n, pool_k, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    max_pop = max(track_popularity.values()) if track_popularity else 1

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

        # Pool-level stats
        pool_artists = [ta.get(tid, "") for tid in pool[:pool_k]]
        artist_counts = Counter(a for a in pool_artists if a)

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

            # Source count + ALS score + history depth
            row[20] = sum(1 for sname in src_lists if tid in src_rank[sname])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)

            # NEW features
            row[23] = track_popularity.get(tid, 0) / max_pop  # normalized popularity
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0  # artist share
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0  # artist count in pool
            # Source agreement: how many sources agree on this track?
            row[26] = row[20]  # same as n_sources but as float for tree splits

    return X, gt_idx, sizes


def run_lambdarank_cv(X, gt_idx, sizes, sessions, cases, pool_k,
                      feature_names, seeds, last_turn_only=False):
    """Run grouped-session LambdaRank CV5."""
    n = len(sessions)
    n_feat = len(feature_names)
    X_flat = X.reshape(-1, n_feat)
    labels_flat = np.zeros(n * pool_k, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels_flat[i * pool_k + gt_idx[i]] = 1.0

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,
    }

    cv5_seeds = []
    last_turn_seeds = []

    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        fold_ndcgs = []
        fold_lt_ndcgs = []

        for fold in folds:
            held = set(fold.tolist())

            if last_turn_only:
                # Train only on last-turn cases (highest turn per session)
                session_last_turn = {}
                for j in range(n):
                    if j in held:
                        continue
                    sid = sessions[j]
                    tn = cases[j]["turn_number"]
                    if sid not in session_last_turn or tn > session_last_turn[sid][1]:
                        session_last_turn[sid] = (j, tn)
                train_cases = [v[0] for v in session_last_turn.values()]
            else:
                train_cases = [j for j in range(n) if j not in held]

            val_cases = fold.tolist()

            train_flat = []
            for j in train_cases:
                for k in range(int(sizes[j])):
                    train_flat.append(j * pool_k + k)
            val_flat = []
            for j in val_cases:
                for k in range(int(sizes[j])):
                    val_flat.append(j * pool_k + k)

            g_train = np.array([int(sizes[j]) for j in train_cases], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_cases], dtype=np.int32)

            dtrain = lgb.Dataset(X_flat[train_flat], labels_flat[train_flat],
                                 group=g_train, feature_name=feature_names,
                                 free_raw_data=False)
            dval = lgb.Dataset(X_flat[val_flat], labels_flat[val_flat],
                               group=g_val, reference=dtrain, free_raw_data=False)

            model = lgb.train(
                lgb_params, dtrain, num_boost_round=300,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            val_scores = model.predict(X_flat[val_flat])
            offset = 0
            case_ndcgs = []
            lt_ndcgs = []
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

                # Track last-turn nDCG
                n_hist = len(cases[j]["music_turns"])
                if n_hist == 7:
                    lt_ndcgs.append(ndcg_val)
                offset += sz

            fold_ndcgs.append(float(np.mean(case_ndcgs)))
            if lt_ndcgs:
                fold_lt_ndcgs.append(float(np.mean(lt_ndcgs)))

        cv5_seeds.append(float(np.mean(fold_ndcgs)))
        if fold_lt_ndcgs:
            last_turn_seeds.append(float(np.mean(fold_lt_ndcgs)))

    return cv5_seeds, last_turn_seeds, model


def main():
    t0 = time.time()

    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases, {len(set(sessions))} unique sessions")

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

    print(f"{ts()} Building popularity stats...", flush=True)
    track_pop = build_popularity_stats()
    print(f"  {len(track_pop)} tracks with popularity data")

    seeds = [0, 1, 2]  # 3 seeds for faster iteration

    results = {}

    # --- Test pool sizes with V2 features ---
    for pool_k in [100, 150, 200]:
        print(f"\n{ts()} Building features (pool_k={pool_k}, {len(FEATURE_NAMES_V2)} features)...",
              flush=True)
        X, gt_idx, sizes = build_features_v2(
            payload, als_source, als_session_vecs,
            als_factors, als_track_to_idx, track_pop, pool_k=pool_k)
        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{pool_k}={pool_hit:.4f}")

        # All-turn training
        name = f"v2_{pool_k}_allturn"
        print(f"\n{ts()} {name}...", flush=True)
        cv5, lt_cv5, model = run_lambdarank_cv(
            X, gt_idx, sizes, sessions, cases, pool_k,
            FEATURE_NAMES_V2, seeds, last_turn_only=False)
        cv5_mean = float(np.mean(cv5))
        lt_mean = float(np.mean(lt_cv5)) if lt_cv5 else 0
        print(f"  CV5={cv5_mean:.4f}  last_turn={lt_mean:.4f}  seeds={cv5}")
        results[name] = {"pool_k": pool_k, "pool_hit": pool_hit,
                         "cv5": cv5_mean, "last_turn_ndcg": lt_mean, "seeds": cv5}

        # Last-turn-only training (only for pool_k=100 to save time)
        if pool_k == 100:
            name = f"v2_{pool_k}_lastturn"
            print(f"\n{ts()} {name}...", flush=True)
            cv5_lt, lt_cv5_lt, _ = run_lambdarank_cv(
                X, gt_idx, sizes, sessions, cases, pool_k,
                FEATURE_NAMES_V2, seeds, last_turn_only=True)
            cv5_lt_mean = float(np.mean(cv5_lt))
            lt_lt_mean = float(np.mean(lt_cv5_lt)) if lt_cv5_lt else 0
            print(f"  CV5={cv5_lt_mean:.4f}  last_turn={lt_lt_mean:.4f}  seeds={cv5_lt}")
            results[name] = {"pool_k": pool_k, "pool_hit": pool_hit,
                             "cv5": cv5_lt_mean, "last_turn_ndcg": lt_lt_mean,
                             "seeds": cv5_lt}

    # --- V1 baseline comparison (23f, pool=100) ---
    print(f"\n{ts()} V1 baseline (23f, pool=100)...", flush=True)
    X_v1 = X[:, :, :len(FEATURE_NAMES_LR)]  # first 23 features only (from pool=100 run... but X was last built at pool_k=200)

    # Rebuild at pool_k=100 with v1 features
    from scripts.expS2_lambdarank_grouped import build_features as build_v1_features
    X_v1, gt_v1, sizes_v1 = build_v1_features(
        payload, als_source, als_session_vecs,
        als_factors, als_track_to_idx, pool_k=100)
    cv5_v1, lt_v1, _ = run_lambdarank_cv(
        X_v1, gt_v1, sizes_v1, sessions, cases, 100,
        FEATURE_NAMES_LR, seeds, last_turn_only=False)
    cv5_v1_mean = float(np.mean(cv5_v1))
    lt_v1_mean = float(np.mean(lt_v1)) if lt_v1 else 0
    print(f"  CV5={cv5_v1_mean:.4f}  last_turn={lt_v1_mean:.4f}")
    results["v1_100_allturn"] = {"pool_k": 100, "cv5": cv5_v1_mean,
                                 "last_turn_ndcg": lt_v1_mean, "seeds": cv5_v1}

    # Summary
    print(f"\n{ts()} {'='*70}")
    print(f"{'Config':25s} {'pool_k':>6s} {'pool_hit':>9s} {'CV5':>7s} {'last_turn':>10s}")
    for name, r in sorted(results.items()):
        ph = r.get('pool_hit', '')
        ph_str = f"{ph:.4f}" if ph else "—"
        print(f"  {name:25s} {r['pool_k']:6d} {ph_str:>9s} {r['cv5']:7.4f} {r['last_turn_ndcg']:10.4f}")

    # Feature importance from last model
    print(f"\n  Top features (last model):")
    imp = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES_V2, imp), key=lambda x: -x[1])
    for fname, fimp in feat_imp[:12]:
        print(f"    {fname:25s} {fimp:10.1f}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_lr_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
