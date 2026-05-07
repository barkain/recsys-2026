#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""LambdaRank V4 sweep: pool sizes, source weights, LGB hyperparams.

All configs use V2 27-feature set, grouped-session CV5.
Key metric: last-turn nDCG (leaderboard scores last turn only).
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_als_source(cases, als_track_to_idx, als_factors, als_track_ids):
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
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
    return als_source, als_vecs


def build_features(payload, als_source, als_vecs, als_factors, als_track_to_idx,
                   track_pop, source_weights, pool_k):
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
    max_pop = max(track_pop.values()) if track_pop else 1

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        pool = weighted_rrf(src_lists, source_weights, topk=pool_k, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}

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
        sv = als_vecs[i]

        pool_artists = [ta.get(tid, "") for tid in pool[:pool_k]]
        artist_counts = Counter(a for a in pool_artists if a)

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
            for fi, sn in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sn].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sn in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sn] else 0.0
            row[20] = sum(1 for sn in src_lists if tid in src_rank[sn])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]

    return X, gt_idx, sizes


def run_cv(X, gt_idx, sizes, sessions, cases, pool_k, feature_names,
           lgb_params, seeds):
    n = len(sessions)
    n_feat = len(feature_names)
    X_flat = X.reshape(-1, n_feat)
    labels_flat = np.zeros(n * pool_k, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels_flat[i * pool_k + gt_idx[i]] = 1.0

    cv5_seeds, lt_seeds = [], []

    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        fold_ndcgs, fold_lt = [], []

        for fold in folds:
            held = set(fold.tolist())
            train_cases = [j for j in range(n) if j not in held]
            val_cases = fold.tolist()

            train_flat = [j * pool_k + k for j in train_cases for k in range(int(sizes[j]))]
            val_flat = [j * pool_k + k for j in val_cases for k in range(int(sizes[j]))]

            g_train = np.array([int(sizes[j]) for j in train_cases], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_cases], dtype=np.int32)

            dtrain = lgb.Dataset(X_flat[train_flat], labels_flat[train_flat],
                                 group=g_train, feature_name=feature_names,
                                 free_raw_data=False)
            dval = lgb.Dataset(X_flat[val_flat], labels_flat[val_flat],
                               group=g_val, reference=dtrain, free_raw_data=False)

            model = lgb.train(
                lgb_params, dtrain, num_boost_round=lgb_params.get("n_estimators", 300),
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            val_scores = model.predict(X_flat[val_flat])
            offset = 0
            case_ndcgs, lt_ndcgs = [], []
            for j in val_cases:
                sz = int(sizes[j])
                if sz == 0:
                    case_ndcgs.append(0.0)
                    continue
                sc = val_scores[offset:offset + sz]
                gt = gt_idx[j]
                if gt >= 0:
                    gt_score = sc[gt]
                    rank0 = int(np.sum(sc > gt_score) +
                                np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                    ndcg = 1.0 / np.log2(rank0 + 2) if rank0 < 20 else 0.0
                else:
                    ndcg = 0.0
                case_ndcgs.append(ndcg)
                if len(cases[j]["music_turns"]) == 7:
                    lt_ndcgs.append(ndcg)
                offset += sz

            fold_ndcgs.append(float(np.mean(case_ndcgs)))
            if lt_ndcgs:
                fold_lt.append(float(np.mean(lt_ndcgs)))

        cv5_seeds.append(float(np.mean(fold_ndcgs)))
        if fold_lt:
            lt_seeds.append(float(np.mean(fold_lt)))

    return cv5_seeds, lt_seeds


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()

    print(f"{ts()} Building ALS source...", flush=True)
    als_source, als_vecs = build_als_source(cases, als_track_to_idx, als_factors, als_track_ids)

    print(f"{ts()} Building popularity...", flush=True)
    track_pop = build_popularity_stats()

    seeds = [0, 1, 2]

    # Default LGB params
    base_lgb = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "random_state": 42, "force_col_wise": True,
    }
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

    results = {}

    # ---- 1. Pool size sweep ----
    for pool_k in [200, 250, 300]:
        name = f"pool_{pool_k}"
        print(f"\n{ts()} {name}...", flush=True)
        X, gt_idx, sizes = build_features(payload, als_source, als_vecs,
                                          als_factors, als_track_to_idx,
                                          track_pop, base_weights, pool_k)
        ph = float(np.mean(gt_idx >= 0))
        cv5, lt = run_cv(X, gt_idx, sizes, sessions, cases, pool_k,
                         FEATURE_NAMES_V2, base_lgb, seeds)
        cv5m, ltm = np.mean(cv5), np.mean(lt) if lt else 0
        print(f"  pool_hit={ph:.4f}  CV5={cv5m:.4f}  last_turn={ltm:.4f}")
        results[name] = {"pool_k": pool_k, "pool_hit": ph, "cv5": cv5m,
                         "lt": ltm, "seeds": cv5}

    # ---- 2. Source weight variations (pool=200) ----
    sw_configs = {
        "sw_als2":    {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 2.0},
        "sw_als05":   {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 0.5},
        "sw_noD":     {"A": 1.0, "B": 1.0, "C": 1.0, "F": 1.0, "ALS": 1.0},
        "sw_D1":      {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "F": 1.0, "ALS": 1.0},
        "sw_A2":      {"A": 2.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0},
    }
    for name, sw in sw_configs.items():
        print(f"\n{ts()} {name}...", flush=True)
        X, gt_idx, sizes = build_features(payload, als_source, als_vecs,
                                          als_factors, als_track_to_idx,
                                          track_pop, sw, 200)
        ph = float(np.mean(gt_idx >= 0))
        cv5, lt = run_cv(X, gt_idx, sizes, sessions, cases, 200,
                         FEATURE_NAMES_V2, base_lgb, seeds)
        cv5m, ltm = np.mean(cv5), np.mean(lt) if lt else 0
        print(f"  pool_hit={ph:.4f}  CV5={cv5m:.4f}  last_turn={ltm:.4f}")
        results[name] = {"pool_k": 200, "pool_hit": ph, "cv5": cv5m,
                         "lt": ltm, "weights": sw, "seeds": cv5}

    # ---- 3. LGB hyperparams (pool=200, base weights) ----
    X200, gt200, sizes200 = build_features(payload, als_source, als_vecs,
                                           als_factors, als_track_to_idx,
                                           track_pop, base_weights, 200)

    lgb_configs = {
        "lgb_leaves63": {**base_lgb, "num_leaves": 63},
        "lgb_leaves15": {**base_lgb, "num_leaves": 15},
        "lgb_lr01":     {**base_lgb, "learning_rate": 0.01, "n_estimators": 600},
        "lgb_lr1":      {**base_lgb, "learning_rate": 0.1},
        "lgb_500rnd":   {**base_lgb, "n_estimators": 500},
    }
    for name, params in lgb_configs.items():
        print(f"\n{ts()} {name}...", flush=True)
        cv5, lt = run_cv(X200, gt200, sizes200, sessions, cases, 200,
                         FEATURE_NAMES_V2, params, seeds)
        cv5m, ltm = np.mean(cv5), np.mean(lt) if lt else 0
        print(f"  CV5={cv5m:.4f}  last_turn={ltm:.4f}")
        results[name] = {"pool_k": 200, "cv5": cv5m, "lt": ltm,
                         "params": {k: v for k, v in params.items() if k != "verbose"},
                         "seeds": cv5}

    # Summary
    baseline = results.get("pool_200", {})
    base_cv5 = baseline.get("cv5", 0)
    base_lt = baseline.get("lt", 0)
    print(f"\n{ts()} {'='*75}")
    print(f"{'Config':20s} {'pool_k':>6s} {'pool_hit':>9s} {'CV5':>7s} {'Δ CV5':>7s} "
          f"{'last_turn':>10s} {'Δ lt':>7s}")
    for name, r in sorted(results.items()):
        ph = r.get("pool_hit", "")
        ph_s = f"{ph:.4f}" if ph else "—"
        dcv = r["cv5"] - base_cv5
        dlt = r["lt"] - base_lt
        print(f"  {name:20s} {r['pool_k']:6d} {ph_s:>9s} {r['cv5']:7.4f} {dcv:+7.4f} "
              f"{r['lt']:10.4f} {dlt:+7.4f}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_lr_v4_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
