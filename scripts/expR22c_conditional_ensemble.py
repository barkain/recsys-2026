#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R22c: Conditional ensemble — find the R22b config that helps last_turn.

Tests (all with LambdaRank CV5, focus on hist_7/last_turn):
1. R21 baseline (control)
2. R22b as rerank feature only (not in RRF pool, but rank_inv/presence features)
3. R21+R22b w=0.5 in RRF
4. R21+R22b w=0.3 in RRF
5. R21+R22b w=1.0 but R22b only admitted if candidate in >=1 other source
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R22B_LISTS = REPO / "cache" / "r22b" / "dev_r22b_lists.json"
RRF_K = 20
POOL_K = 300

FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEATURE_NAMES_ENS = FEATURE_NAMES_R21 + ["r22b_rank_inv", "r22b_presence"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_features_flexible(cases, payload, als_source, als_vecs, als_factors,
                            als_track_to_idx, track_pop, r21_source, r22b_source,
                            source_weights, feature_names, r22b_pool_mode="source"):
    """
    r22b_pool_mode:
      "source" — R22b in RRF as normal source
      "feature_only" — R22b NOT in RRF, but r22b_rank_inv/presence features computed
      "gated" — R22b in RRF only for candidates also in another source
    """
    n = len(cases)
    n_feat = len(feature_names)
    has_r22b_features = "r22b_rank_inv" in feature_names

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }

        r22b_list_i = r22b_source[i] if r22b_source else []

        if r22b_pool_mode == "source" and "R22b" in source_weights:
            src_lists["R22b"] = r22b_list_i
        elif r22b_pool_mode == "gated" and "R22b" in source_weights:
            other_tids = set()
            for sname in ["A", "B", "C", "D", "F", "ALS", "R21"]:
                other_tids.update(src_lists.get(sname, []))
            gated_r22b = [tid for tid in r22b_list_i if tid in other_tids]
            src_lists["R22b"] = gated_r22b

        pool = weighted_rrf(src_lists, source_weights, topk=POOL_K, k=RRF_K)
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
        sv = als_vecs[i]

        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        r22b_rank_map = {tid: r + 1 for r, tid in enumerate(r22b_list_i[:300])}

        for rank, tid in enumerate(pool[:POOL_K], start=1):
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
            row[20] = sum(1 for sname in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sname, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0
            if has_r22b_features:
                row[29] = 1.0 / r22b_rank_map[tid] if tid in r22b_rank_map else 0.0
                row[30] = 1.0 if tid in r22b_rank_map else 0.0

    return X, gt_idx, sizes


def run_cv_sliced(X, gt_idx, sizes, cases, sessions, n_feat, seed=0):
    n = X.shape[0]
    folds = grouped_session_folds(sessions, seed=seed)
    case_ndcg = np.zeros(n)

    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        train_idx = [j for j in range(n) if j not in val_idx]
        val_list = sorted(val_idx)

        X_flat_train, y_train, g_train = [], [], []
        X_flat_val, y_val, g_val = [], [], []

        for i in train_idx:
            s = int(sizes[i])
            for k in range(s):
                X_flat_train.append(X[i, k])
                y_train.append(1.0 if k == gt_idx[i] else 0.0)
            g_train.append(s)

        for i in val_list:
            s = int(sizes[i])
            for k in range(s):
                X_flat_val.append(X[i, k])
                y_val.append(1.0 if k == gt_idx[i] else 0.0)
            g_val.append(s)

        ds_tr = lgb.Dataset(np.array(X_flat_train), label=np.array(y_train), group=g_train)
        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "eval_at": [20], "num_leaves": 31, "learning_rate": 0.05,
            "min_data_in_leaf": 10, "verbose": -1, "seed": seed,
        }
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          callbacks=[lgb.log_evaluation(0)])

        preds_va = model.predict(np.array(X_flat_val))
        offset = 0
        for i in val_list:
            s = int(sizes[i])
            scores = preds_va[offset:offset + s]
            offset += s
            if gt_idx[i] < 0:
                continue
            ranked = np.argsort(-scores)
            gt_pos = np.where(ranked == gt_idx[i])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[i] = 1.0 / np.log2(gt_pos[0] + 2)

    slices = {}
    for depth in range(8):
        idx = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
        if idx:
            slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx]))

    return {
        "cv5": float(np.mean(case_ndcg)),
        "last_turn": slices.get("hist_7", 0),
        "slices": slices,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R22c: Conditional R22b ensemble sweep")
    print(f"{'='*60}")

    print(f"\n{ts()} Loading data...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R22B_LISTS) as f:
        r22b_source = json.load(f)

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()

    configs = [
        ("1_R21_baseline", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
         FEATURE_NAMES_R21, "none", False),
        ("2_feature_only", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
         FEATURE_NAMES_ENS, "feature_only", True),
        ("3_ens_w0.5", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 0.5},
         FEATURE_NAMES_ENS, "source", True),
        ("4_ens_w0.3", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 0.3},
         FEATURE_NAMES_ENS, "source", True),
        ("5_ens_gated", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 1.0},
         FEATURE_NAMES_ENS, "gated", True),
    ]

    results_all = {}
    for name, sw, feat_names, pool_mode, use_r22b in configs:
        print(f"\n{ts()} {name} ({len(feat_names)}f, pool_mode={pool_mode})")
        X, gt, sz = build_features_flexible(
            cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
            track_pop, r21_source, r22b_source if use_r22b else [],
            sw, feat_names, r22b_pool_mode=pool_mode)
        ph = float(np.mean(gt >= 0))
        r = run_cv_sliced(X, gt, sz, cases, sessions, len(feat_names))
        r["pool_hit"] = ph
        results_all[name] = r
        print(f"  pool_hit={ph:.4f}  CV5={r['cv5']:.5f}  last_turn={r['last_turn']:.5f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY (focus: last_turn / hist_7)")
    print(f"{'='*60}")
    print(f"  {'Config':<20} {'pool_hit':>9} {'CV5':>9} {'last_turn':>10} {'Δ_last':>8}")
    print(f"  {'-'*58}")
    baseline_last = results_all["1_R21_baseline"]["last_turn"]
    for name, r in results_all.items():
        delta = r["last_turn"] - baseline_last
        print(f"  {name:<20} {r['pool_hit']:>9.4f} {r['cv5']:>9.5f} {r['last_turn']:>10.5f} {delta:>+8.5f}")

    # Full slice comparison
    print(f"\nSlice detail (Δ vs baseline):")
    print(f"  {'Config':<20}", end="")
    for d in range(8):
        print(f" {'h'+str(d):>7}", end="")
    print()
    for name, r in results_all.items():
        if name == "1_R21_baseline":
            continue
        print(f"  {name:<20}", end="")
        for d in range(8):
            base = results_all["1_R21_baseline"]["slices"].get(f"hist_{d}", 0)
            val = r["slices"].get(f"hist_{d}", 0)
            print(f" {val-base:>+7.4f}", end="")
        print()

    out = REPO / "exp" / "eval" / "r22c_conditional_ensemble.json"
    with open(out, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n{ts()} Saved: {out}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
