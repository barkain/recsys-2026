#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R20: Learned conditional router between V3 and B2C2_250.

Oracle showed +0.015 last-turn headroom. This trains a binary classifier
to predict when B2C2 beats V3, with V3 as default (conservative routing).
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import re
import sys
import time
from collections import Counter
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


def build_lambdarank_features(cases, payload, als_source, als_vecs, als_factors,
                               als_track_to_idx, ta, tt, track_pop, max_pop,
                               sw, pool_k):
    """Build V2 feature matrices and pools for a given source weight config."""
    n = len(cases)
    nf = len(FEATURE_NAMES_V2)
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    X = np.zeros((n, pool_k, nf), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools = []

    for i, c in enumerate(cases):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        pool = weighted_rrf(sl, sw, topk=pool_k, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r+1 for r, tid in enumerate(s)} for sn, s in sl.items()}
        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:pool_k]]
        artist_counts = Counter(a for a in pool_artists if a)

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
            for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                sr = src_rank[sn].get(tid)
                row[8+fi] = 1.0/sr if sr else 0.0
            for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                row[14+fi] = 1.0 if tid in src_rank[sn] else 0.0
            row[20] = sum(1 for sn in sl if tid in src_rank[sn])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None: row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            pop = track_pop.get(tid, 0)
            row[23] = pop / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]

    return X, gt_idx, sizes, pools


def get_per_case_ndcg(X, gt_idx, sizes, sessions, n, pool_k, seeds=(0, 1, 2)):
    """Run CV and return per-case nDCG averaged over seeds."""
    nf = len(FEATURE_NAMES_V2)
    X_flat = X.reshape(-1, nf)
    lab = np.zeros(n * pool_k, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            lab[i * pool_k + gt_idx[i]] = 1.0

    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }

    per_case = np.zeros(n, dtype=np.float64)
    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        for fold in folds:
            held = set(fold.tolist())
            train_c = [j for j in range(n) if j not in held]
            val_c = fold.tolist()
            train_flat = [j*pool_k+k for j in train_c for k in range(int(sizes[j]))]
            val_flat = [j*pool_k+k for j in val_c for k in range(int(sizes[j]))]
            g_tr = np.array([int(sizes[j]) for j in train_c], dtype=np.int32)
            g_va = np.array([int(sizes[j]) for j in val_c], dtype=np.int32)
            dt = lgb.Dataset(X_flat[train_flat], lab[train_flat], group=g_tr,
                             feature_name=FEATURE_NAMES_V2, free_raw_data=False)
            dv = lgb.Dataset(X_flat[val_flat], lab[val_flat], group=g_va,
                             reference=dt, free_raw_data=False)
            mdl = lgb.train(lgb_params, dt, num_boost_round=300, valid_sets=[dv],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
            vs = mdl.predict(X_flat[val_flat])
            offset = 0
            for j in val_c:
                sz = int(sizes[j])
                if sz == 0: continue
                sc = vs[offset:offset+sz]
                gt = gt_idx[j]
                if gt >= 0:
                    gt_s = sc[gt]
                    r = int(np.sum(sc > gt_s) + np.sum((sc == gt_s) & (np.arange(sz) < gt)))
                    per_case[j] += (1.0/np.log2(r+2) if r < 20 else 0.0) / len(seeds)
                offset += sz
    return per_case


def build_routing_features(cases, payload, als_source, als_vecs, pools_v3, pools_b2c2,
                            ta, tt, track_pop):
    """Build per-case features for the router."""
    n = len(cases)
    feats = []
    feat_names = []

    for i, c in enumerate(cases):
        played = c["music_turns"]
        n_hist = len(played)
        query = c["user_query"]
        ql = query.lower()
        query_words = set(re.findall(r'\w+', ql))

        f = {}

        # History features
        f["n_hist"] = n_hist
        f["is_hist0"] = 1.0 if n_hist == 0 else 0.0

        # Behavioral strength
        src_a = payload["src_a"][i]
        src_d = payload["src_d"][i]
        src_f = payload["src_f"][i]
        als = als_source[i]

        behavioral_in_pool = sum(1 for tid in pools_v3[i][:50]
                                 if any(tid in payload[sn][i][:200]
                                        for sn in ["src_a", "src_d", "src_f"])
                                 or tid in als[:200])
        f["behavioral_top50"] = behavioral_in_pool

        # Source agreement in top20
        top20 = pools_v3[i][:20]
        multi_source = sum(1 for tid in top20
                          if sum(1 for sn in ["src_a", "src_b", "src_c", "src_d", "src_f"]
                                 if tid in payload[sn][i][:200]) + (tid in als[:200]) >= 3)
        f["multi_source_top20"] = multi_source

        # ALS top score (proxy)
        if als_vecs[i] is not None and als:
            idx = als_track_to_idx_global.get(als[0])
            if idx is not None:
                f["als_top_score"] = float(np.dot(als_vecs[i], als_factors_global[idx]))
            else:
                f["als_top_score"] = 0.0
        else:
            f["als_top_score"] = 0.0

        # Content strength
        src_b = payload["src_b"][i]
        src_c = payload["src_c"][i]
        bm25_only_top50 = sum(1 for tid in pools_v3[i][:50]
                              if (tid in src_b[:200] or tid in src_c[:200])
                              and not any(tid in payload[sn][i][:200]
                                          for sn in ["src_a", "src_d", "src_f"])
                              and tid not in als[:200])
        f["bm25_only_top50"] = bm25_only_top50

        bc_agreement = len(set(src_b[:100]) & set(src_c[:100]))
        f["bc_agreement_100"] = bc_agreement

        # Query intent
        genre_words = {"rock", "pop", "jazz", "hip", "hop", "rap", "metal", "punk",
                       "blues", "soul", "funk", "electronic", "edm", "classical",
                       "country", "folk", "indie", "grunge", "alternative", "r&b"}
        discovery_words = {"recommend", "suggest", "something", "similar", "like",
                           "discover", "new", "different", "explore", "introduce",
                           "surprise", "mood", "vibe", "feel"}

        f["query_len"] = len(query_words)
        f["query_genre_count"] = len(query_words & genre_words)
        f["query_discovery"] = len(query_words & discovery_words)
        f["query_has_quote"] = 1.0 if "'" in query or '"' in query else 0.0
        f["query_has_by"] = 1.0 if " by " in ql else 0.0

        # Pool composition
        pool = pools_v3[i]
        if pool:
            same_artist_count = 0
            if played:
                last_a = ta.get(played[-1], "")
                if isinstance(last_a, list): last_a = last_a[0] if last_a else ""
                for tid in pool[:50]:
                    ca = ta.get(tid, "")
                    if isinstance(ca, list): ca = ca[0] if ca else ""
                    if ca and ca == last_a:
                        same_artist_count += 1
            f["same_artist_top50"] = same_artist_count
        else:
            f["same_artist_top50"] = 0

        # Pool popularity
        pool_pops = [track_pop.get(tid, 0) for tid in pool[:50]]
        f["pool_median_pop"] = float(np.median(pool_pops)) if pool_pops else 0
        f["pool_pop0_share"] = sum(1 for p in pool_pops if p == 0) / max(len(pool_pops), 1)

        # Pool difference between V3 and B2C2
        v3_set = set(pools_v3[i][:200])
        b2c2_set = set(pools_b2c2[i][:250])
        f["pool_overlap"] = len(v3_set & b2c2_set) / max(len(v3_set), 1)
        f["b2c2_unique"] = len(b2c2_set - v3_set)

        feats.append(f)

    if not feat_names:
        feat_names = list(feats[0].keys())

    X_route = np.zeros((n, len(feat_names)), dtype=np.float64)
    for i, f in enumerate(feats):
        for j, fn in enumerate(feat_names):
            X_route[i, j] = f[fn]

    return X_route, feat_names


# Globals for routing features
als_factors_global = None
als_track_to_idx_global = None


def main():
    global als_factors_global, als_track_to_idx_global
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values())

    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music": train_tracks.add(str(c["content"]).strip())
    unseen_mask = np.array([c["gt"] not in train_tracks for c in cases])

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_factors_global = als_factors
    als_track_to_idx_global = als_track_to_idx
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx: scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    sw_v3 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    sw_b2c2 = {"A": 1.0, "B": 2.0, "C": 2.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

    print(f"{ts()} Building V3 pools + features...", flush=True)
    X_v3, gt_v3, sz_v3, pools_v3 = build_lambdarank_features(
        cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
        ta, tt, track_pop, max_pop, sw_v3, 200)

    print(f"{ts()} Building B2C2_250 pools + features...", flush=True)
    X_b2, gt_b2, sz_b2, pools_b2 = build_lambdarank_features(
        cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
        ta, tt, track_pop, max_pop, sw_b2c2, 250)

    print(f"{ts()} Computing per-case nDCG (V3)...", flush=True)
    ndcg_v3 = get_per_case_ndcg(X_v3, gt_v3, sz_v3, sessions, n, 200)
    print(f"  V3 mean: {ndcg_v3.mean():.4f}")

    print(f"{ts()} Computing per-case nDCG (B2C2)...", flush=True)
    ndcg_b2 = get_per_case_ndcg(X_b2, gt_b2, sz_b2, sessions, n, 250)
    print(f"  B2C2 mean: {ndcg_b2.mean():.4f}")

    oracle = np.maximum(ndcg_v3, ndcg_b2)
    lt_mask = np.array([len(c["music_turns"]) == 7 for c in cases])
    print(f"  Oracle mean: {oracle.mean():.4f}  last_turn: {oracle[lt_mask].mean():.4f}")

    # Build routing features
    print(f"\n{ts()} Building routing features...", flush=True)
    X_route, route_feat_names = build_routing_features(
        cases, payload, als_source, als_vecs, pools_v3, pools_b2, ta, tt, track_pop)
    print(f"  {len(route_feat_names)} features: {route_feat_names}")

    # Router labels: B2C2 wins by various margins
    delta = ndcg_b2 - ndcg_v3

    # Grouped-session CV for router
    print(f"\n{ts()} === ROUTER EVALUATION ===")

    for margin_name, margin in [("any_gain", 0.0), ("margin_0.05", 0.05), ("margin_0.10", 0.10)]:
        print(f"\n  --- Label: B2C2 wins by >= {margin} ---")
        y_route = (delta >= margin).astype(np.float32)
        pos_rate = y_route.mean()
        print(f"  Positive rate: {pos_rate:.1%}")

        if pos_rate < 0.02 or pos_rate > 0.95:
            print(f"  Skipping: degenerate label distribution")
            continue

        # Router CV: train router, apply, measure nDCG of routed system
        routed_ndcgs = {
            "v3_default": [],
            "learned_router": [],
            "conservative_router": [],
        }

        for seed in [0, 1, 2]:
            folds = grouped_session_folds(sessions, seed)
            for fold in folds:
                held = set(fold.tolist())
                train_c = np.array([j for j in range(n) if j not in held])
                val_c = np.array(fold.tolist())

                # Train router
                router = lgb.LGBMClassifier(
                    n_estimators=100, num_leaves=15, learning_rate=0.05,
                    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, verbose=-1)
                router.fit(X_route[train_c], y_route[train_c])

                # Predict on val
                proba = router.predict_proba(X_route[val_c])[:, 1]

                for j_local, j_global in enumerate(val_c):
                    # V3 default
                    routed_ndcgs["v3_default"].append(ndcg_v3[j_global])

                    # Learned router: route to B2C2 if predicted positive
                    pred = proba[j_local] > 0.5
                    routed_ndcgs["learned_router"].append(
                        ndcg_b2[j_global] if pred else ndcg_v3[j_global])

                    # Conservative: route only if high confidence
                    conservative = proba[j_local] > 0.7
                    routed_ndcgs["conservative_router"].append(
                        ndcg_b2[j_global] if conservative else ndcg_v3[j_global])

        for rname, vals in routed_ndcgs.items():
            arr = np.array(vals) / 3  # average over 3 seeds (each case appears 3x)
            # Reconstruct per-case
            per_case_routed = np.zeros(n)
            counts = np.zeros(n)
            idx = 0
            for seed in [0, 1, 2]:
                folds = grouped_session_folds(sessions, seed)
                for fold in folds:
                    for j in fold.tolist():
                        per_case_routed[j] += vals[idx]
                        counts[j] += 1
                        idx += 1
            per_case_routed = per_case_routed / np.maximum(counts, 1)

            mean_ndcg = per_case_routed.mean()
            lt_ndcg = per_case_routed[lt_mask].mean()
            seen_ndcg = per_case_routed[~unseen_mask].mean()
            unseen_ndcg = per_case_routed[unseen_mask].mean()

            # Same/diff artist
            sa_vals, da_vals = [], []
            for j in range(n):
                played = cases[j]["music_turns"]
                if played:
                    la = ta.get(played[-1], "")
                    ga = ta.get(cases[j]["gt"], "")
                    if isinstance(la, list): la = la[0] if la else ""
                    if isinstance(ga, list): ga = ga[0] if ga else ""
                    if ga and la:
                        if ga == la: sa_vals.append(per_case_routed[j])
                        else: da_vals.append(per_case_routed[j])

            sa = np.mean(sa_vals) if sa_vals else 0
            da = np.mean(da_vals) if da_vals else 0

            # Route rate
            if rname != "v3_default":
                route_rate = np.mean(per_case_routed != ndcg_v3)
            else:
                route_rate = 0

            print(f"    {rname:25s}: CV={mean_ndcg:.4f}  lt={lt_ndcg:.4f}  "
                  f"sa={sa:.4f}  da={da:.4f}  "
                  f"seen={seen_ndcg:.4f}  unseen={unseen_ndcg:.4f}  "
                  f"route_rate={route_rate:.1%}")

        # Feature importance for the router
        router_full = lgb.LGBMClassifier(
            n_estimators=100, num_leaves=15, learning_rate=0.05,
            min_child_samples=30, random_state=42, verbose=-1)
        router_full.fit(X_route, y_route)
        imp = sorted(zip(route_feat_names, router_full.feature_importances_),
                     key=lambda x: -x[1])
        print(f"  Router top features:")
        for fn, iv in imp[:8]:
            print(f"    {fn}: {iv}")

    elapsed = time.time() - t0
    print(f"\n{ts()} R20 complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
