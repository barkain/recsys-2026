#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R19: BM25 unseen-pool admission diagnostic.

R18 showed unseen GTs convert at 93% when in pool, but only 46% make it in.
This experiment measures BM25 depth curves and tests query/pool admission policies.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import bm25s
import lightgbm as lgb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens
from offline_retrieval_sweep import CachedBM25, meta_text, load_track_metadata, query_parts

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]

    # Train track set
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    unseen_mask = np.array([c["gt"] not in train_tracks for c in cases])
    print(f"  Unseen GTs: {unseen_mask.sum()}/{n} ({unseen_mask.mean():.1%})")

    # Load BM25 and metadata
    print(f"{ts()} Loading BM25 + metadata...", flush=True)
    bm25 = CachedBM25()
    metadata = load_track_metadata()

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    # Build ALS + V3 pools
    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
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

    v3_pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        v3_pools.append(set(weighted_rrf(sl, SOURCE_WEIGHTS, topk=200, k=RRF_K)))

    # =========================================================
    # STAGE 1: BM25 depth curves
    # =========================================================
    print(f"\n{ts()} === STAGE 1: BM25 DEPTH CURVES ===")

    # Retrieve at multiple depths
    depths = [50, 100, 200, 500, 1000, 2000]
    max_depth = max(depths)

    # Build B and C queries (same as V3)
    queries_b, queries_c = [], []
    for c in cases:
        history = c["history"]
        user_query = c["user_query"]
        q_b = " ".join(query_parts(history, user_query, metadata, "last_music_meta"))
        q_c = " ".join(query_parts(history, user_query, metadata, "full"))
        queries_b.append(q_b or user_query)
        queries_c.append(q_c or user_query)

    print(f"{ts()} Retrieving B@{max_depth} and C@{max_depth}...", flush=True)
    src_b_deep = bm25.retrieve_batch(queries_b, topk=max_depth)
    src_c_deep = bm25.retrieve_batch(queries_c, topk=max_depth)

    print(f"\n  BM25 depth curves (unseen GTs only, n={unseen_mask.sum()}):")
    print(f"  {'depth':>6s}  {'B_hit':>7s}  {'C_hit':>7s}  {'B∪C':>7s}  {'new_vs_V3':>10s}")
    for d in depths:
        b_hit = sum(1 for i in range(n) if unseen_mask[i] and cases[i]["gt"] in src_b_deep[i][:d])
        c_hit = sum(1 for i in range(n) if unseen_mask[i] and cases[i]["gt"] in src_c_deep[i][:d])
        bc_hit = sum(1 for i in range(n) if unseen_mask[i] and
                     (cases[i]["gt"] in src_b_deep[i][:d] or cases[i]["gt"] in src_c_deep[i][:d]))
        new_vs_v3 = sum(1 for i in range(n) if unseen_mask[i] and
                        (cases[i]["gt"] in src_b_deep[i][:d] or cases[i]["gt"] in src_c_deep[i][:d]) and
                        cases[i]["gt"] not in v3_pools[i])
        nu = unseen_mask.sum()
        print(f"  {d:6d}  {b_hit:5d} ({b_hit/nu:5.1%})  {c_hit:5d} ({c_hit/nu:5.1%})  "
              f"{bc_hit:5d} ({bc_hit/nu:5.1%})  {new_vs_v3:5d}")

    # Categorize unseen misses
    cat_shallow = 0  # in BM25@200
    cat_deep = 0     # in BM25@2000 but not @200
    cat_unreachable = 0  # not in BM25@2000
    for i in range(n):
        if not unseen_mask[i]: continue
        gt = cases[i]["gt"]
        in_200 = gt in src_b_deep[i][:200] or gt in src_c_deep[i][:200]
        in_2000 = gt in src_b_deep[i][:2000] or gt in src_c_deep[i][:2000]
        if in_200: cat_shallow += 1
        elif in_2000: cat_deep += 1
        else: cat_unreachable += 1

    nu = int(unseen_mask.sum())
    print(f"\n  Unseen GT categories:")
    print(f"    Shallow (BM25@200):       {cat_shallow}/{nu} ({cat_shallow/nu:.1%})")
    print(f"    Deep (BM25 201-2000):     {cat_deep}/{nu} ({cat_deep/nu:.1%})")
    print(f"    Unreachable (>2000):      {cat_unreachable}/{nu} ({cat_unreachable/nu:.1%})")

    # =========================================================
    # STAGE 2: Query construction sweep
    # =========================================================
    print(f"\n{ts()} === STAGE 2: QUERY CONSTRUCTION SWEEP ===")

    query_variants = {}

    # V1: current query only
    query_variants["current_only"] = [c["user_query"] for c in cases]

    # V2: all user utterances
    q_all_user = []
    for c in cases:
        user_turns = [str(h["content"]) for h in c["history"] if h["role"] == "user"]
        q_all_user.append(" ".join(user_turns + [c["user_query"]]))
    query_variants["all_user"] = q_all_user

    # V3: last user + last music metadata (= source B query)
    query_variants["last_music_meta"] = queries_b

    # V4: full conversation (= source C query)
    query_variants["full_conv"] = queries_c

    # V5: recent played track metadata only
    q_recent_meta = []
    for c in cases:
        played = c["music_turns"]
        parts = []
        for tid in played[-3:]:
            parts.append(meta_text(tid, metadata, include_track_name=True))
        q_recent_meta.append(" ".join(parts) if parts else c["user_query"])
    query_variants["recent_meta"] = q_recent_meta

    # V6: recent played artist+tags only
    q_artist_tags = []
    for c in cases:
        played = c["music_turns"]
        parts = []
        for tid in played[-5:]:
            a = ta.get(tid, "")
            if isinstance(a, list): a = " ".join(a)
            t = tt.get(tid, set())
            parts.append(f"{a} {' '.join(t)}")
        q_artist_tags.append(" ".join(parts) if parts else c["user_query"])
    query_variants["artist_tags"] = q_artist_tags

    # V7: current query + recent tags
    q_query_tags = []
    for c in cases:
        played = c["music_turns"]
        tag_parts = []
        for tid in played[-3:]:
            t = tt.get(tid, set())
            tag_parts.extend(t)
        q_query_tags.append(c["user_query"] + " " + " ".join(tag_parts))
    query_variants["query_plus_tags"] = q_query_tags

    # V8: current query + recent artists
    q_query_artists = []
    for c in cases:
        played = c["music_turns"]
        artists = []
        for tid in played[-5:]:
            a = ta.get(tid, "")
            if isinstance(a, list): artists.extend(a)
            elif a: artists.append(a)
        q_query_artists.append(c["user_query"] + " " + " ".join(artists))
    query_variants["query_plus_artists"] = q_query_artists

    # V9: current query repeated 3x (boost current intent)
    query_variants["query_3x"] = [c["user_query"] + " " + c["user_query"] + " " + c["user_query"]
                                   for c in cases]

    # Evaluate each variant at multiple depths
    print(f"\n  {'variant':25s} {'@200':>8s} {'@500':>8s} {'@1000':>8s} {'@2000':>8s} {'new@500':>8s}")
    best_variant = None
    best_new = 0

    for vname, queries in query_variants.items():
        results = bm25.retrieve_batch(queries, topk=2000)

        hits = {}
        for d in [200, 500, 1000, 2000]:
            hits[d] = sum(1 for i in range(n) if unseen_mask[i] and cases[i]["gt"] in results[i][:d])

        new_500 = sum(1 for i in range(n) if unseen_mask[i] and
                      cases[i]["gt"] in results[i][:500] and cases[i]["gt"] not in v3_pools[i])

        print(f"  {vname:25s} {hits[200]:5d} ({hits[200]/nu:5.1%}) "
              f"{hits[500]:5d} ({hits[500]/nu:5.1%}) "
              f"{hits[1000]:5d} ({hits[1000]/nu:5.1%}) "
              f"{hits[2000]:5d} ({hits[2000]/nu:5.1%}) "
              f"{new_500:5d}")

        if new_500 > best_new:
            best_new = new_500
            best_variant = vname

    print(f"\n  Best for new unseen@500: {best_variant} ({best_new} new)")

    # =========================================================
    # STAGE 3: Pool admission policies with LambdaRank
    # =========================================================
    print(f"\n{ts()} === STAGE 3: POOL ADMISSION POLICIES ===")

    ttl_p = payload["track_title_toks"]
    tat_p = payload["track_artist_toks"]
    tmt_p = payload["track_meta_toks"]

    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }

    # Pool configs to test
    pool_configs = [
        {"name": "v3_baseline_200", "pool_k": 200, "bm25_tail": 0, "bm25_boost": 1.0},
        {"name": "v3+bm25_tail_20", "pool_k": 220, "bm25_tail": 20, "bm25_boost": 1.0},
        {"name": "v3+bm25_tail_40", "pool_k": 240, "bm25_tail": 40, "bm25_boost": 1.0},
        {"name": "v3+bm25_tail_60", "pool_k": 260, "bm25_tail": 60, "bm25_boost": 1.0},
        {"name": "v3_b2c2_200", "pool_k": 200, "bm25_tail": 0, "bm25_boost": 2.0},
        {"name": "v3_b3c3_200", "pool_k": 200, "bm25_tail": 0, "bm25_boost": 3.0},
        {"name": "v3_b2c2_250", "pool_k": 250, "bm25_tail": 0, "bm25_boost": 2.0},
    ]

    for pcfg in pool_configs:
        pname = pcfg["name"]
        pk = pcfg["pool_k"]
        bm25_tail = pcfg["bm25_tail"]
        bm25_boost = pcfg["bm25_boost"]
        nf = len(FEATURE_NAMES_V2)

        print(f"\n{ts()} {pname}...", flush=True)

        X = np.zeros((n, pk, nf), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        pool_sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            src_lists = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                         "C": payload["src_c"][i], "D": payload["src_d"][i],
                         "F": payload["src_f"][i], "ALS": als_source[i]}

            sw = dict(SOURCE_WEIGHTS)
            if bm25_boost != 1.0:
                sw["B"] = bm25_boost
                sw["C"] = bm25_boost

            base_pool_k = pk - bm25_tail
            pool = weighted_rrf(src_lists, sw, topk=base_pool_k, k=RRF_K)

            if bm25_tail > 0:
                pool_set = set(pool)
                tail_candidates = []
                for tid in src_b_deep[i][:500] + src_c_deep[i][:500]:
                    if tid not in pool_set and tid not in set(tail_candidates):
                        tail_candidates.append(tid)
                    if len(tail_candidates) >= bm25_tail:
                        break
                pool = list(pool) + tail_candidates[:bm25_tail]

            pool = pool[:pk]
            pool_sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])

            src_rank = {sn: {tid: r+1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}
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
            pool_artists = [ta.get(tid, "") for tid in pool[:pk]]
            artist_counts = Counter(a for a in pool_artists if a)

            for rank, tid in enumerate(pool[:pk], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank-1]
                row[0] = 1.0/rank
                row[1] = 1.0 if ca and ca == l_artist else 0.0
                if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
                row[3] = float(len(tat_p.get(tid, set()) & now_tok))
                row[4] = float(len(ttl_p.get(tid, set()) & now_tok))
                row[5] = float(len(tmt_p.get(tid, set()) & all_tok))
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
                row[20] = sum(1 for sn in src_lists if tid in src_rank[sn])
                if sv is not None:
                    aidx = als_track_to_idx.get(tid)
                    if aidx is not None: row[21] = float(np.dot(sv, als_factors[aidx]))
                row[22] = float(n_hist)
                pop = track_pop.get(tid, 0)
                row[23] = pop / max_pop
                row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
                row[25] = float(artist_counts.get(ca, 0)) if ca else 0
                row[26] = row[20]

        pool_hit = float(np.mean(gt_idx >= 0))
        unseen_pool_hit = float(np.mean([gt_idx[i] >= 0 for i in range(n) if unseen_mask[i]]))
        seen_pool_hit = float(np.mean([gt_idx[i] >= 0 for i in range(n) if not unseen_mask[i]]))
        print(f"  pool_hit={pool_hit:.4f}  unseen={unseen_pool_hit:.4f}  seen={seen_pool_hit:.4f}")

        # LambdaRank CV
        X_flat = X.reshape(-1, nf)
        lab = np.zeros(n * pk, dtype=np.float32)
        for i in range(n):
            if gt_idx[i] >= 0:
                lab[i * pk + gt_idx[i]] = 1.0

        cv5_seeds, lt_seeds, sa_vals, da_vals = [], [], [], []
        unseen_ndcgs, seen_ndcgs = [], []

        for seed in [0, 1, 2]:
            folds = grouped_session_folds(sessions, seed)
            fold_ndcgs = []
            for fold in folds:
                held = set(fold.tolist())
                train_c = [j for j in range(n) if j not in held]
                val_c = fold.tolist()
                train_flat = [j*pk+k for j in train_c for k in range(int(pool_sizes[j]))]
                val_flat = [j*pk+k for j in val_c for k in range(int(pool_sizes[j]))]
                g_train = np.array([int(pool_sizes[j]) for j in train_c], dtype=np.int32)
                g_val = np.array([int(pool_sizes[j]) for j in val_c], dtype=np.int32)
                dtrain = lgb.Dataset(X_flat[train_flat], lab[train_flat],
                                     group=g_train, feature_name=FEATURE_NAMES_V2, free_raw_data=False)
                dval = lgb.Dataset(X_flat[val_flat], lab[val_flat],
                                   group=g_val, reference=dtrain, free_raw_data=False)
                mdl = lgb.train(lgb_params, dtrain, num_boost_round=300,
                                valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
                val_scores = mdl.predict(X_flat[val_flat])
                offset = 0
                case_ndcgs = []
                for j in val_c:
                    sz = int(pool_sizes[j])
                    if sz == 0: case_ndcgs.append(0.0); continue
                    sc = val_scores[offset:offset+sz]
                    gt = gt_idx[j]
                    if gt >= 0:
                        gt_score = sc[gt]
                        rank0 = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                        ndcg = 1.0/np.log2(rank0+2) if rank0 < 20 else 0.0
                    else:
                        ndcg = 0.0
                    case_ndcgs.append(ndcg)
                    played = cases[j]["music_turns"]
                    gt_tid = cases[j]["gt"]
                    if len(played) == 7: lt_seeds.append(ndcg)
                    if played:
                        la_v = ta.get(played[-1], "")
                        ga_v = ta.get(gt_tid, "")
                        if isinstance(la_v, list): la_v = la_v[0] if la_v else ""
                        if isinstance(ga_v, list): ga_v = ga_v[0] if ga_v else ""
                        if ga_v and la_v:
                            if ga_v == la_v: sa_vals.append(ndcg)
                            else: da_vals.append(ndcg)
                    if unseen_mask[j]: unseen_ndcgs.append(ndcg)
                    else: seen_ndcgs.append(ndcg)
                    offset += sz
                fold_ndcgs.append(float(np.mean(case_ndcgs)))
            cv5_seeds.append(float(np.mean(fold_ndcgs)))

        cv5 = float(np.mean(cv5_seeds))
        lt = float(np.mean(lt_seeds)) if lt_seeds else 0
        sa = float(np.mean(sa_vals)) if sa_vals else 0
        da = float(np.mean(da_vals)) if da_vals else 0
        unseen = float(np.mean(unseen_ndcgs)) if unseen_ndcgs else 0
        seen = float(np.mean(seen_ndcgs)) if seen_ndcgs else 0

        print(f"  CV5={cv5:.4f}  last_turn={lt:.4f}  sa={sa:.4f}  da={da:.4f}")
        print(f"  seen_nDCG={seen:.4f}  unseen_nDCG={unseen:.4f}")

        lt_seeds, sa_vals, da_vals, unseen_ndcgs, seen_ndcgs = [], [], [], [], []

    elapsed = time.time() - t0
    print(f"\n{ts()} R19 complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
