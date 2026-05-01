#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R18: Unseen/content-aware ranking — teach LambdaRank when to trust BM25.

Key finding: 39.4% of dev GTs never appear in training. BM25 finds 46% of them,
but LambdaRank under-promotes BM25-only candidates. This experiment adds
content-awareness features and measures whether conversion improves.

Stage 1: Error decomposition by seen/unseen GT status
Stage 2: Content-aware features + LambdaRank CV5
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import pickle
import re
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
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

DISCOVERY_WORDS = {
    "recommend", "suggestion", "suggest", "something", "similar", "like",
    "discover", "new", "different", "explore", "introduce", "surprise",
    "mood", "vibe", "feel", "chill", "upbeat", "energetic", "relaxing",
    "genre", "style", "type", "kind",
}

FEATURE_NAMES_R18 = FEATURE_NAMES_V2 + [
    "is_bm25_only",        # in B or C but not A/D/F/ALS
    "bm25_max_rank_inv",   # best BM25 rank (max of B, C reciprocal)
    "behavioral_source_count",  # how many of A/D/F/ALS have this
    "content_source_count",     # how many of B/C have this
    "query_has_artist",    # user query mentions an artist name
    "query_has_genre",     # user query has genre/tag words
    "query_is_discovery",  # query has discovery/mood/explore language
    "query_has_title",     # query mentions a specific track title
    "title_bm25_match",    # track title tokens overlap with query
    "artist_in_query",     # this candidate's artist appears in query
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def detect_query_intent(query, all_artist_names=None):
    """Classify query intent from text."""
    ql = query.lower()
    words = set(re.findall(r'\w+', ql))

    has_genre = bool(words & {"rock", "pop", "jazz", "hip", "hop", "rap", "metal",
                              "punk", "blues", "soul", "funk", "electronic", "edm",
                              "classical", "country", "folk", "indie", "r&b", "rnb",
                              "reggae", "latin", "grunge", "alternative"})
    has_discovery = bool(words & DISCOVERY_WORDS)

    # Check for quoted titles
    has_title = bool(re.search(r"['\"]([^'\"]+)['\"]", query))

    # Simple artist detection from quotes or "by X"
    has_artist = bool(re.search(r"\bby\s+[A-Z]", query))

    return {
        "has_genre": has_genre,
        "has_discovery": has_discovery,
        "has_title": has_title,
        "has_artist": has_artist,
    }


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
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    # Build train track set
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())
    print(f"  Train tracks: {len(train_tracks)}")

    # Build ALS
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

    # Build pools
    print(f"{ts()} Building V3 pools...", flush=True)
    pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        pools.append(weighted_rrf(sl, SOURCE_WEIGHTS, topk=200, k=RRF_K))

    # =========================================================
    # STAGE 1: Error decomposition
    # =========================================================
    print(f"\n{ts()} === STAGE 1: ERROR DECOMPOSITION ===")

    pool_k = 200
    n_feat_v2 = len(FEATURE_NAMES_V2)

    # Build V2 features and get LambdaRank ranks for analysis
    X_v2 = np.zeros((n, pool_k, n_feat_v2), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    # Track source membership for each GT
    gt_source_info = []

    for i, c in enumerate(cases):
        pool = pools[i][:pool_k]
        sizes[i] = len(pool)
        gt = c["gt"]
        if gt in pool:
            gt_idx[i] = pool.index(gt)

        src_lists = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                     "C": payload["src_c"][i], "D": payload["src_d"][i],
                     "F": payload["src_f"][i], "ALS": als_source[i]}
        src_rank = {sn: {tid: r+1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}

        # GT source membership
        gt_in = {sn: gt in src_rank[sn] for sn in src_lists}
        bm25_only = (gt_in["B"] or gt_in["C"]) and not any(gt_in[s] for s in ["A", "D", "F", "ALS"])
        behavioral_only = any(gt_in[s] for s in ["A", "D", "F", "ALS"]) and not (gt_in["B"] or gt_in["C"])
        gt_source_info.append({
            "in_sources": gt_in,
            "bm25_only": bm25_only,
            "behavioral_only": behavioral_only,
            "is_unseen": gt not in train_tracks,
            "in_pool": gt_idx[i] >= 0,
        })

        # Build V2 features (same as V3 baseline)
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
        pool_artists = [ta.get(tid, "") for tid in pool]
        artist_counts = Counter(a for a in pool_artists if a)

        for rank, tid in enumerate(pool[:pool_k], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X_v2[i, rank-1]
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

    # Error decomposition stats
    seen_in_pool = sum(1 for g in gt_source_info if not g["is_unseen"] and g["in_pool"])
    seen_total = sum(1 for g in gt_source_info if not g["is_unseen"])
    unseen_in_pool = sum(1 for g in gt_source_info if g["is_unseen"] and g["in_pool"])
    unseen_total = sum(1 for g in gt_source_info if g["is_unseen"])

    print(f"  Seen GTs:   {seen_total} total, {seen_in_pool} in pool ({seen_in_pool/seen_total:.1%})")
    print(f"  Unseen GTs: {unseen_total} total, {unseen_in_pool} in pool ({unseen_in_pool/unseen_total:.1%})")

    # BM25-only GT stats
    bm25_only_in_pool = sum(1 for g in gt_source_info if g["bm25_only"] and g["in_pool"])
    bm25_only_total = sum(1 for g in gt_source_info if g["bm25_only"])
    print(f"  BM25-only GTs: {bm25_only_total} total (in any BM25 source)")
    print(f"    In pool: {bm25_only_in_pool}")

    # Now train V2 LambdaRank to get per-case ranks for decomposition
    print(f"\n{ts()} Training V2 LambdaRank for rank analysis...", flush=True)
    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }

    X_flat = X_v2.reshape(-1, n_feat_v2)
    labels = np.zeros(n * pool_k, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * pool_k + gt_idx[i]] = 1.0

    # Get per-case LambdaRank ranks via CV
    lr_ranks = np.full(n, -1, dtype=np.int64)  # -1 = not in pool
    for seed in [42]:
        folds = grouped_session_folds(sessions, seed)
        for fold in folds:
            held = set(fold.tolist())
            train_c = [j for j in range(n) if j not in held]
            val_c = fold.tolist()
            train_flat = [j*pool_k+k for j in train_c for k in range(int(sizes[j]))]
            val_flat = [j*pool_k+k for j in val_c for k in range(int(sizes[j]))]
            g_train = np.array([int(sizes[j]) for j in train_c], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_c], dtype=np.int32)
            dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat],
                                 group=g_train, feature_name=FEATURE_NAMES_V2, free_raw_data=False)
            dval = lgb.Dataset(X_flat[val_flat], labels[val_flat],
                               group=g_val, reference=dtrain, free_raw_data=False)
            model = lgb.train(lgb_params, dtrain, num_boost_round=300,
                              valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
            val_scores = model.predict(X_flat[val_flat])
            offset = 0
            for j in val_c:
                sz = int(sizes[j])
                if sz == 0: offset += 0; continue
                sc = val_scores[offset:offset+sz]
                gt = gt_idx[j]
                if gt >= 0:
                    gt_score = sc[gt]
                    rank = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                    lr_ranks[j] = rank
                offset += sz

    # Decomposition report
    print(f"\n  LambdaRank conversion rates:")

    for label, pred in [("Seen GTs", lambda g: not g["is_unseen"]),
                        ("Unseen GTs", lambda g: g["is_unseen"]),
                        ("BM25-only GTs", lambda g: g["bm25_only"]),
                        ("Behavioral-only GTs", lambda g: g["behavioral_only"]),
                        ("Multi-source GTs", lambda g: not g["bm25_only"] and not g["behavioral_only"])]:
        mask = [i for i in range(n) if pred(gt_source_info[i])]
        in_pool = [i for i in mask if gt_idx[i] >= 0]
        in_top20 = [i for i in in_pool if lr_ranks[i] < 20]
        in_top5 = [i for i in in_pool if lr_ranks[i] < 5]
        median_rank = np.median([lr_ranks[i] for i in in_pool if lr_ranks[i] >= 0]) if in_pool else -1

        print(f"    {label}: total={len(mask)}  in_pool={len(in_pool)} ({len(in_pool)/len(mask):.1%})  "
              f"→top20={len(in_top20)} ({len(in_top20)/max(len(in_pool),1):.1%} conversion)  "
              f"→top5={len(in_top5)}  median_rank={median_rank:.0f}")

    # =========================================================
    # STAGE 2: Content-aware features + LambdaRank
    # =========================================================
    print(f"\n{ts()} === STAGE 2: CONTENT-AWARE RANKING ===")

    n_feat_r18 = len(FEATURE_NAMES_R18)

    # Pre-compute query intent for all cases
    query_intents = [detect_query_intent(c["user_query"]) for c in cases]

    # Build R18 feature matrix (V2 + 10 new features)
    X_r18 = np.zeros((n, pool_k, n_feat_r18), dtype=np.float64)
    X_r18[:, :, :n_feat_v2] = X_v2  # copy V2 features

    for i, c in enumerate(cases):
        pool = pools[i][:pool_k]
        src_lists = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                     "C": payload["src_c"][i], "D": payload["src_d"][i],
                     "F": payload["src_f"][i], "ALS": als_source[i]}
        src_rank = {sn: {tid: r+1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}

        intent = query_intents[i]
        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        query_toks = tokens(c["user_query"])

        for rank, tid in enumerate(pool[:pool_k], start=1):
            row = X_r18[i, rank-1]

            in_b = tid in src_rank["B"]
            in_c = tid in src_rank["C"]
            in_behavioral = any(tid in src_rank[s] for s in ["A", "D", "F", "ALS"])

            # is_bm25_only
            row[27] = 1.0 if (in_b or in_c) and not in_behavioral else 0.0

            # bm25_max_rank_inv
            b_rank = src_rank["B"].get(tid)
            c_rank = src_rank["C"].get(tid)
            best_bm25 = min(r for r in [b_rank, c_rank] if r is not None) if (b_rank or c_rank) else 0
            row[28] = 1.0 / best_bm25 if best_bm25 else 0.0

            # behavioral_source_count
            row[29] = sum(1 for s in ["A", "D", "F", "ALS"] if tid in src_rank[s])

            # content_source_count
            row[30] = sum(1 for s in ["B", "C"] if tid in src_rank[s])

            # query_has_artist
            row[31] = 1.0 if intent["has_artist"] else 0.0

            # query_has_genre
            row[32] = 1.0 if intent["has_genre"] else 0.0

            # query_is_discovery
            row[33] = 1.0 if intent["has_discovery"] else 0.0

            # query_has_title
            row[34] = 1.0 if intent["has_title"] else 0.0

            # title_bm25_match: track title tokens overlap with query
            title_toks = ttl.get(tid, set())
            row[35] = float(len(title_toks & query_toks)) if title_toks else 0.0

            # artist_in_query: this candidate's artist name appears in query
            ca = ta.get(tid, "")
            if isinstance(ca, list): ca = ca[0] if ca else ""
            if ca and len(ca) > 2:
                row[36] = 1.0 if ca.lower() in c["user_query"].lower() else 0.0

    # Run CV5 for both V2 baseline and R18
    seeds = [0, 1, 2]
    for config_name, X_config, feat_names in [
        ("v3_baseline", X_v2, FEATURE_NAMES_V2),
        ("r18_content_aware", X_r18, FEATURE_NAMES_R18),
    ]:
        nf = len(feat_names)
        print(f"\n{ts()} {config_name} ({nf} features)...", flush=True)

        X_f = X_config.reshape(-1, nf)
        lab = np.zeros(n * pool_k, dtype=np.float32)
        for i in range(n):
            if gt_idx[i] >= 0:
                lab[i * pool_k + gt_idx[i]] = 1.0

        cv5_seeds, lt_seeds = [], []
        sa_vals, da_vals, h0_vals, p0_vals = [], [], [], []
        unseen_ndcgs, seen_ndcgs = [], []
        bm25only_ndcgs = []

        for seed in seeds:
            folds = grouped_session_folds(sessions, seed)
            fold_ndcgs = []
            for fold in folds:
                held = set(fold.tolist())
                train_c = [j for j in range(n) if j not in held]
                val_c = fold.tolist()
                train_flat = [j*pool_k+k for j in train_c for k in range(int(sizes[j]))]
                val_flat = [j*pool_k+k for j in val_c for k in range(int(sizes[j]))]
                g_train = np.array([int(sizes[j]) for j in train_c], dtype=np.int32)
                g_val = np.array([int(sizes[j]) for j in val_c], dtype=np.int32)
                dtrain = lgb.Dataset(X_f[train_flat], lab[train_flat],
                                     group=g_train, feature_name=feat_names, free_raw_data=False)
                dval = lgb.Dataset(X_f[val_flat], lab[val_flat],
                                   group=g_val, reference=dtrain, free_raw_data=False)
                mdl = lgb.train(lgb_params, dtrain, num_boost_round=300,
                                valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
                val_scores = mdl.predict(X_f[val_flat])

                # Feature importance (last fold of last seed)
                if seed == seeds[-1] and fold is folds[-1]:
                    imp = mdl.feature_importance(importance_type="gain")
                    top_feats = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:15]

                offset = 0
                case_ndcgs = []
                for j in val_c:
                    sz = int(sizes[j])
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
                        la = ta.get(played[-1], "")
                        ga = ta.get(gt_tid, "")
                        if isinstance(la, list): la = la[0] if la else ""
                        if isinstance(ga, list): ga = ga[0] if ga else ""
                        if ga and la:
                            if ga == la: sa_vals.append(ndcg)
                            else: da_vals.append(ndcg)
                    if len(played) == 0: h0_vals.append(ndcg)
                    if track_pop.get(gt_tid, 0) == 0: p0_vals.append(ndcg)

                    if gt_source_info[j]["is_unseen"]: unseen_ndcgs.append(ndcg)
                    else: seen_ndcgs.append(ndcg)
                    if gt_source_info[j]["bm25_only"]: bm25only_ndcgs.append(ndcg)

                    offset += sz
                fold_ndcgs.append(float(np.mean(case_ndcgs)))
            cv5_seeds.append(float(np.mean(fold_ndcgs)))

        cv5 = float(np.mean(cv5_seeds))
        lt = float(np.mean(lt_seeds)) if lt_seeds else 0
        sa = float(np.mean(sa_vals)) if sa_vals else 0
        da = float(np.mean(da_vals)) if da_vals else 0
        h0 = float(np.mean(h0_vals)) if h0_vals else 0
        p0 = float(np.mean(p0_vals)) if p0_vals else 0
        unseen_ndcg = float(np.mean(unseen_ndcgs)) if unseen_ndcgs else 0
        seen_ndcg = float(np.mean(seen_ndcgs)) if seen_ndcgs else 0
        bm25only_ndcg = float(np.mean(bm25only_ndcgs)) if bm25only_ndcgs else 0

        print(f"  CV5={cv5:.4f}  last_turn={lt:.4f}")
        print(f"  same_artist={sa:.4f}  diff_artist={da:.4f}  hist_0={h0:.4f}  pop_0={p0:.4f}")
        print(f"  seen_nDCG={seen_ndcg:.4f}  unseen_nDCG={unseen_ndcg:.4f}  bm25only_nDCG={bm25only_ndcg:.4f}")

        print(f"  Top features (gain):")
        for fname, imp_val in top_feats[:10]:
            print(f"    {fname}: {imp_val:.0f}")

        lt_seeds, sa_vals, da_vals, h0_vals, p0_vals = [], [], [], [], []
        unseen_ndcgs, seen_ndcgs, bm25only_ndcgs = [], [], []

    elapsed = time.time() - t0
    print(f"\n{ts()} R18 complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
