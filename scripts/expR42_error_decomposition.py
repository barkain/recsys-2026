#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R42: Post-R39 error decomposition.

Classify all hist_7 cases into miss buckets (A-E) and cross-tabulate
by observable properties to find the next large, structural, low-noise
miss bucket for improvement.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1",
    "same_album_last3",
    "same_album_any",
    "album_history_count",
    "pool_same_album_count",
]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    """Load track_id -> album_id mapping."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds:
        tid = str(item["track_id"])
        alb_id = item.get("album_id", [])
        if isinstance(alb_id, list) and alb_id:
            track_album[tid] = str(alb_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                track_album[tid] = str(alb_name[0])
            else:
                track_album[tid] = ""
    return track_album


def load_track_artists_from_meta():
    """Load track_id -> artist_name mapping from metadata."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_artist_meta = {}
    for item in ds:
        tid = str(item["track_id"])
        artist = item.get("artist_name", [])
        if isinstance(artist, list) and artist:
            track_artist_meta[tid] = str(artist[0]).lower().strip()
        else:
            track_artist_meta[tid] = str(artist).lower().strip() if artist else ""
    return track_artist_meta


def load_training_track_set():
    """Get all track_ids that appear in training sessions."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]
    seen = set()
    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                seen.add(str(c["content"]).strip())
    return seen


def main():
    t0 = time.time()
    print(f"{ts()} R42: Post-R39 Error Decomposition")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data (same as R39a)
    # ---------------------------------------------------------------
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading artist metadata...")
    track_artist_meta = load_track_artists_from_meta()

    print(f"{ts()} Loading training track set...")
    training_tracks = load_training_track_set()
    print(f"  {len(training_tracks)} unique tracks seen in training")

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids[j] for j in top])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    # ---------------------------------------------------------------
    # Step 1: Build features & run CV5 LambdaRank (R39 config exactly)
    # ---------------------------------------------------------------
    n = len(cases)
    n_feat = len(ALL_FEAT)
    n_feat_base = len(FEAT_BASE)
    folds = grouped_session_folds(sessions, seed=0)

    print(f"\n{ts()} Building R39 features ({n_feat} features, pool@{POOL_K})...")

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]  # store pools for later analysis

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools[i] = pool
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
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

        # Album precomputation
        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums if a)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features (identical to R39a)
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
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"]
                         if tid in src_rank.get(sn, {}))
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

            # Album features
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # ---------------------------------------------------------------
    # CV5 LambdaRank — collect per-case OOF ranks
    # ---------------------------------------------------------------
    print(f"\n{ts()} Running CV5 LambdaRank (R39 config)...")
    case_lr_rank = np.full(n, -1, dtype=np.int64)  # GT rank from LR, -1 if GT not in pool

    for fi in range(5):
        val_set = set(folds[fi].tolist())
        tr = [j for j in range(n) if j not in val_set]
        va = sorted(val_set)
        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in tr:
            s = int(sizes[idx])
            for k in range(s):
                X_tr.append(X[idx, k])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(sizes[idx])
            for k in range(s):
                X_va.append(X[idx, k])
                y_va.append(1.0 if k == gt_idx[idx] else 0.0)
            g_va.append(s)
        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(ALL_FEAT))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            offset += s
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1  # 1-indexed rank

    # Quick sanity: h7 nDCG
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg_vals = []
    for i in h7:
        if case_lr_rank[i] > 0 and case_lr_rank[i] <= 20:
            h7_ndcg_vals.append(1.0 / np.log2(case_lr_rank[i] + 1))
        else:
            h7_ndcg_vals.append(0.0)
    h7_ndcg = float(np.mean(h7_ndcg_vals))
    print(f"  h7 nDCG@20 = {h7_ndcg:.5f} (sanity check)")

    # ---------------------------------------------------------------
    # Step 2: Classify hist_7 cases into buckets A-E
    # ---------------------------------------------------------------
    print(f"\n{ts()} Classifying {len(h7)} hist_7 cases into buckets...")

    # Build source union for each case (all sources, unlimited depth)
    bucket_counts = Counter()
    case_bucket = {}

    for i in h7:
        c = cases[i]
        gt = c["gt"]

        if case_lr_rank[i] >= 1 and case_lr_rank[i] <= 20:
            bucket = "A"
        elif gt_idx[i] >= 0 and case_lr_rank[i] >= 21 and case_lr_rank[i] <= 100:
            bucket = "B"
        elif gt_idx[i] >= 0 and case_lr_rank[i] > 100:
            bucket = "C"
        elif gt_idx[i] < 0:
            # GT not in pool@300, check if in source union
            src_union = set()
            src_union.update(payload["src_a"][i])
            src_union.update(payload["src_b"][i])
            src_union.update(payload["src_c"][i])
            src_union.update(payload["src_d"][i])
            src_union.update(payload["src_f"][i])
            src_union.update(als_source[i])
            src_union.update(r21_source[i])
            if gt in src_union:
                bucket = "D"
            else:
                bucket = "E"
        else:
            bucket = "C"  # fallback

        case_bucket[i] = bucket
        bucket_counts[bucket] += 1

    print("  Bucket distribution:")
    for b in ["A", "B", "C", "D", "E"]:
        cnt = bucket_counts[b]
        print(f"    {b}: {cnt:>4d} ({cnt/len(h7)*100:.1f}%)")

    # ---------------------------------------------------------------
    # Step 3: Cross-tabulate each bucket by observable properties
    # ---------------------------------------------------------------
    print(f"\n{ts()} Cross-tabulating by observable properties...")

    # Precompute BM25 source ranks (B=BM25_meta, C=BM25_text — take best rank)
    # In the payload: src_b = BM25 source B, src_c = BM25 source C
    # "Best BM25" = min rank across B and C

    # Results: bucket -> {prop -> count}
    cross_tab = {b: Counter() for b in ["A", "B", "C", "D", "E"]}

    for i in h7:
        c = cases[i]
        gt = c["gt"]
        bucket = case_bucket[i]
        played = c["music_turns"]
        query = c["user_query"]

        # Properties
        played_artists = {ta.get(t, "") for t in played} - {""}
        gt_artist = ta.get(gt, "")

        # same_artist / diff_artist
        if gt_artist and gt_artist in played_artists:
            cross_tab[bucket]["same_artist"] += 1
        else:
            cross_tab[bucket]["diff_artist"] += 1

        # same_album
        played_album_ids = {track_album.get(t, "") for t in played} - {""}
        gt_album = track_album.get(gt, "")
        if gt_album and gt_album in played_album_ids:
            cross_tab[bucket]["same_album"] += 1

        # seen_track / unseen_track
        if gt in training_tracks:
            cross_tab[bucket]["seen_track"] += 1
        else:
            cross_tab[bucket]["unseen_track"] += 1

        # artist_in_query
        gt_artist_name = track_artist_meta.get(gt, "")
        if gt_artist_name and len(gt_artist_name) > 2 and gt_artist_name in query.lower():
            cross_tab[bucket]["artist_in_query"] += 1

        # r21_top20
        r21_list = r21_source[i]
        gt_r21_rank = (r21_list.index(gt) + 1) if gt in r21_list[:300] else -1
        if gt_r21_rank >= 1 and gt_r21_rank <= 20:
            cross_tab[bucket]["r21_top20"] += 1
            # r21_top20_demoted: only for B/C (GT in pool but LR ranked >20)
            if bucket in ("B", "C"):
                cross_tab[bucket]["r21_top20_demoted"] += 1

        # bm25_top20: best of B and C source rank
        src_b_list = payload["src_b"][i]
        src_c_list = payload["src_c"][i]
        gt_b_rank = (src_b_list.index(gt) + 1) if gt in src_b_list[:300] else 999
        gt_c_rank = (src_c_list.index(gt) + 1) if gt in src_c_list[:300] else 999
        best_bm25 = min(gt_b_rank, gt_c_rank)
        if best_bm25 <= 20:
            cross_tab[bucket]["bm25_top20"] += 1
            if bucket in ("B", "C"):
                cross_tab[bucket]["bm25_top20_demoted"] += 1

    # ---------------------------------------------------------------
    # Step 4: Estimate recoverability
    # ---------------------------------------------------------------
    print(f"\n{ts()} Estimating recoverability...")

    # Total h7 misses = B + C + D + E
    total_misses = sum(bucket_counts[b] for b in ["B", "C", "D", "E"])

    recovery_methods = []

    bc_count = bucket_counts["B"] + bucket_counts["C"]

    # r21_top20_demoted in B/C — LR demoted a high-R21 candidate
    r21_demoted = cross_tab["B"].get("r21_top20_demoted", 0) + cross_tab["C"].get("r21_top20_demoted", 0)
    bm25_demoted = cross_tab["B"].get("bm25_top20_demoted", 0) + cross_tab["C"].get("bm25_top20_demoted", 0)

    # Source protection: cases where GT was top-20 in a strong source but LR dropped it
    source_protect_cases = max(r21_demoted, bm25_demoted)  # overlap likely, take max as proxy

    recovery_methods.append({
        "method": "Better LR features (rerank B+C)",
        "cases": bc_count,
        "pct": bc_count / total_misses * 100 if total_misses else 0,
        "approach": "Add discriminative features to push GT above wrong top-20 in pool",
    })
    recovery_methods.append({
        "method": "Source protection (demoted GT)",
        "cases": source_protect_cases,
        "pct": source_protect_cases / total_misses * 100 if total_misses else 0,
        "approach": "Protect high-source-rank candidates from LR demotion (floor rank)",
    })
    recovery_methods.append({
        "method": "Pool expansion / new retriever (D)",
        "cases": bucket_counts["D"],
        "pct": bucket_counts["D"] / total_misses * 100 if total_misses else 0,
        "approach": "GT in source union but RRF@300 drops it — tune RRF or enlarge pool",
    })
    recovery_methods.append({
        "method": "New retrieval source (E)",
        "cases": bucket_counts["E"],
        "pct": bucket_counts["E"] / total_misses * 100 if total_misses else 0,
        "approach": "GT not in any source — need fundamentally new retriever",
    })

    # ---------------------------------------------------------------
    # Step 5: Output
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("R42 ERROR DECOMPOSITION — BUCKET TABLE")
    print(sep)

    header = f"{'Bucket':>6s} | {'Count':>5s} | {'same_art':>8s} | {'diff_art':>8s} | {'same_alb':>8s} | {'seen_trk':>8s} | {'unseen':>6s} | {'art_qry':>7s} | {'r21_t20':>7s} | {'bm25_t20':>8s} | {'r21_dem':>7s} | {'bm25_dem':>8s}"
    print(header)
    print("-" * len(header))

    for b in ["A", "B", "C", "D", "E"]:
        cnt = bucket_counts[b]
        ct = cross_tab[b]
        row = (
            f"{b:>6s} | {cnt:>5d} | "
            f"{ct['same_artist']:>8d} | {ct['diff_artist']:>8d} | {ct['same_album']:>8d} | "
            f"{ct['seen_track']:>8d} | {ct['unseen_track']:>6d} | {ct['artist_in_query']:>7d} | "
            f"{ct['r21_top20']:>7d} | {ct['bm25_top20']:>8d} | "
            f"{ct.get('r21_top20_demoted', 0):>7d} | {ct.get('bm25_top20_demoted', 0):>8d}"
        )
        print(row)

    # Percentage view
    print(f"\n{'Bucket':>6s} | {'Count':>5s} | {'same_art':>8s} | {'diff_art':>8s} | {'same_alb':>8s} | {'seen_trk':>8s} | {'unseen':>6s} | {'art_qry':>7s} | {'r21_t20':>7s} | {'bm25_t20':>8s}")
    print("-" * 110)
    for b in ["A", "B", "C", "D", "E"]:
        cnt = bucket_counts[b]
        if cnt == 0:
            continue
        ct = cross_tab[b]
        def pct(v):
            return f"{v/cnt*100:.0f}%"
        row = (
            f"{b:>6s} | {cnt:>5d} | "
            f"{pct(ct['same_artist']):>8s} | {pct(ct['diff_artist']):>8s} | {pct(ct['same_album']):>8s} | "
            f"{pct(ct['seen_track']):>8s} | {pct(ct['unseen_track']):>6s} | {pct(ct['artist_in_query']):>7s} | "
            f"{pct(ct['r21_top20']):>7s} | {pct(ct['bm25_top20']):>8s}"
        )
        print(row)

    # Recovery summary
    print(f"\n{sep}")
    print("RECOVERABLE MASS SUMMARY")
    print(sep)
    print(f"Total h7 cases: {len(h7)}")
    print(f"Total h7 misses (B+C+D+E): {total_misses}")
    print(f"h7 hits (A): {bucket_counts['A']} ({bucket_counts['A']/len(h7)*100:.1f}%)")
    print()
    print(f"{'Recovery method':<40s} | {'Est cases':>9s} | {'% misses':>8s} | Candidate approach")
    print("-" * 120)
    for rm in recovery_methods:
        print(f"{rm['method']:<40s} | {rm['cases']:>9d} | {rm['pct']:>7.1f}% | {rm['approach']}")

    # Detailed sub-analysis of B+C (rerank targets)
    print(f"\n{sep}")
    print("DETAILED B+C RERANK ANALYSIS")
    print(sep)
    bc_cases = [i for i in h7 if case_bucket[i] in ("B", "C")]
    print(f"B+C total: {len(bc_cases)} cases")

    # Median/mean LR rank for B vs C
    b_ranks = [case_lr_rank[i] for i in h7 if case_bucket[i] == "B"]
    c_ranks = [case_lr_rank[i] for i in h7 if case_bucket[i] == "C"]
    if b_ranks:
        print(f"  B (rank 21-100): n={len(b_ranks)}, median={np.median(b_ranks):.0f}, mean={np.mean(b_ranks):.1f}")
    if c_ranks:
        print(f"  C (rank 101-300): n={len(c_ranks)}, median={np.median(c_ranks):.0f}, mean={np.mean(c_ranks):.1f}")

    # Of B+C same_artist cases, how many are same_artist + r21_top20?
    bc_same_art_r21 = 0
    bc_same_art_bm25 = 0
    bc_diff_art_unseen = 0
    bc_artist_query = 0
    for i in bc_cases:
        gt = cases[i]["gt"]
        played = cases[i]["music_turns"]
        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in played} - {""}
        is_same = gt_artist and gt_artist in played_artists
        r21_list_i = r21_source[i]
        r21_rank_i = (r21_list_i.index(gt) + 1) if gt in r21_list_i[:20] else -1
        bm25_b = (payload["src_b"][i].index(gt) + 1) if gt in payload["src_b"][i][:20] else 999
        bm25_c_rank = (payload["src_c"][i].index(gt) + 1) if gt in payload["src_c"][i][:20] else 999
        bm25_best = min(bm25_b, bm25_c_rank)
        if is_same and r21_rank_i >= 1:
            bc_same_art_r21 += 1
        if is_same and bm25_best <= 20:
            bc_same_art_bm25 += 1
        if not is_same and gt not in training_tracks:
            bc_diff_art_unseen += 1
        gt_artist_name = track_artist_meta.get(gt, "")
        if gt_artist_name and len(gt_artist_name) > 2 and gt_artist_name in cases[i]["user_query"].lower():
            bc_artist_query += 1

    print("\n  B+C sub-populations:")
    print(f"    same_artist + r21_top20:   {bc_same_art_r21:>4d} (structural rerank target)")
    print(f"    same_artist + bm25_top20:  {bc_same_art_bm25:>4d} (structural rerank target)")
    print(f"    artist_in_query:           {bc_artist_query:>4d} (query-driven rerank target)")
    print(f"    diff_artist + unseen:      {bc_diff_art_unseen:>4d} (hard/unobservable)")

    # Detailed sub-analysis of D (pool miss)
    print(f"\n{sep}")
    print("DETAILED D (POOL MISS) ANALYSIS")
    print(sep)
    d_cases = [i for i in h7 if case_bucket[i] == "D"]
    if d_cases:
        # Which sources had the GT?
        d_source_had_gt = Counter()
        d_gt_best_source_rank = []
        for i in d_cases:
            gt = cases[i]["gt"]
            best = 9999
            for sname, slist_key in [("A", "src_a"), ("B", "src_b"), ("C", "src_c"),
                                      ("D", "src_d"), ("F", "src_f")]:
                sl = payload[slist_key][i]
                if gt in sl:
                    d_source_had_gt[sname] += 1
                    rank_in = sl.index(gt) + 1
                    best = min(best, rank_in)
            if gt in als_source[i]:
                d_source_had_gt["ALS"] += 1
                rank_in = als_source[i].index(gt) + 1
                best = min(best, rank_in)
            if gt in r21_source[i]:
                d_source_had_gt["R21"] += 1
                rank_in = r21_source[i].index(gt) + 1
                best = min(best, rank_in)
            d_gt_best_source_rank.append(best)

        print(f"  D cases: {len(d_cases)} — GT in source union but not in pool@{POOL_K}")
        print("  Source coverage (which source had GT):")
        for sname in ["A", "B", "C", "D", "F", "ALS", "R21"]:
            print(f"    {sname:>4s}: {d_source_had_gt[sname]:>3d} ({d_source_had_gt[sname]/len(d_cases)*100:.1f}%)")
        if d_gt_best_source_rank:
            print(f"  Best source rank of GT: median={np.median(d_gt_best_source_rank):.0f}, "
                  f"mean={np.mean(d_gt_best_source_rank):.0f}")

    # Detailed sub-analysis of E (unreachable)
    print(f"\n{sep}")
    print("DETAILED E (UNREACHABLE) ANALYSIS")
    print(sep)
    e_cases = [i for i in h7 if case_bucket[i] == "E"]
    if e_cases:
        e_same_art = sum(1 for i in e_cases
                         if ta.get(cases[i]["gt"], "") and
                         ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]})
        e_seen = sum(1 for i in e_cases if cases[i]["gt"] in training_tracks)
        e_art_query = 0
        for i in e_cases:
            gt_artist_name = track_artist_meta.get(cases[i]["gt"], "")
            if gt_artist_name and len(gt_artist_name) > 2 and gt_artist_name in cases[i]["user_query"].lower():
                e_art_query += 1
        print(f"  E cases: {len(e_cases)} — GT not in any source")
        print(f"    same_artist as played: {e_same_art} ({e_same_art/len(e_cases)*100:.1f}%)")
        print(f"    seen in training:      {e_seen} ({e_seen/len(e_cases)*100:.1f}%)")
        print(f"    artist_in_query:       {e_art_query} ({e_art_query/len(e_cases)*100:.1f}%)")

    # ---------------------------------------------------------------
    # Priority ranking
    # ---------------------------------------------------------------
    print(f"\n{sep}")
    print("PRIORITY RANKING — Next target")
    print(sep)

    targets = []
    # B bucket (close misses)
    targets.append(("Rerank B (close miss)", bucket_counts["B"],
                    "Structural — GT in pool and near top, fixable with better features"))
    # D bucket (pool miss, sourceable)
    targets.append(("Pool coverage D", bucket_counts["D"],
                    "Structural — tune RRF weights, enlarge pool, or add source weight"))
    # C bucket (far miss)
    targets.append(("Rerank C (far miss)", bucket_counts["C"],
                    "Harder — GT deep in pool, large rank jump needed"))
    # E bucket (unreachable)
    targets.append(("New retrieval E", bucket_counts["E"],
                    "Hardest — fundamentally new retriever needed"))

    targets.sort(key=lambda x: -x[1])
    for rank, (name, cnt, note) in enumerate(targets, 1):
        print(f"  {rank}. {name:<30s} {cnt:>4d} cases ({cnt/len(h7)*100:.1f}% of h7) — {note}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    out_data = {
        "h7_count": len(h7),
        "h7_ndcg": h7_ndcg,
        "total_misses": total_misses,
        "bucket_counts": dict(bucket_counts),
        "cross_tab": {b: dict(ct) for b, ct in cross_tab.items()},
        "recovery_methods": recovery_methods,
        "pool_hit": pool_hit,
    }
    out_path = REPO / "exp" / "eval" / "expR42_error_decomposition.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
