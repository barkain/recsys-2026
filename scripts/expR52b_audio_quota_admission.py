#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R52b: Audio quota-based admission experiment.

Phase 2 showed audio CLAP candidates can't compete in 8-source RRF fusion.
This experiment forcibly reserves pool slots for top audio candidates:
  quota_5:  reserve 5 slots
  quota_10: reserve 10 slots
  quota_20: reserve 20 slots
  gated_quota_10: reserve 10 slots only for discovery-regime cases

Baseline: R39 album-aware LambdaRank (h7=0.24298, pool_hit=0.6000).
"""
from __future__ import annotations

import gc
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
BUCKET_CACHE = REPO / "cache" / "r52_bucket_labels.json"
CLAP_LISTS_CACHE = REPO / "cache" / "r52_audio_clap_lists.json"
OUT_PATH = REPO / "exp" / "eval" / "expR52b_audio_quota_admission.json"
RRF_K = 20
POOL_K = 300
SW_R39 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R39 = FEAT_BASE + FEAT_ALBUM  # 34 features

FEAT_AUDIO = [
    "audio_rank_inv",
    "audio_presence",
    "audio_cosine",
]
FEAT_QUOTA = FEAT_R39 + FEAT_AUDIO  # 37 features


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_album_mapping():
    """Load album mapping from HF metadata dataset."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds:
        tid = str(item["track_id"])
        album_id_raw = item.get("album_id", [])
        if isinstance(album_id_raw, list) and album_id_raw:
            album_id = str(album_id_raw[0])
        else:
            album_id = ""
        if not album_id:
            album_name_raw = item.get("album_name", [])
            if isinstance(album_name_raw, list) and album_name_raw:
                album_id = str(album_name_raw[0])
            else:
                album_id = str(album_name_raw) if album_name_raw else ""
        track_album[tid] = album_id
    del ds
    return track_album


def load_clap_embeddings():
    """Load CLAP embeddings from HF embeddings dataset.
    Returns (matrix, track_ids, tid_to_idx, n_valid).
    """
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]

    track_ids = []
    vecs = []
    dim = 512
    for item in ds:
        track_ids.append(str(item["track_id"]))
        v = item["audio-laion_clap"]
        if v is not None and len(v) > 0:
            vecs.append(v)
        else:
            vecs.append(None)

    arr = np.zeros((len(vecs), dim), dtype=np.float32)
    n_valid = 0
    for i, v in enumerate(vecs):
        if v is not None and len(v) == dim:
            arr[i] = v
            if np.any(arr[i] != 0):
                n_valid += 1

    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms

    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    del ds, vecs
    gc.collect()

    return arr, track_ids, tid_to_idx, n_valid


def compute_clap_query(played, clap_matrix, clap_tid_to_idx):
    """Compute recency-weighted CLAP query for a case's history.
    Weight = 0.9^(n-1-pos), L2-normalized.
    Returns query vector (512,) or None.
    """
    n_played = len(played)
    valid_idx = []
    weights = []
    for pos, t in enumerate(played):
        if t in clap_tid_to_idx:
            valid_idx.append(clap_tid_to_idx[t])
            weights.append(0.9 ** (n_played - 1 - pos))

    if not valid_idx:
        return None

    hist_embs = clap_matrix[valid_idx]  # (k, 512)
    w = np.array(weights, dtype=np.float32)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    query = (hist_embs * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm
    return query


def build_r39_baseline(cases, payload, als_source, als_vecs, r21_source,
                       track_pop, max_pop, track_album,
                       als_factors, als_track_to_idx, folds):
    """Build R39 baseline pools, features, and run CV5. Returns everything needed."""
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    n = len(cases)
    n_feat_base = len(FEAT_BASE)
    n_feat = len(FEAT_R39)

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = []
    rrf_scores_all: list[dict[str, float]] = []

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW_R39, topk=POOL_K, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        # Compute RRF scores for pool ordering (needed for quota replacement)
        rrf_sc: dict[str, float] = {}
        for name, ranked in src_lists.items():
            w = SW_R39.get(name, 0.0)
            if w == 0 or not ranked:
                continue
            for rank, tid in enumerate(ranked, start=1):
                rrf_sc[tid] = rrf_sc.get(tid, 0.0) + w / (RRF_K + rank)
        rrf_scores_all.append(rrf_sc)

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

            # Base 29 features
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

            # Album features (indices 29-33)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    return X, gt_idx, sizes, pools, rrf_scores_all


def build_quota_pool(r39_pool, r39_rrf_scores, audio_top300, quota):
    """Build a quota-modified pool.

    1. Start with R39 pool (300 candidates from weighted RRF)
    2. Find top audio candidates NOT already in the R39 pool
    3. Remove the `quota` lowest-RRF-ranked candidates from R39 pool
    4. Insert audio candidates, maintaining pool size at 300

    Returns modified pool (list of track IDs).
    """
    r39_set = set(r39_pool)
    audio_new = [t for t in audio_top300 if t not in r39_set][:quota]

    if not audio_new:
        return list(r39_pool)  # no new audio candidates to add

    n_to_add = len(audio_new)

    # Remove bottom N candidates (lowest RRF score = last in pool)
    # R39 pool is sorted by RRF score descending, so bottom is at the end
    modified = list(r39_pool[:len(r39_pool) - n_to_add])
    modified.extend(audio_new)

    return modified


def build_quota_features(cases, payload, als_source, als_vecs, r21_source,
                         track_pop, max_pop, track_album,
                         als_factors, als_track_to_idx,
                         quota_pools, clap_lists, clap_matrix, clap_tid_to_idx):
    """Build 37-feature matrix for quota-modified pools."""
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    n = len(cases)
    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = len(FEAT_R39)
    n_feat = len(FEAT_QUOTA)

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        pool = quota_pools[i]
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        # Source ranks (from original sources, NOT the quota pool)
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
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

        # Audio CLAP precomputation
        clap_rank_map: dict[str, int] = {}
        if clap_lists[i]:
            for r, tid in enumerate(clap_lists[i], start=1):
                clap_rank_map[tid] = r

        clap_query = compute_clap_query(played, clap_matrix, clap_tid_to_idx)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features
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

            # Album features (indices 29-33)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

            # Audio features (indices 34-36)
            # audio_rank_inv: 1/rank if in audio top-300
            clap_r = clap_rank_map.get(tid)
            row[n_feat_r39 + 0] = 1.0 / clap_r if clap_r is not None else 0.0

            # audio_presence: 1.0 if in audio top-300
            row[n_feat_r39 + 1] = 1.0 if clap_r is not None else 0.0

            # audio_cosine: cosine sim between candidate and recency-weighted history query
            if clap_query is not None and tid in clap_tid_to_idx:
                cand_emb = clap_matrix[clap_tid_to_idx[tid]]
                row[n_feat_r39 + 2] = float(np.dot(clap_query, cand_emb))

    return X, gt_idx, sizes


def run_cv5_lambdarank(X, gt_idx, sizes, folds, feat_names, n):
    """Run CV5 LambdaRank and return (case_ndcg, lr_scores)."""
    case_ndcg = np.zeros(n)
    lr_scores = np.full((n, POOL_K), -np.inf, dtype=np.float64)
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }

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
                            group=g_tr, feature_name=list(feat_names))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        model = lgb.train(lr_params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            lr_scores[idx, :s] = sc
            offset += s
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    return case_ndcg, lr_scores


def compute_metrics(cases, case_ndcg, lr_scores, gt_idx, sizes, pools,
                    h7, ta, bucket_labels,
                    baseline_ndcg, baseline_lr_scores,
                    baseline_gt_idx, baseline_pools, baseline_sizes,
                    is_baseline=False):
    """Compute all metrics for a config."""
    # h7 nDCG
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))

    # pool_hit on h7
    pool_hit = float(np.mean([1.0 if gt_idx[i] >= 0 else 0.0 for i in h7]))

    # same_artist_h7 / diff_artist_h7
    same_artist_ndcg = []
    diff_artist_ndcg = []
    for i in h7:
        c = cases[i]
        played = c["music_turns"]
        if not played:
            diff_artist_ndcg.append(case_ndcg[i])
            continue
        gt = c["gt"]
        last_artist = ta.get(played[-1], "")
        gt_artist = ta.get(gt, "")
        if last_artist and gt_artist and last_artist == gt_artist:
            same_artist_ndcg.append(case_ndcg[i])
        else:
            diff_artist_ndcg.append(case_ndcg[i])

    same_h7 = float(np.mean(same_artist_ndcg)) if same_artist_ndcg else 0.0
    diff_h7 = float(np.mean(diff_artist_ndcg)) if diff_artist_ndcg else 0.0

    # Bucket admission and recovery
    bucket_e_admitted = 0
    bucket_d_admitted = 0
    bucket_e_recovered_top20 = 0
    if bucket_labels:
        for i in h7:
            si = str(i)
            if si not in bucket_labels:
                continue
            bl = bucket_labels[si]
            if bl["bucket"] == "E" and gt_idx[i] >= 0:
                bucket_e_admitted += 1
                # Check if actually ranked in top-20
                ps = int(sizes[i])
                if ps > 0:
                    sc = lr_scores[i, :ps]
                    ranked = np.argsort(-sc)
                    gt_pos_arr = np.where(ranked == gt_idx[i])[0]
                    if len(gt_pos_arr) > 0 and gt_pos_arr[0] < 20:
                        bucket_e_recovered_top20 += 1
            elif bl["bucket"] == "D" and gt_idx[i] >= 0:
                bucket_d_admitted += 1

    # recovered / lost / net / top20_overlap / top1_changed
    recovered = 0
    lost = 0
    top20_overlap_vals = []
    top1_changed = 0

    if not is_baseline and baseline_lr_scores is not None:
        for i in h7:
            ps = int(sizes[i])
            bps = int(baseline_sizes[i])

            # Config rank of GT
            config_rank = -1
            if gt_idx[i] >= 0 and ps > 0:
                sc = lr_scores[i, :ps]
                ranked = np.argsort(-sc)
                gt_pos_arr = np.where(ranked == gt_idx[i])[0]
                if len(gt_pos_arr) > 0:
                    config_rank = int(gt_pos_arr[0]) + 1

            # Baseline rank of GT
            baseline_rank = -1
            if baseline_gt_idx[i] >= 0 and bps > 0:
                bsc = baseline_lr_scores[i, :bps]
                branked = np.argsort(-bsc)
                bgt_pos_arr = np.where(branked == baseline_gt_idx[i])[0]
                if len(bgt_pos_arr) > 0:
                    baseline_rank = int(bgt_pos_arr[0]) + 1

            if baseline_rank > 20 and 1 <= config_rank <= 20:
                recovered += 1
            elif 1 <= baseline_rank <= 20 and (config_rank > 20 or config_rank < 0):
                lost += 1

            # top-20 overlap (fraction of config top-20 that's in baseline top-20)
            config_top20 = set()
            if ps > 0:
                sc = lr_scores[i, :ps]
                ranked = np.argsort(-sc)
                config_top20 = {pools[i][int(ranked[j])] for j in range(min(20, ps))}

            baseline_top20 = set()
            if bps > 0:
                bsc = baseline_lr_scores[i, :bps]
                branked = np.argsort(-bsc)
                baseline_top20 = {baseline_pools[i][int(branked[j])] for j in range(min(20, bps))}

            if config_top20 and baseline_top20:
                overlap = len(config_top20 & baseline_top20) / 20.0
                top20_overlap_vals.append(overlap)

            # top-1 changed
            config_top1 = pools[i][int(np.argsort(-lr_scores[i, :ps])[0])] if ps > 0 else None
            baseline_top1 = baseline_pools[i][int(np.argsort(-baseline_lr_scores[i, :bps])[0])] if bps > 0 else None
            if config_top1 != baseline_top1:
                top1_changed += 1

    net = recovered - lost
    top20_overlap = float(np.mean(top20_overlap_vals)) if top20_overlap_vals else 1.0

    return {
        "h7": round(h7_ndcg, 5),
        "pool_hit": round(pool_hit, 4),
        "same_h7": round(same_h7, 5),
        "diff_h7": round(diff_h7, 5),
        "E_admitted": bucket_e_admitted,
        "D_admitted": bucket_d_admitted,
        "E_recovered_top20": bucket_e_recovered_top20,
        "recovered": recovered,
        "lost": lost,
        "net": net,
        "top20_overlap": round(top20_overlap, 4),
        "top1_changed": top1_changed,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R52b: Audio Quota-Based Admission Experiment")
    print("=" * 70)

    # ---------------------------------------------------------------
    # STEP 1: Load shared data
    # ---------------------------------------------------------------
    print(f"\n{ts()} STEP 1: Loading shared data")
    print("-" * 70)

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    n = len(cases)

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Loading album mapping...")
    track_album = load_album_mapping()

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

    folds = grouped_session_folds(sessions, seed=0)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"  Cases: {n}, h7: {len(h7)}")

    # Load bucket labels
    print(f"{ts()} Loading bucket labels...")
    with open(BUCKET_CACHE) as f:
        bucket_data = json.load(f)
    bucket_labels = bucket_data["bucket_labels"]

    # ---------------------------------------------------------------
    # STEP 1b: R39 baseline reproduction
    # ---------------------------------------------------------------
    print(f"\n{ts()} STEP 1b: R39 Baseline Reproduction")
    print("-" * 70)

    print(f"{ts()} Building R39 feature matrix ({len(FEAT_R39)} features)...")
    X_bl, gt_idx_bl, sizes_bl, pools_bl, rrf_scores_bl = build_r39_baseline(
        cases, payload, als_source, als_vecs, r21_source,
        track_pop, max_pop, track_album,
        als_factors, als_track_to_idx, folds)

    pool_hit_bl = float(np.mean(gt_idx_bl >= 0))
    print(f"  pool_hit@{POOL_K} (all): {pool_hit_bl:.4f}")

    print(f"{ts()} Running CV5 LambdaRank (R39 baseline)...")
    case_ndcg_bl, lr_scores_bl = run_cv5_lambdarank(
        X_bl, gt_idx_bl, sizes_bl, folds, FEAT_R39, n)

    h7_ndcg_bl = float(np.mean([case_ndcg_bl[i] for i in h7]))
    print(f"  R39 reproduction: h7={h7_ndcg_bl:.5f}")

    # HARD CHECK
    expected_h7 = 0.24298
    if abs(h7_ndcg_bl - expected_h7) > 0.001:
        print(f"\n  *** HARD CHECK FAILED: h7={h7_ndcg_bl:.5f}, expected ~{expected_h7} ***")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={h7_ndcg_bl:.5f} within +/-0.001 of {expected_h7}")

    baseline_metrics = compute_metrics(
        cases, case_ndcg_bl, lr_scores_bl, gt_idx_bl, sizes_bl, pools_bl,
        h7, ta, bucket_labels,
        None, None, None, None, None, is_baseline=True)
    print(f"  Baseline: h7={baseline_metrics['h7']}, pool_hit={baseline_metrics['pool_hit']}, "
          f"same_h7={baseline_metrics['same_h7']}, diff_h7={baseline_metrics['diff_h7']}")

    # Free baseline feature matrix (keep scores, pools, gt_idx, sizes for comparison)
    del X_bl
    gc.collect()

    # ---------------------------------------------------------------
    # STEP 2: Load audio CLAP data
    # ---------------------------------------------------------------
    print(f"\n{ts()} STEP 2: Loading audio CLAP data")
    print("-" * 70)

    print(f"{ts()} Loading CLAP retrieval lists from cache...")
    with open(CLAP_LISTS_CACHE) as f:
        clap_lists = json.load(f)
    non_empty = sum(1 for x in clap_lists if len(x) > 0)
    print(f"  CLAP lists: {len(clap_lists)} total, {non_empty} non-empty")

    print(f"{ts()} Loading CLAP embeddings...")
    clap_matrix, clap_track_ids, clap_tid_to_idx, clap_valid = load_clap_embeddings()
    print(f"  CLAP embeddings: shape={clap_matrix.shape}, valid={clap_valid}/{len(clap_track_ids)}")

    # ---------------------------------------------------------------
    # STEP 3-6: Run quota configs
    # ---------------------------------------------------------------
    quota_configs = {
        "quota_5": {"quota": 5, "gated": False},
        "quota_10": {"quota": 10, "gated": False},
        "quota_20": {"quota": 20, "gated": False},
        "gated_quota_10": {"quota": 10, "gated": True},
    }

    # Precompute discovery mask for gated config
    discovery_mask = np.zeros(n, dtype=bool)
    for i, c in enumerate(cases):
        played = c["music_turns"]
        played_artists = [ta.get(t, "") for t in played]
        unique_artists = len(set(a for a in played_artists if a))
        last3_artists = played_artists[-3:] if len(played_artists) >= 3 else played_artists
        all_same = len(set(last3_artists)) == 1
        if unique_artists >= 5 and not all_same:
            discovery_mask[i] = True

    h7_discovery = sum(1 for i in h7 if discovery_mask[i])
    print(f"\n  Discovery cases: {int(discovery_mask.sum())}/{n} total, "
          f"{h7_discovery}/{len(h7)} h7")

    all_config_metrics = {}

    for config_name, config in quota_configs.items():
        print(f"\n{ts()} CONFIG: {config_name}")
        print("-" * 70)

        quota = config["quota"]
        gated = config["gated"]

        # Build quota pools
        print(f"{ts()} Building quota pools (quota={quota}, gated={gated})...")
        quota_pools = []
        n_modified = 0
        n_audio_new_total = 0

        for i in range(n):
            if gated and not discovery_mask[i]:
                # Non-discovery: use R39 pool unchanged
                quota_pools.append(pools_bl[i])
                continue

            audio_top300 = clap_lists[i] if i < len(clap_lists) else []
            if not audio_top300:
                quota_pools.append(pools_bl[i])
                continue

            modified = build_quota_pool(pools_bl[i], rrf_scores_bl[i], audio_top300, quota)
            quota_pools.append(modified)

            # Count how many are actually new
            r39_set = set(pools_bl[i])
            n_new = sum(1 for t in modified if t not in r39_set)
            if n_new > 0:
                n_modified += 1
                n_audio_new_total += n_new

        print(f"  Modified pools: {n_modified}/{n}")
        print(f"  Total new audio candidates inserted: {n_audio_new_total}")
        if n_modified > 0:
            print(f"  Avg new per modified pool: {n_audio_new_total/n_modified:.1f}")

        # Build features
        print(f"{ts()} Building feature matrix ({len(FEAT_QUOTA)} features)...")
        X_q, gt_idx_q, sizes_q = build_quota_features(
            cases, payload, als_source, als_vecs, r21_source,
            track_pop, max_pop, track_album,
            als_factors, als_track_to_idx,
            quota_pools, clap_lists, clap_matrix, clap_tid_to_idx)

        pool_hit_q = float(np.mean([1.0 if gt_idx_q[i] >= 0 else 0.0 for i in h7]))
        print(f"  pool_hit@{POOL_K} (h7): {pool_hit_q:.4f}")

        # Run LambdaRank
        print(f"{ts()} Running CV5 LambdaRank...")
        case_ndcg_q, lr_scores_q = run_cv5_lambdarank(
            X_q, gt_idx_q, sizes_q, folds, FEAT_QUOTA, n)

        h7_ndcg_q = float(np.mean([case_ndcg_q[i] for i in h7]))
        dh7 = h7_ndcg_q - baseline_metrics["h7"]
        print(f"  h7={h7_ndcg_q:.5f}  dh7={dh7:+.5f}  pool_hit={pool_hit_q:.4f}")

        # Compute metrics
        q_metrics = compute_metrics(
            cases, case_ndcg_q, lr_scores_q, gt_idx_q, sizes_q, quota_pools,
            h7, ta, bucket_labels,
            case_ndcg_bl, lr_scores_bl, gt_idx_bl, pools_bl, sizes_bl)
        q_metrics["dh7"] = round(q_metrics["h7"] - baseline_metrics["h7"], 5)
        q_metrics["dpool_hit"] = round(q_metrics["pool_hit"] - baseline_metrics["pool_hit"], 4)

        all_config_metrics[config_name] = q_metrics
        print(f"  {config_name}: {json.dumps(q_metrics, indent=2)}")

        # Cleanup
        del X_q, gt_idx_q, sizes_q, lr_scores_q, case_ndcg_q, quota_pools
        gc.collect()

    # ---------------------------------------------------------------
    # STEP 7: Summary and gate check
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("R52b AUDIO QUOTA ADMISSION SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'config':<20} {'h7':>8} {'dh7':>8} {'pool':>6} {'dpool':>7} "
          f"{'same':>7} {'diff':>7} {'E_adm':>5} {'D_adm':>5} {'E_r20':>5} "
          f"{'rec':>4} {'lost':>4} {'net':>4} {'J20':>5} {'t1chg':>5}")
    print(f"  {'-'*115}")

    print(f"  {'BASELINE (R39)':<20} {baseline_metrics['h7']:>8.5f} {'---':>8} "
          f"{baseline_metrics['pool_hit']:>6.4f} {'---':>7} "
          f"{baseline_metrics['same_h7']:>7.5f} {baseline_metrics['diff_h7']:>7.5f} "
          f"{'---':>5} {'---':>5} {'---':>5} {'---':>4} {'---':>4} {'---':>4} {'---':>5} {'---':>5}")

    for cname, cmetrics in all_config_metrics.items():
        print(f"  {cname:<20} {cmetrics['h7']:>8.5f} {cmetrics['dh7']:>+8.5f} "
              f"{cmetrics['pool_hit']:>6.4f} {cmetrics['dpool_hit']:>+7.4f} "
              f"{cmetrics['same_h7']:>7.5f} {cmetrics['diff_h7']:>7.5f} "
              f"{cmetrics['E_admitted']:>5} {cmetrics['D_admitted']:>5} {cmetrics['E_recovered_top20']:>5} "
              f"{cmetrics['recovered']:>4} {cmetrics['lost']:>4} {cmetrics['net']:>4} "
              f"{cmetrics['top20_overlap']:>5.3f} {cmetrics['top1_changed']:>5}")

    # Gate check
    print(f"\n{'='*70}")
    print("GATE CHECK")
    print(f"{'='*70}")

    gate_pass = False
    gate_reason = ""
    best_config = ""

    for cname, cmetrics in all_config_metrics.items():
        dh7 = cmetrics["dh7"]
        dpool = cmetrics["dpool_hit"]
        diff_h7_improved = cmetrics["diff_h7"] > baseline_metrics["diff_h7"]

        if dh7 >= 0.010:
            gate_pass = True
            gate_reason = f"{cname}: dh7={dh7:+.5f} >= +0.010"
            best_config = cname
            break
        elif dpool >= 0.020 and dh7 >= 0.0 and diff_h7_improved:
            gate_pass = True
            gate_reason = f"{cname}: dpool={dpool:+.4f} >= +0.020, dh7={dh7:+.5f} >= 0, diff_h7 improved"
            best_config = cname
            break

    if not gate_pass:
        # Find best config by dh7
        best_config = max(all_config_metrics.keys(), key=lambda k: all_config_metrics[k]["dh7"])
        best_m = all_config_metrics[best_config]

        # Diagnostic: check if E cases are admitted but not recovered
        any_e_admitted = any(m["E_admitted"] > 0 for m in all_config_metrics.values())
        any_e_recovered = any(m["E_recovered_top20"] > 0 for m in all_config_metrics.values())

        if any_e_admitted and not any_e_recovered:
            gate_reason = ("Audio quota admits Bucket E cases to pool but ranker cannot rank them into top-20. "
                           "Audio is diagnostic-only: finds unreachable tracks but signal too weak for ranking.")
        elif any_e_admitted and any_e_recovered:
            max_e_rec = max(m["E_recovered_top20"] for m in all_config_metrics.values())
            gate_reason = (f"Audio quota admits E cases and recovers {max_e_rec} to top-20, "
                           f"but overall dh7={best_m['dh7']:+.5f} < +0.010 threshold. "
                           f"Recovery insufficient to offset disruption from pool modification.")
        else:
            gate_reason = (f"No E cases admitted (audio candidates already in R39 pool or quota too small). "
                           f"Best dh7={best_m['dh7']:+.5f}.")

    recommendation = "proceed to blind" if gate_pass else "archive audio quota as diagnostic-only; R39 remains production best"

    print(f"  Gate pass: {gate_pass}")
    print(f"  Reason: {gate_reason}")
    print(f"  Best config: {best_config}")
    print(f"  Recommendation: {recommendation}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "experiment": "R52b_audio_quota_admission",
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "h7": baseline_metrics["h7"],
            "pool_hit": baseline_metrics["pool_hit"],
            "same_h7": baseline_metrics["same_h7"],
            "diff_h7": baseline_metrics["diff_h7"],
        },
        "configs": all_config_metrics,
        "best_config": best_config,
        "gate": {
            "pass": gate_pass,
            "reason": gate_reason,
        },
        "recommendation": recommendation,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved results to {OUT_PATH}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
