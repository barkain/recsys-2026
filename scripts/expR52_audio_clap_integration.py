#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R52: Audio CLAP integration experiment.

Tests 3 integration approaches for audio CLAP embeddings:
  Config 1 (feature_only):  Add 3 audio features to R39 pool (unchanged pool)
  Config 2 (pool_add_low):  Add CLAP as 8th RRF source at low weights [0.03, 0.05, 0.10]
  Config 3 (gated_pool_add): Add CLAP only for discovery-regime cases (weight 0.10)

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
OUT_PATH = REPO / "exp" / "eval" / "expR52_audio_clap_integration.json"
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
    "audio_clap_sim_max",
    "audio_clap_sim_recency",
    "audio_clap_rank_inv",
]
FEAT_FEATURE_ONLY = FEAT_R39 + FEAT_AUDIO  # 37 features


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_album_mapping():
    """Load album mapping from HF metadata dataset."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    track_artist_map = {}
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

        artist_name_raw = item.get("artist_name", [])
        if isinstance(artist_name_raw, list):
            artist_name = ", ".join(str(a) for a in artist_name_raw)
        else:
            artist_name = str(artist_name_raw)
        track_artist_map[tid] = artist_name
    del ds
    return track_album, track_artist_map


def load_clap_embeddings():
    """Load CLAP embeddings from HF embeddings dataset. Returns (matrix, track_ids, tid_to_idx, n_valid)."""
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
    valid_mask = np.zeros(len(vecs), dtype=bool)
    n_valid = 0
    for i, v in enumerate(vecs):
        if v is not None and len(v) == dim:
            arr[i] = v
            if np.any(arr[i] != 0):
                valid_mask[i] = True
                n_valid += 1

    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms

    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    del ds, vecs
    gc.collect()

    return arr, track_ids, tid_to_idx, n_valid, valid_mask


def build_clap_retrieval_lists(cases, clap_matrix, clap_track_ids, clap_tid_to_idx):
    """Build CLAP history_recency retrieval lists for all cases.

    Returns:
        clap_lists: list of top-300 track ID lists per case
        clap_all_sims: list of full similarity arrays per case (for feature computation)
    """
    n = len(cases)
    clap_lists = []
    clap_all_sims = []

    for i, c in enumerate(cases):
        played = c["music_turns"]
        if not played:
            clap_lists.append([])
            clap_all_sims.append(None)
            continue

        n_played = len(played)
        valid_idx = []
        weights = []
        for pos, t in enumerate(played):
            if t in clap_tid_to_idx:
                valid_idx.append(clap_tid_to_idx[t])
                weights.append(0.9 ** (n_played - 1 - pos))

        if not valid_idx:
            clap_lists.append([])
            clap_all_sims.append(None)
            continue

        # Compute weighted query
        hist_embs = clap_matrix[valid_idx]  # (k, 512)
        w = np.array(weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        query = (hist_embs * w[:, None]).sum(axis=0)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Compute cosine similarity with all catalog tracks
        sims = clap_matrix @ query  # (N,)

        # Exclude played tracks
        played_idx_set = {clap_tid_to_idx[t] for t in played if t in clap_tid_to_idx}
        sims_copy = sims.copy()
        for pi in played_idx_set:
            sims_copy[pi] = -np.inf

        # Top-300 by argpartition
        topk = min(300, len(sims_copy))
        top = np.argpartition(-sims_copy, topk)[:topk]
        top = top[np.argsort(-sims_copy[top])]
        top_tids = [clap_track_ids[j] for j in top]

        clap_lists.append(top_tids)
        clap_all_sims.append(sims)  # raw sims (before excluding played)

        if (i + 1) % 2000 == 0:
            print(f"    {ts()} CLAP retrieval: {i+1}/{n} cases processed")

    return clap_lists, clap_all_sims


def build_r39_pools_and_features(cases, payload, als_source, als_vecs,
                                 r21_source, track_pop, max_pop, track_album,
                                 als_factors, als_track_to_idx,
                                 feat_names, n_feat, n_feat_base,
                                 source_weights, clap_lists=None,
                                 clap_matrix=None, clap_tid_to_idx=None,
                                 clap_all_sims=None,
                                 per_case_sw=None):
    """Build pools and feature matrix. Returns (X, gt_idx, sizes, pools, src_lists_all).

    Args:
        per_case_sw: If provided, list of per-case source weights (for gated config)
    """
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    n = len(cases)
    has_audio_feats = len(feat_names) > len(FEAT_R39)

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = []
    src_lists_all: list[dict[str, list[str]]] = []

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        # Add CLAP source if in source_weights or per_case_sw
        sw = per_case_sw[i] if per_case_sw is not None else source_weights
        if "CLAP" in sw and clap_lists is not None:
            src_lists["CLAP"] = clap_lists[i]

        pool = weighted_rrf(src_lists, sw, topk=POOL_K, k=RRF_K)
        pools.append(pool)
        src_lists_all.append(src_lists)
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

        # Audio feature precomputation (for feature_only config)
        hist_clap_embs = None
        recency_embs = None
        rw = None
        clap_rank_map: dict[str, int] = {}
        if has_audio_feats and clap_matrix is not None and clap_tid_to_idx is not None:
            # Get history track CLAP embeddings for max-sim computation
            hist_clap_idx = [clap_tid_to_idx[t] for t in played if t in clap_tid_to_idx]
            hist_clap_embs = clap_matrix[hist_clap_idx] if hist_clap_idx else None  # (k, 512)

            # Recency weights for history
            n_played = len(played)
            recency_weights = []
            recency_idx = []
            for pos, t in enumerate(played):
                if t in clap_tid_to_idx:
                    recency_idx.append(clap_tid_to_idx[t])
                    recency_weights.append(0.9 ** (n_played - 1 - pos))
            if recency_weights:
                rw = np.array(recency_weights, dtype=np.float32)
                rw_sum = rw.sum()
                if rw_sum > 0:
                    rw = rw / rw_sum
                recency_embs = clap_matrix[recency_idx]

            # CLAP rank map for this case
            if clap_lists is not None and i < len(clap_lists):
                for r, tid in enumerate(clap_lists[i], start=1):
                    clap_rank_map[tid] = r

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features (EXACT copy from R39)
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

            # Audio CLAP features (indices 34-36, only for feature_only config)
            if has_audio_feats and clap_matrix is not None and clap_tid_to_idx is not None:
                n_r39 = len(FEAT_R39)
                if tid in clap_tid_to_idx:
                    cand_emb = clap_matrix[clap_tid_to_idx[tid]]  # (512,)

                    # audio_clap_sim_max: max cosine sim with ANY history track
                    if hist_clap_embs is not None and len(hist_clap_embs) > 0:
                        sims_with_hist = hist_clap_embs @ cand_emb  # (k,)
                        row[n_r39 + 0] = float(np.max(sims_with_hist))

                        # audio_clap_sim_recency: recency-weighted cosine sim
                        if recency_embs is not None and rw is not None:
                            sims_recency = recency_embs @ cand_emb  # (k,)
                            row[n_r39 + 1] = float(np.dot(rw, sims_recency))

                    # audio_clap_rank_inv: 1/rank if in CLAP top-300
                    clap_r = clap_rank_map.get(tid)
                    if clap_r is not None:
                        row[n_r39 + 2] = 1.0 / clap_r

    return X, gt_idx, sizes, pools, src_lists_all


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
                    baseline_ndcg=None, baseline_lr_scores=None,
                    baseline_gt_idx=None, baseline_pools=None,
                    baseline_sizes=None):
    """Compute all metrics for a config."""
    # h7 nDCG
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))

    # pool_hit
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

    # Bucket recovery
    bucket_e_recovered = 0
    bucket_d_recovered = 0
    if bucket_labels:
        for i in h7:
            si = str(i)
            if si not in bucket_labels:
                continue
            bl = bucket_labels[si]
            if bl["bucket"] == "E" and gt_idx[i] >= 0:
                bucket_e_recovered += 1
            elif bl["bucket"] == "D" and gt_idx[i] >= 0:
                bucket_d_recovered += 1

    # recovered / lost / net / top20_overlap / top1_changed
    recovered = 0
    lost = 0
    top20_jaccard_vals = []
    top1_changed = 0

    if baseline_lr_scores is not None:
        for i in h7:
            ps = int(sizes[i])
            bps = int(baseline_sizes[i]) if baseline_sizes is not None else 0

            # Compute config rank of GT
            config_rank = -1
            if gt_idx[i] >= 0 and ps > 0:
                sc = lr_scores[i, :ps]
                ranked = np.argsort(-sc)
                gt_pos_arr = np.where(ranked == gt_idx[i])[0]
                if len(gt_pos_arr) > 0:
                    config_rank = int(gt_pos_arr[0]) + 1

            # Compute baseline rank of GT
            baseline_rank = -1
            if baseline_gt_idx is not None and baseline_gt_idx[i] >= 0 and bps > 0:
                bsc = baseline_lr_scores[i, :bps]
                branked = np.argsort(-bsc)
                bgt_pos_arr = np.where(branked == baseline_gt_idx[i])[0]
                if len(bgt_pos_arr) > 0:
                    baseline_rank = int(bgt_pos_arr[0]) + 1

            # recovered: baseline rank > 20 AND config rank <= 20
            if baseline_rank > 20 and 1 <= config_rank <= 20:
                recovered += 1
            elif 1 <= baseline_rank <= 20 and (config_rank > 20 or config_rank < 0):
                lost += 1

            # top-20 Jaccard
            config_top20 = set()
            if ps > 0:
                sc = lr_scores[i, :ps]
                ranked = np.argsort(-sc)
                config_top20 = {pools[i][int(ranked[j])] for j in range(min(20, ps))}

            baseline_top20 = set()
            if bps > 0 and baseline_pools is not None:
                bsc = baseline_lr_scores[i, :bps]
                branked = np.argsort(-bsc)
                baseline_top20 = {baseline_pools[i][int(branked[j])] for j in range(min(20, bps))}

            if config_top20 and baseline_top20:
                jaccard = len(config_top20 & baseline_top20) / len(config_top20 | baseline_top20)
                top20_jaccard_vals.append(jaccard)

            # top-1 changed
            config_top1 = pools[i][int(np.argsort(-lr_scores[i, :ps])[0])] if ps > 0 else None
            baseline_top1 = None
            if bps > 0 and baseline_pools is not None:
                baseline_top1 = baseline_pools[i][int(np.argsort(-baseline_lr_scores[i, :bps])[0])]
            if config_top1 != baseline_top1:
                top1_changed += 1

    net = recovered - lost
    top20_overlap = float(np.mean(top20_jaccard_vals)) if top20_jaccard_vals else 0.0

    return {
        "h7": round(h7_ndcg, 5),
        "pool_hit": round(pool_hit, 4),
        "same_h7": round(same_h7, 5),
        "diff_h7": round(diff_h7, 5),
        "E_recovered": bucket_e_recovered,
        "D_recovered": bucket_d_recovered,
        "recovered": recovered,
        "lost": lost,
        "net": net,
        "top20_overlap": round(top20_overlap, 4),
        "top1_changed": top1_changed,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R52: Audio CLAP Integration Experiment")
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
    track_album, _track_artist_map = load_album_mapping()

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
    # STEP 2: Load CLAP embeddings and build retrieval lists
    # ---------------------------------------------------------------
    print(f"\n{ts()} STEP 2: Loading CLAP embeddings")
    print("-" * 70)

    clap_matrix, clap_track_ids, clap_tid_to_idx, clap_valid, _ = load_clap_embeddings()
    print(f"  CLAP: shape={clap_matrix.shape}, valid={clap_valid}/{len(clap_track_ids)} "
          f"({clap_valid/len(clap_track_ids)*100:.1f}%)")

    print(f"{ts()} Building CLAP history_recency retrieval lists...")
    clap_lists, clap_all_sims = build_clap_retrieval_lists(
        cases, clap_matrix, clap_track_ids, clap_tid_to_idx)

    # Save CLAP lists cache
    CLAP_LISTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CLAP_LISTS_CACHE, "w") as f:
        json.dump(clap_lists, f)
    print(f"  Saved CLAP lists to {CLAP_LISTS_CACHE}")

    # ---------------------------------------------------------------
    # BASELINE: R39 reproduction
    # ---------------------------------------------------------------
    print(f"\n{ts()} BASELINE: R39 Reproduction")
    print("-" * 70)

    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = len(FEAT_R39)

    print(f"{ts()} Building R39 feature matrix ({n_feat_r39} features)...")
    X_bl, gt_idx_bl, sizes_bl, pools_bl, src_lists_bl = build_r39_pools_and_features(
        cases, payload, als_source, als_vecs, r21_source,
        track_pop, max_pop, track_album,
        als_factors, als_track_to_idx,
        FEAT_R39, n_feat_r39, n_feat_base,
        SW_R39)

    pool_hit_all_bl = float(np.mean(gt_idx_bl >= 0))
    print(f"  pool_hit@{POOL_K} (all): {pool_hit_all_bl:.4f}")

    print(f"{ts()} Running CV5 LambdaRank (R39 baseline)...")
    case_ndcg_bl, lr_scores_bl = run_cv5_lambdarank(X_bl, gt_idx_bl, sizes_bl, folds, FEAT_R39, n)

    h7_ndcg_bl = float(np.mean([case_ndcg_bl[i] for i in h7]))
    pool_hit_h7_bl = float(np.mean([1.0 if gt_idx_bl[i] >= 0 else 0.0 for i in h7]))
    print(f"  R39 reproduction: h7={h7_ndcg_bl:.5f}  pool_hit_all={pool_hit_all_bl:.4f}  pool_hit_h7={pool_hit_h7_bl:.4f}")

    # HARD CHECK: h7 on h7 cases, pool_hit on ALL cases (matches reference scripts)
    expected_h7 = 0.24298
    expected_pool_hit = 0.6000
    if abs(h7_ndcg_bl - expected_h7) > 0.001:
        print(f"\n  *** HARD CHECK FAILED: h7={h7_ndcg_bl:.5f}, expected ~{expected_h7} ***")
        sys.exit(1)
    if abs(pool_hit_all_bl - expected_pool_hit) > 0.002:
        print(f"\n  *** POOL HIT CHECK FAILED: pool_hit_all={pool_hit_all_bl:.4f}, expected ~{expected_pool_hit} ***")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={h7_ndcg_bl:.5f}, pool_hit_all={pool_hit_all_bl:.4f}")

    baseline_metrics = compute_metrics(
        cases, case_ndcg_bl, lr_scores_bl, gt_idx_bl, sizes_bl, pools_bl,
        h7, ta, bucket_labels)
    print(f"  Baseline metrics: {json.dumps(baseline_metrics, indent=2)}")

    # ---------------------------------------------------------------
    # CONFIG 1: feature_only (unchanged R39 pool + 3 audio features)
    # ---------------------------------------------------------------
    print(f"\n{ts()} CONFIG 1: feature_only (R39 pool + 3 audio features)")
    print("-" * 70)

    n_feat_fo = len(FEAT_FEATURE_ONLY)
    print(f"{ts()} Building feature_only matrix ({n_feat_fo} features)...")

    X_fo, gt_idx_fo, sizes_fo, pools_fo, _ = build_r39_pools_and_features(
        cases, payload, als_source, als_vecs, r21_source,
        track_pop, max_pop, track_album,
        als_factors, als_track_to_idx,
        FEAT_FEATURE_ONLY, n_feat_fo, n_feat_base,
        SW_R39,
        clap_lists=clap_lists,
        clap_matrix=clap_matrix,
        clap_tid_to_idx=clap_tid_to_idx,
        clap_all_sims=clap_all_sims)

    print(f"{ts()} Running CV5 LambdaRank (feature_only)...")
    case_ndcg_fo, lr_scores_fo = run_cv5_lambdarank(X_fo, gt_idx_fo, sizes_fo, folds, FEAT_FEATURE_ONLY, n)

    h7_ndcg_fo = float(np.mean([case_ndcg_fo[i] for i in h7]))
    pool_hit_h7_fo = float(np.mean([1.0 if gt_idx_fo[i] >= 0 else 0.0 for i in h7]))
    print(f"  feature_only: h7={h7_ndcg_fo:.5f}  pool_hit={pool_hit_h7_fo:.4f}")

    fo_metrics = compute_metrics(
        cases, case_ndcg_fo, lr_scores_fo, gt_idx_fo, sizes_fo, pools_fo,
        h7, ta, bucket_labels,
        baseline_ndcg=case_ndcg_bl, baseline_lr_scores=lr_scores_bl,
        baseline_gt_idx=gt_idx_bl, baseline_pools=pools_bl,
        baseline_sizes=sizes_bl)
    fo_metrics["dh7"] = round(fo_metrics["h7"] - baseline_metrics["h7"], 5)
    fo_metrics["dpool_hit"] = round(fo_metrics["pool_hit"] - baseline_metrics["pool_hit"], 4)
    print(f"  feature_only metrics: {json.dumps(fo_metrics, indent=2)}")

    # Free feature_only arrays
    del X_fo, gt_idx_fo, sizes_fo, lr_scores_fo, case_ndcg_fo
    gc.collect()

    # ---------------------------------------------------------------
    # CONFIG 2: pool_add_low (CLAP as 8th RRF source at low weights)
    # ---------------------------------------------------------------
    print(f"\n{ts()} CONFIG 2: pool_add_low (CLAP as 8th RRF source)")
    print("-" * 70)

    pool_add_results = {}
    for audio_weight in [0.03, 0.05, 0.10]:
        config_name = f"pool_add_{audio_weight:.2f}"
        sw = dict(SW_R39)
        sw["CLAP"] = audio_weight
        print(f"\n{ts()} {config_name}: CLAP weight={audio_weight}")

        X_pa, gt_idx_pa, sizes_pa, pools_pa, _ = build_r39_pools_and_features(
            cases, payload, als_source, als_vecs, r21_source,
            track_pop, max_pop, track_album,
            als_factors, als_track_to_idx,
            FEAT_R39, n_feat_r39, n_feat_base,
            sw,
            clap_lists=clap_lists)

        pool_hit_h7_pa = float(np.mean([1.0 if gt_idx_pa[i] >= 0 else 0.0 for i in h7]))
        print(f"  pool_hit: {pool_hit_h7_pa:.4f}")

        print("  Running CV5 LambdaRank...")
        case_ndcg_pa, lr_scores_pa = run_cv5_lambdarank(X_pa, gt_idx_pa, sizes_pa, folds, FEAT_R39, n)

        h7_ndcg_pa = float(np.mean([case_ndcg_pa[i] for i in h7]))
        print(f"  {config_name}: h7={h7_ndcg_pa:.5f}  pool_hit={pool_hit_h7_pa:.4f}")

        pa_metrics = compute_metrics(
            cases, case_ndcg_pa, lr_scores_pa, gt_idx_pa, sizes_pa, pools_pa,
            h7, ta, bucket_labels,
            baseline_ndcg=case_ndcg_bl, baseline_lr_scores=lr_scores_bl,
            baseline_gt_idx=gt_idx_bl, baseline_pools=pools_bl,
            baseline_sizes=sizes_bl)
        pa_metrics["dh7"] = round(pa_metrics["h7"] - baseline_metrics["h7"], 5)
        pa_metrics["dpool_hit"] = round(pa_metrics["pool_hit"] - baseline_metrics["pool_hit"], 4)
        pool_add_results[config_name] = pa_metrics
        print(f"  {config_name} metrics: {json.dumps(pa_metrics, indent=2)}")

        del X_pa, gt_idx_pa, sizes_pa, lr_scores_pa, case_ndcg_pa
        gc.collect()

    # ---------------------------------------------------------------
    # CONFIG 3: gated_pool_add (CLAP only for discovery-regime cases)
    # ---------------------------------------------------------------
    print(f"\n{ts()} CONFIG 3: gated_pool_add (CLAP for discovery cases only)")
    print("-" * 70)

    # Build per-case source weights
    per_case_sw = []
    discovery_count = 0
    for i, c in enumerate(cases):
        played = c["music_turns"]
        played_artists = [ta.get(t, "") for t in played]
        unique_artists = len(set(a for a in played_artists if a))
        last3_artists = played_artists[-3:] if len(played_artists) >= 3 else played_artists
        all_same = len(set(last3_artists)) == 1

        if unique_artists >= 5 and not all_same:
            # Discovery regime
            sw = dict(SW_R39)
            sw["CLAP"] = 0.10
            per_case_sw.append(sw)
            discovery_count += 1
        else:
            per_case_sw.append(dict(SW_R39))

    print(f"  Discovery cases: {discovery_count}/{n} ({discovery_count/n*100:.1f}%)")
    h7_discovery = sum(1 for i in h7
                       if "CLAP" in per_case_sw[i])
    print(f"  h7 discovery: {h7_discovery}/{len(h7)} ({h7_discovery/len(h7)*100:.1f}%)")

    X_gp, gt_idx_gp, sizes_gp, pools_gp, _ = build_r39_pools_and_features(
        cases, payload, als_source, als_vecs, r21_source,
        track_pop, max_pop, track_album,
        als_factors, als_track_to_idx,
        FEAT_R39, n_feat_r39, n_feat_base,
        SW_R39,  # not used directly, per_case_sw takes over
        clap_lists=clap_lists,
        per_case_sw=per_case_sw)

    pool_hit_h7_gp = float(np.mean([1.0 if gt_idx_gp[i] >= 0 else 0.0 for i in h7]))
    print(f"  pool_hit: {pool_hit_h7_gp:.4f}")

    print(f"{ts()} Running CV5 LambdaRank (gated_pool_add)...")
    case_ndcg_gp, lr_scores_gp = run_cv5_lambdarank(X_gp, gt_idx_gp, sizes_gp, folds, FEAT_R39, n)

    h7_ndcg_gp = float(np.mean([case_ndcg_gp[i] for i in h7]))
    print(f"  gated_pool_add: h7={h7_ndcg_gp:.5f}  pool_hit={pool_hit_h7_gp:.4f}")

    gp_metrics = compute_metrics(
        cases, case_ndcg_gp, lr_scores_gp, gt_idx_gp, sizes_gp, pools_gp,
        h7, ta, bucket_labels,
        baseline_ndcg=case_ndcg_bl, baseline_lr_scores=lr_scores_bl,
        baseline_gt_idx=gt_idx_bl, baseline_pools=pools_bl,
        baseline_sizes=sizes_bl)
    gp_metrics["dh7"] = round(gp_metrics["h7"] - baseline_metrics["h7"], 5)
    gp_metrics["dpool_hit"] = round(gp_metrics["pool_hit"] - baseline_metrics["pool_hit"], 4)
    print(f"  gated_pool_add metrics: {json.dumps(gp_metrics, indent=2)}")

    del X_gp, gt_idx_gp, sizes_gp, lr_scores_gp, case_ndcg_gp
    gc.collect()

    # ---------------------------------------------------------------
    # STEP 6: Summary and gate check
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("R52 INTEGRATION SUMMARY")
    print(f"{'='*70}")

    all_configs = {
        "feature_only": fo_metrics,
        **pool_add_results,
        "gated_pool_add": gp_metrics,
    }

    print(f"\n  {'config':<20} {'h7':>8} {'dh7':>8} {'pool_hit':>9} {'dpool':>7} "
          f"{'same_h7':>8} {'diff_h7':>8} {'E_rec':>5} {'D_rec':>5} "
          f"{'rec':>4} {'lost':>4} {'net':>4} {'J20':>5} {'t1chg':>5}")
    print(f"  {'-'*110}")

    # Print baseline
    print(f"  {'BASELINE (R39)':<20} {baseline_metrics['h7']:>8.5f} {'---':>8} "
          f"{baseline_metrics['pool_hit']:>9.4f} {'---':>7} "
          f"{baseline_metrics['same_h7']:>8.5f} {baseline_metrics['diff_h7']:>8.5f} "
          f"{'---':>5} {'---':>5} {'---':>4} {'---':>4} {'---':>4} {'---':>5} {'---':>5}")

    for cname, cmetrics in all_configs.items():
        print(f"  {cname:<20} {cmetrics['h7']:>8.5f} {cmetrics['dh7']:>+8.5f} "
              f"{cmetrics['pool_hit']:>9.4f} {cmetrics['dpool_hit']:>+7.4f} "
              f"{cmetrics['same_h7']:>8.5f} {cmetrics['diff_h7']:>8.5f} "
              f"{cmetrics['E_recovered']:>5} {cmetrics['D_recovered']:>5} "
              f"{cmetrics['recovered']:>4} {cmetrics['lost']:>4} {cmetrics['net']:>4} "
              f"{cmetrics['top20_overlap']:>5.3f} {cmetrics['top1_changed']:>5}")

    # Gate check
    print(f"\n{'='*70}")
    print("GATE CHECK")
    print(f"{'='*70}")

    gate_pass = False
    gate_reason = ""
    best_config = ""

    for cname, cmetrics in all_configs.items():
        dh7 = cmetrics["dh7"]
        dpool = cmetrics["dpool_hit"]
        diff_h7_improved = cmetrics["diff_h7"] > baseline_metrics["diff_h7"]

        # Check gate criteria
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
        # Check R36 pattern: pool up, h7/same down
        pool_up_h7_down = all(
            m.get("dpool_hit", 0) > 0 and m.get("dh7", 0) < 0
            for cname, m in all_configs.items()
            if "pool_add" in cname
        )
        if pool_up_h7_down:
            gate_reason = "R36 pattern repeats: pool up + h7 down for all pool_add configs"
        else:
            gate_reason = "No config meets gate criteria (dh7 >= +0.010 or dpool >= +0.020 with h7 non-negative)"
        # Pick config with best dh7 as 'best' for reporting
        best_config = max(all_configs.keys(), key=lambda k: all_configs[k]["dh7"])

    recommendation = "proceed to blind" if gate_pass else "archive audio as diagnostic-only"

    print(f"  Gate pass: {gate_pass}")
    print(f"  Reason: {gate_reason}")
    print(f"  Best config: {best_config}")
    print(f"  Recommendation: {recommendation}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "experiment": "R52_audio_clap_integration",
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "h7": baseline_metrics["h7"],
            "pool_hit": baseline_metrics["pool_hit"],
            "same_h7": baseline_metrics["same_h7"],
            "diff_h7": baseline_metrics["diff_h7"],
        },
        "configs": all_configs,
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
