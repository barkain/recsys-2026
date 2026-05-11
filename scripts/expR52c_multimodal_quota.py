#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R52c: Multimodal quota experiment.

Audio CLAP + Image SigLIP quota-based pool modification.
R52b showed audio quota_20 admits 6 Bucket E cases with 5 recovered (83%).
Union diagnostic showed audio+image recovers 54 Bucket E (vs 40 audio-only).

4 configs:
  audio_q20:                audio=20, image=0,  total=20, all cases
  audio15_image5:           audio=15, image=5,  total=20, all cases
  audio20_image10:          audio=20, image=10, total=30, all cases
  audio20_image10_gated:    audio=20, image=10, total=30, discovery only

Baseline: R39 album-aware LambdaRank (h7=0.24298, pool_hit=0.598).
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
IMAGE_LISTS_CACHE = REPO / "cache" / "r52_image_siglip_lists.json"
OUT_PATH = REPO / "exp" / "eval" / "expR52c_multimodal_quota.json"
RRF_K = 20
POOL_K = 300
SW_R39 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R39 = FEAT_BASE + FEAT_ALBUM  # 34 features

FEAT_MULTIMODAL = [
    "audio_rank_inv",
    "audio_presence",
    "audio_cosine",
    "image_rank_inv",
    "image_presence",
    "image_cosine",
    "source_added_audio",
    "source_added_image",
]
FEAT_ALL = FEAT_R39 + FEAT_MULTIMODAL  # 42 features


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
    """Load CLAP embeddings (512 dims) from HF dataset. Returns (matrix, track_ids, tid_to_idx, n_valid)."""
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


def load_siglip_embeddings():
    """Load SigLIP embeddings (768 dims) from HF dataset. Returns (matrix, track_ids, tid_to_idx, n_valid)."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]

    track_ids = []
    vecs = []
    dim = 768
    for item in ds:
        track_ids.append(str(item["track_id"]))
        v = item["image-siglip2"]
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


def compute_query_embedding(played, emb_matrix, tid_to_idx):
    """Compute recency-weighted query: weight = 0.9^(n-1-pos), L2-normalized.
    Returns query vector or None.
    """
    n_played = len(played)
    valid_idx = []
    weights = []
    for pos, t in enumerate(played):
        if t in tid_to_idx:
            valid_idx.append(tid_to_idx[t])
            weights.append(0.9 ** (n_played - 1 - pos))

    if not valid_idx:
        return None

    hist_embs = emb_matrix[valid_idx]
    w = np.array(weights, dtype=np.float32)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    query = (hist_embs * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm
    return query


def build_image_retrieval_lists(cases, siglip_matrix, siglip_track_ids, siglip_tid_to_idx):
    """Build SigLIP history_recency retrieval lists (top-300 per case).
    Same strategy as audio CLAP: recency-weighted average, cosine, exclude played.
    """
    n = len(cases)
    image_lists = []

    for i, c in enumerate(cases):
        played = c["music_turns"]
        if not played:
            image_lists.append([])
            continue

        query = compute_query_embedding(played, siglip_matrix, siglip_tid_to_idx)
        if query is None:
            image_lists.append([])
            continue

        sims = siglip_matrix @ query

        # Exclude played tracks
        played_idx_set = {siglip_tid_to_idx[t] for t in played if t in siglip_tid_to_idx}
        sims_copy = sims.copy()
        for pi in played_idx_set:
            sims_copy[pi] = -np.inf

        topk = min(300, len(sims_copy))
        top = np.argpartition(-sims_copy, topk)[:topk]
        top = top[np.argsort(-sims_copy[top])]
        top_tids = [siglip_track_ids[j] for j in top]

        image_lists.append(top_tids)

        if (i + 1) % 2000 == 0:
            print(f"    {ts()} Image retrieval: {i+1}/{n} cases processed")

    return image_lists


def build_r39_baseline(cases, payload, als_source, als_vecs, r21_source,
                       track_pop, max_pop, track_album,
                       als_factors, als_track_to_idx, folds):
    """Build R39 baseline pools and features. Returns (X, gt_idx, sizes, pools, rrf_scores_all)."""
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


def build_multimodal_quota_pool(r39_pool, audio_top300, image_top300,
                                audio_quota, image_quota):
    """Build a multimodal quota-modified pool.

    1. Start with R39 pool (300 candidates)
    2. audio_new = audio candidates NOT in R39 pool, take[:audio_quota]
    3. image_new = image candidates NOT in R39 pool AND NOT in audio_new, take[:image_quota]
    4. Replace bottom N candidates with audio_new + image_new

    Returns (modified_pool, audio_added_set, image_added_set).
    """
    r39_set = set(r39_pool)
    audio_new = [t for t in audio_top300 if t not in r39_set][:audio_quota]
    audio_new_set = set(audio_new)

    image_new = []
    if image_quota > 0:
        image_new = [t for t in image_top300
                     if t not in r39_set and t not in audio_new_set][:image_quota]

    n_to_add = len(audio_new) + len(image_new)
    if n_to_add == 0:
        return list(r39_pool), set(), set()

    # Remove bottom N candidates (lowest RRF = end of pool)
    modified = list(r39_pool[:len(r39_pool) - n_to_add])
    modified.extend(audio_new)
    modified.extend(image_new)

    return modified, set(audio_new), set(image_new)


def build_multimodal_features(cases, payload, als_source, als_vecs, r21_source,
                              track_pop, max_pop, track_album,
                              als_factors, als_track_to_idx,
                              quota_pools, audio_added_sets, image_added_sets,
                              clap_lists, clap_matrix, clap_tid_to_idx,
                              image_lists, siglip_matrix, siglip_tid_to_idx):
    """Build 42-feature matrix for multimodal quota-modified pools."""
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    n = len(cases)
    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = len(FEAT_R39)
    n_feat = len(FEAT_ALL)

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
        clap_query = compute_query_embedding(played, clap_matrix, clap_tid_to_idx)

        # Image SigLIP precomputation
        image_rank_map: dict[str, int] = {}
        if image_lists[i]:
            for r, tid in enumerate(image_lists[i], start=1):
                image_rank_map[tid] = r
        siglip_query = compute_query_embedding(played, siglip_matrix, siglip_tid_to_idx)

        # Added sets for this case
        audio_added = audio_added_sets[i]
        image_added = image_added_sets[i]

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

            # --- Multimodal features (indices 34-41) ---

            # audio_rank_inv (idx 34)
            clap_r = clap_rank_map.get(tid)
            row[n_feat_r39 + 0] = 1.0 / clap_r if clap_r is not None else 0.0

            # audio_presence (idx 35)
            row[n_feat_r39 + 1] = 1.0 if clap_r is not None else 0.0

            # audio_cosine (idx 36)
            if clap_query is not None and tid in clap_tid_to_idx:
                cand_emb = clap_matrix[clap_tid_to_idx[tid]]
                row[n_feat_r39 + 2] = float(np.dot(clap_query, cand_emb))

            # image_rank_inv (idx 37)
            img_r = image_rank_map.get(tid)
            row[n_feat_r39 + 3] = 1.0 / img_r if img_r is not None else 0.0

            # image_presence (idx 38)
            row[n_feat_r39 + 4] = 1.0 if img_r is not None else 0.0

            # image_cosine (idx 39)
            if siglip_query is not None and tid in siglip_tid_to_idx:
                cand_img_emb = siglip_matrix[siglip_tid_to_idx[tid]]
                row[n_feat_r39 + 5] = float(np.dot(siglip_query, cand_img_emb))

            # source_added_audio (idx 40)
            row[n_feat_r39 + 6] = 1.0 if tid in audio_added else 0.0

            # source_added_image (idx 41)
            row[n_feat_r39 + 7] = 1.0 if tid in image_added else 0.0

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
                    audio_added_sets=None, image_added_sets=None,
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
    bucket_d_recovered_top20 = 0
    if bucket_labels:
        for i in h7:
            si = str(i)
            if si not in bucket_labels:
                continue
            bl = bucket_labels[si]
            if bl["bucket"] == "E" and gt_idx[i] >= 0:
                bucket_e_admitted += 1
                ps = int(sizes[i])
                if ps > 0:
                    sc = lr_scores[i, :ps]
                    ranked = np.argsort(-sc)
                    gt_pos_arr = np.where(ranked == gt_idx[i])[0]
                    if len(gt_pos_arr) > 0 and gt_pos_arr[0] < 20:
                        bucket_e_recovered_top20 += 1
            elif bl["bucket"] == "D" and gt_idx[i] >= 0:
                bucket_d_admitted += 1
                ps = int(sizes[i])
                if ps > 0:
                    sc = lr_scores[i, :ps]
                    ranked = np.argsort(-sc)
                    gt_pos_arr = np.where(ranked == gt_idx[i])[0]
                    if len(gt_pos_arr) > 0 and gt_pos_arr[0] < 20:
                        bucket_d_recovered_top20 += 1

    # recovered / lost / net / top20_overlap / top1_changed
    recovered = 0
    lost = 0
    top20_overlap_vals = []
    top1_changed = 0

    # Per-source contribution for recovered cases
    recovered_from_audio = 0
    recovered_from_image = 0
    recovered_from_existing = 0

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
                # Determine source of recovered GT
                gt_tid = cases[i]["gt"]
                if audio_added_sets is not None and gt_tid in audio_added_sets[i]:
                    recovered_from_audio += 1
                elif image_added_sets is not None and gt_tid in image_added_sets[i]:
                    recovered_from_image += 1
                else:
                    recovered_from_existing += 1
            elif 1 <= baseline_rank <= 20 and (config_rank > 20 or config_rank < 0):
                lost += 1

            # top-20 overlap
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

    result = {
        "h7": round(h7_ndcg, 5),
        "pool_hit": round(pool_hit, 4),
        "same_h7": round(same_h7, 5),
        "diff_h7": round(diff_h7, 5),
        "bucket_E_admitted": bucket_e_admitted,
        "bucket_D_admitted": bucket_d_admitted,
        "E_recovered_top20": bucket_e_recovered_top20,
        "D_recovered_top20": bucket_d_recovered_top20,
        "recovered": recovered,
        "lost": lost,
        "net": net,
        "top20_overlap": round(top20_overlap, 4),
        "top1_changed": top1_changed,
    }
    if not is_baseline:
        result["recovered_from_audio"] = recovered_from_audio
        result["recovered_from_image"] = recovered_from_image
        result["recovered_from_existing"] = recovered_from_existing

    return result


def main():
    t0 = time.time()
    print(f"{ts()} R52c: Multimodal Quota Experiment")
    print("=" * 70)

    # ---------------------------------------------------------------
    # STEP 1: Load shared data + R39 baseline reproduction
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

    # --- R39 baseline ---
    print(f"\n{ts()} R39 Baseline Reproduction")
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

    # Free baseline feature matrix
    del X_bl
    gc.collect()

    # ---------------------------------------------------------------
    # STEP 2: Load embedding sources
    # ---------------------------------------------------------------
    print(f"\n{ts()} STEP 2: Loading embedding sources")
    print("-" * 70)

    # Audio CLAP lists (from cache)
    print(f"{ts()} Loading CLAP retrieval lists from cache...")
    with open(CLAP_LISTS_CACHE) as f:
        clap_lists = json.load(f)
    non_empty = sum(1 for x in clap_lists if len(x) > 0)
    print(f"  CLAP lists: {len(clap_lists)} total, {non_empty} non-empty")

    # Audio CLAP embeddings (for cosine features)
    print(f"{ts()} Loading CLAP embeddings...")
    clap_matrix, clap_track_ids, clap_tid_to_idx, clap_valid = load_clap_embeddings()
    print(f"  CLAP: shape={clap_matrix.shape}, valid={clap_valid}/{len(clap_track_ids)}")

    # Image SigLIP embeddings
    print(f"{ts()} Loading SigLIP embeddings...")
    siglip_matrix, siglip_track_ids, siglip_tid_to_idx, siglip_valid = load_siglip_embeddings()
    print(f"  SigLIP: shape={siglip_matrix.shape}, valid={siglip_valid}/{len(siglip_track_ids)}")

    # Build image retrieval lists (or load from cache)
    if IMAGE_LISTS_CACHE.exists():
        print(f"{ts()} Loading image retrieval lists from cache...")
        with open(IMAGE_LISTS_CACHE) as f:
            image_lists = json.load(f)
        non_empty_img = sum(1 for x in image_lists if len(x) > 0)
        print(f"  Image lists: {len(image_lists)} total, {non_empty_img} non-empty")
    else:
        print(f"{ts()} Building image retrieval lists (history_recency)...")
        image_lists = build_image_retrieval_lists(
            cases, siglip_matrix, siglip_track_ids, siglip_tid_to_idx)
        non_empty_img = sum(1 for x in image_lists if len(x) > 0)
        print(f"  Image lists: {len(image_lists)} total, {non_empty_img} non-empty")

        # Save cache
        IMAGE_LISTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(IMAGE_LISTS_CACHE, "w") as f:
            json.dump(image_lists, f)
        print(f"  Saved to {IMAGE_LISTS_CACHE}")

    # ---------------------------------------------------------------
    # STEP 3-5: Run 4 quota configs
    # ---------------------------------------------------------------
    quota_configs = {
        "audio_q20": {"audio_quota": 20, "image_quota": 0, "total_quota": 20, "gated": False},
        "audio15_image5": {"audio_quota": 15, "image_quota": 5, "total_quota": 20, "gated": False},
        "audio20_image10": {"audio_quota": 20, "image_quota": 10, "total_quota": 30, "gated": False},
        "audio20_image10_gated": {"audio_quota": 20, "image_quota": 10, "total_quota": 30, "gated": True},
    }

    # Precompute discovery mask
    discovery_mask = np.zeros(n, dtype=bool)
    for i, c in enumerate(cases):
        played = c["music_turns"]
        played_artists = [ta.get(t, "") for t in played]
        n_unique_artists = len(set(a for a in played_artists if a))
        last3 = played_artists[-3:] if len(played_artists) >= 3 else played_artists
        recent3_same = len(set(a for a in last3 if a)) == 1 and any(a for a in last3)
        if n_unique_artists >= 5 and not recent3_same:
            discovery_mask[i] = True

    h7_discovery = sum(1 for i in h7 if discovery_mask[i])
    print(f"\n  Discovery cases: {int(discovery_mask.sum())}/{n} total, "
          f"{h7_discovery}/{len(h7)} h7")

    all_config_metrics = {}

    for config_name, config in quota_configs.items():
        print(f"\n{ts()} CONFIG: {config_name}")
        print("-" * 70)

        audio_quota = config["audio_quota"]
        image_quota = config["image_quota"]
        total_quota = config["total_quota"]
        gated = config["gated"]

        # Build quota pools
        print(f"{ts()} Building quota pools (audio={audio_quota}, image={image_quota}, "
              f"total={total_quota}, gated={gated})...")
        quota_pools = []
        audio_added_sets = []
        image_added_sets = []
        n_modified = 0
        n_audio_new_total = 0
        n_image_new_total = 0

        for i in range(n):
            if gated and not discovery_mask[i]:
                # Non-discovery: use R39 pool unchanged
                quota_pools.append(pools_bl[i])
                audio_added_sets.append(set())
                image_added_sets.append(set())
                continue

            audio_top300 = clap_lists[i] if i < len(clap_lists) else []
            image_top300 = image_lists[i] if i < len(image_lists) else []

            if not audio_top300 and not image_top300:
                quota_pools.append(pools_bl[i])
                audio_added_sets.append(set())
                image_added_sets.append(set())
                continue

            modified, a_added, i_added = build_multimodal_quota_pool(
                pools_bl[i], audio_top300, image_top300,
                audio_quota, image_quota)
            quota_pools.append(modified)
            audio_added_sets.append(a_added)
            image_added_sets.append(i_added)

            n_new = len(a_added) + len(i_added)
            if n_new > 0:
                n_modified += 1
                n_audio_new_total += len(a_added)
                n_image_new_total += len(i_added)

        print(f"  Modified pools: {n_modified}/{n}")
        print(f"  Total audio new: {n_audio_new_total}, image new: {n_image_new_total}")
        if n_modified > 0:
            print(f"  Avg new per modified pool: {(n_audio_new_total+n_image_new_total)/n_modified:.1f}")

        # Build features
        print(f"{ts()} Building feature matrix ({len(FEAT_ALL)} features)...")
        X_q, gt_idx_q, sizes_q = build_multimodal_features(
            cases, payload, als_source, als_vecs, r21_source,
            track_pop, max_pop, track_album,
            als_factors, als_track_to_idx,
            quota_pools, audio_added_sets, image_added_sets,
            clap_lists, clap_matrix, clap_tid_to_idx,
            image_lists, siglip_matrix, siglip_tid_to_idx)

        pool_hit_q = float(np.mean([1.0 if gt_idx_q[i] >= 0 else 0.0 for i in h7]))
        print(f"  pool_hit@{POOL_K} (h7): {pool_hit_q:.4f}")

        # Run LambdaRank
        print(f"{ts()} Running CV5 LambdaRank...")
        case_ndcg_q, lr_scores_q = run_cv5_lambdarank(
            X_q, gt_idx_q, sizes_q, folds, FEAT_ALL, n)

        h7_ndcg_q = float(np.mean([case_ndcg_q[i] for i in h7]))
        dh7 = h7_ndcg_q - baseline_metrics["h7"]
        print(f"  h7={h7_ndcg_q:.5f}  dh7={dh7:+.5f}  pool_hit={pool_hit_q:.4f}")

        # Compute metrics
        q_metrics = compute_metrics(
            cases, case_ndcg_q, lr_scores_q, gt_idx_q, sizes_q, quota_pools,
            h7, ta, bucket_labels,
            case_ndcg_bl, lr_scores_bl, gt_idx_bl, pools_bl, sizes_bl,
            audio_added_sets=audio_added_sets,
            image_added_sets=image_added_sets)
        q_metrics["dh7"] = round(q_metrics["h7"] - baseline_metrics["h7"], 5)
        q_metrics["dpool_hit"] = round(q_metrics["pool_hit"] - baseline_metrics["pool_hit"], 4)

        all_config_metrics[config_name] = q_metrics
        print(f"  {config_name}: {json.dumps(q_metrics, indent=2)}")

        # Cleanup
        del X_q, gt_idx_q, sizes_q, lr_scores_q, case_ndcg_q, quota_pools
        del audio_added_sets, image_added_sets
        gc.collect()

    # ---------------------------------------------------------------
    # STEP 6: Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("R52c MULTIMODAL QUOTA SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'config':<25} {'h7':>8} {'dh7':>8} {'pool':>6} {'dpool':>7} "
          f"{'same':>7} {'diff':>7} {'E_adm':>5} {'D_adm':>5} {'E_r20':>5} {'D_r20':>5} "
          f"{'rec':>4} {'lost':>4} {'net':>4} {'J20':>5} {'t1chg':>5} "
          f"{'r_aud':>5} {'r_img':>5} {'r_ex':>4}")
    print(f"  {'-'*140}")

    print(f"  {'BASELINE (R39)':<25} {baseline_metrics['h7']:>8.5f} {'---':>8} "
          f"{baseline_metrics['pool_hit']:>6.4f} {'---':>7} "
          f"{baseline_metrics['same_h7']:>7.5f} {baseline_metrics['diff_h7']:>7.5f} "
          f"{'---':>5} {'---':>5} {'---':>5} {'---':>5} "
          f"{'---':>4} {'---':>4} {'---':>4} {'---':>5} {'---':>5} "
          f"{'---':>5} {'---':>5} {'---':>4}")

    for cname, cmetrics in all_config_metrics.items():
        print(f"  {cname:<25} {cmetrics['h7']:>8.5f} {cmetrics['dh7']:>+8.5f} "
              f"{cmetrics['pool_hit']:>6.4f} {cmetrics['dpool_hit']:>+7.4f} "
              f"{cmetrics['same_h7']:>7.5f} {cmetrics['diff_h7']:>7.5f} "
              f"{cmetrics['bucket_E_admitted']:>5} {cmetrics['bucket_D_admitted']:>5} "
              f"{cmetrics['E_recovered_top20']:>5} {cmetrics['D_recovered_top20']:>5} "
              f"{cmetrics['recovered']:>4} {cmetrics['lost']:>4} {cmetrics['net']:>4} "
              f"{cmetrics['top20_overlap']:>5.3f} {cmetrics['top1_changed']:>5} "
              f"{cmetrics.get('recovered_from_audio', 0):>5} "
              f"{cmetrics.get('recovered_from_image', 0):>5} "
              f"{cmetrics.get('recovered_from_existing', 0):>4}")

    # ---------------------------------------------------------------
    # STEP 7: Gate check
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GATE CHECK")
    print(f"{'='*70}")

    gate_pass = False
    gate_reason = ""
    best_config = ""
    recommendation = ""

    # Find best config by dh7
    best_by_dh7 = max(all_config_metrics.keys(), key=lambda k: all_config_metrics[k]["dh7"])
    best_m = all_config_metrics[best_by_dh7]

    for cname, cmetrics in all_config_metrics.items():
        dh7 = cmetrics["dh7"]
        if dh7 >= 0.010:
            gate_pass = True
            gate_reason = f"{cname}: dh7={dh7:+.5f} >= +0.010 -> candidate for full validation"
            best_config = cname
            recommendation = "proceed to full validation"
            break

    if not gate_pass:
        best_config = best_by_dh7
        best_dh7 = best_m["dh7"]
        best_e_admitted = max(m["bucket_E_admitted"] for m in all_config_metrics.values())

        if 0.005 <= best_dh7 < 0.010 and best_e_admitted > 6:
            gate_reason = (f"{best_config}: dh7={best_dh7:+.5f} in [+0.005, +0.010) "
                           f"with E_admitted={best_e_admitted} > 6 -> consider one more tuned quota")
            recommendation = "tune quota further (adjust audio/image split)"
        elif best_dh7 <= 0.005:
            same_h7_regressed = best_m["same_h7"] < baseline_metrics["same_h7"]
            gate_reason = (f"{best_config}: dh7={best_dh7:+.5f} <= +0.005"
                           + (", same_h7 regressed" if same_h7_regressed else "")
                           + " -> archive multimodal quota")
            recommendation = "archive multimodal quota; R39 remains production best"
        else:
            gate_reason = f"{best_config}: dh7={best_dh7:+.5f}, no gate condition met"
            recommendation = "archive multimodal quota; R39 remains production best"

    print(f"  Gate pass: {gate_pass}")
    print(f"  Best config: {best_config}")
    print(f"  Reason: {gate_reason}")
    print(f"  Recommendation: {recommendation}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "experiment": "R52c_multimodal_quota",
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
