#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R52: Bucket E diagnostic — find new retrieval sources for unretrievable GT tracks.

Phase 1 ONLY: source discovery, NO integration.

Two-phase execution to avoid lightgbm/torch memory conflicts:
  --phase buckets   : Reproduce R39, classify h7 cases into buckets A-E (lightgbm, no torch)
  --phase diagnostic: Test embedding sources as standalone retrievers (numpy only, no lightgbm)
  (no flag)         : Run both phases sequentially
"""
from __future__ import annotations

import argparse
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

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

BUCKET_CACHE = REPO / "cache" / "r52_bucket_labels.json"
POOL_CACHE = REPO / "cache" / "r52_pools.pkl"
OUT_PATH = REPO / "exp" / "eval" / "expR52_bucket_e_diagnostic.json"

# Embedding modalities in HF dataset
EMB_COL_MAP = {
    "audio_clap": "audio-laion_clap",
    "lyrics_qwen": "lyrics-qwen3_embedding_0.6b",
    "attributes_qwen": "attributes-qwen3_embedding_0.6b",
    "metadata_qwen": "metadata-qwen3_embedding_0.6b",
    "image_siglip": "image-siglip2",
    "cf_bpr": "cf-bpr",
}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ===================================================================
# PHASE 1: Reproduce R39, build buckets
# ===================================================================

def phase_buckets():
    """Reproduce R39 LambdaRank, classify h7 cases into buckets A-E."""
    import lightgbm as lgb

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
    from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
    from scripts.tune_postrank_v23 import tokens

    t0 = time.time()
    print(f"{ts()} R52 Phase 1: R39 Reproduction + Bucket Classification")
    print("=" * 70)

    # --- Load payload ---
    print(f"{ts()} Loading payload...")
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

    # --- Album mapping ---
    print(f"{ts()} Loading album mapping...")
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

    # --- ALS ---
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

    # --- Feature names ---
    FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
    FEAT_ALBUM = [
        "same_album_last1", "same_album_last3", "same_album_any",
        "album_history_count", "pool_same_album_count",
    ]
    FEAT_ALL = FEAT_BASE + FEAT_ALBUM

    n = len(cases)
    n_feat_base = len(FEAT_BASE)
    n_feat = len(FEAT_ALL)
    folds = grouped_session_folds(sessions, seed=0)

    print(f"{ts()} Building R39 feature matrix ({n_feat} features)...")

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
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
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

        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums if a)

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

            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # --- CV5 LambdaRank ---
    print(f"{ts()} Running CV5 LambdaRank (R39 config)...")
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
                            group=g_tr, feature_name=list(FEAT_ALL))
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

    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
    cv5 = float(np.mean(case_ndcg))
    print(f"  R39 reproduction: h7={h7_ndcg:.5f}  cv5={cv5:.5f}")

    # HARD CHECK
    expected_h7 = 0.24298
    if abs(h7_ndcg - expected_h7) > 0.001:
        print("\n  *** HARD CHECK FAILED ***")
        print(f"  Expected h7 ~{expected_h7}, got {h7_ndcg:.5f}")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={h7_ndcg:.5f} within +/-0.001 of {expected_h7}")

    expected_pool_hit = 0.6000
    if abs(pool_hit - expected_pool_hit) > 0.002:
        print("\n  *** POOL HIT CHECK FAILED ***")
        print(f"  Expected pool_hit ~{expected_pool_hit}, got {pool_hit:.4f}")
        sys.exit(1)
    print(f"  POOL HIT CHECK PASSED: {pool_hit:.4f} within +/-0.002 of {expected_pool_hit}")

    # --- Classify into buckets ---
    print(f"\n{ts()} Classifying h7 cases into buckets A-E...")
    bucket_labels = {}
    bucket_counts = Counter()

    for i in h7:
        c = cases[i]
        gt = c["gt"]
        pool = pools[i]
        ps = int(sizes[i])

        if gt in pool:
            gt_pool_idx = pool.index(gt)
            sc = lr_scores[i, :ps]
            lr_ranked = np.argsort(-sc)
            gt_lr_rank_arr = np.where(lr_ranked == gt_pool_idx)[0]
            if len(gt_lr_rank_arr) == 0:
                bucket_labels[str(i)] = {"bucket": "X", "gt": gt, "lr_rank": -1}
                bucket_counts["X"] += 1
                continue
            gt_rank = int(gt_lr_rank_arr[0]) + 1  # 1-indexed

            if gt_rank <= 20:
                bkt = "A"
            elif gt_rank <= 100:
                bkt = "B"
            else:
                bkt = "C"
            bucket_labels[str(i)] = {"bucket": bkt, "gt": gt, "lr_rank": gt_rank}
        else:
            # GT not in pool — check source union
            src_lists = src_lists_all[i]
            source_union = set()
            for sl in src_lists.values():
                source_union.update(sl)
            if gt in source_union:
                bkt = "D"
                bucket_labels[str(i)] = {"bucket": bkt, "gt": gt, "lr_rank": -1}
            else:
                bkt = "E"
                bucket_labels[str(i)] = {"bucket": bkt, "gt": gt, "lr_rank": -1}
        bucket_counts[bkt] += 1

    print(f"  Total h7 cases: {len(h7)}")
    for bk in ["A", "B", "C", "D", "E"]:
        print(f"  Bucket {bk}: {bucket_counts.get(bk, 0)}")

    # Save bucket labels
    BUCKET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(BUCKET_CACHE, "w") as f:
        json.dump({
            "h7_indices": h7,
            "bucket_labels": bucket_labels,
            "bucket_counts": dict(bucket_counts),
            "h7_ndcg": h7_ndcg,
            "pool_hit": pool_hit,
        }, f, indent=2)
    print(f"  Saved bucket labels to {BUCKET_CACHE}")

    # Save pools and R39 pool sets for diagnostic phase
    pool_sets_h7 = {str(i): set(pools[i]) for i in h7}
    r21_sets_h7 = {str(i): set(r21_source[i][:300]) for i in h7}
    r39_pools_h7 = {str(i): pools[i] for i in h7}

    with open(POOL_CACHE, "wb") as f:
        pickle.dump({
            "pool_sets_h7": pool_sets_h7,
            "r21_sets_h7": r21_sets_h7,
            "r39_pools_h7": r39_pools_h7,
            "h7_indices": h7,
            "cases_h7_played": {str(i): cases[i]["music_turns"] for i in h7},
            "cases_h7_gt": {str(i): cases[i]["gt"] for i in h7},
            "track_artist": ta,
        }, f)
    print(f"  Saved pool data to {POOL_CACHE}")
    print(f"  Phase 1 elapsed: {time.time()-t0:.1f}s")

    return bucket_labels, h7


# ===================================================================
# PHASE 2: Embedding diagnostic
# ===================================================================

def load_single_embedding(col_name: str, hf_col: str):
    """Load a single embedding modality from HF, L2-normalize, return (matrix, track_ids, tid_to_idx, n_valid)."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]

    track_ids = []
    vecs = []
    dim = None
    for item in ds:
        track_ids.append(str(item["track_id"]))
        v = item[hf_col]
        if v is not None and len(v) > 0:
            if dim is None:
                dim = len(v)
            vecs.append(v)
        else:
            vecs.append(None)

    if dim is None:
        print(f"  {col_name}: no valid embeddings found!")
        return None, track_ids, {}, 0

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
    print(f"  {col_name}: {arr.shape}, valid={n_valid}/{len(vecs)} ({n_valid/len(vecs)*100:.1f}%)")

    del ds, vecs
    gc.collect()

    return arr, track_ids, tid_to_idx, n_valid


def retrieve_recent1_nn(played, emb_matrix, tid_to_idx, topk=300):
    """Last played track's nearest neighbors."""
    if not played:
        return []
    last = played[-1]
    if last not in tid_to_idx:
        return []
    query_idx = tid_to_idx[last]
    query_emb = emb_matrix[query_idx]

    sims = emb_matrix @ query_emb
    played_idx = {tid_to_idx[t] for t in played if t in tid_to_idx}
    for pi in played_idx:
        sims[pi] = -np.inf

    top = np.argpartition(-sims, topk)[:topk]
    top = top[np.argsort(-sims[top])]
    return top.tolist()


def retrieve_recent3_max(played, emb_matrix, tid_to_idx, topk=300):
    """Max similarity across last 3 played tracks."""
    recent = played[-3:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []

    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    recent_embs = emb_matrix[recent_idx]  # (k, dim)
    sims = emb_matrix @ recent_embs.T     # (N, k)
    max_sims = sims.max(axis=1)

    for pi in played_set:
        max_sims[pi] = -np.inf

    top = np.argpartition(-max_sims, topk)[:topk]
    top = top[np.argsort(-max_sims[top])]
    return top.tolist()


def retrieve_recent3_avg(played, emb_matrix, tid_to_idx, topk=300):
    """Average embedding of last 3 played tracks."""
    recent = played[-3:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []

    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    avg_emb = emb_matrix[recent_idx].mean(axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm
    sims = emb_matrix @ avg_emb

    for pi in played_set:
        sims[pi] = -np.inf

    top = np.argpartition(-sims, topk)[:topk]
    top = top[np.argsort(-sims[top])]
    return top.tolist()


def retrieve_history_recency(played, emb_matrix, tid_to_idx, topk=300):
    """All history tracks weighted by recency: weight = 0.9^(n-i)."""
    if not played:
        return []
    n_played = len(played)
    valid_idx = []
    weights = []
    for pos, t in enumerate(played):
        if t in tid_to_idx:
            valid_idx.append(tid_to_idx[t])
            weights.append(0.9 ** (n_played - 1 - pos))
    if not valid_idx:
        return []

    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    hist_embs = emb_matrix[valid_idx]  # (k, dim)
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    weighted_emb = (hist_embs * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(weighted_emb)
    if norm > 0:
        weighted_emb = weighted_emb / norm
    sims = emb_matrix @ weighted_emb

    for pi in played_set:
        sims[pi] = -np.inf

    top = np.argpartition(-sims, topk)[:topk]
    top = top[np.argsort(-sims[top])]
    return top.tolist()


STRATEGIES = [
    ("recent1_nn", retrieve_recent1_nn),
    ("recent3_max", retrieve_recent3_max),
    ("recent3_avg", retrieve_recent3_avg),
    ("history_recency", retrieve_history_recency),
]


def compute_ndcg_at_k(gt_rank_0based: int, k: int = 20) -> float:
    """Compute nDCG@k for a single query with one relevant doc."""
    if gt_rank_0based < 0 or gt_rank_0based >= k:
        return 0.0
    return 1.0 / np.log2(gt_rank_0based + 2)


def phase_diagnostic():
    """Test each embedding source x strategy on h7 cases."""
    t0 = time.time()
    print(f"\n{ts()} R52 Phase 2: Embedding Source Diagnostic")
    print("=" * 70)

    # Load bucket labels and pool data
    with open(BUCKET_CACHE) as f:
        bucket_data = json.load(f)
    bucket_labels = bucket_data["bucket_labels"]
    h7_indices = bucket_data["h7_indices"]

    with open(POOL_CACHE, "rb") as f:
        pool_data = pickle.load(f)
    pool_sets_h7 = pool_data["pool_sets_h7"]
    r21_sets_h7 = pool_data["r21_sets_h7"]
    cases_h7_played = pool_data["cases_h7_played"]
    cases_h7_gt = pool_data["cases_h7_gt"]
    ta = pool_data["track_artist"]

    # Identify bucket E and D cases
    bucket_e_indices = [i for i in h7_indices if bucket_labels[str(i)]["bucket"] == "E"]
    bucket_d_indices = [i for i in h7_indices if bucket_labels[str(i)]["bucket"] == "D"]
    print(f"  h7 cases: {len(h7_indices)}")
    print(f"  Bucket E (unretrievable): {len(bucket_e_indices)}")
    print(f"  Bucket D (pool miss): {len(bucket_d_indices)}")

    # --- Inventory embeddings ---
    print(f"\n{ts()} STEP 2: Embedding Inventory")
    print("-" * 70)

    all_results = {}
    source_inventory = {}

    for emb_name, hf_col in EMB_COL_MAP.items():
        print(f"\n{ts()} Processing source: {emb_name}")
        print("-" * 50)

        emb_matrix, track_ids, tid_to_idx, n_valid = load_single_embedding(emb_name, hf_col)
        if emb_matrix is None:
            print("  SKIPPED: no valid embeddings")
            continue

        n_total = len(track_ids)
        source_inventory[emb_name] = {
            "dimensions": int(emb_matrix.shape[1]),
            "coverage": n_valid / n_total * 100,
            "n_valid": n_valid,
            "n_total": n_total,
            "hf_column": hf_col,
        }

        # Run each strategy
        for strat_name, strat_fn in STRATEGIES:
            key = f"{emb_name}__{strat_name}"
            print(f"  Strategy: {strat_name}...", end=" ", flush=True)
            strat_t0 = time.time()

            # Compute retrieval lists for all h7 cases
            source_lists = {}  # case_idx -> list of catalog indices
            for ci, case_i in enumerate(h7_indices):
                played = cases_h7_played[str(case_i)]
                source_lists[case_i] = strat_fn(played, emb_matrix, tid_to_idx, topk=300)

            # --- Compute metrics ---
            # standalone nDCG@20, hit@100, hit@300
            ndcg_at20_vals = []
            hit_at100 = 0
            hit_at300 = 0
            bucket_e_recovered = 0
            bucket_d_recovered = 0
            unique_vs_r39 = 0
            same_artist_top20_total = 0
            same_artist_top20_count = 0
            overlap_r21_vals = []
            overlap_r39_vals = []

            for case_i in h7_indices:
                gt = cases_h7_gt[str(case_i)]
                played = cases_h7_played[str(case_i)]
                sl = source_lists[case_i]
                sl_tids = [track_ids[j] for j in sl]
                sl_set = set(sl_tids[:300])

                # nDCG@20
                gt_rank = -1
                for r, tid in enumerate(sl_tids[:300]):
                    if tid == gt:
                        gt_rank = r
                        break
                ndcg_at20_vals.append(compute_ndcg_at_k(gt_rank, 20))

                # hit@100, hit@300
                if gt in set(sl_tids[:100]):
                    hit_at100 += 1
                if gt in sl_set:
                    hit_at300 += 1

                # same_artist_pct in top-20
                played_artists = {ta.get(t, "") for t in played} - {""}
                top20_tids = sl_tids[:20]
                if top20_tids:
                    same_count = sum(1 for t in top20_tids
                                     if ta.get(t, "") and ta.get(t, "") in played_artists)
                    same_artist_top20_total += same_count
                    same_artist_top20_count += len(top20_tids)

                # Jaccard overlap with R21 top-300
                r21_set = r21_sets_h7.get(str(case_i), set())
                if sl_set and r21_set:
                    jaccard = len(sl_set & r21_set) / len(sl_set | r21_set)
                    overlap_r21_vals.append(jaccard)

                # Jaccard overlap with R39 pool
                r39_set = pool_sets_h7.get(str(case_i), set())
                if sl_set and r39_set:
                    jaccard = len(sl_set & r39_set) / len(sl_set | r39_set)
                    overlap_r39_vals.append(jaccard)

            # Bucket E recovery
            for case_i in bucket_e_indices:
                gt = cases_h7_gt[str(case_i)]
                sl = source_lists[case_i]
                sl_tids_set = set(track_ids[j] for j in sl[:300])
                if gt in sl_tids_set:
                    bucket_e_recovered += 1
                    r39_set = pool_sets_h7.get(str(case_i), set())
                    if gt not in r39_set:
                        unique_vs_r39 += 1

            # Bucket D recovery
            unique_d_vs_r39 = 0
            for case_i in bucket_d_indices:
                gt = cases_h7_gt[str(case_i)]
                sl = source_lists[case_i]
                sl_tids_set = set(track_ids[j] for j in sl[:300])
                if gt in sl_tids_set:
                    bucket_d_recovered += 1
                    r39_set = pool_sets_h7.get(str(case_i), set())
                    if gt not in r39_set:
                        unique_d_vs_r39 += 1

            total_unique = unique_vs_r39 + unique_d_vs_r39

            same_artist_pct = (same_artist_top20_total / same_artist_top20_count * 100
                              if same_artist_top20_count > 0 else 0)
            diff_artist_pct = 100 - same_artist_pct

            result = {
                "source": emb_name,
                "strategy": strat_name,
                "standalone_h7_at20": float(np.mean(ndcg_at20_vals)),
                "standalone_h7_at100": hit_at100,
                "standalone_h7_at300": hit_at300,
                "bucket_E_recovered": bucket_e_recovered,
                "bucket_D_recovered": bucket_d_recovered,
                "unique_vs_r39_pool": total_unique,
                "unique_E_vs_r39": unique_vs_r39,
                "unique_D_vs_r39": unique_d_vs_r39,
                "same_artist_pct": round(same_artist_pct, 1),
                "diff_artist_pct": round(diff_artist_pct, 1),
                "overlap_with_r21": round(float(np.mean(overlap_r21_vals)) if overlap_r21_vals else 0, 4),
                "overlap_with_r39_pool": round(float(np.mean(overlap_r39_vals)) if overlap_r39_vals else 0, 4),
                "coverage": round(n_valid / n_total * 100, 1),
            }
            all_results[key] = result

            elapsed = time.time() - strat_t0
            print(f"{elapsed:.1f}s  E_rec={bucket_e_recovered}  D_rec={bucket_d_recovered}  "
                  f"uniq={total_unique}  h7@20={result['standalone_h7_at20']:.4f}  "
                  f"diff_art={diff_artist_pct:.0f}%")

        # Free memory before loading next source
        del emb_matrix, track_ids, tid_to_idx
        gc.collect()

    # --- Compute diff_artist_pct for recovered E cases per source*strategy ---
    # (Recompute specifically for recovered bucket E cases)
    # We already have this in the general stats. For the pass check we use the
    # general diff_artist_pct which applies to all h7 top-20, which is conservative.

    # --- Summary ---
    print(f"\n{'='*90}")
    print("R52 DIAGNOSTIC SUMMARY")
    print(f"{'='*90}")

    print(f"\n  {'source__strategy':<40} {'E_rec':>6} {'D_rec':>6} {'uniq':>6} "
          f"{'h7@20':>7} {'h7@300':>7} {'diff%':>6} {'J_r21':>6} {'J_r39':>6}")
    print(f"  {'-'*96}")

    # Sort by bucket E recovery
    sorted_keys = sorted(all_results.keys(), key=lambda k: all_results[k]["bucket_E_recovered"], reverse=True)
    for key in sorted_keys:
        r = all_results[key]
        print(f"  {key:<40} {r['bucket_E_recovered']:>6} {r['bucket_D_recovered']:>6} "
              f"{r['unique_vs_r39_pool']:>6} {r['standalone_h7_at20']:>7.4f} "
              f"{r['standalone_h7_at300']:>7} {r['diff_artist_pct']:>5.0f}% "
              f"{r['overlap_with_r21']:>6.3f} {r['overlap_with_r39_pool']:>6.3f}")

    # --- Pass check ---
    print(f"\n{'='*70}")
    print("PASS CHECK")
    print(f"{'='*70}")

    best_e_recovery = max(r["bucket_E_recovered"] for r in all_results.values()) if all_results else 0
    best_e_with_diversity = max(
        (r["bucket_E_recovered"] for r in all_results.values() if r["diff_artist_pct"] > 60),
        default=0
    )

    pass_threshold_1 = best_e_recovery >= 40
    pass_threshold_2 = best_e_with_diversity >= 25
    passed = pass_threshold_1 or pass_threshold_2

    print(f"  Best Bucket E recovery: {best_e_recovery} (threshold: >= 40) -> {'PASS' if pass_threshold_1 else 'FAIL'}")
    print(f"  Best E recovery with diff_artist > 60%: {best_e_with_diversity} (threshold: >= 25) -> {'PASS' if pass_threshold_2 else 'FAIL'}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")

    # --- Save full results ---
    full_output = {
        "experiment": "R52_bucket_e_diagnostic",
        "timestamp": datetime.now().isoformat(),
        "bucket_summary": {
            "h7_count": len(h7_indices),
            "bucket_counts": bucket_data["bucket_counts"],
            "h7_ndcg": bucket_data["h7_ndcg"],
            "pool_hit": bucket_data["pool_hit"],
        },
        "source_inventory": source_inventory,
        "diagnostic_results": all_results,
        "ranking_by_bucket_e_recovery": sorted_keys,
        "pass_check": {
            "best_e_recovery": best_e_recovery,
            "best_e_with_diversity": best_e_with_diversity,
            "threshold_1_pass": pass_threshold_1,
            "threshold_2_pass": pass_threshold_2,
            "overall_pass": passed,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n{ts()} Saved results to {OUT_PATH}")
    print(f"  Phase 2 elapsed: {time.time()-t0:.1f}s")

    return full_output


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="R52 Bucket E Diagnostic")
    parser.add_argument("--phase", choices=["buckets", "diagnostic"], default=None,
                       help="Run specific phase (default: both)")
    args = parser.parse_args()

    t0 = time.time()

    if args.phase is None or args.phase == "buckets":
        phase_buckets()
        gc.collect()

    if args.phase is None or args.phase == "diagnostic":
        if not BUCKET_CACHE.exists():
            print("ERROR: Bucket cache not found. Run --phase buckets first.")
            sys.exit(1)
        phase_diagnostic()

    print(f"\n{ts()} Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
