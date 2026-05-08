#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R39b: Album-aware LambdaRank blind submission.

Same retrieval pool as R21 production.
Same response strategy as R27b.
Only change: add 5 album features to LambdaRank.

Steps:
1. Train production LambdaRank with album features on full dev
2. Blind inference (track-only)
3. Hybrid responses from R27b
4. Validate and package
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.tune_postrank_v23 import tokens


def parse_last_turn_local(item):
    """Parse last user turn from blind session (no external imports)."""
    import pandas as pd
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    last_user = user_rows.iloc[-1]
    turn_num = int(last_user["turn_number"])
    user_query = str(last_user["content"])
    prior = df[df["turn_number"] < turn_num]
    history = [{"role": str(r["role"]), "content": r["content"],
                "turn_number": int(r["turn_number"])} for _, r in prior.iterrows()]
    music_turns = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
    return turn_num, user_query, history, music_turns

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_MODEL = REPO / "cache" / "r21_production" / "model"
R21_TRACK_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
BLIND_OUT = REPO / "exp" / "inference" / "blind_a"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE_NAMES = [
    "rrf_rank_inv", "last_artist_match", "last_tag_jaccard",
    "query_artist_tok_overlap", "query_title_tok_overlap", "query_meta_tok_overlap",
    "is_played", "recency_score",
    "src_a_rank_inv", "src_b_rank_inv", "src_c_rank_inv", "src_d_rank_inv",
    "src_f_rank_inv", "src_als_rank_inv",
    "src_a_pres", "src_b_pres", "src_c_pres", "src_d_pres",
    "src_f_pres", "src_als_pres",
    "n_sources", "als_dot", "n_history",
    "popularity", "pool_artist_frac", "pool_artist_count", "source_count_v2",
    "r21_rank_inv", "r21_presence",
]
FEAT_BASE = FEAT_BASE_NAMES
FEAT_ALBUM = [
    "same_album_last1",
    "same_album_last3",
    "same_album_any",
    "album_history_count",
    "pool_same_album_count",
]
FEAT_ALL = FEAT_BASE + FEAT_ALBUM


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds:
        tid = str(item["track_id"])
        # Prefer album_id over album_name to avoid name collisions
        alb_id = item.get("album_id", [])
        if isinstance(alb_id, list) and alb_id:
            track_album[tid] = str(alb_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                track_album[tid] = str(alb_name[0])
            else:
                track_album[tid] = ""
    n_with = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {n_with} with album_id")
    return track_album


def build_features_with_album(cases, payload, als_source, als_vecs, als_factors,
                                als_track_to_idx, track_pop, r21_source, track_album):
    """Build feature matrix with album features for all dev cases."""
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1
    n_feat = len(FEAT_ALL)
    n_base = len(FEAT_BASE)

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
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
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

            # Base 29 features (identical to production)
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

            # Album features (R39)
            c_album = track_album.get(tid, "")
            row[n_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_alb_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_base + 4] = pool_alb_count / max(len(pool), 1)

    return X, gt_idx, sizes


def main():
    print(f"{ts()} R39b: Album-Aware Blind Submission")
    print("=" * 70)

    # Step 1: Train production LambdaRank with album features
    import lightgbm as lgb
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector
    from scripts.expS2_lr_v2 import build_popularity_stats

    print(f"\n{ts()} Step 1: Training production LambdaRank...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    track_album = load_track_albums()

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

    X, gt_idx, sizes = build_features_with_album(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, track_album)

    n = len(cases)
    X_flat, y, groups = [], [], []
    for i in range(n):
        s = int(sizes[i])
        for k in range(s):
            X_flat.append(X[i, k])
            y.append(1.0 if k == gt_idx[i] else 0.0)
        groups.append(s)

    ds = lgb.Dataset(np.array(X_flat), label=np.array(y),
                     group=groups, feature_name=list(FEAT_ALL))
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
              "verbose": -1, "seed": 0}
    lr_model = lgb.train(params, ds, num_boost_round=300)
    model_path = REPO / "cache" / "r39_lr_model.txt"
    lr_model.save_model(str(model_path))
    print(f"  Trained on {n} cases, {len(FEAT_ALL)} features")
    print(f"  Model saved to {model_path}")

    # Save ALS artifacts for Phase 2 (avoids lightgbm import in blind process)
    als_cache = REPO / "cache" / "r39_als.npz"
    np.savez(als_cache, factors=als_factors, track_ids=np.array(als_track_ids))
    print(f"  ALS saved to {als_cache}")

    # Save track_pop
    pop_cache = REPO / "cache" / "r39_track_pop.json"
    with open(pop_cache, "w") as f:
        json.dump(track_pop, f)

    # Save payload maps needed for blind
    maps_cache = REPO / "cache" / "r39_payload_maps.pkl"
    with open(maps_cache, "wb") as f:
        pickle.dump({
            "track_artist": payload["track_artist"],
            "track_tags": payload["track_tags"],
            "track_title_toks": payload["track_title_toks"],
            "track_artist_toks": payload["track_artist_toks"],
            "track_meta_toks": payload["track_meta_toks"],
        }, f)

    print("  Phase 1 done. Run with --phase blind for inference.")
    return


def als_retrieve_simple(played, als_to_idx, als_factors, als_ids, topk=200):
    """ALS retrieval without importing expS2_lambdarank."""
    played_idx = [als_to_idx[t] for t in played if t in als_to_idx]
    if not played_idx:
        return [], None
    weights = np.array([0.8 ** i for i in range(len(played_idx))], dtype=np.float32)
    weights = weights / weights.sum()
    sv = np.zeros(als_factors.shape[1], dtype=np.float32)
    for w, idx in zip(weights, played_idx):
        sv += w * als_factors[idx]
    norm = np.linalg.norm(sv)
    if norm < 1e-8:
        return [], None
    sv = sv / norm
    scores = als_factors @ sv
    for t in played:
        if t in als_to_idx:
            scores[als_to_idx[t]] = -np.inf
    top = np.argpartition(-scores, topk)[:topk]
    top = top[np.argsort(-scores[top])]
    return [als_ids[j] for j in top], sv


def phase_blind():
    """Phase 2: blind retrieve + features (NO lightgbm). Saves features for scoring."""

    print(f"\n{ts()} Loading saved artifacts (no lightgbm)...")

    track_album = load_track_albums()

    with open(REPO / "cache" / "r39_track_pop.json") as f:
        track_pop = json.load(f)

    als_data = np.load(REPO / "cache" / "r39_als.npz", allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}
    print(f"  ALS: {als_factors.shape}")

    with open(REPO / "cache" / "r39_payload_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1
    n_base = len(FEAT_BASE)
    n_feat = len(FEAT_ALL)

    # Step 2: Blind inference
    print(f"\n{ts()} Step 2: Blind inference...")
    from datasets import DownloadConfig, load_dataset
    from run_inference_blind_r3_det import APrimeMaxRecent
    from run_inference_blind_f1 import CFBPRMaxRecent, load_cfbpr_index
    from offline_retrieval_sweep import CachedBM25
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
    from sentence_transformers import SentenceTransformer

    from offline_retrieval_sweep import load_track_metadata, query_parts
    metadata = load_track_metadata()

    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=3)
    cf_ids, cf_vecs, cf_idx = load_cfbpr_index()
    cfbpr = CFBPRMaxRecent(cf_ids, cf_vecs, cf_idx, recent_k=3)

    r21_model = SentenceTransformer(str(R21_MODEL), device="cpu")
    r21_track_ids_list = json.loads(Path(R21_TRACK_IDS).read_text())
    r21_track_embs_arr = np.load(R21_TRACK_EMBS)

    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))

    n_base = len(FEAT_BASE)
    n_feat = len(FEAT_ALL)

    blind_rows: list[dict] = []
    for idx, item in enumerate(db):
        sid = str(item["session_id"])
        turn_num, user_query, history, music_turns = parse_last_turn_local(item)

        user_parts = [str(h["content"]) for h in history if h["role"] == "user"]
        user_parts.append(user_query)
        r21_query = " ".join(user_parts[-3:])

        q_emb = r21_model.encode([r21_query], normalize_embeddings=True).astype(np.float32)[0]
        r21_scores = r21_track_embs_arr @ q_emb
        played_set = set(music_turns)
        for ti, tid in enumerate(r21_track_ids_list):
            if tid in played_set:
                r21_scores[ti] = -np.inf
        r21_top = np.argpartition(-r21_scores, 300)[:300]
        r21_top = r21_top[np.argsort(-r21_scores[r21_top])]
        r21_list = [r21_track_ids_list[j] for j in r21_top]

        src_a = a_prime.topn(music_turns, topn=100) if music_turns else []

        # B/C use different BM25 query modes (production parity)
        history_dicts = [{"role": h["role"], "content": h["content"]} for h in history]
        q_b = " ".join(query_parts(history_dicts, user_query, metadata, "last_music_meta"))
        q_c = " ".join(query_parts(history_dicts, user_query, metadata, "full"))
        src_b = bm25.retrieve(q_b or user_query, topk=100)
        src_c = bm25.retrieve(q_c or user_query, topk=100)

        anchor = music_turns[-1] if music_turns else None
        src_d = track_sim.track_id_to_neighbors(anchor, topk=100) if anchor else []
        src_f = cfbpr.topn(music_turns, topn=100) if music_turns else []
        als_tracks, als_vec = als_retrieve_simple(music_turns, als_to_idx, als_factors, als_ids)

        src_lists = {"A": src_a, "B": src_b, "C": src_c, "D": src_d,
                     "F": src_f, "ALS": als_tracks, "R21": r21_list}
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = user_parts
        played = music_turns
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_list[:300])}

        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums_hist = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums_hist if a)

        feats = np.zeros((len(pool), n_feat), dtype=np.float64)
        for rank, tid in enumerate(pool, start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = feats[rank - 1]
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
            if als_vec is not None:
                aidx = als_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(als_vec, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

            c_album = track_album.get(tid, "")
            row[n_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_alb = sum(1 for t2 in pool if track_album.get(t2, "") == c_album) if c_album else 0
            row[n_base + 4] = pool_alb / max(len(pool), 1)

        blind_rows.append({
            "session_id": sid,
            "turn_number": turn_num,
            "pool": pool,
            "feats": feats,
        })
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/80 blind rows retrieved", flush=True)

    # Save for scoring phase
    blind_cache = REPO / "cache" / "r39_blind_features.pkl"
    with open(blind_cache, "wb") as f:
        pickle.dump(blind_rows, f)
    print(f"  Saved {len(blind_rows)} rows to {blind_cache}")
    print("  Phase blind done. Run with --phase score")

def phase_score():
    """Phase 3: score with LR model + responses + validation (lightgbm, no SentenceTransformer)."""
    import lightgbm as lgb

    print(f"\n{ts()} Phase score: Loading artifacts...")
    lr_model = lgb.Booster(model_file=str(REPO / "cache" / "r39_lr_model.txt"))
    track_album = load_track_albums()

    with open(REPO / "cache" / "r39_blind_features.pkl", "rb") as f:
        blind_rows = pickle.load(f)
    from datasets import DownloadConfig, load_dataset
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))
    print(f"  {len(blind_rows)} blind rows loaded")

    results = []
    for row in blind_rows:
        scores = lr_model.predict(row["feats"])
        ranked_idx = np.argsort(-scores)
        top20 = [row["pool"][j] for j in ranked_idx[:20]]
        results.append({
            "session_id": row["session_id"],
            "turn_number": row["turn_number"],
            "predicted_track_ids": top20,
            "predicted_response": "",
        })

    # Step 3: Hybrid responses
    print(f"\n{ts()} Step 3: Response assembly...")
    r27b_path = BLIND_OUT / "r27b_agent_audit_submission.json"
    r25_path = BLIND_OUT / "r25_lexdiv_v2.json"
    r21_path = BLIND_OUT / "lr_r21_v1_hybrid.json"

    prev_by_sid: dict[str, dict] = {}
    for path, name in [(r27b_path, "R27b"), (r25_path, "R25"), (r21_path, "R21")]:
        if path.exists():
            with open(path) as f:
                for r in json.load(f):
                    sid = r["session_id"]
                    if sid not in prev_by_sid:
                        prev_by_sid[sid] = {"response": r["predicted_response"],
                                           "tracks": r["predicted_track_ids"],
                                           "source": name}

    reused = 0
    need_gen = []
    for r in results:
        sid = r["session_id"]
        prev = prev_by_sid.get(sid)
        if prev and prev["tracks"][0] in set(r["predicted_track_ids"]):
            r["predicted_response"] = prev["response"]
            reused += 1
        else:
            need_gen.append(r)

    print(f"  Reused: {reused}, need generation: {len(need_gen)}")

    if need_gen:
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule
        from run_inference_blind_r3_det import build_session_memory_for_response

        item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                                  split_types=["all_tracks"])
        prompts_dir = REPO / "mcrs" / "system_prompts"
        sys_prompt = (prompts_dir / "roleplay.txt").read_text() + "\n" + (prompts_dir / "response_generation.txt").read_text()
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")
        blind_by_sid = {str(item["session_id"]): item for item in db}

        for r in need_gen:
            item = blind_by_sid[r["session_id"]]
            turn_num_r, user_query_r, history_r, music_turns_r = parse_last_turn_local(item)
            top_id = r["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(history_r, user_query_r, item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            r["predicted_response"] = (response or "").lstrip(",").lstrip()

    # Step 4: Validate
    print(f"\n{ts()} Step 4: Validation...")
    valid_catalog = set(track_album.keys())
    sids = set()
    for r in results:
        sid = r["session_id"]
        if len(r["predicted_track_ids"]) != 20:
            raise ValueError(f"Row {sid}: {len(r['predicted_track_ids'])} tracks")
        if len(set(r["predicted_track_ids"])) != 20:
            raise ValueError(f"Row {sid}: duplicate tracks")
        if not r["predicted_response"].strip():
            raise ValueError(f"Row {sid}: empty response")
        if r["predicted_response"].startswith(","):
            raise ValueError(f"Row {sid}: leading comma")
        invalid = [t for t in r["predicted_track_ids"] if t not in valid_catalog]
        if invalid:
            raise ValueError(f"Row {sid}: invalid track IDs: {invalid[:3]}")
        sids.add(sid)
    if len(results) != 80 or len(sids) != 80:
        raise ValueError(f"Expected 80 rows, got {len(results)}")

    # Compare vs R27b
    if r27b_path.exists():
        with open(r27b_path) as f:
            r27b = {r["session_id"]: r for r in json.load(f)}
        top1_changed = sum(1 for r in results
                          if r["predicted_track_ids"][0] != r27b.get(r["session_id"], {}).get("predicted_track_ids", [""])[0])
        overlaps = [len(set(r["predicted_track_ids"]) &
                       set(r27b.get(r["session_id"], {}).get("predicted_track_ids", [])))
                   for r in results]
        print(f"  vs R27b: top-1 changed={top1_changed}/80, avg overlap={np.mean(overlaps):.1f}/20")

    out_json = BLIND_OUT / "r39_album_submission.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_zip = BLIND_OUT / "r39_album_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")

    print(f"\n  SUBMISSION: {out_zip}")
    print(f"  80 rows, {reused} reused, {len(need_gen)} generated")
    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="train", choices=["train", "blind", "score"])
    args = parser.parse_args()

    if args.phase == "train":
        main()
    elif args.phase == "blind":
        phase_blind()
    elif args.phase == "score":
        phase_score()
