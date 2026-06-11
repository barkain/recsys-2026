#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 3 blind submission (EXPLORATORY — not promoted over R39).

Treats R54 as an additive 8th RRF source with r54_rank_inv + r54_presence
+ r54_cosine features. Configured at the only safe weight: R54_w=1.0_feats.

Pipeline:
  Phase train  : Train R39+R54 LambdaRank on all 8000 dev cases (37 features)
                 using cache/r54/phase3_full/oof_r54_lists.json for R54 features.
  Phase blind  : Retrieve A/B/C/D/F/ALS/R21/R54 for 80 blind queries.
                 R54 retrieval uses cached cache/r54_production/blind_r54_lists.json.
                 (Run Colab cell first to produce it.)
  Phase score  : Apply LR, top-20, reuse R39/R27b responses where top-1 matches,
                 regenerate changed top-1s with Haiku, validate, package.

Output: exp/inference/blind_a/r54_phase3_exploratory_submission.{json,zip}

Decision rationale (recorded in submission metadata):
  Dev h7 +0.0055 sub-gate (+0.010), but pool_hit +0.0248, net +48, same/diff
  both improve. Structural retriever change, not noisy metadata. Used as one
  information-gathering blind probe.
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
R54_OOF = REPO / "cache" / "r54" / "phase3_full" / "oof_r54_lists.json"
R54_BLIND_LISTS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
BLIND_OUT = REPO / "exp" / "inference" / "blind_a"
LOCAL_TRACK_METADATA = REPO / "cache" / "metadata" / "track_metadata_all_tracks.json"

RRF_K = 20
POOL_K = 300
# R54 weight: ONLY w=1.0 (w=1.5 and w=2.0 regress h7 in integration).
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

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
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
FEAT_R39_ALL = FEAT_BASE + FEAT_ALBUM
FEAT_ALL = FEAT_R39_ALL + FEAT_R54


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    def first_value(value):
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value) if value else ""

    if LOCAL_TRACK_METADATA.exists():
        raw = json.load(open(LOCAL_TRACK_METADATA))
        return {
            str(tid): first_value(meta.get("album_id")) or first_value(meta.get("album_name"))
            for tid, meta in raw.items()
        }

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


def _featurize_row(pool, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
                    case_query, history, played, played_set, ta, tt, ttl, tat, tmt,
                    als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
                    pool_K=POOL_K, n_feat=len(FEAT_ALL)):
    n_base = len(FEAT_BASE)
    n_r39 = len(FEAT_R39_ALL)
    feats = np.zeros((len(pool), n_feat), dtype=np.float64)

    user_msgs = [str(r["content"]) for r in history if r["role"] == "user"] + [case_query]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    l_artist = ta.get(played[-1], "") if played else ""
    l_tags = tt.get(played[-1], set()) if played else set()
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]
    pool_artists = [ta.get(tid, "") for tid in pool[:pool_K]]
    artist_counts = Counter(a for a in pool_artists if a)
    src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                for sn, sl in src_lists.items()}

    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
    all_albums = [track_album.get(t, "") for t in played]
    album_hist_counts = Counter(a for a in all_albums if a)

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
        row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sn, {}))
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

        # R54 features
        row[n_r39 + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
        row[n_r39 + 1] = 1.0 if tid in r54_rank_map else 0.0
        row[n_r39 + 2] = r54_score_map.get(tid, 0.0)

    return feats


def phase_train():
    """Train R39+R54 LambdaRank on all 8000 dev cases."""
    import lightgbm as lgb
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector
    from scripts.expS2_lr_v2 import build_popularity_stats

    print(f"\n{ts()} Phase TRAIN: Production LR on all dev (37 features)")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"  Loading R54 OOF from {R54_OOF}")
    with open(R54_OOF) as f:
        r54_data = json.load(f)
    r54_raw = r54_data["lists"]
    assert len(r54_raw) == n, f"R54 OOF length {len(r54_raw)} != {n}"
    r54_source = []
    r54_scores = []
    for case_lists in r54_raw:
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        r54_source.append(tids)
        r54_scores.append(score_map)
    print(f"  Validated R54 OOF for {n} cases")

    track_pop = build_popularity_stats()
    track_album = load_track_albums()
    max_pop = max(track_pop.values()) if track_pop else 1

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

    print(f"  Building features for {n} cases (this is the slow step)...")
    t_feat = datetime.now()
    X_flat, y, groups = [], [], []
    gt_in_pool_count = 0
    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "R54": r54_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        gt = c["gt"]
        gi = pool.index(gt) if gt in pool else -1
        if gi >= 0:
            gt_in_pool_count += 1

        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_source[i][:300])}
        feats = _featurize_row(pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
                                c["user_query"], c["history"], c["music_turns"],
                                set(c["music_turns"]), ta, tt, ttl, tat, tmt,
                                als_factors, als_track_to_idx, als_vecs[i],
                                track_pop, max_pop, track_album)
        s = len(pool)
        for k in range(s):
            X_flat.append(feats[k])
            y.append(1.0 if k == gi else 0.0)
        groups.append(s)
        if (i + 1) % 1000 == 0:
            elapsed = (datetime.now() - t_feat).total_seconds()
            print(f"  features {i + 1}/{n} ({elapsed:.0f}s)", flush=True)

    print(f"  pool_hit@{POOL_K}: {gt_in_pool_count}/{n} ({gt_in_pool_count/n:.4f})")
    print(f"  Training LR on {len(groups)} groups, {len(y)} candidates, {len(FEAT_ALL)} features")
    ds = lgb.Dataset(np.array(X_flat), label=np.array(y),
                     group=groups, feature_name=list(FEAT_ALL))
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
              "verbose": -1, "seed": 0}
    lr_model = lgb.train(params, ds, num_boost_round=300)
    model_path = REPO / "cache" / "r54_phase3_lr_model.txt"
    lr_model.save_model(str(model_path))
    print(f"  Saved LR model -> {model_path}")

    # Save shared artifacts for blind phase (avoid lightgbm imports there)
    als_cache = REPO / "cache" / "r54_phase3_als.npz"
    np.savez(als_cache, factors=als_factors, track_ids=np.array(als_track_ids))
    pop_cache = REPO / "cache" / "r54_phase3_track_pop.json"
    with open(pop_cache, "w") as f:
        json.dump(track_pop, f)
    maps_cache = REPO / "cache" / "r54_phase3_payload_maps.pkl"
    with open(maps_cache, "wb") as f:
        pickle.dump({
            "track_artist": ta, "track_tags": tt,
            "track_title_toks": ttl, "track_artist_toks": tat,
            "track_meta_toks": tmt,
        }, f)
    print(f"  Saved ALS / pop / maps caches.")
    print(f"  Phase train done. Next: --phase blind")


def als_retrieve_simple(played, als_to_idx, als_factors, als_ids, topk=200):
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
    """Retrieve A/B/C/D/F/ALS/R21/R54 for 80 blind queries, build features."""
    if not R54_BLIND_LISTS.exists():
        print(f"ERROR: {R54_BLIND_LISTS} not found.")
        print(f"  Run the Colab cell to produce R54 production blind retrieval first.")
        sys.exit(1)

    print(f"\n{ts()} Phase BLIND: feature extraction for 80 queries")
    track_album = load_track_albums()
    with open(REPO / "cache" / "r54_phase3_track_pop.json") as f:
        track_pop = json.load(f)
    als_data = np.load(REPO / "cache" / "r54_phase3_als.npz", allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}
    with open(REPO / "cache" / "r54_phase3_payload_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    ta, tt, ttl, tat, tmt = (maps[k] for k in
                              ["track_artist", "track_tags", "track_title_toks",
                               "track_artist_toks", "track_meta_toks"])
    max_pop = max(track_pop.values()) if track_pop else 1
    n_feat = len(FEAT_ALL)

    print(f"  Loading R54 blind lists...")
    with open(R54_BLIND_LISTS) as f:
        r54_blind_data = json.load(f)
    # Format: { session_id: [(tid, score), ...] }
    r54_blind_by_sid = {sid: case for sid, case in r54_blind_data["lists"].items()}
    print(f"  R54 blind: {len(r54_blind_by_sid)} sessions")

    from datasets import DownloadConfig, load_dataset
    from run_inference_blind_r3_det import APrimeMaxRecent
    from run_inference_blind_f1 import CFBPRMaxRecent, load_cfbpr_index
    from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
    from sentence_transformers import SentenceTransformer

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

    blind_rows: list[dict] = []
    for idx, item in enumerate(db):
        sid = str(item["session_id"])
        turn_num, user_query, history, music_turns = parse_last_turn_local(item)
        played_set = set(music_turns)

        user_parts = [str(h["content"]) for h in history if h["role"] == "user"]
        user_parts.append(user_query)
        r21_query = " ".join(user_parts[-3:])

        q_emb = r21_model.encode([r21_query], normalize_embeddings=True).astype(np.float32)[0]
        r21_scores = r21_track_embs_arr @ q_emb
        for ti, tid in enumerate(r21_track_ids_list):
            if tid in played_set:
                r21_scores[ti] = -np.inf
        r21_top = np.argpartition(-r21_scores, 300)[:300]
        r21_top = r21_top[np.argsort(-r21_scores[r21_top])]
        r21_list = [r21_track_ids_list[j] for j in r21_top]

        # R54 from cached Colab retrieval
        if sid not in r54_blind_by_sid:
            print(f"  WARNING: sid={sid} missing from R54 blind cache, falling back to empty R54")
            r54_pairs = []
        else:
            r54_pairs = r54_blind_by_sid[sid]
        r54_list = [t for t, _ in r54_pairs[:300]]
        r54_score_map = {t: float(s) for t, s in r54_pairs}

        src_a = a_prime.topn(music_turns, topn=100) if music_turns else []
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
                     "F": src_f, "ALS": als_tracks, "R21": r21_list, "R54": r54_list}
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)

        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_list[:300])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_list[:300])}
        feats = _featurize_row(pool, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
                                user_query, history, music_turns, played_set,
                                ta, tt, ttl, tat, tmt,
                                als_factors, als_to_idx, als_vec, track_pop, max_pop,
                                track_album)

        blind_rows.append({
            "session_id": sid, "turn_number": turn_num,
            "pool": pool, "feats": feats,
        })
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/80 blind rows retrieved", flush=True)

    blind_cache = REPO / "cache" / "r54_phase3_blind_features.pkl"
    with open(blind_cache, "wb") as f:
        pickle.dump(blind_rows, f)
    print(f"  Saved {len(blind_rows)} rows to {blind_cache}")
    print(f"  Phase blind done. Next: --phase score")


def phase_score():
    """Apply LR, top-20, reuse responses where top-1 matches, regenerate changed, validate, package."""
    import lightgbm as lgb

    print(f"\n{ts()} Phase SCORE: rank + responses + package")
    lr_model = lgb.Booster(model_file=str(REPO / "cache" / "r54_phase3_lr_model.txt"))
    track_album = load_track_albums()
    valid_catalog = set(track_album.keys())

    with open(REPO / "cache" / "r54_phase3_blind_features.pkl", "rb") as f:
        blind_rows = pickle.load(f)

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

    # Diff vs R39 BEFORE response assembly
    r39_path = BLIND_OUT / "r39_album_submission.json"
    r27b_path = BLIND_OUT / "r27b_agent_audit_submission.json"
    with open(r39_path) as f:
        r39_results = {r["session_id"]: r for r in json.load(f)}
    with open(r27b_path) as f:
        r27b_results = {r["session_id"]: r for r in json.load(f)}

    top1_changed = 0
    overlaps = []
    for r in results:
        sid = r["session_id"]
        r39 = r39_results.get(sid)
        if r39:
            if r["predicted_track_ids"][0] != r39["predicted_track_ids"][0]:
                top1_changed += 1
            overlaps.append(len(set(r["predicted_track_ids"]) &
                                set(r39["predicted_track_ids"])))
    print(f"  vs R39: top-1 changed={top1_changed}/80  "
          f"avg top-20 overlap={np.mean(overlaps):.1f}/20")

    # Response assembly: reuse from R39, R27b, R25, R21 by top-1 compatibility
    print(f"\n{ts()} Response assembly (reuse where top-1 in candidate list)")
    prev_by_sid: dict[str, dict] = {}
    for path, name in [(r39_path, "R39"), (r27b_path, "R27b"),
                       (BLIND_OUT / "r25_lexdiv_v2.json", "R25"),
                       (BLIND_OUT / "lr_r21_v1_hybrid.json", "R21")]:
        if path.exists():
            with open(path) as f:
                for r in json.load(f):
                    sid = r["session_id"]
                    if sid not in prev_by_sid:
                        prev_by_sid[sid] = {"response": r["predicted_response"],
                                           "tracks": r["predicted_track_ids"],
                                           "source": name}

    reused = 0
    reuse_sources = Counter()
    need_gen = []
    for r in results:
        sid = r["session_id"]
        prev = prev_by_sid.get(sid)
        # Reuse if the previous top-1 is still in our top-20 → response stays compatible
        if prev and prev["tracks"][0] in set(r["predicted_track_ids"]):
            r["predicted_response"] = prev["response"]
            reused += 1
            reuse_sources[prev["source"]] += 1
        else:
            need_gen.append(r)
    print(f"  reused: {reused}  need generation: {len(need_gen)}  sources: {dict(reuse_sources)}")

    if need_gen:
        print(f"\n{ts()} Generating {len(need_gen)} new responses with Haiku")
        from datasets import DownloadConfig, load_dataset
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule
        from run_inference_blind_r3_det import build_session_memory_for_response

        item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                                  split_types=["all_tracks"])
        prompts_dir = REPO / "mcrs" / "system_prompts"
        sys_prompt = (prompts_dir / "roleplay.txt").read_text() + "\n" + (prompts_dir / "response_generation.txt").read_text()
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")

        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                          download_config=DownloadConfig(local_files_only=True))
        blind_by_sid = {str(item["session_id"]): item for item in db}

        for r in need_gen:
            item = blind_by_sid[r["session_id"]]
            _, user_query_r, history_r, _ = parse_last_turn_local(item)
            top_id = r["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(history_r, user_query_r, item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            r["predicted_response"] = (response or "").lstrip(",").lstrip()

    # Validation
    print(f"\n{ts()} Validation")
    sids = set()
    for r in results:
        sid = r["session_id"]
        if len(r["predicted_track_ids"]) != 20:
            raise ValueError(f"Row {sid}: {len(r['predicted_track_ids'])} tracks (expected 20)")
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
        raise ValueError(f"Expected 80 unique rows, got {len(results)}/{len(sids)}")

    # LexDiv estimate (Distinct-2)
    from itertools import chain
    all_bigrams = []
    for r in results:
        toks = r["predicted_response"].lower().split()
        all_bigrams.extend(zip(toks, toks[1:]))
    lexdiv = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    print(f"  LexDiv estimate (Distinct-2 across 80 responses): {lexdiv:.4f}")

    # Diff report
    report = {
        "exploratory_label": "R54_phase3_w1.0_feats",
        "vs_r39": {
            "top1_changed": int(top1_changed),
            "avg_top20_overlap": float(np.mean(overlaps)),
            "min_top20_overlap": int(min(overlaps)),
            "max_top20_overlap": int(max(overlaps)),
        },
        "response_strategy": {
            "reused": reused, "generated": len(need_gen),
            "reuse_sources": dict(reuse_sources),
        },
        "lexdiv_distinct2_estimate": float(lexdiv),
        "validation": "PASS",
        "dev_metrics": {
            "h7_baseline": 0.24298, "h7_r54": 0.24853, "delta_h7": 0.00554,
            "pool_hit_baseline": 0.6000, "pool_hit_r54": 0.6248, "delta_pool": 0.0248,
            "net_top20": 48, "gate_status": "FAIL (h7 -0.0045 short)",
            "fold_h7_deltas": [0.0295, 0.0105, 0.0045, -0.0169, 0.0001],
        },
        "rationale": "Exploratory probe. Dev h7 below gate but structural retriever change with positive pool/net. Not promoted over R39.",
        "created_at": datetime.now().isoformat(),
    }

    out_json = BLIND_OUT / "r54_phase3_exploratory_submission.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_meta = BLIND_OUT / "r54_phase3_exploratory_metadata.json"
    with open(out_meta, "w") as f:
        json.dump(report, f, indent=2)
    out_zip = BLIND_OUT / "r54_phase3_exploratory_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")

    print(f"\n  SUBMISSION: {out_zip}")
    print(f"  METADATA:   {out_meta}")
    print(f"  80 rows. reused {reused}, generated {len(need_gen)}. top-1 changed {top1_changed}/80")
    print(f"  LexDiv estimate: {lexdiv:.4f}")
    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="train", choices=["train", "blind", "score"])
    args = parser.parse_args()
    if args.phase == "train":
        phase_train()
    elif args.phase == "blind":
        phase_blind()
    elif args.phase == "score":
        phase_score()
