#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S101
"""R50: View-aware ranking features for LambdaRank.

Phase 1 (--phase encode): Encode per-view query embeddings for 5 folds.
Phase 2 (--phase evaluate): Build per-view cosine/rank features, run 5 LambdaRank configs.

CRITICAL: encode imports sentence_transformers (torch). evaluate imports lightgbm (no torch).
Run as separate processes to avoid macOS segfault.

  .venv/bin/python scripts/expR50_view_aware_ranking.py --phase encode
  .venv/bin/python scripts/expR50_view_aware_ranking.py --phase evaluate
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
TRACK_IDS_PATH = REPO / "cache" / "r21_production" / "track_ids.json"
FOLD_INDICES_PATH = REPO / "cache" / "r21_production" / "fold_indices.json"
MANIFEST_PATH = REPO / "cache" / "r50_encode_manifest.json"

MODEL_DIRS = {
    0: REPO / "cache" / "r21_supervised" / "fold_0",
    1: REPO / "cache" / "r21_supervised" / "fold_1",
    2: REPO / "cache" / "r21_production" / "oof" / "model_fold_2",
    3: REPO / "cache" / "r21_production" / "oof" / "model_fold_3",
    4: REPO / "cache" / "r21_production" / "oof" / "model_fold_4",
}

VIEW_NAMES = ["current", "lastquery", "history"]

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# ── Track text encoding (R21 format with album+tags) ──────────────


def _first_or_str(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value else ""


def _artists_to_str(value):
    if isinstance(value, list):
        return ", ".join(str(a) for a in value)
    return str(value) if value else ""


def load_catalog_metadata():
    """Load track metadata from HF dataset (cached locally)."""
    from datasets import Dataset
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found in HF cache")
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        meta[tid] = {
            "track_name": cols.get("track_name", [""])[i],
            "artist_name": cols.get("artist_name", [""])[i],
            "album_name": cols.get("album_name", [""])[i],
            "tag_list": cols.get("tag_list", [[]])[i],
        }
    return meta


def build_track_text_r21(tid, meta):
    """Exact R21 catalog text format. Required for OOF parity."""
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])

    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)

    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


# ── Query view builders ───────────────────────────────────────────


def build_view_current(case):
    """View 1: Last 3 user messages (R21 default query)."""
    user_msgs = [str(h["content"]) for h in case["history"] if h["role"] == "user"]
    user_msgs.append(case["user_query"])
    return " ".join(user_msgs[-3:])


def build_view_lastquery(case):
    """View 2: Only the current user query."""
    return case["user_query"]


def build_view_history(case, track_meta):
    """View 3: Structured summary of played tracks (last 10)."""
    played = case["music_turns"]
    items = []
    for tid in played[-10:]:
        m = track_meta.get(tid, {})
        name = _first_or_str(m.get("track_name", "")) or "unknown"
        artist = _artists_to_str(m.get("artist_name", "")) or "unknown"
        items.append(f"{name} by {artist}")
    if items:
        return "User listened to: " + "; ".join(items)
    return "No prior plays"


# ── Phase 1: Encode ───────────────────────────────────────────────


def run_encode():
    """Encode per-view query embeddings for all 5 folds."""
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    print(f"{ts()} R50: Encode per-view query embeddings")
    print("=" * 70)

    # Load payload
    print(f"{ts()} Loading dev payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]

    # Load fold indices
    with open(FOLD_INDICES_PATH) as f:
        fold_indices_raw = json.load(f)
    fold_indices = {int(k): v for k, v in fold_indices_raw.items()}

    # Load track IDs
    with open(TRACK_IDS_PATH) as f:
        all_track_ids = json.load(f)
    n_tracks = len(all_track_ids)

    # Load track metadata
    print(f"{ts()} Loading track metadata...")
    track_meta = load_catalog_metadata()

    # Build track texts
    print(f"{ts()} Building track texts...")
    track_texts = [build_track_text_r21(tid, track_meta) for tid in all_track_ids]

    manifest_folds = []

    for fold_i in range(5):
        fold_t0 = time.time()
        held = fold_indices[fold_i]
        model_dir = MODEL_DIRS[fold_i]
        print(f"\n{ts()} === Fold {fold_i} ({len(held)} val cases) ===")
        print(f"  Model: {model_dir}")

        # Load model
        model = SentenceTransformer(str(model_dir), device="cpu")
        set_eval = getattr(model, "eval")
        set_eval()

        # Check for cached track embeddings
        track_emb_cache = REPO / "cache" / f"r49a_fold_{fold_i}_track_embs.npy"
        if track_emb_cache.exists():
            print(f"  {ts()} Reusing cached track embeddings: {track_emb_cache}")
            # Don't need to load track embeddings here — we only need query embeddings
            # Track embeddings will be loaded in evaluate phase
        else:
            print(f"  {ts()} Encoding {n_tracks} tracks (cache missing)...")
            track_embs = model.encode(track_texts, batch_size=128,
                                      show_progress_bar=True,
                                      normalize_embeddings=True).astype(np.float32)
            np.save(track_emb_cache, track_embs)
            print(f"  Cached: {track_emb_cache}")
            del track_embs

        # Build queries for each view
        val_cases = [cases[j] for j in held]
        view_queries = {
            "current": [build_view_current(c) for c in val_cases],
            "lastquery": [build_view_lastquery(c) for c in val_cases],
            "history": [build_view_history(c, track_meta) for c in val_cases],
        }

        # Encode query embeddings for each view
        for vn in VIEW_NAMES:
            out_path = REPO / "cache" / f"r50_fold_{fold_i}_query_embs_{vn}.npy"
            print(f"  {ts()} Encoding {len(held)} queries [{vn}]...")
            q_embs = model.encode(view_queries[vn], batch_size=64,
                                  show_progress_bar=True,
                                  normalize_embeddings=True).astype(np.float32)
            np.save(out_path, q_embs)
            print(f"    Saved: {out_path} (shape={q_embs.shape})")

        del model
        manifest_folds.append({
            "fold_id": fold_i,
            "model_dir": str(model_dir),
            "n_val": len(held),
            "views": list(VIEW_NAMES),
        })
        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold_i} elapsed: {fold_elapsed:.1f}s")

    # Save manifest
    manifest = {
        "folds": manifest_folds,
        "track_text_format": "r21_album_tags",
        "n_tracks": n_tracks,
        "created_at": datetime.now().isoformat(),
        "track_emb_source": "r49a cache or re-encoded",
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{ts()} Saved manifest: {MANIFEST_PATH}")

    total_elapsed = time.time() - t0
    print(f"{ts()} Encode complete. Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


# ── Phase 2: Evaluate ─────────────────────────────────────────────


def run_evaluate():
    """Load cached embeddings and per-view lists, run 5 LambdaRank configs."""
    import lightgbm as lgb

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
    from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
    from scripts.tune_postrank_v23 import tokens

    t0 = time.time()
    print(f"{ts()} R50: View-aware ranking evaluation")
    print("=" * 70)

    # ── Check manifest ─────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print("ERROR: Manifest not found. Run --phase encode first.")
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    if manifest.get("track_text_format") != "r21_album_tags":
        print(f"ERROR: track_text_format={manifest.get('track_text_format')}, expected r21_album_tags")
        sys.exit(1)
    print(f"{ts()} Manifest OK: {manifest['n_tracks']} tracks, {len(manifest['folds'])} folds")

    # ── Feature names ──────────────────────────────────────────────
    FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
    FEAT_ALBUM = [
        "same_album_last1", "same_album_last3", "same_album_any",
        "album_history_count", "pool_same_album_count",
    ]
    FEAT_R39 = FEAT_BASE + FEAT_ALBUM  # 34 features

    FEAT_VIEW = [
        "current_cosine",                   # 34
        "lastquery_rank_inv",               # 35
        "lastquery_presence",               # 36
        "lastquery_cosine",                 # 37
        "history_rank_inv",                 # 38
        "history_presence",                 # 39
        "history_cosine",                   # 40
        "max_cosine",                       # 41
        "n_views_present",                  # 42
        "rank_std",                         # 43
        "current_minus_lastquery_cosine",   # 44
    ]
    FEAT_ALL = FEAT_R39 + FEAT_VIEW  # 45 features

    n_feat_r39 = len(FEAT_R39)   # 34
    n_feat_all = len(FEAT_ALL)   # 45

    # ── Config definitions ─────────────────────────────────────────
    # Indices of view features (starting from 34)
    VIEW_BASE = n_feat_r39  # 34
    configs = {
        "baseline": {
            "n_feat": n_feat_r39,
            "desc": "R39 exact (34 features)",
        },
        "cosine_only": {
            "indices": list(range(n_feat_r39)) + [
                VIEW_BASE + 0,  # current_cosine
                VIEW_BASE + 3,  # lastquery_cosine
                VIEW_BASE + 6,  # history_cosine
                VIEW_BASE + 7,  # max_cosine
            ],
            "n_feat": n_feat_r39 + 4,
            "desc": "R39 + 4 cosine features (38 total)",
        },
        "alt_rank_only": {
            "indices": list(range(n_feat_r39)) + [
                VIEW_BASE + 1,  # lastquery_rank_inv
                VIEW_BASE + 2,  # lastquery_presence
                VIEW_BASE + 4,  # history_rank_inv
                VIEW_BASE + 5,  # history_presence
            ],
            "n_feat": n_feat_r39 + 4,
            "desc": "R39 + 4 alt-view rank features (38 total)",
        },
        "cross_view_only": {
            "indices": list(range(n_feat_r39)) + [
                VIEW_BASE + 7,   # max_cosine
                VIEW_BASE + 8,   # n_views_present
                VIEW_BASE + 9,   # rank_std
                VIEW_BASE + 10,  # current_minus_lastquery_cosine
            ],
            "n_feat": n_feat_r39 + 4,
            "desc": "R39 + 4 cross-view features (38 total)",
        },
        "full_safe": {
            "indices": list(range(n_feat_all)),
            "n_feat": n_feat_all,
            "desc": f"R39 + all 11 view features ({n_feat_all} total)",
        },
    }

    # ── Load data ──────────────────────────────────────────────────
    print(f"{ts()} Loading payload...")
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

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    # Track IDs and mapping
    with open(TRACK_IDS_PATH) as f:
        all_track_ids = json.load(f)
    tid_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

    # Fold indices
    with open(FOLD_INDICES_PATH) as f:
        fold_indices_raw = json.load(f)
    fold_indices = {int(k): v for k, v in fold_indices_raw.items()}

    # Build case_to_fold mapping
    case_fold = np.full(n, -1, dtype=np.int32)
    case_local_idx = np.full(n, -1, dtype=np.int32)
    for fi in range(5):
        for local_i, global_i in enumerate(fold_indices[fi]):
            case_fold[global_i] = fi
            case_local_idx[global_i] = local_i

    # ── Load per-view OOF lists (from R49A) ────────────────────────
    print(f"{ts()} Loading per-view OOF lists...")
    view_oof_lists = {}
    for vn in VIEW_NAMES:
        path = REPO / "cache" / f"r49a_view_{vn}_oof_lists.json"
        with open(path) as f:
            view_oof_lists[vn] = json.load(f)
        print(f"  {vn}: {len(view_oof_lists[vn])} lists")

    # ── Load per-fold embeddings ───────────────────────────────────
    print(f"{ts()} Loading per-fold track and query embeddings...")
    fold_track_embs = {}
    fold_query_embs = {}  # fold_i -> view_name -> np.array

    for fi in range(5):
        # Track embeddings
        track_emb_path = REPO / "cache" / f"r49a_fold_{fi}_track_embs.npy"
        fold_track_embs[fi] = np.load(track_emb_path)
        print(f"  Fold {fi} track embs: {fold_track_embs[fi].shape}")

        # Query embeddings per view
        fold_query_embs[fi] = {}
        for vn in VIEW_NAMES:
            q_path = REPO / "cache" / f"r50_fold_{fi}_query_embs_{vn}.npy"
            fold_query_embs[fi][vn] = np.load(q_path)
            print(f"  Fold {fi} query embs [{vn}]: {fold_query_embs[fi][vn].shape}")

    # ── Load album mapping ─────────────────────────────────────────
    print(f"{ts()} Loading track metadata for albums...")
    from scripts.expR49c_structural_forensics import load_full_track_metadata, load_track_albums
    track_meta = load_full_track_metadata()
    track_album = load_track_albums(track_meta)
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {non_empty} with album_id")

    # ── Build ALS ──────────────────────────────────────────────────
    print(f"{ts()} Building ALS...")
    track_pop = build_popularity_stats()
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

    folds = grouped_session_folds(sessions, seed=0)
    n_feat_base = len(FEAT_BASE)

    # ── Build per-view rank maps ───────────────────────────────────
    print(f"{ts()} Building per-view rank maps...")
    # For lastquery and history views: tid -> rank (1-indexed) per case
    view_rank_maps = {}
    for vn in ["lastquery", "history"]:
        view_rank_maps[vn] = []
        for i in range(n):
            rm = {tid: r + 1 for r, tid in enumerate(view_oof_lists[vn][i][:300])}
            view_rank_maps[vn].append(rm)

    # ── Build feature matrix (all 45 features) ─────────────────────
    print(f"{ts()} Building feature matrix ({n_feat_all} features)...")
    X = np.zeros((n, POOL_K, n_feat_all), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = []

    for i, c in enumerate(cases):
        fi = int(case_fold[i])
        li = int(case_local_idx[i])

        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = [str(r["content"]) for r in c["history"]
                     if r["role"] == "user"] + [c["user_query"]]
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

        # Per-view rank maps for this case
        lq_rank_map = view_rank_maps["lastquery"][i]
        hist_rank_map = view_rank_maps["history"][i]

        # Query embeddings for this case from the correct fold
        q_emb_current = fold_query_embs[fi]["current"][li]
        q_emb_lastquery = fold_query_embs[fi]["lastquery"][li]
        q_emb_history = fold_query_embs[fi]["history"][li]

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
            for fi2, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi2] = 1.0 / sr if sr else 0.0
            for fi2, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi2] = 1.0 if tid in src_rank[sname] else 0.0
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

            # Album features (indices 29-33, EXACT copy from R39)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K]
                                   if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

            # ── View features (indices 34-44) ──────────────────────
            emb_idx = tid_to_idx.get(tid)
            if emb_idx is not None:
                track_emb = fold_track_embs[fi][emb_idx]

                # [34] current_cosine
                current_cos = float(np.dot(q_emb_current, track_emb))
                row[VIEW_BASE + 0] = current_cos

                # [37] lastquery_cosine
                lastquery_cos = float(np.dot(q_emb_lastquery, track_emb))
                row[VIEW_BASE + 3] = lastquery_cos

                # [40] history_cosine
                history_cos = float(np.dot(q_emb_history, track_emb))
                row[VIEW_BASE + 6] = history_cos

                # [41] max_cosine
                row[VIEW_BASE + 7] = max(current_cos, lastquery_cos, history_cos)

                # [44] current_minus_lastquery_cosine
                row[VIEW_BASE + 10] = current_cos - lastquery_cos
            else:
                current_cos = 0.0
                lastquery_cos = 0.0
                history_cos = 0.0

            # [35] lastquery_rank_inv
            lq_r = lq_rank_map.get(tid)
            row[VIEW_BASE + 1] = 1.0 / lq_r if lq_r else 0.0

            # [36] lastquery_presence
            lq_pres = 1.0 if lq_r else 0.0
            row[VIEW_BASE + 2] = lq_pres

            # [38] history_rank_inv
            hist_r = hist_rank_map.get(tid)
            row[VIEW_BASE + 4] = 1.0 / hist_r if hist_r else 0.0

            # [39] history_presence
            hist_pres = 1.0 if hist_r else 0.0
            row[VIEW_BASE + 5] = hist_pres

            # [42] n_views_present = r21_presence (current) + lastquery_presence + history_presence
            r21_pres = row[28]  # r21_presence from R39 features
            row[VIEW_BASE + 8] = r21_pres + lq_pres + hist_pres

            # [43] rank_std: std of ranks for views where present
            ranks_present = []
            # Current view: use r21_rank_inv (index 27) to get rank
            if row[27] > 0:
                ranks_present.append(1.0 / row[27])
            if lq_r:
                ranks_present.append(float(lq_r))
            if hist_r:
                ranks_present.append(float(hist_r))
            if len(ranks_present) > 1:
                row[VIEW_BASE + 9] = float(np.std(ranks_present))
            else:
                row[VIEW_BASE + 9] = 0.0

        if (i + 1) % 2000 == 0:
            print(f"  {ts()} Built features for {i+1}/{n} cases...")

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # ── CV5 LambdaRank per config ──────────────────────────────────
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }

    all_results = {}
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]

    for cfg_name, cfg in configs.items():
        cfg_t0 = time.time()
        print(f"\n{ts()} Config: {cfg_name} ({cfg['desc']})")
        print("-" * 70)

        if cfg_name == "baseline":
            # Use first n_feat_r39 columns
            feat_indices = list(range(n_feat_r39))
            feat_names = list(FEAT_R39)
            n_f = n_feat_r39
        else:
            feat_indices = cfg["indices"]
            feat_names = [FEAT_ALL[j] for j in feat_indices]
            n_f = cfg["n_feat"]

        case_ndcg = np.zeros(n)
        lr_scores = np.full((n, POOL_K), -np.inf, dtype=np.float64)
        feat_imp_accum = np.zeros(n_f, dtype=np.float64)

        for fi in range(5):
            val_set = set(folds[fi].tolist())
            tr = [j for j in range(n) if j not in val_set]
            va = sorted(val_set)
            X_tr, y_tr, g_tr = [], [], []
            X_va, y_va, g_va = [], [], []
            for idx in tr:
                s = int(sizes[idx])
                for k in range(s):
                    X_tr.append(X[idx, k, feat_indices])
                    y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
                g_tr.append(s)
            for idx in va:
                s = int(sizes[idx])
                for k in range(s):
                    X_va.append(X[idx, k, feat_indices])
                    y_va.append(1.0 if k == gt_idx[idx] else 0.0)
                g_va.append(s)
            ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                                group=g_tr, feature_name=list(feat_names))
            ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                                group=g_va, reference=ds_tr)
            model = lgb.train(lr_params, ds_tr, num_boost_round=300,
                              valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
            preds = model.predict(np.array(X_va))
            feat_imp_accum += np.array(model.feature_importance(importance_type='gain'),
                                       dtype=np.float64)
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

        feat_imp_avg = feat_imp_accum / 5.0

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7_idx]))
        cv5_ndcg = float(np.mean(case_ndcg))
        print(f"  h7={h7_ndcg:.5f}  cv5={cv5_ndcg:.5f}  pool_hit={pool_hit:.4f}")

        # HARD CHECK for baseline
        if cfg_name == "baseline":
            expected_h7 = 0.24298
            if abs(h7_ndcg - expected_h7) > 0.001:
                print("\n  *** HARD CHECK FAILED ***")
                print(f"  Expected h7 ~{expected_h7}, got {h7_ndcg:.5f}")
                print(f"  Difference: {abs(h7_ndcg - expected_h7):.5f} > 0.001")
                print("  ABORTING.")
                sys.exit(1)
            print(f"  HARD CHECK PASSED: h7={h7_ndcg:.5f} within +/-0.001 of {expected_h7}")

        all_results[cfg_name] = {
            "h7": h7_ndcg,
            "cv5": cv5_ndcg,
            "pool_hit": pool_hit,
            "case_ndcg": case_ndcg.copy(),
            "lr_scores": lr_scores.copy(),
            "feat_imp_avg": feat_imp_avg.copy(),
            "feat_names": feat_names,
        }

        cfg_elapsed = time.time() - cfg_t0
        print(f"  Config elapsed: {cfg_elapsed:.1f}s")

    # ── Config comparison ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONFIG COMPARISON:")
    print(f"{'config':<20s} | {'h7':>8s} | {'cv5':>8s} | {'delta_h7':>8s} | {'n_feat':>6s}")
    print("-" * 60)

    baseline_h7 = all_results["baseline"]["h7"]
    best_cfg = "baseline"
    best_h7 = baseline_h7

    for cfg_name in configs:
        r = all_results[cfg_name]
        delta = r["h7"] - baseline_h7
        n_f = configs[cfg_name]["n_feat"] if "n_feat" in configs[cfg_name] else n_feat_r39
        marker = " ***" if cfg_name != "baseline" and r["h7"] > best_h7 else ""
        print(f"{cfg_name:<20s} | {r['h7']:>8.5f} | {r['cv5']:>8.5f} | {delta:>+8.5f} | {n_f:>6d}{marker}")
        if r["h7"] > best_h7:
            best_h7 = r["h7"]
            best_cfg = cfg_name

    print(f"\nBest non-baseline config: {best_cfg} (h7={best_h7:.5f}, delta={best_h7-baseline_h7:+.5f})")

    # ── Detailed analysis for best non-baseline config ─────────────
    if best_cfg == "baseline":
        # All configs were worse, pick the one closest to baseline
        non_base = {k: v for k, v in all_results.items() if k != "baseline"}
        best_cfg = max(non_base, key=lambda k: non_base[k]["h7"])
        best_h7 = all_results[best_cfg]["h7"]
        print(f"  (No improvement; analyzing {best_cfg} anyway)")

    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: {best_cfg} vs baseline")
    print("-" * 70)

    best = all_results[best_cfg]
    base = all_results["baseline"]

    n_h7 = len(h7_idx)

    # Recovered / lost
    base_hits = set(i for i in h7_idx if base["case_ndcg"][i] > 0)
    best_hits = set(i for i in h7_idx if best["case_ndcg"][i] > 0)
    recovered = best_hits - base_hits
    lost = base_hits - best_hits

    print(f"  Base h7 hits: {len(base_hits)}")
    print(f"  Best h7 hits: {len(best_hits)}")
    print(f"  Recovered (miss->hit): {len(recovered)}")
    print(f"  Lost (hit->miss): {len(lost)}")
    print(f"  Net: {len(recovered) - len(lost):+d}")

    # Top-20 churn (membership change)
    top20_changed = 0
    for i in h7_idx:
        ps = int(sizes[i])
        if ps == 0:
            continue
        base_sc = base["lr_scores"][i, :ps]
        best_sc = best["lr_scores"][i, :ps]
        base_top20 = set(int(x) for x in np.argsort(-base_sc)[:20])
        best_top20 = set(int(x) for x in np.argsort(-best_sc)[:20])
        if base_top20 != best_top20:
            top20_changed += 1
    print(f"  Top-20 membership changed: {top20_changed}/{n_h7}")

    # Feature importance top-15 by gain
    print(f"\n  Feature importance top-15 ({best_cfg}, avg gain across 5 folds):")
    fi_list = list(zip(best["feat_names"], best["feat_imp_avg"]))
    fi_sorted = sorted(fi_list, key=lambda x: -x[1])
    top15 = fi_sorted[:15]
    for fname, gain in top15:
        print(f"    {fname:>40s}: {gain:.2f}")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "experiment": "R50_view_aware_ranking",
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "best_non_baseline": best_cfg,
        "analysis": {
            "base_hits": len(base_hits),
            "best_hits": len(best_hits),
            "recovered": len(recovered),
            "lost": len(lost),
            "net": len(recovered) - len(lost),
            "recovered_cases": sorted(recovered),
            "lost_cases": sorted(lost),
            "top20_membership_changed": top20_changed,
            "feature_importance_top15": [
                {"name": fname, "gain": round(float(gain), 2)}
                for fname, gain in top15
            ],
        },
    }

    for cfg_name in configs:
        r = all_results[cfg_name]
        output["configs"][cfg_name] = {
            "desc": configs[cfg_name]["desc"],
            "h7": round(r["h7"], 5),
            "cv5": round(r["cv5"], 5),
            "pool_hit": round(r["pool_hit"], 4),
            "delta_h7": round(r["h7"] - baseline_h7, 5),
            "n_features": configs[cfg_name]["n_feat"] if "n_feat" in configs[cfg_name] else n_feat_r39,
        }

    out_path = REPO / "exp" / "eval" / "expR50_view_aware_ranking.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{ts()} Saved to {out_path}")

    total_elapsed = time.time() - t0
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R50: View-aware ranking features")
    parser.add_argument("--phase", required=True,
                        choices=["encode", "evaluate"],
                        help="encode: per-view query embeddings, evaluate: LambdaRank configs")
    args = parser.parse_args()

    if args.phase == "encode":
        run_encode()
    elif args.phase == "evaluate":
        run_evaluate()
