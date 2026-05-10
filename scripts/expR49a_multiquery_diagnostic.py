#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S101
"""R49a: Multi-query inference diagnostic.

Phase 1 (--phase encode): Build per-view OOF top-300 lists using 3 query views
  across 5 folds of R21 models. Saves per-view lists.

Phase 2 (--phase evaluate): RRF-fuse views into R21_MULTI source, then
  evaluate with R39 LambdaRank at multiple weights.

Run:
  uv run scripts/expR49a_multiquery_diagnostic.py --phase encode
  uv run scripts/expR49a_multiquery_diagnostic.py --phase evaluate
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

MODEL_DIRS = {
    0: REPO / "cache" / "r21_supervised" / "fold_0",
    1: REPO / "cache" / "r21_supervised" / "fold_1",
    2: REPO / "cache" / "r21_production" / "oof" / "model_fold_2",
    3: REPO / "cache" / "r21_production" / "oof" / "model_fold_3",
    4: REPO / "cache" / "r21_production" / "oof" / "model_fold_4",
}

VIEW_NAMES = ["current", "lastquery", "history"]
TOP_K = 300


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# ── Track text encoding (same as rebuild_r46) ───────────────────────


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


def _first_or_str(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value else ""


def _artists_to_str(value):
    if isinstance(value, list):
        return ", ".join(str(a) for a in value)
    return str(value) if value else ""


# ── Query view builders ─────────────────────────────────────────────


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


# ── Phase 1: Encode ─────────────────────────────────────────────────


def encode_single_fold(fold_i):
    """Encode one fold's 3 query views and save per-fold results."""
    t0 = time.time()
    print(f"{ts()} R49a: Encoding fold {fold_i} (3 views, top-{TOP_K})")
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
    held = fold_indices[fold_i]
    print(f"  {len(held)} val cases for fold {fold_i}")

    # Load track IDs
    with open(TRACK_IDS_PATH) as f:
        all_track_ids = json.load(f)
    n_tracks = len(all_track_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

    # Load track metadata
    print(f"{ts()} Loading track metadata...")
    track_meta = load_catalog_metadata()

    # Build track texts
    print(f"{ts()} Building track texts...")
    track_texts = [build_track_text_r21(tid, track_meta) for tid in all_track_ids]

    # Load model
    model_dir = MODEL_DIRS[fold_i]
    print(f"{ts()} Loading model: {model_dir}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_dir), device="cpu")
    set_eval = getattr(model, "eval")
    set_eval()

    # Check for cached track embeddings
    cache_path = REPO / "cache" / f"r49a_fold_{fold_i}_track_embs.npy"
    if cache_path.exists():
        print(f"  {ts()} Loading cached track embeddings: {cache_path}")
        track_embs = np.load(cache_path)
    else:
        print(f"  {ts()} Encoding {n_tracks} tracks...")
        track_embs = model.encode(track_texts, batch_size=128,
                                  show_progress_bar=True,
                                  normalize_embeddings=True).astype(np.float32)
        np.save(cache_path, track_embs)
        print(f"  Cached: {cache_path}")

    # Build queries for all 3 views
    val_cases = [cases[j] for j in held]
    view_queries = {
        "current": [build_view_current(c) for c in val_cases],
        "lastquery": [build_view_lastquery(c) for c in val_cases],
        "history": [build_view_history(c, track_meta) for c in val_cases],
    }

    # Per-view retrieval
    fold_results = {vn: {} for vn in VIEW_NAMES}  # vn -> {j_global: list}

    for vn in VIEW_NAMES:
        print(f"  {ts()} Encoding {len(held)} queries [{vn}]...")
        q_embs = model.encode(view_queries[vn], batch_size=64,
                              show_progress_bar=True,
                              normalize_embeddings=True).astype(np.float32)

        print(f"  {ts()} Retrieving top-{TOP_K} [{vn}]...")
        for j_local in range(len(val_cases)):
            scores = track_embs @ q_embs[j_local]

            # Exclude played tracks
            played_set = set(val_cases[j_local]["music_turns"])
            for tid in played_set:
                idx = tid_to_idx.get(tid)
                if idx is not None:
                    scores[idx] = -np.inf

            # Top-K by argpartition
            top_idx = np.argpartition(-scores, TOP_K - 1)[:TOP_K]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            fold_list = [all_track_ids[k] for k in top_idx]

            j_global = held[j_local]
            fold_results[vn][j_global] = fold_list

    # Sanity check: current view vs existing OOF
    with open(R21_OOF) as f:
        existing_oof = json.load(f)
    match_count = 0
    for j_global in held:
        if fold_results["current"][j_global] == existing_oof[j_global]:
            match_count += 1
    pct = match_count / len(held) * 100
    print(f"  Sanity: current view vs existing OOF: {match_count}/{len(held)} ({pct:.1f}%)")

    # GT coverage stats
    for vn in VIEW_NAMES:
        gt_found = sum(1 for j in held if cases[j]["gt"] in fold_results[vn][j])
        print(f"  GT in top-{TOP_K} [{vn}]: {gt_found}/{len(held)}")

    # Save per-fold results
    out_path = REPO / "cache" / f"r49a_fold_{fold_i}_view_lists.json"
    # Convert to serializable format
    save_data = {}
    for vn in VIEW_NAMES:
        save_data[vn] = {str(k): v for k, v in fold_results[vn].items()}
    with open(out_path, "w") as f:
        json.dump(save_data, f)
    print(f"{ts()} Saved fold {fold_i} results to {out_path}")

    del model, track_embs
    elapsed = time.time() - t0
    print(f"{ts()} Fold {fold_i} complete. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


def phase_merge():
    """Merge per-fold results into combined per-view OOF lists."""
    t0 = time.time()
    print(f"{ts()} R49a: Merging per-fold results into combined view lists")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    view_lists = {vn: [None] * n for vn in VIEW_NAMES}

    for fold_i in range(5):
        fold_path = REPO / "cache" / f"r49a_fold_{fold_i}_view_lists.json"
        print(f"  Loading fold {fold_i}: {fold_path}")
        with open(fold_path) as f:
            fold_data = json.load(f)
        for vn in VIEW_NAMES:
            for j_str, lst in fold_data[vn].items():
                j = int(j_str)
                view_lists[vn][j] = lst

    # Verify completeness
    for vn in VIEW_NAMES:
        assert all(x is not None for x in view_lists[vn]), f"Missing lists for view {vn}!"

    # Save per-view lists
    for vn in VIEW_NAMES:
        out_path = REPO / "cache" / f"r49a_view_{vn}_oof_lists.json"
        with open(out_path, "w") as f:
            json.dump(view_lists[vn], f)
        print(f"{ts()} Saved {out_path} ({n} lists)")

    # GT coverage stats
    print(f"\n{'='*70}")
    print("GT coverage per view (all 8000 cases):")
    for vn in VIEW_NAMES:
        gt_found = sum(1 for i in range(n) if cases[i]["gt"] in view_lists[vn][i])
        print(f"  {vn}: {gt_found}/{n} ({gt_found/n:.4f})")

    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"\nGT coverage h7 ({len(h7)} cases):")
    for vn in VIEW_NAMES:
        gt_found = sum(1 for i in h7 if cases[i]["gt"] in view_lists[vn][i])
        print(f"  {vn}: {gt_found}/{len(h7)} ({gt_found/len(h7):.4f})")

    print(f"\n{ts()} Merge complete. Elapsed: {time.time()-t0:.1f}s")


# ── Phase 2: Evaluate ───────────────────────────────────────────────


def phase_evaluate():
    t0 = time.time()
    print(f"{ts()} R49a Phase 2: Evaluate multi-query configs with R39 LambdaRank")
    print("=" * 70)

    # Imports for evaluation (lightgbm OK, no sentence_transformers)
    import lightgbm as lgb

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
    from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
    from scripts.tune_postrank_v23 import tokens

    RRF_K = 20
    POOL_K = 300

    FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
    FEAT_ALBUM = [
        "same_album_last1",
        "same_album_last3",
        "same_album_any",
        "album_history_count",
        "pool_same_album_count",
    ]
    FEAT_ALL = FEAT_BASE + FEAT_ALBUM
    n_feat_base = len(FEAT_BASE)
    n_feat = len(FEAT_ALL)

    # Load payload
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

    # Load R21 OOF (baseline)
    with open(R21_OOF) as f:
        r21_oof_baseline = json.load(f)

    # Load per-view lists
    print(f"{ts()} Loading per-view OOF lists...")
    view_lists = {}
    for vn in VIEW_NAMES:
        path = REPO / "cache" / f"r49a_view_{vn}_oof_lists.json"
        with open(path) as f:
            view_lists[vn] = json.load(f)
        print(f"  {vn}: {len(view_lists[vn])} lists")

    # Build R21_MULTI via RRF across views
    print(f"{ts()} Building R21_MULTI via RRF (k=20)...")
    r21_multi = []
    for i in range(n):
        view_sources = {
            "v1": view_lists["current"][i],
            "v2": view_lists["lastquery"][i],
            "v3": view_lists["history"][i],
        }
        view_weights = {"v1": 1.0, "v2": 1.0, "v3": 1.0}
        r21_multi.append(weighted_rrf(view_sources, view_weights, topk=300, k=20))

    # Save R21_MULTI
    multi_path = REPO / "cache" / "r49a_r21_multi_oof_lists.json"
    with open(multi_path, "w") as f:
        json.dump(r21_multi, f)
    print(f"  Saved R21_MULTI to {multi_path}")

    # Track metadata for album features
    print(f"{ts()} Loading track metadata...")
    from scripts.expR49c_structural_forensics import load_full_track_metadata, load_track_albums
    track_meta = load_full_track_metadata()
    track_album = load_track_albums(track_meta)
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {non_empty} with album_id")

    # Build ALS
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

    # ── Define configs ──────────────────────────────────────────────
    configs = {
        "baseline": {
            "r21_source": r21_oof_baseline,
            "sw": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
            "desc": "R39 exact (R21 OOF baseline)",
        },
        "multi_w0.3": {
            "r21_source": r21_multi,
            "sw": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 0.3},
            "desc": "R21_MULTI weight=0.3",
        },
        "multi_w0.5": {
            "r21_source": r21_multi,
            "sw": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 0.5},
            "desc": "R21_MULTI weight=0.5",
        },
        "multi_w1.0": {
            "r21_source": r21_multi,
            "sw": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
            "desc": "R21_MULTI weight=1.0",
        },
    }

    # ── Run configs ─────────────────────────────────────────────────
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }

    all_results = {}

    for cfg_name, cfg in configs.items():
        cfg_t0 = time.time()
        print(f"\n{ts()} Config: {cfg_name} ({cfg['desc']})")
        print("-" * 70)

        r21_source = cfg["r21_source"]
        sw = cfg["sw"]

        # Build feature matrix
        X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        pools: list[list[str]] = []

        for i, c in enumerate(cases):
            src_lists = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
                "R21": r21_source[i],
            }
            pool = weighted_rrf(src_lists, sw, topk=POOL_K, k=RRF_K)
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

            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank - 1]

                # Base 29 features (EXACT copy from R39a/R49c)
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

                # Album features (EXACT copy from R39a/R49c)
                c_album = track_album.get(tid, "")
                row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
                row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
                row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
                row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
                pool_album_count = sum(1 for tid2 in pool[:POOL_K]
                                       if track_album.get(tid2, "") == c_album) if c_album else 0
                row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

        # CV5 LambdaRank
        print(f"  {ts()} Running CV5 LambdaRank...")
        case_ndcg = np.zeros(n)
        lr_scores = np.full((n, POOL_K), -np.inf, dtype=np.float64)

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
        cv5_ndcg = float(np.mean(case_ndcg))
        print(f"  h7={h7_ndcg:.5f}  cv5={cv5_ndcg:.5f}  pool_hit={pool_hit:.4f}")

        # HARD CHECK for baseline
        if cfg_name == "baseline":
            expected_h7 = 0.24298
            if abs(h7_ndcg - expected_h7) > 0.001:
                print("\n  *** HARD CHECK FAILED ***")
                print(f"  Expected h7 ~{expected_h7}, got {h7_ndcg:.5f}")
                print(f"  Difference: {abs(h7_ndcg - expected_h7):.5f} > 0.001")
                sys.exit(1)
            print(f"  HARD CHECK PASSED: h7={h7_ndcg:.5f} within +/-0.001 of {expected_h7}")

        all_results[cfg_name] = {
            "h7": h7_ndcg,
            "cv5": cv5_ndcg,
            "pool_hit": pool_hit,
            "case_ndcg": case_ndcg,
            "lr_scores": lr_scores,
            "pools": pools,
            "gt_idx": gt_idx,
            "sizes": sizes,
        }

        cfg_elapsed = time.time() - cfg_t0
        print(f"  Config elapsed: {cfg_elapsed:.1f}s")

    # ── Find best config ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONFIG COMPARISON:")
    print(f"{'config':<20s} | {'h7':>8s} | {'cv5':>8s} | {'pool_hit':>9s} | {'delta_h7':>8s}")
    print("-" * 65)

    baseline_h7 = all_results["baseline"]["h7"]
    best_cfg = "baseline"
    best_h7 = baseline_h7

    for cfg_name in configs:
        r = all_results[cfg_name]
        delta = r["h7"] - baseline_h7
        marker = " <-- best" if r["h7"] == max(rr["h7"] for rr in all_results.values()) else ""
        print(f"{cfg_name:<20s} | {r['h7']:>8.5f} | {r['cv5']:>8.5f} | {r['pool_hit']:>9.4f} | {delta:>+8.5f}{marker}")
        if r["h7"] > best_h7:
            best_h7 = r["h7"]
            best_cfg = cfg_name

    print(f"\nBest config: {best_cfg} (h7={best_h7:.5f})")

    # ── Detailed analysis for best config ───────────────────────────
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: {best_cfg} vs baseline")
    print("-" * 70)

    best = all_results[best_cfg]
    base = all_results["baseline"]

    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    n_h7 = len(h7)

    # Hit/miss analysis
    base_hits = set(i for i in h7 if base["case_ndcg"][i] > 0)
    best_hits = set(i for i in h7 if best["case_ndcg"][i] > 0)

    recovered = best_hits - base_hits
    lost = base_hits - best_hits
    stable_hits = base_hits & best_hits
    stable_misses = set(h7) - base_hits - best_hits

    print(f"  Base hits:    {len(base_hits)}")
    print(f"  Best hits:    {len(best_hits)}")
    print(f"  Recovered:    {len(recovered)} (miss->hit)")
    print(f"  Lost:         {len(lost)} (hit->miss)")
    print(f"  Net:          {len(recovered) - len(lost):+d}")
    print(f"  Stable hits:  {len(stable_hits)}")
    print(f"  Stable miss:  {len(stable_misses)}")

    # Bucket recovery analysis
    print("\nBucket recovery analysis (baseline miss -> best hit):")

    # Classify baseline misses into buckets
    bucket_recovered = Counter()
    bucket_total = Counter()

    for i in h7:
        if i in base_hits:
            continue  # already a hit in baseline

        gt = cases[i]["gt"]
        base_pool = base["pools"][i]
        base_ps = int(base["sizes"][i])
        base_sc = base["lr_scores"][i, :base_ps]

        if gt in base_pool:
            gt_pool_idx = base_pool.index(gt)
            lr_ranked = np.argsort(-base_sc)
            gt_lr_rank_arr = np.where(lr_ranked == gt_pool_idx)[0]
            if len(gt_lr_rank_arr) > 0:
                gt_rank = int(gt_lr_rank_arr[0]) + 1
                if gt_rank <= 100:
                    bucket = "B"
                else:
                    bucket = "C"
            else:
                bucket = "D"
        else:
            # Check if GT is in any source
            src_union = set()
            src_union.update(payload["src_a"][i])
            src_union.update(payload["src_b"][i])
            src_union.update(payload["src_c"][i])
            src_union.update(payload["src_d"][i])
            src_union.update(payload["src_f"][i])
            src_union.update(als_source[i])
            src_union.update(r21_oof_baseline[i])
            if gt in src_union:
                bucket = "D"
            else:
                bucket = "E"

        bucket_total[bucket] += 1
        if i in recovered:
            bucket_recovered[bucket] += 1

    for b in sorted(set(list(bucket_total.keys()) + list(bucket_recovered.keys()))):
        total = bucket_total.get(b, 0)
        rec = bucket_recovered.get(b, 0)
        print(f"  Bucket {b}: {rec}/{total} recovered")

    # Top-1 / Top-20 churn
    print("\nTop-1 churn:")
    top1_same = 0
    top1_diff = 0
    for i in h7:
        base_pool = base["pools"][i]
        best_pool = best["pools"][i]
        base_ps = int(base["sizes"][i])
        best_ps = int(best["sizes"][i])

        if base_ps > 0 and best_ps > 0:
            base_sc = base["lr_scores"][i, :base_ps]
            best_sc = best["lr_scores"][i, :best_ps]
            base_top1 = base_pool[np.argmax(base_sc)]
            best_top1 = best_pool[np.argmax(best_sc)]
            if base_top1 == best_top1:
                top1_same += 1
            else:
                top1_diff += 1
    print(f"  Same top-1: {top1_same}/{n_h7}")
    print(f"  Diff top-1: {top1_diff}/{n_h7}")

    # Top-20 churn
    print("\nTop-20 churn:")
    top20_overlaps = []
    for i in h7:
        base_pool = base["pools"][i]
        best_pool = best["pools"][i]
        base_ps = int(base["sizes"][i])
        best_ps = int(best["sizes"][i])

        if base_ps > 0 and best_ps > 0:
            base_sc = base["lr_scores"][i, :base_ps]
            best_sc = best["lr_scores"][i, :best_ps]
            base_top20_idx = np.argsort(-base_sc)[:20]
            best_top20_idx = np.argsort(-best_sc)[:20]
            base_top20 = set(base_pool[j] for j in base_top20_idx if j < len(base_pool))
            best_top20 = set(best_pool[j] for j in best_top20_idx if j < len(best_pool))
            overlap = len(base_top20 & best_top20) / max(len(base_top20), 1)
            top20_overlaps.append(overlap)

    if top20_overlaps:
        print(f"  Mean top-20 overlap: {np.mean(top20_overlaps):.4f}")
        print(f"  Median top-20 overlap: {np.median(top20_overlaps):.4f}")

    # Same/diff artist analysis for recovered/lost
    print("\nSame/diff artist analysis:")
    rec_same_artist = 0
    rec_diff_artist = 0
    for i in recovered:
        gt = cases[i]["gt"]
        played = cases[i]["music_turns"]
        gt_artist = ta.get(gt, "")
        played_artists = set(ta.get(t, "") for t in played) - {""}
        if gt_artist and gt_artist in played_artists:
            rec_same_artist += 1
        else:
            rec_diff_artist += 1
    print(f"  Recovered same_artist: {rec_same_artist}")
    print(f"  Recovered diff_artist: {rec_diff_artist}")

    lost_same_artist = 0
    lost_diff_artist = 0
    for i in lost:
        gt = cases[i]["gt"]
        played = cases[i]["music_turns"]
        gt_artist = ta.get(gt, "")
        played_artists = set(ta.get(t, "") for t in played) - {""}
        if gt_artist and gt_artist in played_artists:
            lost_same_artist += 1
        else:
            lost_diff_artist += 1
    print(f"  Lost same_artist: {lost_same_artist}")
    print(f"  Lost diff_artist: {lost_diff_artist}")

    # ── Save results ────────────────────────────────────────────────
    output = {
        "experiment": "R49a_multiquery_diagnostic",
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "best_config": best_cfg,
        "analysis": {
            "base_hits": len(base_hits),
            "best_hits": len(best_hits),
            "recovered": len(recovered),
            "lost": len(lost),
            "net": len(recovered) - len(lost),
            "recovered_cases": sorted(recovered),
            "lost_cases": sorted(lost),
            "bucket_recovery": {b: {"recovered": bucket_recovered.get(b, 0),
                                     "total": bucket_total.get(b, 0)}
                                for b in sorted(set(list(bucket_total.keys()) +
                                                    list(bucket_recovered.keys())))},
            "top1_same": top1_same,
            "top1_diff": top1_diff,
            "top20_mean_overlap": float(np.mean(top20_overlaps)) if top20_overlaps else 0,
            "recovered_same_artist": rec_same_artist,
            "recovered_diff_artist": rec_diff_artist,
            "lost_same_artist": lost_same_artist,
            "lost_diff_artist": lost_diff_artist,
        },
    }

    for cfg_name in configs:
        r = all_results[cfg_name]
        output["configs"][cfg_name] = {
            "desc": configs[cfg_name]["desc"],
            "h7": r["h7"],
            "cv5": r["cv5"],
            "pool_hit": r["pool_hit"],
            "delta_h7": r["h7"] - baseline_h7,
        }

    out_path = REPO / "exp" / "eval" / "expR49a_multiquery_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{ts()} Saved to {out_path}")

    total_elapsed = time.time() - t0
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


# ── Main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R49a: Multi-query diagnostic")
    parser.add_argument("--phase", required=True,
                        choices=["encode", "merge", "evaluate"],
                        help="encode: per-fold encoding, merge: combine folds, evaluate: LambdaRank")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (0-4) for encode phase")
    args = parser.parse_args()

    if args.phase == "encode":
        if args.fold is not None:
            encode_single_fold(args.fold)
        else:
            for fi in range(5):
                encode_single_fold(fi)
    elif args.phase == "merge":
        phase_merge()
    elif args.phase == "evaluate":
        phase_evaluate()
