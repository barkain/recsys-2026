#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R49c: Structural metadata forensics.

Find NEW stable structural metadata features beyond album.
Examines release_date, artist_id, duration, ISRC patterns in R39 miss cases.
"""
from __future__ import annotations

import json
import os
import pickle
import re
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
FEAT_ALL = FEAT_BASE + FEAT_ALBUM


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def extract_year(release_date: str) -> int | None:
    if not release_date:
        return None
    m = re.match(r"(\d{4})", release_date)
    if m:
        return int(m.group(1))
    return None


def extract_decade(year: int | None) -> str | None:
    if year is None:
        return None
    return f"{(year // 10) * 10}s"


def load_full_track_metadata():
    """Load full track metadata including all structural fields."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    for item in ds:
        tid = str(item["track_id"])
        artist_name_raw = item.get("artist_name", [])
        if isinstance(artist_name_raw, list):
            artist_name = ", ".join(str(a) for a in artist_name_raw)
        else:
            artist_name = str(artist_name_raw)

        track_name_raw = item.get("track_name", [])
        if isinstance(track_name_raw, list) and track_name_raw:
            track_name = str(track_name_raw[0])
        else:
            track_name = str(track_name_raw) if track_name_raw else ""

        album_name_raw = item.get("album_name", [])
        if isinstance(album_name_raw, list) and album_name_raw:
            album_name = str(album_name_raw[0])
        else:
            album_name = str(album_name_raw) if album_name_raw else ""

        album_id_raw = item.get("album_id", [])
        if isinstance(album_id_raw, list) and album_id_raw:
            album_id = str(album_id_raw[0])
        else:
            album_id = ""

        artist_id_raw = item.get("artist_id", [])
        if isinstance(artist_id_raw, list) and artist_id_raw:
            artist_id = str(artist_id_raw[0])
        else:
            artist_id = ""

        release_date = str(item.get("release_date", "")) if item.get("release_date") else ""
        popularity = float(item.get("popularity", 0.0)) if item.get("popularity") is not None else 0.0
        duration = int(item.get("duration", 0)) if item.get("duration") is not None else 0

        isrc_raw = item.get("ISRC", [])
        if isinstance(isrc_raw, list) and isrc_raw:
            isrc = str(isrc_raw[0])
        else:
            isrc = str(isrc_raw) if isrc_raw else ""

        meta[tid] = {
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,
            "album_id": album_id,
            "artist_id": artist_id,
            "release_date": release_date,
            "popularity": popularity,
            "duration": duration,
            "isrc": isrc,
        }
    return meta


def load_track_albums(track_meta):
    """Build track_id -> album_id mapping from full metadata."""
    track_album = {}
    for tid, m in track_meta.items():
        alb_id = m.get("album_id", "")
        if alb_id:
            track_album[tid] = alb_id
        else:
            alb_name = m.get("album_name", "")
            track_album[tid] = alb_name if alb_name else ""
    return track_album


def main():
    t0 = time.time()
    print(f"{ts()} R49c: Structural Metadata Forensics")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Phase 1: Metadata inventory
    # ---------------------------------------------------------------
    print(f"\n{ts()} PHASE 1: Metadata Inventory")
    print("-" * 70)

    print(f"{ts()} Loading full track metadata...")
    track_meta = load_full_track_metadata()
    n_tracks = len(track_meta)
    print(f"  Total tracks: {n_tracks}")

    # Coverage stats
    coverage = {}
    fields_to_check = ["release_date", "artist_id", "duration", "isrc"]
    for field in fields_to_check:
        if field == "duration":
            count = sum(1 for m in track_meta.values() if m.get(field, 0) > 0)
        else:
            count = sum(1 for m in track_meta.values() if m.get(field, ""))
        pct = count / n_tracks * 100
        coverage[field] = round(pct, 2)
        print(f"  {field}: {count}/{n_tracks} ({pct:.1f}%)")

    # Check for album_artist or contributor role fields (they don't exist in this dataset)
    print("  album_artist: NOT PRESENT in dataset schema")
    print("  contributor_roles: NOT PRESENT in dataset schema")

    # ---------------------------------------------------------------
    # Phase 2: Build R39 baseline predictions
    # ---------------------------------------------------------------
    print(f"\n{ts()} PHASE 2: R39 Baseline Reproduction")
    print("-" * 70)

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

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums(track_meta)
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {non_empty} with album_id")

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

    # Build feature matrix
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

        # Album precomputation
        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums if a)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features (EXACT copy from R39a/R40)
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

            # Album features (EXACT copy from R39a/R40)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # CV5 LambdaRank
    print(f"{ts()} Running CV5 LambdaRank (R39 config)...")
    case_ndcg = np.zeros(n)
    lr_scores = np.full((n, POOL_K), -np.inf, dtype=np.float64)
    lr_params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                 "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                 "verbose": -1, "seed": 0}

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

    # Metrics
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
    cv5 = float(np.mean(case_ndcg))
    print(f"  R39 reproduction: h7={h7_ndcg:.5f}  cv5={cv5:.5f}")

    # HARD CHECK
    expected_h7 = 0.24298
    if abs(h7_ndcg - expected_h7) > 0.001:
        print("\n  *** HARD CHECK FAILED ***")
        print(f"  Expected h7 ~{expected_h7}, got {h7_ndcg:.5f}")
        print(f"  Difference: {abs(h7_ndcg - expected_h7):.5f} > 0.001")
        print("  STOPPING. Do not proceed to forensics.")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={h7_ndcg:.5f} within +/-0.001 of {expected_h7}")

    # ---------------------------------------------------------------
    # Classify miss cases into buckets
    # ---------------------------------------------------------------
    print(f"\n{ts()} Classifying h7 miss cases...")

    # Build source union per case
    bucket_counts = Counter()
    miss_cases = []

    for i in h7:
        c = cases[i]
        gt = c["gt"]
        pool = pools[i]
        ps = int(sizes[i])

        # Check if GT is in the pool
        if gt in pool:
            gt_pool_idx = pool.index(gt)
            sc = lr_scores[i, :ps]
            lr_ranked = np.argsort(-sc)
            gt_lr_rank_arr = np.where(lr_ranked == gt_pool_idx)[0]
            if len(gt_lr_rank_arr) == 0:
                continue
            gt_rank = int(gt_lr_rank_arr[0]) + 1  # 1-indexed

            if gt_rank <= 20:
                bucket_counts["A_hit"] += 1
            elif gt_rank <= 100:
                bucket_counts["B_rank21_100"] += 1
                wrong_top5_pool_idx = [int(lr_ranked[j]) for j in range(min(5, ps))
                                       if int(lr_ranked[j]) != gt_pool_idx][:5]
                miss_cases.append({
                    "case_idx": i, "gt": gt, "gt_rank": gt_rank,
                    "bucket": "B",
                    "wrong_top5": [pool[idx] for idx in wrong_top5_pool_idx],
                    "pool": pool, "pool_size": ps,
                })
            else:
                bucket_counts["C_rank100plus"] += 1
                wrong_top5_pool_idx = [int(lr_ranked[j]) for j in range(min(5, ps))
                                       if int(lr_ranked[j]) != gt_pool_idx][:5]
                miss_cases.append({
                    "case_idx": i, "gt": gt, "gt_rank": gt_rank,
                    "bucket": "C",
                    "wrong_top5": [pool[idx] for idx in wrong_top5_pool_idx],
                    "pool": pool, "pool_size": ps,
                })
        else:
            # GT not in pool — check if in source union
            src_lists = src_lists_all[i]
            source_union = set()
            for sl in src_lists.values():
                source_union.update(sl)
            if gt in source_union:
                bucket_counts["D_pool_miss_in_sources"] += 1
                miss_cases.append({
                    "case_idx": i, "gt": gt, "gt_rank": -1,
                    "bucket": "D",
                    "wrong_top5": pool[:5],
                    "pool": pool, "pool_size": ps,
                })
            else:
                bucket_counts["E_unretrievable"] += 1
                miss_cases.append({
                    "case_idx": i, "gt": gt, "gt_rank": -1,
                    "bucket": "E",
                    "wrong_top5": pool[:5],
                    "pool": pool, "pool_size": ps,
                })

    print(f"  Total h7 cases: {len(h7)}")
    for bk, cnt in sorted(bucket_counts.items()):
        print(f"  {bk}: {cnt}")
    print(f"  Total miss cases (B+C+D+E): {len(miss_cases)}")

    # ---------------------------------------------------------------
    # Phase 3: Forensic comparison
    # ---------------------------------------------------------------
    print(f"\n{ts()} PHASE 3: Forensic Comparison")
    print("-" * 70)

    # Pattern accumulators
    patterns = {
        "release_decade_match": {
            "gt_has_pattern_count": 0,
            "gt_has_pattern_not_top1": 0,
            "gt_has_pattern_not_any_top5": 0,
        },
        "release_year_proximity": {
            "gt_has_pattern_count": 0,
            "gt_has_pattern_not_top1": 0,
            "gt_has_pattern_not_any_top5": 0,
        },
        "artist_id_match": {
            "gt_has_pattern_count": 0,
            "gt_has_pattern_not_top1": 0,
            "gt_has_pattern_not_any_top5": 0,
            "adds_beyond_artist_name": 0,
        },
        "duration_bucket_match": {
            "gt_has_pattern_count": 0,
            "gt_has_pattern_not_top1": 0,
            "gt_has_pattern_not_any_top5": 0,
        },
        "isrc_prefix_match": {
            "gt_has_pattern_count": 0,
            "gt_has_pattern_not_top1": 0,
            "gt_has_pattern_not_any_top5": 0,
        },
    }

    for mc in miss_cases:
        i = mc["case_idx"]
        c = cases[i]
        gt = mc["gt"]
        played = c["music_turns"]
        wrong_top5 = mc["wrong_top5"]
        wrong_top1 = wrong_top5[0] if wrong_top5 else None

        gt_meta = track_meta.get(gt, {})
        gt_release = gt_meta.get("release_date", "")
        gt_year = extract_year(gt_release)
        gt_decade = extract_decade(gt_year)
        gt_artist_id = gt_meta.get("artist_id", "")
        gt_artist_name = gt_meta.get("artist_name", "")
        gt_duration = gt_meta.get("duration", 0)
        gt_isrc = gt_meta.get("isrc", "")
        gt_isrc_prefix = gt_isrc[:2] if len(gt_isrc) >= 2 else ""

        # History metadata for last 3 plays
        last3 = played[-3:] if len(played) >= 3 else played
        hist_years = []
        hist_decades = []
        hist_artist_ids = set()
        hist_artist_names = set()
        hist_durations = []
        hist_isrc_prefixes = set()

        for t in last3:
            m = track_meta.get(t, {})
            y = extract_year(m.get("release_date", ""))
            if y is not None:
                hist_years.append(y)
                d = extract_decade(y)
                if d:
                    hist_decades.append(d)
            aid = m.get("artist_id", "")
            if aid:
                hist_artist_ids.add(aid)
            aname = m.get("artist_name", "")
            if aname:
                hist_artist_names.add(aname.lower())
            dur = m.get("duration", 0)
            if dur > 0:
                hist_durations.append(dur)
            isrc = m.get("isrc", "")
            if len(isrc) >= 2:
                hist_isrc_prefixes.add(isrc[:2])

        # Also collect from ALL history for artist_id match
        all_hist_artist_ids = set()
        all_hist_artist_names = set()
        for t in played:
            m = track_meta.get(t, {})
            aid = m.get("artist_id", "")
            if aid:
                all_hist_artist_ids.add(aid)
            aname = m.get("artist_name", "")
            if aname:
                all_hist_artist_names.add(aname.lower())

        # Wrong top-1 and top-5 metadata
        w1_meta = track_meta.get(wrong_top1, {}) if wrong_top1 else {}
        w5_metas = [track_meta.get(tid, {}) for tid in wrong_top5]

        # ---- Pattern: release_decade_match ----
        if gt_decade and hist_decades:
            if gt_decade in hist_decades:
                patterns["release_decade_match"]["gt_has_pattern_count"] += 1
                # Check top-1
                w1_year = extract_year(w1_meta.get("release_date", ""))
                w1_decade = extract_decade(w1_year)
                if w1_decade not in hist_decades:
                    patterns["release_decade_match"]["gt_has_pattern_not_top1"] += 1
                # Check all top-5
                any_top5_match = False
                for wm in w5_metas:
                    wy = extract_year(wm.get("release_date", ""))
                    wd = extract_decade(wy)
                    if wd in hist_decades:
                        any_top5_match = True
                        break
                if not any_top5_match:
                    patterns["release_decade_match"]["gt_has_pattern_not_any_top5"] += 1

        # ---- Pattern: release_year_proximity ----
        if gt_year is not None and hist_years:
            mean_hist_year = sum(hist_years) / len(hist_years)
            if abs(gt_year - mean_hist_year) <= 3:
                patterns["release_year_proximity"]["gt_has_pattern_count"] += 1
                # Check top-1
                w1_year = extract_year(w1_meta.get("release_date", ""))
                w1_close = (w1_year is not None and abs(w1_year - mean_hist_year) <= 3)
                if not w1_close:
                    patterns["release_year_proximity"]["gt_has_pattern_not_top1"] += 1
                # Check all top-5
                any_top5_close = False
                for wm in w5_metas:
                    wy = extract_year(wm.get("release_date", ""))
                    if wy is not None and abs(wy - mean_hist_year) <= 3:
                        any_top5_close = True
                        break
                if not any_top5_close:
                    patterns["release_year_proximity"]["gt_has_pattern_not_any_top5"] += 1

        # ---- Pattern: artist_id_match ----
        if gt_artist_id and gt_artist_id in all_hist_artist_ids:
            patterns["artist_id_match"]["gt_has_pattern_count"] += 1
            # CRITICAL: adds_beyond_artist_name = artist_id matches but artist_name does NOT
            gt_name_lower = gt_artist_name.lower() if gt_artist_name else ""
            if gt_name_lower and gt_name_lower not in all_hist_artist_names:
                patterns["artist_id_match"]["adds_beyond_artist_name"] += 1
            # Check top-1
            w1_aid = w1_meta.get("artist_id", "")
            if w1_aid not in all_hist_artist_ids:
                patterns["artist_id_match"]["gt_has_pattern_not_top1"] += 1
            # Check all top-5
            any_top5_match = False
            for wm in w5_metas:
                waid = wm.get("artist_id", "")
                if waid in all_hist_artist_ids:
                    any_top5_match = True
                    break
            if not any_top5_match:
                patterns["artist_id_match"]["gt_has_pattern_not_any_top5"] += 1

        # ---- Pattern: duration_bucket_match ----
        if gt_duration > 0 and hist_durations:
            last_dur = hist_durations[-1]  # last played track duration
            if abs(gt_duration - last_dur) <= 30000:  # within 30 seconds
                patterns["duration_bucket_match"]["gt_has_pattern_count"] += 1
                # Check top-1
                w1_dur = w1_meta.get("duration", 0)
                w1_close = (w1_dur > 0 and abs(w1_dur - last_dur) <= 30000)
                if not w1_close:
                    patterns["duration_bucket_match"]["gt_has_pattern_not_top1"] += 1
                # Check all top-5
                any_top5_close = False
                for wm in w5_metas:
                    wd = wm.get("duration", 0)
                    if wd > 0 and abs(wd - last_dur) <= 30000:
                        any_top5_close = True
                        break
                if not any_top5_close:
                    patterns["duration_bucket_match"]["gt_has_pattern_not_any_top5"] += 1

        # ---- Pattern: isrc_prefix_match ----
        if gt_isrc_prefix and hist_isrc_prefixes:
            if gt_isrc_prefix in hist_isrc_prefixes:
                patterns["isrc_prefix_match"]["gt_has_pattern_count"] += 1
                # Check top-1
                w1_isrc = w1_meta.get("isrc", "")
                w1_prefix = w1_isrc[:2] if len(w1_isrc) >= 2 else ""
                if w1_prefix not in hist_isrc_prefixes:
                    patterns["isrc_prefix_match"]["gt_has_pattern_not_top1"] += 1
                # Check all top-5
                any_top5_match = False
                for wm in w5_metas:
                    wi = wm.get("isrc", "")
                    wp = wi[:2] if len(wi) >= 2 else ""
                    if wp in hist_isrc_prefixes:
                        any_top5_match = True
                        break
                if not any_top5_match:
                    patterns["isrc_prefix_match"]["gt_has_pattern_not_any_top5"] += 1

    # ---------------------------------------------------------------
    # Phase 4: Print and save forensic table
    # ---------------------------------------------------------------
    print(f"\n{ts()} PHASE 4: Forensic Results")
    print("=" * 70)

    n_miss = len(miss_cases)
    print(f"  Total miss cases (B+C+D+E): {n_miss}")

    feature_proposals = {
        "release_decade_match": "same_decade_last3",
        "release_year_proximity": "year_proximity_last3",
        "artist_id_match": "same_artist_id_any",
        "duration_bucket_match": "duration_close_last1",
        "isrc_prefix_match": "isrc_country_match",
    }

    risk_map = {
        "release_decade_match": "low",
        "release_year_proximity": "med",
        "artist_id_match": "low",
        "duration_bucket_match": "med",
        "isrc_prefix_match": "low",
    }

    print(f"\n{'field_pattern':<30} | {'coverage':>8} | {'gt_has':>6} | {'not_top1':>8} | {'not_any5':>8} | {'adds_new':>8} | {'feature':>25} | {'risk':>4}")
    print("-" * 120)

    forensic_table = []
    for pname, pdata in patterns.items():
        cov = coverage.get(pname.replace("_match", "").replace("_proximity", "").replace("release_decade", "release_date").replace("artist_id", "artist_id").replace("duration_bucket", "duration").replace("isrc_prefix", "isrc"), 0)
        # Map to correct coverage field
        if "decade" in pname or "year" in pname:
            cov = coverage.get("release_date", 0)
        elif "artist_id" in pname:
            cov = coverage.get("artist_id", 0)
        elif "duration" in pname:
            cov = coverage.get("duration", 0)
        elif "isrc" in pname:
            cov = coverage.get("isrc", 0)

        gt_has = pdata["gt_has_pattern_count"]
        not_t1 = pdata["gt_has_pattern_not_top1"]
        not_t5 = pdata["gt_has_pattern_not_any_top5"]
        adds_new = pdata.get("adds_beyond_artist_name")
        fp = feature_proposals.get(pname, "")
        risk = risk_map.get(pname, "med")

        adds_str = str(adds_new) if adds_new is not None else "n/a"
        print(f"{pname:<30} | {cov:>7.1f}% | {gt_has:>6} | {not_t1:>8} | {not_t5:>8} | {adds_str:>8} | {fp:>25} | {risk:>4}")

        entry = {
            "field_pattern": pname,
            "coverage": cov,
            "gt_has_pattern_count": gt_has,
            "gt_has_pattern_not_top1": not_t1,
            "gt_has_pattern_not_any_top5": not_t5,
            "adds_beyond_existing": adds_new if pname == "artist_id_match" else None,
            "feature_proposal": fp,
            "risk": risk,
        }
        forensic_table.append(entry)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  h7 reproduction: {h7_ndcg:.5f} (expected ~{expected_h7})")
    print(f"  Total h7 cases: {len(h7)}")
    print(f"  Miss cases: {n_miss}")
    for bk, cnt in sorted(bucket_counts.items()):
        print(f"    {bk}: {cnt}")

    print("\n  Top patterns by gt_has_pattern_not_any_top5 (most actionable):")
    sorted_patterns = sorted(forensic_table, key=lambda x: x["gt_has_pattern_not_any_top5"], reverse=True)
    for entry in sorted_patterns:
        pct = entry["gt_has_pattern_not_any_top5"] / n_miss * 100 if n_miss > 0 else 0
        adds = f", adds_beyond_name={entry['adds_beyond_existing']}" if entry['adds_beyond_existing'] is not None else ""
        print(f"    {entry['field_pattern']}: {entry['gt_has_pattern_not_any_top5']} cases ({pct:.1f}%){adds}")
        print(f"      -> Feature: {entry['feature_proposal']} (risk: {entry['risk']})")

    # Save results
    results = {
        "metadata_coverage": coverage,
        "h7_reproduction": {"h7": h7_ndcg, "pass": abs(h7_ndcg - expected_h7) <= 0.001},
        "miss_case_counts": {
            "total_h7": len(h7),
            **{bk: cnt for bk, cnt in bucket_counts.items()},
        },
        "forensic_table": forensic_table,
    }

    out_path = REPO / "exp" / "eval" / "expR49c_structural_forensics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
