#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R40: Second forensic pass on R39 baseline.

R38 forensics found album match as missing feature -> R39 added it -> nDCG +0.006 blind.
Now repeat: recompute R39 LR misses, analyze new structural patterns in miss cases.
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

# R39 feature set: 29 base + 2 R21 + 5 album = 36 total
# (FEATURE_NAMES_V2 has 27, then +2 R21 from FEAT_BASE, +5 album)
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


def load_full_track_metadata():
    """Load full track metadata for forensic analysis."""
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

        tag_list_raw = item.get("tag_list", [])
        if isinstance(tag_list_raw, list):
            tag_list = [str(t).lower().strip() for t in tag_list_raw if str(t).strip()]
        else:
            tag_list = []

        release_date = str(item.get("release_date", "")) if item.get("release_date") else ""
        popularity = float(item.get("popularity", 0.0)) if item.get("popularity") is not None else 0.0

        meta[tid] = {
            "track_name": track_name,
            "artist_name": artist_name,
            "album_name": album_name,
            "album_id": album_id,
            "artist_id": artist_id,
            "tag_list": tag_list,
            "tag_set": set(tag_list),
            "release_date": release_date,
            "popularity": popularity,
        }
    return meta


def extract_year(release_date: str) -> int | None:
    """Extract year from release_date string like '2006-12-06'."""
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


def main():
    t0 = time.time()
    print(f"{ts()} R40: Second Forensic Pass on R39 Baseline")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data (same as R39a)
    # ---------------------------------------------------------------
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
    track_album = load_track_albums()

    print(f"{ts()} Loading full track metadata...")
    track_meta = load_full_track_metadata()

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
    # Phase 1: Build features EXACTLY as R39a (+album config)
    # ---------------------------------------------------------------
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

            # Base 29 features (EXACT copy from R39a)
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

            # Album features (EXACT copy from R39a)
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
    # Phase 1 cont: CV5 LambdaRank, collect per-case LR scores
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Phase 2: Identify miss cases and run forensic analysis
    # ---------------------------------------------------------------
    print(f"\n{ts()} Phase 2: Forensic analysis of R39 miss cases")

    # Compute tag frequencies across full catalog for rare-tag detection
    print(f"{ts()} Computing catalog-wide tag frequencies...")
    tag_freq: Counter[str] = Counter()
    for tid, meta in track_meta.items():
        for tag in meta["tag_list"]:
            tag_freq[tag] += 1
    rare_tag_threshold = 500
    n_rare = sum(1 for f in tag_freq.values() if f < rare_tag_threshold)
    print(f"  {len(tag_freq)} unique tags, {n_rare} rare (freq < {rare_tag_threshold})")

    # Collect miss cases: GT in pool@300 but R39 LR ranks it > 20
    miss_cases = []
    for i in h7:
        c = cases[i]
        gt = c["gt"]
        pool = pools[i]
        if gt not in pool:
            continue
        gt_pool_idx = pool.index(gt)
        ps = int(sizes[i])
        sc = lr_scores[i, :ps]
        lr_ranked = np.argsort(-sc)
        gt_lr_rank_arr = np.where(lr_ranked == gt_pool_idx)[0]
        if len(gt_lr_rank_arr) == 0:
            continue
        gt_rank = int(gt_lr_rank_arr[0]) + 1  # 1-indexed
        if gt_rank <= 20:
            continue
        # This is a miss case
        wrong_top5_pool_idx = [int(lr_ranked[j]) for j in range(min(5, ps))
                               if int(lr_ranked[j]) != gt_pool_idx][:5]
        wrong_top1_tid = pool[wrong_top5_pool_idx[0]] if wrong_top5_pool_idx else None
        wrong_top5_tids = [pool[idx] for idx in wrong_top5_pool_idx]

        miss_cases.append({
            "case_idx": i,
            "gt": gt,
            "gt_rank": gt_rank,
            "gt_pool_idx": gt_pool_idx,
            "wrong_top1": wrong_top1_tid,
            "wrong_top5": wrong_top5_tids,
            "pool": pool,
            "pool_size": ps,
        })

    n_miss = len(miss_cases)
    print(f"  {len(h7)} hist_7 cases, {n_miss} LR miss-in-pool cases (GT in pool but rank > 20)")

    # ---------------------------------------------------------------
    # Phase 2: Analyze each miss case on 11 dimensions
    # ---------------------------------------------------------------
    pattern_counts = Counter()
    pattern_details: dict[str, list] = {}
    examples = []

    for mc in miss_cases:
        i = mc["case_idx"]
        c = cases[i]
        gt = mc["gt"]
        gt_rank = mc["gt_rank"]
        played = c["music_turns"]
        query = c["user_query"]
        query_lower = query.lower()
        wrong_top1 = mc["wrong_top1"]
        wrong_top5 = mc["wrong_top5"]
        pool = mc["pool"]
        src_lists = src_lists_all[i]

        # GT metadata
        gt_meta = track_meta.get(gt, {})
        gt_artist = gt_meta.get("artist_name", "")
        gt_artist_lower = gt_artist.lower()
        gt_track_name = gt_meta.get("track_name", "")
        gt_track_lower = gt_track_name.lower()
        gt_tags = gt_meta.get("tag_set", set())
        gt_album = gt_meta.get("album_name", "")
        gt_album_id = gt_meta.get("album_id", "")
        gt_artist_id = gt_meta.get("artist_id", "")
        gt_release = gt_meta.get("release_date", "")
        gt_year = extract_year(gt_release)
        gt_decade = extract_decade(gt_year)
        gt_popularity = gt_meta.get("popularity", 0.0)

        # Wrong top-1 metadata
        w1_meta = track_meta.get(wrong_top1, {}) if wrong_top1 else {}
        w1_artist = w1_meta.get("artist_name", "")
        w1_tags = w1_meta.get("tag_set", set())
        # History metadata
        played_artist_ids = {track_meta.get(t, {}).get("artist_id", "") for t in played} - {""}
        last3_tags: list[set[str]] = []
        hist_years = []
        hist_decades = []
        hist_popularity = []
        for t in played:
            m = track_meta.get(t, {})
            last3_tags.append(m.get("tag_set", set()))
            y = extract_year(m.get("release_date", ""))
            if y is not None:
                hist_years.append(y)
                d = extract_decade(y)
                if d:
                    hist_decades.append(d)
            hist_popularity.append(m.get("popularity", 0.0))

        # Recent (last 3) tags union
        recent_tags = set()
        for t in played[-3:]:
            recent_tags |= track_meta.get(t, {}).get("tag_set", set())

        # ---------------------------------------------------------------
        # Dimension 1: artist_in_query
        # ---------------------------------------------------------------
        dim_artist_in_query = False
        if gt_artist_lower and len(gt_artist_lower) > 2:
            # Check each artist name (could be multi-artist)
            for a_part in gt_artist_lower.split(","):
                a_part = a_part.strip()
                if a_part and len(a_part) > 2 and a_part in query_lower:
                    dim_artist_in_query = True
                    break

        # ---------------------------------------------------------------
        # Dimension 2: title_in_query
        # ---------------------------------------------------------------
        dim_title_in_query = False
        if gt_track_lower and len(gt_track_lower) > 3 and gt_track_lower in query_lower:
            dim_title_in_query = True
        elif gt_track_lower and len(gt_track_lower) > 3:
            # Check significant substrings (>3 chars, first word or parenthetical)
            parts = re.split(r"[(\[/\-,]", gt_track_lower)
            for part in parts:
                part = part.strip().rstrip(")].").strip()
                if len(part) > 3 and part in query_lower:
                    dim_title_in_query = True
                    break

        # ---------------------------------------------------------------
        # Dimension 3: album_artist_mismatch (compilation albums)
        # ---------------------------------------------------------------
        dim_album_artist_mismatch = False
        if gt_album_id:
            # Check if any track in pool has same album but different artist
            for tid2 in pool[:50]:
                m2 = track_meta.get(tid2, {})
                if (m2.get("album_id", "") == gt_album_id
                        and m2.get("artist_id", "") != gt_artist_id
                        and m2.get("artist_id", "")):
                    dim_album_artist_mismatch = True
                    break

        # ---------------------------------------------------------------
        # Dimension 4: release_year_decade match
        # ---------------------------------------------------------------
        dim_decade_match = False
        if gt_decade and hist_decades:
            decade_counts = Counter(hist_decades)
            dominant_decade = decade_counts.most_common(1)[0][0]
            dim_decade_match = (gt_decade == dominant_decade)
        # Also check if wrong top-1 doesn't match
        w1_year = extract_year(w1_meta.get("release_date", ""))
        w1_decade = extract_decade(w1_year)
        dim_decade_gt_better = (dim_decade_match and
                                w1_decade is not None and
                                gt_decade is not None and
                                w1_decade != gt_decade)

        # ---------------------------------------------------------------
        # Dimension 5: rare_tag_overlap
        # ---------------------------------------------------------------
        gt_rare_tags = {t for t in gt_tags if tag_freq.get(t, 0) < rare_tag_threshold}
        recent_rare_tags = {t for t in recent_tags if tag_freq.get(t, 0) < rare_tag_threshold}
        gt_rare_overlap = gt_rare_tags & recent_rare_tags

        w1_rare_tags = {t for t in w1_tags if tag_freq.get(t, 0) < rare_tag_threshold}
        w1_rare_overlap = w1_rare_tags & recent_rare_tags

        dim_rare_tag = len(gt_rare_overlap) > 0 and len(gt_rare_overlap) > len(w1_rare_overlap)

        # ---------------------------------------------------------------
        # Dimension 6: tag_motif (repeated tag pattern across last 3)
        # ---------------------------------------------------------------
        dim_tag_motif = False
        if len(played) >= 3:
            # Tags that appear in ALL of last 3 tracks
            last3_tag_sets = [track_meta.get(t, {}).get("tag_set", set()) for t in played[-3:]]
            if all(s for s in last3_tag_sets):
                motif_tags = last3_tag_sets[0]
                for s in last3_tag_sets[1:]:
                    motif_tags = motif_tags & s
                # Filter to meaningful tags (not too common)
                motif_tags = {t for t in motif_tags if tag_freq.get(t, 0) < 5000}
                if motif_tags:
                    gt_motif_overlap = gt_tags & motif_tags
                    w1_motif_overlap = w1_tags & motif_tags
                    if len(gt_motif_overlap) > len(w1_motif_overlap):
                        dim_tag_motif = True

        # ---------------------------------------------------------------
        # Dimension 7: r21_top20_demoted
        # ---------------------------------------------------------------
        gt_r21_rank = r21_source[i].index(gt) + 1 if gt in r21_source[i][:300] else -1
        dim_r21_demoted = gt_r21_rank > 0 and gt_r21_rank <= 20

        # ---------------------------------------------------------------
        # Dimension 8: bm25_top20_demoted
        # ---------------------------------------------------------------
        gt_bm25a_rank = payload["src_a"][i].index(gt) + 1 if gt in payload["src_a"][i][:300] else -1
        gt_bm25c_rank = payload["src_c"][i].index(gt) + 1 if gt in payload["src_c"][i][:300] else -1
        dim_bm25_demoted = ((gt_bm25a_rank > 0 and gt_bm25a_rank <= 20) or
                            (gt_bm25c_rank > 0 and gt_bm25c_rank <= 20))

        # ---------------------------------------------------------------
        # Dimension 9: deeper_album_cut
        # ---------------------------------------------------------------
        dim_deeper_cut = False
        if gt_artist_id and gt_artist_id in played_artist_ids:
            # GT is same artist as history but less popular
            same_artist_hist_pops = []
            for t in played:
                if track_meta.get(t, {}).get("artist_id", "") == gt_artist_id:
                    same_artist_hist_pops.append(track_meta.get(t, {}).get("popularity", 0.0))
            if same_artist_hist_pops:
                avg_hist_pop = sum(same_artist_hist_pops) / len(same_artist_hist_pops)
                if gt_popularity < avg_hist_pop * 0.5:  # GT is <50% as popular
                    dim_deeper_cut = True

        # ---------------------------------------------------------------
        # Dimension 10: negative_constraint_violated
        # ---------------------------------------------------------------
        dim_negative_constraint = False
        # Check for negative expressions in user query
        neg_patterns = [
            r"(?:not|don't|no|never|avoid|skip|without|doesn't|isn't|stop|hate)\s+(?:the\s+)?(\w+)",
            r"(?:tired of|enough of|sick of|bored of)\s+(\w+)",
        ]
        neg_entities = set()
        for pat in neg_patterns:
            for m in re.finditer(pat, query_lower):
                neg_entities.add(m.group(1).lower())
        if neg_entities:
            # Check if wrong top candidates contain negated entity
            for w_tid in wrong_top5[:3]:
                w_meta = track_meta.get(w_tid, {})
                w_artist_lower = w_meta.get("artist_name", "").lower()
                w_track_lower = w_meta.get("track_name", "").lower()
                for ne in neg_entities:
                    if ne in w_artist_lower or ne in w_track_lower:
                        dim_negative_constraint = True
                        break

        # ---------------------------------------------------------------
        # Dimension 11: played_exclusion_edge
        # ---------------------------------------------------------------
        dim_played_edge = False
        played_set = set(played)
        # Check if any of the wrong top-5 appear in played set
        for w_tid in wrong_top5[:3]:
            if w_tid in played_set:
                dim_played_edge = True
                break

        # ---------------------------------------------------------------
        # Collect results
        # ---------------------------------------------------------------
        patterns_found = []
        pattern_map = {
            "artist_in_query": dim_artist_in_query,
            "title_in_query": dim_title_in_query,
            "album_artist_mismatch": dim_album_artist_mismatch,
            "decade_match_gt_better": dim_decade_gt_better,
            "rare_tag_overlap": dim_rare_tag,
            "tag_motif": dim_tag_motif,
            "r21_top20_demoted": dim_r21_demoted,
            "bm25_top20_demoted": dim_bm25_demoted,
            "deeper_album_cut": dim_deeper_cut,
            "negative_constraint": dim_negative_constraint,
            "played_exclusion_edge": dim_played_edge,
        }

        for name, val in pattern_map.items():
            if val:
                patterns_found.append(name)
                pattern_counts[name] += 1
                if name not in pattern_details:
                    pattern_details[name] = []
                if len(pattern_details[name]) < 5:
                    pattern_details[name].append({
                        "case_idx": i,
                        "gt": gt,
                        "gt_name": gt_track_name,
                        "gt_artist": gt_artist,
                        "gt_rank": gt_rank,
                        "query": query[:150],
                    })

        if not patterns_found:
            pattern_counts["no_clear_pattern"] += 1
            patterns_found.append("no_clear_pattern")

        if len(examples) < 30:
            examples.append({
                "case_idx": i,
                "gt": gt,
                "gt_name": gt_track_name,
                "gt_artist": gt_artist,
                "gt_rank": gt_rank,
                "gt_album": gt_album,
                "gt_decade": gt_decade,
                "gt_popularity": round(gt_popularity, 1),
                "gt_r21_rank": gt_r21_rank,
                "gt_bm25a_rank": gt_bm25a_rank,
                "gt_bm25c_rank": gt_bm25c_rank,
                "w1_artist": w1_artist,
                "w1_decade": w1_decade,
                "patterns": patterns_found,
                "query": query[:150],
            })

    # ---------------------------------------------------------------
    # Phase 3: Output ranked table
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"R40 FORENSICS — {n_miss} R39 LR miss-in-pool cases (hist_7)")
    print(sep)

    # Assess each pattern for actionability
    pattern_info = {
        "artist_in_query": {
            "observable": True,
            "candidate_feature": "artist_name_in_current_query",
            "risk": "low",
            "desc": "GT artist name mentioned in user query but LR missed",
        },
        "title_in_query": {
            "observable": True,
            "candidate_feature": "track_title_in_current_query",
            "risk": "low",
            "desc": "GT track title mentioned in query but LR missed",
        },
        "album_artist_mismatch": {
            "observable": True,
            "candidate_feature": "album_artist_differs_from_track_artist",
            "risk": "med",
            "desc": "GT album_artist != track_artist (compilations)",
        },
        "decade_match_gt_better": {
            "observable": True,
            "candidate_feature": "release_decade_matches_history_mode",
            "risk": "med",
            "desc": "GT decade matches history pattern but wrong top-1 doesn't",
        },
        "rare_tag_overlap": {
            "observable": True,
            "candidate_feature": "rare_tag_intersection_count",
            "risk": "low",
            "desc": "GT shares rare tags with recent history, wrong candidate doesn't",
        },
        "tag_motif": {
            "observable": True,
            "candidate_feature": "tag_motif_match_last3",
            "risk": "low",
            "desc": "Repeated tag across last 3 tracks, GT matches, wrong doesn't",
        },
        "r21_top20_demoted": {
            "observable": True,
            "candidate_feature": "r21_rank_bucket",
            "risk": "low",
            "desc": "GT was R21 top-20 but LR demoted below 20",
        },
        "bm25_top20_demoted": {
            "observable": True,
            "candidate_feature": "bm25_rank_bucket",
            "risk": "low",
            "desc": "GT was BM25-A or BM25-C top-20 but LR demoted below 20",
        },
        "deeper_album_cut": {
            "observable": True,
            "candidate_feature": "track_popularity_vs_artist_hist_avg",
            "risk": "med",
            "desc": "GT is same artist but much less popular (deeper catalog)",
        },
        "negative_constraint": {
            "observable": True,
            "candidate_feature": "negative_mention_overlap",
            "risk": "high",
            "desc": "User expressed negative preference, wrong candidates match negated entity",
        },
        "played_exclusion_edge": {
            "observable": True,
            "candidate_feature": "already_played_improved",
            "risk": "low",
            "desc": "Wrong top candidates appear in played set (edge case of already_played)",
        },
        "no_clear_pattern": {
            "observable": False,
            "candidate_feature": "n/a",
            "risk": "n/a",
            "desc": "No clear structural pattern found",
        },
    }

    # Estimate h7 case impact: miss_count is how many cases could potentially flip to hits
    # Expected h7 cases that could benefit = miss_count (already filtered to h7)
    print(f"\n{'feature_pattern':<30} | {'miss_count':>10} | {'pct_misses':>10} | {'observable':>11} | {'candidate_feature':<40} | {'risk':>5} | {'exp_h7':>7}")
    print("-" * 130)
    for name, count in pattern_counts.most_common():
        info = pattern_info.get(name, {"observable": False, "candidate_feature": "n/a", "risk": "n/a"})
        pct = count / n_miss * 100
        obs = "yes" if info["observable"] else "no"
        cf = info["candidate_feature"]
        risk = info["risk"]
        print(f"{name:<30} | {count:>10} | {pct:>9.1f}% | {obs:>11} | {cf:<40} | {risk:>5} | {count:>7}")

    # Show detailed examples for top patterns
    print(f"\n{sep}")
    print("DETAILED EXAMPLES BY PATTERN")
    print(sep)
    for name, count in pattern_counts.most_common(8):
        if name == "no_clear_pattern":
            continue
        if name not in pattern_details:
            continue
        print(f"\n  [{name}] ({count} cases)")
        for ex in pattern_details[name][:3]:
            print(f"    Case {ex['case_idx']}: GT rank {ex['gt_rank']} — "
                  f"{ex['gt_name']} by {ex['gt_artist']}")
            print(f"      Query: {ex['query']}")

    # Show overall miss examples
    print(f"\n{sep}")
    print("FIRST 20 MISS EXAMPLES")
    print(sep)
    for ex in examples[:20]:
        print(f"\n  Case {ex['case_idx']} — GT rank {ex['gt_rank']}")
        print(f"    GT: {ex['gt_name']} by {ex['gt_artist']} (album: {ex['gt_album']}, "
              f"decade: {ex['gt_decade']}, pop: {ex['gt_popularity']})")
        print(f"    R21={ex['gt_r21_rank']} BM25a={ex['gt_bm25a_rank']} BM25c={ex['gt_bm25c_rank']}")
        print(f"    Wrong#1: {ex['w1_artist']} (decade: {ex['w1_decade']})")
        print(f"    Patterns: {ex['patterns']}")
        print(f"    Query: {ex['query']}")

    # Summary recommendation
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    actionable = [(name, count) for name, count in pattern_counts.most_common()
                  if name != "no_clear_pattern" and count >= 5]
    print(f"\n  Total miss cases: {n_miss}")
    print("  Actionable patterns (>= 5 cases):")
    for name, count in actionable:
        info = pattern_info.get(name, {})
        print(f"    {name}: {count} ({count/n_miss*100:.1f}%) — {info.get('desc', '')}")
    print("\n  Recommended next feature to add (highest count, observable, low risk):")
    for name, count in actionable:
        info = pattern_info.get(name, {})
        if info.get("observable") and info.get("risk") in ("low", "med"):
            print(f"    -> {name}: {count} cases ({count/n_miss*100:.1f}%)")
            print(f"       Feature: {info.get('candidate_feature', '')}")
            print(f"       Description: {info.get('desc', '')}")
            break

    # Save results
    out_path = REPO / "exp" / "eval" / "expR40_forensics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "n_h7": len(h7),
        "n_miss": n_miss,
        "r39_h7_ndcg": h7_ndcg,
        "r39_cv5_ndcg": cv5,
        "pool_hit": pool_hit,
        "pattern_counts": dict(pattern_counts.most_common()),
        "pattern_percentages": {name: round(count / n_miss * 100, 1)
                                for name, count in pattern_counts.most_common()},
        "pattern_info": {name: {k: v for k, v in info.items() if k != "desc"}
                         for name, info in pattern_info.items()
                         if name in pattern_counts},
        "examples": examples,
        "pattern_details": {k: v for k, v in pattern_details.items()},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
