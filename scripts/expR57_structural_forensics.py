#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R57 structural forensics — refresh on R39+R54 baseline.

Pure forensics. No feature implementation, no LR retraining beyond the
already-CV5 LR that defines the baseline.

Builds the R39+R54 baseline (same as R55_post_refresh_decomp / R56), gets
per-case LR ranks, then for each candidate structural-metadata pattern
counts:

  gt_has_pattern                  : # DEMOTED cases where GT has the pattern
  gt_has_pattern_not_top1         : # DEMOTED cases where GT has the pattern
                                    AND the LR top-1 (a wrong track) does NOT
  gt_has_pattern_not_any_top5     : same, but checked against all of LR top-5

Patterns evaluated (all observable at blind inference, all non-textual):

  release_year_proximity_last1    : |gt year - last_played year| <= 3
  release_year_proximity_last3    : |gt year - any of last 3 played years| <= 3
  release_decade_match_last1      : gt decade == last_played decade
  release_decade_match_history    : gt decade in history decades
  duration_close_last1            : |gt dur - last_played dur| <= 30s
  duration_close_last3            : same, last 3
  isrc_country_match              : ISRC country (chars 0-1) in history
  isrc_registrant_match           : ISRC registrant (chars 2-4) in history
  artist_id_match_history         : gt artist_id in history artist_ids (adds
                                    beyond artist-name? counted separately)
  album_artist_id_match_history   : gt's album_artist_id in history

Reporting filters (per user spec):
  >= 30 cases where gt_has_pattern_not_top1
  coverage >= 90%
  same_artist / diff_artist split shown for context
  ranking_only vs pool_admission flagged per pattern

Output:
  exp/eval/expR57_structural_forensics.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expS2_lambdarank import build_als  # noqa: E402
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds  # noqa: E402
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats  # noqa: E402
from scripts.tune_postrank_v23 import tokens  # noqa: E402

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
OUT = REPO / "exp" / "eval" / "expR57_structural_forensics.json"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM + FEAT_R54

# Reporting filters
MIN_CASES = 30
MIN_COVERAGE = 0.90


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def first_or_none(v):
    if isinstance(v, list):
        return v[0] if v else None
    if v in (None, ""):
        return None
    return v


def parse_year(date_str):
    if not date_str:
        return None
    s = str(date_str)
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def load_track_metadata():
    """Return dict track_id -> {artist_id, album_id, album_artist_id, release_year,
    release_decade, duration_ms, isrc, isrc_country, isrc_registrant, album_name}.
    """
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    coverage = Counter()
    total = 0
    for item in ds:
        tid = str(item["track_id"])
        total += 1

        def _scalar(key):
            v = item.get(key, None)
            if isinstance(v, list):
                return v[0] if v else None
            if v == "":
                return None
            return v

        artist_id = _scalar("artist_id")
        album_id = _scalar("album_id")
        album_name = _scalar("album_name")
        release_date = _scalar("release_date")
        # HF schema: `duration` is an int in ms, `ISRC` is uppercase list
        duration_ms = _scalar("duration")
        isrc = _scalar("ISRC")
        # No album_artist_id column in this dataset; drop that pattern.
        album_artist_id = None

        release_year = parse_year(release_date)
        release_decade = (release_year // 10) * 10 if release_year is not None else None

        isrc_country = None
        isrc_registrant = None
        if isrc:
            s = str(isrc).strip().upper()
            if len(s) >= 5:
                isrc_country = s[:2]
                isrc_registrant = s[2:5]

        if artist_id:
            coverage["artist_id"] += 1
        if album_id:
            coverage["album_id"] += 1
        if album_artist_id:
            coverage["album_artist_id"] += 1
        if release_year is not None:
            coverage["release_year"] += 1
        if duration_ms is not None:
            coverage["duration_ms"] += 1
        if isrc:
            coverage["isrc"] += 1
        if isrc_country:
            coverage["isrc_country"] += 1
        if isrc_registrant:
            coverage["isrc_registrant"] += 1

        meta[tid] = {
            "artist_id": str(artist_id) if artist_id else None,
            "album_id": str(album_id) if album_id else None,
            "album_artist_id": str(album_artist_id) if album_artist_id else None,
            "release_year": release_year,
            "release_decade": release_decade,
            "duration_ms": float(duration_ms) if duration_ms is not None else None,
            "isrc": str(isrc) if isrc else None,
            "isrc_country": isrc_country,
            "isrc_registrant": isrc_registrant,
        }

    print(f"  {total} tracks, coverage:")
    for k, c in coverage.most_common():
        print(f"    {k:>20s}: {c / total:.4f}")
    return meta, total, coverage


def build_baseline_ranks():
    """Reproduce baseline CV5 LR ranks + pools. Same logic as R56."""
    print(f"{ts()} Loading R12 + R21 OOF + R54 Phase 2 OOF...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_raw = r54_data["lists"]
    r54_source = []
    r54_scores = []
    for cl in r54_raw:
        tids = [t for t, _ in cl]
        sm = {t: float(s) for t, s in cl}
        r54_source.append(tids)
        r54_scores.append(sm)
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    from scripts.expR42_error_decomposition import load_track_albums as load_albums_for_feat
    track_album = load_albums_for_feat()

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
    n_feat_r39 = n_feat_base + len(FEAT_ALBUM)
    X = np.zeros((n, POOL_K, len(ALL_FEAT)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]
    src_union_has_gt = np.zeros(n, dtype=bool)
    print(f"{ts()} Building features...")
    t_feat = time.time()
    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "R54": r54_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools[i] = pool
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])
        if c["gt"] in (set(payload["src_a"][i]) | set(payload["src_b"][i]) | set(payload["src_c"][i])
                       | set(payload["src_d"][i]) | set(payload["src_f"][i])
                       | set(als_source[i]) | set(r21_source[i]) | set(r54_source[i])):
            src_union_has_gt[i] = True

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
        pool_artists_all = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists_all if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_source[i][:300])}
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
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sn, {}))
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
            pool_album_count = sum(1 for t2 in pool[:POOL_K] if track_album.get(t2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)
            row[n_feat_r39 + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 1] = 1.0 if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 2] = r54_scores[i].get(tid, 0.0)
        if (i + 1) % 2000 == 0:
            print(f"  features {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)

    print(f"{ts()} CV5 LambdaRank for baseline ranks...")
    case_lr_rank = np.full(n, -1, dtype=np.int64)
    top5_per_case: list[list[str]] = [[] for _ in range(n)]
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
                            group=g_tr, feature_name=list(ALL_FEAT))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            offset += s
            ranked = np.argsort(-sc)
            top5_per_case[idx] = [pools[idx][int(j)] for j in ranked[:5]]
            if gt_idx[idx] < 0:
                continue
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1

    return {
        "cases": cases,
        "ta": ta,
        "pools": pools,
        "gt_idx": gt_idx,
        "src_union_has_gt": src_union_has_gt,
        "case_lr_rank": case_lr_rank,
        "top5_per_case": top5_per_case,
    }


def classify_buckets(baseline):
    cases = baseline["cases"]
    gt_idx = baseline["gt_idx"]
    case_lr_rank = baseline["case_lr_rank"]
    src_union_has_gt = baseline["src_union_has_gt"]
    n = len(cases)
    bucket = []
    for i in range(n):
        if gt_idx[i] >= 0 and case_lr_rank[i] >= 1 and case_lr_rank[i] <= 20:
            bucket.append("HIT")
        elif gt_idx[i] >= 0:
            bucket.append("DEMOTED")
        elif src_union_has_gt[i]:
            bucket.append("POOL_MISS")
        else:
            bucket.append("UNREACHABLE")
    return bucket


PATTERN_DEFS = {
    "release_year_proximity_last1": {
        "field": "release_year", "scope": "ranking_only",
    },
    "release_year_proximity_last3": {
        "field": "release_year", "scope": "ranking_only",
    },
    "release_decade_match_last1": {
        "field": "release_decade", "scope": "ranking_only",
    },
    "release_decade_match_history": {
        "field": "release_decade", "scope": "ranking_only",
    },
    "duration_close_last1": {
        "field": "duration_ms", "scope": "ranking_only",
    },
    "duration_close_last3": {
        "field": "duration_ms", "scope": "ranking_only",
    },
    "isrc_country_match": {
        "field": "isrc_country", "scope": "ranking_only",
    },
    "isrc_registrant_match": {
        "field": "isrc_registrant", "scope": "ranking_only",
    },
    "artist_id_match_history": {
        "field": "artist_id",
        "scope": "ranking_only",
        "note": "Largely overlaps with track_artist_name match (already a feature). Adds-beyond reported separately.",
    },
    # album_artist_id removed: column does not exist in HF metadata schema.
}


def check_pattern(name, tid, meta, played_meta_list):
    """Return True if track `tid` matches pattern relative to history (list of dicts)."""
    m = meta.get(tid, {})
    if name == "release_year_proximity_last1":
        gy = m.get("release_year")
        if gy is None or not played_meta_list:
            return False
        ly = played_meta_list[-1].get("release_year")
        return ly is not None and abs(gy - ly) <= 3
    if name == "release_year_proximity_last3":
        gy = m.get("release_year")
        if gy is None:
            return False
        for h in played_meta_list[-3:]:
            ly = h.get("release_year")
            if ly is not None and abs(gy - ly) <= 3:
                return True
        return False
    if name == "release_decade_match_last1":
        gd = m.get("release_decade")
        if gd is None or not played_meta_list:
            return False
        return gd == played_meta_list[-1].get("release_decade")
    if name == "release_decade_match_history":
        gd = m.get("release_decade")
        if gd is None:
            return False
        return any(gd == h.get("release_decade") for h in played_meta_list)
    if name == "duration_close_last1":
        gd = m.get("duration_ms")
        if gd is None or not played_meta_list:
            return False
        ld = played_meta_list[-1].get("duration_ms")
        return ld is not None and abs(gd - ld) <= 30_000  # 30s tolerance
    if name == "duration_close_last3":
        gd = m.get("duration_ms")
        if gd is None:
            return False
        for h in played_meta_list[-3:]:
            ld = h.get("duration_ms")
            if ld is not None and abs(gd - ld) <= 30_000:
                return True
        return False
    if name == "isrc_country_match":
        gc = m.get("isrc_country")
        if not gc:
            return False
        return any(gc == h.get("isrc_country") for h in played_meta_list if h.get("isrc_country"))
    if name == "isrc_registrant_match":
        gr = m.get("isrc_registrant")
        if not gr:
            return False
        return any(gr == h.get("isrc_registrant") for h in played_meta_list if h.get("isrc_registrant"))
    if name == "artist_id_match_history":
        ga = m.get("artist_id")
        if not ga:
            return False
        return any(ga == h.get("artist_id") for h in played_meta_list if h.get("artist_id"))
    return False


def main():
    t0 = time.time()
    print("R57 structural forensics (refresh on R39+R54 baseline)")
    print("=" * 70)

    print(f"\n{ts()} Loading track metadata...")
    track_meta, total_tracks, field_cov = load_track_metadata()

    baseline = build_baseline_ranks()
    cases = baseline["cases"]
    ta = baseline["ta"]
    case_lr_rank = baseline["case_lr_rank"]
    top5_per_case = baseline["top5_per_case"]
    n = len(cases)

    buckets = classify_buckets(baseline)
    bucket_counts = Counter(buckets)
    print(f"\n  Bucket counts (R39+R54 baseline): {dict(bucket_counts)}")

    demoted_idxs = [i for i in range(n) if buckets[i] == "DEMOTED"]
    print(f"  DEMOTED cases: {len(demoted_idxs)} (target for pattern check)")

    # Coverage at catalog level (already printed inside load_track_metadata).
    # But the user wants coverage on the cases we're actually evaluating (the
    # GT tracks). Re-check on the GT-set of DEMOTED cases.
    demoted_gt_field_cov = Counter()
    for i in demoted_idxs:
        gt_m = track_meta.get(cases[i]["gt"], {})
        for f in ("release_year", "release_decade", "duration_ms", "isrc_country",
                  "isrc_registrant", "artist_id"):
            if gt_m.get(f) is not None:
                demoted_gt_field_cov[f] += 1

    print(f"\n  Field coverage on DEMOTED GT tracks (n={len(demoted_idxs)}):")
    for f, c in sorted(demoted_gt_field_cov.items(), key=lambda kv: -kv[1]):
        print(f"    {f:>20s}: {c}/{len(demoted_idxs)}  ({c / len(demoted_idxs):.2%})")

    # For each DEMOTED case, precompute history's per-turn metadata
    history_meta_list = []
    for i in range(n):
        history_meta_list.append([track_meta.get(t, {}) for t in cases[i]["music_turns"]])

    # Evaluate each pattern on DEMOTED bucket
    print(f"\n{ts()} Evaluating patterns on {len(demoted_idxs)} DEMOTED cases...")
    pattern_results = {}
    for pat_name, pat_def in PATTERN_DEFS.items():
        gt_has_pattern = 0
        gt_has_pattern_not_top1 = 0
        gt_has_pattern_not_any_top5 = 0
        same_artist_count = 0
        diff_artist_count = 0
        same_art_has_pat = 0
        diff_art_has_pat = 0

        for i in demoted_idxs:
            gt = cases[i]["gt"]
            played = cases[i]["music_turns"]
            played_meta = history_meta_list[i]
            gt_pat = check_pattern(pat_name, gt, track_meta, played_meta)
            # same/diff artist (history) — purely for reporting
            gt_artist = ta.get(gt, "")
            played_artists = {ta.get(t, "") for t in played} - {""}
            is_same = bool(gt_artist) and gt_artist in played_artists
            if is_same:
                same_artist_count += 1
            else:
                diff_artist_count += 1
            if gt_pat:
                gt_has_pattern += 1
                if is_same:
                    same_art_has_pat += 1
                else:
                    diff_art_has_pat += 1
                # Check LR top-1: does it ALSO have the pattern?
                top5 = top5_per_case[i]
                if not top5:
                    continue
                top1_pat = check_pattern(pat_name, top5[0], track_meta, played_meta)
                if not top1_pat:
                    gt_has_pattern_not_top1 += 1
                # Any of top-5?
                any_top5_pat = any(check_pattern(pat_name, t, track_meta, played_meta)
                                   for t in top5)
                if not any_top5_pat:
                    gt_has_pattern_not_any_top5 += 1

        # Coverage of the field needed for the pattern (catalog level)
        # Derived fields (release_decade) inherit from their source (release_year)
        field = pat_def["field"]
        cov_field = "release_year" if field == "release_decade" else field
        coverage = field_cov.get(cov_field, 0) / total_tracks if total_tracks else 0

        # Adds-beyond: only meaningful for artist_id (where artist_name match is already a feature)
        adds_beyond = None
        if pat_name == "artist_id_match_history":
            # Among gt_has_pattern cases, how many have GT artist_id in history
            # BUT GT artist_name NOT in history? Those are the genuine net-new.
            ab = 0
            for i in demoted_idxs:
                gt = cases[i]["gt"]
                ga_id = track_meta.get(gt, {}).get("artist_id")
                if not ga_id:
                    continue
                ga_name = ta.get(gt, "")
                hist_ids = {track_meta.get(t, {}).get("artist_id") for t in cases[i]["music_turns"]} - {None}
                hist_names = {ta.get(t, "") for t in cases[i]["music_turns"]} - {""}
                if ga_id in hist_ids and (not ga_name or ga_name not in hist_names):
                    ab += 1
            adds_beyond = ab

        pattern_results[pat_name] = {
            "field": field,
            "scope": pat_def["scope"],
            "note": pat_def.get("note"),
            "coverage_catalog": coverage,
            "gt_has_pattern": gt_has_pattern,
            "gt_has_pattern_not_top1": gt_has_pattern_not_top1,
            "gt_has_pattern_not_any_top5": gt_has_pattern_not_any_top5,
            "same_artist_has_pattern": same_art_has_pat,
            "diff_artist_has_pattern": diff_art_has_pat,
            "adds_beyond_artist_name": adds_beyond,
        }

    # ---- Report ----
    print(f"\n{'=' * 100}")
    print(f"{'pattern':<35s} | {'cov':>5s} | {'gt_has':>6s} | {'!top1':>5s} | {'!top5':>5s} | {'same':>4s} | {'diff':>4s} | scope | passes")
    print("-" * 130)

    passes = []
    for name, r in pattern_results.items():
        cov_pct = r["coverage_catalog"]
        not_top1 = r["gt_has_pattern_not_top1"]
        cov_ok = cov_pct >= MIN_COVERAGE
        count_ok = not_top1 >= MIN_CASES
        verdict = "PASS" if (cov_ok and count_ok) else "fail"
        if cov_ok and count_ok:
            passes.append(name)
        ann = "  [PASS]" if verdict == "PASS" else ""
        print(f"{name:<35s} | {cov_pct:>5.2%} | {r['gt_has_pattern']:>6d} | {not_top1:>5d} | "
              f"{r['gt_has_pattern_not_any_top5']:>5d} | {r['same_artist_has_pattern']:>4d} | "
              f"{r['diff_artist_has_pattern']:>4d} | {r['scope']:<13s} | {verdict}{ann}")

    print(f"\n  Patterns passing filters (>= {MIN_CASES} cases, >= {MIN_COVERAGE:.0%} coverage):")
    if not passes:
        print(f"    (none)")
    else:
        for p in passes:
            r = pattern_results[p]
            print(f"    - {p}: {r['gt_has_pattern_not_top1']} cases  "
                  f"(coverage {r['coverage_catalog']:.0%}, scope {r['scope']})")

    # Compare to R49C historical numbers (R39-only baseline)
    print(f"\n  Historical reference — R49C (R39-only baseline, before R54 was integrated):")
    print(f"    release_decade_match: 45 cases (now: {pattern_results['release_decade_match_last1']['gt_has_pattern_not_top1']})")
    print(f"    release_year_proximity: 66 cases (now: {pattern_results['release_year_proximity_last1']['gt_has_pattern_not_top1']})")
    print(f"    artist_id_match: 17 cases, adds_beyond_artist_name=5  (now: "
          f"{pattern_results['artist_id_match_history']['gt_has_pattern_not_top1']} cases, "
          f"adds_beyond={pattern_results['artist_id_match_history']['adds_beyond_artist_name']})")
    print(f"    duration_bucket_match: 97 cases (now: {pattern_results['duration_close_last1']['gt_has_pattern_not_top1']})")
    print(f"    isrc_prefix_match: 35 cases (now: {pattern_results['isrc_country_match']['gt_has_pattern_not_top1']})")
    print(f"  Note: R49C used hist_7 only (n=578 misses). This refresh uses ALL DEMOTED ({len(demoted_idxs)}). Counts are NOT directly comparable.")

    out_data = {
        "baseline": {
            "n_cases": n,
            "bucket_counts": dict(bucket_counts),
            "demoted_cases": len(demoted_idxs),
        },
        "metadata_coverage": {
            "catalog_level": {f: c / total_tracks for f, c in field_cov.items()},
            "demoted_gt_level": {f: c / len(demoted_idxs)
                                  for f, c in demoted_gt_field_cov.items()},
        },
        "patterns": pattern_results,
        "filters": {"min_cases": MIN_CASES, "min_coverage": MIN_COVERAGE},
        "passing_patterns": passes,
        "historical_r49c_h7": {
            "release_decade_match": 45,
            "release_year_proximity": 66,
            "artist_id_match": 17,
            "artist_id_adds_beyond_name": 5,
            "duration_bucket_match": 97,
            "isrc_prefix_match": 35,
            "scope_note": "R49C used hist_7 only (n=578 misses); this refresh uses all DEMOTED.",
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved: {OUT}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
