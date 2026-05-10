#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R49C-b: Release year features experiment.

Config 1 (baseline): R39 exact — 34 features (29 base + 5 album). Must reproduce h7=0.24298.
Config 2 (+year): R39 + 6 release year features = 40 features.
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
FEAT_YEAR = [
    "year_diff_abs_min_to_history",
]

FEAT_R39 = FEAT_BASE + FEAT_ALBUM  # 34 features
FEAT_PLUS_YEAR = FEAT_R39 + FEAT_YEAR  # 40 features

N_FEAT_BASE = len(FEAT_BASE)  # 29
N_FEAT_R39 = len(FEAT_R39)  # 34
N_FEAT_YEAR = len(FEAT_PLUS_YEAR)  # 40
YEAR_OFFSET = N_FEAT_R39  # 34


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def extract_year(release_date: str) -> int | None:
    """Extract year from release_date string like '2006-12-06'."""
    if not release_date:
        return None
    m = re.match(r"(\d{4})", release_date)
    if m:
        return int(m.group(1))
    return None


def load_track_year_map() -> dict[str, int]:
    """Load track_id -> year (int) from HF metadata. Missing/bad = 0."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    year_map: dict[str, int] = {}
    for item in ds:
        tid = str(item["track_id"])
        rd = str(item.get("release_date", "")) if item.get("release_date") else ""
        y = extract_year(rd)
        year_map[tid] = y if y is not None else 0
    return year_map


def load_track_albums_from_hf() -> dict[str, str]:
    """Build track_id -> album_id mapping from HF metadata."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album: dict[str, str] = {}
    for item in ds:
        tid = str(item["track_id"])
        album_id_raw = item.get("album_id", [])
        if isinstance(album_id_raw, list) and album_id_raw:
            album_id = str(album_id_raw[0])
        else:
            album_id = ""
        if album_id:
            track_album[tid] = album_id
        else:
            album_name_raw = item.get("album_name", [])
            if isinstance(album_name_raw, list) and album_name_raw:
                track_album[tid] = str(album_name_raw[0])
            else:
                track_album[tid] = str(album_name_raw) if album_name_raw else ""
    return track_album


def run_cv5(cases, X, gt_idx, sizes, folds, feat_names, n_feat, label):
    """Run CV5 LambdaRank and return case_ndcg, lr_ranks, feature_importances."""
    n = len(cases)
    case_ndcg = np.zeros(n)
    lr_ranks = np.full(n, 999, dtype=np.int64)  # rank of GT (1-indexed), 999 if not in pool
    fold_importances = []

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
                X_tr.append(X[idx, k, :n_feat])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(sizes[idx])
            for k in range(s):
                X_va.append(X[idx, k, :n_feat])
                y_va.append(1.0 if k == gt_idx[idx] else 0.0)
            g_va.append(s)
        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(feat_names))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        model = lgb.train(lr_params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        fold_importances.append(model.feature_importance(importance_type="gain"))
        preds = model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            offset += s
            ranked = np.argsort(-sc)
            if gt_idx[idx] >= 0:
                gt_pos = np.where(ranked == gt_idx[idx])[0]
                if len(gt_pos) > 0:
                    lr_ranks[idx] = int(gt_pos[0]) + 1  # 1-indexed
                    if gt_pos[0] < 20:
                        case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    # Average feature importance across folds
    avg_imp = np.mean(fold_importances, axis=0)
    feat_imp = sorted(zip(feat_names, avg_imp.tolist()), key=lambda x: -x[1])

    return case_ndcg, lr_ranks, feat_imp


def main():
    t0 = time.time()
    print(f"{ts()} R49C-b: Release Year Features Experiment")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    print(f"\n{ts()} Loading payload...")
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
    track_album = load_track_albums_from_hf()
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {non_empty} with album_id")

    print(f"{ts()} Loading year mapping...")
    track_year = load_track_year_map()
    has_year = sum(1 for v in track_year.values() if v > 0)
    print(f"  Year mapping: {len(track_year)} tracks, {has_year} with year")

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
    # Build feature matrix (40 features — superset)
    # ---------------------------------------------------------------
    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)

    print(f"{ts()} Building feature matrix ({N_FEAT_YEAR} features = {N_FEAT_R39} R39 + {len(FEAT_YEAR)} year)...")

    X = np.zeros((n, POOL_K, N_FEAT_YEAR), dtype=np.float64)
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
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools.append(pool)
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

        # All history years for min diff
        all_hist_years = [track_year.get(t, 0) for t in played]
        all_hist_years_valid = [y for y in all_hist_years if y > 0]

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

            # Album features (indices 29-33, EXACT copy from R39)
            c_album = track_album.get(tid, "")
            row[N_FEAT_BASE + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[N_FEAT_BASE + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[N_FEAT_BASE + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[N_FEAT_BASE + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[N_FEAT_BASE + 4] = pool_album_count / max(len(pool), 1)

            # Release year feature (single)
            cand_year = track_year.get(tid, 0)

            # year_diff_abs_min_to_history (single feature)
            if cand_year > 0 and all_hist_years_valid:
                row[YEAR_OFFSET + 0] = float(min(abs(cand_year - hy) for hy in all_hist_years_valid))
            else:
                row[YEAR_OFFSET + 0] = 100.0

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # ---------------------------------------------------------------
    # Config 1: Baseline (34 features = R39)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config 1: R39 baseline ({N_FEAT_R39} features)")
    print("-" * 70)

    base_ndcg, base_ranks, base_imp = run_cv5(
        cases, X, gt_idx, sizes, folds,
        feat_names=FEAT_R39, n_feat=N_FEAT_R39, label="baseline",
    )

    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]

    base_h7 = float(np.mean([base_ndcg[i] for i in h7]))
    base_cv5 = float(np.mean(base_ndcg))
    base_same = float(np.mean([base_ndcg[i] for i in h7_same])) if h7_same else 0.0
    base_diff = float(np.mean([base_ndcg[i] for i in h7_diff])) if h7_diff else 0.0

    print(f"  h7={base_h7:.5f}  cv5={base_cv5:.5f}  same={base_same:.5f}  diff={base_diff:.5f}")

    # HARD STOP
    expected_h7 = 0.24298
    if abs(base_h7 - expected_h7) > 0.001:
        print(f"\n  *** HARD STOP: Expected h7 ~{expected_h7}, got {base_h7:.5f} ***")
        print(f"  Difference: {abs(base_h7 - expected_h7):.5f} > 0.001")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={base_h7:.5f} within +/-0.001 of {expected_h7}")

    # ---------------------------------------------------------------
    # Config 2: +year (40 features)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config 2: +year ({N_FEAT_YEAR} features)")
    print("-" * 70)

    year_ndcg, year_ranks, year_imp = run_cv5(
        cases, X, gt_idx, sizes, folds,
        feat_names=FEAT_PLUS_YEAR, n_feat=N_FEAT_YEAR, label="+year",
    )

    year_h7 = float(np.mean([year_ndcg[i] for i in h7]))
    year_cv5 = float(np.mean(year_ndcg))
    year_same = float(np.mean([year_ndcg[i] for i in h7_same])) if h7_same else 0.0
    year_diff = float(np.mean([year_ndcg[i] for i in h7_diff])) if h7_diff else 0.0

    print(f"  h7={year_h7:.5f}  cv5={year_cv5:.5f}  same={year_same:.5f}  diff={year_diff:.5f}")

    # ---------------------------------------------------------------
    # Deltas
    # ---------------------------------------------------------------
    delta_h7 = year_h7 - base_h7
    delta_cv5 = year_cv5 - base_cv5
    delta_same = year_same - base_same
    delta_diff = year_diff - base_diff

    print(f"\n{'='*70}")
    print("DELTAS")
    print(f"{'='*70}")
    print(f"  Δh7={delta_h7:+.5f}  Δcv5={delta_cv5:+.5f}  Δsame={delta_same:+.5f}  Δdiff={delta_diff:+.5f}")

    # ---------------------------------------------------------------
    # Recovered / Lost analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RECOVERED / LOST ANALYSIS")
    print(f"{'='*70}")

    recovered_cases = []
    lost_cases = []
    for i in h7:
        br = base_ranks[i]
        yr = year_ranks[i]
        if br > 20 and yr <= 20:
            recovered_cases.append(i)
        elif br <= 20 and yr > 20:
            lost_cases.append(i)

    recovered = len(recovered_cases)
    lost = len(lost_cases)
    net = recovered - lost
    print(f"  Recovered: {recovered}  Lost: {lost}  Net: {net:+d}")

    # Bucket recovery — classify recovered cases into B/C/D buckets
    b_recovered = 0
    c_recovered = 0
    d_recovered = 0
    for i in recovered_cases:
        br = base_ranks[i]
        if gt_idx[i] < 0:
            d_recovered += 1  # GT not in pool at all
        elif br <= 100:
            b_recovered += 1  # rank 21-100
        else:
            c_recovered += 1  # rank 100+

    print(f"  Bucket recovery: B={b_recovered}  C={c_recovered}  D={d_recovered}")

    # ---------------------------------------------------------------
    # Churn analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CHURN ANALYSIS")
    print(f"{'='*70}")

    # Need to rerun to get top-1 and top-20 sets. Use stored LR ranks approach:
    # Re-derive from the raw scores. We need to re-run CV5 to get raw rankings.
    # Actually, we already have lr_ranks for GT only. For churn, we need the actual
    # predicted ranking of ALL items, not just GT position.
    # Let's do a dedicated pass for churn.

    # Re-run CV5 to capture full predicted rankings for h7 cases
    print(f"{ts()} Computing churn (full ranking pass)...")
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }

    base_top1 = {}   # case_idx -> pool track_id at rank 1
    base_top20 = {}  # case_idx -> set of pool track_ids in top 20
    year_top1 = {}
    year_top20 = {}

    for config_label, n_f, feat_names, top1_dict, top20_dict in [
        ("base", N_FEAT_R39, FEAT_R39, base_top1, base_top20),
        ("+year", N_FEAT_YEAR, FEAT_PLUS_YEAR, year_top1, year_top20),
    ]:
        for fi in range(5):
            val_set = set(folds[fi].tolist())
            tr = [j for j in range(n) if j not in val_set]
            va = sorted(val_set)
            X_tr, y_tr, g_tr = [], [], []
            X_va, y_va, g_va = [], [], []
            for idx in tr:
                s = int(sizes[idx])
                for k in range(s):
                    X_tr.append(X[idx, k, :n_f])
                    y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
                g_tr.append(s)
            for idx in va:
                s = int(sizes[idx])
                for k in range(s):
                    X_va.append(X[idx, k, :n_f])
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
                offset += s
                if idx in set(h7):
                    ranked = np.argsort(-sc)
                    top1_dict[idx] = pools[idx][int(ranked[0])]
                    top20_set = {pools[idx][int(ranked[j])] for j in range(min(20, s))}
                    top20_dict[idx] = top20_set

    top1_changed = sum(1 for i in h7 if base_top1.get(i) != year_top1.get(i))
    top20_changed = sum(1 for i in h7 if base_top20.get(i, set()) != year_top20.get(i, set()))

    print(f"  top1_changed: {top1_changed}/{len(h7)}")
    print(f"  top20_changed: {top20_changed}/{len(h7)}")

    # ---------------------------------------------------------------
    # Feature importance (from +year config, averaged across folds)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE (top 10 by gain, +year config, 5-fold avg)")
    print(f"{'='*70}")
    for fname, gain in year_imp[:10]:
        print(f"  {fname:<30} {gain:.1f}")

    # Also show year features specifically
    print("\n  Year features specifically:")
    year_only = [(fname, gain) for fname, gain in year_imp if fname in FEAT_YEAR]
    for fname, gain in year_only:
        rank_in_all = [fn for fn, _ in year_imp].index(fname) + 1
        print(f"    {fname:<30} {gain:.1f}  (rank {rank_in_all}/{len(year_imp)})")

    # ---------------------------------------------------------------
    # Gate
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("R49C-b GATE")
    print(f"{'='*70}")

    h7_pass = delta_h7 >= 0.010
    net_pass = net >= 20
    same_diff_pass = (delta_same >= 0) and (delta_diff >= 0)

    print(f"  baseline:  h7={base_h7:.5f}  cv5={base_cv5:.5f}  same={base_same:.5f}  diff={base_diff:.5f}")
    print(f"  +year:     h7={year_h7:.5f}  cv5={year_cv5:.5f}  same={year_same:.5f}  diff={year_diff:.5f}")
    print(f"  Δh7={delta_h7:+.5f}  Δcv5={delta_cv5:+.5f}  Δsame={delta_same:+.5f}  Δdiff={delta_diff:+.5f}")
    print(f"  Recovered={recovered}  Lost={lost}  Net={net:+d}")
    print()
    print(f"  Δh7 >= +0.010:     {delta_h7:+.5f}  {'PASS' if h7_pass else 'FAIL'}")
    print(f"  net >= +20:        {net:+d}  {'PASS' if net_pass else 'FAIL'}")
    print(f"  same/diff >= 0:    Δsame={delta_same:+.5f} Δdiff={delta_diff:+.5f}  {'PASS' if same_diff_pass else 'FAIL'}")

    overall = "PASS" if (h7_pass and net_pass and same_diff_pass) else "FAIL"
    print(f"\n  OVERALL: {overall}")

    if not h7_pass and delta_h7 >= 0.005:
        print("  WARNING: Δh7 in [0.005, 0.010) — R41a risk zone, marking FAIL per gate policy")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "experiment": "R49C-b_release_year",
        "configs": {
            "baseline": {"n_features": N_FEAT_R39, "h7": round(base_h7, 5), "cv5": round(base_cv5, 5),
                         "same": round(base_same, 5), "diff": round(base_diff, 5)},
            "+year": {"n_features": N_FEAT_YEAR, "h7": round(year_h7, 5), "cv5": round(year_cv5, 5),
                      "same": round(year_same, 5), "diff": round(year_diff, 5)},
        },
        "delta_h7": round(delta_h7, 5),
        "delta_cv5": round(delta_cv5, 5),
        "delta_same": round(delta_same, 5),
        "delta_diff": round(delta_diff, 5),
        "recovered_lost": {"recovered": recovered, "lost": lost, "net": net},
        "bucket_recovery": {"B_recovered": b_recovered, "C_recovered": c_recovered, "D_recovered": d_recovered},
        "churn": {"top1_changed": top1_changed, "top20_changed": top20_changed},
        "feature_importance_top10": [{"name": fn, "gain": round(g, 2)} for fn, g in year_imp[:10]],
        "gate": {
            "h7_pass": h7_pass,
            "net_pass": net_pass,
            "same_diff_pass": same_diff_pass,
            "overall": overall,
        },
        "gate_thresholds": {"h7_min": 0.010, "net_min": 20},
    }

    out_path = REPO / "exp" / "eval" / "expR49c_b_release_year_single.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
