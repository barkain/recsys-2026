#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R49C-a: Duration features experiment.

Config 1 (baseline): R39 exact -- 34 features (29 base + 5 album). Must reproduce h7=0.24298.
Config 2 (+duration): R39 + 6 duration features = 40 features.

Both configs: same pool (POOL_K=300, RRF_K=20), same R21 OOF, same ALS.
"""
from __future__ import annotations

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
FEAT_ALL_R39 = FEAT_BASE + FEAT_ALBUM  # 34 features

FEAT_DURATION = [
    "duration_diff_abs_min_to_history",
]
FEAT_ALL_PLUS_DUR = FEAT_ALL_R39 + FEAT_DURATION  # 35 features


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_durations():
    """Build track_id -> duration (ms) mapping from HF metadata."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    dur_map = {}
    for item in ds:
        tid = str(item["track_id"])
        dur = item.get("duration")
        dur_map[tid] = int(dur) if dur is not None else 0
    return dur_map


def load_track_albums():
    """Build track_id -> album_id mapping from HF metadata."""
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
        if album_id:
            track_album[tid] = album_id
        else:
            album_name_raw = item.get("album_name", [])
            if isinstance(album_name_raw, list) and album_name_raw:
                track_album[tid] = str(album_name_raw[0])
            else:
                track_album[tid] = str(album_name_raw) if album_name_raw else ""
    return track_album


def build_feature_matrix(cases, payload, als_source, als_vecs, als_factors,
                         als_track_to_idx, r21_source, track_pop, max_pop,
                         track_album, track_dur, n_feat, include_duration):
    """Build feature matrix for all cases.

    Returns X, gt_idx, sizes, pools.
    """
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    n = len(cases)
    n_feat_base = len(FEAT_BASE)

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

        # Duration precomputation (only need all history for min diff)
        all_hist_durs = [track_dur.get(t, 0) for t in played]
        all_hist_durs = [d for d in all_hist_durs if d > 0]

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

            # Album features (EXACT copy from R39)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

            # Duration feature (single: duration_diff_abs_min_to_history)
            if include_duration:
                cand_dur = track_dur.get(tid, 0)
                dur_base = n_feat_base + 5  # = 34

                if cand_dur > 0 and all_hist_durs:
                    row[dur_base + 0] = min(abs(cand_dur - hd) for hd in all_hist_durs) / 1000.0
                else:
                    row[dur_base + 0] = 0.0

    return X, gt_idx, sizes, pools


def run_cv5(X, gt_idx, sizes, folds, n, feat_names, n_feat):
    """Run CV5 LambdaRank, return case_ndcg, lr_scores, feature_importance."""
    case_ndcg = np.zeros(n)
    lr_scores = np.full((n, POOL_K), -np.inf, dtype=np.float64)
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }
    feat_imp_accum = np.zeros(n_feat, dtype=np.float64)

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
        preds = model.predict(np.array(X_va))
        feat_imp_accum += np.array(model.feature_importance(importance_type='gain'), dtype=np.float64)
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
    return case_ndcg, lr_scores, feat_imp_avg


def compute_metrics(case_ndcg, cases, n):
    """Compute h7, cv5, same, diff metrics."""
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    same_idx = [i for i in range(n) if cases[i].get("scenario") == "same"]
    diff_idx = [i for i in range(n) if cases[i].get("scenario") == "different"]

    h7 = float(np.mean([case_ndcg[i] for i in h7_idx])) if h7_idx else 0.0
    cv5 = float(np.mean(case_ndcg))
    same = float(np.mean([case_ndcg[i] for i in same_idx])) if same_idx else 0.0
    diff = float(np.mean([case_ndcg[i] for i in diff_idx])) if diff_idx else 0.0

    return h7, cv5, same, diff, h7_idx


def main():
    t0 = time.time()
    print(f"{ts()} R49C-a: Duration Features Experiment")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load shared data
    # ---------------------------------------------------------------
    print(f"\n{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Album mapping: {len(track_album)} tracks, {non_empty} with album_id")

    print(f"{ts()} Loading duration mapping...")
    track_dur = load_track_durations()
    dur_nonzero = sum(1 for v in track_dur.values() if v > 0)
    print(f"  Duration mapping: {len(track_dur)} tracks, {dur_nonzero} with duration > 0")

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

    # ---------------------------------------------------------------
    # CONFIG 1: Baseline (R39 exact, 34 features)
    # ---------------------------------------------------------------
    n_feat_baseline = len(FEAT_ALL_R39)
    print(f"\n{ts()} CONFIG 1: Baseline ({n_feat_baseline} features)")
    print("-" * 70)

    # Build feature matrix with 40 columns (includes duration slots)
    # but we only use first 34 for baseline training
    n_feat_full = len(FEAT_ALL_PLUS_DUR)
    print(f"{ts()} Building feature matrix ({n_feat_full} columns, duration included)...")
    X, gt_idx, sizes, pools = build_feature_matrix(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, r21_source, track_pop, max_pop,
        track_album, track_dur, n_feat_full, include_duration=True)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    print(f"{ts()} Running CV5 LambdaRank (baseline, {n_feat_baseline} features)...")
    base_ndcg, base_scores, base_imp = run_cv5(
        X, gt_idx, sizes, folds, n, FEAT_ALL_R39, n_feat_baseline)

    base_h7, base_cv5, base_same, base_diff, h7_idx = compute_metrics(base_ndcg, cases, n)
    print(f"  Baseline: h7={base_h7:.5f}  cv5={base_cv5:.5f}  same={base_same:.5f}  diff={base_diff:.5f}")

    # HARD CHECK
    expected_h7 = 0.24298
    if abs(base_h7 - expected_h7) > 0.001:
        print("\n  *** HARD CHECK FAILED ***")
        print(f"  Expected h7 ~{expected_h7}, got {base_h7:.5f}")
        print(f"  Difference: {abs(base_h7 - expected_h7):.5f} > 0.001")
        print("  STOPPING.")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={base_h7:.5f} within +/-0.001 of {expected_h7}")

    # ---------------------------------------------------------------
    # CONFIG 2: +duration (40 features)
    # ---------------------------------------------------------------
    print(f"\n{ts()} CONFIG 2: +duration ({n_feat_full} features)")
    print("-" * 70)

    print(f"{ts()} Running CV5 LambdaRank (+duration, {n_feat_full} features)...")
    dur_ndcg, dur_scores, dur_imp = run_cv5(
        X, gt_idx, sizes, folds, n, FEAT_ALL_PLUS_DUR, n_feat_full)

    dur_h7, dur_cv5, dur_same, dur_diff, _ = compute_metrics(dur_ndcg, cases, n)
    print(f"  +duration: h7={dur_h7:.5f}  cv5={dur_cv5:.5f}  same={dur_same:.5f}  diff={dur_diff:.5f}")

    # ---------------------------------------------------------------
    # Deltas
    # ---------------------------------------------------------------
    delta_h7 = dur_h7 - base_h7
    delta_cv5 = dur_cv5 - base_cv5
    delta_same = dur_same - base_same
    delta_diff = dur_diff - base_diff
    print(f"\n  delta_h7:   {delta_h7:+.5f}")
    print(f"  delta_cv5:  {delta_cv5:+.5f}")
    print(f"  delta_same: {delta_same:+.5f}")
    print(f"  delta_diff: {delta_diff:+.5f}")

    # ---------------------------------------------------------------
    # Recovered / Lost analysis
    # ---------------------------------------------------------------
    print(f"\n{ts()} Recovered / Lost Analysis")
    print("-" * 70)

    recovered = 0
    lost = 0
    bucket_recovered = {"B": 0, "C": 0, "D": 0}
    top1_changed = 0
    top20_changed = 0

    for i in h7_idx:
        ps = int(sizes[i])
        if ps == 0:
            continue

        base_sc = base_scores[i, :ps]
        dur_sc = dur_scores[i, :ps]
        base_ranked = np.argsort(-base_sc)
        dur_ranked = np.argsort(-dur_sc)

        gt_pool_idx = int(gt_idx[i])

        # Check recovered/lost only for cases where GT is in pool
        if gt_pool_idx >= 0:
            base_gt_pos = np.where(base_ranked == gt_pool_idx)[0]
            dur_gt_pos = np.where(dur_ranked == gt_pool_idx)[0]
            base_rank = int(base_gt_pos[0]) + 1 if len(base_gt_pos) > 0 else ps + 1
            dur_rank = int(dur_gt_pos[0]) + 1 if len(dur_gt_pos) > 0 else ps + 1

            if base_rank > 20 and dur_rank <= 20:
                recovered += 1
                # Classify bucket
                if base_rank <= 100:
                    bucket_recovered["B"] += 1
                elif base_rank <= ps:
                    bucket_recovered["C"] += 1
            elif base_rank <= 20 and dur_rank > 20:
                lost += 1

        # Churn: top-1 changed
        if int(base_ranked[0]) != int(dur_ranked[0]):
            top1_changed += 1

        # Churn: top-20 set membership differs
        base_top20 = set(int(base_ranked[j]) for j in range(min(20, ps)))
        dur_top20 = set(int(dur_ranked[j]) for j in range(min(20, ps)))
        if base_top20 != dur_top20:
            top20_changed += 1

    net = recovered - lost
    print(f"  Recovered (miss->hit): {recovered}")
    print(f"  Lost (hit->miss):      {lost}")
    print(f"  Net:                   {net:+d}")
    print(f"  B recovered: {bucket_recovered['B']}")
    print(f"  C recovered: {bucket_recovered['C']}")
    print(f"  D recovered: {bucket_recovered['D']}")
    print(f"  top1_changed: {top1_changed}")
    print(f"  top20_changed: {top20_changed}")

    # ---------------------------------------------------------------
    # Feature importance (top 10 from +duration model)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Feature Importance (top 10, +duration model)")
    print("-" * 70)

    feat_imp_list = list(zip(FEAT_ALL_PLUS_DUR, dur_imp))
    feat_imp_sorted = sorted(feat_imp_list, key=lambda x: -x[1])
    top10 = feat_imp_sorted[:10]
    for fname, gain in top10:
        print(f"  {fname:>40s}: {gain:.2f}")

    # ---------------------------------------------------------------
    # Gate
    # ---------------------------------------------------------------
    h7_pass = delta_h7 >= 0.010
    net_pass = net >= 20
    same_diff_pass = delta_same >= 0.0 and delta_diff >= 0.0
    overall = "PASS" if (h7_pass and net_pass and same_diff_pass) else "FAIL"

    print(f"\n{ts()} GATE")
    print("-" * 70)
    print(f"  h7_delta >= 0.010:  {delta_h7:+.5f} -> {'PASS' if h7_pass else 'FAIL'}")
    print(f"  net >= 20:          {net:+d} -> {'PASS' if net_pass else 'FAIL'}")
    print(f"  same/diff >= 0:     same={delta_same:+.5f} diff={delta_diff:+.5f} -> {'PASS' if same_diff_pass else 'FAIL'}")
    print(f"  OVERALL:            {overall}")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    results = {
        "experiment": "R49C-a_duration",
        "configs": {
            "baseline": {
                "n_features": n_feat_baseline,
                "h7": round(base_h7, 5),
                "cv5": round(base_cv5, 5),
                "same": round(base_same, 5),
                "diff": round(base_diff, 5),
            },
            "+duration": {
                "n_features": n_feat_full,
                "h7": round(dur_h7, 5),
                "cv5": round(dur_cv5, 5),
                "same": round(dur_same, 5),
                "diff": round(dur_diff, 5),
            },
        },
        "delta_h7": round(delta_h7, 5),
        "delta_cv5": round(delta_cv5, 5),
        "delta_same": round(delta_same, 5),
        "delta_diff": round(delta_diff, 5),
        "recovered_lost": {
            "recovered": recovered,
            "lost": lost,
            "net": net,
        },
        "bucket_recovery": {
            "B_recovered": bucket_recovered["B"],
            "C_recovered": bucket_recovered["C"],
            "D_recovered": bucket_recovered["D"],
        },
        "churn": {
            "top1_changed": top1_changed,
            "top20_changed": top20_changed,
        },
        "feature_importance_top10": [
            {"name": fname, "gain": round(float(gain), 2)} for fname, gain in top10
        ],
        "gate": {
            "h7_pass": h7_pass,
            "net_pass": net_pass,
            "same_diff_pass": same_diff_pass,
            "overall": overall,
        },
        "gate_thresholds": {
            "h7_min": 0.010,
            "net_min": 20,
        },
    }

    out_path = REPO / "exp" / "eval" / "expR49c_a_duration_single.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
