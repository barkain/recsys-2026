#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 2 full 5-fold R39 integration with R54 features.

Loads OOF R54 retrieval lists (one per case from the fold where that case was held out).
Integrates R54 as 8th RRF source and adds R54 features to LambdaRank.

Sources: A, B, C, D, F, ALS, R21, R54
Features: 34 R39 features + r54_rank_inv + r54_presence + r54_cosine = 37

CV5 LambdaRank, full evaluation:
- h7 nDCG
- cv5 nDCG
- pool_hit@300
- same/diff h7
- recovered/lost/net top-20 vs R39 baseline (no R54)
- Bucket D/E recovery
- fold-by-fold deltas

Gate: Δh7 >= +0.010, Δpool_hit >= +0.020, net positive, no same/diff regression.
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
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"

RRF_K = 20
POOL_K = 300

SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
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


def build_features(pool, case, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
                   als_factors, als_track_to_idx, als_vec, track_pop, max_pop,
                   ta, tt, ttl, tat, tmt, track_album, feat_names):
    n_pool = len(pool)
    n_feat = len(feat_names)
    X = np.zeros((n_pool, n_feat), dtype=np.float64)

    use_r54 = "r54_rank_inv" in feat_names
    n_feat_base = len(FEAT_BASE)

    user_msgs = [str(r["content"]) for r in case["history"] if r["role"] == "user"] + [case["user_query"]]
    played = case["music_turns"]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    played_set = set(played)
    l_artist = ta.get(played[-1], "") if played else ""
    l_tags = tt.get(played[-1], set()) if played else set()
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]
    pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
    artist_counts = Counter(a for a in pool_artists if a)

    src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                for sn, sl in src_lists.items()}

    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
    all_albums = [track_album.get(t, "") for t in played]
    album_hist_counts = Counter(a for a in all_albums if a)

    for rank, tid in enumerate(pool[:POOL_K], start=1):
        ca = ta.get(tid, "")
        ct = tt.get(tid, set())
        row = X[rank - 1]

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
            aidx = als_track_to_idx.get(tid)
            if aidx is not None:
                row[21] = float(np.dot(als_vec, als_factors[aidx]))
        row[22] = float(n_hist)
        row[23] = track_pop.get(tid, 0) / max_pop
        row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
        row[25] = float(artist_counts.get(ca, 0)) if ca else 0
        row[26] = row[20]
        row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
        row[28] = 1.0 if tid in r21_rank_map else 0.0

        # Album features (29..33)
        c_album = track_album.get(tid, "")
        row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
        row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
        row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
        row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
        pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
        row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

        # R54 features (34..36) — only when configured
        if use_r54:
            r54_off = len(FEAT_R39_ALL)
            row[r54_off + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
            row[r54_off + 1] = 1.0 if tid in r54_rank_map else 0.0
            row[r54_off + 2] = r54_score_map.get(tid, 0.0)

    return X


def classify_buckets_baseline(cases, payload, r21_source_full, als_source,
                               h7_idx, ta, sources_for_union):
    """Classify h7 cases into buckets using R39 baseline pool + union.

    A: hit (in baseline top-20 — needs LR scores, defer)
    For now classify based on POOL ONLY:
        in_pool: GT in R39 baseline pool
        in_union: GT in union of all source top-300
        bucket_D: not in pool, but in union
        bucket_E: not in union
    """
    buckets = {}
    for i in h7_idx:
        gt = cases[i]["gt"]
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source_full[i],
        }
        pool = weighted_rrf(src_lists, SW_BASE, topk=POOL_K, k=RRF_K)
        in_pool = gt in pool

        union = set()
        for sl in src_lists.values():
            union.update(sl[:300])
        in_union = gt in union

        if in_pool:
            bucket = "in_pool"
        elif in_union:
            bucket = "D"
        else:
            bucket = "E"
        buckets[i] = bucket
    return buckets


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 2 full 5-fold R39+R54 integration")
    print("=" * 70)

    print(f"{ts()} Loading payload...")
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
        r21_source_full = json.load(f)

    print(f"{ts()} Loading R54 OOF lists...")
    r54_data = json.load(open(R54_OOF))
    r54_oof_with_scores = r54_data["lists"]

    # --- Validation: format consistency before integration ---
    print(f"{ts()} Validating OOF list format...")
    assert len(r54_oof_with_scores) == n, \
        f"OOF length {len(r54_oof_with_scores)} != cases {n}"
    list_lengths = []
    for case_idx, case_lists in enumerate(r54_oof_with_scores):
        assert case_lists is not None, f"case {case_idx}: R54 list is None"
        assert isinstance(case_lists, list) and len(case_lists) > 0, \
            f"case {case_idx}: R54 list empty or wrong type"
        for item_idx, item in enumerate(case_lists):
            assert isinstance(item, (list, tuple)) and len(item) == 2, \
                f"case {case_idx} item {item_idx}: format != (tid, score)"
            tid, score = item
            assert isinstance(tid, str), f"case {case_idx} item {item_idx}: tid not str"
            assert isinstance(score, (int, float)) and np.isfinite(score), \
                f"case {case_idx} item {item_idx}: score invalid ({score})"
        list_lengths.append(len(case_lists))
    print(f"  Validation passed: {n} cases, lengths min={min(list_lengths)} "
          f"median={int(np.median(list_lengths))} max={max(list_lengths)}")

    r54_source_full = []
    r54_scores_full = []
    for case_lists in r54_oof_with_scores:
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        r54_source_full.append(tids)
        r54_scores_full.append(score_map)
    n_with_r54 = sum(1 for x in r54_source_full if x)
    print(f"  R54 lists: {n_with_r54}/{n} cases")

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading popularity & ALS...")
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

    # Pre-compute baseline buckets for h7 cases
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"\n{ts()} Classifying baseline buckets on {len(h7_idx)} h7 cases...")
    src_for_union = ["A", "B", "C", "D", "F", "ALS", "R21"]
    baseline_buckets = classify_buckets_baseline(
        cases, payload, r21_source_full, als_source, h7_idx, ta, src_for_union)
    bucket_counts = Counter(baseline_buckets.values())
    print(f"  Baseline (R39): {dict(bucket_counts)}")

    # ============================================================
    # Configs
    # ============================================================
    configs = [
        ("baseline_R39", SW_BASE, False, FEAT_R39_ALL),
        ("R39+R54_w0.5_pool_only", {**SW_BASE, "R54": 0.5}, False, FEAT_R39_ALL),
        ("R39+R54_w1.0_pool_only", {**SW_BASE, "R54": 1.0}, False, FEAT_R39_ALL),
        ("R39+R54_w1.0_feats", {**SW_BASE, "R54": 1.0}, True, FEAT_ALL),
        ("R39+R54_w1.5_feats", {**SW_BASE, "R54": 1.5}, True, FEAT_ALL),
        ("R39+R54_w0.5_feats", {**SW_BASE, "R54": 0.5}, True, FEAT_ALL),
    ]

    all_results = {}
    baseline_top20_per_case = None
    baseline_pool_per_case = None

    for cfg_name, weights, use_r54, feat_names in configs:
        cfg_t0 = time.time()
        print(f"\n{ts()} === Config: {cfg_name} ({len(feat_names)} feats, R54 in pool: {'R54' in weights}) ===")

        n_feat = len(feat_names)
        X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        pools_by_case = {}

        for i in range(n):
            src_lists = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
                "R21": r21_source_full[i],
            }
            if "R54" in weights:
                src_lists["R54"] = r54_source_full[i]

            pool = weighted_rrf(src_lists, weights, topk=POOL_K, k=RRF_K)
            sizes[i] = len(pool)
            pools_by_case[i] = pool
            if cases[i]["gt"] in pool:
                gt_idx[i] = pool.index(cases[i]["gt"])

            r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source_full[i][:300])}
            r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_source_full[i][:300])}
            r54_score_map = r54_scores_full[i]

            Xi = build_features(pool, cases[i], src_lists, r21_rank_map,
                                 r54_rank_map, r54_score_map,
                                 als_factors, als_track_to_idx, als_vecs[i],
                                 track_pop, max_pop, ta, tt, ttl, tat, tmt,
                                 track_album, feat_names)
            X[i, :len(pool)] = Xi

            if (i + 1) % 1000 == 0:
                print(f"  features {i + 1}/{n} ({time.time() - cfg_t0:.0f}s)", flush=True)

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

        # CV5 LambdaRank
        case_ndcg = np.zeros(n)
        case_top20 = {}
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
                                group=g_tr, feature_name=list(feat_names))
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
                pool = pools_by_case[idx]
                if s == 0:
                    case_top20[idx] = []
                    continue
                ranked = np.argsort(-sc)
                case_top20[idx] = [pool[j] for j in ranked[:20]]
                if gt_idx[idx] < 0:
                    continue
                gt_pos = np.where(ranked == gt_idx[idx])[0]
                if len(gt_pos) > 0 and gt_pos[0] < 20:
                    case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

        # Aggregate metrics
        h7_same = [i for i in h7_idx if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7_idx if i not in set(h7_same)]

        h7 = float(np.mean([case_ndcg[i] for i in h7_idx]))
        cv5 = float(np.mean(case_ndcg))
        same = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0.0
        diff = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0.0

        print(f"  h7={h7:.5f}  cv5={cv5:.5f}  same={same:.5f}  diff={diff:.5f}")

        # Recovered/lost vs baseline
        recovered = lost = unchanged = 0
        if baseline_top20_per_case:
            for i in range(n):
                base = set(baseline_top20_per_case.get(i, []))
                now = set(case_top20.get(i, []))
                gt = cases[i]["gt"]
                if gt in now and gt not in base:
                    recovered += 1
                elif gt in base and gt not in now:
                    lost += 1
                elif gt in base and gt in now:
                    unchanged += 1
            net = recovered - lost
            print(f"  vs baseline: recovered={recovered}  lost={lost}  net={net:+d}")
        else:
            net = 0

        # Bucket D/E recovery
        bucket_recovery = {"D_recovered_to_pool": 0, "D_total": 0,
                           "E_recovered_to_pool": 0, "E_total": 0,
                           "D_into_top20": 0, "E_into_top20": 0}
        for i in h7_idx:
            b = baseline_buckets[i]
            gt = cases[i]["gt"]
            pool = pools_by_case[i]
            in_pool_now = gt in pool
            in_top20_now = gt in set(case_top20.get(i, []))
            if b == "D":
                bucket_recovery["D_total"] += 1
                if in_pool_now:
                    bucket_recovery["D_recovered_to_pool"] += 1
                if in_top20_now:
                    bucket_recovery["D_into_top20"] += 1
            elif b == "E":
                bucket_recovery["E_total"] += 1
                if in_pool_now:
                    bucket_recovery["E_recovered_to_pool"] += 1
                if in_top20_now:
                    bucket_recovery["E_into_top20"] += 1

        # Fold-by-fold h7
        fold_h7 = {}
        for fi in range(5):
            fold_h7_idx = [i for i in h7_idx if i in set(folds[fi].tolist())]
            if fold_h7_idx:
                fold_h7[fi] = float(np.mean([case_ndcg[i] for i in fold_h7_idx]))

        # Top-20 overlap with baseline (fraction of top-20 same)
        top20_overlap = 0.0
        if baseline_top20_per_case:
            ov = []
            for i in range(n):
                base = set(baseline_top20_per_case.get(i, []))
                now = set(case_top20.get(i, []))
                if base or now:
                    ov.append(len(base & now) / 20.0)
            top20_overlap = float(np.mean(ov)) if ov else 0.0

        all_results[cfg_name] = {
            "name": cfg_name,
            "pool_hit": pool_hit,
            "h7_ndcg": h7, "cv5_ndcg": cv5,
            "h7_same_ndcg": same, "h7_diff_ndcg": diff,
            "n_h7": len(h7_idx), "n_h7_same": len(h7_same), "n_h7_diff": len(h7_diff),
            "recovered": recovered, "lost": lost, "net": net,
            "top20_overlap": top20_overlap,
            "bucket_recovery": bucket_recovery,
            "fold_h7": fold_h7,
            "config": {"weights": weights, "feat_names": list(feat_names)},
            "elapsed_s": time.time() - cfg_t0,
        }

        if cfg_name == "baseline_R39":
            baseline_top20_per_case = case_top20
            baseline_pool_per_case = pools_by_case

        print(f"  bucket D recovery: {bucket_recovery['D_recovered_to_pool']}/{bucket_recovery['D_total']} to pool, "
              f"{bucket_recovery['D_into_top20']}/{bucket_recovery['D_total']} to top20")
        print(f"  bucket E recovery: {bucket_recovery['E_recovered_to_pool']}/{bucket_recovery['E_total']} to pool, "
              f"{bucket_recovery['E_into_top20']}/{bucket_recovery['E_total']} to top20")
        if fold_h7:
            print(f"  fold-by-fold h7: " + "  ".join(f"f{k}={v:.4f}" for k, v in sorted(fold_h7.items())))

    # ============================================================
    # Summary
    # ============================================================
    base = all_results["baseline_R39"]
    print(f"\n{ts()} === SUMMARY ===")
    print(f"  {'Config':<30} {'pool':>7} {'h7':>8} {'cv5':>8} {'same':>8} {'diff':>8} "
          f"{'rec':>5} {'lost':>5} {'net':>5} {'ΔD/p':>5} {'ΔE/p':>5}")
    for cfg_name, _, _, _ in configs:
        r = all_results[cfg_name]
        dh7 = r["h7_ndcg"] - base["h7_ndcg"]
        dp = r["pool_hit"] - base["pool_hit"]
        dD = r["bucket_recovery"]["D_recovered_to_pool"]
        dE = r["bucket_recovery"]["E_recovered_to_pool"]
        marker = ""
        if cfg_name != "baseline_R39":
            marker = f"  Δh7={dh7:+.4f}  Δpool={dp:+.4f}"
        print(f"  {cfg_name:<30} {r['pool_hit']:>7.4f} {r['h7_ndcg']:>8.5f} {r['cv5_ndcg']:>8.5f} "
              f"{r['h7_same_ndcg']:>8.5f} {r['h7_diff_ndcg']:>8.5f} "
              f"{r['recovered']:>5} {r['lost']:>5} {r['net']:>+5} {dD:>5} {dE:>5}{marker}")

    # ============================================================
    # Gate
    # ============================================================
    print(f"\n{ts()} === GATE CHECK ===")
    best_cfg = max((c for c in all_results if c != "baseline_R39"),
                    key=lambda c: all_results[c]["h7_ndcg"])
    best = all_results[best_cfg]
    dh7 = best["h7_ndcg"] - base["h7_ndcg"]
    dpool = best["pool_hit"] - base["pool_hit"]
    dsame = best["h7_same_ndcg"] - base["h7_same_ndcg"]
    ddiff = best["h7_diff_ndcg"] - base["h7_diff_ndcg"]

    print(f"  Best: {best_cfg}")
    print(f"  Δh7: {dh7:+.5f}  (gate >= +0.010)  {'PASS' if dh7 >= 0.010 else 'FAIL'}")
    print(f"  Δpool: {dpool:+.5f}  (gate >= +0.020)  {'PASS' if dpool >= 0.020 else 'FAIL'}")
    print(f"  Net: {best['net']:+d}  (gate > 0)  {'PASS' if best['net'] > 0 else 'FAIL'}")
    print(f"  Δsame: {dsame:+.5f}  Δdiff: {ddiff:+.5f}  (no regression)  "
          f"{'PASS' if min(dsame, ddiff) >= -0.005 else 'FAIL'}")

    gates_pass = (dh7 >= 0.010 and best["net"] > 0 and dsame >= -0.005 and ddiff >= -0.005)
    print(f"\n  Overall: {'PASS — production candidate' if gates_pass else 'FAIL — not production-ready'}")

    out_path = REPO / "exp" / "eval" / "expR54_phase2_full5fold_integration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "results": all_results,
            "best_cfg": best_cfg,
            "deltas_vs_baseline": {
                "dh7": dh7, "dpool": dpool, "dsame": dsame, "ddiff": ddiff,
                "net": best["net"],
            },
            "gates_pass": gates_pass,
            "baseline_buckets": dict(bucket_counts),
            "elapsed_s": time.time() - t0,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{ts()} Integration complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
