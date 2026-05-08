#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R33b: Enhanced LambdaRank with R22b/Q3/artist features.

Test R22b and Q3 as ranking features (not pool expansion).
Test same-artist features. Evaluate movement diagnostics on hist_7.

Configs:
1. Baseline (29 features)
2. + same-artist features only
3. + R22b/Q3 rank features only
4. + same-artist + R22b/Q3 features
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
R22B_OOF = REPO / "cache" / "r22b" / "dev_r22b_lists.json"
Q3_LISTS = REPO / "cache" / "r26" / "q3_dense_results.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ARTIST = [
    "same_artist_last1", "same_artist_last3", "same_artist_any",
    "artist_session_freq", "artist_session_frac",
]
FEAT_SOURCES = [
    "r22b_rank_inv", "r22b_presence",
    "q3_rank_inv", "q3_presence",
    "source_agree_r21_r22b", "source_agree_r21_q3",
    "source_agree_behavioral",
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_features(cases, payload, als_source, als_vecs, als_factors,
                   als_track_to_idx, track_pop, r21_source,
                   r22b_source, q3_source, feature_names):
    n = len(cases)
    n_feat = len(feature_names)
    has_artist = "same_artist_last1" in feature_names
    has_sources = "r22b_rank_inv" in feature_names

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

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

        # Precompute for new features
        last1_artist: set[str] = set()
        last3_artists: set[str] = set()
        all_artists: set[str] = set()
        artist_freq: Counter[str] = Counter()
        max_freq = 1
        r22b_map: dict[str, int] = {}
        q3_map: dict[str, int] = {}
        r21_top20: set[str] = set()
        als_top20: set[str] = set()
        f_top20: set[str] = set()

        if has_artist:
            last1_artist = {ta.get(played[-1], "")} - {""} if played else set()
            last3_artists = {ta.get(t, "") for t in played[-3:]} - {""}
            all_artists = {ta.get(t, "") for t in played} - {""}
            artist_freq = Counter(ta.get(t, "") for t in played)
            artist_freq.pop("", None)
            max_freq = max(artist_freq.values()) if artist_freq else 1

        if has_sources:
            r22b_map = {tid: r + 1 for r, tid in enumerate(r22b_source[i][:300])}
            q3_map = {tid: r + 1 for r, tid in enumerate(q3_source[i][:300])}
            r21_top20 = set(r21_source[i][:20])
            als_top20 = set(als_source[i][:20])
            f_top20 = set(src_lists["F"][:20])

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features
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

            base_end = len(FEAT_BASE)

            if has_artist:
                row[base_end + 0] = 1.0 if ca and ca in last1_artist else 0.0
                row[base_end + 1] = 1.0 if ca and ca in last3_artists else 0.0
                row[base_end + 2] = 1.0 if ca and ca in all_artists else 0.0
                row[base_end + 3] = float(artist_freq.get(ca, 0)) / max_freq if ca else 0.0
                row[base_end + 4] = float(artist_freq.get(ca, 0)) / max(n_hist, 1) if ca else 0.0

            if has_sources:
                src_offset = base_end + (len(FEAT_ARTIST) if has_artist else 0)
                row[src_offset + 0] = 1.0 / r22b_map[tid] if tid in r22b_map else 0.0
                row[src_offset + 1] = 1.0 if tid in r22b_map else 0.0
                row[src_offset + 2] = 1.0 / q3_map[tid] if tid in q3_map else 0.0
                row[src_offset + 3] = 1.0 if tid in q3_map else 0.0
                row[src_offset + 4] = 1.0 if tid in r22b_map and tid in r21_rank_map else 0.0
                row[src_offset + 5] = 1.0 if tid in q3_map and tid in r21_rank_map else 0.0
                row[src_offset + 6] = sum(1 for s in [r21_top20, als_top20, f_top20]
                                          if tid in s) / 3.0

    return X, gt_idx, sizes, pools


def run_cv5_with_diagnostics(X, gt_idx, sizes, cases, sessions, pools, ta,
                              feature_names, label=""):
    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)
    case_ndcg = np.zeros(n)
    case_gt_rank = np.full(n, -1)

    # First get baseline rankings (from fold 0 of the standard 29-feature model)
    # Actually we compute both in one pass
    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        train_list = [j for j in range(n) if j not in val_idx]
        val_list = sorted(val_idx)

        X_flat_tr, y_tr, g_tr = [], [], []
        X_flat_va, y_va, g_va = [], [], []

        for idx in train_list:
            s = int(sizes[idx])
            for k in range(s):
                X_flat_tr.append(X[idx, k])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in val_list:
            s = int(sizes[idx])
            for k in range(s):
                X_flat_va.append(X[idx, k])
                y_va.append(1.0 if k == gt_idx[idx] else 0.0)
            g_va.append(s)

        ds_tr = lgb.Dataset(np.array(X_flat_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(feature_names))
        ds_va = lgb.Dataset(np.array(X_flat_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        preds = model.predict(np.array(X_flat_va))
        offset = 0
        for idx in val_list:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            offset += s
            ranked = np.argsort(-sc)
            top20 = [pools[idx][j] for j in ranked[:20]]
            if cases[idx]["gt"] in top20:
                pos = top20.index(cases[idx]["gt"])
                case_ndcg[idx] = 1.0 / np.log2(pos + 2)
            if gt_idx[idx] >= 0:
                gt_rank_pos = np.where(ranked == gt_idx[idx])[0]
                if len(gt_rank_pos) > 0:
                    case_gt_rank[idx] = int(gt_rank_pos[0]) + 1

    # Metrics
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]

    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
    h7_same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
    h7_diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0
    all_ndcg = float(np.mean(case_ndcg))

    h7_in_top20 = sum(1 for i in h7 if case_ndcg[i] > 0)
    h7_in_pool_ranked = [i for i in h7 if case_gt_rank[i] > 0]
    h7_median_gt_rank = float(np.median([case_gt_rank[i] for i in h7_in_pool_ranked])) if h7_in_pool_ranked else -1
    h7_same_median = float(np.median([case_gt_rank[i] for i in h7_same if case_gt_rank[i] > 0])) if h7_same else -1

    print(f"  {label:<35} h7={h7_ndcg:.5f}  all={all_ndcg:.5f}  "
          f"h7_same={h7_same_ndcg:.5f}  h7_diff={h7_diff_ndcg:.5f}  "
          f"h7_top20={h7_in_top20}  med_rank={h7_median_gt_rank:.0f}  "
          f"same_med={h7_same_median:.0f}")

    return {
        "h7": h7_ndcg, "all": all_ndcg,
        "h7_same": h7_same_ndcg, "h7_diff": h7_diff_ndcg,
        "h7_in_top20": h7_in_top20,
        "h7_median_gt_rank": h7_median_gt_rank,
        "h7_same_median_rank": h7_same_median,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R33b: Enhanced LambdaRank with Artist/Source Features")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R22B_OOF) as f:
        r22b_source = json.load(f)
    with open(Q3_LISTS) as f:
        q3_source = json.load(f)

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

    track_pop = build_popularity_stats()

    configs = [
        ("1_baseline", FEAT_BASE),
        ("2_+artist", FEAT_BASE + FEAT_ARTIST),
        ("3_+sources", FEAT_BASE + FEAT_SOURCES),
        ("4_+artist+sources", FEAT_BASE + FEAT_ARTIST + FEAT_SOURCES),
    ]

    print(f"\n{ts()} Running configs...")
    header = (f"  {'Config':<35} {'h7':>10}  {'all':>10}  {'h7_same':>10}  {'h7_diff':>10}  "
              f"{'h7_t20':>7}  {'med_rk':>7}  {'same_med':>8}")
    print(header)
    print(f"  {'-'*100}")

    results = {}
    for name, feat_names in configs:
        print(f"\n{ts()} Building features: {name} ({len(feat_names)} features)")
        X, gt_idx, sizes, pools = build_features(
            cases, payload, als_source, als_vecs, als_factors,
            als_track_to_idx, track_pop, r21_source,
            r22b_source, q3_source, feat_names)
        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")
        r = run_cv5_with_diagnostics(X, gt_idx, sizes, cases, sessions, pools, ta,
                                      feat_names, label=name)
        results[name] = r

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("R33b ENHANCED LAMBDARANK — SUMMARY")
    print(sep)
    base = results["1_baseline"]
    for name, r in results.items():
        dh7 = r["h7"] - base["h7"]
        print(f"  {name:<35} h7={r['h7']:.5f} ({dh7:+.5f})  "
              f"h7_same={r['h7_same']:.5f}  h7_diff={r['h7_diff']:.5f}  "
              f"med_rank={r['h7_median_gt_rank']:.0f}")

    # Gate
    print(f"\n{sep}")
    print("GATE CHECK")
    for name, r in results.items():
        if name == "1_baseline":
            continue
        dh7 = r["h7"] - base["h7"]
        g = dh7 >= 0.005
        print(f"  {name:<35} Δh7={dh7:+.5f} {'PASS' if g else 'FAIL'}")

    out_path = REPO / "exp" / "eval" / "expR33b_feature_reranker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
