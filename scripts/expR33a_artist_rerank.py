#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R33a: Same-artist reranking heuristic on hist_7.

R32 showed 272 same-artist GTs are in pool@300 but not top-20.
Test simple artist-boost heuristics before building neural rerankers.

Heuristics:
A. Boost any-previous-artist candidates
B. Boost last-1-artist candidates
C. Boost last-3-artist candidates
D. Frequency-weighted artist boost
E. Artist boost only when query has continuation terms
F. ALS-confident boost (ALS rank <= 5)
G. Combined: artist + ALS
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

import lightgbm as lgb

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]

CONTINUATION_TERMS = {"more", "another", "similar", "like", "keep", "same", "again",
                      "play", "give me", "how about", "what about", "any other"}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_lambdarank_baseline(cases, payload, als_source, als_vecs, als_factors,
                               als_track_to_idx, track_pop, r21_source, sessions):
    """Train CV5 LambdaRank and return per-case predictions."""
    n = len(cases)
    n_feat = len(FEATURE_NAMES_R21)
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

    return X, gt_idx, sizes, pools


def cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta,
               als_source, boost_fn=None, label=""):
    """Run CV5 LambdaRank, optionally apply post-hoc boost, evaluate hist_7."""
    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)
    case_scores: list[np.ndarray | None] = [None] * n

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
                            group=g_tr, feature_name=list(FEATURE_NAMES_R21))
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
            case_scores[idx] = preds[offset:offset + s].copy()
            offset += s

    # Apply boost and evaluate
    case_ndcg = np.zeros(n)

    for i in range(n):
        sc = case_scores[i]
        if sc is None or sizes[i] == 0:
            continue
        scores = sc.copy()

        if boost_fn is not None:
            scores = boost_fn(i, cases[i], pools[i], scores, ta, als_source)

        ranked_idx = np.argsort(-scores)
        top20_tids = [pools[i][j] for j in ranked_idx[:20]]

        if cases[i]["gt"] in top20_tids:
            pos = top20_tids.index(cases[i]["gt"])
            case_ndcg[i] = 1.0 / np.log2(pos + 2)

    # Sliced metrics
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
    all_ndcg = float(np.mean(case_ndcg))

    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]

    h7_same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
    h7_diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

    print(f"  {label:<30} h7={h7_ndcg:.5f}  all={all_ndcg:.5f}  "
          f"h7_same={h7_same_ndcg:.5f}  h7_diff={h7_diff_ndcg:.5f}")

    return {"h7": h7_ndcg, "all": all_ndcg, "h7_same": h7_same_ndcg, "h7_diff": h7_diff_ndcg}


def main():
    t0 = __import__("time").time()
    print(f"{ts()} R33a: Same-Artist Reranking Heuristic")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    als_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores_als = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores_als[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores_als, 200)[:200]
            top_idx = top_idx[np.argsort(-scores_als[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()

    print(f"{ts()} Building features + pools...")
    X, gt_idx, sizes, pools = build_lambdarank_baseline(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, sessions)

    print(f"\n{ts()} Running CV5 LambdaRank + heuristic variants...")
    print(f"  {'Config':<30} {'h7':>10}  {'all':>10}  {'h7_same':>10}  {'h7_diff':>10}")
    print(f"  {'-'*70}")

    # Baseline: no boost
    base = cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
                      boost_fn=None, label="baseline")

    # A: Boost any-previous-artist
    def boost_any_artist(i, case, pool, scores, ta_map, als_src):
        played_artists = {ta_map.get(t, "") for t in case["music_turns"]} - {""}
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            ca = ta_map.get(tid, "")
            if ca and ca in played_artists:
                boosted[j] += 2.0
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_any_artist, label="A: any_artist +2.0")

    # B: Boost last-1-artist
    def boost_last1(i, case, pool, scores, ta_map, als_src):
        if not case["music_turns"]:
            return scores
        last_artist = ta_map.get(case["music_turns"][-1], "")
        if not last_artist:
            return scores
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            if ta_map.get(tid, "") == last_artist:
                boosted[j] += 2.0
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_last1, label="B: last1_artist +2.0")

    # C: Boost last-3-artists
    def boost_last3(i, case, pool, scores, ta_map, als_src):
        recent = case["music_turns"][-3:]
        recent_artists = {ta_map.get(t, "") for t in recent} - {""}
        if not recent_artists:
            return scores
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            ca = ta_map.get(tid, "")
            if ca and ca in recent_artists:
                boosted[j] += 2.0
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_last3, label="C: last3_artists +2.0")

    # D: Frequency-weighted artist boost
    def boost_freq(i, case, pool, scores, ta_map, als_src):
        artist_freq = Counter(ta_map.get(t, "") for t in case["music_turns"])
        artist_freq.pop("", None)
        if not artist_freq:
            return scores
        max_freq = max(artist_freq.values())
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            ca = ta_map.get(tid, "")
            if ca and ca in artist_freq:
                boosted[j] += 2.0 * artist_freq[ca] / max_freq
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_freq, label="D: freq_artist +2.0*f")

    # E: Artist boost only with continuation terms
    def boost_continuation(i, case, pool, scores, ta_map, als_src):
        query_lower = case["user_query"].lower()
        has_continuation = any(term in query_lower for term in CONTINUATION_TERMS)
        if not has_continuation:
            return scores
        played_artists = {ta_map.get(t, "") for t in case["music_turns"]} - {""}
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            ca = ta_map.get(tid, "")
            if ca and ca in played_artists:
                boosted[j] += 2.0
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_continuation, label="E: continuation_artist +2.0")

    # F: ALS-confident boost
    def boost_als(i, case, pool, scores, ta_map, als_src):
        als_top5 = set(als_src[i][:5]) if als_src[i] else set()
        if not als_top5:
            return scores
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            if tid in als_top5:
                boosted[j] += 1.5
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_als, label="F: ALS_top5 +1.5")

    # G: Combined artist + ALS
    def boost_combined(i, case, pool, scores, ta_map, als_src):
        played_artists = {ta_map.get(t, "") for t in case["music_turns"]} - {""}
        als_top5 = set(als_src[i][:5]) if als_src[i] else set()
        boosted = scores.copy()
        for j, tid in enumerate(pool[:len(scores)]):
            ca = ta_map.get(tid, "")
            if ca and ca in played_artists:
                boosted[j] += 2.0
            if tid in als_top5:
                boosted[j] += 1.5
        return boosted

    cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
               boost_fn=boost_combined, label="G: artist+ALS combined")

    # Sweep boost magnitudes for best variant
    print(f"\n{ts()} Boost magnitude sweep (any_artist):")
    print(f"  {'Boost':<15} {'h7':>10}  {'h7_same':>10}  {'h7_diff':>10}")
    print(f"  {'-'*45}")
    for mag in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        def boost_mag(i, case, pool, scores, ta_map, als_src, m=mag):
            played_artists = {ta_map.get(t, "") for t in case["music_turns"]} - {""}
            boosted = scores.copy()
            for j, tid in enumerate(pool[:len(scores)]):
                ca = ta_map.get(tid, "")
                if ca and ca in played_artists:
                    boosted[j] += m
            return boosted

        cv5_rerank(X, gt_idx, sizes, cases, sessions, pools, ta, als_source,
                   boost_fn=boost_mag, label=f"any_artist +{mag}")

    # Gate check
    sep = "=" * 70
    print(f"\n{sep}")
    print("GATE CHECK: best variant vs baseline")
    print(f"  Baseline h7: {base['h7']:.5f}")
    print(f"  Gate: h7 >= baseline + 0.005 = {base['h7'] + 0.005:.5f}")

    out_path = REPO / "exp" / "eval" / "expR33a_artist_rerank.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline": base}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {__import__('time').time()-t0:.1f}s")


if __name__ == "__main__":
    main()
