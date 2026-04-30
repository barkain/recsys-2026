#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""V3 error analysis: pool coverage vs ranking failures, by category.

Read-only diagnostic. No pipeline changes, no API calls.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, build_als
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
POOL_K = 200
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()

    print(f"{ts()} Building ALS source...", flush=True)
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            played_idx = [als_track_to_idx[t] for t in played if t in als_track_to_idx]
            for idx in played_idx:
                scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    print(f"{ts()} Building popularity...", flush=True)
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values())

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    # --- Build pools and per-source GT presence ---
    print(f"{ts()} Building pools and analyzing...", flush=True)

    per_case = []  # diagnostic record per case

    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        n_hist = len(played)

        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }

        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)
        gt_in_pool = gt in pool
        gt_pool_rank = pool.index(gt) + 1 if gt_in_pool else -1

        # Per-source GT presence and rank
        gt_sources = {}
        for sname, slist in src_lists.items():
            if gt in slist:
                gt_sources[sname] = slist.index(gt) + 1
            else:
                gt_sources[sname] = -1

        # GT track properties
        gt_artist = ta.get(gt, "")
        gt_pop = track_pop.get(gt, 0)
        gt_pop_decile = min(int(gt_pop / max_pop * 10), 9)

        # Same-artist as last played?
        last_artist = ta.get(played[-1], "") if played else ""
        same_artist = gt_artist != "" and gt_artist == last_artist

        rec = {
            "idx": i, "gt": gt, "n_hist": n_hist,
            "gt_in_pool": gt_in_pool, "gt_pool_rank": gt_pool_rank,
            "gt_sources": gt_sources,
            "gt_artist": gt_artist, "gt_pop": gt_pop,
            "gt_pop_decile": gt_pop_decile,
            "same_artist": same_artist,
            "n_sources_with_gt": sum(1 for v in gt_sources.values() if v > 0),
        }
        per_case.append(rec)

    # ==============================
    # SECTION 1: Pool coverage analysis
    # ==============================
    print(f"\n{'='*70}")
    print("SECTION 1: POOL COVERAGE (pool_k={})".format(POOL_K))
    print(f"{'='*70}")

    in_pool = [r for r in per_case if r["gt_in_pool"]]
    not_in_pool = [r for r in per_case if not r["gt_in_pool"]]
    print(f"  GT in pool: {len(in_pool)}/{n} ({len(in_pool)/n:.1%})")
    print(f"  GT NOT in pool: {len(not_in_pool)}/{n} ({len(not_in_pool)/n:.1%})")

    # By hist depth
    print(f"\n  By history depth:")
    print(f"    {'hist':8s} {'total':>6s} {'in_pool':>8s} {'rate':>6s} {'not_in':>7s}")
    for h in range(8):
        total = sum(1 for r in per_case if r["n_hist"] == h)
        hit = sum(1 for r in per_case if r["n_hist"] == h and r["gt_in_pool"])
        print(f"    hist_{h:1d}   {total:6d} {hit:8d} {hit/total:6.1%} {total-hit:7d}")

    # By popularity decile
    print(f"\n  By GT popularity decile:")
    print(f"    {'decile':8s} {'total':>6s} {'in_pool':>8s} {'rate':>6s}")
    for d in range(10):
        total = sum(1 for r in per_case if r["gt_pop_decile"] == d)
        hit = sum(1 for r in per_case if r["gt_pop_decile"] == d and r["gt_in_pool"])
        if total:
            print(f"    d{d:1d}       {total:6d} {hit:8d} {hit/total:6.1%}")

    # By same-artist
    print(f"\n  By same-artist:")
    for label, pred in [("same", True), ("diff", False)]:
        total = sum(1 for r in per_case if r["same_artist"] == pred and r["n_hist"] > 0)
        hit = sum(1 for r in per_case if r["same_artist"] == pred and r["n_hist"] > 0 and r["gt_in_pool"])
        if total:
            print(f"    {label:8s} {total:6d} {hit:8d} {hit/total:6.1%}")

    # For NOT in pool: which sources had the GT?
    print(f"\n  GT NOT in pool — source coverage of missed GTs:")
    src_coverage = Counter()
    no_source = 0
    for r in not_in_pool:
        sources_with = [s for s, v in r["gt_sources"].items() if v > 0]
        if not sources_with:
            no_source += 1
        for s in sources_with:
            src_coverage[s] += 1
    print(f"    No source has GT: {no_source}/{len(not_in_pool)} ({no_source/max(len(not_in_pool),1):.1%})")
    for s in ["A", "B", "C", "D", "F", "ALS"]:
        print(f"    {s:5s} has GT: {src_coverage.get(s,0):5d} ({src_coverage.get(s,0)/max(len(not_in_pool),1):.1%})")

    # For NOT in pool and GT in some source: why didn't RRF promote it?
    print(f"\n  GT in source but not in pool@200 — source rank distribution:")
    for s in ["A", "B", "C", "D", "F", "ALS"]:
        ranks = [r["gt_sources"][s] for r in not_in_pool if r["gt_sources"][s] > 0]
        if ranks:
            print(f"    {s:5s}: n={len(ranks):4d}  median_rank={np.median(ranks):.0f}  "
                  f"mean={np.mean(ranks):.0f}  <=50={sum(1 for x in ranks if x<=50)}  "
                  f"<=200={sum(1 for x in ranks if x<=200)}")

    # ==============================
    # SECTION 2: Ranking failures (GT in pool but not top-20)
    # ==============================
    print(f"\n{'='*70}")
    print("SECTION 2: RANKING FAILURES (GT in pool but would miss top-20)")
    print(f"{'='*70}")

    # Train LambdaRank on all data to get scores
    print(f"\n  {ts()} Training LambdaRank for scoring...", flush=True)

    from collections import Counter as _Counter
    n_feat = len(FEATURE_NAMES_V2)
    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools = []

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}
        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j,t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = _Counter(a for a in pool_artists if a)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank-1]
            row[0] = 1.0/rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
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
            for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                sr = src_rank[sn].get(tid)
                row[8+fi] = 1.0/sr if sr else 0.0
            for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                row[14+fi] = 1.0 if tid in src_rank[sn] else 0.0
            row[20] = sum(1 for sn in src_lists if tid in src_rank[sn])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None: row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]

    # Train on all data
    X_flat = X.reshape(-1, n_feat)
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0

    train_flat = [i * POOL_K + k for i in range(n) for k in range(int(sizes[i]))]
    g = np.array([int(sizes[i]) for i in range(n)], dtype=np.int32)
    dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat], group=g,
                         feature_name=FEATURE_NAMES_V2, free_raw_data=False)
    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }
    model = lgb.train(lgb_params, dtrain, num_boost_round=300)

    # Score all cases
    all_scores = model.predict(X_flat[train_flat])
    offset = 0
    for i in range(n):
        sz = int(sizes[i])
        sc = all_scores[offset:offset+sz]
        gt = gt_idx[i]
        if gt >= 0:
            gt_score = sc[gt]
            lr_rank = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(sz) < gt))) + 1
        else:
            lr_rank = -1
        per_case[i]["lr_rank"] = lr_rank
        per_case[i]["lr_hit20"] = gt >= 0 and lr_rank <= 20

        # Top-1 track info
        if sz > 0:
            top1_idx = np.argmax(sc)
            top1_tid = pools[i][top1_idx]
            per_case[i]["top1_pop"] = track_pop.get(top1_tid, 0)
            per_case[i]["top1_artist"] = ta.get(top1_tid, "")
        offset += sz

    # Analyze ranking failures
    rank_fail = [r for r in per_case if r["gt_in_pool"] and not r["lr_hit20"]]
    rank_hit = [r for r in per_case if r["gt_in_pool"] and r["lr_hit20"]]
    print(f"\n  GT in pool: {len(in_pool)}")
    print(f"  LR promotes to top-20: {len(rank_hit)} ({len(rank_hit)/max(len(in_pool),1):.1%})")
    print(f"  LR misses (rank > 20): {len(rank_fail)} ({len(rank_fail)/max(len(in_pool),1):.1%})")

    # Ranking failure by hist depth
    print(f"\n  Ranking failures by hist depth:")
    print(f"    {'hist':8s} {'in_pool':>8s} {'hit20':>6s} {'miss':>6s} {'conv%':>6s}")
    for h in range(8):
        ip = sum(1 for r in per_case if r["n_hist"] == h and r["gt_in_pool"])
        hit = sum(1 for r in per_case if r["n_hist"] == h and r.get("lr_hit20"))
        miss = ip - hit
        conv = hit / ip if ip else 0
        print(f"    hist_{h:1d}   {ip:8d} {hit:6d} {miss:6d} {conv:6.1%}")

    # GT rank distribution for misses
    miss_ranks = [r["lr_rank"] for r in rank_fail if r["lr_rank"] > 0]
    if miss_ranks:
        print(f"\n  Missed GT rank distribution:")
        print(f"    median={np.median(miss_ranks):.0f}  mean={np.mean(miss_ranks):.0f}  "
              f"21-50={sum(1 for x in miss_ranks if 21<=x<=50)}  "
              f"51-100={sum(1 for x in miss_ranks if 51<=x<=100)}  "
              f"100+={sum(1 for x in miss_ranks if x>100)}")

    # Pool rank of GT vs LR rank
    print(f"\n  Pool rank vs LR rank for missed GTs:")
    print(f"    {'pool_rank':>10s} {'count':>6s} {'median_lr':>10s}")
    for lo, hi in [(1, 20), (21, 50), (51, 100), (101, 200)]:
        subset = [r for r in rank_fail if lo <= r["gt_pool_rank"] <= hi]
        if subset:
            lr_ranks = [r["lr_rank"] for r in subset]
            print(f"    {lo:3d}-{hi:3d}     {len(subset):6d} {np.median(lr_ranks):10.0f}")

    # Popularity of missed GT vs hit GT
    print(f"\n  GT popularity: hit vs miss:")
    hit_pop = [r["gt_pop"] for r in rank_hit]
    miss_pop = [r["gt_pop"] for r in rank_fail]
    print(f"    Hit:  median={np.median(hit_pop):.0f}  mean={np.mean(hit_pop):.0f}")
    print(f"    Miss: median={np.median(miss_pop):.0f}  mean={np.mean(miss_pop):.0f}")

    # Same artist: hit vs miss
    print(f"\n  Same-artist GT: hit vs miss (hist>0 only):")
    for label, pred in [("same", True), ("diff", False)]:
        h = sum(1 for r in rank_hit if r["same_artist"] == pred and r["n_hist"] > 0)
        m = sum(1 for r in rank_fail if r["same_artist"] == pred and r["n_hist"] > 0)
        total = h + m
        if total:
            print(f"    {label:8s} hit={h:5d}  miss={m:5d}  conv={h/total:.1%}")

    # ==============================
    # SECTION 3: Missing GT source analysis
    # ==============================
    print(f"\n{'='*70}")
    print("SECTION 3: MISSING GTs — WHAT SOURCE WOULD HELP?")
    print(f"{'='*70}")

    # Categorize missing GTs
    categories = {
        "no_source": [],      # no source has GT anywhere
        "deep_text": [],      # B or C has GT but rank > 200
        "deep_behav": [],     # ALS or F has GT but rank > 200
        "shallow_miss": [],   # some source has GT <= 200 but RRF didn't promote
    }

    for r in not_in_pool:
        sources_with = {s: v for s, v in r["gt_sources"].items() if v > 0}
        if not sources_with:
            categories["no_source"].append(r)
        elif any(v <= 200 for v in sources_with.values()):
            categories["shallow_miss"].append(r)
        elif any(s in ("B", "C") for s in sources_with):
            categories["deep_text"].append(r)
        else:
            categories["deep_behav"].append(r)

    for cat, items in categories.items():
        print(f"\n  {cat}: {len(items)} ({len(items)/max(len(not_in_pool),1):.1%} of misses)")
        if items:
            pops = [r["gt_pop"] for r in items]
            hists = [r["n_hist"] for r in items]
            print(f"    pop: median={np.median(pops):.0f} mean={np.mean(pops):.0f}")
            print(f"    hist: median={np.median(hists):.0f} mean={np.mean(hists):.0f}")

    # nDCG impact estimate
    print(f"\n  nDCG impact if we could retrieve all missing GTs at rank 1:")
    current_ndcg = sum(1 for r in per_case if r.get("lr_hit20")) / n
    perfect_pool = (len(in_pool) + len(not_in_pool)) / n  # = 1.0
    for cat, items in categories.items():
        added = len(items) / n  # each adds 1.0 nDCG (rank 1)
        print(f"    {cat:15s}: +{added:.4f} nDCG (fixing {len(items)} cases)")

    # ==============================
    # SECTION 4: Ranked recommendations
    # ==============================
    print(f"\n{'='*70}")
    print("SECTION 4: RANKED NEXT ACTIONS BY EXPECTED IMPACT")
    print(f"{'='*70}")

    pool_miss_n = len(not_in_pool)
    rank_miss_n = len(rank_fail)
    no_source_n = len(categories["no_source"])
    shallow_n = len(categories["shallow_miss"])

    print(f"\n  Total misses: {pool_miss_n + rank_miss_n}")
    print(f"    Pool misses (GT not in top-200): {pool_miss_n}")
    print(f"      - No source has GT at all: {no_source_n}")
    print(f"      - Shallow miss (source has GT <= 200): {shallow_n}")
    print(f"    Rank misses (GT in pool, not top-20): {rank_miss_n}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
