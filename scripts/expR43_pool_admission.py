#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R43: Pool admission analysis for Bucket D cases.

Bucket D = GT exists in at least one retrieval source but gets dropped
by RRF@300.  Analyse which sources contain these GTs and at what ranks,
then simulate admission policies (larger pool, guaranteed source slots)
and evaluate each with CV5 LambdaRank using R39 features.
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
MAX_POOL = 600  # max pool size for admission policies
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1",
    "same_album_last3",
    "same_album_any",
    "album_history_count",
    "pool_same_album_count",
]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM


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


def build_features_for_pool(
    pool: list[str],
    case: dict,
    src_rank: dict[str, dict[str, int]],
    payload: dict,
    ta: dict,
    tt: dict,
    ttl: dict,
    tat: dict,
    tmt: dict,
    track_pop: dict,
    max_pop: float,
    track_album: dict,
    als_vecs_i,
    als_factors,
    als_track_to_idx: dict,
    r21_source_i: list[str],
    n_feat: int,
    n_feat_base: int,
) -> np.ndarray:
    """Build feature matrix for a single case's pool. Returns (pool_size, n_feat)."""
    pool_size = len(pool)
    X = np.zeros((pool_size, n_feat), dtype=np.float64)

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
    sv = als_vecs_i
    pool_artists = [ta.get(tid, "") for tid in pool]
    artist_counts = Counter(a for a in pool_artists if a)
    r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source_i[:300])}

    # Album precomputation
    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
    all_albums = [track_album.get(t, "") for t in played]
    album_hist_counts = Counter(a for a in all_albums if a)

    for rank, tid in enumerate(pool, start=1):
        ca = ta.get(tid, "")
        ct = tt.get(tid, set())
        row = X[rank - 1]

        # Base 29 features (identical to R39a)
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

        # Album features
        c_album = track_album.get(tid, "")
        row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
        row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
        row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
        row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
        pool_album_count = sum(1 for tid2 in pool if track_album.get(tid2, "") == c_album) if c_album else 0
        row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    return X


def run_cv5_lambdarank(X_all, gt_idx, sizes, folds, feat_names, n, cases, ta):
    """Run CV5 LambdaRank, return per-case nDCG array."""
    case_ndcg = np.zeros(n)
    for fi in range(5):
        val_set = set(folds[fi].tolist())
        tr = [j for j in range(n) if j not in val_set]
        va = sorted(val_set)
        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in tr:
            s = int(sizes[idx])
            for k in range(s):
                X_tr.append(X_all[idx][k])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(sizes[idx])
            for k in range(s):
                X_va.append(X_all[idx][k])
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
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)
    return case_ndcg


def main():
    t0 = time.time()
    print(f"{ts()} R43: Pool Admission Analysis for Bucket D")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data (same as R39a)
    # ---------------------------------------------------------------
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

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source: list[list[str]] = []
    als_vecs: list = []
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

    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)
    n_feat = len(ALL_FEAT)
    n_feat_base = len(FEAT_BASE)

    # ---------------------------------------------------------------
    # Step 1: Build baseline RRF pools and identify hist_7
    # ---------------------------------------------------------------
    print(f"\n{ts()} Step 1: Building baseline RRF@{POOL_K} pools...", flush=True)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"  hist_7 cases: {len(h7)}")

    # Build source lists for all cases (needed for all policies)
    all_src_lists: list[dict[str, list[str]]] = []
    for i in range(n):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        all_src_lists.append(src_lists)

    # Baseline pools
    baseline_pools: list[list[str]] = []
    baseline_gt_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        pool = weighted_rrf(all_src_lists[i], SW, topk=POOL_K, k=RRF_K)
        baseline_pools.append(pool)
        if cases[i]["gt"] in pool:
            baseline_gt_idx[i] = pool.index(cases[i]["gt"])

    pool_hit_baseline = float(np.mean(baseline_gt_idx[h7] >= 0))
    print(f"  baseline pool_hit@{POOL_K} (h7): {pool_hit_baseline:.4f}", flush=True)

    # ---------------------------------------------------------------
    # Step 2: Identify Bucket D cases
    # ---------------------------------------------------------------
    print(f"\n{ts()} Step 2: Identifying Bucket D cases...")
    bucket_d_cases: list[int] = []
    d_source_info: list[dict] = []  # per D case: which sources, ranks

    for i in h7:
        gt = cases[i]["gt"]
        # GT already in pool -> not bucket D
        if baseline_gt_idx[i] >= 0:
            continue
        # Check if GT in any source
        src_containing_gt = {}
        for sname, slist in all_src_lists[i].items():
            if gt in slist:
                src_containing_gt[sname] = slist.index(gt) + 1  # 1-indexed rank
        if src_containing_gt:
            bucket_d_cases.append(i)
            best_src = min(src_containing_gt, key=lambda k: src_containing_gt[k])
            d_source_info.append({
                "case_idx": i,
                "sources": src_containing_gt,
                "best_rank": min(src_containing_gt.values()),
                "best_source": best_src,
            })

    print(f"  Bucket D cases: {len(bucket_d_cases)}")

    # ---------------------------------------------------------------
    # Step 3: Source x rank analysis for D cases
    # ---------------------------------------------------------------
    print(f"\n{ts()} Step 3: Source x rank analysis for Bucket D")
    print("=" * 70)

    source_names = ["A", "B", "C", "D", "F", "ALS", "R21"]
    header = f"{'Source':>6s} | {'D w/ GT':>7s} | {'median':>6s} | {'mean':>6s} | {'top20':>5s} | {'top50':>5s} | {'top100':>6s}"
    print(header)
    print("-" * len(header))

    for sname in source_names:
        ranks = [info["sources"][sname] for info in d_source_info if sname in info["sources"]]
        if not ranks:
            print(f"{sname:>6s} | {0:>7d} | {'—':>6s} | {'—':>6s} | {0:>5d} | {0:>5d} | {0:>6d}")
            continue
        ranks_arr = np.array(ranks)
        cnt = len(ranks)
        median_r = float(np.median(ranks_arr))
        mean_r = float(np.mean(ranks_arr))
        top20 = int(np.sum(ranks_arr <= 20))
        top50 = int(np.sum(ranks_arr <= 50))
        top100 = int(np.sum(ranks_arr <= 100))
        print(f"{sname:>6s} | {cnt:>7d} | {median_r:>6.0f} | {mean_r:>6.1f} | {top20:>5d} | {top50:>5d} | {top100:>6d}")

    # Best source distribution
    best_source_counts = Counter(info["best_source"] for info in d_source_info)
    print("\n  Best source distribution (which source had GT at lowest rank):")
    for sname in source_names:
        cnt = best_source_counts.get(sname, 0)
        if cnt > 0:
            print(f"    {sname:>4s}: {cnt:>3d} ({cnt/len(bucket_d_cases)*100:.1f}%)")

    best_ranks = [info["best_rank"] for info in d_source_info]
    print(f"\n  Best rank across all sources: median={np.median(best_ranks):.0f}, mean={np.mean(best_ranks):.1f}")
    print(f"  Best rank <= 50: {sum(1 for r in best_ranks if r <= 50)}")
    print(f"  Best rank <= 100: {sum(1 for r in best_ranks if r <= 100)}")
    print(f"  Best rank <= 200: {sum(1 for r in best_ranks if r <= 200)}")

    # ---------------------------------------------------------------
    # Step 4: Simulate admission policies
    # ---------------------------------------------------------------
    print(f"\n{ts()} Step 4: Simulating admission policies...")
    print("=" * 70)

    # Define admission policies
    policies = {
        "baseline": {
            "desc": "RRF@300 (R39 baseline)",
            "pool_fn": lambda i: weighted_rrf(all_src_lists[i], SW, topk=POOL_K, k=RRF_K),
        },
        "pool_k=400": {
            "desc": "RRF@400",
            "pool_fn": lambda i: weighted_rrf(all_src_lists[i], SW, topk=400, k=RRF_K),
        },
        "pool_k=500": {
            "desc": "RRF@500",
            "pool_fn": lambda i: weighted_rrf(all_src_lists[i], SW, topk=500, k=RRF_K),
        },
        "guar_R21_top50": {
            "desc": "RRF@300 + guarantee R21 top 50",
            "guarantee": {"R21": 50},
        },
        "guar_BM25_top30": {
            "desc": "RRF@300 + guarantee best-of-B/C top 30",
            "guarantee_bm25": 30,
        },
        "guar_ALS_top50": {
            "desc": "RRF@300 + guarantee ALS top 50",
            "guarantee": {"ALS": 50},
        },
        "guar_all_top30": {
            "desc": "RRF@300 + guarantee top 30 from R21, B, C, ALS",
            "guarantee": {"R21": 30, "B": 30, "C": 30, "ALS": 30},
        },
    }

    # Run baseline first to get reference nDCG
    results = {}

    for pi, (policy_name, policy_cfg) in enumerate(policies.items()):
        print(f"\n{ts()} Policy {pi+1}/{len(policies)}: {policy_name} — {policy_cfg['desc']}", flush=True)

        # Build pools for this policy
        policy_pools: list[list[str]] = []
        policy_gt_idx = np.full(n, -1, dtype=np.int64)
        policy_sizes = np.zeros(n, dtype=np.int64)

        for i in range(n):
            if "pool_fn" in policy_cfg:
                pool = policy_cfg["pool_fn"](i)
            else:
                # Start with baseline RRF@300
                pool = list(baseline_pools[i])
                pool_set = set(pool)

                if "guarantee" in policy_cfg:
                    for src_name, top_k in policy_cfg["guarantee"].items():
                        src_list = all_src_lists[i].get(src_name, [])
                        for tid in src_list[:top_k]:
                            if tid not in pool_set:
                                pool.append(tid)
                                pool_set.add(tid)

                if "guarantee_bm25" in policy_cfg:
                    top_k = policy_cfg["guarantee_bm25"]
                    # Merge B and C by taking best rank
                    b_list = all_src_lists[i].get("B", [])
                    c_list = all_src_lists[i].get("C", [])
                    # Build combined BM25 ranked list: RRF of B and C only
                    bm25_combined = weighted_rrf({"B": b_list, "C": c_list},
                                                 {"B": 1.0, "C": 1.0},
                                                 topk=top_k, k=RRF_K)
                    for tid in bm25_combined:
                        if tid not in pool_set:
                            pool.append(tid)
                            pool_set.add(tid)

            policy_pools.append(pool)
            policy_sizes[i] = len(pool)
            gt = cases[i]["gt"]
            if gt in pool:
                policy_gt_idx[i] = pool.index(gt)

        pool_hit = float(np.mean(policy_gt_idx[h7] >= 0))
        max_pool_size = int(np.max(policy_sizes))
        mean_pool_size = float(np.mean(policy_sizes))
        print(f"  pool_hit (h7): {pool_hit:.4f}  max_pool: {max_pool_size}  mean_pool: {mean_pool_size:.0f}", flush=True)

        # Build features for all cases
        print("  Building features...", flush=True)
        X_all: list[np.ndarray] = []
        for i in range(n):
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{n} cases...", flush=True)
            pool = policy_pools[i]
            src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                        for sn, sl in all_src_lists[i].items()}
            feat_mat = build_features_for_pool(
                pool=pool, case=cases[i], src_rank=src_rank,
                payload=payload, ta=ta, tt=tt, ttl=ttl, tat=tat, tmt=tmt,
                track_pop=track_pop, max_pop=max_pop, track_album=track_album,
                als_vecs_i=als_vecs[i], als_factors=als_factors,
                als_track_to_idx=als_track_to_idx, r21_source_i=r21_source[i],
                n_feat=n_feat, n_feat_base=n_feat_base,
            )
            X_all.append(feat_mat)

        # Run CV5 LambdaRank
        print("  Running CV5 LambdaRank...", flush=True)
        case_ndcg = run_cv5_lambdarank(
            X_all, policy_gt_idx, policy_sizes, folds, ALL_FEAT, n, cases, ta,
        )

        # Metrics
        h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7 if i not in set(h7_same)]

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
        same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
        diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

        print(f"  h7={h7_ndcg:.5f}  same={same_ndcg:.5f}  diff={diff_ndcg:.5f}", flush=True)

        results[policy_name] = {
            "pool_hit": pool_hit,
            "h7_ndcg": h7_ndcg,
            "same": same_ndcg,
            "diff": diff_ndcg,
            "max_pool": max_pool_size,
            "mean_pool": mean_pool_size,
            "case_ndcg": case_ndcg,
        }

    # ---------------------------------------------------------------
    # Step 5: Summary table
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("R43 POOL ADMISSION — SUMMARY TABLE")
    print(sep)

    base_ndcg_arr = results["baseline"]["case_ndcg"]
    base_h7_ndcg = results["baseline"]["h7_ndcg"]

    header = f"{'Policy':<22s} | {'pool_hit':>8s} | {'h7_ndcg':>8s} | {'dh7':>7s} | {'same':>7s} | {'diff':>7s} | {'recov':>5s} | {'lost':>4s} | {'net':>4s}"
    print(header)
    print("-" * len(header))

    for policy_name in policies:
        r = results[policy_name]
        dh7 = r["h7_ndcg"] - base_h7_ndcg

        # Recovered: cases where baseline nDCG=0 but policy nDCG>0
        # Lost: cases where baseline nDCG>0 but policy nDCG=0
        recovered = 0
        lost = 0
        for i in h7:
            b = base_ndcg_arr[i]
            p = r["case_ndcg"][i]
            if b == 0 and p > 0:
                recovered += 1
            elif b > 0 and p == 0:
                lost += 1

        net = recovered - lost
        print(f"{policy_name:<22s} | {r['pool_hit']:>8.4f} | {r['h7_ndcg']:>8.5f} | {dh7:>+7.5f} | {r['same']:>7.5f} | {r['diff']:>7.5f} | {recovered:>5d} | {lost:>4d} | {net:>+4d}")

    # ---------------------------------------------------------------
    # Save results (strip numpy arrays for JSON)
    # ---------------------------------------------------------------
    out_data = {}
    for policy_name, r in results.items():
        out_data[policy_name] = {
            "pool_hit": r["pool_hit"],
            "h7_ndcg": r["h7_ndcg"],
            "same": r["same"],
            "diff": r["diff"],
            "max_pool": r["max_pool"],
            "mean_pool": r["mean_pool"],
        }
    out_data["bucket_d_count"] = len(bucket_d_cases)
    out_data["d_source_info_summary"] = {
        sname: {
            "count": sum(1 for info in d_source_info if sname in info["sources"]),
            "median_rank": float(np.median([info["sources"][sname] for info in d_source_info if sname in info["sources"]])) if any(sname in info["sources"] for info in d_source_info) else None,
        }
        for sname in source_names
    }

    out_path = REPO / "exp" / "eval" / "expR43_pool_admission.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
