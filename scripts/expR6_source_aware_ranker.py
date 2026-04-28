#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R6: Source-aware ranker — compact source-evidence features.

Ranks candidates from ABCD + F(CF-BPR) + G(session cooccur) using
per-source rank scores instead of text-overlap features.

No API. No blind. No LightGBM.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (
    build_or_load_payload,
    cv_folds,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from scripts.expR5_sequential_retrieval import SessionTransitionGraph, stage1_audit
from offline_retrieval_sweep import load_track_metadata
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever

POOL_K = 50
RRF_K = 20

# R6 feature set (11 features)
R6_FEATURES = [
    "rrf_score_abcd",           # RRF score from ABCD sources only
    "cf_rank_recip",            # 1/(cf_rank+1), 0 if not from CF
    "cooccur_rank_recip",       # 1/(cooccur_rank+1), 0 if not from G
    "a_prime_rank_recip",       # 1/(A'_rank+1)
    "bm25_lastmusic_rank_recip",# 1/(B_rank+1)
    "bm25_full_rank_recip",     # 1/(C_rank+1)
    "tracksim_rank_recip",      # 1/(D_rank+1)
    "source_count",             # number of sources containing this candidate
    "already_played",           # 1 if in played set
    "same_artist_last_played",  # artist match with last played
    "tag_overlap_last_played",  # Jaccard tag overlap with last played
]

R6_INIT = {
    "rrf_score_abcd": 3.0,
    "cf_rank_recip": 1.5,
    "cooccur_rank_recip": 1.0,
    "a_prime_rank_recip": 1.0,
    "bm25_lastmusic_rank_recip": 1.0,
    "bm25_full_rank_recip": 0.5,
    "tracksim_rank_recip": 0.5,
    "source_count": 0.5,
    "already_played": -3.0,
    "same_artist_last_played": 1.0,
    "tag_overlap_last_played": 0.3,
}

# Bounds for constrained Powell (L-BFGS-B)
R6_BOUNDS = [
    (0.0, 20.0),   # rrf_score_abcd
    (0.0, 20.0),   # cf_rank_recip
    (0.0, 20.0),   # cooccur_rank_recip
    (0.0, 20.0),   # a_prime_rank_recip
    (0.0, 20.0),   # bm25_lastmusic_rank_recip
    (0.0, 20.0),   # bm25_full_rank_recip
    (0.0, 20.0),   # tracksim_rank_recip
    (0.0, 10.0),   # source_count
    (-10.0, 0.0),  # already_played (must be non-positive)
    (0.0, 10.0),   # same_artist_last_played
    (0.0, 10.0),   # tag_overlap_last_played
]


def vec_ndcg(X, gt_idx, sizes, weights, idx):
    pool_axis = np.arange(X.shape[1])[None, :]
    valid_pool = pool_axis < sizes[idx, None]
    scores = X[idx] @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    gt = gt_idx[idx]
    has_gt = gt >= 0
    safe_gt = np.where(has_gt, gt, 0)
    gt_scores = scores[np.arange(len(idx)), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_bounded(X, gt_idx, sizes, train_idx):
    """L-BFGS-B fit with bounds."""
    init = np.array([R6_INIT[f] for f in R6_FEATURES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="L-BFGS-B", bounds=R6_BOUNDS,
                   options={"maxiter": 500, "ftol": 1e-6})
    return res.x, -float(res.fun)


def fit_powell(X, gt_idx, sizes, train_idx):
    """Unconstrained Powell for comparison."""
    init = np.array([R6_INIT[f] for f in R6_FEATURES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def eval_cv5(X, gt_idx, sizes, sessions, seeds, fit_fn):
    n = len(sessions)
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([i for i in range(n) if i not in held], dtype=np.int64)
            w, _ = fit_fn(X, gt_idx, sizes, train)
            fold_scores.append(vec_ndcg(X, gt_idx, sizes, w, fold))
        per_seed.append(float(np.mean(fold_scores)))
    return per_seed


def main():
    t0 = time.time()

    # Load sources
    print("Loading sources...", flush=True)
    payload = build_or_load_payload()
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    metadata = load_track_metadata()

    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]

    def ensure_meta(tids):
        for tid in tids:
            if tid in track_artist:
                continue
            meta = metadata.get(str(tid), {})
            artist = str(meta.get("artist_name", "")).lower().strip()
            raw_tags = meta.get("tag_list") or []
            tags = {str(t).lower().strip() for t in raw_tags if str(t).strip()} if isinstance(raw_tags, list) else set()
            track_artist[tid] = artist
            track_tags[tid] = tags

    # A' source
    print("Computing A' (qwen3)...", flush=True)
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    src_a = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = qwen_sim.vectors[a_idxs]
            sims = qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims.max(axis=1)
            exclude_a = {qwen_sim._id_to_idx[t] for t in played if t in qwen_sim._id_to_idx}
            cap = min(len(scores_a), 200 + len(exclude_a))
            cand = np.argpartition(-scores_a, cap - 1)[:cap]
            cand = cand[np.argsort(-scores_a[cand])]
            out = [qwen_sim.track_ids[int(ii)] for ii in cand if int(ii) not in exclude_a][:200]
            src_a.append(out)
        else:
            src_a.append([])

    # CF-BPR source
    print("Computing CF-BPR...", flush=True)
    cf_ids, cf_vecs, cf_idx = build_cfbpr_index()
    src_f = []
    for c in cases:
        played = c["music_turns"]
        result = cfbpr_max_recent(played, cf_vecs, cf_idx, cf_ids, 5, 200) if played else []
        ensure_meta(result)
        src_f.append(result)

    # Session co-occurrence source
    print("Building session transition graph...", flush=True)
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    session_sequences = {}
    for item in ds["train"]:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        tracks = [str(c["content"]).strip() for c in convs if c["role"] == "music"]
        session_sequences[sid] = tracks
    graph = SessionTransitionGraph(session_sequences, metadata)

    src_g = []
    for c in cases:
        played = c["music_turns"]
        result = graph.g_session_cooccur(played, 200) if played else []
        ensure_meta(result)
        src_g.append(result)

    # =====================================================================
    # Build R6 feature matrices
    # =====================================================================
    print("\n" + "=" * 70)
    print("R6: SOURCE-AWARE RANKER")
    print("=" * 70)

    # Pool: 6-source weighted RRF
    # Try multiple pool constructions
    pool_configs = [
        ("abcdfg_equal", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": 0.5}),
        ("abcdfg_cf_heavy", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 2.0, "G": 1.0}),
        ("abcdf_only", {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": 0.0}),
    ]

    seeds = [0, 1, 2, 3, 4]
    all_results = {}

    for pool_name, pool_weights in pool_configs:
        print(f"\n--- Pool: {pool_name} ---")
        X = np.zeros((n, POOL_K, len(R6_FEATURES)), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            # Build pool
            all_sources = {
                "A": src_a[i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": src_f[i], "G": src_g[i],
            }
            pool = weighted_rrf(all_sources, pool_weights, topk=POOL_K, k=RRF_K)
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])

            # Per-source rank maps
            a_ranks = {tid: r for r, tid in enumerate(src_a[i])}
            b_ranks = {tid: r for r, tid in enumerate(payload["src_b"][i])}
            c_ranks = {tid: r for r, tid in enumerate(payload["src_c"][i])}
            d_ranks = {tid: r for r, tid in enumerate(payload["src_d"][i])}
            f_ranks = {tid: r for r, tid in enumerate(src_f[i])}
            g_ranks = {tid: r for r, tid in enumerate(src_g[i])}

            # ABCD-only RRF score for each candidate
            abcd_scores = {}
            for sname, slist in [("A", src_a[i]), ("B", payload["src_b"][i]),
                                  ("C", payload["src_c"][i]), ("D", payload["src_d"][i])]:
                w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}[sname]
                for rank, tid in enumerate(slist, start=1):
                    abcd_scores[tid] = abcd_scores.get(tid, 0.0) + w / (RRF_K + rank)

            played = c["music_turns"]
            played_set = set(played)
            last_artist = track_artist.get(played[-1], "") if played else ""
            last_tags = track_tags.get(played[-1], set()) if played else set()

            for rank, tid in enumerate(pool[:POOL_K]):
                row = X[i, rank]
                row[0] = abcd_scores.get(tid, 0.0)
                row[1] = 1.0 / (f_ranks[tid] + 1) if tid in f_ranks else 0.0
                row[2] = 1.0 / (g_ranks[tid] + 1) if tid in g_ranks else 0.0
                row[3] = 1.0 / (a_ranks[tid] + 1) if tid in a_ranks else 0.0
                row[4] = 1.0 / (b_ranks[tid] + 1) if tid in b_ranks else 0.0
                row[5] = 1.0 / (c_ranks[tid] + 1) if tid in c_ranks else 0.0
                row[6] = 1.0 / (d_ranks[tid] + 1) if tid in d_ranks else 0.0
                # Source count
                count = 0
                if tid in a_ranks: count += 1
                if tid in b_ranks: count += 1
                if tid in c_ranks: count += 1
                if tid in d_ranks: count += 1
                if tid in f_ranks: count += 1
                if tid in g_ranks: count += 1
                row[7] = float(count)
                row[8] = 1.0 if tid in played_set else 0.0
                cand_artist = track_artist.get(tid, "")
                row[9] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
                cand_tags = track_tags.get(tid, set())
                if cand_tags or last_tags:
                    row[10] = len(cand_tags & last_tags) / len(cand_tags | last_tags)

        pool_hit = float(np.mean(gt_idx >= 0))
        med_rank = float(np.median(gt_idx[gt_idx >= 0] + 1)) if (gt_idx >= 0).any() else 999
        print(f"  pool_hit@50: {pool_hit:.4f} ({(gt_idx >= 0).sum()}/{n})")
        print(f"  median GT rank: {med_rank:.1f}")

        # Evaluate with both fit methods
        for fit_name, fit_fn in [("L-BFGS-B", fit_bounded), ("Powell", fit_powell)]:
            cv5 = eval_cv5(X, gt_idx, sizes, sessions, seeds, fit_fn)
            cv5_mean = float(np.mean(cv5))
            cv5_std = float(np.std(cv5, ddof=1))
            key = f"{pool_name}_{fit_name}"
            all_results[key] = {
                "pool": pool_name, "fit": fit_name,
                "pool_hit": pool_hit, "median_gt_rank": med_rank,
                "cv5_mean": cv5_mean, "cv5_std": cv5_std,
                "cv5_per_seed": cv5,
            }
            print(f"  {fit_name:10s}  CV5={cv5_mean:.4f} ± {cv5_std:.4f}  {cv5}")

        # Also compare: old 8-feature Powell on same pool
        from scripts.tune_postrank_v23 import INIT_WEIGHTS as OLD_INIT
        track_title_toks = payload["track_title_toks"]
        track_artist_toks = payload["track_artist_toks"]
        track_meta_toks = payload["track_meta_toks"]

        X_8f = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
        for i, c in enumerate(cases):
            all_sources_i = {
                "A": src_a[i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": src_f[i], "G": src_g[i],
            }
            pool = weighted_rrf(all_sources_i, pool_weights, topk=POOL_K, k=RRF_K)
            user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
            played = c["music_turns"]
            now_tok = tokens(user_msgs[-1]) if user_msgs else set()
            all_tok = tokens(" ".join(user_msgs))
            played_set = set(played)
            l_artist = track_artist.get(played[-1], "") if played else ""
            l_tags = track_tags.get(played[-1], set()) if played else set()
            prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set()))
                     for j,t in enumerate(reversed(played))]
            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = track_artist.get(tid, "")
                ct = track_tags.get(tid, set())
                row = X_8f[i, rank-1]
                row[0] = 1.0/rank
                row[1] = 1.0 if ca and ca == l_artist else 0.0
                if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
                row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
                row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
                row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
                row[6] = 1.0 if tid in played_set else 0.0
                rec = 0.0
                for wd, pa, pt in prior:
                    am = 1.0 if ca and ca == pa else 0.0
                    tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                    rec += wd * (am + tm)
                row[7] = rec

        from scripts.r3_confirm_400_deterministic import fit_weights as fit_8f
        from scripts.r3_confirm_400_deterministic import vec_ndcg as ndcg_8f
        cv5_8f = eval_cv5(X_8f, gt_idx, sizes, sessions, seeds,
                          lambda X, g, s, t: fit_8f(X, g, s, t))
        cv5_8f_mean = float(np.mean(cv5_8f))
        cv5_8f_std = float(np.std(cv5_8f, ddof=1))
        key_8f = f"{pool_name}_8f_Powell"
        all_results[key_8f] = {
            "pool": pool_name, "fit": "8f_Powell",
            "pool_hit": pool_hit, "median_gt_rank": med_rank,
            "cv5_mean": cv5_8f_mean, "cv5_std": cv5_8f_std,
            "cv5_per_seed": cv5_8f,
        }
        print(f"  {'8f_Powell':10s}  CV5={cv5_8f_mean:.4f} ± {cv5_8f_std:.4f}  {cv5_8f}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    best_key = max(all_results, key=lambda k: all_results[k]["cv5_mean"])
    best = all_results[best_key]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"F1 baseline reference:  CV5 ≈ 0.1700 (5-seed)")
    print(f"Best R6 config:         CV5={best['cv5_mean']:.4f} ± {best['cv5_std']:.4f}  [{best_key}]")
    print(f"  pool_hit={best['pool_hit']:.4f}  med_rank={best['median_gt_rank']:.1f}")

    for k in sorted(all_results, key=lambda k: all_results[k]["cv5_mean"], reverse=True):
        r = all_results[k]
        print(f"  {k:40s}  CV5={r['cv5_mean']:.4f} ± {r['cv5_std']:.4f}  hit={r['pool_hit']:.4f}")

    if best["cv5_mean"] >= 0.178:
        verdict = "PASS"
    elif best["cv5_mean"] >= 0.174:
        verdict = "PROMISING"
    elif best["cv5_mean"] <= 0.170:
        verdict = "FAIL"
    else:
        verdict = "WEAK"

    print(f"\nGATE VERDICT: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR6_source_aware_ranker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "best": best_key, "verdict": verdict,
                    "elapsed": elapsed, "features": R6_FEATURES}, f, indent=2)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
