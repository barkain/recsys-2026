#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Phase A1.1: Ablation-based CV5 bounds for each embedding-based source.

Uses the cached R12 all-turns payload (8000 cases) for efficiency.
No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import cv_folds
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import weighted_rrf

POOL_K = 50
RRF_K = 20
R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"


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


def fit_powell(X, gt_idx, sizes, train_idx):
    init = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def build_features(payload, source_weights):
    cases = payload["cases"]
    n = len(cases)
    X = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        sources = {}
        if source_weights.get("A", 0) > 0: sources["A"] = payload["src_a"][i]
        if source_weights.get("B", 0) > 0: sources["B"] = payload["src_b"][i]
        if source_weights.get("C", 0) > 0: sources["C"] = payload["src_c"][i]
        if source_weights.get("D", 0) > 0: sources["D"] = payload["src_d"][i]
        if source_weights.get("F", 0) > 0: sources["F"] = payload["src_f"][i]
        if source_weights.get("G", 0) > 0: sources["G"] = payload["src_g"][i]

        pool = weighted_rrf(sources, source_weights, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j,t in enumerate(reversed(played))]
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

    return X, gt_idx, sizes


def eval_cv5(X, gt_idx, sizes, sessions, seeds):
    n = len(sessions)
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_powell(X, gt_idx, sizes, train)
            fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
        per_seed.append(float(np.mean(fold_sc)))
    return per_seed


def main():
    t0 = time.time()

    print("Loading R12 cached payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases, {len(set(sessions))} sessions")

    seeds = [0, 1, 2, 3, 4]

    configs = {
        "Full (A'+B+C+D+F)": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
        "No F (CF-BPR)":     {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
        "No A' (qwen3 max)": {"B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
        "No D (qwen3 nbrs)": {"A": 1.0, "B": 1.0, "C": 1.0, "F": 1.0},
        "No A' no D":        {"B": 1.0, "C": 1.0, "F": 1.0},
        "B+C only (BM25)":   {"B": 1.0, "C": 1.0},
    }

    results = {}
    for name, weights in configs.items():
        print(f"\n  {name}...", flush=True)
        X, gt_idx, sizes = build_features(payload, weights)
        hit50 = float(np.mean(gt_idx >= 0))
        cv5 = eval_cv5(X, gt_idx, sizes, sessions, seeds)
        cv5_mean = float(np.mean(cv5))
        cv5_std = float(np.std(cv5, ddof=1))
        results[name] = {"cv5": cv5_mean, "cv5_std": cv5_std, "hit50": hit50, "cv5_seeds": cv5}
        print(f"    CV5={cv5_mean:.4f} ± {cv5_std:.4f}  hit@50={hit50:.4f}")

    # Print summary table
    full_cv5 = results["Full (A'+B+C+D+F)"]["cv5"]
    print(f"\n{'='*60}")
    print(f"{'Config':25s} {'CV5':>7s} {'Δ':>7s} {'hit@50':>7s}")
    for name, r in results.items():
        delta = r["cv5"] - full_cv5
        print(f"  {name:25s} {r['cv5']:7.4f} {delta:+7.4f} {r['hit50']:7.4f}")

    # Check leakage flag
    print(f"\nLeakage flag threshold: Δ > 0.02 for any single source removal")
    for name, r in results.items():
        if name == "Full (A'+B+C+D+F)":
            continue
        delta = full_cv5 - r["cv5"]
        if delta > 0.02:
            print(f"  WARNING: {name} drops CV5 by {delta:.4f} — flagged for further audit")
        else:
            print(f"  OK: {name} drops CV5 by {delta:.4f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expA1_ablation_cv5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
