#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R105 — Dev simulation of the CatalogDiv tail-rank guard (p_gt estimation).

We reproduce the PRODUCTION dev ranker (R84c selective routing: per-case use the
R84 sibling LR when the R54 sibling-LR top-1 margin is <LOW or >=HIGH, else R54;
LOW=0.5, HIGH=2.0 — the rule that produced the blind R92 p11 submission's ranking)
to get dev `lr_top20` lists for all 8000 cases, each with its known GT.

We then replay the variant-C tail-rank removal logic (edit ranks 11-20: a slot is
*removable* iff its track is a cross-CASE duplicate AND this is NOT its min-rank
occurrence) and measure, stratified by cross-case duplicate count:

    p_gt = (# removable tail-dup slots whose track == that case's GT)
           / (# removable tail-dup slots)

This is the dev proxy for "fraction of removed tail-dup slots that are the true GT",
i.e. the per-removed-slot probability that a CatalogDiv swap clips a real GT.

Guard (a) dup-count: only remove duplicates whose occurrence count is >= D.
We report p_gt and removable-slot count for D in {2,3,4} so the blind-side
expected GTs-clipped (= p_gt * blind_removable_swaps) can be computed.

Output: exp/eval/expR105_catdiv_devsim.json
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore
import numpy as np  # type: ignore

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_blind_submission import FEAT_R39_ALL, FEAT_ALL  # noqa
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa
    load_supporting_maps,
)

N_FOLDS = 5
TOP_K = 20
EDIT_LO, EDIT_HI = 11, 20  # variant C window
LOW_THR, HIGH_THR = 0.5, 2.0  # production selective routing rule

W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR105_catdiv_devsim.json"

FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def train_sibling_lr(case_features, train_idx, feat_key, feat_names):
    X, y, gt = [], [], []
    for i in train_idx:
        cf = case_features[i]
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf[feat_key][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=feat_names)
    return lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R105 dev-sim — production dev top20 + guard p_gt")
    print("=" * 70)

    payload, _, _, _ = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    print(f"  {n} dev cases")

    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_to_idx = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)

    print(f"{ts()} Loading case_features ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)...",
          flush=True)
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)

    # Per-fold sibling LRs, score eval folds (OOF) — both R54 and R84-only.
    print(f"{ts()} Training per-fold sibling LRs (OOF) ...", flush=True)
    per_case_scores: dict[int, dict[str, np.ndarray]] = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t_lr = time.time()
        lr_r54 = train_sibling_lr(case_features, train_idx, "feats_r54", list(FEAT_ALL))
        lr_r84 = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                  FEAT_NAMES_R84_ONLY)
        for i in eval_idx:
            cf = case_features[i]
            per_case_scores[i] = {
                "r54": lr_r54.predict(cf["feats_r54"]),
                "r84": lr_r84.predict(cf["feats_r84_only"]),
            }
        print(f"  fold {fold_k}: {time.time()-t_lr:.0f}s", flush=True)

    # Build production dev top20 lists via selective routing (LOW/HIGH).
    print(f"{ts()} Building production dev top20 (selective routing "
          f"LOW={LOW_THR}, HIGH={HIGH_THR}) ...", flush=True)
    dev_top20: list[list[str]] = [None] * n  # type: ignore
    gt_rank20: list[int] = [-1] * n
    n_r84 = 0
    for i in range(n):
        sd = per_case_scores[i]
        cf = case_features[i]
        s_r54 = sd["r54"]
        if len(s_r54) >= 2:
            sorted_s = np.sort(s_r54)[::-1]
            margin = float(sorted_s[0] - sorted_s[1])
        else:
            margin = 0.0
        use_r84 = (margin < LOW_THR) or (margin >= HIGH_THR)
        if use_r84:
            n_r84 += 1
        s = sd["r84"] if use_r84 else sd["r54"]
        order = np.argsort(-s, kind="mergesort")
        top20 = [cf["pool"][int(j)] for j in order[:TOP_K]]
        dev_top20[i] = top20
        if cf["gt_pos"] >= 0:
            p = np.where(order == cf["gt_pos"])[0]
            if len(p) and int(p[0]) < TOP_K:
                gt_rank20[i] = int(p[0]) + 1  # 1-based
    print(f"  routed_r84={n_r84}/{n}  routed_r54={n - n_r84}/{n}")

    gt = [c["gt"] for c in cases]
    in_top20 = sum(1 for r in gt_rank20 if r > 0)
    print(f"  dev GT-in-top20: {in_top20}/{n} ({in_top20/n:.4f})")

    # --- Cross-CASE duplicate bookkeeping over dev top20 lists ---
    allt = [t for L in dev_top20 for t in L]
    cnt = Counter(allt)
    occ = defaultdict(list)  # tid -> [(case_i, rank1based)]
    for ci, L in enumerate(dev_top20):
        for r, t in enumerate(L, start=1):
            occ[t].append((ci, r))
    kept = {t: min(os, key=lambda x: x[1]) for t, os in occ.items()}

    dup_slots_total = sum(c - 1 for c in cnt.values() if c > 1)
    print(f"  dev dup_slots (cross-case, all windows): {dup_slots_total}")

    # pool-rank (retrieval rank) of every track in its own case deep pool.
    # cf["pool"] is the per-case RRF-ordered deep candidate pool (len 300).
    pool_rank_of = []  # list[dict tid->1based pool rank] per case
    for i in range(n):
        pr = {t: r for r, t in enumerate(case_features[i]["pool"], start=1)}
        pool_rank_of.append(pr)

    # --- Replay variant-C removal logic, stratified by guards ---
    # A slot (ci, p) with track t in window [EDIT_LO, EDIT_HI] is REMOVABLE iff:
    #   cnt[t] >= D  (cross-case duplicate, count threshold D)  AND  kept[t] != (ci, p)
    # Guard (b) pool-rank: additionally require the removed track's own-case deep
    #   pool rank to be > PR (i.e. retrieval scored it low -> filler, not near-miss).
    # For each removable slot, record whether t == gt[ci]  (-> p_gt numerator).
    def measure(D: int, pr_min: int | None = None):
        removable = 0
        is_gt = 0
        for ci in range(n):
            L = dev_top20[ci]
            g = gt[ci]
            prc = pool_rank_of[ci]
            for p in range(EDIT_LO, EDIT_HI + 1):
                t = L[p - 1]
                if cnt[t] < D:
                    continue
                if kept[t] == (ci, p):
                    continue
                if pr_min is not None:
                    rk = prc.get(t)  # own-case deep pool rank (None = absent from pool)
                    # keep (don't remove) if the retriever ranked it high (<= pr_min);
                    # only remove when it is deep filler (rank > pr_min) or absent.
                    if rk is not None and rk <= pr_min:
                        continue
                removable += 1
                if t == g:
                    is_gt += 1
        p_gt = (is_gt / removable) if removable else 0.0
        return {"D": D, "pr_min": pr_min, "removable_slots": removable,
                "gt_in_removable": is_gt, "p_gt": p_gt}

    results = {}
    print(f"\n{ts()} === (a) p_gt by dup-count threshold D (window {EDIT_LO}-{EDIT_HI}) ===")
    for D in (2, 3, 4, 5):
        m = measure(D)
        results[f"dupcount_D{D}"] = m
        print(f"  D>={D}: removable={m['removable_slots']:5d}  "
              f"gt_in_removable={m['gt_in_removable']:3d}  p_gt={m['p_gt']*100:.4f}%")

    print(f"\n{ts()} === (b) p_gt by pool-rank guard (D>=2, keep if own-pool rank<=PR) ===")
    for PR in (10, 20, 30, 50, 100):
        m = measure(2, PR)
        results[f"poolrank_PR{PR}"] = m
        print(f"  PR>{PR}: removable={m['removable_slots']:5d}  "
              f"gt_in_removable={m['gt_in_removable']:3d}  p_gt={m['p_gt']*100:.4f}%")

    print(f"\n{ts()} === (c) combined: dup-count D>=3 AND pool-rank>PR ===")
    for PR in (20, 30, 50):
        m = measure(3, PR)
        results[f"combined_D3_PR{PR}"] = m
        print(f"  D>=3 & PR>{PR}: removable={m['removable_slots']:5d}  "
              f"gt_in_removable={m['gt_in_removable']:3d}  p_gt={m['p_gt']*100:.4f}%")

    out = {
        "experiment": "R105 dev-sim: production dev top20 + CatalogDiv guard p_gt",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "config": {
            "n_dev_cases": n, "edit_window": [EDIT_LO, EDIT_HI],
            "routing": {"rule": "R84c selective", "low_thr": LOW_THR, "high_thr": HIGH_THR},
            "routed_r84": n_r84, "routed_r54": n - n_r84,
        },
        "dev_gt_in_top20": {"count": in_top20, "frac": in_top20 / n},
        "dev_dup_slots_all_windows": dup_slots_total,
        "p_gt_guards": results,
        "note": ("Cross-CASE dup-count on 8000 dev lists is NOT scale-comparable to "
                 "cross-SUBMISSION dup-count on 80 blind lists (everything is a dup on "
                 "dev), so the dup-count threshold barely moves dev p_gt. The pool-rank "
                 "guard IS per-case and transferable; use guard (b)/(c) for the dev p_gt "
                 "estimate of the pool-rank-guarded blind variant."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\n{ts()} saved {OUT_JSON}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
