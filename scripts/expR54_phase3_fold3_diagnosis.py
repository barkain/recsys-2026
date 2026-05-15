#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 3 fold 3 regression diagnosis.

Phase 3 full 5-fold integration showed:
  fold 0  Δh7 = +0.0295
  fold 1  Δh7 = +0.0105
  fold 2  Δh7 = +0.0045
  fold 3  Δh7 = -0.0169  ← REGRESSION
  fold 4  Δh7 = +0.0001

Question: is fold 3's regression a structural property of its dev split,
or a noise artifact?

Tests:
  1. Distributional comparison of fold 3 dev cases vs other folds (history
     length, GT artist freq, same/diff h7 ratio, played-track popularity, ...).
  2. Standalone Phase 3 vs Phase 2 retrieval on fold 3 h7 cases — does R54
     retrieve worse on fold 3 specifically, or only does ranking suffer?
  3. Per-h7-case attribution on fold 3: admitted-but-buried count + R54
     cosine for recovered vs buried.
  4. Compare fold 3 train pairs (= what Phase 3 fold-3 model sees) vs other
     folds — is the train set particularly noisy?
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
P2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
P3_OOF = REPO / "cache" / "r54" / "phase3_full" / "oof_r54_lists.json"
INTEGRATION_RES = REPO / "exp" / "eval" / "expR54_phase3_full5fold_integration.json"


def load_lists_with_scores(path):
    data = json.load(open(path))["lists"]
    return [[t for t, _ in case] for case in data], \
           [{t: float(s) for t, s in case} for case in data]


def hit_at_k(retrieval, gt, k):
    return gt in set(retrieval[:k])


def main():
    print("R54 Phase 3 fold 3 regression diagnosis")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]

    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)

    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    fold_h7 = {fi: [i for i in h7_idx if i in set(folds[fi].tolist())]
               for fi in range(5)}

    print("\n=== 1. Distributional comparison of h7 cases per fold ===")
    print(f"  {'fold':<6} {'n_h7':>5} {'same%':>7} {'diff%':>7} {'hist_len_mean':>13} {'gt_artist_freq_mean':>20}")
    artist_counter = Counter()
    for c in cases:
        a = ta.get(c["gt"], "")
        if a:
            artist_counter[a] += 1

    for fi in range(5):
        idxs = fold_h7[fi]
        n_h7 = len(idxs)
        same = sum(1 for i in idxs
                   if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in
                   {ta.get(t, "") for t in cases[i]["music_turns"]})
        same_pct = same / max(n_h7, 1) * 100
        hist_lens = [len(cases[i]["history"]) for i in idxs]
        gt_artist_freqs = [artist_counter[ta.get(cases[i]["gt"], "")] for i in idxs]
        print(f"  fold {fi:<2} {n_h7:>5} {same_pct:>6.1f}% {100-same_pct:>6.1f}% "
              f"{np.mean(hist_lens):>13.1f} {np.mean(gt_artist_freqs):>20.1f}")

    print("\n=== 2. Standalone retrieval comparison on fold 3 h7 ===")
    print("  Loading P2 + P3 OOF...")
    p2_lists, _ = load_lists_with_scores(P2_OOF)
    p3_lists, p3_scores = load_lists_with_scores(P3_OOF)
    r21_lists = json.load(open(R21_OOF))

    for fi in range(5):
        idxs = fold_h7[fi]
        p2_h300 = sum(1 for i in idxs if hit_at_k(p2_lists[i], cases[i]["gt"], 300))
        p3_h300 = sum(1 for i in idxs if hit_at_k(p3_lists[i], cases[i]["gt"], 300))
        p2_h20 = sum(1 for i in idxs if hit_at_k(p2_lists[i], cases[i]["gt"], 20))
        p3_h20 = sum(1 for i in idxs if hit_at_k(p3_lists[i], cases[i]["gt"], 20))
        r21_h300 = sum(1 for i in idxs if hit_at_k(r21_lists[i], cases[i]["gt"], 300))
        delta = p3_h300 - p2_h300
        marker = " ← f3" if fi == 3 else ""
        print(f"  fold {fi} h7: R21@300={r21_h300}  P2@300={p2_h300}  P3@300={p3_h300}  "
              f"ΔP3-P2={delta:+d}  P2@20={p2_h20}  P3@20={p3_h20}{marker}")

    print("\n=== 3. Where do fold 3 h7 GTs sit in P3 retrieval? ===")
    f3_h7 = fold_h7[3]
    rank_bins = {"1-20": 0, "21-50": 0, "51-100": 0, "101-300": 0, "missed_300": 0}
    cosines_in_top20 = []
    cosines_buried = []
    cosines_missed = []
    for i in f3_h7:
        gt = cases[i]["gt"]
        lst = p3_lists[i]
        cos = p3_scores[i].get(gt, None)
        if gt in lst[:20]:
            rank_bins["1-20"] += 1
            if cos is not None:
                cosines_in_top20.append(cos)
        elif gt in lst[:50]:
            rank_bins["21-50"] += 1
            if cos is not None:
                cosines_buried.append(cos)
        elif gt in lst[:100]:
            rank_bins["51-100"] += 1
            if cos is not None:
                cosines_buried.append(cos)
        elif gt in lst[:300]:
            rank_bins["101-300"] += 1
            if cos is not None:
                cosines_buried.append(cos)
        else:
            rank_bins["missed_300"] += 1
            if cos is not None:
                cosines_missed.append(cos)
    for b, c in rank_bins.items():
        pct = c / max(len(f3_h7), 1) * 100
        print(f"  {b:<14}: {c:>4} ({pct:.1f}%)")
    if cosines_in_top20:
        print(f"  cos in_top20  (n={len(cosines_in_top20)}): "
              f"mean={np.mean(cosines_in_top20):.4f}")
    if cosines_buried:
        print(f"  cos buried    (n={len(cosines_buried)}): "
              f"mean={np.mean(cosines_buried):.4f}")

    print("\n=== 4. Cross-fold P3-only and P2-only cases (h7) ===")
    for fi in range(5):
        idxs = fold_h7[fi]
        p3_only = sum(1 for i in idxs
                      if hit_at_k(p3_lists[i], cases[i]["gt"], 300) and
                      not hit_at_k(p2_lists[i], cases[i]["gt"], 300))
        p2_only = sum(1 for i in idxs
                      if hit_at_k(p2_lists[i], cases[i]["gt"], 300) and
                      not hit_at_k(p3_lists[i], cases[i]["gt"], 300))
        net = p3_only - p2_only
        marker = " ← f3" if fi == 3 else ""
        print(f"  fold {fi} h7: P3-only={p3_only}  P2-only={p2_only}  net={net:+d}{marker}")

    print("\n=== 5. R21 baseline pool overlap with P3 retrieval (fold 3 h7) ===")
    avg_overlap_by_fold = {}
    for fi in range(5):
        overlaps = []
        for i in fold_h7[fi]:
            p3_set = set(p3_lists[i][:300])
            r21_set = set(r21_lists[i][:300])
            overlap = len(p3_set & r21_set)
            overlaps.append(overlap)
        avg_overlap_by_fold[fi] = np.mean(overlaps)
        marker = " ← f3" if fi == 3 else ""
        print(f"  fold {fi} h7: avg P3-R21 overlap={np.mean(overlaps):.1f}/300{marker}")

    print("\n=== 6. Per-fold integration deltas (from artifact) ===")
    with open(INTEGRATION_RES) as f:
        integ = json.load(f)
    base_fold_h7 = integ["results"]["baseline_R39"]["fold_h7"]
    best_cfg = integ.get("best_cfg", "R39+R54_w1.0_feats")
    r54_fold_h7 = integ["results"][best_cfg]["fold_h7"]
    for fi in range(5):
        b = base_fold_h7.get(str(fi), 0)
        r = r54_fold_h7.get(str(fi), 0)
        d = r - b
        marker = " ← f3" if fi == 3 else ""
        print(f"  fold {fi}: baseline h7={b:.4f}  {best_cfg}={r:.4f}  Δ={d:+.4f}{marker}")

    print("\n=== 7. Phase 3 fold 3 training was 26,400 pairs ===")
    print("  But fold 3's TRAIN-side pairs are dev folds 0,1,2,4 ∪ 20K train-split.")
    print("  The 6,400 dev fold-train pairs differ from fold 0's train set.")
    print("  If fold 3's dev train set is qualitatively different (e.g. fewer")
    print("  cross-artist GTs), the resulting retriever may help in-fold but")
    print("  not generalize to fold 3's val distribution.")
    print("  This requires comparing the 6,400-case dev train per fold by:")
    print("    - same/diff GT ratio")
    print("    - artist concentration (Gini)")
    print("    - GT track popularity quartiles")
    print("    - history length distribution")

    print("\n=== 8. Dev train set composition per fold ===")
    print(f"  {'fold':<6} {'train_n':>8} {'same_GT%':>9} {'pop0%':>7} {'unique_artists':>15}")
    for fi in range(5):
        val_set = set(folds[fi].tolist())
        train_idx = [j for j in range(n) if j not in val_set]
        train_cases = [cases[j] for j in train_idx]
        same_train = sum(1 for c in train_cases
                          if ta.get(c["gt"], "") and
                          ta.get(c["gt"], "") in
                          {ta.get(t, "") for t in c["music_turns"]})
        same_pct = same_train / len(train_cases) * 100
        unique_artists = len({ta.get(c["gt"], "") for c in train_cases})
        marker = " ← f3" if fi == 3 else ""
        print(f"  fold {fi:<2} {len(train_idx):>8} {same_pct:>8.1f}% "
              f"{'-':>7} {unique_artists:>15}{marker}")


if __name__ == "__main__":
    main()
