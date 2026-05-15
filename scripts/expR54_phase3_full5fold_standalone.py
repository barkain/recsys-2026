#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 3 full 5-fold OOF standalone aggregate.

Reads OOF R54-Phase3 retrieval lists (assembled from all 5 folds).
Compares against R21 OOF AND Phase 2 OOF.

Fails fast if expected artifacts are missing.

Reports:
- hit@20/100/200/300 overall
- h7 hit@200/300
- same/diff h7 hit@300
- unique vs R21 OOF and vs Phase 2 OOF
- lost vs R21 OOF and vs Phase 2 OOF
- Bucket D/E retrieved
- fold-by-fold hit@200
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

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
P2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
P3_OOF = REPO / "cache" / "r54" / "phase3_full" / "oof_r54_lists.json"

RRF_K = 20
POOL_K = 300
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def fail_fast_check():
    """Validate all required artifacts exist before doing any work."""
    missing = []
    for label, path in [("R12 payload", R12_CACHE), ("R21 OOF", R21_OOF),
                         ("Phase 2 OOF", P2_OOF), ("Phase 3 OOF", P3_OOF)]:
        if not path.exists():
            missing.append(f"  {label}: {path}")
    if missing:
        print("MISSING ARTIFACTS — CANNOT RUN:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)


def normalize_r54_lists(raw_lists, n_expected, label):
    """Validate and extract tids from (tid, score) tuple format."""
    assert len(raw_lists) == n_expected, \
        f"{label}: length {len(raw_lists)} != expected {n_expected}"
    lists = []
    for i, case_lists in enumerate(raw_lists):
        assert case_lists is not None and len(case_lists) > 0, \
            f"{label}: case {i} has empty/None list"
        for item_idx, item in enumerate(case_lists):
            assert isinstance(item, (list, tuple)) and len(item) == 2, \
                f"{label}: case {i} item {item_idx} not (tid, score)"
            tid, score = item
            assert isinstance(tid, str) and np.isfinite(score), \
                f"{label}: case {i} item {item_idx} invalid"
        lists.append([t for t, _ in case_lists])
    return lists


def hit_at_k(retrieval, gt, k):
    return gt in set(retrieval[:k])


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 3 full 5-fold standalone aggregate")
    print("=" * 70)

    fail_fast_check()

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]

    print(f"{ts()} Loading R21 OOF...")
    r21_lists = json.load(open(R21_OOF))
    assert len(r21_lists) == n, f"R21 OOF length {len(r21_lists)} != {n}"

    print(f"{ts()} Loading Phase 2 OOF...")
    p2_data = json.load(open(P2_OOF))
    p2_lists = normalize_r54_lists(p2_data["lists"], n, "Phase 2")

    print(f"{ts()} Loading Phase 3 OOF...")
    p3_data = json.load(open(P3_OOF))
    p3_lists = normalize_r54_lists(p3_data["lists"], n, "Phase 3")
    print(f"  R54 lists validated for {n} cases")

    folds = grouped_session_folds(sessions, seed=0)
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7_idx if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7_idx if i not in set(h7_same)]
    print(f"  h7 cases: {len(h7_idx)}  same={len(h7_same)}  diff={len(h7_diff)}")

    print(f"{ts()} Building ALS for bucket classification...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
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
        else:
            als_source.append([])

    print(f"{ts()} Classifying h7 buckets (R39 baseline)...")
    buckets = {}
    for i in h7_idx:
        gt = cases[i]["gt"]
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_lists[i],
        }
        pool = weighted_rrf(src_lists, SW_BASE, topk=POOL_K, k=RRF_K)
        if gt in pool:
            buckets[i] = "in_pool"
        else:
            union = set()
            for sl in src_lists.values():
                union.update(sl[:300])
            buckets[i] = "D" if gt in union else "E"
    bucket_counts = Counter(buckets.values())
    print(f"  Baseline (R39): {dict(bucket_counts)}")

    # Aggregate hits
    print(f"\n{ts()} === AGGREGATE HITS ===")
    r54_hits = {"@20": 0, "@100": 0, "@200": 0, "@300": 0}
    p2_hits = {"@20": 0, "@100": 0, "@200": 0, "@300": 0}
    r21_hits = {"@20": 0, "@100": 0, "@200": 0, "@300": 0}
    for i in range(n):
        gt = cases[i]["gt"]
        for label, k in [("@20", 20), ("@100", 100), ("@200", 200), ("@300", 300)]:
            if hit_at_k(p3_lists[i], gt, k):
                r54_hits[label] += 1
            if hit_at_k(p2_lists[i], gt, k):
                p2_hits[label] += 1
            if hit_at_k(r21_lists[i], gt, k):
                r21_hits[label] += 1

    print(f"  {'k':<8} {'P3':>10} {'P2':>10} {'R21':>10} {'P3-P2':>8} {'P3-R21':>8}")
    for k in ["@20", "@100", "@200", "@300"]:
        print(f"  hit{k:<6} {r54_hits[k]:>5}/{n} ({r54_hits[k]/n:.4f})  "
              f"{p2_hits[k]:>5}/{n} ({p2_hits[k]/n:.4f})  "
              f"{r21_hits[k]:>5}/{n} ({r21_hits[k]/n:.4f})  "
              f"{r54_hits[k]-p2_hits[k]:>+8d} {r54_hits[k]-r21_hits[k]:>+8d}")

    # h7 hits
    print(f"\n{ts()} === h7 HITS ===")
    h7_metrics = {}
    for label, idx_list in [("h7", h7_idx), ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        p3_h = {"@200": 0, "@300": 0}
        p2_h = {"@200": 0, "@300": 0}
        r21_h = {"@200": 0, "@300": 0}
        for i in idx_list:
            gt = cases[i]["gt"]
            for kl, k in [("@200", 200), ("@300", 300)]:
                if hit_at_k(p3_lists[i], gt, k):
                    p3_h[kl] += 1
                if hit_at_k(p2_lists[i], gt, k):
                    p2_h[kl] += 1
                if hit_at_k(r21_lists[i], gt, k):
                    r21_h[kl] += 1
        n_idx = len(idx_list)
        for kl in ["@200", "@300"]:
            d_p2 = p3_h[kl] - p2_h[kl]
            d_r21 = p3_h[kl] - r21_h[kl]
            print(f"  {label} hit{kl}: P3={p3_h[kl]}/{n_idx} ({p3_h[kl]/max(n_idx,1):.4f})  "
                  f"P2={p2_h[kl]}  R21={r21_h[kl]}  ΔP2={d_p2:+d}  ΔR21={d_r21:+d}")
        h7_metrics[label] = {"n": n_idx, "p3": p3_h, "p2": p2_h, "r21": r21_h}

    # Unique/lost vs R21 and P2
    print(f"\n{ts()} === UNIQUE / LOST @300 ===")
    for ref_name, ref_lists in [("R21", r21_lists), ("P2", p2_lists)]:
        both = p3_only = ref_only = neither = 0
        for i in range(n):
            gt = cases[i]["gt"]
            in_p3 = hit_at_k(p3_lists[i], gt, 300)
            in_ref = hit_at_k(ref_lists[i], gt, 300)
            if in_p3 and in_ref:
                both += 1
            elif in_p3:
                p3_only += 1
            elif in_ref:
                ref_only += 1
            else:
                neither += 1
        print(f"  P3 vs {ref_name}: both={both}  P3-only={p3_only}  {ref_name}-only={ref_only}  "
              f"neither={neither}  net={p3_only - ref_only:+d}")

    # Bucket D/E recovery
    print(f"\n{ts()} === BUCKET D/E RECOVERY by P3 ===")
    d_total = sum(1 for v in buckets.values() if v == "D")
    e_total = sum(1 for v in buckets.values() if v == "E")
    d_recovered = sum(1 for i, b in buckets.items() if b == "D" and hit_at_k(p3_lists[i], cases[i]["gt"], 300))
    e_recovered = sum(1 for i, b in buckets.items() if b == "E" and hit_at_k(p3_lists[i], cases[i]["gt"], 300))
    print(f"  Bucket D recovered: {d_recovered}/{d_total} ({d_recovered/max(d_total,1):.2%})")
    print(f"  Bucket E recovered: {e_recovered}/{e_total} ({e_recovered/max(e_total,1):.2%})")

    # Fold-by-fold
    print(f"\n{ts()} === FOLD-BY-FOLD hit@200 ===")
    fold_hits = {}
    for fi, fold in enumerate(folds):
        idx_list = fold.tolist()
        p3_h = sum(1 for i in idx_list if hit_at_k(p3_lists[i], cases[i]["gt"], 200))
        p2_h = sum(1 for i in idx_list if hit_at_k(p2_lists[i], cases[i]["gt"], 200))
        r21_h = sum(1 for i in idx_list if hit_at_k(r21_lists[i], cases[i]["gt"], 200))
        print(f"  fold {fi}: P3={p3_h}/{len(idx_list)} ({p3_h/len(idx_list):.4f})  "
              f"P2={p2_h}  R21={r21_h}  ΔP2={p3_h - p2_h:+d}  ΔR21={p3_h - r21_h:+d}")
        fold_hits[fi] = {"p3": p3_h, "p2": p2_h, "r21": r21_h, "n": len(idx_list)}

    out = {
        "p3_hits": r54_hits, "p2_hits": p2_hits, "r21_hits": r21_hits,
        "h7_metrics": h7_metrics,
        "bucket_recovery_p3": {
            "D_total": d_total, "D_recovered": d_recovered,
            "E_total": e_total, "E_recovered": e_recovered,
        },
        "baseline_buckets": dict(bucket_counts),
        "fold_hits": fold_hits,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase3_full5fold_standalone.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Standalone complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
