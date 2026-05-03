#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R22c fine sweep: w={0.2,0.3,0.4,0.5} × pool_k={300,350}.

Reuses build_features_flexible and run_cv_sliced from R22c.
Hard stop: last_turn >= R21 + 0.005 AND CV5 >= R21 + 0.003.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.expR22c_conditional_ensemble import (
    FEATURE_NAMES_R21, FEATURE_NAMES_ENS,
    build_features_flexible, run_cv_sliced,
)

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R22B_LISTS = REPO / "cache" / "r22b" / "dev_r22b_lists.json"
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()
    print(f"{ts()} R22c fine sweep")
    print(f"{'='*60}")

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R22B_LISTS) as f:
        r22b_source = json.load(f)

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()

    # Baseline
    print(f"\n{ts()} R21 baseline (pool_k=300)")
    from scripts.expR22c_conditional_ensemble import FEATURE_NAMES_R21
    # Need to monkey-patch POOL_K for the imported module
    import scripts.expR22c_conditional_ensemble as mod
    orig_pool_k = mod.POOL_K

    mod.POOL_K = 300
    X0, gt0, sz0 = build_features_flexible(
        cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, r21_source, [],
        {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
        FEATURE_NAMES_R21, r22b_pool_mode="none")
    r0 = run_cv_sliced(X0, gt0, sz0, cases, sessions, len(FEATURE_NAMES_R21))
    ph0 = float(np.mean(gt0 >= 0))
    print(f"  pool_hit={ph0:.4f}  CV5={r0['cv5']:.5f}  last_turn={r0['last_turn']:.5f}")

    # Sweep
    results = {"baseline": {"pool_hit": ph0, **r0}}
    configs = []
    for w in [0.2, 0.3, 0.4, 0.5]:
        for pk in [300, 350]:
            configs.append((w, pk))

    print(f"\n{ts()} Running {len(configs)} configs...")
    for w, pk in configs:
        name = f"w{w}_pk{pk}"
        mod.POOL_K = pk
        sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": w}
        X, gt, sz = build_features_flexible(
            cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
            track_pop, r21_source, r22b_source,
            sw, FEATURE_NAMES_ENS, r22b_pool_mode="source")
        ph = float(np.mean(gt >= 0))
        r = run_cv_sliced(X, gt, sz, cases, sessions, len(FEATURE_NAMES_ENS))
        results[name] = {"pool_hit": ph, **r}

        d_cv5 = r["cv5"] - r0["cv5"]
        d_last = r["last_turn"] - r0["last_turn"]
        gate_last = d_last >= 0.005
        gate_cv5 = d_cv5 >= 0.003
        gate_pool = ph >= 0.620
        status = "SUBMIT" if (gate_last and gate_cv5 and gate_pool) else ""
        print(f"  {name:<12} ph={ph:.4f} cv5={r['cv5']:.5f}({d_cv5:+.4f}) "
              f"last={r['last_turn']:.5f}({d_last:+.4f}) {status}")

    # Summary
    mod.POOL_K = orig_pool_k
    print(f"\n{'='*60}")
    print(f"{'Config':<12} {'pool_hit':>9} {'CV5':>9} {'Δcv5':>7} {'last_t':>9} {'Δlast':>7} {'gates':>8}")
    print(f"{'-'*62}")
    for name, r in results.items():
        d_cv5 = r["cv5"] - r0["cv5"]
        d_last = r["last_turn"] - r0["last_turn"]
        g = ("P" if r["pool_hit"] >= 0.620 else ".") + \
            ("C" if d_cv5 >= 0.003 else ".") + \
            ("L" if d_last >= 0.005 else ".")
        print(f"{name:<12} {r['pool_hit']:>9.4f} {r['cv5']:>9.5f} {d_cv5:>+7.4f} "
              f"{r['last_turn']:>9.5f} {d_last:>+7.4f} {g:>8}")

    # Best by last_turn
    best_last = max((n, r) for n, r in results.items() if n != "baseline"
                    and r["last_turn"] - r0["last_turn"] > 0)
    print(f"\nBest last_turn: {best_last[0]} ({best_last[1]['last_turn']-r0['last_turn']:+.5f})")

    any_submit = any(
        r["pool_hit"] >= 0.620 and r["cv5"] - r0["cv5"] >= 0.003 and r["last_turn"] - r0["last_turn"] >= 0.005
        for n, r in results.items() if n != "baseline")
    if any_submit:
        print("\n>>> SUBMISSION CANDIDATE FOUND <<<")
    else:
        print("\n>>> NO CONFIG MEETS ALL GATES — archive R22b for Blind-A <<<")

    out = REPO / "exp" / "eval" / "r22c_fine_sweep.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
