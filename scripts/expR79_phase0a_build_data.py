#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R79 Phase 0A — Mac-side prep for top-rank discriminator retriever.

Builds hard-negative training pairs for fine-tuning a retriever to BEAT
R54c's top-20 false positives, not optimize hit@300.

Method:
- Train 5 sibling R54-style LRs (one per held-out fold).
- For each case i, get its OOF R54c top-20 from the LR that didn't see
  fold-of-i during training. Those top-20 are the "hard candidates" for case i.
- Build training pairs: (query_for_case_i, positive=GT_i, hard_negs=top20_non_GT_i)
- Save as a Python pickle for downstream A100 fine-tuning.

Also locks the standalone OOF R54c top-20 nDCG as Phase 0B's baseline.

No GPU. No model training. ~15-20 min on Mac.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_R39_ALL, FEAT_R54, FEAT_ALL,
    _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
POOL_K = 300
RRF_K = 20
TOP_20 = 20
N_FOLDS = 5

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_DIR = REPO / "cache" / "r79"
OUT_PAIRS = OUT_DIR / "training_pairs.pkl"
OUT_BASELINE = OUT_DIR / "eval_baseline.json"
OUT_STATS = REPO / "exp" / "eval" / "expR79_phase0a_stats.json"
OUT_DOC = REPO / "docs" / "r79_phase0a_data_audit.md"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R79 Phase 0A — build training data + lock baseline")
    print("=" * 70)

    print(f"{ts()} Loading payloads ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    track_artist = maps["track_artist"]

    print(f"{ts()} Loading W0 fold map ...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_idx_map = {k: [i for i in range(n) if case_fold[i] == k]
                    for k in range(N_FOLDS)}
    for k, idxs in fold_idx_map.items():
        print(f"  fold {k}: {len(idxs)} cases")

    print(f"{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize37(case_idx, src_lists, pool):
        case = cases[case_idx]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[case_idx],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )

    # ---- Precompute pool + features per case (used by all 5 sibling LRs) ----
    print(f"\n{ts()} === Step 1: Precompute pool + features for all {n} cases ===")
    case_data = {}
    t_feat = time.time()
    for ki in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], ki)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        feats = featurize37(ki, src_lists, pool)
        gt = cases[ki]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        case_data[ki] = {
            "pool": pool,
            "feats": feats,
            "gt": gt,
            "gt_pool_idx": gi,
        }
        if (ki + 1) % 1000 == 0:
            print(f"    {ki + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)

    # ---- Train 5 sibling LRs (one per held-out fold), use to score that fold OOF ----
    print(f"\n{ts()} === Step 2: Train 5 sibling LRs + OOF-score held-out folds ===")
    oof_top20_by_case: dict[int, list[str]] = {}
    oof_score_by_case: dict[int, list[float]] = {}

    for fk in range(N_FOLDS):
        held = fold_idx_map[fk]
        train = [i for k, idxs in fold_idx_map.items() if k != fk for i in idxs]
        print(f"\n  fold {fk}: train={len(train)} held={len(held)}", flush=True)

        # Build training arrays
        X_tr, y_tr, g_tr = [], [], []
        for i in train:
            cd = case_data[i]
            feats = cd["feats"]
            for k_row in range(len(cd["pool"])):
                X_tr.append(feats[k_row])
                y_tr.append(1.0 if k_row == cd["gt_pool_idx"] else 0.0)
            g_tr.append(len(cd["pool"]))

        t_lr = time.time()
        ds = lgb.Dataset(np.array(X_tr, dtype=np.float64),
                         label=np.array(y_tr, dtype=np.float64),
                         group=g_tr, feature_name=list(FEAT_ALL))
        sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
        print(f"    LR trained in {time.time() - t_lr:.0f}s", flush=True)
        del X_tr, y_tr, g_tr, ds

        # Score held-out cases
        for i in held:
            cd = case_data[i]
            scores = sibling_lr.predict(cd["feats"])
            order = np.argsort(-scores, kind="mergesort")
            top20_pos = order[:TOP_20]
            oof_top20_by_case[i] = [cd["pool"][int(p)] for p in top20_pos]
            oof_score_by_case[i] = [float(scores[int(p)]) for p in top20_pos]
        del sibling_lr

    # ---- Step 3: Build training pairs (positive=GT, hard negs=OOF top-20 non-GT) ----
    print(f"\n{ts()} === Step 3: Build hard-negative training pairs ===")
    training_pairs = []
    n_with_gt_in_top20 = 0
    for i in range(n):
        case = cases[i]
        gt = cases[i]["gt"]
        top20 = oof_top20_by_case[i]
        # Hard negatives = top-20 that aren't GT
        hard_negs = [t for t in top20 if t != gt]
        gt_in_top20 = gt in set(top20)
        if gt_in_top20:
            n_with_gt_in_top20 += 1
        # Build the structured query text (same as R54's encoding format)
        user_msgs = [str(r["content"]) for r in case["history"] if r["role"] == "user"] + [case["user_query"]]
        # Brief query rep — pre-compose so R79 model can re-tokenize
        history_text = " | ".join(user_msgs)
        played = case["music_turns"]
        played_meta = []
        for pt in played:
            artist = track_artist.get(pt, "?")
            played_meta.append(f"{artist}")
        history_summary = " | ".join(played_meta) if played_meta else ""

        training_pairs.append({
            "case_idx": i,
            "session_id": case["session_id"],
            "turn_number": int(case.get("turn_number", 0)),
            "fold": case_fold[i],
            "is_h7": int(case["n_prior_music"]) == 7,
            "is_same_artist": same_artist_case(case, track_artist),
            "gt": gt,
            "hard_negs": hard_negs,                # R54c top-20 minus GT
            "oof_top20": top20,                    # full top-20 for ref
            "oof_top20_scores": oof_score_by_case[i],
            "gt_was_in_top20": gt_in_top20,
            "user_query": case["user_query"],
            "history_text": history_text,
            "played_tracks": list(played),
            "history_summary": history_summary,
        })

    print(f"  total cases: {n}")
    print(f"  GT in OOF R54c top-20: {n_with_gt_in_top20}/{n} = {n_with_gt_in_top20/n:.4f}")
    print(f"  avg hard_negs per case: "
          f"{sum(len(p['hard_negs']) for p in training_pairs)/n:.2f}")

    # ---- Step 4: Compute Phase 0B baseline (OOF R54c standalone top-20 nDCG) ----
    print(f"\n{ts()} === Step 4: Lock Phase 0B baseline (OOF R54c standalone) ===")
    rows = []
    for i in range(n):
        cd = case_data[i]
        gt = cd["gt"]
        top20 = oof_top20_by_case[i]
        b_rank = -1
        for r_idx, tid in enumerate(top20):
            if tid == gt:
                b_rank = r_idx + 1
                break
        rows.append({
            "case_idx": i,
            "fold": case_fold[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "is_h7": int(cases[i]["n_prior_music"]) == 7,
            "is_same_artist": same_artist_case(cases[i], track_artist),
            "b_gt_rank": b_rank,
            "b_in_top20": 0 < b_rank <= TOP_20,
            "b_ndcg": ndcg_at_k(b_rank, TOP_20),
            "oof_top20": top20,
        })

    def avg(rs, key, where=None):
        if where is not None:
            rs = [r for r in rs if where(r)]
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    # Per-fold + aggregate
    baseline_metrics = {}
    for label, where in [
        ("all", lambda r: True),
        ("h7", lambda r: r["is_h7"]),
        ("same_artist", lambda r: r["is_same_artist"]),
        ("diff_artist", lambda r: not r["is_same_artist"]),
        ("h7_same", lambda r: r["is_h7"] and r["is_same_artist"]),
        ("h7_diff", lambda r: r["is_h7"] and not r["is_same_artist"]),
        ("fold_0", lambda r: r["fold"] == 0),
        ("fold_0_h7", lambda r: r["fold"] == 0 and r["is_h7"]),
    ]:
        subset = [r for r in rows if where(r)]
        baseline_metrics[label] = {
            "n": len(subset),
            "ndcg_at_20": avg(subset, "b_ndcg"),
            "in_top20": sum(1 for r in subset if r["b_in_top20"]),
        }

    print(f"  baseline ndcg@20 (OOF R54c standalone, used as R79 Phase 0B baseline):")
    for label, m in baseline_metrics.items():
        rate = m["in_top20"] / max(m["n"], 1)
        print(f"    {label:14}  n={m['n']:5d}  nDCG@20={m['ndcg_at_20']:.4f}  "
              f"in_top20={m['in_top20']} ({rate:.4f})")

    # Save artifacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PAIRS, "wb") as f:
        pickle.dump({
            "training_pairs": training_pairs,
            "metadata": {
                "n_cases": n,
                "n_folds": N_FOLDS,
                "fold_sizes": {k: len(v) for k, v in fold_idx_map.items()},
                "gt_in_top20_count": n_with_gt_in_top20,
                "gt_in_top20_rate": n_with_gt_in_top20 / n,
                "top_k_used_for_hard_negs": TOP_20,
            },
        }, f)
    print(f"\n{ts()} Saved training pairs → {OUT_PAIRS}")

    with open(OUT_BASELINE, "w") as f:
        json.dump({
            "experiment": "R79 Phase 0A — OOF R54c standalone baseline (per-fold)",
            "created_at": datetime.now().isoformat(),
            "baseline_metrics": baseline_metrics,
        }, f, indent=2)
    print(f"{ts()} Saved baseline → {OUT_BASELINE}")

    stats = {
        "experiment": "R79 Phase 0A — data + baseline",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "n_cases": n,
        "n_folds": N_FOLDS,
        "fold_sizes": {k: len(v) for k, v in fold_idx_map.items()},
        "training_pairs_count": len(training_pairs),
        "gt_in_oof_top20": {
            "count": n_with_gt_in_top20,
            "rate": n_with_gt_in_top20 / n,
        },
        "phase0b_baseline": baseline_metrics,
        "files": {
            "training_pairs": str(OUT_PAIRS),
            "baseline": str(OUT_BASELINE),
        },
    }
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"{ts()} Saved stats → {OUT_STATS}")

    md = [
        "# R79 Phase 0A — data + baseline audit",
        "",
        f"Elapsed: {stats['elapsed_s']:.0f}s",
        f"Total cases: {n}  Folds: {N_FOLDS}",
        "",
        "## Training pair stats",
        "",
        f"- Cases: {len(training_pairs)}",
        f"- GT in OOF R54c top-20: **{n_with_gt_in_top20}/{n} = {n_with_gt_in_top20/n:.4f}**",
        f"- Avg hard negs per case: ~{sum(len(p['hard_negs']) for p in training_pairs)/n:.1f}",
        "",
        "## Phase 0B baseline (OOF R54c standalone top-20)",
        "",
        "This is what R79's standalone retriever must beat to clear gates.",
        "",
        "| Subset | n | nDCG@20 | in_top20 |",
        "|---|---:|---:|---:|",
    ]
    for label, m in baseline_metrics.items():
        rate = m["in_top20"] / max(m["n"], 1)
        md.append(f"| {label} | {m['n']} | {m['ndcg_at_20']:.4f} | {m['in_top20']} ({rate:.3f}) |")
    md += [
        "",
        "## Phase 0B gates",
        "",
        "Use fold-0 subset specifically (n=1600, h7=200).",
        "",
        "- h7 nDCG Δ ≥ +0.005 vs baseline (h7 = 0.2213 above)",
        "- same-artist Δ ≥ -0.002 (subset)",
        "- recovered (R79 in top-20, R54c not) > lost (R54c in top-20, R79 not) on h7",
        "- top-1 churn /80 ≤ 25 on fold-0",
        "",
        "## Files",
        "",
        f"- Training pairs: `{OUT_PAIRS}`",
        f"- Baseline: `{OUT_BASELINE}`",
        f"- Stats: `{OUT_STATS}`",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved doc → {OUT_DOC}")


if __name__ == "__main__":
    main()
