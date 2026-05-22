#!/usr/bin/env python3
"""R70b 5-fold OOF baseline — true OOF R54c-equivalent across all 5 folds.

Per the R70b critical finding, frozen R54c was trained on all 8000 dev (in-
sample). The ~−0.08 h7 / ~−0.16 same-artist deltas seen across 20 sprints are
dominated by this OOF-vs-in-sample bias.

This script confirms (or refutes) that interpretation at 5-fold scale:
- For each fold k in {0..4}: train sibling LR with 37 features (R39+R54, no
  r68_*, identical schema to R54c) on the other 4 folds' cases. Score fold-k
  dev. Identical hyperparams to R54c.
- Aggregate: 5-fold OOF h7 nDCG and same-artist nDCG.
- Compare to frozen R54c on the same 8000 cases (where R54c is in-sample).

If 5-fold R70b shows Δh7 ≈ −0.08 / Δsame-artist ≈ −0.16 averaged across all
folds → artifact theory definitively confirmed.

This gives us the TRUE OOF baseline. Future sibling-OOF experiments (e.g.,
5-fold R70 with r68_* features) should be compared against this baseline,
not frozen R54c.

Output: exp/eval/expR70b_5fold_oof.json (per-fold + aggregated metrics).
"""
from __future__ import annotations
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
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
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR70b_5fold_oof.json"
OUT_MD = REPO / "docs" / "r70b_5fold_oof_result.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        return "no-git"
    return subprocess.check_output(  # noqa: S603
        [git_bin, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R70b 5-fold OOF — true OOF R54c-equivalent baseline")
    print(f"  features: {len(FEAT_ALL)} (R39+R54, identical schema to R54c)")
    print(f"  folds: {N_FOLDS}  hyperparams: identical to R54c")
    print("=" * 70)

    print(f"{ts()} Loading payload + R21/R54 OOF ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Loading fold map from W0 stats ...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_idx = {k: [i for i in range(n) if case_fold[i] == k] for k in range(N_FOLDS)}
    for k in range(N_FOLDS):
        print(f"  fold {k}: {len(fold_idx[k])} cases", flush=True)
    assert sum(len(v) for v in fold_idx.values()) == n

    print(f"{ts()} Building case index (ALS, R21, R54 source lists) ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize37(case_idx: int, src_lists: dict[str, list[str]],
                    pool: list[str]) -> np.ndarray:
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

    # ---- Pre-build all features and pools (8000 cases × 300 pool = 2.4M rows)
    # cached so we don't rebuild per fold. ~3 min total featurization.
    print(f"\n{ts()} === Pre-building all-case features (8000 × 300) ===",
          flush=True)
    case_feats = {}   # case_idx -> (X, gt_idx_in_pool, pool, gt_in_pool)
    t_feat = time.time()
    for ki in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], ki)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[ki]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        feats = featurize37(ki, src_lists, pool)
        case_feats[ki] = (feats, gi, pool, gi >= 0)
        if (ki + 1) % 1000 == 0:
            print(f"    {ki + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  all-case features built in {time.time() - t_feat:.0f}s", flush=True)

    # ---- 5-fold: train on other 4 folds, eval on held-out fold ----
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))
    if r54_baseline_ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError("R54 LR feature count mismatch")

    per_fold = {}
    all_rows_global: list[dict[str, Any]] = []
    for fk in range(N_FOLDS):
        print(f"\n{ts()} === Fold {fk}: train on others, eval on fold {fk} ===",
              flush=True)
        held = fold_idx[fk]
        train = [i for k, idxs in fold_idx.items() if k != fk for i in idxs]
        print(f"  train: {len(train)}  held: {len(held)}", flush=True)

        # Build train arrays (already-cached features)
        X_train, y_train, groups_train = [], [], []
        for i in train:
            feats, gi, pool, _ = case_feats[i]
            for k_row in range(len(pool)):
                X_train.append(feats[k_row])
                y_train.append(1.0 if k_row == gi else 0.0)
            groups_train.append(len(pool))

        t_train = time.time()
        ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                         label=np.array(y_train, dtype=np.float64),
                         group=groups_train, feature_name=list(FEAT_ALL))
        sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
        print(f"  fold {fk} LR trained in {time.time() - t_train:.0f}s",
              flush=True)
        del X_train, y_train, groups_train, ds

        # Eval on held-out fold
        fold_rows = []
        for i in held:
            feats, gi, pool, gt_in_pool = case_feats[i]
            b_scores = r54_baseline_ranker.predict(feats)
            s_scores = sibling_lr.predict(feats)
            b_order = np.argsort(-b_scores, kind="mergesort")
            s_order = np.argsort(-s_scores, kind="mergesort")
            b_gt_rank = -1
            s_gt_rank = -1
            if gt_in_pool:
                bp = np.where(b_order == gi)[0]
                sp = np.where(s_order == gi)[0]
                if len(bp):
                    b_gt_rank = int(bp[0]) + 1
                if len(sp):
                    s_gt_rank = int(sp[0]) + 1
            row = {
                "case_idx": i, "fold": fk,
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "b_gt_rank": b_gt_rank, "s_gt_rank": s_gt_rank,
                "b_ndcg": ndcg_at_k(b_gt_rank, TOP_K),
                "s_ndcg": ndcg_at_k(s_gt_rank, TOP_K),
                "b_in_top20": b_gt_rank > 0 and b_gt_rank <= TOP_K,
                "s_in_top20": s_gt_rank > 0 and s_gt_rank <= TOP_K,
                "top1_changed": 1 if (gt_in_pool and len(b_order) > 0 and len(s_order) > 0
                                       and b_order[0] != s_order[0]) else 0,
            }
            fold_rows.append(row)
            all_rows_global.append(row)

        def avg(rows, key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        f_all = fold_rows
        f_h7 = [r for r in fold_rows if r["n_prior_music"] == 7]
        f_same = [r for r in fold_rows if r["same_artist"]]
        b_all = avg(f_all, "b_ndcg")
        s_all = avg(f_all, "s_ndcg")
        b_h7 = avg(f_h7, "b_ndcg")
        s_h7 = avg(f_h7, "s_ndcg")
        b_same = avg(f_same, "b_ndcg")
        s_same = avg(f_same, "s_ndcg")
        per_fold[fk] = {
            "n_held": len(f_all), "n_h7": len(f_h7), "n_same": len(f_same),
            "baseline_all": b_all, "sibling_all": s_all, "delta_all": s_all - b_all,
            "baseline_h7": b_h7, "sibling_h7": s_h7, "delta_h7": s_h7 - b_h7,
            "baseline_same": b_same, "sibling_same": s_same, "delta_same": s_same - b_same,
        }
        print(f"  fold {fk}  all Δ={s_all - b_all:+.4f}  h7 Δ={s_h7 - b_h7:+.4f}  "
              f"same_art Δ={s_same - b_same:+.4f}", flush=True)
        del sibling_lr

    # ---- Aggregate across folds (case-weighted) ----
    def agg(rows, key, b_or_s):
        return float(np.mean([r[f"{b_or_s}_ndcg"] for r in rows if (key is None
                              or r["n_prior_music"] == 7 if key == "h7" else r[key])]))

    g_all_b = float(np.mean([r["b_ndcg"] for r in all_rows_global]))
    g_all_s = float(np.mean([r["s_ndcg"] for r in all_rows_global]))
    g_h7_rows = [r for r in all_rows_global if r["n_prior_music"] == 7]
    g_h7_b = float(np.mean([r["b_ndcg"] for r in g_h7_rows]))
    g_h7_s = float(np.mean([r["s_ndcg"] for r in g_h7_rows]))
    g_same_rows = [r for r in all_rows_global if r["same_artist"]]
    g_same_b = float(np.mean([r["b_ndcg"] for r in g_same_rows]))
    g_same_s = float(np.mean([r["s_ndcg"] for r in g_same_rows]))
    g_diff_rows = [r for r in all_rows_global if not r["same_artist"]]
    g_diff_b = float(np.mean([r["b_ndcg"] for r in g_diff_rows]))
    g_diff_s = float(np.mean([r["s_ndcg"] for r in g_diff_rows]))

    h7_recovered = sum(1 for r in g_h7_rows if r["s_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in g_h7_rows if r["b_in_top20"] and not r["s_in_top20"])
    top1_churn = sum(r["top1_changed"] for r in all_rows_global)
    churn_per_80 = top1_churn / max(len(all_rows_global), 1) * 80

    print(f"\n{ts()} === 5-fold OOF aggregate ===", flush=True)
    print(f"  all_dev    n={len(all_rows_global)}   "
          f"R54c={g_all_b:.4f}  R70b_5fold={g_all_s:.4f}  Δ={g_all_s - g_all_b:+.4f}",
          flush=True)
    print(f"  h7         n={len(g_h7_rows):5d}  "
          f"R54c={g_h7_b:.4f}  R70b_5fold={g_h7_s:.4f}  Δ={g_h7_s - g_h7_b:+.4f}",
          flush=True)
    print(f"  same_art   n={len(g_same_rows):5d}  "
          f"R54c={g_same_b:.4f}  R70b_5fold={g_same_s:.4f}  Δ={g_same_s - g_same_b:+.4f}",
          flush=True)
    print(f"  diff_art   n={len(g_diff_rows):5d}  "
          f"R54c={g_diff_b:.4f}  R70b_5fold={g_diff_s:.4f}  Δ={g_diff_s - g_diff_b:+.4f}",
          flush=True)
    print(f"  h7 recovered={h7_recovered}  lost={h7_lost}  net={h7_recovered - h7_lost:+d}",
          flush=True)
    print(f"  top1_churn /80 = {churn_per_80:.2f}", flush=True)

    # Interpretation
    h7_d = g_h7_s - g_h7_b
    sa_d = g_same_s - g_same_b
    if abs(h7_d + 0.08) <= 0.02 and abs(sa_d + 0.15) <= 0.04:
        interp = "ARTIFACT_CONFIRMED_5FOLD"
        meaning = ("5-fold OOF R70b regression matches fold-0 (~-0.08 h7, "
                   "-0.15 same-artist) almost exactly across all folds. The "
                   "'LR conversion wall' is the train/dev OOF-vs-in-sample "
                   "memorization gap. Frozen R54c is artifact-locked (in-sample "
                   "for all 8000 dev cases). Future sibling experiments must "
                   "compare against this 5-fold OOF baseline, not frozen R54c.")
    elif abs(h7_d) < 0.02:
        interp = "PARTIAL_ARTIFACT"
        meaning = ("5-fold OOF R70b regresses less than fold-0. The artifact "
                   "is real but smaller in magnitude than fold-0 suggested. "
                   "Some component of the prior conversion failures may be "
                   "genuine.")
    else:
        interp = "UNCLEAR"
        meaning = "5-fold OOF result doesn't cleanly match either prediction. Investigate fold variance."

    print(f"\n  INTERPRETATION: {interp}", flush=True)
    print(f"  {meaning}", flush=True)

    out = {
        "experiment": "R70b 5-fold OOF — true OOF R54c-equivalent baseline",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "interpretation": interp,
        "meaning": meaning,
        "feature_stack": {
            "names": list(FEAT_ALL), "n_features": len(FEAT_ALL),
            "structure": "FEAT_R39_ALL + FEAT_R54 (identical schema to R54c)",
        },
        "per_fold": per_fold,
        "aggregate": {
            "all_dev": {"n": len(all_rows_global),
                        "R54c_in_sample": g_all_b,
                        "R70b_5fold_OOF": g_all_s,
                        "delta": g_all_s - g_all_b},
            "h7": {"n": len(g_h7_rows),
                   "R54c_in_sample": g_h7_b,
                   "R70b_5fold_OOF": g_h7_s,
                   "delta": g_h7_s - g_h7_b},
            "same_artist": {"n": len(g_same_rows),
                            "R54c_in_sample": g_same_b,
                            "R70b_5fold_OOF": g_same_s,
                            "delta": g_same_s - g_same_b},
            "diff_artist": {"n": len(g_diff_rows),
                            "R54c_in_sample": g_diff_b,
                            "R70b_5fold_OOF": g_diff_s,
                            "delta": g_diff_s - g_diff_b},
        },
        "h7_recovery": {"recovered": h7_recovered, "lost": h7_lost,
                        "net": h7_recovered - h7_lost},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80},
        "note": (
            "R70b 5-fold OOF establishes the TRUE OOF baseline for sibling "
            "experiments with 37-feature R54c schema. Frozen R54c is in-sample "
            "for all 8000 dev cases. Future sibling-OOF experiments (e.g., "
            "with r68_* features added) should compare against this baseline."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    md = [
        "# R70b 5-fold OOF — true OOF R54c-equivalent baseline",
        "",
        f"HEAD: `{out['head_sha'][:10]}`  Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Interpretation: **{interp}**",
        "",
        meaning,
        "",
        "## 5-fold aggregate",
        "",
        "| Subset | n | R54c in-sample | R70b 5-fold OOF | Δ |",
        "|---|---:|---:|---:|---:|",
        f"| all_dev | {len(all_rows_global)} | {g_all_b:.4f} | {g_all_s:.4f} | {g_all_s - g_all_b:+.4f} |",
        f"| h7 | {len(g_h7_rows)} | {g_h7_b:.4f} | {g_h7_s:.4f} | {g_h7_s - g_h7_b:+.4f} |",
        f"| same_artist | {len(g_same_rows)} | {g_same_b:.4f} | {g_same_s:.4f} | {g_same_s - g_same_b:+.4f} |",
        f"| diff_artist | {len(g_diff_rows)} | {g_diff_b:.4f} | {g_diff_s:.4f} | {g_diff_s - g_diff_b:+.4f} |",
        "",
        "## Per-fold deltas",
        "",
        "| Fold | n | Δ all | Δ h7 | Δ same-artist |",
        "|---|---:|---:|---:|---:|",
    ]
    for fk, m in per_fold.items():
        md.append(f"| {fk} | {m['n_held']} | {m['delta_all']:+.4f} | "
                  f"{m['delta_h7']:+.4f} | {m['delta_same']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_recovered}, lost={h7_lost}, net={h7_recovered - h7_lost:+d}",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
