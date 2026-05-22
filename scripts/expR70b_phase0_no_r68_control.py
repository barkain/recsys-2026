#!/usr/bin/env python3
"""R70b — Sibling LR control: 37 features (R39+R54, NO r68).

Discriminator (per Codex consultation 2026-05-22) between two hypotheses for
R70's failure to reproduce R54c-level ranking when retraining a sibling LR:

  A) "Retraining LR loses R54c calibration" — sibling LR converges to a
     different decision boundary regardless of feature set, because of recipe
     drift (hyperparams, candidate pool generation, label construction,
     LightGBM version, etc.).
  B) "r68 features specifically poison the LR surface" — adding r68_* causes
     LightGBM to over-weight BGE-large cosines and overwrite r54 calibration.

R70b removes r68 entirely. Feature stack = FEAT_R39_ALL + FEAT_R54 (37 cols,
identical schema to R54c frozen). Pool unchanged (R54-stacked). Same fold-0
train, same LightGBM hyperparams as R70 (LR_PARAMS, 300 rounds, leaf=10,
seed=0).

Interpretation:
- If R70b ~matches R54c (Δh7 ~ 0): retraining is benign → r68 IS toxic;
  next step is feature scaling / monotone constraints on r68.
- If R70b regresses ~similarly to R70 (Δh7 ~ -0.08, same-artist ~ -0.16):
  R54c is artifact-locked; the LR cannot be re-derived from the visible
  artifacts. Sprint pivots to frozen-ranker-compatible interfaces.
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
FOLD = 0

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR70b_phase0_no_r68_control.json"
OUT_MD = REPO / "docs" / "r70b_phase0_no_r68_control_result.md"

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
    print(f"{ts()} R70b — sibling LR control (37 feats, NO r68)")
    print(f"  features: {len(FEAT_ALL)} = R39({len(FEAT_R39_ALL)}) + R54({len(FEAT_R54)})")
    print(f"  pool: R54-stacked (SW_BASELINE), unchanged")
    print("=" * 70)

    print(f"{ts()} Loading payload + R21/R54 OOF ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Loading W0 fold map ...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold0_idx = [i for i in range(n) if case_fold[i] == FOLD]
    train_idx = [i for i in range(n) if case_fold[i] != FOLD]
    print(f"  fold-0: {len(fold0_idx)}  train: {len(train_idx)}", flush=True)

    print(f"{ts()} Building case index ...", flush=True)
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

    # ---- Step 1: TRAIN feature matrix ----
    print(f"\n{ts()} === Step 1: TRAIN feature matrix ({len(train_idx)} cases) ===",
          flush=True)
    X_train, y_train, groups_train = [], [], []
    gt_in_pool_train = 0
    t_feat = time.time()
    for ki, i in enumerate(train_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        if gi >= 0:
            gt_in_pool_train += 1
        feats = featurize37(i, src_lists, pool)
        for k_row in range(len(pool)):
            X_train.append(feats[k_row])
            y_train.append(1.0 if k_row == gi else 0.0)
        groups_train.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} "
                  f"({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  TRAIN pool_hit: {gt_in_pool_train}/{len(train_idx)} "
          f"({gt_in_pool_train/max(len(train_idx),1):.4f})", flush=True)

    # ---- Step 2: Train sibling LR (37 features) ----
    print(f"\n{ts()} === Step 2: Train sibling LR (37 features) ===", flush=True)
    ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                     label=np.array(y_train, dtype=np.float64),
                     group=groups_train, feature_name=list(FEAT_ALL))
    sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    sibling_lr_path = REPO / "cache" / "r70b_phase0_sibling_lr_no_r68.txt"
    sibling_lr.save_model(str(sibling_lr_path))
    print(f"  Saved sibling LR -> {sibling_lr_path}", flush=True)

    del X_train, y_train, groups_train, ds

    # ---- Step 3: Score fold-0 dev — baseline (R54c frozen) vs R70b sibling ----
    print(f"\n{ts()} === Step 3: Score fold-0 dev ===", flush=True)
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))
    if r54_baseline_ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError(
            f"R54 LR feature count mismatch: "
            f"model={r54_baseline_ranker.num_feature()} expected={len(FEAT_ALL)}")

    fold0_rows: list[dict[str, Any]] = []
    t_eval = time.time()
    for ki, i in enumerate(fold0_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gt_in_pool = gt in pool

        feats = featurize37(i, src_lists, pool)
        b_scores = r54_baseline_ranker.predict(feats)
        b_order = np.argsort(-b_scores, kind="mergesort")
        s_scores = sibling_lr.predict(feats)
        s_order = np.argsort(-s_scores, kind="mergesort")
        b_gt_rank = -1
        s_gt_rank = -1
        if gt_in_pool:
            gt_pos = pool.index(gt)
            bp = np.where(b_order == gt_pos)[0]
            sp = np.where(s_order == gt_pos)[0]
            if len(bp):
                b_gt_rank = int(bp[0]) + 1
            if len(sp):
                s_gt_rank = int(sp[0]) + 1

        b_top20 = [pool[int(j)] for j in b_order[:TOP_K]]
        s_top20 = [pool[int(j)] for j in s_order[:TOP_K]]
        overlap = len(set(b_top20) & set(s_top20))
        top1_changed = 1 if b_top20[0] != s_top20[0] else 0

        fold0_rows.append({
            "case_idx": i,
            "session_id": cases[i]["session_id"],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "b_gt_rank": b_gt_rank,
            "s_gt_rank": s_gt_rank,
            "b_ndcg_at_20": ndcg_at_k(b_gt_rank, TOP_K),
            "s_ndcg_at_20": ndcg_at_k(s_gt_rank, TOP_K),
            "top20_overlap": overlap,
            "top1_changed": top1_changed,
            "b_in_top20": b_gt_rank > 0 and b_gt_rank <= TOP_K,
            "s_in_top20": s_gt_rank > 0 and s_gt_rank <= TOP_K,
        })
        if (ki + 1) % 200 == 0:
            print(f"    dev {ki + 1}/{len(fold0_idx)} ({time.time() - t_eval:.0f}s)",
                  flush=True)

    # ---- Step 4: Metrics + verdict ----
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7_rows = [r for r in fold0_rows if r["n_prior_music"] == 7]
    same_rows = [r for r in fold0_rows if r["same_artist"]]
    diff_rows = [r for r in fold0_rows if not r["same_artist"]]
    h7_same = [r for r in h7_rows if r["same_artist"]]
    h7_diff = [r for r in h7_rows if not r["same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", fold0_rows), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows),
                       ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        b = avg(rows, "b_ndcg_at_20")
        s = avg(rows, "s_ndcg_at_20")
        metrics[name] = {"n": len(rows), "baseline": b, "sibling": s, "delta": s - b}

    h7_recovered = sum(1 for r in h7_rows if r["s_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["s_in_top20"])
    h7_net = h7_recovered - h7_lost
    top1_churn = sum(r["top1_changed"] for r in fold0_rows)
    churn_per_80 = top1_churn / len(fold0_rows) * 80
    overlap_mean = avg(fold0_rows, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]
    all_d = metrics["all_fold0"]["delta"]

    # R70 reference deltas for comparison
    r70_h7_d = -0.0798
    r70_sa_d = -0.1595
    r70_all_d = -0.0787

    h7_diff_from_r70 = h7_d - r70_h7_d
    sa_diff_from_r70 = sa_d - r70_sa_d

    # Interpret
    if abs(h7_d) < 0.005 and abs(sa_d) < 0.005:
        interp = "MATCHES_R54C"
        meaning = ("Retraining with 37 features reproduces R54c. "
                   "→ r68 features are SPECIFICALLY TOXIC. Next: feature "
                   "scaling, monotone constraints, or regularization on r68_*.")
    elif h7_d <= -0.05 and sa_d <= -0.10:
        interp = "ARTIFACT_LOCKED"
        meaning = ("Retraining 37 features regresses ~similarly to R70's "
                   "40-feature regression. → R54c is artifact-locked. "
                   "Sibling LR cannot reproduce R54c from the same feature "
                   "schema. Sprint pivots to frozen-ranker-compatible "
                   "interfaces (stacker, candidate injection, residual rerank).")
    else:
        interp = "MIXED"
        meaning = ("R70b regresses but less severely than R70. r68 has SOME "
                   "specific toxicity, but retraining alone also loses some "
                   "calibration. Both directions need work.")

    print(f"\n{ts()} === Results ===", flush=True)
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  base={m['baseline']:.4f}  "
              f"sibling={m['sibling']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7_recovered={h7_recovered}  h7_lost={h7_lost}  net={h7_net:+d}",
          flush=True)
    print(f"  top1_changed={top1_churn}  churn_per_80={churn_per_80:.2f}  "
          f"overlap_mean={overlap_mean:.2f}/20", flush=True)
    print(f"\n  vs R70 (40-feat with r68):", flush=True)
    print(f"    R70 h7 Δ:        {r70_h7_d:+.4f}", flush=True)
    print(f"    R70b h7 Δ:       {h7_d:+.4f}  (diff vs R70: {h7_diff_from_r70:+.4f})",
          flush=True)
    print(f"    R70 same_art Δ:  {r70_sa_d:+.4f}", flush=True)
    print(f"    R70b same_art Δ: {sa_d:+.4f}  (diff vs R70: {sa_diff_from_r70:+.4f})",
          flush=True)
    print(f"\n  INTERPRETATION: {interp}", flush=True)
    print(f"  {meaning}", flush=True)

    out = {
        "experiment": "R70b — sibling LR control (37 feats, NO r68)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "interpretation": interp,
        "meaning": meaning,
        "fold": FOLD,
        "n_fold0": len(fold0_idx),
        "n_train": len(train_idx),
        "feature_stack": {
            "names": list(FEAT_ALL), "n_features": len(FEAT_ALL),
            "structure": "FEAT_R39_ALL + FEAT_R54 (identical schema to R54c)",
        },
        "pool": {"weights": SW_BASELINE, "K": POOL_K,
                 "note": "R54-stacked, identical to R54c production"},
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_recovered, "lost": h7_lost, "net": h7_net},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "comparison_to_R70": {
            "R70_h7_delta": r70_h7_d, "R70b_h7_delta": h7_d,
            "h7_diff_from_R70": h7_diff_from_r70,
            "R70_same_artist_delta": r70_sa_d, "R70b_same_artist_delta": sa_d,
            "same_artist_diff_from_R70": sa_diff_from_r70,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    md = [
        "# R70b Phase 0 — Sibling LR Control (37 feats, NO r68)",
        "",
        f"HEAD: `{out['head_sha'][:10]}`  Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Interpretation: **{interp}**",
        "",
        meaning,
        "",
        "## Metrics",
        "",
        "| Subset | n | R54c frozen | R70b sibling (37f) | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['sibling']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_recovered}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80={churn_per_80:.2f}  top-20 overlap mean={overlap_mean:.2f}/20",
        "",
        "## R70b vs R70 (with r68) — the discriminator",
        "",
        "| | R70 (40 feats, +r68) | R70b (37 feats, no r68) | diff |",
        "|---|---:|---:|---:|",
        f"| h7 nDCG Δ | {r70_h7_d:+.4f} | {h7_d:+.4f} | {h7_diff_from_r70:+.4f} |",
        f"| same-artist Δ | {r70_sa_d:+.4f} | {sa_d:+.4f} | {sa_diff_from_r70:+.4f} |",
        f"| all_fold0 Δ | {r70_all_d:+.4f} | {all_d:+.4f} | {all_d - r70_all_d:+.4f} |",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
