#!/usr/bin/env python3
"""R71 — Stacker on top of R54c-style LR + R68 features (fold-0 OOF).

Bypasses both:
- The OOF-vs-in-sample artifact (we train an OOF R54c-equivalent on folds 1-4
  and score fold-0 cases — truly OOF).
- The recipe drift (we use a current-recipe sibling LR for BOTH baseline and
  the stacker input — same recipe, same drift, isolated comparison).

Design:
  1. Train R54c-style LR on cases in folds 1-4 (6400 cases, current recipe) →
     `OOF_R54c_for_fold0`.
  2. Score fold-0 cases (1600 cases, R54-stacked RRF top-300 pool) →
     per-case rank + score.
  3. For each fold-0 candidate in OOF_R54c top-30, compute
     [r54c_score, r54c_rank_inv, r68_rank_inv, r68_presence, r68_cosine].
  4. Train a tiny stacker (LightGBM LambdaRank with 5 features, OR linear)
     via 5-way CV within fold-0 cases.
  5. Compare stacker top-20 nDCG vs OOF_R54c top-20 nDCG (baseline).

Gates:
  - h7 Δ ≥ +0.005 (stacker vs OOF R54c)
  - same-artist Δ ≥ -0.002
  - recovered > lost
  - top-1 churn / 80 ≤ 25
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
TOP_30 = 30
TOP_K = 20
N_FOLDS = 5
STACKER_INNER_FOLDS = 5

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R68_DEV_LISTS = REPO / "cache" / "r68" / "phase0_fold0" / "oof_r68_lists_fold0.json"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR71b_stacker_simpler.json"
OUT_MD = REPO / "docs" / "r71b_stacker_simpler_result.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

STACKER_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 7, "learning_rate": 0.05, "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbose": -1, "seed": 1,
}
STACKER_NUM_BOOST = 20

FEAT_STACKER = ["r54c_score", "r54c_rank_inv", "r68_rank_inv",
                "r68_presence", "r68_cosine"]


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
    print(f"{ts()} R71 — Stacker on OOF R54c + R68 features (fold-0)")
    print(f"  stacker features: {FEAT_STACKER}")
    print(f"  inner CV folds within fold-0: {STACKER_INNER_FOLDS}")
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
    fold0_idx = [i for i in range(n) if case_fold[i] == 0]
    train_idx = [i for i in range(n) if case_fold[i] != 0]
    print(f"  fold-0 dev: {len(fold0_idx)}  train (folds 1-4): {len(train_idx)}",
          flush=True)

    print(f"{ts()} Loading R68 fold-0 lists ...", flush=True)
    with open(R68_DEV_LISTS) as f:
        r68_data = json.load(f)
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"]["val_idx"]
    r68_lists_by_case = {}
    for k_pos, case_idx in enumerate(r68_val_idx):
        r68_lists_by_case[int(case_idx)] = [
            (str(t), float(s)) for t, s in r68_data["lists"][k_pos]]
    print(f"  R68 lists for fold-0: {len(r68_lists_by_case)} cases", flush=True)

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

    # ---- Step 1: Train OOF R54c-style LR on folds 1-4 ----
    print(f"\n{ts()} === Step 1: Train OOF R54c on folds 1-4 ({len(train_idx)} cases) ===",
          flush=True)
    X_tr, y_tr, g_tr = [], [], []
    t_feat = time.time()
    for ki, i in enumerate(train_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        feats = featurize37(i, src_lists, pool)
        for k_row in range(len(pool)):
            X_tr.append(feats[k_row])
            y_tr.append(1.0 if k_row == gi else 0.0)
        g_tr.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} "
                  f"({time.time() - t_feat:.0f}s)", flush=True)

    t_lr = time.time()
    ds = lgb.Dataset(np.array(X_tr, dtype=np.float64),
                     label=np.array(y_tr, dtype=np.float64),
                     group=g_tr, feature_name=list(FEAT_ALL))
    oof_r54c = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    print(f"  OOF R54c trained in {time.time() - t_lr:.0f}s", flush=True)
    del X_tr, y_tr, g_tr, ds

    # ---- Step 2: Score fold-0 cases (R54-stacked pool, top-30) ----
    print(f"\n{ts()} === Step 2: Score fold-0 dev with OOF R54c ===", flush=True)
    fold0_case_data = {}  # case_idx -> {pool, gt, oof_r54c_top30 (list of (tid, score))}
    t_eval = time.time()
    for ki, i in enumerate(fold0_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        feats = featurize37(i, src_lists, pool)
        scores = oof_r54c.predict(feats)
        order = np.argsort(-scores, kind="mergesort")
        gt = cases[i]["gt"]
        top30_idx = order[:TOP_30]
        top30_tids = [pool[int(j)] for j in top30_idx]
        top30_scores = [float(scores[int(j)]) for j in top30_idx]
        fold0_case_data[i] = {
            "pool": pool,
            "gt": gt,
            "gt_in_pool": gt in pool,
            "top30_tids": top30_tids,
            "top30_scores": top30_scores,
        }
        if (ki + 1) % 400 == 0:
            print(f"    dev {ki + 1}/{len(fold0_idx)} ({time.time() - t_eval:.0f}s)",
                  flush=True)
    print(f"  done scoring fold-0", flush=True)

    # ---- Step 3: Build stacker training data (per-fold-0-case, 30-candidate groups) ----
    print(f"\n{ts()} === Step 3: Build stacker features ===", flush=True)
    # For each fold-0 case: 30 candidates, each with 5 features.
    X_stack_all = []
    y_stack_all = []
    g_stack_all = []
    case_ids_aligned = []  # which case each group belongs to (for CV split)
    for i in fold0_idx:
        d = fold0_case_data[i]
        tids = d["top30_tids"]
        scores = d["top30_scores"]
        r68_list = r68_lists_by_case.get(i, [])
        r68_rank = {str(t): r + 1 for r, (t, _) in enumerate(r68_list[:POOL_K])}
        r68_smap = {str(t): float(s) for t, s in r68_list}

        # max R54c score for rank_inv normalization (use rank position 1/r)
        case_X = []
        case_y = []
        for r, tid in enumerate(tids):
            r54_score = scores[r]
            r54_rank_inv = 1.0 / (r + 1)
            r68_r_inv = 1.0 / r68_rank.get(tid, POOL_K + 1)  # 0 if not in R68 top
            r68_pres = 1.0 if tid in r68_rank else 0.0
            r68_cos = r68_smap.get(tid, 0.0)
            case_X.append([r54_score, r54_rank_inv, r68_r_inv, r68_pres, r68_cos])
            case_y.append(1.0 if tid == d["gt"] else 0.0)
        X_stack_all.append(np.array(case_X, dtype=np.float64))
        y_stack_all.append(np.array(case_y, dtype=np.float64))
        g_stack_all.append(len(case_X))
        case_ids_aligned.append(i)

    n_cases = len(case_ids_aligned)
    print(f"  built features for {n_cases} fold-0 cases  "
          f"({sum(g_stack_all)} candidate rows)", flush=True)

    # ---- Step 4: Stacker 5-way CV within fold-0 ----
    print(f"\n{ts()} === Step 4: Stacker 5-way CV within fold-0 ===", flush=True)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_cases)
    inner_folds = np.array_split(perm, STACKER_INNER_FOLDS)
    inner_fold_idx = [np.array(f) for f in inner_folds]

    case_rows_all = []  # collected predictions per case
    for fk in range(STACKER_INNER_FOLDS):
        test_pos = set(inner_fold_idx[fk].tolist())
        train_pos = [p for p in range(n_cases) if p not in test_pos]
        # Train stacker
        Xtr = np.concatenate([X_stack_all[p] for p in train_pos], axis=0)
        ytr = np.concatenate([y_stack_all[p] for p in train_pos], axis=0)
        gtr = [g_stack_all[p] for p in train_pos]
        ds = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=list(FEAT_STACKER))
        stacker = lgb.train(STACKER_PARAMS, ds, num_boost_round=STACKER_NUM_BOOST)
        # Predict on test
        for p in inner_fold_idx[fk]:
            X_te = X_stack_all[int(p)]
            preds = stacker.predict(X_te)
            order = np.argsort(-preds, kind="mergesort")
            i = case_ids_aligned[int(p)]
            d = fold0_case_data[i]
            tids = d["top30_tids"]
            stacker_top20 = [tids[int(j)] for j in order[:TOP_K]]
            baseline_top20 = tids[:TOP_K]  # OOF R54c top-20
            gt = d["gt"]
            b_rank = -1
            s_rank = -1
            for r, t in enumerate(baseline_top20):
                if t == gt:
                    b_rank = r + 1
                    break
            for r, t in enumerate(stacker_top20):
                if t == gt:
                    s_rank = r + 1
                    break
            case_rows_all.append({
                "case_idx": i,
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "b_rank": b_rank,
                "s_rank": s_rank,
                "b_ndcg": ndcg_at_k(b_rank, TOP_K),
                "s_ndcg": ndcg_at_k(s_rank, TOP_K),
                "b_in_top20": 0 < b_rank <= TOP_K,
                "s_in_top20": 0 < s_rank <= TOP_K,
                "top1_changed": 1 if baseline_top20[0] != stacker_top20[0] else 0,
                "top20_overlap": len(set(baseline_top20) & set(stacker_top20)),
            })
    print(f"  collected {len(case_rows_all)} per-case rows", flush=True)

    # ---- Step 5: Metrics + gates ----
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    all_rows = case_rows_all
    h7_rows = [r for r in all_rows if r["n_prior_music"] == 7]
    same_rows = [r for r in all_rows if r["same_artist"]]
    diff_rows = [r for r in all_rows if not r["same_artist"]]
    h7_same = [r for r in h7_rows if r["same_artist"]]
    h7_diff = [r for r in h7_rows if not r["same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", all_rows), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows),
                       ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        b = avg(rows, "b_ndcg")
        s = avg(rows, "s_ndcg")
        metrics[name] = {"n": len(rows), "oof_r54c": b, "stacker": s, "delta": s - b}

    h7_recovered = sum(1 for r in h7_rows if r["s_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["s_in_top20"])
    h7_net = h7_recovered - h7_lost

    top1_churn = sum(r["top1_changed"] for r in all_rows)
    churn_per_80 = top1_churn / len(all_rows) * 80
    overlap_mean = avg(all_rows, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]
    all_d = metrics["all_fold0"]["delta"]

    gate_h7 = h7_d >= 0.005
    gate_same = sa_d >= -0.002
    gate_net = h7_net > 0
    gate_churn = churn_per_80 <= 25
    all_pass = gate_h7 and gate_same and gate_net and gate_churn

    if all_pass:
        verdict = "STACKER_WINS_FOLD0"
    elif h7_d >= 0.0 and sa_d >= -0.005:
        verdict = "STACKER_MARGINAL"
    else:
        verdict = "STACKER_FAIL"

    print(f"\n{ts()} === Results (within-fold-0 CV) ===", flush=True)
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  "
              f"oof_r54c={m['oof_r54c']:.4f}  stacker={m['stacker']:.4f}  "
              f"Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_recovered}  lost={h7_lost}  net={h7_net:+d}",
          flush=True)
    print(f"  top1_churn={top1_churn}  per_80={churn_per_80:.2f}  "
          f"overlap={overlap_mean:.2f}/20", flush=True)
    print(f"\n  Gates:", flush=True)
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})", flush=True)
    print(f"    same-artist Δ ≥ -0.002: {gate_same}  ({sa_d:+.4f})", flush=True)
    print(f"    h7 net > 0:           {gate_net}  ({h7_net:+d})", flush=True)
    print(f"    churn /80 ≤ 25:       {gate_churn}  ({churn_per_80:.2f})", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R71 — Stacker on OOF R54c + R68 features (fold-0)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "stacker_features": FEAT_STACKER,
        "stacker_params": STACKER_PARAMS,
        "stacker_num_boost": STACKER_NUM_BOOST,
        "n_inner_folds": STACKER_INNER_FOLDS,
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_recovered, "lost": h7_lost, "net": h7_net},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "gates": {
            "h7_delta_>=_0.005": {"value": h7_d, "pass": gate_h7},
            "same_artist_delta_>=_-0.002": {"value": sa_d, "pass": gate_same},
            "h7_net_>_0": {"value": h7_net, "pass": gate_net},
            "churn_per_80_<=_25": {"value": churn_per_80, "pass": gate_churn},
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    md = [
        "# R71 — Stacker on OOF R54c + R68 features (fold-0)",
        "",
        f"HEAD: `{out['head_sha'][:10]}`  Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Design",
        "",
        "- OOF R54c-style LR trained on folds 1-4 (6400 cases). Scores fold-0 candidates.",
        "- Take OOF R54c top-30 per fold-0 case.",
        "- For each, compute features: r54c_score, r54c_rank_inv, r68_rank_inv, r68_presence, r68_cosine.",
        f"- Stacker: LightGBM LambdaRank ({STACKER_NUM_BOOST} rounds), 5-fold CV within fold-0.",
        "",
        "## Metrics (5-fold inner CV)",
        "",
        "| Subset | n | OOF R54c top-20 | Stacker top-20 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['oof_r54c']:.4f} | {m['stacker']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_recovered}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap mean = {overlap_mean:.2f}/20",
        "",
        "## Gates",
        f"- h7 Δ ≥ +0.005:        **{gate_h7}** ({h7_d:+.4f})",
        f"- same-artist Δ ≥ -0.002: **{gate_same}** ({sa_d:+.4f})",
        f"- h7 net > 0:           **{gate_net}** ({h7_net:+d})",
        f"- churn /80 ≤ 25:       **{gate_churn}** ({churn_per_80:.2f})",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
