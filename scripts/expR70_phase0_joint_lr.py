#!/usr/bin/env python3
"""R70 Phase 0 — Joint R54+R68 LR (addition form) on fold-0.

Hypothesis. The R68/R68.1 *substitution* experiments (drop r54_*, replace with
r68_*) failed at LR conversion: Δh7=−0.081, same-artist=−0.156. That falsified
the substitution path, but it did NOT test whether R68 features carry
*incremental* signal when *added* alongside the R54 features.

R70 keeps the R54-stacked RRF pool (SW_BASELINE) bitwise identical to R54c
production — same candidates the frozen LR sees. We then train a sibling LR
with the 40-feature stack:

    FEAT_R70 = FEAT_R39_ALL + FEAT_R54 + FEAT_R68
             = 34 R39 feats + (r54_rank_inv, r54_presence, r54_cosine)
                            + (r68_rank_inv, r68_presence, r68_cosine)

This is materially different from prior failed paths:
  - R68/R68.1 substitution: pool changed (R54→R68) AND features substituted.
  - R60 matched-pool retrain: pool changed (C3 admission).
  - R70 addition: pool unchanged, features ADDED — auxiliary evidence.

Phase 0 gate (predeclared):
  - h7 nDCG Δ ≥ +0.005     vs R54c frozen-LR baseline (on R54-stacked pool)
  - same-artist Δ ≥ −0.002 vs same baseline
  - all_fold0 Δ ≥ 0
  - recovered > lost (h7)

Verdict: PROCEED_TO_PHASE_1 / ARCHIVE_PHASE_0 / EXPLORATORY.

All required artifacts already exist in cache/r68/phase0_fold0/ from R68.1.
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

# Pool weights — identical to R54c production
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
FOLD = 0

FEAT_R68 = ["r68_rank_inv", "r68_presence", "r68_cosine"]
FEAT_R70 = list(FEAT_R39_ALL) + list(FEAT_R54) + list(FEAT_R68)  # 40 features

# Paths (all already exist from R68.1)
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
R68_DIR = REPO / "cache" / "r68" / "phase0_fold0"
R68_LISTS = R68_DIR / "oof_r68_lists_fold0.json"
R68_TRAIN_LISTS = R68_DIR / "oof_r68_lists_train.json"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR70_phase0_joint_lr.json"
OUT_MD = REPO / "docs" / "r70_phase0_joint_lr_result.md"

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


def fail_fast() -> None:
    for label, p in [
        ("R12 payload", R12_CACHE),
        ("R21 OOF", R21_OOF),
        ("R54 OOF", R54_OOF),
        ("R54 frozen LR", R54_LR),
        ("R68 fold-0 lists", R68_LISTS),
        ("R68 TRAIN lists", R68_TRAIN_LISTS),
        ("W0 stats", W0_STATS),
    ]:
        if not p.exists():
            print(f"MISSING: {label} -> {p}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R70 Phase 0 — joint R54+R68 LR (addition form)")
    print(f"  features: {len(FEAT_R70)} = "
          f"R39({len(FEAT_R39_ALL)}) + R54({len(FEAT_R54)}) + R68({len(FEAT_R68)})")
    print(f"  pool: R54-stacked (SW_BASELINE), unchanged from R54c")
    print("=" * 70)
    fail_fast()

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
    print(f"  fold-0 held-out: {len(fold0_idx)} cases  train: {len(train_idx)} cases",
          flush=True)

    print(f"{ts()} Loading R68 fold-0 + TRAIN lists ...", flush=True)
    with open(R68_LISTS) as f:
        r68_data = json.load(f)
    r68_lists_raw = r68_data["lists"]
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"].get("val_idx")
    if set(r68_val_idx) != set(fold0_idx):
        raise RuntimeError("R68 val_idx mismatch vs W0 fold-0")
    r68_lists_by_case: dict[int, list[tuple[str, float]]] = {}
    for k_pos, case_idx in enumerate(r68_val_idx):
        r68_lists_by_case[int(case_idx)] = [
            (str(t), float(s)) for t, s in r68_lists_raw[k_pos]]

    with open(R68_TRAIN_LISTS) as f:
        r68_train_lists_raw = json.load(f)

    def _case_id_str(c: dict) -> str:
        return c.get("case_id") or f"{c['session_id']}_{c['turn_number']}"
    r68_train_lists_by_case: dict[int, list[tuple[str, float]]] = {}
    for i, c in enumerate(cases):
        cid = _case_id_str(c)
        if cid in r68_train_lists_raw:
            r68_train_lists_by_case[i] = [
                (str(t), float(s)) for t, s in r68_train_lists_raw[cid]]
    print(f"  R68 dev lists: {len(r68_lists_by_case)}  "
          f"R68 train lists: {len(r68_train_lists_by_case)}", flush=True)

    print(f"{ts()} Building case index (ALS, R21, R54 source lists) ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def r68_rank_map_for(case_idx: int, is_train: bool) -> dict[str, int]:
        if is_train:
            lst = r68_train_lists_by_case.get(case_idx, [])
        else:
            lst = r68_lists_by_case.get(case_idx, [])
        return {str(t): r + 1 for r, (t, _) in enumerate(lst[:POOL_K])}

    def r68_score_map_for(case_idx: int, is_train: bool) -> dict[str, float]:
        if is_train:
            lst = r68_train_lists_by_case.get(case_idx, [])
        else:
            lst = r68_lists_by_case.get(case_idx, [])
        return {str(t): float(s) for t, s in lst}

    def featurize40(case_idx: int, src_lists: dict[str, list[str]],
                    pool: list[str], is_train: bool) -> np.ndarray:
        """Build the 40-feature matrix: R39 + R54 + R68.

        _featurize_row returns (n_pool, 37) with the trailing 3 cols being
        (rank_inv, presence, cosine) for whatever rank/score map was passed.
        We call it twice (once with R54 maps, once with R68 maps) and stitch
        the last 3 cols of the second call onto the first → 40 cols total.
        """
        case = cases[case_idx]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}

        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[case_idx],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )

        r68_rank_map = r68_rank_map_for(case_idx, is_train)
        r68_smap = r68_score_map_for(case_idx, is_train)
        feats_r68 = _featurize_row(
            pool, src_lists, r21_rank_map, r68_rank_map, r68_smap,
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )

        # feats_r54 cols: [R39_ALL ... | r54_rank_inv | r54_presence | r54_cosine]
        # feats_r68 cols: [R39_ALL ... | r68_rank_inv | r68_presence | r68_cosine]
        # Concatenate last 3 of feats_r68 onto feats_r54 → 40 cols.
        return np.concatenate([feats_r54, feats_r68[:, -3:]], axis=1)

    # ---- Step 1: Build TRAIN feature matrix (R54-stacked pool, 40 features) ----
    print(f"\n{ts()} === Step 1: TRAIN feature matrix ({len(train_idx)} cases) ===",
          flush=True)
    X_train, y_train, groups_train = [], [], []
    gt_in_pool_train = 0
    t_feat = time.time()
    train_missing_r68 = 0
    for ki, i in enumerate(train_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        # R54-stacked pool (bitwise identical to R54c)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        if gi >= 0:
            gt_in_pool_train += 1
        if i not in r68_train_lists_by_case:
            train_missing_r68 += 1
        feats = featurize40(i, src_lists, pool, is_train=True)
        for k_row in range(len(pool)):
            X_train.append(feats[k_row])
            y_train.append(1.0 if k_row == gi else 0.0)
        groups_train.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} "
                  f"({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  TRAIN pool_hit: {gt_in_pool_train}/{len(train_idx)} "
          f"({gt_in_pool_train/max(len(train_idx),1):.4f})", flush=True)
    print(f"  TRAIN cases missing R68 list: {train_missing_r68} "
          f"(features default to 0,0,0)", flush=True)

    # ---- Step 2: Train sibling LR ----
    print(f"\n{ts()} === Step 2: Train sibling LR ({len(FEAT_R70)} features) ===",
          flush=True)
    ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                     label=np.array(y_train, dtype=np.float64),
                     group=groups_train, feature_name=list(FEAT_R70))
    sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    sibling_lr_path = REPO / "cache" / "r70_phase0_sibling_lr.txt"
    sibling_lr.save_model(str(sibling_lr_path))
    print(f"  Saved sibling LR -> {sibling_lr_path}", flush=True)

    # Free train data
    del X_train, y_train, groups_train, ds

    # ---- Step 3: Score fold-0 dev — baseline (R54c frozen LR) vs R70 sibling ----
    print(f"\n{ts()} === Step 3: Score fold-0 dev ===", flush=True)
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))
    if r54_baseline_ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError(
            f"R54 LR feature count mismatch: "
            f"model={r54_baseline_ranker.num_feature()} expected={len(FEAT_ALL)}")

    fold0_rows: list[dict[str, Any]] = []
    t_eval = time.time()
    dev_missing_r68 = 0
    for ki, i in enumerate(fold0_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        # R54-stacked pool (same as R54c)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gt_in_pool = gt in pool

        # ---- Baseline: R54c frozen LR over the R54-stacked pool ----
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        baseline_feats = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            cases[i]["user_query"], cases[i]["history"], cases[i]["music_turns"],
            set(cases[i]["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album,
        )
        b_scores = r54_baseline_ranker.predict(baseline_feats)
        b_order = np.argsort(-b_scores, kind="mergesort")
        b_gt_rank = -1
        if gt_in_pool:
            pos = np.where(b_order == pool.index(gt))[0]
            if len(pos):
                b_gt_rank = int(pos[0]) + 1

        # ---- R70: sibling LR over the SAME R54-stacked pool, 40 features ----
        if i not in r68_lists_by_case:
            dev_missing_r68 += 1
        r70_feats = featurize40(i, src_lists, pool, is_train=False)
        r70_scores = sibling_lr.predict(r70_feats)
        r_order = np.argsort(-r70_scores, kind="mergesort")
        r_gt_rank = -1
        if gt_in_pool:
            pos = np.where(r_order == pool.index(gt))[0]
            if len(pos):
                r_gt_rank = int(pos[0]) + 1

        # top-20 churn diagnostics
        b_top20 = [pool[int(j)] for j in b_order[:TOP_K]]
        r_top20 = [pool[int(j)] for j in r_order[:TOP_K]]
        overlap = len(set(b_top20) & set(r_top20))
        top1_changed = 1 if b_top20[0] != r_top20[0] else 0

        fold0_rows.append({
            "case_idx": i,
            "session_id": cases[i]["session_id"],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "gt_in_pool": gt_in_pool,
            "b_gt_rank": b_gt_rank,
            "r_gt_rank": r_gt_rank,
            "b_ndcg_at_20": ndcg_at_k(b_gt_rank, TOP_K),
            "r_ndcg_at_20": ndcg_at_k(r_gt_rank, TOP_K),
            "top20_overlap": overlap,
            "top1_changed": top1_changed,
            "b_in_top20": b_gt_rank > 0 and b_gt_rank <= TOP_K,
            "r_in_top20": r_gt_rank > 0 and r_gt_rank <= TOP_K,
        })
        if (ki + 1) % 200 == 0:
            print(f"    dev {ki + 1}/{len(fold0_idx)} ({time.time() - t_eval:.0f}s)",
                  flush=True)
    print(f"  dev cases missing R68 list: {dev_missing_r68}", flush=True)

    # ---- Step 4: Compute deltas + gates ----
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
        r = avg(rows, "r_ndcg_at_20")
        metrics[name] = {"n": len(rows), "baseline": b, "r70": r, "delta": r - b}

    h7_recovered = sum(1 for r in h7_rows if r["r_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["r_in_top20"])
    h7_net = h7_recovered - h7_lost

    top1_churn = sum(r["top1_changed"] for r in fold0_rows)
    churn_per_80 = top1_churn / len(fold0_rows) * 80
    overlap_mean = avg(fold0_rows, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]
    all_d = metrics["all_fold0"]["delta"]
    gate_h7 = h7_d >= 0.005
    gate_same = sa_d >= -0.002
    gate_all = all_d >= 0.0
    gate_net = h7_net > 0
    all_pass = gate_h7 and gate_same and gate_all and gate_net
    expl_pass = (h7_d >= 0.0 and sa_d >= -0.005)

    if all_pass:
        verdict = "PROCEED_TO_PHASE_1"
    elif expl_pass:
        verdict = "EXPLORATORY"
    else:
        verdict = "ARCHIVE_PHASE_0"

    print(f"\n{ts()} === Results ===", flush=True)
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  base={m['baseline']:.4f}  "
              f"r70={m['r70']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7_recovered={h7_recovered}  h7_lost={h7_lost}  net={h7_net:+d}",
          flush=True)
    print(f"  top1_changed={top1_churn}  churn_per_80={churn_per_80:.2f}  "
          f"overlap_mean={overlap_mean:.2f}/20", flush=True)
    print(f"\n  Gates:", flush=True)
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})", flush=True)
    print(f"    same-artist Δ ≥ −0.002: {gate_same}  ({sa_d:+.4f})", flush=True)
    print(f"    all_fold0 Δ ≥ 0:      {gate_all}  ({all_d:+.4f})", flush=True)
    print(f"    h7 net > 0:           {gate_net}  ({h7_net:+d})", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    # ---- Step 5: Save artifacts ----
    out = {
        "experiment": "R70 Phase 0 — joint R54+R68 LR addition (fold-0)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "branch": "r70-joint-r54-r68-features",
        "head_sha": head_sha(),
        "verdict": verdict,
        "fold": FOLD,
        "n_fold0": len(fold0_idx),
        "n_train": len(train_idx),
        "feature_stack": {
            "names": FEAT_R70, "n_features": len(FEAT_R70),
            "structure": "FEAT_R39_ALL + FEAT_R54 + FEAT_R68",
        },
        "pool": {
            "weights": SW_BASELINE, "K": POOL_K,
            "note": "R54-stacked, bitwise identical to R54c production pool",
        },
        "missing_r68": {"train": train_missing_r68, "dev": dev_missing_r68},
        "metrics": metrics,
        "h7_recovery": {
            "recovered": h7_recovered, "lost": h7_lost, "net": h7_net,
        },
        "churn": {
            "top1_changed": top1_churn, "churn_per_80": churn_per_80,
            "top20_overlap_mean": overlap_mean,
        },
        "gates": {
            "h7_delta_>=_0.005": {"value": h7_d, "pass": gate_h7},
            "same_artist_delta_>=_-0.002": {"value": sa_d, "pass": gate_same},
            "all_fold0_delta_>=_0": {"value": all_d, "pass": gate_all},
            "h7_net_>_0": {"value": h7_net, "pass": gate_net},
        },
        "design": (
            "ADDITION form: pool unchanged (R54-stacked), feature stack expanded "
            "from 37 (R54c LR) to 40 by appending R68's (rank_inv, presence, "
            "cosine) alongside R54's. Sibling LR retrained on fold-0 train; "
            "evaluated against R54c frozen LR on the SAME pool. Distinct from "
            "R68 (substitution, dropped R54 features) and R60 (matched-pool "
            "retrain with pool changes)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# R70 Phase 0 — joint R54+R68 LR (addition form), fold-0",
        "",
        f"Branch: `r70-joint-r54-r68-features`  HEAD: `{out['head_sha'][:10]}`  "
        f"Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Design",
        "",
        "**Pool**: R54-stacked RRF top-300 (`SW_BASELINE`), bitwise identical to R54c production.",
        "",
        f"**Features (40)**: FEAT_R39_ALL ({len(FEAT_R39_ALL)}) + FEAT_R54 ({len(FEAT_R54)}) + FEAT_R68 ({len(FEAT_R68)})",
        "",
        "**Sibling LR**: LightGBM LambdaRank, 300 rounds, same hyperparams as R54c, retrained on fold-0 train cases.",
        "",
        "Distinct from prior failed paths:",
        "- R68/R68.1 substitution: pool changed (R54→R68) AND r54_* features dropped.",
        "- R60 matched-pool: pool changed (C3 admission).",
        "- R70 addition: pool unchanged, r68_* features added.",
        "",
        "## Metrics",
        "",
        "| Subset | n | Baseline (R54c frozen LR) | R70 sibling LR | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r70']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_recovered}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80={churn_per_80:.2f}  top-20 overlap mean={overlap_mean:.2f}/20",
        f"- dev cases missing R68 list: {dev_missing_r68}  train cases missing R68: {train_missing_r68}",
        "",
        "## Gates (predeclared)",
        "",
        f"- h7 Δ ≥ +0.005: **{gate_h7}** ({h7_d:+.4f})",
        f"- same-artist Δ ≥ −0.002: **{gate_same}** ({sa_d:+.4f})",
        f"- all_fold0 Δ ≥ 0: **{gate_all}** ({all_d:+.4f})",
        f"- h7 net > 0: **{gate_net}** ({h7_net:+d})",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
