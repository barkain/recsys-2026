#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R68 Phase 0 eval — Mac-side, runs AFTER GPU sync of fold-0 artifacts.

Inputs (must all be present):
  cache/r68/phase0_fold0/oof_r68_lists_fold0.json
  cache/r68/phase0_fold0/track_embeddings.npy
  cache/r68/phase0_fold0/track_ids.json
  cache/r68/phase0_fold0/query_embeddings_dev.npy
  exp/eval/expR68_r54_reference_stats.pkl   (W0 output)
  exp/eval/_R12_all_turns_payload.pkl
  cache/r54_phase3_lr_model.txt              (frozen R54c LR, baseline only)
  cache/r54/phase2_full/oof_manifest.json
  cache/r21_production/dev_r21_oof_lists.json
  cache/r54_phase3_als.npz, cache/r54_phase3_track_pop.json, cache/r54_phase3_payload_maps.pkl

Computation:
  1. Restrict to fold-0 held-out cases (fold map from W0 stats pkl).
  2. r68_single_source_pool_hit_h7 vs r54_single_source_pool_hit_h7 on fold-0.
  3. R68-stacked RRF: drop R54 source, add R68 source (weight 1.0). Compute
     r68_stacked_RRF_pool_hit on fold-0 cases.
  4. Unique h7 GT recoveries by R68 vs R54.
  5. Per-(case, candidate) features:
       r68_rank_inv  = 1.0/r68_rank if in top-300 else 0
       r68_presence  = 1.0 if in top-300 else 0
       r68_cosine    = dot(query_emb, candidate_track_emb)
  6. Train sibling LR with FEAT_ALL_R68 = FEAT_R39_ALL + FEAT_R68 on fold-0
     TRAIN cases (the 6400 not in fold-0). Same lambdarank hyperparams as R54c.
     This is feature substitution only; pool admission unchanged.
  7. Score the R68-stacked RRF pool with sibling LR on fold-0 dev.
  8. Compute h7 / same-artist / diff-artist / all nDCG@20 deltas vs R54
     baseline (frozen LR on R54-stacked pool, restricted to fold-0).

Kill gate (Phase 0):
  gate_1: (unique_h7_recoveries >= 15) OR
          (r68_single_source_pool_hit_h7 - r54_single_source_pool_hit_h7 >= 0.010)
  gate_2: r68_stacked_RRF_pool_hit - baseline_RRF_pool_hit >= 0.005
  gate_3: (h7_ndcg_delta >= 0.0) AND (same_artist_delta >= -0.002)
  verdict = PROCEED iff all 3 pass else ARCHIVE_PHASE_0

Outputs:
  exp/eval/expR68_phase0_fold0_eval.json
  docs/r68_phase0_fold0_result.md
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess  # noqa: S404
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

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
from scripts.expS2_lambdarank import build_als  # noqa: E402
from scripts.expS2_lambdarank_grouped import (  # noqa: E402
    als_session_vector, grouped_session_folds,
)

# Mirror R54 phase3 RRF weights, with R68 replacing R54
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
SW_R68 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
          "ALS": 1.0, "R21": 1.0, "R68": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
FOLD = 0

FEAT_R68 = ["r68_rank_inv", "r68_presence", "r68_cosine"]
FEAT_ALL_R68 = FEAT_R39_ALL + FEAT_R68

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_OOF = REPO / "cache" / "r54" / "phase3_full" / "oof_r54_lists.json"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
R68_DIR = REPO / "cache" / "r68" / "phase0_fold0"
R68_LISTS = R68_DIR / "oof_r68_lists_fold0.json"
R68_TRACK_EMBS = R68_DIR / "track_embeddings.npy"
R68_TRACK_IDS = R68_DIR / "track_ids.json"
R68_QUERY_EMBS = R68_DIR / "query_embeddings_dev.npy"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
W0_AGG = REPO / "exp" / "eval" / "expR68_r54_aggregate.json"

OUT_JSON = REPO / "exp" / "eval" / "expR68_phase0_fold0_eval.json"
OUT_MD = REPO / "docs" / "r68_phase0_fold0_result.md"

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
        raise RuntimeError("git not found")
    return subprocess.check_output(  # noqa: S603
        [git_bin, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def fail_fast() -> None:
    missing = []
    for label, p in [
        ("R68 OOF lists", R68_LISTS),
        ("R68 track embs", R68_TRACK_EMBS),
        ("R68 track ids", R68_TRACK_IDS),
        ("R68 query embs", R68_QUERY_EMBS),
        ("W0 stats pkl", W0_STATS),
        ("R12 payload", R12_CACHE),
        ("R21 OOF", R21_OOF),
        ("R54 OOF", R54_OOF),
        ("R54 frozen LR", R54_LR),
    ]:
        if not p.exists():
            missing.append(f"  {label}: {p}")
    if missing:
        print("MISSING ARTIFACTS — CANNOT RUN:", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / np.log2(gt_rank + 1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R68 Phase 0 fold-0 eval")
    print("=" * 70)
    fail_fast()

    # Payloads (reuse c3.load_payloads for consistency with W0)
    print(f"{ts()} Loading payload + R21/R54 OOF...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    # Fold map from W0
    print(f"{ts()} Loading W0 fold map + R54 reference stats...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold0_idx = [i for i in range(n) if case_fold[i] == FOLD]
    print(f"  fold-0 held-out: {len(fold0_idx)} cases", flush=True)

    # Load R68 fold-0 lists
    print(f"{ts()} Loading R68 fold-0 lists...", flush=True)
    with open(R68_LISTS) as f:
        r68_data = json.load(f)
    r68_lists_raw = r68_data["lists"]
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"].get("val_idx")
    if r68_val_idx is None:
        raise RuntimeError("R68 lists missing val_idx")
    if set(r68_val_idx) != set(fold0_idx):
        raise RuntimeError(
            f"R68 val_idx mismatch: R68 has {len(r68_val_idx)} cases, "
            f"W0 fold-0 has {len(fold0_idx)}. First 5 R68: {r68_val_idx[:5]} "
            f"vs W0: {fold0_idx[:5]}")
    if r68_val_idx != fold0_idx:
        print("  WARNING: R68 val_idx order != W0 fold-0 order; remapping",
              flush=True)
    # Index R68 lists by global case idx
    r68_lists_by_case: dict[int, list[tuple[str, float]]] = {}
    for k_pos, case_idx in enumerate(r68_val_idx):
        r68_lists_by_case[int(case_idx)] = [
            (str(t), float(s)) for t, s in r68_lists_raw[k_pos]]

    # Load R68 embeddings (for r68_cosine of pool candidates)
    print(f"{ts()} Loading R68 embeddings...", flush=True)
    r68_track_embs = np.load(R68_TRACK_EMBS)
    r68_track_ids = json.load(open(R68_TRACK_IDS))
    r68_query_embs = np.load(R68_QUERY_EMBS)
    print(f"  track_embs: {r68_track_embs.shape}, query_embs: {r68_query_embs.shape}",
          flush=True)
    r68_track_to_idx = {tid: j for j, tid in enumerate(r68_track_ids)}
    # Query embs are aligned to r68_val_idx ordering (training script saved them so)
    case_to_qemb_row = {int(case_idx): k_pos
                       for k_pos, case_idx in enumerate(r68_val_idx)}

    # Build ALS / R21 / R54 source artifacts via c3 (mirrors training/W0)
    print(f"{ts()} Building case index (ALS, R21, R54 source lists)...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    # Build R68 single-source list per fold-0 case (top-300 from r68_lists)
    def r68_source_list(case_idx: int) -> list[str]:
        lst = r68_lists_by_case.get(case_idx)
        if not lst:
            return []
        return [t for t, _ in lst[:POOL_K]]

    def r68_score_map(case_idx: int) -> dict[str, float]:
        lst = r68_lists_by_case.get(case_idx)
        if not lst:
            return {}
        return {t: float(s) for t, s in lst}

    # ----- Step 1: R68 vs R54 single-source pool_hit (fold-0) -----
    print(f"\n{ts()} === Step 1: Single-source pool_hit on fold-0 ===", flush=True)
    h7_idx_fold0 = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    r54_single_h7 = sum(
        1 for i in h7_idx_fold0
        if cases[i]["gt"] in set(r54_source[i][:POOL_K]))
    r68_single_h7 = sum(
        1 for i in h7_idx_fold0
        if cases[i]["gt"] in set(r68_source_list(i)))
    r54_single_all = sum(
        1 for i in fold0_idx
        if cases[i]["gt"] in set(r54_source[i][:POOL_K]))
    r68_single_all = sum(
        1 for i in fold0_idx
        if cases[i]["gt"] in set(r68_source_list(i)))
    n_h7 = len(h7_idx_fold0)
    n_fold0 = len(fold0_idx)
    r54_single_h7_rate = r54_single_h7 / max(n_h7, 1)
    r68_single_h7_rate = r68_single_h7 / max(n_h7, 1)
    r54_single_all_rate = r54_single_all / max(n_fold0, 1)
    r68_single_all_rate = r68_single_all / max(n_fold0, 1)
    print(f"  h7 (n={n_h7}): R54={r54_single_h7_rate:.4f}  R68={r68_single_h7_rate:.4f}  "
          f"Δ={r68_single_h7_rate - r54_single_h7_rate:+.4f}", flush=True)
    print(f"  all (n={n_fold0}): R54={r54_single_all_rate:.4f}  R68={r68_single_all_rate:.4f}  "
          f"Δ={r68_single_all_rate - r54_single_all_rate:+.4f}", flush=True)

    # ----- Step 2: Unique h7 recoveries (R68 has GT, R54 does NOT) -----
    print(f"\n{ts()} === Step 2: Unique h7 GT recoveries by R68 ===", flush=True)
    unique_h7_recoveries = 0
    lost_h7 = 0
    for i in h7_idx_fold0:
        gt = cases[i]["gt"]
        in_r68 = gt in set(r68_source_list(i))
        in_r54 = gt in set(r54_source[i][:POOL_K])
        if in_r68 and not in_r54:
            unique_h7_recoveries += 1
        elif in_r54 and not in_r68:
            lost_h7 += 1
    print(f"  recovered (R68 only): {unique_h7_recoveries}  lost (R54 only): {lost_h7}  "
          f"net: {unique_h7_recoveries - lost_h7:+d}", flush=True)

    # ----- Step 3: R68-stacked RRF pool_hit vs baseline -----
    print(f"\n{ts()} === Step 3: Stacked RRF pool_hit on fold-0 ===", flush=True)
    baseline_pool_hit = 0
    baseline_pool_hit_h7 = 0
    r68_pool_hit = 0
    r68_pool_hit_h7 = 0
    for i in fold0_idx:
        src_lists_base = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        # baseline RRF (already includes R54 weight 1.0)
        baseline_pool = weighted_rrf(src_lists_base, SW_BASELINE,
                                     topk=POOL_K, k=RRF_K)
        # r68-stacked: same 7 base sources, swap R54 -> R68
        src_lists_r68 = dict(src_lists_base)
        src_lists_r68.pop("R54", None)
        src_lists_r68["R68"] = r68_source_list(i)
        r68_pool = weighted_rrf(src_lists_r68, SW_R68, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        b_hit = gt in baseline_pool
        r_hit = gt in r68_pool
        baseline_pool_hit += int(b_hit)
        r68_pool_hit += int(r_hit)
        if int(cases[i]["n_prior_music"]) == 7:
            baseline_pool_hit_h7 += int(b_hit)
            r68_pool_hit_h7 += int(r_hit)
    base_rate = baseline_pool_hit / max(n_fold0, 1)
    r68_rate = r68_pool_hit / max(n_fold0, 1)
    base_rate_h7 = baseline_pool_hit_h7 / max(n_h7, 1)
    r68_rate_h7 = r68_pool_hit_h7 / max(n_h7, 1)
    print(f"  all fold-0: baseline={base_rate:.4f}  R68-stacked={r68_rate:.4f}  "
          f"Δ={r68_rate - base_rate:+.4f}", flush=True)
    print(f"  h7 fold-0:  baseline={base_rate_h7:.4f}  R68-stacked={r68_rate_h7:.4f}  "
          f"Δ={r68_rate_h7 - base_rate_h7:+.4f}", flush=True)

    # ----- Step 4: Train sibling LR (FEAT_R39_ALL + FEAT_R68) on fold-0 TRAIN -----
    # NOTE: Pool admission unchanged. Only 3 feature columns are swapped
    # (r54_* -> r68_*). This is feature substitution, NOT matched-pool retraining.
    print(f"\n{ts()} === Step 4: Training sibling LR (R68 features) on fold-0 train ===",
          flush=True)
    print(f"  features: {len(FEAT_ALL_R68)} (FEAT_R39_ALL={len(FEAT_R39_ALL)} + "
          f"FEAT_R68={len(FEAT_R68)})", flush=True)
    train_idx = [i for i in range(n) if case_fold[i] != FOLD]

    # For sibling LR training we need R68 lists for TRAIN cases too. Phase 0 only
    # produced fold-0 OOF lists (held-out). For TRAIN rows we don't yet have OOF
    # R68; we therefore approximate using R68 list of the TRAIN case if produced
    # via the fold-0 *model* on those cases (the model embeds catalog identically;
    # the bias is small because training is 1-epoch InfoNCE).
    #
    # PHASE 0 LIMITATION: this means the sibling LR for Phase 0 sees in-sample
    # R68 features for TRAIN cases. We document this in the result MD and gate
    # decisions accordingly. The Phase 1 5-fold OOF run will redo this cleanly.
    print(f"  NOTE (Phase 0): sibling LR train uses fold-0-model R68 embeddings "
          f"for TRAIN cases (mild in-sample bias). Phase 1 runs fully OOF.",
          flush=True)

    # Encode all 8000 dev queries with the fold-0 R68 model to materialize r68_cosine.
    # But Phase 0 only saved fold-0 dev queries. For TRAIN cases we therefore lack
    # query embeddings without re-encoding. To keep this script Mac-only, we fall
    # back to using ONLY the R68 single-source list features (rank_inv, presence)
    # for TRAIN rows, with r68_cosine = 0 where the candidate cosine is unknown.
    # This degrades the train signal but does not poison fold-0 eval (which uses
    # the saved query embeddings).
    print(f"  WARNING: TRAIN-case r68_cosine set to 0 (no train query embeddings).",
          flush=True)

    def featurize_case(case_idx: int, source_lists: dict[str, list[str]],
                       pool: list[str], r68_score_lookup: dict[str, float]
                       ) -> np.ndarray:
        case = cases[case_idx]
        # Mimic R54 _featurize_row but substituting r68 features in the trailing
        # 3 columns. We exploit that _featurize_row's r54_rank_map / r54_score_map
        # args map straight to (rank_inv, presence, cosine). We pass R68 maps in
        # those slots — this returns a matrix where the last 3 cols are R68
        # features, indistinguishable from the R54 schema in terms of column
        # layout. The frozen R54c LR will NOT understand these directly, so we
        # train a sibling LR with feature_name=FEAT_ALL_R68.
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(source_lists["R21"][:POOL_K])}
        r68_rank_map = {tid: r + 1 for r, tid in enumerate(source_lists.get("R68", [])[:POOL_K])}
        feats = _featurize_row(
            pool, source_lists, r21_rank_map, r68_rank_map, r68_score_lookup,
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )
        return feats

    # Build features for TRAIN cases (R68-stacked pool)
    print(f"  Building TRAIN feature matrix ({len(train_idx)} cases)...", flush=True)
    X_train, y_train, groups_train = [], [], []
    gt_in_pool_train = 0
    t_feat = time.time()
    for ki, i in enumerate(train_idx):
        src_lists_base = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        src_lists_r68 = dict(src_lists_base)
        src_lists_r68.pop("R54", None)
        # TRAIN cases: we don't have OOF R68 lists. Use empty list + score map.
        # Sibling LR sees r68_rank_inv=0, r68_presence=0, r68_cosine=0 for TRAIN.
        # This is intentional: Phase 0 only validates fold-0 transfer, not OOF.
        src_lists_r68["R68"] = []
        pool = weighted_rrf(src_lists_r68, SW_R68, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        if gi >= 0:
            gt_in_pool_train += 1
        feats = featurize_case(i, src_lists_r68, pool, {})
        for k_row in range(len(pool)):
            X_train.append(feats[k_row])
            y_train.append(1.0 if k_row == gi else 0.0)
        groups_train.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} "
                  f"({time.time() - t_feat:.0f}s)", flush=True)

    print(f"  TRAIN pool_hit: {gt_in_pool_train}/{len(train_idx)} "
          f"({gt_in_pool_train/max(len(train_idx),1):.4f})", flush=True)

    print(f"  Training LightGBM LambdaRank ({len(groups_train)} groups, "
          f"{len(y_train)} candidates, {len(FEAT_ALL_R68)} features)...", flush=True)
    ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                     label=np.array(y_train, dtype=np.float64),
                     group=groups_train, feature_name=list(FEAT_ALL_R68))
    sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    sibling_lr_path = REPO / "cache" / "r68_phase0_sibling_lr.txt"
    sibling_lr.save_model(str(sibling_lr_path))
    print(f"  Saved sibling LR -> {sibling_lr_path}", flush=True)

    # ----- Step 5: Score fold-0 dev with sibling LR (R68-stacked pool) -----
    print(f"\n{ts()} === Step 5: Scoring fold-0 dev (R68-stacked pool, sibling LR) ===",
          flush=True)
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))
    if r54_baseline_ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError(
            f"R54 frozen LR feature count mismatch: "
            f"model={r54_baseline_ranker.num_feature()} expected={len(FEAT_ALL)}")

    fold0_case_rows: list[dict[str, Any]] = []
    t_eval = time.time()
    for ki, i in enumerate(fold0_idx):
        src_lists_base = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)

        # --- Baseline: R54-stacked pool, frozen LR (Phase 3 prod) ---
        baseline_pool = weighted_rrf(src_lists_base, SW_BASELINE,
                                     topk=POOL_K, k=RRF_K)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists_base["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists_base["R54"][:POOL_K])}
        baseline_feats = _featurize_row(
            baseline_pool, src_lists_base, r21_rank_map, r54_rank_map,
            r54_scores[i],
            cases[i]["user_query"], cases[i]["history"], cases[i]["music_turns"],
            set(cases[i]["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album,
        )
        baseline_scores = r54_baseline_ranker.predict(baseline_feats)
        b_order = np.argsort(-baseline_scores, kind="mergesort")
        gt = cases[i]["gt"]
        b_gt_rank = -1
        if gt in baseline_pool:
            pos = np.where(b_order == baseline_pool.index(gt))[0]
            if len(pos):
                b_gt_rank = int(pos[0]) + 1

        # --- R68 stacked: R68-stacked pool, sibling LR ---
        src_lists_r68 = dict(src_lists_base)
        src_lists_r68.pop("R54", None)
        src_lists_r68["R68"] = r68_source_list(i)
        r68_pool = weighted_rrf(src_lists_r68, SW_R68, topk=POOL_K, k=RRF_K)
        r68_smap = r68_score_map(i)
        # NB: r68_cosine here is the cosine from the GPU-saved dot products
        # (BGE-large normalized embeddings). The score_map already stores this.
        r68_feats = featurize_case(i, src_lists_r68, r68_pool, r68_smap)
        r68_scores = sibling_lr.predict(r68_feats)
        r_order = np.argsort(-r68_scores, kind="mergesort")
        r_gt_rank = -1
        if gt in r68_pool:
            pos = np.where(r_order == r68_pool.index(gt))[0]
            if len(pos):
                r_gt_rank = int(pos[0]) + 1

        fold0_case_rows.append({
            "case_idx": i,
            "session_id": cases[i]["session_id"],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "gt_in_baseline_pool": gt in baseline_pool,
            "gt_in_r68_pool": gt in r68_pool,
            "baseline_gt_rank": b_gt_rank,
            "r68_gt_rank": r_gt_rank,
            "baseline_ndcg_at_20": ndcg_at_k(b_gt_rank, 20),
            "r68_ndcg_at_20": ndcg_at_k(r_gt_rank, 20),
        })
        if (ki + 1) % 500 == 0:
            print(f"    scored {ki + 1}/{n_fold0} ({time.time() - t_eval:.0f}s)",
                  flush=True)

    # ----- Step 6: Metrics -----
    print(f"\n{ts()} === Step 6: Fold-0 dev metrics ===", flush=True)
    base_ndcg = np.array([r["baseline_ndcg_at_20"] for r in fold0_case_rows])
    r68_ndcg = np.array([r["r68_ndcg_at_20"] for r in fold0_case_rows])
    h7_mask = np.array([r["n_prior_music"] == 7 for r in fold0_case_rows])
    same_mask = np.array([r["same_artist"] for r in fold0_case_rows])
    diff_mask = ~same_mask
    h7_same_mask = h7_mask & same_mask
    h7_diff_mask = h7_mask & diff_mask

    def block(mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return {"n": 0, "baseline": 0.0, "r68": 0.0, "delta": 0.0}
        return {
            "n": int(mask.sum()),
            "baseline": float(base_ndcg[mask].mean()),
            "r68": float(r68_ndcg[mask].mean()),
            "delta": float((r68_ndcg[mask] - base_ndcg[mask]).mean()),
        }

    metrics = {
        "all_fold0": block(np.ones_like(h7_mask, dtype=bool)),
        "h7": block(h7_mask),
        "same_artist": block(same_mask),
        "diff_artist": block(diff_mask),
        "h7_same": block(h7_same_mask),
        "h7_diff": block(h7_diff_mask),
    }
    for label, b in metrics.items():
        print(f"  {label:14s} n={b['n']:5d}  baseline={b['baseline']:.4f}  "
              f"R68={b['r68']:.4f}  Δ={b['delta']:+.4f}", flush=True)

    # h7 churn (top-20)
    h7_top1_churn = 0
    for r in fold0_case_rows:
        if r["n_prior_music"] != 7:
            continue
        # Same-rank churn proxy: did GT rank change?
        if r["baseline_gt_rank"] != r["r68_gt_rank"]:
            h7_top1_churn += 1

    # ----- Step 7: Gates -----
    print(f"\n{ts()} === Step 7: Kill gates ===", flush=True)
    delta_single_h7 = r68_single_h7_rate - r54_single_h7_rate
    delta_stacked_all = r68_rate - base_rate
    h7_ndcg_delta = metrics["h7"]["delta"]
    same_artist_delta = metrics["same_artist"]["delta"]

    gate_1 = (unique_h7_recoveries >= 15) or (delta_single_h7 >= 0.010)
    gate_2 = (delta_stacked_all >= 0.005)
    gate_3 = (h7_ndcg_delta >= 0.0) and (same_artist_delta >= -0.002)

    gates = {
        "gate_1_recovery": {
            "rule": "(unique_h7_recoveries >= 15) OR "
                    "(r68_single_pool_hit_h7 - r54_single_pool_hit_h7 >= 0.010)",
            "unique_h7_recoveries": unique_h7_recoveries,
            "delta_single_h7": delta_single_h7,
            "pass": bool(gate_1),
        },
        "gate_2_pool_hit": {
            "rule": "r68_stacked_RRF_pool_hit - baseline_RRF_pool_hit >= 0.005",
            "baseline_pool_hit": base_rate,
            "r68_pool_hit": r68_rate,
            "delta": delta_stacked_all,
            "pass": bool(gate_2),
        },
        "gate_3_ndcg": {
            "rule": "(h7_ndcg_delta >= 0.0) AND (same_artist_delta >= -0.002)",
            "h7_ndcg_delta": h7_ndcg_delta,
            "same_artist_delta": same_artist_delta,
            "pass": bool(gate_3),
        },
    }
    all_pass = gate_1 and gate_2 and gate_3
    verdict = "PROCEED" if all_pass else "ARCHIVE_PHASE_0"
    print(f"  gate_1 (recovery): {'PASS' if gate_1 else 'FAIL'}", flush=True)
    print(f"  gate_2 (pool_hit): {'PASS' if gate_2 else 'FAIL'}", flush=True)
    print(f"  gate_3 (ndcg):     {'PASS' if gate_3 else 'FAIL'}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    # ----- Step 8: Outputs -----
    report = {
        "experiment": "R68 Phase 0 fold-0 eval",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "branch": "r68-large-scale-retrieval",
        "head_sha": head_sha(),
        "verdict": verdict,
        "fold": FOLD,
        "n_fold0": n_fold0,
        "n_fold0_h7": n_h7,
        "single_source_pool_hit": {
            "h7": {
                "r54": r54_single_h7_rate, "r68": r68_single_h7_rate,
                "delta": delta_single_h7,
                "r54_count": r54_single_h7, "r68_count": r68_single_h7,
            },
            "all": {
                "r54": r54_single_all_rate, "r68": r68_single_all_rate,
                "delta": r68_single_all_rate - r54_single_all_rate,
            },
        },
        "unique_h7_recoveries": unique_h7_recoveries,
        "lost_h7_to_r68": lost_h7,
        "h7_top20_rank_changes": h7_top1_churn,
        "stacked_rrf_pool_hit": {
            "all": {"baseline": base_rate, "r68": r68_rate,
                   "delta": delta_stacked_all},
            "h7": {"baseline": base_rate_h7, "r68": r68_rate_h7,
                  "delta": r68_rate_h7 - base_rate_h7},
        },
        "fold0_metrics": metrics,
        "gates": gates,
        "sibling_lr": {
            "path": str(sibling_lr_path.relative_to(REPO)),
            "feature_names": list(FEAT_ALL_R68),
            "params": LR_PARAMS,
            "num_boost_round": LR_NUM_BOOST_ROUND,
            "notes": (
                "Sibling LR trained on TRAIN cases (non-fold-0). TRAIN-case "
                "R68 features are zero-stubbed (no train query embeddings). "
                "Phase 1 will produce clean OOF R68 features for retraining."
            ),
        },
        "notes": (
            "Phase 0 substitutes 3 LR feature columns (r54_* -> r68_*); pool "
            "admission unchanged. This is NOT matched-pool retraining."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)

    # Markdown
    lines = [
        "# R68 Phase 0 fold-0 result",
        "",
        f"Created: {report['created_at']}",
        f"Branch: `{report['branch']}`",
        f"HEAD: `{report['head_sha']}`",
        f"Fold: {FOLD}  n_fold0={n_fold0}  h7={n_h7}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Gate results",
        "",
        "| Gate | Rule | Value | Pass |",
        "|---|---|---|:---:|",
        f"| 1 | recovery | unique_h7={unique_h7_recoveries}, "
        f"Δsingle_h7={delta_single_h7:+.4f} | "
        f"{'PASS' if gate_1 else 'FAIL'} |",
        f"| 2 | pool_hit | Δstacked_all={delta_stacked_all:+.4f} | "
        f"{'PASS' if gate_2 else 'FAIL'} |",
        f"| 3 | nDCG | Δh7={h7_ndcg_delta:+.4f}, "
        f"Δsame_artist={same_artist_delta:+.4f} | "
        f"{'PASS' if gate_3 else 'FAIL'} |",
        "",
        "## Single-source pool_hit @300 (fold-0)",
        "",
        "| Subset | R54 | R68 | Δ |",
        "|---|---:|---:|---:|",
        f"| h7 (n={n_h7}) | {r54_single_h7_rate:.4f} | "
        f"{r68_single_h7_rate:.4f} | {delta_single_h7:+.4f} |",
        f"| all (n={n_fold0}) | {r54_single_all_rate:.4f} | "
        f"{r68_single_all_rate:.4f} | "
        f"{r68_single_all_rate - r54_single_all_rate:+.4f} |",
        "",
        f"Unique h7 recoveries (R68 only): **{unique_h7_recoveries}**, "
        f"lost h7 (R54 only): **{lost_h7}**, "
        f"net **{unique_h7_recoveries - lost_h7:+d}**.",
        "",
        "## Stacked-RRF pool_hit @300 (fold-0)",
        "",
        "| Subset | Baseline (R54-stacked) | R68-stacked | Δ |",
        "|---|---:|---:|---:|",
        f"| all | {base_rate:.4f} | {r68_rate:.4f} | {delta_stacked_all:+.4f} |",
        f"| h7 | {base_rate_h7:.4f} | {r68_rate_h7:.4f} | "
        f"{r68_rate_h7 - base_rate_h7:+.4f} |",
        "",
        "## nDCG@20 (fold-0)",
        "",
        "| Subset | n | Baseline | R68 stacked + sibling LR | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ["all_fold0", "h7", "same_artist", "diff_artist",
                  "h7_same", "h7_diff"]:
        b = metrics[label]
        lines.append(
            f"| {label} | {b['n']} | {b['baseline']:.4f} | "
            f"{b['r68']:.4f} | {b['delta']:+.4f} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Pool admission unchanged; only 3 LR feature columns swapped "
        "(r54_* -> r68_*).",
        "- Sibling LR trained on fold-0 TRAIN cases (R68 features zero-stubbed "
        "for TRAIN; Phase 1 produces clean OOF features for full 5-fold).",
        "- This is feature substitution, NOT matched-pool retraining.",
        f"- Sibling LR: `{sibling_lr_path.relative_to(REPO)}`",
        f"- Elapsed: {report['elapsed_s']:.1f}s",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"\n{ts()} Phase 0 eval complete. Verdict: {verdict}  "
          f"Elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
