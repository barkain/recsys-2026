"""R84 Phase 0B compare — R84 sibling LR (37 feats, r84 substituted for r54) vs
the R70b OOF sibling LR baseline (37 feats with r54).

Apples-to-apples test per feedback_lr_wall_was_artifact: both sides are OOF
sibling LRs trained on fold {1..4} and evaluated on fold 0, with identical
LightGBM hyperparams (LR_PARAMS) and identical R54-stacked RRF pool. Only the
last 3 feature columns differ.

Also reports:
- R84 source-alone vs R70b source-alone (pool_hit / nDCG)
- R84 RRF-add (9-source) and R84 RRF-replace (R84 swaps R54 as one of 8) pool diagnostics
- recovered/lost h7 vs R70b sibling LR
- Gate verdict: PROCEED_TO_PHASE_1 / ARCHIVE_SPRINT / INVESTIGATE
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
SW_R84_REPLACE = {**SW_BASELINE, "R84": 1.0}  # one used in RRF-replace (R54 removed)
RRF_K = 20
POOL_K = 300
TOP_K = 20
FOLD = 0

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
R84_LISTS = REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
R70B_RESULT = REPO / "exp" / "eval" / "expR70b_phase0_no_r68_control.json"

OUT_JSON = REPO / "exp" / "eval" / "expR84_phase0b.json"
OUT_MD = REPO / "docs" / "r84_phase0b_result.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

# R84 feature column positions in FEAT_ALL (replace r54 in cols n_r39+0..2)
N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = (
    list(FEAT_R39_ALL)
    + ["r84_rank_inv", "r84_presence", "r84_cosine"]
)
assert len(FEAT_NAMES_R84) == len(FEAT_ALL), (
    f"feature count mismatch: {len(FEAT_NAMES_R84)} vs {len(FEAT_ALL)}"
)


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


def load_r84_per_case_maps() -> dict[int, dict]:
    """Returns {case_idx: {"ranks": dict[tid -> 1..300], "scores": dict[tid -> cosine],
                            "lists": list[tid]}} from oof_r84_lists.json."""
    with open(R84_LISTS) as f:
        raw = json.load(f)
    out = {}
    for case_idx_str, pairs in raw.items():
        case_idx = int(case_idx_str)
        tids = [t for t, _ in pairs]
        out[case_idx] = {
            "ranks": {t: r + 1 for r, t in enumerate(tids)},
            "scores": {t: float(s) for t, s in pairs},
            "lists": tids,
        }
    return out


def overwrite_r84_features(feats: np.ndarray, pool: list[str],
                            r84_ranks: dict[str, int],
                            r84_scores: dict[str, float]) -> np.ndarray:
    """Replace cols n_r39+{0,1,2} in `feats` with R84-derived values."""
    for i, tid in enumerate(pool):
        feats[i, N_R39 + 0] = (1.0 / r84_ranks[tid]) if tid in r84_ranks else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in r84_ranks else 0.0
        feats[i, N_R39 + 2] = r84_scores.get(tid, 0.0)
    return feats


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R84 Phase 0B compare — sibling LR (37 feats, R84 swapped for R54)")
    print(f"  pool: R54-stacked SW_BASELINE, K={POOL_K} (unchanged from R70b)")
    print(f"  features: {len(FEAT_NAMES_R84)} = R39({N_R39}) + R84(3)")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + R21/R54 OOF ...", flush=True)
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

    print(f"{ts()} Loading R84 OOF lists from {R84_LISTS.name} ...", flush=True)
    r84_maps = load_r84_per_case_maps()
    print(f"  R84 cases with lists: {len(r84_maps)}")
    # Sanity: every fold-0 case must have R84 lists
    missing = [i for i in fold0_idx if i not in r84_maps]
    if missing:
        raise RuntimeError(f"R84 missing for {len(missing)} fold-0 cases (first: {missing[:5]})")
    # train_idx (folds 1-4) won't have R84 lists; we need to substitute differently
    # for training. R84 was trained per-fold (fold-0 OOF only). For sibling LR
    # training on fold-{1..4}, we DON'T have R84 features for those cases yet.
    # The clean OOF protocol would be: train R84 separately per fold (Phase 1).
    # For Phase 0B fold-0-only comparison, we can either:
    #   (a) Train sibling LR on fold-0 ALONE (no OOF) — biased
    #   (b) For training cases, use R54 features as a proxy (matches R70b
    #       baseline) — preserves the LR structure but compares only fold-0
    #       conversion of R84 features
    # Approach (b) is what R66/R70b implicitly did: train sibling on fold {1..4}
    # with R54 features (we don't have r84 for those folds), then SWAP to r84
    # features only on the fold-0 eval pool. This tests "does R84 supply a
    # better signal than R54 for the ranker, given the same trained weights?"
    # That's NOT a true OOF test of R84 — Phase 1 would need per-fold R84 training.
    # We label this as a "partial-OOF probe" and report it honestly.

    print(f"{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize37_r54(case_idx: int, src_lists: dict[str, list[str]],
                         pool: list[str]) -> np.ndarray:
        """R70b-equivalent feature matrix (37 feats with r54)."""
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

    # ---- Step 1: TRAIN sibling LR on fold-{1..4} with R54 features (R70b mirror) ----
    print(f"\n{ts()} === Step 1: TRAIN sibling LR on fold-{{1..4}} (R54 features) ===",
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
        feats = featurize37_r54(i, src_lists, pool)
        for k_row in range(len(pool)):
            X_train.append(feats[k_row])
            y_train.append(1.0 if k_row == gi else 0.0)
        groups_train.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} "
                  f"({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  TRAIN pool_hit: {gt_in_pool_train}/{len(train_idx)} "
          f"({gt_in_pool_train/max(len(train_idx),1):.4f})", flush=True)

    ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                     label=np.array(y_train, dtype=np.float64),
                     group=groups_train, feature_name=list(FEAT_ALL))
    sibling_lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    sibling_lr_path = REPO / "cache" / "r84_phase0b_sibling_lr.txt"
    sibling_lr.save_model(str(sibling_lr_path))
    print(f"  Saved sibling LR -> {sibling_lr_path.name}", flush=True)
    del X_train, y_train, groups_train, ds

    # ---- Step 2: Score fold-0 with: (a) frozen R54c (in-sample), (b) R70b sibling,
    #            (c) R84-substituted features through SAME sibling LR ----
    print(f"\n{ts()} === Step 2: Score fold-0 dev with 3 scorers ===", flush=True)
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))

    fold0_rows = []
    t_eval = time.time()
    n_r84_unique_recoveries_top30 = 0
    n_r84_lost_h7_top20 = 0
    n_r84_recovered_h7_top20 = 0
    pool_hit_r84_replace = 0  # # of cases where GT in R84-replace pool
    pool_hit_r84_add = 0
    pool_hit_baseline = 0
    r84_source_alone_hit20_h7 = 0
    r84_source_alone_hit30_h7 = 0
    r54_source_alone_hit20_h7 = 0
    r54_source_alone_hit30_h7 = 0

    for ki, i in enumerate(fold0_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gt_in_pool = gt in pool
        if gt_in_pool:
            pool_hit_baseline += 1

        # R84 single-source top-30 metrics
        r84_top30 = r84_maps[i]["lists"][:30]
        r84_top20 = r84_maps[i]["lists"][:20]
        is_h7 = cases[i].get("n_prior_music") == 7
        if is_h7:
            if gt in r84_top20:
                r84_source_alone_hit20_h7 += 1
            if gt in r84_top30:
                r84_source_alone_hit30_h7 += 1
            r54_top20 = r54_source[i][:20]
            r54_top30 = r54_source[i][:30]
            if gt in r54_top20:
                r54_source_alone_hit20_h7 += 1
            if gt in r54_top30:
                r54_source_alone_hit30_h7 += 1
            # Unique recovery: in R84 top-30 but NOT R54 top-30
            if gt in r84_top30 and gt not in r54_top30:
                n_r84_unique_recoveries_top30 += 1

        # RRF-add diagnostic: 9-source pool with R84 as additional source
        src_lists_with_r84 = {**src_lists, "R84": r84_maps[i]["lists"]}
        sw_add = {**SW_BASELINE, "R84": 1.0}
        pool_add = weighted_rrf(src_lists_with_r84, sw_add, topk=POOL_K, k=RRF_K)
        if gt in pool_add:
            pool_hit_r84_add += 1

        # RRF-replace diagnostic: replace R54 with R84
        src_lists_replace = {k: v for k, v in src_lists.items() if k != "R54"}
        src_lists_replace["R84"] = r84_maps[i]["lists"]
        sw_replace = {k: v for k, v in SW_BASELINE.items() if k != "R54"}
        sw_replace["R84"] = 1.0
        pool_replace = weighted_rrf(src_lists_replace, sw_replace, topk=POOL_K, k=RRF_K)
        if gt in pool_replace:
            pool_hit_r84_replace += 1

        # Featurize with R54 (the version used by trained scorers)
        feats_r54 = featurize37_r54(i, src_lists, pool)
        # Featurize with R84 (last-3-cols swap), using SAME pool
        feats_r84 = feats_r54.copy()
        overwrite_r84_features(
            feats_r84, pool, r84_maps[i]["ranks"], r84_maps[i]["scores"],
        )

        # 3 scoring runs
        b_scores = r54_baseline_ranker.predict(feats_r54)
        s_r54_scores = sibling_lr.predict(feats_r54)
        s_r84_scores = sibling_lr.predict(feats_r84)
        b_order = np.argsort(-b_scores, kind="mergesort")
        s_r54_order = np.argsort(-s_r54_scores, kind="mergesort")
        s_r84_order = np.argsort(-s_r84_scores, kind="mergesort")

        b_gt_rank = -1
        s_r54_gt_rank = -1
        s_r84_gt_rank = -1
        if gt_in_pool:
            gt_pos = pool.index(gt)
            bp = np.where(b_order == gt_pos)[0]
            sp_r54 = np.where(s_r54_order == gt_pos)[0]
            sp_r84 = np.where(s_r84_order == gt_pos)[0]
            if len(bp):
                b_gt_rank = int(bp[0]) + 1
            if len(sp_r54):
                s_r54_gt_rank = int(sp_r54[0]) + 1
            if len(sp_r84):
                s_r84_gt_rank = int(sp_r84[0]) + 1

        b_top20 = [pool[int(j)] for j in b_order[:TOP_K]]
        s_r84_top20 = [pool[int(j)] for j in s_r84_order[:TOP_K]]
        overlap_b_r84 = len(set(b_top20) & set(s_r84_top20))
        top1_changed = 1 if b_top20[0] != s_r84_top20[0] else 0
        s_r54_top20 = [pool[int(j)] for j in s_r54_order[:TOP_K]]

        # h7 recovery of R84-sibling LR vs R70b-sibling LR (apples-to-apples)
        if is_h7:
            r84_in_top20 = s_r84_gt_rank > 0 and s_r84_gt_rank <= TOP_K
            r54_in_top20 = s_r54_gt_rank > 0 and s_r54_gt_rank <= TOP_K
            if r84_in_top20 and not r54_in_top20:
                n_r84_recovered_h7_top20 += 1
            if r54_in_top20 and not r84_in_top20:
                n_r84_lost_h7_top20 += 1

        fold0_rows.append({
            "case_idx": i,
            "session_id": cases[i]["session_id"],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "b_gt_rank": b_gt_rank,
            "s_r54_gt_rank": s_r54_gt_rank,
            "s_r84_gt_rank": s_r84_gt_rank,
            "b_ndcg_at_20": ndcg_at_k(b_gt_rank, TOP_K),
            "s_r54_ndcg_at_20": ndcg_at_k(s_r54_gt_rank, TOP_K),
            "s_r84_ndcg_at_20": ndcg_at_k(s_r84_gt_rank, TOP_K),
            "overlap_b_r84": overlap_b_r84,
            "top1_changed_b_r84": top1_changed,
            "b_in_top20": b_gt_rank > 0 and b_gt_rank <= TOP_K,
            "s_r54_in_top20": s_r54_gt_rank > 0 and s_r54_gt_rank <= TOP_K,
            "s_r84_in_top20": s_r84_gt_rank > 0 and s_r84_gt_rank <= TOP_K,
        })
        if (ki + 1) % 200 == 0:
            print(f"    dev {ki + 1}/{len(fold0_idx)} ({time.time() - t_eval:.0f}s)",
                  flush=True)

    # ---- Step 3: Metrics ----
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
        s_r54 = avg(rows, "s_r54_ndcg_at_20")
        s_r84 = avg(rows, "s_r84_ndcg_at_20")
        metrics[name] = {
            "n": len(rows),
            "frozen_r54c_baseline": b,        # in-sample (upper bound)
            "sibling_r54_R70b": s_r54,         # apples-to-apples OOF baseline
            "sibling_r84": s_r84,
            "delta_r84_vs_sibling_r54": s_r84 - s_r54,
            "delta_r84_vs_frozen_r54c": s_r84 - b,
        }

    # Churn
    top1_churn = sum(r["top1_changed_b_r84"] for r in fold0_rows)
    churn_per_80 = top1_churn / len(fold0_rows) * 80
    overlap_mean = avg(fold0_rows, "overlap_b_r84")

    # Source-alone deltas (h7)
    n_h7 = len(h7_rows)
    r84_source_h7_20 = r84_source_alone_hit20_h7 / max(1, n_h7)
    r54_source_h7_20 = r54_source_alone_hit20_h7 / max(1, n_h7)
    r84_source_h7_30 = r84_source_alone_hit30_h7 / max(1, n_h7)
    r54_source_h7_30 = r54_source_alone_hit30_h7 / max(1, n_h7)

    n_fold0 = len(fold0_idx)
    pool_baseline_rate = pool_hit_baseline / n_fold0
    pool_add_rate = pool_hit_r84_add / n_fold0
    pool_replace_rate = pool_hit_r84_replace / n_fold0

    # Load R70b reference for cross-check
    r70b_ref = json.load(open(R70B_RESULT))
    r70b_h7_delta_vs_r54c = r70b_ref["metrics"]["h7"]["delta"]
    r70b_sa_delta_vs_r54c = r70b_ref["metrics"]["same_artist"]["delta"]

    # Gates (per plan)
    h7_d = metrics["h7"]["delta_r84_vs_sibling_r54"]
    sa_d = metrics["same_artist"]["delta_r84_vs_sibling_r54"]
    diff_d = metrics["diff_artist"]["delta_r84_vs_sibling_r54"]
    all_d = metrics["all_fold0"]["delta_r84_vs_sibling_r54"]

    A1 = h7_d >= 0.005
    A2 = n_r84_recovered_h7_top20 > n_r84_lost_h7_top20
    A3 = n_r84_unique_recoveries_top30 >= 10
    A4 = (h7_d >= -0.003) and (n_r84_unique_recoveries_top30 >= 5)
    B1 = sa_d >= -0.005
    B2 = diff_d >= -0.005
    B3 = overlap_mean >= 8.0  # top-20 overlap b_top20 vs r84_top20

    catastrophic_pool = pool_baseline_rate * pool_hit_baseline / max(1, n_fold0) < 0.55
    a_passes = A1 or A2 or A3 or A4
    b_passes = B1 and B2 and B3

    if a_passes and b_passes:
        verdict = "PROCEED_TO_PHASE_1"
    elif catastrophic_pool or not b_passes:
        verdict = "ARCHIVE_SPRINT"
    else:
        verdict = "INVESTIGATE"

    print(f"\n{ts()} === RESULTS ===", flush=True)
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  "
              f"frozen_r54c={m['frozen_r54c_baseline']:.4f}  "
              f"sibling_r54_R70b={m['sibling_r54_R70b']:.4f}  "
              f"sibling_r84={m['sibling_r84']:.4f}  "
              f"Δ_r84_vs_r54={m['delta_r84_vs_sibling_r54']:+.4f}", flush=True)
    print(f"\n  h7_recovered (r84 vs r54 sibling) = {n_r84_recovered_h7_top20}")
    print(f"  h7_lost      (r84 vs r54 sibling) = {n_r84_lost_h7_top20}")
    print(f"  h7 top-30 unique recoveries (R84 source vs R54 source) = "
          f"{n_r84_unique_recoveries_top30}")
    print(f"  source-alone h7 hit@20: R54={r54_source_h7_20:.4f}  "
          f"R84={r84_source_h7_20:.4f}  Δ={r84_source_h7_20 - r54_source_h7_20:+.4f}")
    print(f"  source-alone h7 hit@30: R54={r54_source_h7_30:.4f}  "
          f"R84={r84_source_h7_30:.4f}  Δ={r84_source_h7_30 - r54_source_h7_30:+.4f}")
    print(f"  RRF pool_hit (8-source baseline): {pool_baseline_rate:.4f}")
    print(f"  RRF pool_hit (R84-add 9-source):  {pool_add_rate:.4f}  "
          f"(Δ={pool_add_rate - pool_baseline_rate:+.4f})")
    print(f"  RRF pool_hit (R84-replace 8-src): {pool_replace_rate:.4f}  "
          f"(Δ={pool_replace_rate - pool_baseline_rate:+.4f})")
    print(f"  top-1 churn (sibling_r84 vs frozen_r54c): {top1_churn}/{n_fold0} "
          f"= {churn_per_80:.2f}/80")
    print(f"  top-20 overlap (sibling_r84 vs frozen_r54c): {overlap_mean:.2f}/20")
    print(f"\n  Gates:")
    print(f"    A1: h7 Δ ≥ +0.005:        {A1}  ({h7_d:+.4f})")
    print(f"    A2: h7 rec > lost:        {A2}  ({n_r84_recovered_h7_top20} > "
          f"{n_r84_lost_h7_top20})")
    print(f"    A3: ≥10 h7 top-30 unique: {A3}  ({n_r84_unique_recoveries_top30})")
    print(f"    A4: ambig-positive:       {A4}")
    print(f"    B1: same-artist Δ ≥ -0.005: {B1}  ({sa_d:+.4f})")
    print(f"    B2: diff-artist Δ ≥ -0.005: {B2}  ({diff_d:+.4f})")
    print(f"    B3: overlap ≥ 8/20:         {B3}  ({overlap_mean:.2f})")
    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R84 Phase 0B compare — r84 sibling LR vs r70b r54 sibling LR",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "fold": FOLD,
        "n_fold0": n_fold0,
        "n_train": len(train_idx),
        "n_h7": n_h7,
        "feature_stack": {
            "names": FEAT_NAMES_R84, "n_features": len(FEAT_NAMES_R84),
            "structure": "FEAT_R39_ALL + [r84_rank_inv, r84_presence, r84_cosine]",
            "note": ("sibling LR trained on fold-{1..4} with R54 features "
                     "(no R84 lists for those folds); r84 features swapped in at "
                     "fold-0 eval only. Partial-OOF probe of R84 conversion."),
        },
        "pool": {"weights": SW_BASELINE, "K": POOL_K,
                 "note": "R54-stacked, identical to R54c production"},
        "metrics": metrics,
        "source_alone_h7": {
            "r54_hit20": r54_source_h7_20, "r84_hit20": r84_source_h7_20,
            "r54_hit30": r54_source_h7_30, "r84_hit30": r84_source_h7_30,
            "delta_hit20": r84_source_h7_20 - r54_source_h7_20,
            "delta_hit30": r84_source_h7_30 - r54_source_h7_30,
        },
        "pool_diagnostics": {
            "baseline_8src": pool_baseline_rate,
            "add_9src": pool_add_rate,
            "replace_8src": pool_replace_rate,
            "add_delta": pool_add_rate - pool_baseline_rate,
            "replace_delta": pool_replace_rate - pool_baseline_rate,
        },
        "h7_recovery_r84_vs_r54_sibling": {
            "recovered_top20": n_r84_recovered_h7_top20,
            "lost_top20": n_r84_lost_h7_top20,
            "net_top20": n_r84_recovered_h7_top20 - n_r84_lost_h7_top20,
            "top30_unique_source_alone": n_r84_unique_recoveries_top30,
        },
        "churn_b_vs_r84sibling": {
            "top1_changed": top1_churn,
            "churn_per_80": churn_per_80,
            "top20_overlap_mean": overlap_mean,
        },
        "gates": {
            "A1_h7_delta_ge_p005": {"value": h7_d, "pass": A1},
            "A2_h7_recov_gt_lost": {"value": [n_r84_recovered_h7_top20,
                                              n_r84_lost_h7_top20], "pass": A2},
            "A3_h7_top30_unique_ge_10": {"value": n_r84_unique_recoveries_top30,
                                          "pass": A3},
            "A4_ambig_positive": {"pass": A4},
            "B1_same_artist_delta_ge_n005": {"value": sa_d, "pass": B1},
            "B2_diff_artist_delta_ge_n005": {"value": diff_d, "pass": B2},
            "B3_overlap_ge_8": {"value": overlap_mean, "pass": B3},
        },
        "r70b_reference": {
            "h7_delta_vs_frozen_r54c": r70b_h7_delta_vs_r54c,
            "same_artist_delta_vs_frozen_r54c": r70b_sa_delta_vs_r54c,
            "note": ("R70b is the OOF R54 sibling baseline (same features, "
                     "same pool, retrained on fold-{1..4}). R84 sibling here "
                     "should be compared against R70b sibling for "
                     "apples-to-apples per feedback_lr_wall_was_artifact."),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
