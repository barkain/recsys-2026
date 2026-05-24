"""R84 Phase 1 — 5-fold OOF compare. Proper apples-to-apples LR conversion test.

For each held-out fold k:
- Sibling-R54 LR: train on cases from other 4 folds with R54 features, eval on fold k.
- Sibling-R84 LR: train on cases from other 4 folds with R84 features (now
  available everywhere from Phase 0B fold-0 + Phase 1 folds 1-4), eval on fold k.
- Pool: R54-stacked RRF (unchanged), K=300.
- Compares apples-to-apples per feedback_lr_wall_was_artifact (sibling vs sibling,
  not vs frozen R54c in-sample).

Reports aggregate over all 5 folds (8000 cases, 1000 h7) + per-fold breakdown.
Closes the question that Phase 0B's partial-OOF probe left open:
"does the sibling LR re-calibrate to R84 features and close the same-artist gap?"
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
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR84_phase1.json"
OUT_MD = REPO / "docs" / "r84_phase1_result.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def load_r84_lists_for_fold(fold: int) -> dict[int, dict]:
    """Load R84 OOF lists for cases in `fold` (eval fold).

    Phase 0B used cache/r84/phase0b_fold0/ for fold 0.
    Phase 1 used cache/r84/phase1_fold{k}/ for folds 1-4.
    """
    path = (
        REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
        if fold == 0
        else REPO / "cache" / "r84" / f"phase1_fold{fold}" / "oof_r84_lists.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"R84 OOF lists missing for fold {fold}: {path}")
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for case_idx_str, pairs in raw.items():
        case_idx = int(case_idx_str)
        tids = [t for t, _ in pairs]
        out[case_idx] = {
            "ranks": {t: r + 1 for r, t in enumerate(tids)},
            "scores": {t: float(s) for t, s in pairs},
        }
    return out


def overwrite_r84_features(feats: np.ndarray, pool: list[str],
                            r84_ranks: dict[str, int],
                            r84_scores: dict[str, float]) -> np.ndarray:
    """Replace cols n_r39+{0,1,2} with R84-derived values."""
    for i, tid in enumerate(pool):
        feats[i, N_R39 + 0] = (1.0 / r84_ranks[tid]) if tid in r84_ranks else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in r84_ranks else 0.0
        feats[i, N_R39 + 2] = r84_scores.get(tid, 0.0)
    return feats


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R84 Phase 1 — 5-fold OOF compare (proper sibling-vs-sibling)")
    print(f"  features: {len(FEAT_ALL)} (R39={N_R39} + 3-of-{r'r54|r84'})")
    print(f"  folds: 0..4 each held out in turn")
    print(f"  pool: R54-stacked SW_BASELINE, K={POOL_K}")
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
    fold_to_idx: dict[int, list[int]] = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)
    for k in range(N_FOLDS):
        print(f"  fold {k}: {len(fold_to_idx[k])} cases", flush=True)

    print(f"\n{ts()} Loading R84 OOF lists for all 5 folds ...", flush=True)
    r84_per_fold = {k: load_r84_lists_for_fold(k) for k in range(N_FOLDS)}
    for k in range(N_FOLDS):
        # Sanity: every case in fold k must have R84 lists
        miss = [i for i in fold_to_idx[k] if i not in r84_per_fold[k]]
        if miss:
            raise RuntimeError(f"fold {k} missing R84 lists for {len(miss)} cases "
                               f"(first: {miss[:3]})")
        print(f"  fold {k}: {len(r84_per_fold[k])} R84 cases ✓")

    print(f"\n{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize37_r54(i: int, src_lists: dict[str, list[str]],
                         pool: list[str]) -> np.ndarray:
        case = cases[i]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album,
        )

    # --- Pre-build feature matrices ONCE per case (re-use across folds) ---
    print(f"\n{ts()} === Pre-building all 8000 case features ===", flush=True)
    case_features: dict[int, dict] = {}  # case_idx -> {pool, feats_r54, feats_r84, gt_pos}
    t_feat = time.time()
    gt_in_pool_total = 0
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gt_pos = pool.index(gt) if gt in pool else -1
        if gt_pos >= 0:
            gt_in_pool_total += 1
        feats_r54 = featurize37_r54(i, src_lists, pool)
        # R84 substitution — but which fold's R84 lists do we use?
        # For an HONEST OOF eval: the R84 lists for case i are the ones from the
        # fold where case i was held out (i.e., fold case_fold[i]). That fold's
        # R84 model never saw case i during training.
        owning_fold = case_fold[i]
        r84_maps_for_case = r84_per_fold[owning_fold][i]
        feats_r84 = feats_r54.copy()
        overwrite_r84_features(
            feats_r84, pool, r84_maps_for_case["ranks"], r84_maps_for_case["scores"]
        )
        case_features[i] = {
            "pool": pool,
            "feats_r54": feats_r54,
            "feats_r84": feats_r84,
            "gt_pos": gt_pos,
            "gt": gt,
        }
        if (i + 1) % 1000 == 0:
            print(f"    feats {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  pool_hit (all cases): {gt_in_pool_total}/{n} ({gt_in_pool_total/n:.4f})",
          flush=True)
    print(f"  total feat-build elapsed: {time.time() - t_feat:.0f}s", flush=True)

    # --- 5-fold OOF: train sibling LRs, score held-out folds ---
    r54_baseline_ranker = lgb.Booster(model_file=str(R54_LR))
    all_results: list[dict] = []
    for fold_k in range(N_FOLDS):
        print(f"\n{ts()} === FOLD {fold_k} held out ===", flush=True)
        eval_idx = fold_to_idx[fold_k]
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        print(f"  train cases: {len(train_idx)}, eval cases: {len(eval_idx)}", flush=True)

        # Stack training matrices
        Xt_r54, Xt_r84, yt, gt = [], [], [], []
        for i in train_idx:
            cf = case_features[i]
            pool_len = len(cf["pool"])
            for k_row in range(pool_len):
                Xt_r54.append(cf["feats_r54"][k_row])
                Xt_r84.append(cf["feats_r84"][k_row])
                yt.append(1.0 if k_row == cf["gt_pos"] else 0.0)
            gt.append(pool_len)
        Xt_r54 = np.array(Xt_r54, dtype=np.float64)
        Xt_r84 = np.array(Xt_r84, dtype=np.float64)
        yt = np.array(yt, dtype=np.float64)

        # Train sibling LRs
        print(f"  training sibling_r54_LR ({Xt_r54.shape[0]} rows)...", flush=True)
        t_lr = time.time()
        ds_r54 = lgb.Dataset(Xt_r54, label=yt, group=gt, feature_name=list(FEAT_ALL))
        lr_r54 = lgb.train(LR_PARAMS, ds_r54, num_boost_round=LR_NUM_BOOST_ROUND)
        print(f"    elapsed {time.time() - t_lr:.0f}s", flush=True)

        print(f"  training sibling_r84_LR ({Xt_r84.shape[0]} rows)...", flush=True)
        t_lr = time.time()
        ds_r84 = lgb.Dataset(Xt_r84, label=yt, group=gt, feature_name=FEAT_NAMES_R84)
        lr_r84 = lgb.train(LR_PARAMS, ds_r84, num_boost_round=LR_NUM_BOOST_ROUND)
        print(f"    elapsed {time.time() - t_lr:.0f}s", flush=True)
        del Xt_r54, Xt_r84, yt, ds_r54, ds_r84

        # Score eval fold
        print(f"  scoring fold {fold_k} ...", flush=True)
        for i in eval_idx:
            cf = case_features[i]
            pool, gt_pos, gt_id = cf["pool"], cf["gt_pos"], cf["gt"]
            b_scores = r54_baseline_ranker.predict(cf["feats_r54"])
            s_r54_scores = lr_r54.predict(cf["feats_r54"])
            s_r84_scores = lr_r84.predict(cf["feats_r84"])
            b_order = np.argsort(-b_scores, kind="mergesort")
            s_r54_order = np.argsort(-s_r54_scores, kind="mergesort")
            s_r84_order = np.argsort(-s_r84_scores, kind="mergesort")

            b_rank = s_r54_rank = s_r84_rank = -1
            if gt_pos >= 0:
                bp = np.where(b_order == gt_pos)[0]
                if len(bp): b_rank = int(bp[0]) + 1
                sp = np.where(s_r54_order == gt_pos)[0]
                if len(sp): s_r54_rank = int(sp[0]) + 1
                sp = np.where(s_r84_order == gt_pos)[0]
                if len(sp): s_r84_rank = int(sp[0]) + 1

            b_top20 = [pool[int(j)] for j in b_order[:TOP_K]]
            s_r54_top20 = [pool[int(j)] for j in s_r54_order[:TOP_K]]
            s_r84_top20 = [pool[int(j)] for j in s_r84_order[:TOP_K]]

            all_results.append({
                "case_idx": i,
                "fold": fold_k,
                "session_id": cases[i]["session_id"],
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "b_rank": b_rank, "s_r54_rank": s_r54_rank, "s_r84_rank": s_r84_rank,
                "b_ndcg20": ndcg_at_k(b_rank, TOP_K),
                "s_r54_ndcg20": ndcg_at_k(s_r54_rank, TOP_K),
                "s_r84_ndcg20": ndcg_at_k(s_r84_rank, TOP_K),
                "b_in_top20": b_rank > 0 and b_rank <= TOP_K,
                "s_r54_in_top20": s_r54_rank > 0 and s_r54_rank <= TOP_K,
                "s_r84_in_top20": s_r84_rank > 0 and s_r84_rank <= TOP_K,
                "overlap_r54_r84": len(set(s_r54_top20) & set(s_r84_top20)),
                "overlap_b_r84": len(set(b_top20) & set(s_r84_top20)),
                "top1_r54_eq_r84": int(s_r54_top20[0] == s_r84_top20[0]),
            })

    # --- Aggregate ---
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7_rows = [r for r in all_results if r["n_prior_music"] == 7]
    same_rows = [r for r in all_results if r["same_artist"]]
    diff_rows = [r for r in all_results if not r["same_artist"]]
    h7_same = [r for r in h7_rows if r["same_artist"]]
    h7_diff = [r for r in h7_rows if not r["same_artist"]]

    metrics = {}
    for name, rows in [("all", all_results), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows),
                       ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        b = avg(rows, "b_ndcg20")
        s_r54 = avg(rows, "s_r54_ndcg20")
        s_r84 = avg(rows, "s_r84_ndcg20")
        metrics[name] = {
            "n": len(rows),
            "frozen_r54c_baseline": b,
            "sibling_r54": s_r54,
            "sibling_r84": s_r84,
            "delta_r84_vs_sibling_r54": s_r84 - s_r54,
            "delta_r84_vs_frozen_r54c": s_r84 - b,
        }

    # h7 recovery analysis
    h7_recov = sum(1 for r in h7_rows if r["s_r84_in_top20"] and not r["s_r54_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["s_r54_in_top20"] and not r["s_r84_in_top20"])
    overlap_r54_r84_mean = avg(all_results, "overlap_r54_r84")
    overlap_b_r84_mean = avg(all_results, "overlap_b_r84")
    top1_match_rate = avg(all_results, "top1_r54_eq_r84")
    churn_per_80 = (1 - top1_match_rate) * 80

    # Per-fold breakdown
    by_fold_h7 = {}
    for k in range(N_FOLDS):
        rows = [r for r in h7_rows if r["fold"] == k]
        if not rows:
            continue
        by_fold_h7[k] = {
            "n": len(rows),
            "sibling_r54": avg(rows, "s_r54_ndcg20"),
            "sibling_r84": avg(rows, "s_r84_ndcg20"),
            "delta": avg(rows, "s_r84_ndcg20") - avg(rows, "s_r54_ndcg20"),
        }

    # Gates (same as Phase 0B but stricter aggregation)
    h7_d = metrics["h7"]["delta_r84_vs_sibling_r54"]
    sa_d = metrics["same_artist"]["delta_r84_vs_sibling_r54"]
    diff_d = metrics["diff_artist"]["delta_r84_vs_sibling_r54"]
    all_d = metrics["all"]["delta_r84_vs_sibling_r54"]

    A1 = h7_d >= 0.005
    A2 = h7_recov > h7_lost
    B1 = sa_d >= -0.005
    B2 = diff_d >= -0.005
    B3 = overlap_r54_r84_mean >= 8.0

    if (A1 or A2) and B1 and B2 and B3:
        verdict = "PROCEED_TO_BLIND"
    elif not B1 or not B2:
        verdict = "ARCHIVE_SPRINT"
    else:
        verdict = "INVESTIGATE"

    # --- Report ---
    print(f"\n{ts()} === AGGREGATE RESULTS (5-fold OOF) ===", flush=True)
    print(f"  {'subset':14}  {'n':>5}  {'frozen_r54c':>10}  {'sib_r54':>8}  "
          f"{'sib_r84':>8}  {'Δr84-r54':>9}")
    for name, m in metrics.items():
        print(f"  {name:14}  {m['n']:5d}  {m['frozen_r54c_baseline']:10.4f}  "
              f"{m['sibling_r54']:8.4f}  {m['sibling_r84']:8.4f}  "
              f"{m['delta_r84_vs_sibling_r54']:+9.4f}")

    print(f"\n  h7 recovered (r84 vs r54 sibling): {h7_recov}")
    print(f"  h7 lost      (r84 vs r54 sibling): {h7_lost}")
    print(f"  h7 net: {h7_recov - h7_lost:+d}")
    print(f"  top-20 overlap sib_r54 vs sib_r84: {overlap_r54_r84_mean:.2f}/20")
    print(f"  top-1 churn sib_r54 vs sib_r84:    {churn_per_80:.2f}/80")

    print(f"\n  Per-fold h7 deltas:")
    for k, m in by_fold_h7.items():
        print(f"    fold {k}: n={m['n']} sib_r54={m['sibling_r54']:.4f} "
              f"sib_r84={m['sibling_r84']:.4f} Δ={m['delta']:+.4f}")

    print(f"\n  Gates:")
    print(f"    A1: h7 Δ ≥ +0.005:           {A1}  ({h7_d:+.4f})")
    print(f"    A2: h7 recov > lost:         {A2}  ({h7_recov} > {h7_lost})")
    print(f"    B1: same-artist Δ ≥ -0.005:  {B1}  ({sa_d:+.4f})")
    print(f"    B2: diff-artist Δ ≥ -0.005:  {B2}  ({diff_d:+.4f})")
    print(f"    B3: overlap ≥ 8/20:          {B3}  ({overlap_r54_r84_mean:.2f})")
    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R84 Phase 1 — 5-fold OOF sibling LR (R84 vs R54 features)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "n_folds": N_FOLDS,
        "n_cases_total": len(all_results),
        "n_h7": len(h7_rows),
        "feature_stack_r84": {
            "names": FEAT_NAMES_R84, "n_features": len(FEAT_NAMES_R84),
            "structure": "FEAT_R39_ALL + [r84_rank_inv, r84_presence, r84_cosine]",
        },
        "pool": {"weights": SW_BASELINE, "K": POOL_K,
                 "note": "R54-stacked, identical to R54c production and Phase 0B"},
        "metrics_aggregate": metrics,
        "metrics_per_fold_h7": by_fold_h7,
        "h7_recovery": {
            "recovered": h7_recov, "lost": h7_lost, "net": h7_recov - h7_lost,
        },
        "churn": {
            "overlap_r54_r84_mean": overlap_r54_r84_mean,
            "overlap_b_r84_mean": overlap_b_r84_mean,
            "top1_match_rate_r54_r84": top1_match_rate,
            "churn_per_80": churn_per_80,
        },
        "gates": {
            "A1_h7_delta_ge_p005": {"value": h7_d, "pass": A1},
            "A2_h7_recov_gt_lost": {"value": [h7_recov, h7_lost], "pass": A2},
            "B1_same_artist_delta_ge_n005": {"value": sa_d, "pass": B1},
            "B2_diff_artist_delta_ge_n005": {"value": diff_d, "pass": B2},
            "B3_overlap_ge_8": {"value": overlap_r54_r84_mean, "pass": B3},
        },
        "phase0b_reference": {
            "h7_delta": -0.0059,
            "same_artist_delta": -0.0112,
            "note": ("Phase 0B was fold-0 only, partial-OOF probe (sibling LR "
                     "trained with R54 features, R84 swapped only at eval). "
                     "Phase 1 is proper 5-fold OOF: sibling LRs trained per fold "
                     "with R84 features available throughout."),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
