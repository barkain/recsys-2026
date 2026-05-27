"""R90 Phase 1 — LR conversion test (5-fold OOF sibling LRs + selective routing).

This is the missing test from expR90_phase1_compare.py: does R90's source-alone
gain survive the LR scoring + selective routing layer that R84c won Blind-A
with? Per feedback_lr_conversion_wall_confirmed, retrieval gains do not
automatically convert through the LR ranker; the LR-substitution test is the
real blind-readiness gate.

Apples-to-apples 5-fold OOF design:

For each held-out fold f in {0..4}:
  1. Train sibling LR on the OTHER 4 folds (6400 cases). Two versions:
     - LR_R84_f: features are R39 + R84 (last 3 cols = r84_rank_inv,
       r84_presence, r84_cosine for each pool track, derived from R84 fold-f
       OOF lists for the TRAINING cases — meaning the LR sees R84 features
       built from THE OTHER FOLDS' OOF, so it's a proper OOF sibling).
     - LR_R90_f: same but with R90 features.
  2. Score fold f's RRF pool with LR_R84_f (R84 feats) → R84-OOF sibling score
  3. Score fold f's RRF pool with LR_R90_f (R90 feats) → R90-OOF sibling score
  4. Apply selective routing using frozen R54c LR's top-1 margin on the same
     fold-f pool (R54c margin is INDEPENDENT of R84/R90 → fair routing signal)

Conditions evaluated on aggregated 8000 OOF cases:
  - R84-OOF-sibling alone (no routing)
  - R90-OOF-sibling alone (no routing)
  - R84-OOF-sibling routed via R54c margin (low=0.5, high=2.0)
  - R84-OOF-sibling routed via R54c margin (low=0.25, high=2.0)
  - R90-OOF-sibling routed via R54c margin (low=0.5, high=2.0)
  - R90-OOF-sibling routed via R54c margin (low=0.25, high=2.0)

Reports per condition: all-dev nDCG@20, h7 nDCG@20, same/diff artist,
n_prior buckets, top-30 rec/lost vs R84-OOF baseline, top-1 churn between
conditions, top-20 overlap.

Blind gate: R90-routed h7 nDCG@20 > R84-routed h7 nDCG@20 (the apples-to-
apples baseline) AND same-artist canary safe (Δ ≥ -0.005).

Outputs:
  exp/eval/expR90_phase1_lr_conversion.json
  docs/r90_phase1_lr_conversion.md
"""
from __future__ import annotations

import argparse
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

# Pool / pool-source configuration — identical to production R84c
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_LR_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"  # frozen R54c LR

R84_FOLD_DIRS = {
    0: REPO / "cache" / "r84" / "phase0b_fold0",
    1: REPO / "cache" / "r84" / "phase1_fold1",
    2: REPO / "cache" / "r84" / "phase1_fold2",
    3: REPO / "cache" / "r84" / "phase1_fold3",
    4: REPO / "cache" / "r84" / "phase1_fold4",
}
R90_FOLD_DIRS = {f: REPO / f"cache/r90/phase1_fold{f}_varA" for f in range(5)}

FEATURE_CACHE = REPO / "cache" / "r90" / "phase1_lr_conv_feature_cache.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR90_phase1_lr_conversion.json"
OUT_MD = REPO / "docs" / "r90_phase1_lr_conversion.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_RANKED = list(FEAT_R39_ALL) + ["ret_rank_inv", "ret_presence", "ret_cosine"]
assert len(FEAT_NAMES_RANKED) == len(FEAT_ALL), \
    f"feature count mismatch: {len(FEAT_NAMES_RANKED)} vs {len(FEAT_ALL)}"

# Routing thresholds to sweep
ROUTING_SWEEP = [(0.5, 2.0), (0.25, 2.0)]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ndcg_at_k(rank: int, k: int) -> float:
    if 1 <= rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def load_per_fold_maps(fold_dirs: dict[int, Path], label: str) -> dict[int, dict]:
    """Load all 5 folds' OOF lists into {case_idx: {ranks, scores, lists}}."""
    out: dict[int, dict] = {}
    for fold, fold_dir in fold_dirs.items():
        path = fold_dir / "oof_r84_lists.json"
        if not path.exists():
            raise FileNotFoundError(f"{label} fold-{fold} missing: {path}")
        with open(path) as f:
            raw = json.load(f)
        for case_idx_str, pairs in raw.items():
            ci = int(case_idx_str)
            tids = [t for t, _ in pairs]
            out[ci] = {
                "ranks": {t: r + 1 for r, t in enumerate(tids)},
                "scores": {t: float(s) for t, s in pairs},
                "lists": tids,
            }
    print(f"  {label}: {len(out)} cases loaded across 5 folds")
    return out


def overwrite_retriever_features(feats: np.ndarray, pool: list[str],
                                   ret_ranks: dict[str, int],
                                   ret_scores: dict[str, float]) -> np.ndarray:
    """Replace cols N_R39+{0,1,2} in `feats` with retriever-derived values."""
    for i, tid in enumerate(pool):
        feats[i, N_R39 + 0] = (1.0 / ret_ranks[tid]) if tid in ret_ranks else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in ret_ranks else 0.0
        feats[i, N_R39 + 2] = ret_scores.get(tid, 0.0)
    return feats


def build_feature_cache(cases, payload, r21_source, r54_source, r54_scores,
                         als_factors, als_to_idx, case_index, maps, track_pop,
                         max_pop, track_album, force: bool = False) -> dict:
    """Build (pool, feats_r54) per case once. Returns {case_idx: dict}."""
    if FEATURE_CACHE.exists() and not force:
        print(f"{ts()} Loading feature cache from {FEATURE_CACHE.name}...")
        with open(FEATURE_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"  loaded {len(cache)} cases")
        return cache

    print(f"{ts()} Building feature cache for {len(cases)} cases...")
    cache = {}
    t0 = time.time()
    for ci, case in enumerate(cases):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], ci)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[ci],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][ci],
            track_pop, max_pop, track_album,
        )
        gt = case["gt"]
        gt_pos = pool.index(gt) if gt in pool else -1
        cache[ci] = {
            "pool": pool,
            "feats_r54": feats_r54,  # 37-col with R54-derived last 3
            "gt_pos": gt_pos,
        }
        if (ci + 1) % 1000 == 0:
            print(f"  {ci + 1}/{len(cases)} ({time.time() - t0:.0f}s)")
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  cached to {FEATURE_CACHE} ({FEATURE_CACHE.stat().st_size/1e6:.1f} MB) "
          f"in {time.time() - t0:.0f}s")
    return cache


def assign_folds(payload, case_count: int) -> list[int]:
    """Use existing R90/R84 fold OOF lists to recover which case_idx is in which fold."""
    # Use R84 fold OOF lists as authoritative fold map (same as R90)
    case_fold = [-1] * case_count
    for fold, fold_dir in R84_FOLD_DIRS.items():
        with open(fold_dir / "oof_r84_lists.json") as f:
            raw = json.load(f)
        for case_idx_str in raw.keys():
            case_fold[int(case_idx_str)] = fold
    return case_fold


def train_sibling_lr(train_idx: list[int], feature_cache: dict,
                     ret_maps: dict, label: str) -> "lgb.Booster":
    """Train sibling LR on train_idx using retriever-substituted features."""
    print(f"  TRAIN sibling LR ({label}): {len(train_idx)} cases...")
    X_train, y_train, groups_train = [], [], []
    gt_in_pool = 0
    t0 = time.time()
    for ki, ci in enumerate(train_idx):
        fc = feature_cache[ci]
        pool = fc["pool"]
        gt_pos = fc["gt_pos"]
        if gt_pos >= 0:
            gt_in_pool += 1
        # Build feats with retriever features substituted
        if ci not in ret_maps:
            # Should not happen — every case has a fold-corresponding ret map
            continue
        feats = fc["feats_r54"].copy()
        overwrite_retriever_features(feats, pool, ret_maps[ci]["ranks"],
                                      ret_maps[ci]["scores"])
        for k_row in range(len(pool)):
            X_train.append(feats[k_row])
            y_train.append(1.0 if k_row == gt_pos else 0.0)
        groups_train.append(len(pool))
        if (ki + 1) % 1500 == 0:
            print(f"    {ki + 1}/{len(train_idx)} ({time.time() - t0:.0f}s)")
    ds = lgb.Dataset(np.array(X_train, dtype=np.float64),
                     label=np.array(y_train, dtype=np.float64),
                     group=groups_train, feature_name=FEAT_NAMES_RANKED)
    lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    print(f"    {label}: gt-in-pool {gt_in_pool}/{len(train_idx)} "
          f"({gt_in_pool/max(len(train_idx),1):.3f}) trained in "
          f"{time.time() - t0:.0f}s")
    return lr


def score_val_cases(val_idx: list[int], feature_cache: dict, ret_maps: dict,
                     lr: "lgb.Booster") -> dict[int, dict]:
    """Score each val case's pool with the retriever-substituted features."""
    out = {}
    for ci in val_idx:
        fc = feature_cache[ci]
        pool = fc["pool"]
        feats = fc["feats_r54"].copy()
        overwrite_retriever_features(feats, pool, ret_maps[ci]["ranks"],
                                      ret_maps[ci]["scores"])
        scores = lr.predict(feats)
        order = np.argsort(-scores, kind="mergesort")
        top20_pos = order[:TOP_K]
        top20_tids = [pool[int(p)] for p in top20_pos]
        # Top-1 margin from this LR (for routing? we use R54c LR for routing actually)
        scores_sorted = np.sort(scores)[::-1]
        top1_margin = float(scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0
        out[ci] = {
            "pool": pool,
            "top20": top20_tids,
            "top1_margin": top1_margin,
            "scores": scores,  # (POOL_K,) — keep for routing & per-case rank
            "order": order,
        }
    return out


def compute_r54c_margins(val_idx: list[int], feature_cache: dict,
                          r54c_lr: "lgb.Booster") -> dict[int, float]:
    """Compute R54c LR's top-1 margin per case using R54-features (frozen LR)."""
    out = {}
    for ci in val_idx:
        feats = feature_cache[ci]["feats_r54"]
        scores = r54c_lr.predict(feats)
        sorted_scores = np.sort(scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
        out[ci] = margin
    return out


def gt_rank_in_top20(top20: list[str], gt: str) -> int:
    """1-indexed rank of GT in top20, or 0 if not present."""
    try:
        return top20.index(gt) + 1
    except ValueError:
        return 0


def apply_routing(r84_scored: dict, r90_scored: dict, r54c_margins: dict,
                   low: float, high: float) -> dict[int, list[str]]:
    """Per case: top-20 = R90 if margin < low or >= high, else R84."""
    out = {}
    for ci in r84_scored:
        margin = r54c_margins[ci]
        use_r90 = (margin < low) or (margin >= high)
        out[ci] = r90_scored[ci]["top20"] if use_r90 else r84_scored[ci]["top20"]
    return out


def compute_condition_metrics(scored_top20: dict[int, list[str]],
                                cases: list[dict], track_artist: dict,
                                baseline_scored_top20: dict[int, list[str]] | None = None,
                                ) -> dict:
    """Per-condition aggregate metrics. If baseline_scored_top20 given, also
    computes rec/lost vs baseline."""
    rows = []
    for ci, top20 in scored_top20.items():
        case = cases[ci]
        gt = case["gt"]
        rank = gt_rank_in_top20(top20, gt)
        rows.append({
            "case_idx": ci, "gt": gt, "rank": rank,
            "ndcg20": ndcg_at_k(rank, 20),
            "is_h7": case.get("n_prior_music") == 7,
            "n_prior": int(case.get("n_prior_music", 0)),
            "same_art": bool(same_artist_case(case, track_artist)),
            "in_top20": 0 < rank <= 20,
        })

    h7 = [r for r in rows if r["is_h7"]]
    same_h7 = [r for r in h7 if r["same_art"]]
    diff_h7 = [r for r in h7 if not r["same_art"]]

    out = {
        "n_cases": len(rows),
        "n_h7": len(h7),
        "all_ndcg20": float(np.mean([r["ndcg20"] for r in rows])) if rows else 0.0,
        "h7_ndcg20": float(np.mean([r["ndcg20"] for r in h7])) if h7 else 0.0,
        "same_artist_h7_ndcg20": float(np.mean([r["ndcg20"] for r in same_h7])) if same_h7 else 0.0,
        "diff_artist_h7_ndcg20": float(np.mean([r["ndcg20"] for r in diff_h7])) if diff_h7 else 0.0,
        "h7_hit20": sum(1 for r in h7 if r["in_top20"]) / max(len(h7), 1),
        "all_hit20": sum(1 for r in rows if r["in_top20"]) / max(len(rows), 1),
    }

    # n_prior buckets
    out["n_prior_buckets"] = {}
    for n_prior in range(0, 8):
        sub = [r for r in rows if r["n_prior"] == n_prior]
        out["n_prior_buckets"][str(n_prior)] = {
            "n": len(sub),
            "ndcg20": float(np.mean([r["ndcg20"] for r in sub])) if sub else 0.0,
        }

    # vs baseline diffs (rec/lost top-30 would need top-30 here; we use top-20)
    if baseline_scored_top20 is not None:
        base_rank = {ci: gt_rank_in_top20(baseline_scored_top20[ci], cases[ci]["gt"])
                     for ci in scored_top20 if ci in baseline_scored_top20}
        n_rec_h7 = sum(1 for r in h7
                       if r["in_top20"] and not (0 < base_rank.get(r["case_idx"], 0) <= 20))
        n_lost_h7 = sum(1 for r in h7
                        if (0 < base_rank.get(r["case_idx"], 0) <= 20) and not r["in_top20"])
        # Top-1 churn between this condition and the baseline
        churn_top1 = sum(1 for ci, t20 in scored_top20.items()
                          if ci in baseline_scored_top20
                          and (not t20 or not baseline_scored_top20[ci]
                               or t20[0] != baseline_scored_top20[ci][0]))
        # Top-20 overlap mean
        overlaps = []
        for ci, t20 in scored_top20.items():
            if ci in baseline_scored_top20:
                overlaps.append(len(set(t20) & set(baseline_scored_top20[ci])))
        out["vs_baseline"] = {
            "h7_rec_top20": n_rec_h7,
            "h7_lost_top20": n_lost_h7,
            "h7_net_top20": n_rec_h7 - n_lost_h7,
            "top1_churn": churn_top1,
            "top1_churn_pct": churn_top1 / max(len(scored_top20), 1),
            "top20_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-feature-cache", action="store_true",
                    help="Force rebuild of feature cache.")
    args = ap.parse_args()

    t0 = time.time()
    print(f"{ts()} R90 Phase 1 LR conversion test — 5-fold OOF sibling LRs")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + R21/R54 OOF + supports...")
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    print(f"  n cases: {n}")

    print(f"{ts()} Building case index...")
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    print(f"{ts()} Loading 5-fold R84 + R90 retriever OOF maps...")
    r84_maps = load_per_fold_maps(R84_FOLD_DIRS, "R84")
    r90_maps = load_per_fold_maps(R90_FOLD_DIRS, "R90")
    assert len(r84_maps) == n and len(r90_maps) == n, (
        f"missing cases: r84={n-len(r84_maps)} r90={n-len(r90_maps)}")

    case_fold = assign_folds(payload, n)
    fold_idx = {f: [i for i in range(n) if case_fold[i] == f] for f in range(5)}
    for f in range(5):
        print(f"  fold {f}: {len(fold_idx[f])} cases")

    # ---- Feature cache (heavy, build once) ----
    feature_cache = build_feature_cache(
        cases, payload, r21_source, r54_source, r54_scores,
        als_factors, als_to_idx, case_index, maps, track_pop, max_pop,
        track_album, force=args.rebuild_feature_cache,
    )

    # ---- Frozen R54c LR (for routing margins) ----
    print(f"\n{ts()} Loading frozen R54c LR for routing margins...")
    r54c_lr = lgb.Booster(model_file=str(R54_LR_PATH))

    # ---- Per-fold OOF: train sibling LRs + score val fold ----
    print(f"\n{ts()} === Per-fold OOF: train sibling LRs + score val ===")
    r84_scored = {}  # case_idx -> {top20, top1_margin, ...}
    r90_scored = {}
    r54c_margins = {}

    for held_out in range(5):
        train_idx = [ci for f in range(5) if f != held_out for ci in fold_idx[f]]
        val_idx = fold_idx[held_out]
        print(f"\n  --- held-out fold {held_out}: train={len(train_idx)} val={len(val_idx)} ---")
        lr_r84 = train_sibling_lr(train_idx, feature_cache, r84_maps, f"R84-fold{held_out}")
        lr_r90 = train_sibling_lr(train_idx, feature_cache, r90_maps, f"R90-fold{held_out}")
        print(f"  scoring fold {held_out} with both sibling LRs...")
        r84_scored.update(score_val_cases(val_idx, feature_cache, r84_maps, lr_r84))
        r90_scored.update(score_val_cases(val_idx, feature_cache, r90_maps, lr_r90))
        r54c_margins.update(compute_r54c_margins(val_idx, feature_cache, r54c_lr))
        del lr_r84, lr_r90

    print(f"\n{ts()} === Computing metrics across conditions ===")
    track_artist = maps["track_artist"]

    # Extract top-20 lists from scored dicts
    r84_top20 = {ci: r84_scored[ci]["top20"] for ci in r84_scored}
    r90_top20 = {ci: r90_scored[ci]["top20"] for ci in r90_scored}

    # Conditions: R84-OOF alone, R90-OOF alone, routed variants
    conditions = {}
    print(f"  - R84 OOF sibling (no routing)")
    conditions["R84_oof_alone"] = compute_condition_metrics(
        r84_top20, cases, track_artist,
    )
    print(f"  - R90 OOF sibling (no routing)")
    conditions["R90_oof_alone"] = compute_condition_metrics(
        r90_top20, cases, track_artist, baseline_scored_top20=r84_top20,
    )
    for low, high in ROUTING_SWEEP:
        key84 = f"R84_routed_{low}_{high}"
        key90 = f"R90_routed_{low}_{high}"
        print(f"  - {key84}")
        # R84 routed = R84 in low/high bands, R54c-LR-top20 in mid... but we don't
        # have R54c-LR top20 here; the BASELINE in R84c blind routing was r84-LR
        # in low/high, R54c-LR in middle. For our OOF apples-to-apples, we use
        # r54c-LR top-20 as the fallback. Compute r54c top-20 per case.
        r54c_top20 = {}
        for ci in r84_scored:
            scores = r54c_lr.predict(feature_cache[ci]["feats_r54"])
            order = np.argsort(-scores, kind="mergesort")
            r54c_top20[ci] = [feature_cache[ci]["pool"][int(p)] for p in order[:TOP_K]]
        r84_routed = {}
        r90_routed = {}
        for ci in r84_scored:
            margin = r54c_margins[ci]
            use_retriever = (margin < low) or (margin >= high)
            r84_routed[ci] = r84_top20[ci] if use_retriever else r54c_top20[ci]
            r90_routed[ci] = r90_top20[ci] if use_retriever else r54c_top20[ci]
        conditions[key84] = compute_condition_metrics(
            r84_routed, cases, track_artist,
        )
        print(f"  - {key90}")
        conditions[key90] = compute_condition_metrics(
            r90_routed, cases, track_artist, baseline_scored_top20=r84_routed,
        )

    # ---- Verdict ----
    base_r84_routed_05 = conditions["R84_routed_0.5_2.0"]
    cand_r90_routed_05 = conditions["R90_routed_0.5_2.0"]
    base_r84_routed_025 = conditions["R84_routed_0.25_2.0"]
    cand_r90_routed_025 = conditions["R90_routed_0.25_2.0"]

    blind_gate = {}
    for label, base, cand in [
        ("0.5/2.0", base_r84_routed_05, cand_r90_routed_05),
        ("0.25/2.0", base_r84_routed_025, cand_r90_routed_025),
    ]:
        h7_delta = cand["h7_ndcg20"] - base["h7_ndcg20"]
        same_delta = cand["same_artist_h7_ndcg20"] - base["same_artist_h7_ndcg20"]
        h7_pos = h7_delta > 0
        same_safe = same_delta >= -0.005
        blind_gate[label] = {
            "h7_delta_vs_r84_routed": h7_delta,
            "same_artist_delta_vs_r84_routed": same_delta,
            "diff_artist_delta_vs_r84_routed": (
                cand["diff_artist_h7_ndcg20"] - base["diff_artist_h7_ndcg20"]),
            "h7_positive": h7_pos,
            "same_artist_safe": same_safe,
            "blind_gate_pass": h7_pos and same_safe,
        }

    overall_pass = any(g["blind_gate_pass"] for g in blind_gate.values())
    verdict = "PROCEED_TO_BLIND_B" if overall_pass else "INVESTIGATE_OR_ARCHIVE"

    report = {
        "experiment": "R90 Phase 1 LR conversion test (5-fold OOF sibling LRs)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "n_cases": n,
        "feature_cache": str(FEATURE_CACHE.relative_to(REPO)),
        "conditions": conditions,
        "blind_gate": blind_gate,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Wrote {OUT_JSON.relative_to(REPO)}")

    write_markdown(report)
    print(f"{ts()} Wrote {OUT_MD.relative_to(REPO)}")

    # Console summary
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"\nAggregate h7 nDCG@20 by condition:")
    for k, v in conditions.items():
        print(f"  {k:30s}  h7={v['h7_ndcg20']:.4f}  "
              f"same={v['same_artist_h7_ndcg20']:.4f}  diff={v['diff_artist_h7_ndcg20']:.4f}  "
              f"hit20_h7={v['h7_hit20']:.3f}")
    print(f"\nBlind gates:")
    for label, g in blind_gate.items():
        print(f"  R90 vs R84 (routed {label}): h7 Δ={g['h7_delta_vs_r84_routed']:+.4f}  "
              f"same Δ={g['same_artist_delta_vs_r84_routed']:+.4f}  "
              f"{'PASS' if g['blind_gate_pass'] else 'FAIL'}")
    print(f"\nDone in {time.time() - t0:.0f}s")


def write_markdown(report: dict) -> None:
    lines = []
    lines.append(f"# R90 Phase 1 LR Conversion Test")
    lines.append("")
    lines.append(f"Date: {report['created_at']}  ")
    lines.append(f"Head: {report['head_sha'][:10]}  ")
    lines.append(f"Elapsed: {report['elapsed_s']:.0f}s  ")
    lines.append("")
    lines.append(f"## Verdict: **{report['verdict']}**")
    lines.append("")
    lines.append("This is the LR scoring + selective routing test that R84c won")
    lines.append("Blind-A with. Source-alone retrieval gains (see")
    lines.append("`docs/r90_phase1_5fold_compare.md`) do NOT automatically convert")
    lines.append("through the LR layer per `feedback_lr_conversion_wall_confirmed`.")
    lines.append("")
    lines.append("### Per-condition aggregate (8000 OOF cases)")
    lines.append("")
    lines.append("| condition | h7 nDCG@20 | same-art h7 | diff-art h7 | h7 hit@20 | all nDCG@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in report["conditions"].items():
        lines.append(f"| {k} | {v['h7_ndcg20']:.4f} | {v['same_artist_h7_ndcg20']:.4f} | "
                     f"{v['diff_artist_h7_ndcg20']:.4f} | {v['h7_hit20']:.3f} | "
                     f"{v['all_ndcg20']:.4f} |")
    lines.append("")
    lines.append("### Blind gate: R90 routed vs R84 routed (apples-to-apples)")
    lines.append("")
    lines.append("Blind-readiness gate: R90-routed h7 nDCG@20 > R84-routed h7 nDCG@20")
    lines.append("AND same-artist Δ ≥ -0.005 (canary safe).")
    lines.append("")
    lines.append("| thresholds | h7 Δ | same-art Δ | diff-art Δ | gate |")
    lines.append("|---|---:|---:|---:|---|")
    for label, g in report["blind_gate"].items():
        lines.append(f"| {label} | {g['h7_delta_vs_r84_routed']:+.4f} | "
                     f"{g['same_artist_delta_vs_r84_routed']:+.4f} | "
                     f"{g['diff_artist_delta_vs_r84_routed']:+.4f} | "
                     f"{'PASS' if g['blind_gate_pass'] else 'FAIL'} |")
    lines.append("")
    lines.append("### Recovered / lost top-20 (R90 vs R84 paired comparisons)")
    lines.append("")
    lines.append("| comparison | h7 rec | h7 lost | net | top-1 churn | top-20 overlap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in report["conditions"].items():
        if "vs_baseline" not in v:
            continue
        vb = v["vs_baseline"]
        lines.append(f"| {k} | {vb['h7_rec_top20']} | {vb['h7_lost_top20']} | "
                     f"{vb['h7_net_top20']:+d} | {vb['top1_churn_pct']*100:.1f}% | "
                     f"{vb['top20_overlap_mean']:.2f} |")
    lines.append("")
    lines.append("### n_prior buckets (h7 nDCG@20 by condition)")
    lines.append("")
    cond_names = list(report["conditions"].keys())
    lines.append("| n_prior | " + " | ".join(cond_names) + " |")
    lines.append("|---:|" + "|".join(["---:"] * len(cond_names)) + "|")
    for n_prior in range(0, 8):
        row = [str(n_prior)]
        for cn in cond_names:
            b = report["conditions"][cn]["n_prior_buckets"].get(str(n_prior), {})
            row.append(f"{b.get('ndcg20', 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"Files: `{OUT_JSON.relative_to(REPO)}` (this JSON), "
                 f"`{OUT_MD.relative_to(REPO)}` (this report).")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
