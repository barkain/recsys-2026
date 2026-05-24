"""R84b — CPU-only calibration/tuning sweep using existing 5-fold R84 artifacts.

Four streams (all on the same R54-stacked POOL_K=300 pool, all proper 5-fold OOF):

A. Feature interface
   - replace: 37 feats with r84 substituted for r54 (Phase 1 baseline = sibling_r84)
   - r54_only: 37 feats with r54 (Phase 1 baseline = sibling_r54)
   - combined: 40 feats with r39 + r54 + r84
B. Score blend
   - score = zscore(sibling_r54_LR) + β * zscore(sibling_r84_LR)  for β in 0.05..0.50
C. LR hyperparam sweep on the best variant from (A)
   - num_leaves {15,31,63} × min_data_in_leaf {10,50} × lambda_l2 {0,1}
D. Segment diagnostics
   - per-case h7 lift by same_artist, n_prior_music, R54 top-1 margin,
     R84 unique top-30 presence, fold

Reports best variant in each, plus single combined verdict against R84b PROCEED gate.

Outputs:
- exp/eval/expR84b_sweep.json
- docs/r84b_sweep_result.md
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
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_JSON = REPO / "exp" / "eval" / "expR84b_sweep.json"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"

LR_PARAMS_DEFAULT = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
# 40-feature combined stack: R39 + R54 + R84
FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
FEAT_NAMES_COMBINED = list(FEAT_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
assert len(FEAT_NAMES_R84_ONLY) == 37
assert len(FEAT_NAMES_COMBINED) == 40

# R84b proceed gate
GATE = {
    "h7_delta_ge": 0.005,
    "all_delta_ge": 0.0,
    "same_artist_delta_ge": -0.005,
    "diff_artist_delta_ge": 0.0,
    "recov_ge_lost": True,
    "overlap_ge": 14.0,
}


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def load_r84_for_fold(fold: int) -> dict[int, dict]:
    path = (
        REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json" if fold == 0
        else REPO / "cache" / "r84" / f"phase1_fold{fold}" / "oof_r84_lists.json"
    )
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for case_idx_str, pairs in raw.items():
        case_idx = int(case_idx_str)
        tids = [t for t, _ in pairs]
        out[case_idx] = {
            "ranks": {t: r + 1 for r, t in enumerate(tids)},
            "scores": {t: float(s) for t, s in pairs},
            "tids": tids,
        }
    return out


def build_or_load_features(cases, payload, r21_source, r54_source, r54_scores,
                            case_index, maps, track_pop, max_pop, track_album,
                            r84_per_fold, case_fold) -> dict:
    """Pre-build per-case feature matrices: feats_r54 (37), feats_r84_only (37),
    feats_combined (40 = R39+R54+R84), plus pool + gt_pos."""
    if FEAT_CACHE.exists():
        print(f"  loading feature cache {FEAT_CACHE.name}", flush=True)
        with open(FEAT_CACHE, "rb") as f:
            return pickle.load(f)

    print(f"  building feature cache (one-time ~2 min)", flush=True)
    case_features = {}
    t_feat = time.time()
    for i in range(len(cases)):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gt_pos = pool.index(gt) if gt in pool else -1
        case = cases[i]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            case_index.get("als_factors", None) or c3.load_als_cache()[0],  # cheap
            case_index["als_to_idx"] if "als_to_idx" in case_index else None,
            case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album,
        )
        # Build r84 features columns
        owning_fold = case_fold[i]
        r84 = r84_per_fold[owning_fold][i]
        n_pool = len(pool)
        r84_cols = np.zeros((n_pool, 3), dtype=np.float64)
        for k_row, tid in enumerate(pool):
            r84_cols[k_row, 0] = (1.0 / r84["ranks"][tid]) if tid in r84["ranks"] else 0.0
            r84_cols[k_row, 1] = 1.0 if tid in r84["ranks"] else 0.0
            r84_cols[k_row, 2] = r84["scores"].get(tid, 0.0)
        # r84-only: replace last 3 cols of feats_r54 (r54_*) with r84_*
        feats_r84_only = feats_r54.copy()
        feats_r84_only[:, N_R39:N_R39 + 3] = r84_cols
        # combined: 40 cols = r39 + r54 + r84
        feats_combined = np.concatenate([feats_r54, r84_cols], axis=1)
        case_features[i] = {
            "pool": pool, "gt_pos": gt_pos, "gt": gt,
            "feats_r54": feats_r54,
            "feats_r84_only": feats_r84_only,
            "feats_combined": feats_combined,
        }
        if (i + 1) % 1000 == 0:
            print(f"    feats {i + 1}/{len(cases)} ({time.time() - t_feat:.0f}s)",
                  flush=True)
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEAT_CACHE, "wb") as f:
        pickle.dump(case_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved cache ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)", flush=True)
    return case_features


def train_sibling_lr(case_features, train_idx, feat_key, feat_names, params):
    X, y, gt = [], [], []
    for i in train_idx:
        cf = case_features[i]
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf[feat_key][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=feat_names)
    return lgb.train(params, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def score_fold(case_features, eval_idx, lr_models, feat_keys):
    """Returns dict: case_idx -> {model_name: np.array(per-pool-track score)}."""
    out = {}
    for i in eval_idx:
        cf = case_features[i]
        scores_per_model = {}
        for name, (lr, fk) in zip(lr_models.keys(),
                                   zip(lr_models.values(), feat_keys)):
            scores_per_model[name] = lr.predict(cf[fk])
        out[i] = scores_per_model
    return out


def metrics_from_scores(cases, case_features, scores_per_case, maps,
                         score_key, fold_for_case):
    """Compute metric rows from precomputed scores for `score_key` ('r54', 'r84_only',
    'combined', or 'blend_X')."""
    rows = []
    for i, scores_dict in scores_per_case.items():
        cf = case_features[i]
        s = scores_dict[score_key]
        order = np.argsort(-s, kind="mergesort")
        rank = -1
        if cf["gt_pos"] >= 0:
            p = np.where(order == cf["gt_pos"])[0]
            if len(p):
                rank = int(p[0]) + 1
        rows.append({
            "case_idx": i,
            "fold": fold_for_case[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank,
            "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
        })
    return rows


def zscore_per_case(arr):
    m = arr.mean()
    s = arr.std()
    if s < 1e-9:
        return np.zeros_like(arr)
    return (arr - m) / s


def blend_scores(scores_r54, scores_r84, beta):
    return zscore_per_case(scores_r54) + beta * zscore_per_case(scores_r84)


def metrics_summary(rows_test, rows_baseline, rows_for_overlap=None):
    """Compute Δ metrics: test vs baseline. rows_for_overlap optional (default=baseline)."""
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    sub_h7 = [r for r in rows_test if r["n_prior_music"] == 7]
    sub_h7_b = [r for r in rows_baseline if r["n_prior_music"] == 7]
    sub_same = [r for r in rows_test if r["same_artist"]]
    sub_same_b = [r for r in rows_baseline if r["same_artist"]]
    sub_diff = [r for r in rows_test if not r["same_artist"]]
    sub_diff_b = [r for r in rows_baseline if not r["same_artist"]]

    out = {
        "h7":          {"n": len(sub_h7), "test": avg(sub_h7, "ndcg20"),
                         "baseline": avg(sub_h7_b, "ndcg20")},
        "all":         {"n": len(rows_test), "test": avg(rows_test, "ndcg20"),
                         "baseline": avg(rows_baseline, "ndcg20")},
        "same_artist": {"n": len(sub_same), "test": avg(sub_same, "ndcg20"),
                         "baseline": avg(sub_same_b, "ndcg20")},
        "diff_artist": {"n": len(sub_diff), "test": avg(sub_diff, "ndcg20"),
                         "baseline": avg(sub_diff_b, "ndcg20")},
    }
    for k in out:
        out[k]["delta"] = out[k]["test"] - out[k]["baseline"]

    # Recovery/lost on h7 top-20
    h7_b_in = {r["case_idx"]: r["in_top20"] for r in sub_h7_b}
    h7_t_in = {r["case_idx"]: r["in_top20"] for r in sub_h7}
    recov = sum(1 for cid, t in h7_t_in.items() if t and not h7_b_in.get(cid, False))
    lost = sum(1 for cid, b in h7_b_in.items() if b and not h7_t_in.get(cid, False))
    out["h7_recovery"] = {"recovered": recov, "lost": lost, "net": recov - lost}

    # top-20 overlap
    cross = rows_for_overlap if rows_for_overlap else rows_baseline
    cross_top20 = {r["case_idx"]: set(r["top20"]) for r in cross}
    overlaps = []
    for r in rows_test:
        ov = len(set(r["top20"]) & cross_top20.get(r["case_idx"], set()))
        overlaps.append(ov)
    out["overlap_mean"] = float(np.mean(overlaps)) if overlaps else 0.0

    return out


def gate_eval(summary):
    """Apply R84b PROCEED gate to a metrics_summary dict. Returns (passes, details)."""
    h7_d = summary["h7"]["delta"]
    all_d = summary["all"]["delta"]
    sa_d = summary["same_artist"]["delta"]
    diff_d = summary["diff_artist"]["delta"]
    recov, lost = summary["h7_recovery"]["recovered"], summary["h7_recovery"]["lost"]
    overlap = summary["overlap_mean"]
    gates = {
        "h7_delta_ge_p005": (h7_d >= GATE["h7_delta_ge"], h7_d),
        "all_delta_ge_0": (all_d >= GATE["all_delta_ge"], all_d),
        "same_artist_delta_ge_n005": (sa_d >= GATE["same_artist_delta_ge"], sa_d),
        "diff_artist_delta_ge_0": (diff_d >= GATE["diff_artist_delta_ge"], diff_d),
        "recov_ge_lost": (recov >= lost, [recov, lost]),
        "overlap_ge_14": (overlap >= GATE["overlap_ge"], overlap),
    }
    passes = all(v[0] for v in gates.values())
    return passes, gates


def main():
    t0 = time.time()
    print(f"{ts()} R84b — calibration/tuning sweep")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + R21/R54 OOF ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_to_idx = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)

    print(f"{ts()} Loading R84 lists for all 5 folds ...", flush=True)
    r84_per_fold = {k: load_r84_for_fold(k) for k in range(N_FOLDS)}

    print(f"{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )
    case_index["als_to_idx"] = als_to_idx

    print(f"\n{ts()} === Build / load feature cache ===", flush=True)
    case_features = build_or_load_features(
        cases, payload, r21_source, r54_source, r54_scores,
        case_index, maps, track_pop, max_pop, track_album,
        r84_per_fold, case_fold,
    )

    # --- 5-fold OOF training and scoring for all 3 feature variants ---
    print(f"\n{ts()} === Train + score per fold: 3 feature variants ===", flush=True)
    # For each fold k:
    #   train sibling_r54_lr (37 r54 feats)
    #   train sibling_r84_lr (37 r84 feats)
    #   train sibling_combined_lr (40 feats)
    # Score held-out fold k with all 3 LRs and store score arrays.
    all_scores: dict[int, dict[str, np.ndarray]] = {}
    for fold_k in range(N_FOLDS):
        print(f"  fold {fold_k} train...", flush=True)
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t_lr = time.time()
        lr_r54 = train_sibling_lr(case_features, train_idx, "feats_r54",
                                    list(FEAT_ALL), LR_PARAMS_DEFAULT)
        lr_r84 = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                    FEAT_NAMES_R84_ONLY, LR_PARAMS_DEFAULT)
        lr_comb = train_sibling_lr(case_features, train_idx, "feats_combined",
                                    FEAT_NAMES_COMBINED, LR_PARAMS_DEFAULT)
        print(f"    LRs trained in {time.time() - t_lr:.0f}s", flush=True)
        for i in eval_idx:
            cf = case_features[i]
            all_scores[i] = {
                "r54": lr_r54.predict(cf["feats_r54"]),
                "r84_only": lr_r84.predict(cf["feats_r84_only"]),
                "combined": lr_comb.predict(cf["feats_combined"]),
            }

    fold_for_case = {i: case_fold[i] for i in range(n)}

    # --- A. Feature interface: 3 variants vs sibling_r54 baseline ---
    print(f"\n{ts()} === A. Feature interface compare ===", flush=True)
    rows_r54      = metrics_from_scores(cases, case_features, all_scores, maps,
                                          "r54", fold_for_case)
    rows_r84_only = metrics_from_scores(cases, case_features, all_scores, maps,
                                          "r84_only", fold_for_case)
    rows_comb     = metrics_from_scores(cases, case_features, all_scores, maps,
                                          "combined", fold_for_case)

    A_results = {
        "r84_only":  metrics_summary(rows_r84_only, rows_r54),
        "combined":  metrics_summary(rows_comb, rows_r54),
    }
    print(f"\n  variant      h7_Δ      all_Δ     same_Δ    diff_Δ    rec/lost  ovl    GATE")
    for name, s in A_results.items():
        p, _ = gate_eval(s)
        rec = s["h7_recovery"]
        print(f"  {name:11}  {s['h7']['delta']:+.4f}  {s['all']['delta']:+.4f}  "
              f"{s['same_artist']['delta']:+.4f}  {s['diff_artist']['delta']:+.4f}  "
              f"{rec['recovered']}/{rec['lost']}     "
              f"{s['overlap_mean']:.2f}  {'PASS' if p else 'fail'}")

    # --- B. Score blend sweep ---
    print(f"\n{ts()} === B. Score blend zscore(r54) + β·zscore(r84) ===", flush=True)
    B_results = {}
    best_blend = None
    best_blend_h7 = -1e9
    for beta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        # Build blended score per case
        blend_scores_per_case = {}
        for i, sd in all_scores.items():
            blend_scores_per_case[i] = {"blend": blend_scores(sd["r54"], sd["r84_only"], beta)}
        rows_b = metrics_from_scores(cases, case_features, blend_scores_per_case, maps,
                                       "blend", fold_for_case)
        summary = metrics_summary(rows_b, rows_r54)
        B_results[f"beta_{beta:.2f}"] = summary
        p, _ = gate_eval(summary)
        rec = summary["h7_recovery"]
        print(f"  β={beta:.2f}  h7_Δ={summary['h7']['delta']:+.4f}  "
              f"all_Δ={summary['all']['delta']:+.4f}  "
              f"same_Δ={summary['same_artist']['delta']:+.4f}  "
              f"diff_Δ={summary['diff_artist']['delta']:+.4f}  "
              f"rec/lost={rec['recovered']}/{rec['lost']}  "
              f"ovl={summary['overlap_mean']:.2f}  {'PASS' if p else 'fail'}")
        if summary["h7"]["delta"] > best_blend_h7:
            best_blend_h7 = summary["h7"]["delta"]
            best_blend = (beta, summary)

    print(f"\n  Best blend by h7_Δ: β={best_blend[0]:.2f} → h7_Δ={best_blend_h7:+.4f}")

    # --- C. LR hyperparam sweep on the best-A variant ---
    # Pick best A by h7 Δ
    best_A_name = max(A_results.keys(), key=lambda k: A_results[k]["h7"]["delta"])
    print(f"\n{ts()} === C. LR hyperparam sweep on '{best_A_name}' ===", flush=True)
    if best_A_name == "r84_only":
        feat_key, feat_names = "feats_r84_only", FEAT_NAMES_R84_ONLY
    else:
        feat_key, feat_names = "feats_combined", FEAT_NAMES_COMBINED

    C_results = {}
    best_C = (None, -1e9)
    for num_leaves in [15, 31, 63]:
        for mdif in [10, 50]:
            for l2 in [0, 1]:
                params = dict(LR_PARAMS_DEFAULT)
                params["num_leaves"] = num_leaves
                params["min_data_in_leaf"] = mdif
                params["lambda_l2"] = l2
                # Score this hyperparam set across 5 folds
                rows_test = []
                for fold_k in range(N_FOLDS):
                    train_idx = [i for i in range(n) if case_fold[i] != fold_k]
                    eval_idx = fold_to_idx[fold_k]
                    lr_t = train_sibling_lr(case_features, train_idx, feat_key,
                                              feat_names, params)
                    for i in eval_idx:
                        cf = case_features[i]
                        s = lr_t.predict(cf[feat_key])
                        order = np.argsort(-s, kind="mergesort")
                        rank = -1
                        if cf["gt_pos"] >= 0:
                            p = np.where(order == cf["gt_pos"])[0]
                            if len(p):
                                rank = int(p[0]) + 1
                        rows_test.append({
                            "case_idx": i, "fold": fold_k,
                            "n_prior_music": int(cases[i]["n_prior_music"]),
                            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
                            "in_top20": rank > 0 and rank <= TOP_K,
                            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
                        })
                summary = metrics_summary(rows_test, rows_r54)
                key = f"nl{num_leaves}_mdif{mdif}_l2{l2}"
                C_results[key] = summary
                p, _ = gate_eval(summary)
                rec = summary["h7_recovery"]
                print(f"  {key}: h7_Δ={summary['h7']['delta']:+.4f} "
                      f"same_Δ={summary['same_artist']['delta']:+.4f} "
                      f"rec/lost={rec['recovered']}/{rec['lost']} "
                      f"ovl={summary['overlap_mean']:.2f} {'PASS' if p else 'fail'}")
                if summary["h7"]["delta"] > best_C[1]:
                    best_C = (key, summary["h7"]["delta"], summary)
    print(f"\n  Best C: {best_C[0]} → h7_Δ={best_C[1]:+.4f}")

    # --- D. Segment diagnostics: per-case h7 lift from r84_only vs r54 ---
    print(f"\n{ts()} === D. Segment diagnostics (h7 cases) ===", flush=True)
    h7_idx = [i for i in range(n) if cases[i].get("n_prior_music") == 7]
    seg = {"same_artist": [], "diff_artist": [], "by_fold": defaultdict(list)}
    # Bucket by R54 margin (top-1 score gap) and R84 unique top-30 presence
    margin_buckets = {"low": [], "mid": [], "high": []}
    unique_buckets = {"r84_unique": [], "shared": []}
    for i in h7_idx:
        cf = case_features[i]
        s_r54 = all_scores[i]["r54"]
        s_r84 = all_scores[i]["r84_only"]
        order_r54 = np.argsort(-s_r54, kind="mergesort")
        margin = float(s_r54[order_r54[0]] - s_r54[order_r54[1]]) if len(order_r54) > 1 else 0
        rank_r54 = -1; rank_r84 = -1
        if cf["gt_pos"] >= 0:
            p = np.where(order_r54 == cf["gt_pos"])[0]
            if len(p):
                rank_r54 = int(p[0]) + 1
            order_r84 = np.argsort(-s_r84, kind="mergesort")
            p = np.where(order_r84 == cf["gt_pos"])[0]
            if len(p):
                rank_r84 = int(p[0]) + 1
        delta = ndcg_at_k(rank_r84, TOP_K) - ndcg_at_k(rank_r54, TOP_K)
        # R84 unique top-30 (the GT appears in R84 top-30 but not R54 top-30)
        owning_fold = case_fold[i]
        r84_top30 = set(r84_per_fold[owning_fold][i]["tids"][:30])
        r54_top30 = set((r54_source[i] or [])[:30])
        is_r84_unique_gt = (cases[i]["gt"] in r84_top30 and cases[i]["gt"] not in r54_top30)
        seg_data = {"case_idx": i, "delta": delta, "rank_r54": rank_r54,
                     "rank_r84": rank_r84, "margin": margin,
                     "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                     "is_r84_unique_gt": is_r84_unique_gt}
        if seg_data["same_artist"]:
            seg["same_artist"].append(seg_data)
        else:
            seg["diff_artist"].append(seg_data)
        seg["by_fold"][case_fold[i]].append(seg_data)
        # Margin buckets
        if margin < 0.5:
            margin_buckets["low"].append(seg_data)
        elif margin < 2.0:
            margin_buckets["mid"].append(seg_data)
        else:
            margin_buckets["high"].append(seg_data)
        unique_buckets["r84_unique" if is_r84_unique_gt else "shared"].append(seg_data)

    print(f"  h7 by same/diff artist:")
    for k in ["same_artist", "diff_artist"]:
        n_k = len(seg[k])
        d = float(np.mean([r["delta"] for r in seg[k]])) if seg[k] else 0
        n_pos = sum(1 for r in seg[k] if r["delta"] > 0)
        n_neg = sum(1 for r in seg[k] if r["delta"] < 0)
        print(f"    {k}: n={n_k}, mean_Δ={d:+.4f}, pos/neg/eq = {n_pos}/{n_neg}/"
              f"{n_k - n_pos - n_neg}")
    print(f"  h7 by R54 top-1 margin (proxy for R54 confidence):")
    for k in ["low", "mid", "high"]:
        n_k = len(margin_buckets[k])
        d = float(np.mean([r["delta"] for r in margin_buckets[k]])) if margin_buckets[k] else 0
        print(f"    margin {k} (n={n_k}): mean_Δ={d:+.4f}")
    print(f"  h7 by R84 unique top-30 GT presence:")
    for k in ["r84_unique", "shared"]:
        n_k = len(unique_buckets[k])
        d = float(np.mean([r["delta"] for r in unique_buckets[k]])) if unique_buckets[k] else 0
        print(f"    {k} (n={n_k}): mean_Δ={d:+.4f}")
    print(f"  h7 by fold:")
    for k in range(N_FOLDS):
        rows = seg["by_fold"][k]
        n_k = len(rows)
        d = float(np.mean([r["delta"] for r in rows])) if rows else 0
        print(f"    fold {k} (n={n_k}): mean_Δ={d:+.4f}")

    # --- Overall verdict ---
    print(f"\n{ts()} === FINAL R84b VERDICT ===", flush=True)
    candidates = []
    for name, s in A_results.items():
        p, g = gate_eval(s)
        candidates.append((f"A:{name}", p, s["h7"]["delta"], s, g))
    for beta_key, s in B_results.items():
        p, g = gate_eval(s)
        candidates.append((f"B:{beta_key}", p, s["h7"]["delta"], s, g))
    for key, s in C_results.items():
        p, g = gate_eval(s)
        candidates.append((f"C:{key}", p, s["h7"]["delta"], s, g))
    # Sort by h7 delta
    candidates.sort(key=lambda x: -x[2])
    print(f"  Top 5 candidates by h7_Δ:")
    for name, p, h7d, s, _ in candidates[:5]:
        print(f"    {name:35s}  h7_Δ={h7d:+.4f}  GATE={'PASS' if p else 'fail'}")
    passing = [c for c in candidates if c[1]]
    verdict = ("PROCEED_TO_BLIND" if passing else
               "INVESTIGATE" if max(c[2] for c in candidates) >= 0.0 else
               "ARCHIVE_SPRINT")
    print(f"\n  VERDICT: {verdict}")
    if passing:
        print(f"  {len(passing)} variant(s) cleared the gate. Best: {passing[0][0]}")

    out = {
        "experiment": "R84b — calibration/tuning sweep (CPU-only on Phase 1 artifacts)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "n_passing": len(passing),
        "best_variant": passing[0][0] if passing else candidates[0][0],
        "best_h7_delta": passing[0][2] if passing else candidates[0][2],
        "A_feature_interface": {k: v for k, v in A_results.items()},
        "B_score_blend": B_results,
        "C_lr_hyperparam": C_results,
        "best_A_chosen_for_C": best_A_name,
        "gate_definition": GATE,
        "top_5_by_h7_delta": [
            {"name": c[0], "h7_delta": c[2], "passes_gate": c[1]}
            for c in candidates[:5]
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
