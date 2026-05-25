"""R88 — Constrained multimodal ranker (CPU-only, 5-fold OOF vs R84c).

Last bounded attempt at multimodal conversion. Tests 3 conservative interfaces
that don't let raw multimodal features reorder the whole list (R85b's failure
mode).

Baseline: R84c sibling-R84 LR on R54-stacked pool.

Variants:
  V1 — Monotone LightGBM (43 cols, shallow trees, monotone constraints)
  V2 — Guarded additive boost (R84c score + beta * indicator boost)
  V3 — Quota injection (preserve R84c top-15, allow ≤2 multimodal swaps)

Gate per variant: h7 Δ ≥ +0.005, all Δ ≥ 0, same Δ ≥ -0.005, diff Δ ≥ 0,
recov ≥ lost, overlap ≥ 14/20.

Output: exp/eval/expR88_constrained_mm.json
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

from scripts.expR54_phase3_blind_submission import FEAT_R39_ALL, FEAT_ALL  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

N_FOLDS = 5
TOP_K = 20
POOL_K = 300
MODALITY_TOP_K = 300

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
MULTIMOD_LISTS_DIR = REPO / "cache" / "r85" / "multimodal_lists"
OUT_JSON = REPO / "exp" / "eval" / "expR88_constrained_mm.json"

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
FEAT_NAMES_R88 = FEAT_NAMES_R84_ONLY + [
    "img_rank_inv", "img_presence", "img_cosine",
    "meta_rank_inv", "meta_presence", "meta_cosine",
]

# Default LR params (matches R85a/c)
LR_PARAMS_BASELINE = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

# Gate (same as R84c, R85)
GATE = {
    "h7_delta_ge": 0.005, "all_delta_ge": 0.0,
    "same_artist_delta_ge": -0.005, "diff_artist_delta_ge": 0.0,
    "recov_ge_lost": True, "overlap_ge": 14.0,
}


def ts(): return f"[{datetime.now():%H:%M:%S}]"
def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def ndcg_at_k(r, k):
    return 1.0 / math.log2(r + 1) if 0 < r <= k else 0.0


def margin_of(s):
    s_sorted = np.sort(s)[::-1]
    return float(s_sorted[0] - s_sorted[1]) if len(s_sorted) >= 2 else 0.0


def train_lr(case_features, train_idx, feat_key, feat_names, params=LR_PARAMS_BASELINE,
              monotone=None):
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
    p = dict(params)
    if monotone is not None:
        p["monotone_constraints"] = monotone
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=feat_names)
    return lgb.train(p, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def extend_features_with_modality(case_features, img_lists, meta_lists):
    """43-col features by appending IMG + META rank/pres/cosine."""
    for i, cf in case_features.items():
        pool = cf["pool"]
        n_pool = len(pool)
        img = img_lists.get(i, [])
        meta = meta_lists.get(i, [])
        img_ranks = {t: r + 1 for r, (t, _) in enumerate(img[:MODALITY_TOP_K])}
        img_scores = {t: float(s) for t, s in img}
        meta_ranks = {t: r + 1 for r, (t, _) in enumerate(meta[:MODALITY_TOP_K])}
        meta_scores = {t: float(s) for t, s in meta}
        extra = np.zeros((n_pool, 6), dtype=np.float64)
        for k_row, tid in enumerate(pool):
            extra[k_row, 0] = (1.0 / img_ranks[tid]) if tid in img_ranks else 0.0
            extra[k_row, 1] = 1.0 if tid in img_ranks else 0.0
            extra[k_row, 2] = img_scores.get(tid, 0.0)
            extra[k_row, 3] = (1.0 / meta_ranks[tid]) if tid in meta_ranks else 0.0
            extra[k_row, 4] = 1.0 if tid in meta_ranks else 0.0
            extra[k_row, 5] = meta_scores.get(tid, 0.0)
        cf["feats_r88"] = np.concatenate([cf["feats_r84_only"], extra], axis=1)


def metrics_summary(rows_test, rows_baseline):
    def avg(rows, k): return float(np.mean([r[k] for r in rows])) if rows else 0.0
    h7 = [r for r in rows_test if r["n_prior_music"] == 7]
    h7_b = [r for r in rows_baseline if r["n_prior_music"] == 7]
    same = [r for r in rows_test if r["same_artist"]]
    same_b = [r for r in rows_baseline if r["same_artist"]]
    diff = [r for r in rows_test if not r["same_artist"]]
    diff_b = [r for r in rows_baseline if not r["same_artist"]]
    out = {}
    for n, (rt, rb) in [("h7", (h7, h7_b)), ("all", (rows_test, rows_baseline)),
                         ("same_artist", (same, same_b)), ("diff_artist", (diff, diff_b))]:
        out[n] = {"n": len(rt), "test": avg(rt, "ndcg20"),
                  "baseline": avg(rb, "ndcg20"),
                  "delta": avg(rt, "ndcg20") - avg(rb, "ndcg20")}
    h7_b_in = {r["case_idx"]: r["in_top20"] for r in h7_b}
    h7_t_in = {r["case_idx"]: r["in_top20"] for r in h7}
    recov = sum(1 for cid, t in h7_t_in.items() if t and not h7_b_in.get(cid, False))
    lost = sum(1 for cid, b in h7_b_in.items() if b and not h7_t_in.get(cid, False))
    out["h7_recovery"] = {"recovered": recov, "lost": lost, "net": recov - lost}
    b_top = {r["case_idx"]: set(r["top20"]) for r in rows_baseline}
    ov = [len(set(r["top20"]) & b_top.get(r["case_idx"], set())) for r in rows_test]
    out["overlap_mean"] = float(np.mean(ov)) if ov else 0.0
    # top-1 churn
    b_top1 = {r["case_idx"]: (r["top20"][0] if r["top20"] else None) for r in rows_baseline}
    t_top1 = {r["case_idx"]: (r["top20"][0] if r["top20"] else None) for r in rows_test}
    churn = sum(1 for cid in t_top1 if t_top1[cid] != b_top1.get(cid))
    out["top1_churn_per_80"] = (churn / len(rows_test)) * 80 if rows_test else 0
    return out


def gate_eval(s):
    g = {
        "h7": (s["h7"]["delta"] >= GATE["h7_delta_ge"], s["h7"]["delta"]),
        "all": (s["all"]["delta"] >= GATE["all_delta_ge"], s["all"]["delta"]),
        "same": (s["same_artist"]["delta"] >= GATE["same_artist_delta_ge"],
                  s["same_artist"]["delta"]),
        "diff": (s["diff_artist"]["delta"] >= GATE["diff_artist_delta_ge"],
                  s["diff_artist"]["delta"]),
        "recov": (s["h7_recovery"]["recovered"] >= s["h7_recovery"]["lost"],
                   [s["h7_recovery"]["recovered"], s["h7_recovery"]["lost"]]),
        "overlap": (s["overlap_mean"] >= GATE["overlap_ge"], s["overlap_mean"]),
    }
    return all(v[0] for v in g.values()), g


def rows_from_scores(cases, case_features, scores_per_case, maps, fold_for_case):
    rows = []
    for i, s in scores_per_case.items():
        cf = case_features[i]
        order = np.argsort(-s, kind="mergesort")
        rank = -1
        if cf["gt_pos"] >= 0:
            p = np.where(order == cf["gt_pos"])[0]
            if len(p):
                rank = int(p[0]) + 1
        rows.append({
            "case_idx": i, "fold": fold_for_case[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
        })
    return rows


def rows_from_orderings(cases, case_features, orderings_per_case, maps, fold_for_case):
    """orderings_per_case: case_idx -> list[int] of pool indices."""
    rows = []
    for i, order in orderings_per_case.items():
        cf = case_features[i]
        rank = -1
        if cf["gt_pos"] >= 0:
            try:
                rank = order.index(cf["gt_pos"]) + 1
            except ValueError:
                rank = -1
        rows.append({
            "case_idx": i, "fold": fold_for_case[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
        })
    return rows


def main():
    t0 = time.time()
    print(f"{ts()} R88 — constrained multimodal ranker")
    print("=" * 70)

    # Load
    print(f"\n{ts()} Loading payload + maps + features + multimodal lists...")
    payload, _, _, _ = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    maps, _, _ = load_supporting_maps()
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = {row["case_idx"]: int(row["fold_idx"]) for row in w0_stats}
    fold_for_case = case_fold
    fold_to_idx = {k: [i for i in range(n) if case_fold[i] == k] for k in range(N_FOLDS)}

    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)
    img_lists = {int(k): v for k, v in json.load(open(
        MULTIMOD_LISTS_DIR / "image_siglip_top300.json")).items()}
    meta_lists = {int(k): v for k, v in json.load(open(
        MULTIMOD_LISTS_DIR / "attributes_qwen_top300.json")).items()}
    extend_features_with_modality(case_features, img_lists, meta_lists)

    # --- Baseline: R84c sibling-R84 LR per fold + per-case margins ---
    print(f"\n{ts()} === Train baseline + sibling-R54 (for margins) ===")
    r84_scores = {}
    r54_margins = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t = time.time()
        lr_r84 = train_lr(case_features, train_idx, "feats_r84_only",
                            FEAT_NAMES_R84_ONLY)
        lr_r54 = train_lr(case_features, train_idx, "feats_r54", list(FEAT_ALL))
        for i in eval_idx:
            cf = case_features[i]
            r84_scores[i] = lr_r84.predict(cf["feats_r84_only"])
            s_r54 = lr_r54.predict(cf["feats_r54"])
            r54_margins[i] = margin_of(s_r54)
        print(f"  fold {fold_k}: {time.time() - t:.0f}s")
    rows_baseline = rows_from_scores(cases, case_features, r84_scores, maps, fold_for_case)
    base_h7 = float(np.mean([r["ndcg20"] for r in rows_baseline if r["n_prior_music"] == 7]))
    base_all = float(np.mean([r["ndcg20"] for r in rows_baseline]))
    print(f"\n  R84c baseline: h7 nDCG={base_h7:.4f}  all nDCG={base_all:.4f}")

    results = {}

    # === V1: MONOTONE LightGBM (43 col, shallow trees) ===
    print(f"\n{ts()} === V1: Monotone LightGBM (43 col, shallow trees) ===")
    # monotone: 37 zeros (R39+R84 features unconstrained), then 6 ones (IMG+META all +1)
    # Then we'll also try variant where R84 features are forced +1 too
    monotone_constraints = ([0] * len(FEAT_R39_ALL) + [1, 1, 1] +  # R39 free, R84 +1
                             [1, 1, 1, 1, 1, 1])  # IMG +1, META +1
    assert len(monotone_constraints) == 43

    v1_results = {}
    for num_leaves in [7, 15]:
        for mdif in [50]:
            for l2 in [1, 5]:
                params = dict(LR_PARAMS_BASELINE)
                params["num_leaves"] = num_leaves
                params["min_data_in_leaf"] = mdif
                params["lambda_l2"] = l2
                cfg_key = f"V1_nl{num_leaves}_mdif{mdif}_l2{l2}"
                v1_scores = {}
                for fold_k in range(N_FOLDS):
                    train_idx = [i for i in range(n) if case_fold[i] != fold_k]
                    eval_idx = fold_to_idx[fold_k]
                    lr = train_lr(case_features, train_idx, "feats_r88",
                                    FEAT_NAMES_R88, params=params,
                                    monotone=monotone_constraints)
                    for i in eval_idx:
                        v1_scores[i] = lr.predict(case_features[i]["feats_r88"])
                rows = rows_from_scores(cases, case_features, v1_scores, maps, fold_for_case)
                summary = metrics_summary(rows, rows_baseline)
                passes, _ = gate_eval(summary)
                results[cfg_key] = summary
                v1_results[cfg_key] = summary
                print(f"  {cfg_key}: h7_Δ={summary['h7']['delta']:+.4f}  "
                      f"same_Δ={summary['same_artist']['delta']:+.4f}  "
                      f"rec/lost={summary['h7_recovery']['recovered']}/"
                      f"{summary['h7_recovery']['lost']}  "
                      f"ovl={summary['overlap_mean']:.2f}  "
                      f"churn={summary['top1_churn_per_80']:.1f}/80  "
                      f"{'PASS' if passes else 'fail'}")

    # === V2: GUARDED ADDITIVE BOOST ===
    print(f"\n{ts()} === V2: Guarded additive boost ===")
    # Pre-compute per-case helpers
    img_tids_per_case = {i: set(t for t, _ in img_lists.get(i, [])) for i in range(n)}
    meta_tids_per_case = {i: set(t for t, _ in meta_lists.get(i, [])) for i in range(n)}
    # R84c top-50 per case from r84_scores
    r84_top50_per_case = {}
    for i, s in r84_scores.items():
        cf = case_features[i]
        order = np.argsort(-s, kind="mergesort")
        r84_top50_per_case[i] = set(cf["pool"][int(j)] for j in order[:50])

    def boost_scores(beta, top_k_mod, margin_low_thr=0.5):
        """Returns dict of orderings (pool index lists) per case."""
        orderings = {}
        n_boosts_total = 0
        for i in range(n):
            cf = case_features[i]
            pool = cf["pool"]
            base = r84_scores[i].copy()
            margin = r54_margins.get(i, 0.5)
            if margin < margin_low_thr:
                # Build the in-top-k modality sets for this case
                img_topk = set(t for t, _ in img_lists.get(i, [])[:top_k_mod])
                meta_topk = set(t for t, _ in meta_lists.get(i, [])[:top_k_mod])
                r84_top50 = r84_top50_per_case[i]
                for k_row, tid in enumerate(pool):
                    if ((tid in img_topk or tid in meta_topk) and tid in r84_top50):
                        base[k_row] += beta
                        n_boosts_total += 1
            order = np.argsort(-base, kind="mergesort").tolist()
            orderings[i] = order
        return orderings, n_boosts_total

    v2_results = {}
    for beta in [0.02, 0.05, 0.10, 0.15]:
        for top_k_mod in [10, 20, 30]:
            t = time.time()
            orderings, n_boosts = boost_scores(beta, top_k_mod)
            rows = rows_from_orderings(cases, case_features, orderings, maps, fold_for_case)
            summary = metrics_summary(rows, rows_baseline)
            passes, _ = gate_eval(summary)
            cfg_key = f"V2_beta{beta}_topK{top_k_mod}"
            results[cfg_key] = summary
            v2_results[cfg_key] = summary
            print(f"  {cfg_key}: h7_Δ={summary['h7']['delta']:+.4f}  "
                  f"same_Δ={summary['same_artist']['delta']:+.4f}  "
                  f"rec/lost={summary['h7_recovery']['recovered']}/"
                  f"{summary['h7_recovery']['lost']}  "
                  f"ovl={summary['overlap_mean']:.2f}  "
                  f"churn={summary['top1_churn_per_80']:.1f}/80  "
                  f"n_boosts={n_boosts}  "
                  f"{'PASS' if passes else 'fail'}")

    # === V3: QUOTA INJECTION ===
    print(f"\n{ts()} === V3: Quota injection (preserve R84c top-15, ≤2 swaps) ===")
    def quota_orderings(max_swaps, replace_rank_window, score_proximity_pct):
        """For each case:
         - Keep R84c top-15 ranks.
         - Look at R84c ranks 16-50 for candidates to swap IN that are in img/meta top-30.
         - Allow up to `max_swaps` swaps where the multimodal candidate's R84c score
           is within `score_proximity_pct` (relative) of the candidate it replaces (top-16..20).
        """
        orderings = {}
        n_swaps_total = 0
        for i in range(n):
            cf = case_features[i]
            pool = cf["pool"]
            base = r84_scores[i]
            order = np.argsort(-base, kind="mergesort").tolist()
            top15 = order[:15]
            top16_to_20 = order[15:20]
            top16_to_50 = order[15:50]
            # Candidate set: tids in img/meta top-30 that are in R84c top16..50 (replacement) AND
            # not already in top15
            img_top30 = set(t for t, _ in img_lists.get(i, [])[:30])
            meta_top30 = set(t for t, _ in meta_lists.get(i, [])[:30])
            top15_tids = set(pool[idx] for idx in top15)
            candidates = []  # list of (pool_idx, score)
            for pool_idx in top16_to_50:
                tid = pool[pool_idx]
                if tid in top15_tids:
                    continue
                if tid in img_top30 or tid in meta_top30:
                    candidates.append((pool_idx, float(base[pool_idx])))
            # Sort candidates by score descending (the most "deserving")
            candidates.sort(key=lambda x: -x[1])
            # Build new top-20: take top15 + select replacements from candidates
            new_top20 = list(top15)
            current_top16_20 = list(top16_to_20)
            swaps_done = 0
            for cand_idx, cand_score in candidates:
                if swaps_done >= max_swaps:
                    break
                if cand_idx in new_top20:
                    continue  # already inside
                # Compare to the WEAKEST current_top16_20 item
                if not current_top16_20:
                    break
                weak_idx = max(current_top16_20, key=lambda x: -base[x])  # weakest = lowest score
                # weakest is actually the LAST one in score order; let me redo
                # current_top16_20 is sorted by score desc already (came from order)
                weak_idx = current_top16_20[-1]  # last = weakest
                weak_score = float(base[weak_idx])
                # Score proximity check
                if weak_score == 0:
                    score_ok = cand_score >= 0
                else:
                    score_ok = (weak_score - cand_score) / abs(weak_score) <= score_proximity_pct
                if score_ok:
                    current_top16_20.remove(weak_idx)
                    current_top16_20.append(cand_idx)
                    # Re-sort current_top16_20 by score desc
                    current_top16_20.sort(key=lambda x: -base[x])
                    swaps_done += 1
            new_top20 = list(top15) + list(current_top16_20)
            # Build full ordering: new_top20 + remaining (original order)
            top20_set = set(new_top20)
            rest = [idx for idx in order if idx not in top20_set]
            orderings[i] = new_top20 + rest
            n_swaps_total += swaps_done
        return orderings, n_swaps_total

    v3_results = {}
    for max_swaps in [1, 2]:
        for prox_pct in [0.05, 0.10, 0.20]:
            orderings, n_swaps = quota_orderings(max_swaps, 50, prox_pct)
            rows = rows_from_orderings(cases, case_features, orderings, maps, fold_for_case)
            summary = metrics_summary(rows, rows_baseline)
            passes, _ = gate_eval(summary)
            cfg_key = f"V3_swaps{max_swaps}_prox{prox_pct}"
            results[cfg_key] = summary
            v3_results[cfg_key] = summary
            print(f"  {cfg_key}: h7_Δ={summary['h7']['delta']:+.4f}  "
                  f"same_Δ={summary['same_artist']['delta']:+.4f}  "
                  f"rec/lost={summary['h7_recovery']['recovered']}/"
                  f"{summary['h7_recovery']['lost']}  "
                  f"ovl={summary['overlap_mean']:.2f}  "
                  f"churn={summary['top1_churn_per_80']:.1f}/80  "
                  f"n_swaps={n_swaps}  "
                  f"{'PASS' if passes else 'fail'}")

    # --- Verdict ---
    print(f"\n{ts()} === VERDICT ===")
    candidates = []
    for k, s in results.items():
        p, _ = gate_eval(s)
        candidates.append((k, p, s["h7"]["delta"], s))
    candidates.sort(key=lambda x: -x[2])
    print(f"  Top 5 by h7 Δ:")
    for k, p, h7d, s in candidates[:5]:
        print(f"    {k:30s}  h7_Δ={h7d:+.4f}  "
              f"same_Δ={s['same_artist']['delta']:+.4f}  "
              f"GATE={'PASS' if p else 'fail'}")
    passing = [c for c in candidates if c[1]]
    verdict = "PROCEED_TO_BLIND" if passing else "ARCHIVE_MM"
    print(f"\n  Passing configs: {len(passing)}")
    print(f"  VERDICT: {verdict}")
    if passing:
        print(f"  Best passing: {passing[0][0]}  h7_Δ={passing[0][2]:+.4f}")

    out = {
        "experiment": "R88 — constrained multimodal ranker (CPU)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "n_passing": len(passing),
        "best_variant": passing[0][0] if passing else candidates[0][0],
        "best_h7_delta": passing[0][2] if passing else candidates[0][2],
        "baseline_h7": base_h7,
        "baseline_all": base_all,
        "V1_monotone_lightgbm": v1_results,
        "V2_guarded_boost": v2_results,
        "V3_quota_injection": v3_results,
        "gate_definition": GATE,
        "top5_by_h7": [
            {"name": c[0], "h7_delta": c[2], "passes": c[1]}
            for c in candidates[:5]
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
