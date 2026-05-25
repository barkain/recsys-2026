"""R85c — selective routing between R85a (10-source pool + multimodal) and R84c
(R54-stacked, R84-LR ranked).

Predeclared rule:
- Use R85a top-20 if R54c sibling-R54 margin < LOW_THR AND IMG top-1 score >= IMG_THR.
- Else use R84c top-20 (= sibling-R84 LR on R54-stacked pool).

This combines:
- R84c's confidence-based routing (R54 unsure → use stronger ranker)
- New: only invoke R85a when image SigLIP is itself confident (avoid noise where
  multimodal anchor weak)

Threshold sweep around predeclared:
- LOW_THR ∈ {0.25, 0.50, 0.75}
- IMG_THR ∈ {0.30, 0.40, 0.50}

Baseline: R84c sibling-R84 LR on R54-stacked pool, 5-fold OOF.

Outputs: exp/eval/expR85c_selective_routing.json
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
    FEAT_R39_ALL, FEAT_ALL, _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

N_FOLDS = 5
TOP_K = 20
POOL_K = 300
RRF_K = 20
MODALITY_TOP_K = 300

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
MULTIMOD_LISTS_DIR = REPO / "cache" / "r85" / "multimodal_lists"
OUT_JSON = REPO / "exp" / "eval" / "expR85c_selective_routing.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]

GATE = {
    "h7_delta_ge": 0.005, "all_delta_ge": 0.0,
    "same_artist_delta_ge": -0.005, "diff_artist_delta_ge": 0.0,
    "recov_ge_lost": True, "overlap_ge": 14.0,
}

PREDECLARED = {"low_thr": 0.5, "img_thr": 0.4}
SWEEP_LOW = [0.25, 0.5, 0.75]
SWEEP_IMG = [0.30, 0.40, 0.50]


def ts(): return f"[{datetime.now():%H:%M:%S}]"
def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def ndcg_at_k(r, k):
    return 1.0 / math.log2(r + 1) if 0 < r <= k else 0.0


def margin_of(s):
    sorted_s = np.sort(s)[::-1]
    return float(sorted_s[0] - sorted_s[1]) if len(sorted_s) >= 2 else 0.0


def train_sibling_lr(case_features, train_idx, feat_key, feat_names):
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
    return lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def compute_metrics(rows_test, rows_baseline):
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


def main():
    t0 = time.time()
    print(f"{ts()} R85c — selective routing R85a vs R84c by R54c margin + IMG strength")
    print("=" * 70)

    print(f"\n{ts()} Loading fundamentals...")
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = {row["case_idx"]: int(row["fold_idx"]) for row in w0_stats}
    fold_to_idx = {k: [i for i in range(n) if case_fold[i] == k] for k in range(N_FOLDS)}

    print(f"\n{ts()} Loading cached multimodal top-300 lists...")
    img_lists = {int(k): v for k, v in json.load(open(
        MULTIMOD_LISTS_DIR / "image_siglip_top300.json")).items()}
    meta_lists = {int(k): v for k, v in json.load(open(
        MULTIMOD_LISTS_DIR / "attributes_qwen_top300.json")).items()}
    print(f"  img: {len(img_lists)}, meta: {len(meta_lists)}")

    # IMG top-1 score per case (for routing rule)
    img_top1_score = {}
    for i, pairs in img_lists.items():
        img_top1_score[i] = float(pairs[0][1]) if pairs else 0.0
    img_arr = np.array([img_top1_score.get(i, 0.0) for i in range(n)])
    print(f"  IMG top-1 score: p25={np.percentile(img_arr, 25):.3f} "
          f"p50={np.percentile(img_arr, 50):.3f} p75={np.percentile(img_arr, 75):.3f} "
          f"max={img_arr.max():.3f}")

    # Load case_features (R54-stacked pool, already built)
    print(f"\n{ts()} Loading case_features cache ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)...")
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)

    # --- Train per-fold sibling LRs: R54 (for margins) + R84 (for baseline) ---
    print(f"\n{ts()} === Train per-fold sibling LRs (R54 + R84) on R54-stacked pool ===")
    r54_scores_pc = {}
    r84_scores_pc = {}
    margins_pc = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t = time.time()
        lr_r54 = train_sibling_lr(case_features, train_idx, "feats_r54", list(FEAT_ALL))
        lr_r84 = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                    FEAT_NAMES_R84_ONLY)
        for i in eval_idx:
            cf = case_features[i]
            s_r54 = lr_r54.predict(cf["feats_r54"])
            s_r84 = lr_r84.predict(cf["feats_r84_only"])
            r54_scores_pc[i] = s_r54
            r84_scores_pc[i] = s_r84
            margins_pc[i] = margin_of(s_r54)
        print(f"  fold {fold_k}: {time.time() - t:.0f}s")

    # --- Baseline rows: R84c sibling on R54-stacked pool ---
    print(f"\n{ts()} Building baseline (R84c sibling) rows...")
    rows_baseline = []
    for i in range(n):
        cf = case_features[i]
        s = r84_scores_pc[i]
        order = np.argsort(-s, kind="mergesort")
        rank = -1
        if cf["gt_pos"] >= 0:
            p = np.where(order == cf["gt_pos"])[0]
            if len(p):
                rank = int(p[0]) + 1
        rows_baseline.append({
            "case_idx": i, "fold": case_fold[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
        })

    # --- Build R85a per-fold (10-source pool + sibling R84 LR) ---
    print(f"\n{ts()} === Build R85a 10-source pool + sibling R84 LR per fold ===")
    print(f"  Building case_index for R85a pool...")
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )
    case_index["als_to_idx"] = als_to_idx

    sw_r85a = {**SW_BASELINE, "IMG": 0.5, "META": 0.5}
    r85a_pool_per_case = {}
    print(f"  Building R85a 10-source RRF pool per case...")
    t = time.time()
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        src_lists["IMG"] = [t for t, _ in img_lists.get(i, [])][:MODALITY_TOP_K]
        src_lists["META"] = [t for t, _ in meta_lists.get(i, [])][:MODALITY_TOP_K]
        r85a_pool_per_case[i] = weighted_rrf(src_lists, sw_r85a, topk=POOL_K, k=RRF_K)
    print(f"    done in {time.time() - t:.0f}s")

    # Pre-load R84 OOF lists for re-featurize
    r84_per_fold = {}
    for k in range(N_FOLDS):
        path = (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
                 if k == 0 else
                 REPO / f"cache/r84/phase1_fold{k}/oof_r84_lists.json")
        raw = json.load(open(path))
        r84_per_fold[k] = {
            int(cidx): {"ranks": {t: r + 1 for r, (t, _) in enumerate(v)},
                         "scores": {t: float(s) for t, s in v}}
            for cidx, v in raw.items()
        }

    # Re-featurize on R85a pool (37 cols, R84-substituted)
    print(f"\n  Re-featurizing on R85a pool...")
    r85a_case_features = {}
    t = time.time()
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        src_lists["IMG"] = [t for t, _ in img_lists.get(i, [])][:MODALITY_TOP_K]
        src_lists["META"] = [t for t, _ in meta_lists.get(i, [])][:MODALITY_TOP_K]
        pool = r85a_pool_per_case[i]
        gt_pos = pool.index(cases[i]["gt"]) if cases[i]["gt"] in pool else -1
        r21_rank_map = {tt: r + 1 for r, tt in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tt: r + 1 for r, tt in enumerate(src_lists["R54"][:POOL_K])}
        case = cases[i]
        feats_r54_new = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album,
        )
        r84_data = r84_per_fold[case_fold[i]].get(i, {"ranks": {}, "scores": {}})
        feats_r84 = feats_r54_new.copy()
        for k_row, tid in enumerate(pool):
            feats_r84[k_row, N_R39 + 0] = (1.0 / r84_data["ranks"][tid]) if tid in r84_data["ranks"] else 0.0
            feats_r84[k_row, N_R39 + 1] = 1.0 if tid in r84_data["ranks"] else 0.0
            feats_r84[k_row, N_R39 + 2] = r84_data["scores"].get(tid, 0.0)
        r85a_case_features[i] = {"pool": pool, "gt_pos": gt_pos,
                                   "feats_r84_only": feats_r84}
    print(f"    done in {time.time() - t:.0f}s")

    # R85a per-fold sibling LR
    print(f"\n  R85a per-fold sibling LR...")
    r85a_scores_pc = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t = time.time()
        lr = train_sibling_lr(r85a_case_features, train_idx, "feats_r84_only",
                                FEAT_NAMES_R84_ONLY)
        for i in eval_idx:
            r85a_scores_pc[i] = lr.predict(r85a_case_features[i]["feats_r84_only"])
        print(f"    fold {fold_k}: {time.time() - t:.0f}s")

    # Build R85a-only rows for sanity
    rows_r85a_only = []
    for i in range(n):
        cf = r85a_case_features[i]
        s = r85a_scores_pc[i]
        order = np.argsort(-s, kind="mergesort")
        rank = -1
        if cf["gt_pos"] >= 0:
            p = np.where(order == cf["gt_pos"])[0]
            if len(p):
                rank = int(p[0]) + 1
        rows_r85a_only.append({
            "case_idx": i, "fold": case_fold[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
        })

    # --- Selective routing variants ---
    def routed_rows(low_thr, img_thr):
        rows = []
        n_to_r85a = 0
        n_to_r84c = 0
        for i in range(n):
            use_r85a = (margins_pc[i] < low_thr) and (img_top1_score.get(i, 0) >= img_thr)
            if use_r85a:
                n_to_r85a += 1
                cf = r85a_case_features[i]
                s = r85a_scores_pc[i]
            else:
                n_to_r84c += 1
                cf = case_features[i]
                s = r84_scores_pc[i]
            order = np.argsort(-s, kind="mergesort")
            rank = -1
            if cf["gt_pos"] >= 0:
                p = np.where(order == cf["gt_pos"])[0]
                if len(p):
                    rank = int(p[0]) + 1
            rows.append({
                "case_idx": i, "fold": case_fold[i],
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
                "in_top20": rank > 0 and rank <= TOP_K,
                "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
            })
        return rows, n_to_r85a, n_to_r84c

    print(f"\n{ts()} === PREDECLARED RULE (low={PREDECLARED['low_thr']}, "
          f"img>={PREDECLARED['img_thr']}) ===")
    rows_pre, n_85a, n_84c = routed_rows(PREDECLARED["low_thr"], PREDECLARED["img_thr"])
    sum_pre = compute_metrics(rows_pre, rows_baseline)
    p_pre, gates_pre = gate_eval(sum_pre)
    print(f"  routed: R85a={n_85a} / R84c={n_84c} ({n_85a/n:.1%} R85a)")
    for k in ["h7", "all", "same_artist", "diff_artist"]:
        m = sum_pre[k]
        print(f"    {k:14}  n={m['n']:5d}  base={m['baseline']:.4f}  "
              f"r85c={m['test']:.4f}  Δ={m['delta']:+.4f}")
    rec = sum_pre["h7_recovery"]
    print(f"    recov/lost = {rec['recovered']}/{rec['lost']}  net={rec['net']:+d}")
    print(f"    overlap = {sum_pre['overlap_mean']:.2f}/20")
    print(f"    GATE: {'PASS' if p_pre else 'fail'}")

    print(f"\n{ts()} === SWEEP ===")
    sweep = {}
    best = (None, -1e9, None)
    for low_thr in SWEEP_LOW:
        for img_thr in SWEEP_IMG:
            rows_s, n85, n84 = routed_rows(low_thr, img_thr)
            s = compute_metrics(rows_s, rows_baseline)
            p, _ = gate_eval(s)
            key = f"low{low_thr}_img{img_thr}"
            sweep[key] = {
                "low_thr": low_thr, "img_thr": img_thr,
                "n_routed_r85a": n85, "n_routed_r84c": n84,
                "summary": s, "passes_gate": p,
            }
            print(f"  {key}: routed={n85}/{n84}  "
                  f"h7_Δ={s['h7']['delta']:+.4f}  "
                  f"all_Δ={s['all']['delta']:+.4f}  "
                  f"same_Δ={s['same_artist']['delta']:+.4f}  "
                  f"diff_Δ={s['diff_artist']['delta']:+.4f}  "
                  f"rec/lost={s['h7_recovery']['recovered']}/"
                  f"{s['h7_recovery']['lost']}  "
                  f"ovl={s['overlap_mean']:.2f}  "
                  f"GATE={'PASS' if p else 'fail'}")
            if s["h7"]["delta"] > best[1]:
                best = (key, s["h7"]["delta"], sweep[key])
    print(f"\n  Best sweep by h7_Δ: {best[0]}  h7_Δ={best[1]:+.4f}")

    # R85a-only (no routing) for comparison
    sum_r85a_only = compute_metrics(rows_r85a_only, rows_baseline)
    p_r85a_only, _ = gate_eval(sum_r85a_only)
    print(f"\n  R85a-only (no routing) baseline: h7_Δ={sum_r85a_only['h7']['delta']:+.4f} "
          f"same_Δ={sum_r85a_only['same_artist']['delta']:+.4f} "
          f"rec/lost={sum_r85a_only['h7_recovery']['recovered']}/"
          f"{sum_r85a_only['h7_recovery']['lost']} "
          f"GATE={'PASS' if p_r85a_only else 'fail'}")

    # --- Verdict ---
    any_pass = p_pre or any(v["passes_gate"] for v in sweep.values()) or p_r85a_only
    verdict = "PROCEED_TO_BLIND_CANDIDATE" if any_pass else "INVESTIGATE_OR_ARCHIVE"
    print(f"\n{ts()} === VERDICT: {verdict} ===")
    if any_pass:
        passing = []
        if p_pre: passing.append(("PRE-DECLARED", sum_pre["h7"]["delta"]))
        if p_r85a_only: passing.append(("R85a-only", sum_r85a_only["h7"]["delta"]))
        for k, v in sweep.items():
            if v["passes_gate"]:
                passing.append((k, v["summary"]["h7"]["delta"]))
        passing.sort(key=lambda x: -x[1])
        for nm, h7d in passing:
            print(f"  ✓ {nm}: h7_Δ={h7d:+.4f}")

    out = {
        "experiment": "R85c — selective routing R85a vs R84c",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "predeclared": {
            "rule": f"use R85a if margin < {PREDECLARED['low_thr']} AND img_top1 >= {PREDECLARED['img_thr']}",
            "n_routed_r85a": n_85a, "n_routed_r84c": n_84c,
            "summary": sum_pre, "passes_gate": p_pre,
        },
        "sweep": sweep,
        "r85a_only": {"summary": sum_r85a_only, "passes_gate": p_r85a_only},
        "best_sweep_key": best[0],
        "best_sweep_h7_delta": best[1],
        "gate_definition": GATE,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
