"""R84c — Selective deployment test (observable-feature routing).

Routing rule: per case, use R84 sibling LR ranking when the R54 sibling LR's
top-1 margin is < LOW_THR or >= HIGH_THR; else use R54 sibling LR ranking.

The R54 top-1 margin is observable at inference time (= softmax-like gap
between the top-1 and top-2 sibling-R54-LR scores), so this is a deployable
gate, not a fold-id-based cheat.

- Pre-declared rule: LOW=0.5, HIGH=2.0
- Threshold sweep: LOW in {0.25, 0.5, 0.75}, HIGH in {1.5, 2.0, 2.5}
- Baseline for Δ: sibling-R54 (5-fold OOF) — apples-to-apples
- All metrics across all 8000 cases / 1000 h7

Output: exp/eval/expR84c_selective.json
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

from scripts.expR54_phase3_blind_submission import FEAT_R39_ALL, FEAT_ALL  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

N_FOLDS = 5
TOP_K = 20
POOL_K = 300

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR84c_selective.json"

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

# R84c PROCEED gate (same as R84b)
GATE = {
    "h7_delta_ge": 0.005,
    "all_delta_ge": 0.0,
    "same_artist_delta_ge": -0.005,
    "diff_artist_delta_ge": 0.0,
    "recov_ge_lost": True,
    "overlap_ge": 14.0,
}

PREDECLARED = {"low_thr": 0.5, "high_thr": 2.0}
SWEEP_LOW = [0.25, 0.5, 0.75]
SWEEP_HIGH = [1.5, 2.0, 2.5]


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


def compute_metrics(rows_test, rows_baseline, rows_overlap):
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    sub_h7 = [r for r in rows_test if r["n_prior_music"] == 7]
    sub_h7_b = [r for r in rows_baseline if r["n_prior_music"] == 7]
    sub_same = [r for r in rows_test if r["same_artist"]]
    sub_same_b = [r for r in rows_baseline if r["same_artist"]]
    sub_diff = [r for r in rows_test if not r["same_artist"]]
    sub_diff_b = [r for r in rows_baseline if not r["same_artist"]]
    out = {}
    for name, (rt, rb) in [("h7", (sub_h7, sub_h7_b)),
                            ("all", (rows_test, rows_baseline)),
                            ("same_artist", (sub_same, sub_same_b)),
                            ("diff_artist", (sub_diff, sub_diff_b))]:
        out[name] = {"n": len(rt), "test": avg(rt, "ndcg20"),
                     "baseline": avg(rb, "ndcg20"),
                     "delta": avg(rt, "ndcg20") - avg(rb, "ndcg20")}

    # h7 recovery (top-20 in/out vs baseline)
    h7_b_in = {r["case_idx"]: r["in_top20"] for r in sub_h7_b}
    h7_t_in = {r["case_idx"]: r["in_top20"] for r in sub_h7}
    recov = sum(1 for cid, t in h7_t_in.items() if t and not h7_b_in.get(cid, False))
    lost = sum(1 for cid, b in h7_b_in.items() if b and not h7_t_in.get(cid, False))
    out["h7_recovery"] = {"recovered": recov, "lost": lost, "net": recov - lost}

    # top-1 churn (test vs baseline)
    b_top1 = {r["case_idx"]: r["top20"][0] if r["top20"] else None for r in rows_baseline}
    t_top1 = {r["case_idx"]: r["top20"][0] if r["top20"] else None for r in rows_test}
    churn = sum(1 for cid in t_top1 if t_top1[cid] != b_top1.get(cid))
    out["top1_churn_per_80"] = (churn / len(rows_test)) * 80 if rows_test else 0

    # top-20 overlap mean (test vs overlap-source)
    ov_top20 = {r["case_idx"]: set(r["top20"]) for r in rows_overlap}
    ovs = [len(set(r["top20"]) & ov_top20.get(r["case_idx"], set())) for r in rows_test]
    out["overlap_mean"] = float(np.mean(ovs)) if ovs else 0.0
    return out


def gate_eval(s):
    h7_d = s["h7"]["delta"]
    all_d = s["all"]["delta"]
    sa_d = s["same_artist"]["delta"]
    diff_d = s["diff_artist"]["delta"]
    recov, lost = s["h7_recovery"]["recovered"], s["h7_recovery"]["lost"]
    overlap = s["overlap_mean"]
    gates = {
        "h7_delta_ge_p005": (h7_d >= GATE["h7_delta_ge"], h7_d),
        "all_delta_ge_0": (all_d >= GATE["all_delta_ge"], all_d),
        "same_artist_delta_ge_n005": (sa_d >= GATE["same_artist_delta_ge"], sa_d),
        "diff_artist_delta_ge_0": (diff_d >= GATE["diff_artist_delta_ge"], diff_d),
        "recov_ge_lost": (recov >= lost, [recov, lost]),
        "overlap_ge_14": (overlap >= GATE["overlap_ge"], overlap),
    }
    return all(v[0] for v in gates.values()), gates


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R84c — selective deployment test")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + maps ...", flush=True)
    payload, _, r54_source, _ = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    maps, _, _ = load_supporting_maps()

    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_to_idx = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)

    # --- Load case_features cache ---
    if not FEAT_CACHE.exists():
        print(f"ERROR: feature cache missing at {FEAT_CACHE}. Run expR84b_sweep.py first.")
        sys.exit(1)
    print(f"{ts()} Loading case_features cache ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)...",
          flush=True)
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)
    print(f"  {len(case_features)} cases loaded", flush=True)

    # --- Train per-fold sibling LRs (R54 + R84-only), score eval folds ---
    print(f"\n{ts()} === Train per-fold sibling LRs and score eval folds ===", flush=True)
    per_case_scores: dict[int, dict[str, np.ndarray]] = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t_lr = time.time()
        lr_r54 = train_sibling_lr(case_features, train_idx, "feats_r54",
                                    list(FEAT_ALL))
        lr_r84 = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                    FEAT_NAMES_R84_ONLY)
        for i in eval_idx:
            cf = case_features[i]
            per_case_scores[i] = {
                "r54": lr_r54.predict(cf["feats_r54"]),
                "r84": lr_r84.predict(cf["feats_r84_only"]),
            }
        print(f"  fold {fold_k}: {time.time() - t_lr:.0f}s", flush=True)

    # --- Build rows for each pure variant ---
    print(f"\n{ts()} === Build baseline (R54) and pure-R84 rows ===", flush=True)
    def make_rows(score_key):
        out = []
        for i, sd in per_case_scores.items():
            cf = case_features[i]
            s = sd[score_key]
            order = np.argsort(-s, kind="mergesort")
            rank = -1
            if cf["gt_pos"] >= 0:
                p = np.where(order == cf["gt_pos"])[0]
                if len(p):
                    rank = int(p[0]) + 1
            out.append({
                "case_idx": i, "fold": case_fold[i],
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
                "in_top20": rank > 0 and rank <= TOP_K,
                "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
            })
        return out

    rows_r54_pure = make_rows("r54")
    rows_r84_pure = make_rows("r84")

    # Pure r84 baseline for cross-check (= Phase 1 baseline)
    pure_summary = compute_metrics(rows_r84_pure, rows_r54_pure, rows_r54_pure)
    p_pure, _ = gate_eval(pure_summary)
    print(f"\n  Pure R84 (Phase 1 baseline): h7_Δ={pure_summary['h7']['delta']:+.4f}  "
          f"same_Δ={pure_summary['same_artist']['delta']:+.4f}  "
          f"rec/lost={pure_summary['h7_recovery']['recovered']}/"
          f"{pure_summary['h7_recovery']['lost']}  GATE={'PASS' if p_pure else 'fail'}")

    # --- Compute R54 top-1 margin per case ---
    print(f"\n{ts()} === Compute R54 top-1 margins ===", flush=True)
    case_margin = {}
    for i, sd in per_case_scores.items():
        s_r54 = sd["r54"]
        if len(s_r54) >= 2:
            sorted_s = np.sort(s_r54)[::-1]
            case_margin[i] = float(sorted_s[0] - sorted_s[1])
        else:
            case_margin[i] = 0.0
    margins = np.array(list(case_margin.values()))
    print(f"  margin distribution: p25={np.percentile(margins, 25):.3f} "
          f"p50={np.percentile(margins, 50):.3f} "
          f"p75={np.percentile(margins, 75):.3f} "
          f"p90={np.percentile(margins, 90):.3f} max={margins.max():.3f}")

    # --- Build selective rows for a given (low, high) threshold ---
    def selective_rows(low_thr, high_thr):
        out = []
        n_routed_r84 = 0
        n_routed_r54 = 0
        for i, sd in per_case_scores.items():
            cf = case_features[i]
            margin = case_margin[i]
            # Routing rule
            use_r84 = (margin < low_thr) or (margin >= high_thr)
            s = sd["r84"] if use_r84 else sd["r54"]
            if use_r84:
                n_routed_r84 += 1
            else:
                n_routed_r54 += 1
            order = np.argsort(-s, kind="mergesort")
            rank = -1
            if cf["gt_pos"] >= 0:
                p = np.where(order == cf["gt_pos"])[0]
                if len(p):
                    rank = int(p[0]) + 1
            out.append({
                "case_idx": i, "fold": case_fold[i],
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
                "in_top20": rank > 0 and rank <= TOP_K,
                "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
            })
        return out, n_routed_r84, n_routed_r54

    # --- Pre-declared rule (LOW=0.5, HIGH=2.0) ---
    print(f"\n{ts()} === PRE-DECLARED: LOW={PREDECLARED['low_thr']}, "
          f"HIGH={PREDECLARED['high_thr']} ===", flush=True)
    rows_pre, n_r84_pre, n_r54_pre = selective_rows(
        PREDECLARED["low_thr"], PREDECLARED["high_thr"])
    sum_pre = compute_metrics(rows_pre, rows_r54_pure, rows_r54_pure)
    p_pre, gates_pre = gate_eval(sum_pre)

    # Per-fold h7 breakdown
    def per_fold_h7(rows, baseline):
        out = {}
        for k in range(N_FOLDS):
            rt = [r for r in rows if r["fold"] == k and r["n_prior_music"] == 7]
            rb = [r for r in baseline if r["fold"] == k and r["n_prior_music"] == 7]
            t = float(np.mean([r["ndcg20"] for r in rt])) if rt else 0
            b = float(np.mean([r["ndcg20"] for r in rb])) if rb else 0
            out[k] = {"n": len(rt), "test": t, "baseline": b, "delta": t - b}
        return out
    per_fold_pre = per_fold_h7(rows_pre, rows_r54_pure)

    def report_summary(name, s, n_r84, n_r54, per_fold, passes):
        print(f"\n  [{name}] routed: R84={n_r84} / R54={n_r54} "
              f"(R84 rate={n_r84/(n_r84+n_r54):.1%})")
        print(f"    h7:          n={s['h7']['n']}  test={s['h7']['test']:.4f}  "
              f"baseline={s['h7']['baseline']:.4f}  Δ={s['h7']['delta']:+.4f}")
        print(f"    all:         n={s['all']['n']}  test={s['all']['test']:.4f}  "
              f"baseline={s['all']['baseline']:.4f}  Δ={s['all']['delta']:+.4f}")
        print(f"    same_artist: n={s['same_artist']['n']}  test={s['same_artist']['test']:.4f}  "
              f"baseline={s['same_artist']['baseline']:.4f}  Δ={s['same_artist']['delta']:+.4f}")
        print(f"    diff_artist: n={s['diff_artist']['n']}  test={s['diff_artist']['test']:.4f}  "
              f"baseline={s['diff_artist']['baseline']:.4f}  Δ={s['diff_artist']['delta']:+.4f}")
        rec = s['h7_recovery']
        print(f"    h7 recovered={rec['recovered']}  lost={rec['lost']}  net={rec['net']:+d}")
        print(f"    top1 churn /80: {s['top1_churn_per_80']:.2f}")
        print(f"    top20 overlap (vs R54): {s['overlap_mean']:.2f}/20")
        print(f"    per-fold h7 Δ: " +
              "  ".join(f"f{k}={per_fold[k]['delta']:+.4f}" for k in range(N_FOLDS)))
        print(f"    GATE: {'PASS' if passes else 'fail'}")

    report_summary("PRE-DECLARED", sum_pre, n_r84_pre, n_r54_pre, per_fold_pre, p_pre)

    # --- Threshold sweep ---
    print(f"\n{ts()} === THRESHOLD SWEEP ===", flush=True)
    sweep_results = {}
    best_sweep = (None, -1e9, None)
    for low_thr in SWEEP_LOW:
        for high_thr in SWEEP_HIGH:
            rows_s, n84, n54 = selective_rows(low_thr, high_thr)
            s = compute_metrics(rows_s, rows_r54_pure, rows_r54_pure)
            p, _ = gate_eval(s)
            key = f"low{low_thr}_high{high_thr}"
            sweep_results[key] = {
                "low_thr": low_thr, "high_thr": high_thr,
                "n_routed_r84": n84, "n_routed_r54": n54,
                "summary": s, "passes_gate": p,
            }
            print(f"  {key}: routed={n84}/{n54}  "
                  f"h7_Δ={s['h7']['delta']:+.4f}  "
                  f"all_Δ={s['all']['delta']:+.4f}  "
                  f"same_Δ={s['same_artist']['delta']:+.4f}  "
                  f"diff_Δ={s['diff_artist']['delta']:+.4f}  "
                  f"rec/lost={s['h7_recovery']['recovered']}/"
                  f"{s['h7_recovery']['lost']}  "
                  f"ovl={s['overlap_mean']:.2f}  "
                  f"GATE={'PASS' if p else 'fail'}")
            if s["h7"]["delta"] > best_sweep[1]:
                best_sweep = (key, s["h7"]["delta"], sweep_results[key])

    print(f"\n  Best sweep by h7_Δ: {best_sweep[0]}  h7_Δ={best_sweep[1]:+.4f}", flush=True)

    # --- Verdict ---
    any_pass = p_pre or any(v["passes_gate"] for v in sweep_results.values())
    verdict = "PROCEED_TO_BLIND_CANDIDATE_PLAN" if any_pass else "INVESTIGATE"
    print(f"\n{ts()} === VERDICT: {verdict} ===", flush=True)
    if any_pass:
        passing = []
        if p_pre:
            passing.append(("PRE-DECLARED", sum_pre["h7"]["delta"]))
        for k, v in sweep_results.items():
            if v["passes_gate"]:
                passing.append((k, v["summary"]["h7"]["delta"]))
        passing.sort(key=lambda x: -x[1])
        for name, h7d in passing:
            print(f"  ✓ {name}: h7_Δ={h7d:+.4f}")

    out = {
        "experiment": "R84c — selective deployment via R54 top-1 margin routing",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "predeclared": {
            "rule": f"use R84 if margin < {PREDECLARED['low_thr']} or >= {PREDECLARED['high_thr']}",
            "low_thr": PREDECLARED["low_thr"],
            "high_thr": PREDECLARED["high_thr"],
            "n_routed_r84": n_r84_pre,
            "n_routed_r54": n_r54_pre,
            "summary": sum_pre,
            "per_fold_h7": per_fold_pre,
            "passes_gate": p_pre,
        },
        "sweep": sweep_results,
        "best_sweep_key": best_sweep[0],
        "best_sweep_h7_delta": best_sweep[1],
        "phase1_pure_r84_baseline": {
            "h7_delta": pure_summary["h7"]["delta"],
            "same_artist_delta": pure_summary["same_artist"]["delta"],
        },
        "gate_definition": GATE,
        "r54_margin_distribution": {
            "p25": float(np.percentile(margins, 25)),
            "p50": float(np.percentile(margins, 50)),
            "p75": float(np.percentile(margins, 75)),
            "p90": float(np.percentile(margins, 90)),
            "max": float(margins.max()),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
