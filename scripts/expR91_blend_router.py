"""R91 — Conservative R84/R90 blend-router (OOF screening only).

R90 was a confirmed blind negative (blind nDCG -0.0185 vs R84c). Instead of
replacing R84 with R90, treat R90 as a weak perturbation of the R84 branch and
look for a *narrow, high-confidence* admission that beats R84c-selective on OOF
under a MUCH stricter gate than R90 used.

Mechanism (all on the shared R54-stacked pool; R54/R84/R90 differ only in the
last 3 retrieval columns, so scores blend at aligned pool indices):
  - Reuse cache/r84b/case_features.pkl (pool, feats_r54, feats_r84_only, gt_pos).
  - Add feats_r90 = feats_r54 with last-3 cols substituted from R90 5-fold OOF.
  - Per fold: train sibling LRs lr_r54 / lr_r84 / lr_r90 on train folds, score
    the eval fold (true OOF).
  - Baseline = R84c-selective (production): route R84 vs R54 by R54 top-1 margin
    (LOW=0.5, HIGH=2.0). Deltas/churn/overlap are measured vs THIS, not R54-pure.
  - Variants = conservative blend within the R84-routed cases only:
      blended = (1-a)*z(s_r84) + a*z(s_r90), per-case z-normalized, sweep a.
    a=0 reproduces R84c-selective; a=1 uses R90 on the R84 branch.

STRICT gate vs R84c-selective:
  h7_delta >= +0.008, per-fold h7_delta >= 0 in >=4/5 folds, recov > lost
  (strict), top-1 churn vs R84c <= 8/80, same_artist_delta >= -0.005,
  overlap vs R84c >= 16/20.

Output: exp/eval/expR91_blend_router.json. Screening only; no blind packaging.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_blind_submission import FEAT_R39_ALL, FEAT_ALL  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)
from scripts.expR84c_selective_deployment import (  # noqa: E402
    N_FOLDS, TOP_K, W0_STATS, FEAT_NAMES_R84_ONLY,
    LR_PARAMS, LR_NUM_BOOST_ROUND,
    ndcg_at_k, train_sibling_lr, compute_metrics,
)

import lightgbm as lgb  # type: ignore[reportMissingImports]

FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
R90_FOLD_OOF_DIRS = {
    0: REPO / "cache" / "r90" / "phase1_fold0_varA",
    1: REPO / "cache" / "r90" / "phase1_fold1_varA",
    2: REPO / "cache" / "r90" / "phase1_fold2_varA",
    3: REPO / "cache" / "r90" / "phase1_fold3_varA",
    4: REPO / "cache" / "r90" / "phase1_fold4_varA",
}
OUT_JSON = REPO / "exp" / "eval" / "expR91_blend_router.json"

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R90 = list(FEAT_R39_ALL) + ["ret_rank_inv", "ret_presence", "ret_cosine"]

# R84c production selective-routing thresholds (the deployed rule).
R84C_LOW, R84C_HIGH = 0.5, 2.0

# Conservative blend sweep (fraction of R90 mixed into the R84 branch).
ALPHA_SWEEP = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

# STRICT gate vs R84c-selective baseline.
GATE = {
    "h7_delta_ge": 0.008,
    "per_fold_nonneg_min": 4,        # h7 delta >= 0 in >= 4/5 folds
    "recov_strictly_gt_lost": True,
    "churn_le": 8.0,                 # top-1 churn vs R84c per 80
    "same_artist_delta_ge": -0.005,
    "overlap_ge": 16.0,
}


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def load_5fold_oof_r90_maps() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for fold, fold_dir in R90_FOLD_OOF_DIRS.items():
        p = fold_dir / "oof_r84_lists.json"
        if not p.exists():
            raise FileNotFoundError(f"R90 OOF fold-{fold} missing: {p}")
        with open(p) as f:
            raw = json.load(f)
        for ci_str, pairs in raw.items():
            ci = int(ci_str)
            tids = [t for t, _ in pairs]
            out[ci] = {
                "ranks": {t: r + 1 for r, t in enumerate(tids)},
                "scores": {t: float(s) for t, s in pairs},
            }
    return out


def add_r90_features(case_features: dict, r90_oof_maps: dict) -> None:
    """In place: cf['feats_r90'] = cf['feats_r54'] with last-3 cols from R90 OOF."""
    for ci, cf in case_features.items():
        pool = cf["pool"]
        ranks = r90_oof_maps[ci]["ranks"]
        scores = r90_oof_maps[ci]["scores"]
        f90 = np.asarray(cf["feats_r54"], dtype=np.float64).copy()
        for k_row, tid in enumerate(pool):
            f90[k_row, N_R39 + 0] = (1.0 / ranks[tid]) if tid in ranks else 0.0
            f90[k_row, N_R39 + 1] = 1.0 if tid in ranks else 0.0
            f90[k_row, N_R39 + 2] = scores.get(tid, 0.0)
        cf["feats_r90"] = f90


def znorm(s: np.ndarray) -> np.ndarray:
    sd = s.std()
    if sd < 1e-12:
        return s - s.mean()
    return (s - s.mean()) / sd


def make_row(i, score, cf, cases, case_fold, maps):
    order = np.argsort(-score, kind="mergesort")
    rank = -1
    if cf["gt_pos"] >= 0:
        p = np.where(order == cf["gt_pos"])[0]
        if len(p):
            rank = int(p[0]) + 1
    return {
        "case_idx": i, "fold": case_fold[i],
        "n_prior_music": int(cases[i]["n_prior_music"]),
        "same_artist": same_artist_case(cases[i], maps["track_artist"]),
        "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
        "in_top20": rank > 0 and rank <= TOP_K,
        "top20": [cf["pool"][int(j)] for j in order[:TOP_K]],
    }


def per_fold_h7(rows, baseline):
    out = {}
    for k in range(N_FOLDS):
        rt = [r for r in rows if r["fold"] == k and r["n_prior_music"] == 7]
        rb = [r for r in baseline if r["fold"] == k and r["n_prior_music"] == 7]
        t = float(np.mean([r["ndcg20"] for r in rt])) if rt else 0.0
        b = float(np.mean([r["ndcg20"] for r in rb])) if rb else 0.0
        out[k] = {"n": len(rt), "delta": t - b}
    return out


def strict_gate(s, per_fold, churn):
    h7_d = s["h7"]["delta"]
    sa_d = s["same_artist"]["delta"]
    recov, lost = s["h7_recovery"]["recovered"], s["h7_recovery"]["lost"]
    overlap = s["overlap_mean"]
    n_nonneg = sum(1 for k in per_fold if per_fold[k]["delta"] >= 0)
    checks = {
        "h7_delta_ge_p008": (h7_d >= GATE["h7_delta_ge"], h7_d),
        "per_fold_nonneg_ge_4": (n_nonneg >= GATE["per_fold_nonneg_min"], n_nonneg),
        "recov_gt_lost": (recov > lost, [recov, lost]),
        "churn_le_8": (churn <= GATE["churn_le"], churn),
        "same_artist_ge_n005": (sa_d >= GATE["same_artist_delta_ge"], sa_d),
        "overlap_ge_16": (overlap >= GATE["overlap_ge"], overlap),
    }
    return all(v[0] for v in checks.values()), checks


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R91 — conservative R84/R90 blend-router (OOF screening)")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + maps ...", flush=True)
    payload, _, _, _ = c3.load_payloads()
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

    if not FEAT_CACHE.exists():
        print(f"ERROR: feature cache missing at {FEAT_CACHE}.")
        sys.exit(1)
    print(f"{ts()} Loading case_features ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)...",
          flush=True)
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)
    print(f"  {len(case_features)} cases", flush=True)

    print(f"{ts()} Loading 5-fold R90 OOF + building feats_r90 ...", flush=True)
    r90_oof_maps = load_5fold_oof_r90_maps()
    assert len(r90_oof_maps) == n, f"R90 OOF missing {n - len(r90_oof_maps)} cases"
    add_r90_features(case_features, r90_oof_maps)

    # --- Per-fold OOF sibling LRs: R54, R84, R90 ---
    print(f"\n{ts()} === Train per-fold sibling LRs (R54/R84/R90) ===", flush=True)
    scores: dict[int, dict[str, np.ndarray]] = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t_lr = time.time()
        lr_r54 = train_sibling_lr(case_features, train_idx, "feats_r54", list(FEAT_ALL))
        lr_r84 = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                  FEAT_NAMES_R84_ONLY)
        lr_r90 = train_sibling_lr(case_features, train_idx, "feats_r90", FEAT_NAMES_R90)
        for i in eval_idx:
            cf = case_features[i]
            scores[i] = {
                "r54": lr_r54.predict(cf["feats_r54"]),
                "r84": lr_r84.predict(cf["feats_r84_only"]),
                "r90": lr_r90.predict(cf["feats_r90"]),
            }
        print(f"  fold {fold_k}: {time.time()-t_lr:.0f}s", flush=True)

    # --- R54 margins + R84c-selective routing decision per case ---
    case_margin, use_r84 = {}, {}
    for i, sd in scores.items():
        s_r54 = sd["r54"]
        if len(s_r54) >= 2:
            top2 = np.sort(s_r54)[::-1][:2]
            case_margin[i] = float(top2[0] - top2[1])
        else:
            case_margin[i] = 0.0
        use_r84[i] = (case_margin[i] < R84C_LOW) or (case_margin[i] >= R84C_HIGH)
    n_r84_branch = sum(use_r84.values())
    print(f"\n  R84c-selective routing: R84 branch={n_r84_branch}/{n} "
          f"({n_r84_branch/n:.1%}), R54 branch={n - n_r84_branch}")

    # --- Baseline rows: R54-pure (reference) and R84c-selective (gate baseline) ---
    rows_r54 = [make_row(i, scores[i]["r54"], case_features[i], cases, case_fold, maps)
                for i in scores]
    rows_r84c = [make_row(i, scores[i]["r84"] if use_r84[i] else scores[i]["r54"],
                          case_features[i], cases, case_fold, maps)
                 for i in scores]

    ref = compute_metrics(rows_r84c, rows_r54, rows_r54)
    print(f"  R84c-selective vs R54-pure: h7_Δ={ref['h7']['delta']:+.4f}  "
          f"all_Δ={ref['all']['delta']:+.4f}  "
          f"same_Δ={ref['same_artist']['delta']:+.4f}  "
          f"rec/lost={ref['h7_recovery']['recovered']}/{ref['h7_recovery']['lost']}")

    # --- Conservative blend variants (vs R84c-selective baseline) ---
    print(f"\n{ts()} === BLEND SWEEP (R90 into R84 branch only, vs R84c-selective) ===",
          flush=True)
    variants = {}
    best = (None, -1e9)
    for a in ALPHA_SWEEP:
        rows = []
        for i, sd in scores.items():
            if use_r84[i]:
                blended = (1.0 - a) * znorm(sd["r84"]) + a * znorm(sd["r90"])
            else:
                blended = sd["r54"]
            rows.append(make_row(i, blended, case_features[i], cases, case_fold, maps))
        s = compute_metrics(rows, rows_r84c, rows_r84c)   # baseline + overlap = R84c
        pf = per_fold_h7(rows, rows_r84c)
        churn = s["top1_churn_per_80"]
        passes, checks = strict_gate(s, pf, churn)
        key = f"alpha_{a:.2f}"
        variants[key] = {
            "alpha": a, "summary": s, "per_fold_h7": pf,
            "churn_vs_r84c": churn, "passes_strict_gate": passes,
            "gate_checks": {k: [bool(v[0]), v[1]] for k, v in checks.items()},
        }
        nn = sum(1 for k in pf if pf[k]["delta"] >= 0)
        print(f"  {key}: h7_Δ={s['h7']['delta']:+.4f}  all_Δ={s['all']['delta']:+.4f}  "
              f"same_Δ={s['same_artist']['delta']:+.4f}  "
              f"rec/lost={s['h7_recovery']['recovered']}/{s['h7_recovery']['lost']}  "
              f"churn={churn:.1f}/80  ovl={s['overlap_mean']:.2f}  "
              f"pf_nonneg={nn}/5  GATE={'PASS' if passes else 'fail'}")
        if s["h7"]["delta"] > best[1]:
            best = (key, s["h7"]["delta"])

    any_pass = any(v["passes_strict_gate"] for v in variants.values())
    verdict = "PROCEED_TO_BLIND_CANDIDATE" if any_pass else "ARCHIVE"
    print(f"\n{ts()} === VERDICT: {verdict} ===")
    print(f"  best by h7_Δ: {best[0]} ({best[1]:+.4f})")
    if any_pass:
        for k, v in variants.items():
            if v["passes_strict_gate"]:
                print(f"  PASS {k}: h7_Δ={v['summary']['h7']['delta']:+.4f}  "
                      f"churn={v['churn_vs_r84c']:.1f}/80")

    out = {
        "experiment": "R91 conservative R84/R90 blend-router (OOF screening)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "baseline": "R84c-selective (R54 margin route LOW=0.5/HIGH=2.0)",
        "r84c_vs_r54_reference": {
            "h7_delta": ref["h7"]["delta"],
            "same_artist_delta": ref["same_artist"]["delta"],
        },
        "n_r84_branch": n_r84_branch,
        "gate_definition": GATE,
        "alpha_sweep": ALPHA_SWEEP,
        "variants": variants,
        "best_variant_by_h7": best[0],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON.relative_to(REPO)}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
