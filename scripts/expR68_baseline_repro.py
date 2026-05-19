#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R68 Wave 0: baseline reproduction of frozen R54c LR on weighted_rrf@300.

Reuses cached intermediates and the production frozen LR. Adapted from
expR66_baseline_repro.py with two extensions:
  1. Per-case R54 reference stats (pickle): single-source top-300, presence,
     LR features (r54_rank_inv, r54_presence, r54_cosine), frozen LR top-20,
     and fold assignment.
  2. Aggregate JSON summarizing the R54 single-source pool_hit and feature
     distributions plus the fold map.

Outputs:
  exp/eval/expR68_baseline_repro.json
  exp/eval/expR68_r54_reference_stats.pkl
  exp/eval/expR68_r54_aggregate.json
  docs/r68_baseline_repro.md
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess  # noqa: S404
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR54_phase3_blind_submission import FEAT_ALL  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps,
    metric_block,
    same_artist_case,
    score_pool,
)
from scripts.expS2_lambdarank_grouped import grouped_session_folds  # noqa: E402

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
OUT_JSON = REPO / "exp" / "eval" / "expR68_baseline_repro.json"
OUT_MD = REPO / "docs" / "r68_baseline_repro.md"
OUT_STATS_PKL = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
OUT_AGG_JSON = REPO / "exp" / "eval" / "expR68_r54_aggregate.json"
REFERENCE_JSON = REPO / "exp" / "eval" / "expR59_c3_phase2_frozen_lr.json"
OOF_MANIFEST = REPO / "cache" / "r54" / "phase2_full" / "oof_manifest.json"

POOL_K = 300
TOP_K = 20
EPSILON = 0.0005

REFERENCE_METRICS = {
    "all_dev_ndcg20": 0.315875,
    "h7_ndcg20": 0.348378,
    "same_artist_ndcg20": 0.628214,
    "diff_artist_ndcg20": 0.142367,
    "pool_hit_all": 0.6220,
    "pool_hit_h7": 0.6130,
}


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError("git executable not found on PATH")
    return subprocess.check_output(  # noqa: S603
        [git_bin, "rev-parse", "HEAD"], cwd=str(REPO),
    ).decode().strip()


def derive_fold_assignment(cases: list[dict[str, Any]], manifest: dict[str, Any]) -> list[int]:
    """Recompute the same K-fold session split used by R54 phase2 training."""
    fold_split = manifest["fold_split"]
    seed = int(fold_split["seed"])
    k = int(fold_split["k"])
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed, k=k)

    # Sanity-check against manifest val_indices_sample.
    for fi in range(k):
        sample = manifest["folds"][str(fi)]["val_indices_sample"]
        fold_indices = folds[fi].tolist()
        if fold_indices[:len(sample)] != list(sample):
            raise RuntimeError(
                f"Fold {fi} mismatch vs manifest: derived={fold_indices[:5]} "
                f"manifest={sample}"
            )

    n_cases = len(cases)
    case_fold = [-1] * n_cases
    for fi, idx_arr in enumerate(folds):
        for idx in idx_arr.tolist():
            case_fold[int(idx)] = fi
    if any(f == -1 for f in case_fold):
        raise RuntimeError("Fold derivation missed some cases")
    return case_fold


def main() -> None:
    start = time.time()
    print("R68 Wave 0 baseline reproduction + R54 reference stats")
    print("=" * 70)

    with open(OOF_MANIFEST) as f:
        oof_manifest = json.load(f)

    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    ranker = lgb.Booster(model_file=str(LR_MODEL))
    if ranker.num_feature() != len(FEAT_ALL):
        raise RuntimeError(
            f"Frozen LR feature count mismatch: model={ranker.num_feature()} expected={len(FEAT_ALL)}"
        )

    case_index = c3.build_case_index(
        payload,
        r21_source,
        r54_source,
        r54_scores,
        als_factors,
        als_track_ids,
        als_to_idx,
    )

    case_fold = derive_fold_assignment(cases, oof_manifest)
    print(f"{ts()} Fold derivation PASS (matches manifest val_indices_sample)", flush=True)

    print(f"{ts()} Scoring RRF pool with frozen LR + collecting per-case R54 stats...", flush=True)
    case_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    rrf_pool_hit = 0
    h7_pool_hit = 0
    r54_single_hit_all = 0
    r54_single_hit_h7 = 0
    h7_n = 0
    feat_rank_inv: list[float] = []
    feat_presence: list[float] = []
    feat_cosine: list[float] = []
    fold_counts = [0] * 5
    score_start = time.time()

    for case_idx, case in enumerate(cases):
        source_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], case_idx,
        )
        rrf_pool = case_index["baseline_pools"][case_idx]
        is_h7 = int(case["n_prior_music"]) == 7
        if is_h7:
            h7_n += 1
        if case["gt"] in rrf_pool:
            rrf_pool_hit += 1
            if is_h7:
                h7_pool_hit += 1

        als_vec = case_index["als_session_vecs"][case_idx]
        r54_score_map = case_index["r54_scores"][case_idx]
        rrf_result = score_pool(
            ranker, rrf_pool, source_lists, case, maps,
            als_factors, als_to_idx, als_vec, track_pop, max_pop,
            track_album, r54_score_map, rrf_full_rank_map=None,
        )
        case_rows.append({
            "case_idx": case_idx,
            "session_id": case["session_id"],
            "n_prior_music": int(case["n_prior_music"]),
            "same_artist": same_artist_case(case, maps["track_artist"]),
            "rrf_gt_in_pool": bool(rrf_result["gt_in_pool"]),
            "rrf_gt_rank": int(rrf_result["gt_rank"]),
            "rrf_ndcg_at_20": float(rrf_result["ndcg_at_20"]),
            "rrf_ndcg_at_10": float(rrf_result["ndcg_at_10"]),
            "rrf_ndcg_at_7": float(rrf_result["ndcg_at_7"]),
        })

        # R54 reference stats
        r54_list = r54_source[case_idx][:POOL_K]
        gt = case["gt"]
        r54_in_top300 = gt in r54_list
        if r54_in_top300:
            r54_single_hit_all += 1
            if is_h7:
                r54_single_hit_h7 += 1
        # Per-case LR feature values for the GT candidate (if present in RRF pool).
        # These are what the frozen LR sees for the GT row; if GT not in pool, set 0.
        if gt and gt in rrf_pool:
            r54_rank_idx = r54_list.index(gt) + 1 if r54_in_top300 else 0
            rank_inv = 1.0 / r54_rank_idx if r54_rank_idx > 0 else 0.0
            presence = 1.0 if r54_in_top300 else 0.0
            cosine = float(r54_score_map.get(gt, 0.0))
        else:
            rank_inv = 0.0
            presence = 1.0 if r54_in_top300 else 0.0
            cosine = float(r54_score_map.get(gt, 0.0)) if gt else 0.0
        feat_rank_inv.append(rank_inv)
        feat_presence.append(presence)
        feat_cosine.append(cosine)

        fold_idx = case_fold[case_idx]
        fold_counts[fold_idx] += 1
        case_id = f"{case['session_id']}_t{int(case['turn_number'])}"
        stats_rows.append({
            "case_id": case_id,
            "case_idx": case_idx,
            "history_depth": int(case["n_prior_music"]),
            "gt_track_id": gt,
            "fold_idx": int(fold_idx),
            "r54_single_source_top300": list(r54_list),
            "r54_in_top300": bool(r54_in_top300),
            "r54_features": {
                "r54_rank_inv": float(rank_inv),
                "r54_presence": float(presence),
                "r54_cosine": float(cosine),
            },
            "lr_top20": list(rrf_result["top20"]),
        })

        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - score_start
            print(f"  scored {case_idx + 1}/{len(cases)} cases ({elapsed:.0f}s)", flush=True)

    rrf_metrics = metric_block("rrf", case_rows, cases, maps)
    pool_hit_all = rrf_pool_hit / len(cases)
    pool_hit_h7 = h7_pool_hit / max(h7_n, 1)

    reproduced = {
        "all_dev_ndcg20": rrf_metrics["all"]["ndcg_at_20"],
        "h7_ndcg20": rrf_metrics["h7"]["ndcg_at_20"],
        "same_artist_ndcg20": rrf_metrics["same_artist"]["ndcg_at_20"],
        "diff_artist_ndcg20": rrf_metrics["diff_artist"]["ndcg_at_20"],
        "pool_hit_all": pool_hit_all,
        "pool_hit_h7": pool_hit_h7,
    }

    metrics_out: dict[str, Any] = {}
    max_abs_delta = 0.0
    for key, ref in REFERENCE_METRICS.items():
        val = float(reproduced[key])
        delta = val - ref
        metrics_out[key] = {
            "reference": ref,
            "reproduced": val,
            "delta": delta,
        }
        max_abs_delta = max(max_abs_delta, abs(delta))

    verdict = "PASS" if max_abs_delta <= EPSILON else "FAIL"

    head = head_sha()
    report = {
        "experiment": "R68 W0 baseline reproduction",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start,
        "branch": "r68-large-scale-retrieval",
        "head_sha": head,
        "verdict": verdict,
        "epsilon_gate": EPSILON,
        "reference_file": str(REFERENCE_JSON.relative_to(REPO)),
        "metrics": metrics_out,
        "max_abs_delta": max_abs_delta,
        "n_cases": len(cases),
        "h7_n": h7_n,
        "fold_counts": fold_counts,
        "notes": (
            "Wave 0 reuses cached intermediates (R58 top50, R54 phase2 OOF lists, ALS, "
            "track pop, payload maps) and the production frozen LR model. Also dumps "
            "per-case R54 reference stats for R68 OOF comparison."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Per-case stats pickle
    with open(OUT_STATS_PKL, "wb") as f:
        pickle.dump(stats_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Aggregate JSON
    def _summ(vals: list[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
        }

    fold_assignments = {row["case_id"]: row["fold_idx"] for row in stats_rows}
    aggregate = {
        "experiment": "R68 W0: R54 reference stats",
        "created_at": datetime.now().isoformat(),
        "branch": "r68-large-scale-retrieval",
        "head_sha": head,
        "r54_single_source_pool_hit_h7": r54_single_hit_h7 / max(h7_n, 1),
        "r54_single_source_pool_hit_all": r54_single_hit_all / max(len(cases), 1),
        "fold_assignments": fold_assignments,
        "fold_counts": fold_counts,
        "n_cases": len(cases),
        "feature_summary": {
            "r54_rank_inv": _summ(feat_rank_inv),
            "r54_presence": _summ(feat_presence),
            "r54_cosine": _summ(feat_cosine),
        },
        "fold_split_source": {
            "manifest": str(OOF_MANIFEST.relative_to(REPO)),
            "function": oof_manifest["fold_split"]["function"],
            "seed": oof_manifest["fold_split"]["seed"],
            "k": oof_manifest["fold_split"]["k"],
        },
    }
    with open(OUT_AGG_JSON, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Markdown summary
    lines = [
        "# R68 Baseline Reproduction (Wave 0)",
        "",
        f"Created: {report['created_at']}",
        f"Branch: `{report['branch']}`",
        f"HEAD: `{report['head_sha']}`",
        f"Reference: `{report['reference_file']}`",
        f"Epsilon gate: {EPSILON}",
        "",
        f"## Verdict: **{verdict}**",
        f"max |delta| = {max_abs_delta:.6f}",
        "",
        "## Metrics",
        "",
        "| Metric | Reference | Reproduced | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "all_dev_ndcg20", "h7_ndcg20",
        "same_artist_ndcg20", "diff_artist_ndcg20",
        "pool_hit_all", "pool_hit_h7",
    ]:
        m = metrics_out[key]
        lines.append(
            f"| {key} | {m['reference']:.6f} | {m['reproduced']:.6f} | {m['delta']:+.6f} |"
        )
    lines.extend([
        "",
        "## R54 Reference Stats",
        "",
        f"- R54 single-source pool_hit @300 (all): "
        f"{aggregate['r54_single_source_pool_hit_all']:.4f} "
        f"({r54_single_hit_all}/{len(cases)})",
        f"- R54 single-source pool_hit @300 (h7): "
        f"{aggregate['r54_single_source_pool_hit_h7']:.4f} "
        f"({r54_single_hit_h7}/{h7_n})",
        "",
        "### Feature distributions (per-case, GT row of frozen LR input)",
        "",
        "| Feature | Mean | Std |",
        "|---|---:|---:|",
    ])
    for fname in ("r54_rank_inv", "r54_presence", "r54_cosine"):
        s = aggregate["feature_summary"][fname]
        lines.append(f"| {fname} | {s['mean']:.6f} | {s['std']:.6f} |")
    lines.extend([
        "",
        "## Fold Assignment",
        "",
        f"Source: `{aggregate['fold_split_source']['manifest']}` "
        f"(function={aggregate['fold_split_source']['function']}, "
        f"seed={aggregate['fold_split_source']['seed']}, "
        f"k={aggregate['fold_split_source']['k']})",
        "",
        "| Fold | Count |",
        "|---|---:|",
    ])
    for fi, n in enumerate(fold_counts):
        lines.append(f"| {fi} | {n} |")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- `{OUT_JSON.relative_to(REPO)}`",
        f"- `{OUT_STATS_PKL.relative_to(REPO)}`",
        f"- `{OUT_AGG_JSON.relative_to(REPO)}`",
        "",
        "## Notes",
        "",
        f"- Elapsed: {report['elapsed_s']:.1f}s",
        "- Wave 0 reuses cached intermediates and frozen production LR.",
        "- Fold split derived via `grouped_session_folds(sessions, seed=0, k=5)`; "
        "matches manifest `val_indices_sample` for all 5 folds.",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved stats pickle: {OUT_STATS_PKL}", flush=True)
    print(f"{ts()} Saved aggregate JSON: {OUT_AGG_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"{ts()} Verdict: {verdict}  max|delta|={max_abs_delta:.6f}", flush=True)
    for key in REFERENCE_METRICS:
        m = metrics_out[key]
        print(
            f"  {key}: ref={m['reference']:.6f} got={m['reproduced']:.6f} "
            f"delta={m['delta']:+.6f}",
            flush=True,
        )
    print(
        f"  R54 single-source pool_hit: all={aggregate['r54_single_source_pool_hit_all']:.4f} "
        f"h7={aggregate['r54_single_source_pool_hit_h7']:.4f}",
        flush=True,
    )
    print(f"  fold_counts: {fold_counts}", flush=True)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
