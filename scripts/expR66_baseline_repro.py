#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R66 Wave 0: baseline reproduction of frozen R54c LR on weighted_rrf@300.

Reuses cached intermediates and the production frozen LR. Skips admission LR
training and learned-pool scoring entirely (we only need the RRF side).

Outputs:
  exp/eval/expR66_baseline_repro.json
  docs/r66_baseline_repro.md
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess  # noqa: S404
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]

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

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
OUT_JSON = REPO / "exp" / "eval" / "expR66_baseline_repro.json"
OUT_MD = REPO / "docs" / "r66_baseline_repro.md"
REFERENCE_JSON = REPO / "exp" / "eval" / "expR59_c3_phase2_frozen_lr.json"

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


def main() -> None:
    start = time.time()
    print("R66 Wave 0 baseline reproduction")
    print("=" * 70)
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

    print(f"{ts()} Scoring RRF pool with frozen LR...", flush=True)
    case_rows: list[dict[str, Any]] = []
    rrf_pool_hit = 0
    h7_pool_hit = 0
    score_start = time.time()
    for case_idx, case in enumerate(cases):
        source_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], case_idx,
        )
        rrf_pool = case_index["baseline_pools"][case_idx]
        if case["gt"] in rrf_pool:
            rrf_pool_hit += 1
            if int(case["n_prior_music"]) == 7:
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
        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - score_start
            print(f"  scored {case_idx + 1}/{len(cases)} cases ({elapsed:.0f}s)", flush=True)

    rrf_metrics = metric_block("rrf", case_rows, cases, maps)
    h7_n = sum(1 for c in cases if int(c["n_prior_music"]) == 7)
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

    report = {
        "experiment": "R66 baseline reproduction",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start,
        "branch": "r66-learned-depth-source-router",
        "head_sha": head_sha(),
        "verdict": verdict,
        "epsilon_gate": EPSILON,
        "reference_file": str(REFERENCE_JSON.relative_to(REPO)),
        "metrics": metrics_out,
        "max_abs_delta": max_abs_delta,
        "weighted_rrf_signature_supports_per_source_weights": True,
        "weighted_rrf_call_signature_note": (
            "weighted_rrf(sources: dict[str, list[str]], weights: dict[str, float], "
            "topk: int, k: int = 20) -> list[str] in scripts/expF1_cfbpr_retrieval.py:158"
        ),
        "notes": (
            "Wave 0 reuses cached intermediates (R58 top50, R54 phase2 OOF lists, ALS, "
            "track pop, payload maps) and the production frozen LR model. Admission LR "
            "training and learned-pool scoring are skipped."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    lines = [
        "# R66 Baseline Reproduction (Wave 0)",
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
        "## weighted_rrf Signature",
        "",
        f"Supports per-source weights: {report['weighted_rrf_signature_supports_per_source_weights']}",
        "",
        f"`{report['weighted_rrf_call_signature_note']}`",
        "",
        "## Notes",
        "",
        f"- Elapsed: {report['elapsed_s']:.1f}s",
        "- Wave 0 reuses cached intermediates and frozen production LR.",
        "- Admission LR training and learned-pool scoring skipped (RRF baseline only).",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"{ts()} Verdict: {verdict}  max|delta|={max_abs_delta:.6f}", flush=True)
    for key in REFERENCE_METRICS:
        m = metrics_out[key]
        print(f"  {key}: ref={m['reference']:.6f} got={m['reproduced']:.6f} delta={m['delta']:+.6f}", flush=True)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
