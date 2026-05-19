#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S311
"""R66 Wave 1 Phase 0: static RRF weight profile conversion sanity check.

For each of 8 hand-designed RRF weight profiles, builds weighted_rrf@300 pool,
scores it with the frozen R54c LR, and computes the Phase 0 kill gate.

Outputs:
  exp/eval/expR66_phase0_static_profile_conversion.json
  docs/r66_phase0_static_profile_result.md
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

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
OUT_JSON = REPO / "exp" / "eval" / "expR66_phase0_static_profile_conversion.json"
OUT_MD = REPO / "docs" / "r66_phase0_static_profile_result.md"

POOL_K = 300
TOP_K = 20
RRF_K = 20
BITWISE_EPS = 0.0005
CHURN_SAMPLE_SIZE = 80
CHURN_SEED = 0

REFERENCE_METRICS = {
    "all_dev_ndcg20": 0.3158755,
    "h7_ndcg20": 0.3483779,
    "same_artist_ndcg20": 0.6282143,
    "diff_artist_ndcg20": 0.1423674,
    "pool_hit_all": 0.6220,
    "pool_hit_h7": 0.6130,
}

PROFILES: dict[str, dict[str, float]] = {
    "P0": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0},
    "P1": {"A": 1.5, "B": 1.5, "C": 1.5, "D": 0.5, "F": 1.5, "ALS": 0.5, "R21": 0.7, "R54": 0.7},
    "P2": {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.3, "F": 0.5, "ALS": 1.5, "R21": 1.5, "R54": 1.5},
    "P3": {"A": 0.7, "B": 0.7, "C": 0.7, "D": 0.3, "F": 0.7, "ALS": 0.7, "R21": 0.7, "R54": 2.0},
    "P4": {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.3, "F": 0.5, "ALS": 0.5, "R21": 1.5, "R54": 1.5},
    "P5": {"A": 1.5, "B": 1.5, "C": 1.5, "D": 0.5, "F": 1.5, "ALS": 0.3, "R21": 0.3, "R54": 0.3},
    "P6": {"A": 0.5, "B": 0.5, "C": 2.0, "D": 0.3, "F": 0.5, "ALS": 0.5, "R21": 0.7, "R54": 2.0},
    "P7": {"A": 0.5, "B": 0.5, "C": 0.5, "D": 0.3, "F": 0.5, "ALS": 2.0, "R21": 0.7, "R54": 2.0},
}

PROFILE_LABELS = {
    "P0": "R54c baseline",
    "P1": "text-heavy",
    "P2": "collaborative-heavy",
    "P3": "R54-heavy",
    "P4": "R21/R54 pair",
    "P5": "BM25-only",
    "P6": "C+R54 dominant",
    "P7": "ALS+R54 dominant",
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


def weighted_rrf_pool(source_lists: dict[str, list[str]], weights: dict[str, float], topk: int) -> list[str]:
    scores: dict[str, float] = {}
    for source_name in c3.SOURCE_NAMES:
        weight = weights.get(source_name, 0.0)
        if weight == 0.0:
            continue
        ranked = source_lists[source_name]
        for rank, track_id in enumerate(ranked, start=1):
            scores[track_id] = scores.get(track_id, 0.0) + weight / (RRF_K + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]


def evaluate_profile(
    profile_id: str,
    weights: dict[str, float],
    cases: list[dict[str, Any]],
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    ranker: lgb.Booster,
    maps: dict[str, Any],
    als_factors: Any,
    als_to_idx: dict[str, int],
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
) -> dict[str, Any]:
    print(f"{ts()} Profile {profile_id} ({PROFILE_LABELS[profile_id]}) — building pools and scoring...", flush=True)
    t0 = time.time()
    case_rows: list[dict[str, Any]] = []
    top20_per_case: list[list[str]] = []
    pools_per_case: list[list[str]] = []
    rrf_pool_hit = 0
    h7_pool_hit = 0
    for case_idx, case in enumerate(cases):
        source_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], case_idx,
        )
        pool = weighted_rrf_pool(source_lists, weights, POOL_K)
        pools_per_case.append(pool)
        if case["gt"] in pool:
            rrf_pool_hit += 1
            if int(case["n_prior_music"]) == 7:
                h7_pool_hit += 1
        als_vec = case_index["als_session_vecs"][case_idx]
        r54_score_map = case_index["r54_scores"][case_idx]
        result = score_pool(
            ranker, pool, source_lists, case, maps,
            als_factors, als_to_idx, als_vec, track_pop, max_pop,
            track_album, r54_score_map, rrf_full_rank_map=None,
        )
        top20_per_case.append(result["top20"])
        case_rows.append({
            "case_idx": case_idx,
            "session_id": case["session_id"],
            "n_prior_music": int(case["n_prior_music"]),
            "same_artist": same_artist_case(case, maps["track_artist"]),
            "rrf_gt_in_pool": bool(result["gt_in_pool"]),
            "rrf_gt_rank": int(result["gt_rank"]),
            "rrf_ndcg_at_20": float(result["ndcg_at_20"]),
            "rrf_ndcg_at_10": float(result["ndcg_at_10"]),
            "rrf_ndcg_at_7": float(result["ndcg_at_7"]),
        })
        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {profile_id}: scored {case_idx + 1}/{len(cases)} ({elapsed:.0f}s)", flush=True)

    metrics = metric_block("rrf", case_rows, cases, maps)
    h7_n = sum(1 for c in cases if int(c["n_prior_music"]) == 7)
    pool_hit_all = rrf_pool_hit / len(cases)
    pool_hit_h7 = h7_pool_hit / max(h7_n, 1)
    elapsed = time.time() - t0
    print(
        f"{ts()} Profile {profile_id} elapsed={elapsed:.1f}s  "
        f"pool_hit_h7={pool_hit_h7:.4f}  h7_ndcg20={metrics['h7']['ndcg_at_20']:.6f}",
        flush=True,
    )

    return {
        "metrics": metrics,
        "pool_hit_all": pool_hit_all,
        "pool_hit_h7": pool_hit_h7,
        "case_rows": case_rows,
        "top20_per_case": top20_per_case,
        "pools_per_case": pools_per_case,
        "elapsed_s": elapsed,
    }


def main() -> None:
    start = time.time()
    print("R66 Wave 1 Phase 0: static profile conversion sanity check")
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
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    # Evaluate all profiles
    profile_results: dict[str, dict[str, Any]] = {}
    for profile_id in PROFILES:
        profile_results[profile_id] = evaluate_profile(
            profile_id, PROFILES[profile_id], cases, payload, r21_source, r54_source,
            case_index, ranker, maps, als_factors, als_to_idx,
            track_pop, max_pop, track_album,
        )

    # --- Baseline (P0) bitwise sanity check ---
    p0 = profile_results["P0"]
    p0_repro = {
        "all_dev_ndcg20": p0["metrics"]["all"]["ndcg_at_20"],
        "h7_ndcg20": p0["metrics"]["h7"]["ndcg_at_20"],
        "same_artist_ndcg20": p0["metrics"]["same_artist"]["ndcg_at_20"],
        "diff_artist_ndcg20": p0["metrics"]["diff_artist"]["ndcg_at_20"],
        "pool_hit_all": p0["pool_hit_all"],
        "pool_hit_h7": p0["pool_hit_h7"],
    }
    p0_check: dict[str, Any] = {}
    p0_max_abs_delta = 0.0
    for key, ref in REFERENCE_METRICS.items():
        val = float(p0_repro[key])
        delta = val - ref
        p0_check[key] = {"reference": ref, "reproduced": val, "delta": delta}
        p0_max_abs_delta = max(p0_max_abs_delta, abs(delta))
    h7_delta_abs = abs(p0_check["h7_ndcg20"]["delta"])
    p0_check["max_abs_delta"] = p0_max_abs_delta
    p0_check["verdict"] = "PASS" if h7_delta_abs <= BITWISE_EPS else "FAIL"
    if p0_check["verdict"] != "PASS":
        print(
            f"{ts()} WARNING: P0 sanity FAILED. h7 |delta|={h7_delta_abs:.6f} > {BITWISE_EPS}",
            flush=True,
        )

    # --- Compute baseline top-20 (P0) for churn / recovered / lost stats ---
    p0_top20 = profile_results["P0"]["top20_per_case"]
    p0_pool_hit_h7 = profile_results["P0"]["pool_hit_h7"]
    p0_h7_ndcg20 = profile_results["P0"]["metrics"]["h7"]["ndcg_at_20"]
    p0_same_artist_ndcg20 = profile_results["P0"]["metrics"]["same_artist"]["ndcg_at_20"]

    # Churn sample
    rng = np.random.default_rng(CHURN_SEED)
    churn_indices = [int(i) for i in rng.choice(len(cases), size=CHURN_SAMPLE_SIZE, replace=False)]

    # --- Per-profile gate ---
    profiles_out: dict[str, Any] = {}
    passers: list[str] = []
    for profile_id, result in profile_results.items():
        m = result["metrics"]
        deltas = {
            "pool_hit_all": result["pool_hit_all"] - profile_results["P0"]["pool_hit_all"],
            "pool_hit_h7": result["pool_hit_h7"] - p0_pool_hit_h7,
            "ndcg20_all": m["all"]["ndcg_at_20"] - profile_results["P0"]["metrics"]["all"]["ndcg_at_20"],
            "ndcg20_h7": m["h7"]["ndcg_at_20"] - p0_h7_ndcg20,
            "ndcg20_same_artist": m["same_artist"]["ndcg_at_20"] - p0_same_artist_ndcg20,
            "ndcg20_diff_artist": m["diff_artist"]["ndcg_at_20"]
                - profile_results["P0"]["metrics"]["diff_artist"]["ndcg_at_20"],
        }

        # Recovered / lost h7
        recovered_h7 = 0
        lost_h7 = 0
        for case_idx, case in enumerate(cases):
            if int(case["n_prior_music"]) != 7:
                continue
            gt = case["gt"]
            in_p0 = gt in p0_top20[case_idx]
            in_pi = gt in result["top20_per_case"][case_idx]
            if in_pi and not in_p0:
                recovered_h7 += 1
            elif in_p0 and not in_pi:
                lost_h7 += 1
        net_h7 = recovered_h7 - lost_h7

        # Top-1 churn over sample
        top1_diff = 0
        for case_idx in churn_indices:
            p0_top1 = p0_top20[case_idx][0] if p0_top20[case_idx] else ""
            pi_top1 = result["top20_per_case"][case_idx][0] if result["top20_per_case"][case_idx] else ""
            if p0_top1 != pi_top1:
                top1_diff += 1
        top1_churn_per_80 = top1_diff  # already over 80-sample

        # Top-20 overlap mean over all cases
        overlaps = []
        for case_idx in range(len(cases)):
            p0_set = set(p0_top20[case_idx])
            pi_set = set(result["top20_per_case"][case_idx])
            overlaps.append(len(p0_set & pi_set))
        top20_overlap_mean = float(sum(overlaps) / max(len(overlaps), 1))

        # Gates (only meaningful for non-P0)
        if profile_id == "P0":
            gate_pool = None
            gate_h7 = None
            gate_same = None
            gate_rec = None
            passes = None
        else:
            gate_pool = bool(deltas["pool_hit_h7"] >= 0.010 - 1e-9)
            gate_h7 = bool(deltas["ndcg20_h7"] >= -1e-12)
            gate_same = bool(deltas["ndcg20_same_artist"] >= -0.002 - 1e-9)
            gate_rec = bool(recovered_h7 > lost_h7)
            passes = bool(gate_pool and gate_h7 and gate_same and gate_rec)
            if passes:
                passers.append(profile_id)

        profiles_out[profile_id] = {
            "label": PROFILE_LABELS[profile_id],
            "weights": PROFILES[profile_id],
            "pool_hit_all": float(result["pool_hit_all"]),
            "pool_hit_h7": float(result["pool_hit_h7"]),
            "ndcg20_all": float(m["all"]["ndcg_at_20"]),
            "ndcg20_h7": float(m["h7"]["ndcg_at_20"]),
            "ndcg20_same_artist": float(m["same_artist"]["ndcg_at_20"]),
            "ndcg20_diff_artist": float(m["diff_artist"]["ndcg_at_20"]),
            "recovered_h7": int(recovered_h7),
            "lost_h7": int(lost_h7),
            "net_h7": int(net_h7),
            "top1_churn_per_80": int(top1_churn_per_80),
            "top20_overlap_mean": top20_overlap_mean,
            "deltas_vs_P0": {k: float(v) for k, v in deltas.items()},
            "gate_pool_hit_h7_lift_ge_001": gate_pool,
            "gate_h7_ndcg_delta_ge_0": gate_h7,
            "gate_same_artist_delta_ge_neg002": gate_same,
            "gate_recovered_gt_lost": gate_rec,
            "passes_phase0_gate": passes,
            "elapsed_s": float(result["elapsed_s"]),
        }

    verdict = "PROCEED" if passers else "ARCHIVE_PHASE_0"

    report = {
        "experiment": "R66 Phase 0 static profile conversion",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start,
        "head_sha_before": "cf1684c",
        "head_sha_after": head_sha(),
        "verdict": verdict,
        "passers": passers,
        "baseline_metrics_P0_check": p0_check,
        "kill_gate_definition": {
            "pool_hit_h7_lift_ge": 0.010,
            "h7_ndcg_delta_ge": 0.0,
            "same_artist_delta_ge": -0.002,
            "recovered_gt_lost": True,
        },
        "churn_sample": {"size": CHURN_SAMPLE_SIZE, "seed": CHURN_SEED},
        "profiles": profiles_out,
        "notes": (
            "Phase 0 sweeps 8 hand-designed RRF weight profiles, each scored by frozen "
            "R54c LR on the resulting weighted_rrf@300 pool. Pass requires all 4 gates: "
            "pool_hit_h7 lift >=+0.010, h7 nDCG delta >=0, same-artist delta >=-0.002, "
            "and recovered_h7 > lost_h7 vs P0 baseline."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # --- Markdown summary ---
    lines = [
        "# R66 Phase 0 — Static Profile Conversion Result",
        "",
        f"Created: {report['created_at']}",
        f"HEAD before: `{report['head_sha_before']}`",
        f"HEAD after: `{report['head_sha_after']}`",
        f"Elapsed: {report['elapsed_s']:.1f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"Passers: `{passers if passers else 'none'}`",
        "",
        "## P0 Bitwise Sanity Check (vs reference)",
        "",
        "| Metric | Reference | Reproduced | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "all_dev_ndcg20", "h7_ndcg20",
        "same_artist_ndcg20", "diff_artist_ndcg20",
        "pool_hit_all", "pool_hit_h7",
    ]:
        m = p0_check[key]
        lines.append(
            f"| {key} | {m['reference']:.6f} | {m['reproduced']:.6f} | {m['delta']:+.6f} |"
        )
    lines.extend([
        "",
        f"P0 sanity verdict: **{p0_check['verdict']}**  "
        f"(h7 |delta|={abs(p0_check['h7_ndcg20']['delta']):.6f}, gate eps={BITWISE_EPS})",
        "",
        "## Phase 0 Kill Gate (per profile, vs P0)",
        "",
        "All 4 conditions must hold (non-P0):",
        "1. `pool_hit_h7` lift >= +0.010",
        "2. `h7_ndcg20` delta >= 0",
        "3. `same_artist_ndcg20` delta >= -0.002",
        "4. `recovered_h7 > lost_h7`",
        "",
        "## Profile Weights",
        "",
        "| Profile | Label | A | B | C | D | F | ALS | R21 | R54 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for pid in PROFILES:
        w = PROFILES[pid]
        lines.append(
            f"| {pid} | {PROFILE_LABELS[pid]} | {w['A']} | {w['B']} | {w['C']} | "
            f"{w['D']} | {w['F']} | {w['ALS']} | {w['R21']} | {w['R54']} |"
        )

    lines.extend([
        "",
        "## Profile Metrics",
        "",
        "| Profile | pool_hit_h7 | Δpool_hit_h7 | h7_ndcg20 | Δh7_ndcg | Δsame_artist | rec | lost | net | churn/80 | overlap@20 | passes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for pid in PROFILES:
        p = profiles_out[pid]
        passes_str = "—" if p["passes_phase0_gate"] is None else ("✓" if p["passes_phase0_gate"] else "✗")
        lines.append(
            f"| {pid} | {p['pool_hit_h7']:.4f} | {p['deltas_vs_P0']['pool_hit_h7']:+.4f} | "
            f"{p['ndcg20_h7']:.6f} | {p['deltas_vs_P0']['ndcg20_h7']:+.6f} | "
            f"{p['deltas_vs_P0']['ndcg20_same_artist']:+.6f} | "
            f"{p['recovered_h7']} | {p['lost_h7']} | {p['net_h7']:+d} | "
            f"{p['top1_churn_per_80']} | {p['top20_overlap_mean']:.2f} | {passes_str} |"
        )

    lines.extend([
        "",
        "## Per-Profile Gate Breakdown",
        "",
        "| Profile | pool_hit_lift≥+0.010 | h7_ndcg_Δ≥0 | same_artist_Δ≥-0.002 | rec>lost | passes |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for pid in PROFILES:
        p = profiles_out[pid]

        def gate_str(val: Any) -> str:
            if val is None:
                return "—"
            return "✓" if val else "✗"

        lines.append(
            f"| {pid} | {gate_str(p['gate_pool_hit_h7_lift_ge_001'])} | "
            f"{gate_str(p['gate_h7_ndcg_delta_ge_0'])} | "
            f"{gate_str(p['gate_same_artist_delta_ge_neg002'])} | "
            f"{gate_str(p['gate_recovered_gt_lost'])} | "
            f"{gate_str(p['passes_phase0_gate'])} |"
        )

    if passers:
        conclusion = (
            f"{', '.join(passers)} cleared all 4 Phase 0 conditions; "
            "Wave 2 may proceed with these profiles."
        )
    else:
        conclusion = (
            "No profile cleared all 4 Phase 0 conditions; sprint archives at Phase 0. "
            "Static RRF re-weighting (within the menu) does not unlock learned routing."
        )

    lines.extend([
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Notes",
        "",
        "- Frozen LR: `cache/r54_phase3_lr_model.txt` (read-only)",
        f"- Churn sample: {CHURN_SAMPLE_SIZE} cases, seed={CHURN_SEED}",
        f"- Pool depth: {POOL_K}; top-K: {TOP_K}",
        "- Per-profile elapsed:",
    ])
    for pid in PROFILES:
        lines.append(f"  - {pid}: {profiles_out[pid]['elapsed_s']:.1f}s")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"{ts()} verdict: {verdict}  passers: {passers}", flush=True)
    print(f"Total elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
