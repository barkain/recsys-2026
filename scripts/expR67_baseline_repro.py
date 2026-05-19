#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R67 Wave 0: baseline reproduction of frozen R54c LR on weighted_rrf@300.

Extends R66 baseline reproduction by also dumping per-case LR top-30 candidate
records with a metadata join. The top-30 artifact is the canonical foundation
for Phase 0/1 R67 LLM-rerank candidate packets.

Outputs:
  exp/eval/expR67_baseline_repro.json
  exp/eval/expR67_top30_candidates.pkl
  docs/r67_baseline_repro.md
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
    POOL_K as REF_POOL_K,
    TOP_K as REF_TOP_K,
    featurize_for_frozen_lr,
    load_supporting_maps,
    metric_block,
    ndcg_from_rank,
    same_artist_case,
)

LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
OUT_JSON = REPO / "exp" / "eval" / "expR67_baseline_repro.json"
OUT_TOP30 = REPO / "exp" / "eval" / "expR67_top30_candidates.pkl"
OUT_MD = REPO / "docs" / "r67_baseline_repro.md"
REFERENCE_JSON = REPO / "exp" / "eval" / "expR59_c3_phase2_frozen_lr.json"
METADATA_JSON = REPO / "cache" / "metadata" / "track_metadata_all_tracks.json"

POOL_K = REF_POOL_K
TOP_K = REF_TOP_K
TOP30_K = 30
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


def first_str(value: Any) -> str:
    """Return first string from a list-or-str value, normalized."""
    if value is None:
        return ""
    if isinstance(value, list):
        for v in value:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def parse_release_year(value: Any) -> int | None:
    s = first_str(value)
    if not s or len(s) < 4:
        return None
    try:
        return int(s[:4])
    except ValueError:
        return None


def build_metadata_lookup(track_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Load track metadata JSON once; return lookup dict and coverage counts."""
    with open(METADATA_JSON) as f:
        raw = json.load(f)
    coverage = {"title": 0, "artist": 0, "album": 0, "tags": 0, "release_year": 0, "n_total": 0}
    lookup: dict[str, dict[str, Any]] = {}
    for tid in track_ids:
        coverage["n_total"] += 1
        rec = raw.get(tid)
        if rec is None:
            lookup[tid] = {
                "title": "", "artist": "", "album": "",
                "tags": [], "release_year": None,
            }
            continue
        title = first_str(rec.get("track_name"))
        artist = first_str(rec.get("artist_name"))
        album = first_str(rec.get("album_name"))
        tags_raw = rec.get("tag_list") or []
        tags = [t for t in tags_raw if isinstance(t, str)][:5]
        rel_year = parse_release_year(rec.get("release_date"))
        lookup[tid] = {
            "title": title,
            "artist": artist,
            "album": album,
            "tags": tags,
            "release_year": rel_year,
        }
        if title:
            coverage["title"] += 1
        if artist:
            coverage["artist"] += 1
        if album:
            coverage["album"] += 1
        if tags:
            coverage["tags"] += 1
        if rel_year is not None:
            coverage["release_year"] += 1
    return lookup, coverage


def score_pool_with_top30(
    ranker: lgb.Booster,
    pool: list[str],
    source_lists: dict[str, list[str]],
    case: dict[str, Any],
    maps: dict[str, Any],
    als_factors: np.ndarray,
    als_to_idx: dict[str, int],
    als_vec: np.ndarray | None,
    track_pop: dict[str, int],
    max_pop: int,
    track_album: dict[str, str],
    r54_score_map: dict[str, float],
) -> dict[str, Any]:
    """Mirrors score_pool but returns LR top-30 with source ranks + lr_scores."""
    r21_rank_map = {tid: rank + 1 for rank, tid in enumerate(source_lists["R21"][:POOL_K])}
    r54_rank_map = {tid: rank + 1 for rank, tid in enumerate(source_lists["R54"][:POOL_K])}
    feats = featurize_for_frozen_lr(
        pool, source_lists, r21_rank_map, r54_rank_map, r54_score_map,
        case, maps, als_factors, als_to_idx, als_vec, track_pop, max_pop,
        track_album, rrf_full_rank_map=None,
    )
    scores = ranker.predict(feats)
    order = np.argsort(-scores, kind="mergesort")
    top20 = [pool[int(idx)] for idx in order[:TOP_K]]
    top30_idx = order[:TOP30_K]
    top30_tracks = [pool[int(i)] for i in top30_idx]
    top30_scores = [float(scores[int(i)]) for i in top30_idx]

    gt = case["gt"]
    gt_rank = -1
    gt_in_top20_rank: int | None = None
    if gt in pool:
        gt_pool_idx = pool.index(gt)
        pos = np.where(order == gt_pool_idx)[0]
        if len(pos):
            gt_rank = int(pos[0]) + 1
            if gt_rank <= 20:
                gt_in_top20_rank = gt_rank

    # Per-source rank maps for the top-30 union
    source_rank_maps: dict[str, dict[str, int]] = {}
    for sname, slist in source_lists.items():
        rmap: dict[str, int] = {}
        for r, t in enumerate(slist[:POOL_K]):
            if t not in rmap:
                rmap[t] = r + 1
        source_rank_maps[sname] = rmap

    top30_records: list[dict[str, Any]] = []
    for rank_i, (tid, sc) in enumerate(zip(top30_tracks, top30_scores), start=1):
        src_ranks: dict[str, Any] = {}
        for sname, rmap in source_rank_maps.items():
            src_ranks[sname] = rmap.get(tid)  # None if not present
        top30_records.append({
            "rank": rank_i,
            "track_id": tid,
            "lr_score": sc,
            "source_ranks": src_ranks,
        })

    return {
        "top20": top20,
        "top1": top20[0] if top20 else "",
        "gt_rank": gt_rank,
        "ndcg_at_20": ndcg_from_rank(gt_rank, 20),
        "ndcg_at_10": ndcg_from_rank(gt_rank, 10),
        "ndcg_at_7": ndcg_from_rank(gt_rank, 7),
        "gt_in_pool": gt in pool,
        "top30_records": top30_records,
        "lr_rank1_score": top30_scores[0] if top30_scores else 0.0,
        "lr_rank5_score": top30_scores[4] if len(top30_scores) >= 5 else 0.0,
        "gt_in_top20_rank": gt_in_top20_rank,
    }


def main() -> None:
    start = time.time()
    print("R67 Wave 0 baseline reproduction + LR top-30 extraction")
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

    print(f"{ts()} Scoring RRF pool with frozen LR + dumping top-30...", flush=True)
    case_rows: list[dict[str, Any]] = []
    top30_dump: list[dict[str, Any]] = []
    union_tracks: set[str] = set()
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
        rrf_result = score_pool_with_top30(
            ranker, rrf_pool, source_lists, case, maps,
            als_factors, als_to_idx, als_vec, track_pop, max_pop,
            track_album, r54_score_map,
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
        # Top-30 dump (metadata join postponed to a second pass)
        for rec in rrf_result["top30_records"]:
            union_tracks.add(rec["track_id"])
        gt_tid = case.get("gt")
        gt_artist = maps["track_artist"].get(gt_tid, "") if gt_tid else ""
        top30_dump.append({
            "case_id": case["session_id"],
            "case_idx": case_idx,
            "history_depth": int(case["n_prior_music"]),
            "gt_track_id": gt_tid if gt_tid else None,
            "gt_artist": gt_artist or None,
            "lr_top30_raw": rrf_result["top30_records"],   # filled with metadata below
            "lr_rank1_score": rrf_result["lr_rank1_score"],
            "lr_rank5_score": rrf_result["lr_rank5_score"],
            "gt_in_top20_rank": rrf_result["gt_in_top20_rank"],
            "gt_rank_in_pool": int(rrf_result["gt_rank"]),
        })
        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - score_start
            print(f"  scored {case_idx + 1}/{len(cases)} cases ({elapsed:.0f}s)", flush=True)

    # Compute baseline metrics
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
        metrics_out[key] = {"reference": ref, "reproduced": val, "delta": delta}
        max_abs_delta = max(max_abs_delta, abs(delta))

    verdict = "PASS" if max_abs_delta <= EPSILON else "FAIL"

    # Metadata join for the top-30 union
    print(f"{ts()} Building metadata lookup for {len(union_tracks)} unique tracks...", flush=True)
    meta_lookup, coverage = build_metadata_lookup(union_tracks)

    # Attach metadata into the dump
    for rec in top30_dump:
        enriched: list[dict[str, Any]] = []
        for c_rec in rec["lr_top30_raw"]:
            tid = c_rec["track_id"]
            meta = meta_lookup.get(tid, {
                "title": "", "artist": "", "album": "", "tags": [], "release_year": None,
            })
            enriched.append({
                "rank": c_rec["rank"],
                "track_id": tid,
                "lr_score": c_rec["lr_score"],
                "source_ranks": c_rec["source_ranks"],
                "title": meta["title"],
                "artist": meta["artist"],
                "album": meta["album"],
                "tags": meta["tags"],
                "release_year": meta["release_year"],
            })
        rec["lr_top30"] = enriched
        del rec["lr_top30_raw"]

    # Persist top-30 pickle
    OUT_TOP30.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TOP30, "wb") as f:
        pickle.dump({
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "n_cases": len(top30_dump),
            "top_k": TOP30_K,
            "records": top30_dump,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Build coverage rates
    total = max(coverage["n_total"], 1)
    coverage_pct = {
        "title": coverage["title"] / total,
        "artist": coverage["artist"] / total,
        "album": coverage["album"] / total,
        "tags": coverage["tags"] / total,
        "release_year": coverage["release_year"] / total,
        "n_unique_candidate_tracks": coverage["n_total"],
    }

    report = {
        "experiment": "R67 baseline reproduction",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start,
        "branch": "r67-llm-semantic-rerank",
        "head_sha": head_sha(),
        "verdict": verdict,
        "epsilon_gate": EPSILON,
        "reference_file": str(REFERENCE_JSON.relative_to(REPO)),
        "metrics": metrics_out,
        "max_abs_delta": max_abs_delta,
        "r63c_response_only_note": (
            "R63c-repair is response-side only; retrieval/LR baseline is bitwise R54c. "
            "The R67 LLM reranker operates on the same R54c LR top-30."
        ),
        "top30_extraction": {
            "n_cases": len(top30_dump),
            "top_k": TOP30_K,
            "unique_candidate_tracks": coverage["n_total"],
            "metadata_coverage_pct": {
                k: float(v) for k, v in coverage_pct.items() if k != "n_unique_candidate_tracks"
            },
            "output_pkl": str(OUT_TOP30.relative_to(REPO)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown
    lines = [
        "# R67 Baseline Reproduction (Wave 0)",
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
        "## R63c response-only clarification",
        "",
        "R63c-repair is response-side only. Retrieval/LR baseline is bitwise R54c. "
        "The R67 LLM reranker operates on the same R54c LR top-30.",
        "",
        "## Top-30 extraction",
        "",
        f"- n_cases: {len(top30_dump)}",
        f"- top_k per case: {TOP30_K}",
        f"- unique candidate tracks (top-30 union): {coverage['n_total']}",
        "",
        "### Metadata coverage (% of unique candidate tracks)",
        "",
        "| Field | Coverage |",
        "|---|---:|",
        f"| title | {coverage_pct['title']:.4f} |",
        f"| artist | {coverage_pct['artist']:.4f} |",
        f"| album | {coverage_pct['album']:.4f} |",
        f"| tags (>=1) | {coverage_pct['tags']:.4f} |",
        f"| release_year | {coverage_pct['release_year']:.4f} |",
        "",
        "## Artifacts",
        "",
        f"- Baseline JSON: `{OUT_JSON.relative_to(REPO)}`",
        f"- Top-30 pickle: `{OUT_TOP30.relative_to(REPO)}`",
        f"- This document: `{OUT_MD.relative_to(REPO)}`",
        f"- Script: `scripts/expR67_baseline_repro.py`",
        "",
        "## Notes",
        "",
        f"- Elapsed: {report['elapsed_s']:.1f}s",
        "- Wave 0 reuses cached intermediates and frozen production LR.",
        "- Top-30 records include LR rank/score, per-source ranks (A/B/C/D/F/ALS/R21/R54), "
        "and metadata join (title/artist/album/tags[:5]/release_year).",
        "",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved top-30 pickle: {OUT_TOP30}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(f"{ts()} Verdict: {verdict}  max|delta|={max_abs_delta:.6f}", flush=True)
    for key in REFERENCE_METRICS:
        m = metrics_out[key]
        print(f"  {key}: ref={m['reference']:.6f} got={m['reproduced']:.6f} delta={m['delta']:+.6f}", flush=True)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
