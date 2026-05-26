"""R90 Phase 0 — full-corpus retriever audit (Mac CPU only).

Head-to-head segmentation of R84 (5-fold OOF) vs R54 OOF, per dev case.
Purpose: quantify where Phase 1 continuation training could plausibly help,
and surface 1-2 observable routing rules to try alongside training.

Outputs:
  exp/eval/expR90_phase0_audit.json
  docs/r90_phase0_audit.md
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
R84_FOLD_LISTS = {
    0: REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json",
    1: REPO / "cache" / "r84" / "phase1_fold1" / "oof_r84_lists.json",
    2: REPO / "cache" / "r84" / "phase1_fold2" / "oof_r84_lists.json",
    3: REPO / "cache" / "r84" / "phase1_fold3" / "oof_r84_lists.json",
    4: REPO / "cache" / "r84" / "phase1_fold4" / "oof_r84_lists.json",
}

OUT_JSON = REPO / "exp" / "eval" / "expR90_phase0_audit.json"
OUT_MD = REPO / "docs" / "r90_phase0_audit.md"

# Segment cutoffs used in reporting
QLEN_BUCKETS = [(0, 400), (400, 800), (800, 1200), (1200, 1800), (1800, 99999)]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(rank: int, k: int) -> float:
    if 1 <= rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def gt_rank(top_list: list[str], gt: str) -> int:
    """1-indexed rank of GT in top_list, or 0 if not present."""
    for i, t in enumerate(top_list, 1):
        if t == gt:
            return i
    return 0


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R90 Phase 0 audit — R84 5-fold OOF vs R54 head-to-head")
    print("=" * 70)

    print(f"{ts()} Loading R12 payload + supporting maps...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    maps, _track_pop, _track_album = load_supporting_maps()
    track_artist = maps["track_artist"]

    print(f"{ts()} Loading R54 OOF lists...", flush=True)
    with open(R54_OOF) as f:
        r54_data = json.load(f)
    r54_lists_by_case = {}
    for case_idx, top_list in enumerate(r54_data["lists"]):
        r54_lists_by_case[case_idx] = [tid for tid, _score in top_list]
    r54_scores_by_case = {}
    for case_idx, top_list in enumerate(r54_data["lists"]):
        r54_scores_by_case[case_idx] = {tid: float(s) for tid, s in top_list}

    print(f"{ts()} Loading R84 5-fold OOF lists...", flush=True)
    r84_lists_by_case = {}
    r84_scores_by_case = {}
    case_fold = {}
    for fold, path in R84_FOLD_LISTS.items():
        with open(path) as f:
            d = json.load(f)
        for case_idx_str, top_list in d.items():
            ci = int(case_idx_str)
            r84_lists_by_case[ci] = [tid for tid, _score in top_list]
            r84_scores_by_case[ci] = {tid: float(s) for tid, s in top_list}
            case_fold[ci] = fold
    assert len(r84_lists_by_case) == n, \
        f"R84 missing for {n - len(r84_lists_by_case)} cases"

    # ---- Per-case features ----
    print(f"{ts()} Computing per-case features ({n} cases)...", flush=True)
    rows = []
    for ci in range(n):
        case = cases[ci]
        gt = case["gt"]
        history = case.get("history", [])
        user_query = case.get("user_query", "")
        played = case.get("music_turns", []) or []
        n_prior = int(case.get("n_prior_music", len(played)))
        is_h7 = (n_prior == 7)

        r54_list = r54_lists_by_case[ci]
        r84_list = r84_lists_by_case[ci]
        r54_rank = gt_rank(r54_list, gt)
        r84_rank = gt_rank(r84_list, gt)

        # R54 top-1 cosine margin: top1 - top2 (proxy for "how confident")
        r54_scores = r54_data["lists"][ci]
        r54_top1_margin = (float(r54_scores[0][1]) - float(r54_scores[1][1])
                            if len(r54_scores) >= 2 else 0.0)
        r84_top1_margin = 0.0
        if len(r84_lists_by_case[ci]) >= 2:
            tids = r84_lists_by_case[ci]
            s = r84_scores_by_case[ci]
            r84_top1_margin = s.get(tids[0], 0.0) - s.get(tids[1], 0.0)

        # Query length (chars) — proxy for token count. ~4 chars/token typical.
        q_text = " ".join([user_query] + [str(h) for h in history[-3:]])
        q_chars = len(q_text)

        same_art = bool(same_artist_case(case, track_artist))

        rows.append({
            "case_idx": ci,
            "fold": case_fold[ci],
            "is_h7": is_h7,
            "n_prior": n_prior,
            "q_chars": q_chars,
            "same_art": same_art,
            "gt": gt,
            "r54_rank": r54_rank,        # 0 = miss
            "r84_rank": r84_rank,
            "r54_top1_margin": r54_top1_margin,
            "r84_top1_margin": r84_top1_margin,
            "r54_top1": r54_list[0] if r54_list else None,
            "r84_top1": r84_list[0] if r84_list else None,
        })

    # ---- Aggregates ----
    print(f"{ts()} Computing aggregates...", flush=True)
    h7_rows = [r for r in rows if r["is_h7"]]
    all_rows = rows

    def hit_at(rs: list[dict], side: str, k: int) -> float:
        col = f"{side}_rank"
        return sum(1 for r in rs if 0 < r[col] <= k) / max(len(rs), 1)

    def ndcg_mean(rs: list[dict], side: str, k: int) -> float:
        col = f"{side}_rank"
        return float(np.mean([ndcg_at_k(r[col], k) for r in rs]))

    summary = {
        "n_cases": n,
        "n_h7": len(h7_rows),
        "all": {
            "r54_hit20": hit_at(all_rows, "r54", 20),
            "r84_hit20": hit_at(all_rows, "r84", 20),
            "r54_hit30": hit_at(all_rows, "r54", 30),
            "r84_hit30": hit_at(all_rows, "r84", 30),
            "r54_hit300": hit_at(all_rows, "r54", 300),
            "r84_hit300": hit_at(all_rows, "r84", 300),
            "r54_ndcg20": ndcg_mean(all_rows, "r54", 20),
            "r84_ndcg20": ndcg_mean(all_rows, "r84", 20),
        },
        "h7": {
            "r54_hit20": hit_at(h7_rows, "r54", 20),
            "r84_hit20": hit_at(h7_rows, "r84", 20),
            "r54_hit30": hit_at(h7_rows, "r54", 30),
            "r84_hit30": hit_at(h7_rows, "r84", 30),
            "r54_hit300": hit_at(h7_rows, "r54", 300),
            "r84_hit300": hit_at(h7_rows, "r84", 300),
            "r54_ndcg20": ndcg_mean(h7_rows, "r54", 20),
            "r84_ndcg20": ndcg_mean(h7_rows, "r84", 20),
        },
    }

    # ---- Head-to-head segments (top-30) ----
    def seg(rs: list[dict]) -> dict:
        both_hit = [r for r in rs if 0 < r["r54_rank"] <= 30 and 0 < r["r84_rank"] <= 30]
        only_r54 = [r for r in rs if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30)]
        only_r84 = [r for r in rs if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30)]
        neither = [r for r in rs if not (0 < r["r54_rank"] <= 30) and not (0 < r["r84_rank"] <= 30)]
        return {
            "both_hit": len(both_hit),
            "only_r54": len(only_r54),
            "only_r84": len(only_r84),
            "neither": len(neither),
            "ceiling_pct": 100.0 * (len(both_hit) + len(only_r54) + len(only_r84)) / max(len(rs), 1),
        }
    summary["seg_top30_all"] = seg(all_rows)
    summary["seg_top30_h7"] = seg(h7_rows)
    # And top-300 ceiling
    def seg300(rs):
        union_hit = [r for r in rs if 0 < r["r54_rank"] <= 300 or 0 < r["r84_rank"] <= 300]
        return {
            "union_hit300": len(union_hit),
            "ceiling_pct_300": 100.0 * len(union_hit) / max(len(rs), 1),
        }
    summary["seg_top300_all"] = seg300(all_rows)
    summary["seg_top300_h7"] = seg300(h7_rows)

    # ---- Bucket only_r84 wins and only_r54 wins by query length ----
    def bucket_by_qlen(rs: list[dict]) -> dict:
        out = {}
        for lo, hi in QLEN_BUCKETS:
            sub = [r for r in rs if lo <= r["q_chars"] < hi]
            r84w = sum(1 for r in sub if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30))
            r54w = sum(1 for r in sub if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30))
            out[f"{lo}-{hi}"] = {"n": len(sub), "r84_wins": r84w, "r54_wins": r54w}
        return out
    summary["qlen_buckets_h7"] = bucket_by_qlen(h7_rows)

    # ---- Bucket by same/diff artist ----
    def bucket_by_artist(rs: list[dict]) -> dict:
        out = {}
        for label, sub in [("same", [r for r in rs if r["same_art"]]),
                           ("diff", [r for r in rs if not r["same_art"]])]:
            r84w = sum(1 for r in sub if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30))
            r54w = sum(1 for r in sub if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30))
            r84_ndcg = float(np.mean([ndcg_at_k(r["r84_rank"], 20) for r in sub])) if sub else 0.0
            r54_ndcg = float(np.mean([ndcg_at_k(r["r54_rank"], 20) for r in sub])) if sub else 0.0
            out[label] = {
                "n": len(sub), "r84_wins": r84w, "r54_wins": r54w,
                "r84_ndcg20": r84_ndcg, "r54_ndcg20": r54_ndcg,
                "delta_ndcg20": r84_ndcg - r54_ndcg,
            }
        return out
    summary["artist_buckets_h7"] = bucket_by_artist(h7_rows)

    # ---- Bucket by R54 top-1 cosine margin (observable!) ----
    def bucket_by_r54_margin(rs: list[dict]) -> dict:
        # Use percentile thresholds matched to R84c selective sweep
        edges = [0.0, 0.10, 0.25, 0.50, 1.00, 2.0, 10.0]
        out = {}
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = [r for r in rs if lo <= r["r54_top1_margin"] < hi]
            r84w = sum(1 for r in sub if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30))
            r54w = sum(1 for r in sub if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30))
            r84_ndcg = float(np.mean([ndcg_at_k(r["r84_rank"], 20) for r in sub])) if sub else 0.0
            r54_ndcg = float(np.mean([ndcg_at_k(r["r54_rank"], 20) for r in sub])) if sub else 0.0
            out[f"[{lo:.2f},{hi:.2f})"] = {
                "n": len(sub), "r84_wins": r84w, "r54_wins": r54w,
                "delta_ndcg20": r84_ndcg - r54_ndcg,
            }
        return out
    summary["r54_margin_buckets_h7"] = bucket_by_r54_margin(h7_rows)
    summary["r54_margin_buckets_all"] = bucket_by_r54_margin(all_rows)

    # ---- n_prior (history depth) bucketing ----
    def bucket_by_history(rs: list[dict]) -> dict:
        out = {}
        for n_prior in sorted({r["n_prior"] for r in rs}):
            sub = [r for r in rs if r["n_prior"] == n_prior]
            r84w = sum(1 for r in sub if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30))
            r54w = sum(1 for r in sub if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30))
            r84_ndcg = float(np.mean([ndcg_at_k(r["r84_rank"], 20) for r in sub])) if sub else 0.0
            r54_ndcg = float(np.mean([ndcg_at_k(r["r54_rank"], 20) for r in sub])) if sub else 0.0
            out[str(n_prior)] = {
                "n": len(sub), "r84_wins": r84w, "r54_wins": r54w,
                "delta_ndcg20": r84_ndcg - r54_ndcg,
            }
        return out
    summary["history_buckets_all"] = bucket_by_history(all_rows)

    # ---- Per-fold breakdown (sanity: is one fold worse?) ----
    def per_fold_h7(rs: list[dict]) -> dict:
        out = {}
        for f in range(5):
            sub = [r for r in rs if r["fold"] == f and r["is_h7"]]
            r84w = sum(1 for r in sub if 0 < r["r84_rank"] <= 30 and not (0 < r["r54_rank"] <= 30))
            r54w = sum(1 for r in sub if 0 < r["r54_rank"] <= 30 and not (0 < r["r84_rank"] <= 30))
            r84_ndcg = float(np.mean([ndcg_at_k(r["r84_rank"], 20) for r in sub])) if sub else 0.0
            r54_ndcg = float(np.mean([ndcg_at_k(r["r54_rank"], 20) for r in sub])) if sub else 0.0
            out[str(f)] = {
                "n": len(sub), "r84_wins": r84w, "r54_wins": r54w,
                "r84_ndcg20": r84_ndcg, "r54_ndcg20": r54_ndcg,
                "delta_ndcg20": r84_ndcg - r54_ndcg,
            }
        return out
    summary["per_fold_h7"] = per_fold_h7(rows)

    # ---- Phase 1 ROI prediction (RETRIEVAL LAYER, not LR-scored layer) ----
    # These numbers are RETRIEVAL-ONLY nDCG (GT rank in retriever top-K).
    # They are NOT comparable to R84c selective routing dev numbers
    # (`exp/eval/expR84c_selective.json`), which are LR-scored top-20 nDCG
    # and live on a different scale (~2.3x larger because the LR concentrates
    # GT into top-20 of the RRF pool).
    perfect_route_h7 = float(np.mean([
        max(ndcg_at_k(r["r54_rank"], 20), ndcg_at_k(r["r84_rank"], 20))
        for r in h7_rows
    ]))
    summary["retrieval_perfect_route_ceiling_h7_ndcg20"] = perfect_route_h7
    summary["r84_source_alone_h7_ndcg20"] = summary["h7"]["r84_ndcg20"]
    summary["retrieval_headroom_h7_ndcg20"] = perfect_route_h7 - summary["h7"]["r84_ndcg20"]
    summary["scale_note"] = (
        "All nDCG values in this audit are retrieval-layer (GT rank within "
        "the retriever's top-K, no LR rerank). The R84c selective sweep "
        "(expR84c_selective.json) reports LR-scored top-20 nDCG and is on "
        "a different scale; do not directly compare deltas across the two."
    )

    # h7 cases where neither R54 nor R84 has GT in top-300 (retrieval bottleneck)
    retrieval_bottleneck_h7 = sum(
        1 for r in h7_rows
        if not (0 < r["r54_rank"] <= 300) and not (0 < r["r84_rank"] <= 300)
    )
    summary["retrieval_bottleneck_h7_count"] = retrieval_bottleneck_h7
    summary["retrieval_bottleneck_h7_pct"] = 100.0 * retrieval_bottleneck_h7 / len(h7_rows)

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "experiment": "R90 Phase 0 audit",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "summary": summary,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload_out, f, indent=2)
    print(f"{ts()} Wrote {OUT_JSON.relative_to(REPO)}")
    print(f"{ts()} Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
