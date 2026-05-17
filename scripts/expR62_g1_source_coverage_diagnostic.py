#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""G1 Phase 0 cached source-coverage diagnostic.

Strictly diagnostic:
- reads cached dev payload/source lists only
- reads cached ALS factors and derives the existing ALS source lists
- does not train, retrieve, tune, touch Blind-A, or build a submission

Outputs:
  exp/eval/expR62_g1_source_coverage.json
  docs/r62_g1_source_coverage_result.md
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R58_TOP50 = REPO / "cache" / "r58" / "top50_dev.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
OUT_JSON = REPO / "exp" / "eval" / "expR62_g1_source_coverage.json"
OUT_MD = REPO / "docs" / "r62_g1_source_coverage_result.md"

SOURCE_NAMES = ["A", "B", "C", "D", "F", "ALS", "R21", "R54"]
CATEGORIES = list("ABCDEFGHIJK")
LOW_CATS = ["C", "K"]
HIGH_CATS = ["F", "H"]
POOL_K = 300
RRF_K = 20
SW = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "D": 0.5,
    "F": 1.0,
    "ALS": 1.0,
    "R21": 1.0,
    "R54": 1.0,
}
CEILING_GATE = 0.010


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def rate(hit: int, n: int) -> float:
    return float(hit / n) if n else 0.0


def pct(x: float) -> str:
    return f"{x:.4f}"


def signed(x: float) -> str:
    return f"{x:+.4f}"


def hit_cell(hit: int, n: int) -> str:
    return f"{hit}/{n} ({rate(hit, n):.3f})"


def als_session_vector(
    played: list[str],
    track_to_idx: dict[str, int],
    factors: np.ndarray,
    decay: float = 0.8,
) -> np.ndarray | None:
    anchors = [track_to_idx[t] for t in played if t in track_to_idx]
    if not anchors:
        return None
    n = len(anchors)
    weights = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    weights /= weights.sum()
    vec = np.zeros(factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        vec += weights[j] * factors[idx]
    return vec


def build_als_source_for_case(
    case: dict[str, Any],
    factors: np.ndarray,
    track_ids: list[str],
    track_to_idx: dict[str, int],
) -> list[str]:
    session_vec = als_session_vector(case["music_turns"], track_to_idx, factors)
    if session_vec is None:
        return []
    scores = factors @ session_vec
    for track_id in case["music_turns"]:
        idx = track_to_idx.get(track_id)
        if idx is not None:
            scores[idx] = -np.inf
    top_count = min(200, len(scores))
    top_idx = np.argpartition(-scores, top_count - 1)[:top_count]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [track_ids[int(idx)] for idx in top_idx]


def rrf_scores(source_lists: dict[str, list[str]], source_names: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for source_name in source_names:
        weight = SW.get(source_name, 0.0)
        if weight == 0:
            continue
        for rank_idx, track_id in enumerate(source_lists[source_name], start=1):
            scores[track_id] = scores.get(track_id, 0.0) + weight / (RRF_K + rank_idx)
    return scores


def ranked_rrf(source_lists: dict[str, list[str]], source_names: list[str], topk: int | None = None) -> list[str]:
    scores = rrf_scores(source_lists, source_names)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return ranked[:topk] if topk is not None else ranked


def source_union(source_lists: dict[str, list[str]], source_names: list[str], topk_each: int | None = None) -> list[str]:
    seen: dict[str, None] = {}
    for source_name in source_names:
        ranked = source_lists[source_name]
        if topk_each is not None:
            ranked = ranked[:topk_each]
        for track_id in ranked:
            if track_id not in seen:
                seen[track_id] = None
    return list(seen)


def first_rank_maps(source_lists: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for source_name, ranked in source_lists.items():
        ranks: dict[str, int] = {}
        for rank_idx, track_id in enumerate(ranked, start=1):
            if track_id not in ranks:
                ranks[track_id] = rank_idx
        out[source_name] = ranks
    return out


def rrf_score_for_track(
    rank_maps: dict[str, dict[str, int]],
    source_names: list[str] | tuple[str, ...],
    track_id: str,
) -> float:
    score = 0.0
    for source_name in source_names:
        rank_idx = rank_maps[source_name].get(track_id)
        if rank_idx is not None:
            score += SW[source_name] / (RRF_K + rank_idx)
    return score


def gt_in_rrf_topk_from_rank_maps(
    source_lists: dict[str, list[str]],
    rank_maps: dict[str, dict[str, int]],
    source_names: tuple[str, ...],
    gt: str,
    topk: int,
) -> bool:
    gt_score = rrf_score_for_track(rank_maps, source_names, gt)
    if gt_score <= 0.0:
        return False
    seen_order: dict[str, int] = {}
    for source_name in source_names:
        for track_id in source_lists[source_name]:
            if track_id not in seen_order:
                seen_order[track_id] = len(seen_order)
    gt_order = seen_order.get(gt, 10**12)
    outrank = 0
    for track_id, order in seen_order.items():
        if track_id == gt:
            continue
        score = rrf_score_for_track(rank_maps, source_names, track_id)
        if score > gt_score or (score == gt_score and order < gt_order):
            outrank += 1
            if outrank >= topk:
                return False
    return True


def load_payloads() -> tuple[dict[str, Any], list[list[str]], list[list[str]]]:
    print(f"{ts()} Loading cached R12/R21/R54 source payloads...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_source = [[track_id for track_id, _score in case_lists] for case_lists in r54_data["lists"]]
    return payload, r21_source, r54_source


def load_als_sources(cases: list[dict[str, Any]]) -> list[list[str]]:
    print(f"{ts()} Loading cached ALS factors and deriving existing ALS lists...", flush=True)
    als_data = np.load(ALS_CACHE, allow_pickle=True)
    factors = np.asarray(als_data["factors"], dtype=np.float32)
    track_ids = [str(track_id) for track_id in als_data["track_ids"].tolist()]
    track_to_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}
    out: list[list[str]] = []
    start = time.time()
    for idx, case in enumerate(cases):
        out.append(build_als_source_for_case(case, factors, track_ids, track_to_idx))
        if (idx + 1) % 1000 == 0:
            print(f"  ALS {idx + 1}/{len(cases)} ({time.time() - start:.0f}s)", flush=True)
    return out


def load_goal_categories(cases: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    print(f"{ts()} Loading conversation_goal.category from cached TalkPlayData arrow...", flush=True)
    from datasets import Dataset  # type: ignore[reportMissingImports]
    from eval_inference import cached_test_arrow_path

    arrow_path = cached_test_arrow_path()
    if not arrow_path:
        raise RuntimeError("cached TalkPlayData test Arrow not found; refusing network access")
    ds = Dataset.from_file(arrow_path)
    goal_by_session_user: dict[tuple[str, str], dict[str, Any]] = {}
    goal_by_session: dict[str, dict[str, Any]] = {}
    for row in ds:
        sid = str(row["session_id"])
        uid = "" if row.get("user_id") is None else str(row.get("user_id"))
        goal = row.get("conversation_goal") or {}
        goal_by_session_user[(sid, uid)] = goal
        goal_by_session[sid] = goal

    categories: list[str] = []
    missing = 0
    unknown = Counter()
    for case in cases:
        sid = str(case["session_id"])
        uid = "" if case.get("user_id") is None else str(case.get("user_id"))
        goal = goal_by_session_user.get((sid, uid)) or goal_by_session.get(sid) or {}
        cat = str(goal.get("category") or "UNKNOWN")
        if cat not in CATEGORIES:
            missing += 1
            unknown[cat] += 1
        categories.append(cat)
    meta = {
        "arrow_path": arrow_path,
        "n_sessions": len(ds),
        "missing_or_unknown_goal_categories": missing,
        "unknown_category_counts": dict(unknown),
        "case_category_counts": dict(Counter(categories)),
    }
    return categories, meta


def make_source_lists(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    als_source: list[list[str]],
    case_idx: int,
) -> dict[str, list[str]]:
    return {
        "A": payload["src_a"][case_idx],
        "B": payload["src_b"][case_idx],
        "C": payload["src_c"][case_idx],
        "D": payload["src_d"][case_idx],
        "F": payload["src_f"][case_idx],
        "ALS": als_source[case_idx],
        "R21": r21_source[case_idx],
        "R54": r54_source[case_idx],
    }


def load_top50_info(n_cases: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"{ts()} Loading expR55/R58 top-50 LR cache...", flush=True)
    info = [
        {
            "gt_in_pool": False,
            "gt_lr_rank": -1,
            "gt_top50_rank": -1,
            "top50_rows": 0,
        }
        for _ in range(n_cases)
    ]
    with open(R58_TOP50, "rb") as f:
        rows = pickle.load(f)
    for row in rows:
        case_idx = int(row["case_idx"])
        item = info[case_idx]
        item["top50_rows"] += 1
        item["gt_in_pool"] = bool(row.get("gt_in_pool", 0))
        item["gt_lr_rank"] = int(row.get("gt_lr_rank", -1))
        if int(row.get("gt_flag", 0)) == 1:
            item["gt_top50_rank"] = int(row.get("candidate_rank", -1))
    meta = {"path": str(R58_TOP50), "rows": len(rows)}
    return info, meta


def init_nested_counts() -> dict[str, dict[str, int]]:
    return {category: defaultdict(int) for category in CATEGORIES}


def option_label(source_names: tuple[str, ...] | str) -> str:
    if isinstance(source_names, str):
        return source_names
    return "+".join(source_names)


def best_option_for_category(
    category: str,
    n_by_category: dict[str, int],
    option_hits_by_category: dict[str, dict[str, int]],
    option_kind: str,
    option_names: list[str],
    baseline_hits_by_category: dict[str, int],
) -> dict[str, Any]:
    n = n_by_category[category]
    best_name = option_names[0]
    best_hits = option_hits_by_category[best_name][category]
    for name in option_names[1:]:
        hits = option_hits_by_category[name][category]
        if hits > best_hits:
            best_name = name
            best_hits = hits
    base_hits = baseline_hits_by_category[category]
    return {
        "kind": option_kind,
        "option": best_name,
        "n": n,
        "hits": best_hits,
        "rate": rate(best_hits, n),
        "baseline_hits": base_hits,
        "baseline_rate": rate(base_hits, n),
        "delta_hits": best_hits - base_hits,
        "delta_rate": rate(best_hits, n) - rate(base_hits, n),
    }


def summarize_selected(
    by_category: dict[str, dict[str, Any]],
    n_cases: int,
    baseline_hits: int,
) -> dict[str, Any]:
    hits = sum(int(row["hits"]) for row in by_category.values())
    return {
        "overall_hits": hits,
        "n_cases": n_cases,
        "overall_rate": rate(hits, n_cases),
        "baseline_hits": baseline_hits,
        "baseline_rate": rate(baseline_hits, n_cases),
        "delta_hits": hits - baseline_hits,
        "delta_rate": rate(hits, n_cases) - rate(baseline_hits, n_cases),
        "by_category": by_category,
    }


def lr_rank_bucket(gt_lr_rank: int, gt_in_pool: bool) -> str:
    if gt_in_pool and 1 <= gt_lr_rank <= 20:
        return "lr_top20"
    if gt_in_pool and 21 <= gt_lr_rank <= 50:
        return "lr_21_50"
    if gt_in_pool:
        return "lr_pool_below50_or_not_captured"
    return "not_in_current_rrf_pool"


def median_or_none(values: list[int]) -> float | None:
    return float(median(values)) if values else None


def main() -> None:
    start = time.time()
    payload, r21_source, r54_source = load_payloads()
    cases = payload["cases"]
    n_cases = len(cases)
    if len(r21_source) != n_cases or len(r54_source) != n_cases:
        raise RuntimeError("source cache length mismatch")

    categories, goal_meta = load_goal_categories(cases)
    als_source = load_als_sources(cases)
    top50_info, top50_meta = load_top50_info(n_cases)

    n_by_category = {category: 0 for category in CATEGORIES}
    coverage_cached = {category: defaultdict(int) for category in CATEGORIES}
    coverage_top300 = {category: defaultdict(int) for category in CATEGORIES}
    source_depths: dict[str, Counter[int]] = {source: Counter() for source in SOURCE_NAMES}
    baseline_hit_by_category = {category: 0 for category in CATEGORIES}
    baseline_rrf_rank_by_category: dict[str, list[int]] = {category: [] for category in CATEGORIES}
    source_union_full_by_category = {category: 0 for category in CATEGORIES}
    source_union_top300_by_category = {category: 0 for category in CATEGORIES}

    single_options = [f"single:{source}" for source in SOURCE_NAMES]
    pair_tuples = list(combinations(SOURCE_NAMES, 2))
    pair_options = [f"pair:{option_label(pair)}" for pair in pair_tuples]
    all_ceiling_options = ["baseline:weighted_rrf@300"] + single_options + pair_options
    option_hits_by_category: dict[str, dict[str, int]] = {
        option: {category: 0 for category in CATEGORIES}
        for option in all_ceiling_options
    }
    option_hit_by_case: dict[str, list[bool]] = {
        option: [False] * n_cases for option in all_ceiling_options
    }

    per_case: list[dict[str, Any]] = []
    print(f"{ts()} Counting source coverage and oracle source-choice hits...", flush=True)
    for case_idx, case in enumerate(cases):
        category = categories[case_idx]
        if category not in CATEGORIES:
            continue
        gt = str(case["gt"])
        n_by_category[category] += 1
        source_lists = make_source_lists(payload, r21_source, r54_source, als_source, case_idx)
        rank_maps = first_rank_maps(source_lists)
        for source_name in SOURCE_NAMES:
            source_depths[source_name][len(source_lists[source_name])] += 1

        union_full_set = set(source_union(source_lists, SOURCE_NAMES, topk_each=None))
        union_top300_set = set(source_union(source_lists, SOURCE_NAMES, topk_each=POOL_K))
        baseline_ranked_full = ranked_rrf(source_lists, SOURCE_NAMES, topk=None)
        baseline_pool = baseline_ranked_full[:POOL_K]
        baseline_hit = gt in baseline_pool
        baseline_rrf_rank = baseline_ranked_full.index(gt) + 1 if gt in union_full_set else -1
        if baseline_rrf_rank > 0:
            baseline_rrf_rank_by_category[category].append(baseline_rrf_rank)

        if gt in union_full_set:
            source_union_full_by_category[category] += 1
            coverage_cached[category]["source-union"] += 1
        if gt in union_top300_set:
            source_union_top300_by_category[category] += 1
            coverage_top300[category]["source-union"] += 1
        if baseline_hit:
            baseline_hit_by_category[category] += 1
            coverage_cached[category]["weighted_rrf@300"] += 1
            coverage_top300[category]["weighted_rrf@300"] += 1
            option_hits_by_category["baseline:weighted_rrf@300"][category] += 1
        option_hit_by_case["baseline:weighted_rrf@300"][case_idx] = baseline_hit

        for source_name in SOURCE_NAMES:
            cached_hit = gt in set(source_lists[source_name])
            top300_hit = gt in set(source_lists[source_name][:POOL_K])
            if cached_hit:
                coverage_cached[category][source_name] += 1
            if top300_hit:
                coverage_top300[category][source_name] += 1
            option_name = f"single:{source_name}"
            if top300_hit:
                option_hits_by_category[option_name][category] += 1
            option_hit_by_case[option_name][case_idx] = top300_hit

        for pair in pair_tuples:
            option_name = f"pair:{option_label(pair)}"
            pair_hit = gt_in_rrf_topk_from_rank_maps(source_lists, rank_maps, pair, gt, POOL_K)
            if pair_hit:
                option_hits_by_category[option_name][category] += 1
            option_hit_by_case[option_name][case_idx] = pair_hit

        per_case.append(
            {
                "case_idx": case_idx,
                "category": category,
                "gt": gt,
                "baseline_hit": baseline_hit,
                "source_union_full_hit": gt in union_full_set,
                "source_union_top300_hit": gt in union_top300_set,
                "baseline_rrf_rank": baseline_rrf_rank,
                "gt_lr_rank": int(top50_info[case_idx]["gt_lr_rank"]),
                "gt_in_top50": int(top50_info[case_idx]["gt_top50_rank"]) > 0,
            }
        )

        if (case_idx + 1) % 1000 == 0:
            print(f"  counted {case_idx + 1}/{n_cases}", flush=True)

    baseline_hits = sum(baseline_hit_by_category.values())
    baseline_rate = rate(baseline_hits, n_cases)

    print(f"{ts()} Selecting per-category oracle source choices...", flush=True)
    best_single_by_category = {
        category: best_option_for_category(
            category,
            n_by_category,
            option_hits_by_category,
            "single_source",
            single_options,
            baseline_hit_by_category,
        )
        for category in CATEGORIES
    }
    best_pair_by_category = {
        category: best_option_for_category(
            category,
            n_by_category,
            option_hits_by_category,
            "two_source_rrf",
            pair_options,
            baseline_hit_by_category,
        )
        for category in CATEGORIES
    }
    best_conditioned_by_category = {
        category: best_option_for_category(
            category,
            n_by_category,
            option_hits_by_category,
            "best_of_baseline_single_pair",
            all_ceiling_options,
            baseline_hit_by_category,
        )
        for category in CATEGORIES
    }
    oracle_single = summarize_selected(best_single_by_category, n_cases, baseline_hits)
    oracle_pair = summarize_selected(best_pair_by_category, n_cases, baseline_hits)
    oracle_conditioned = summarize_selected(best_conditioned_by_category, n_cases, baseline_hits)

    asymmetry_by_category: dict[str, dict[str, Any]] = {}
    for category in LOW_CATS + HIGH_CATS:
        single_row = best_single_by_category[category]
        pair_row = best_pair_by_category[category]
        conditioned_row = best_conditioned_by_category[category]
        best_alt_delta = max(float(single_row["delta_rate"]), float(pair_row["delta_rate"]))
        asymmetry_by_category[category] = {
            "n": n_by_category[category],
            "baseline_hits": baseline_hit_by_category[category],
            "baseline_rate": rate(baseline_hit_by_category[category], n_by_category[category]),
            "best_single_option": single_row["option"],
            "best_single_rate": single_row["rate"],
            "best_single_delta_rate": single_row["delta_rate"],
            "best_pair_option": pair_row["option"],
            "best_pair_rate": pair_row["rate"],
            "best_pair_delta_rate": pair_row["delta_rate"],
            "best_conditioned_option": conditioned_row["option"],
            "best_conditioned_rate": conditioned_row["rate"],
            "best_conditioned_delta_rate": conditioned_row["delta_rate"],
            "source_specific_advantage": best_alt_delta > 0.0,
        }
    ck_advantage_present = any(asymmetry_by_category[category]["source_specific_advantage"] for category in LOW_CATS)

    print(f"{ts()} Running C/K LR-bury analysis...", flush=True)
    lr_bury = build_lr_bury_analysis(
        per_case,
        top50_info,
        option_hit_by_case,
        best_single_by_category,
        best_pair_by_category,
        best_conditioned_by_category,
    )

    consistency = build_consistency(top50_info, option_hit_by_case["baseline:weighted_rrf@300"])
    verdict, verdict_reason = decide_verdict(oracle_conditioned, ck_advantage_present)

    data = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "elapsed_s": time.time() - start,
            "n_cases": n_cases,
            "pool_k": POOL_K,
            "rrf_k": RRF_K,
            "source_names": SOURCE_NAMES,
            "source_weights": SW,
            "inputs": {
                "r12_payload": str(R12_CACHE),
                "r58_top50": str(R58_TOP50),
                "r21_oof": str(R21_OOF),
                "r54_phase2_oof": str(R54_PHASE2_OOF),
                "als_cache": str(ALS_CACHE),
            },
            "goal_meta": goal_meta,
            "top50_meta": top50_meta,
            "notes": [
                "No retrieval, training, LR feature changes, Blind-A access, or submission build was performed.",
                "Cached-depth source coverage uses each cached list as-is; pool-comparable source-choice ceilings enforce pool@300.",
                "R54 uses Phase 2 OOF cached lists, matching the post-refresh diagnostic precedent.",
            ],
        },
        "sanity": {
            "baseline_hits": baseline_hits,
            "baseline_rate": baseline_rate,
            "source_union_cached_depth_hits": sum(source_union_full_by_category.values()),
            "source_union_cached_depth_rate": rate(sum(source_union_full_by_category.values()), n_cases),
            "source_union_top300_hits": sum(source_union_top300_by_category.values()),
            "source_union_top300_rate": rate(sum(source_union_top300_by_category.values()), n_cases),
            "source_depths": {source: dict(counter) for source, counter in source_depths.items()},
            "top50_consistency": consistency,
        },
        "per_category": build_per_category_output(
            n_by_category,
            coverage_cached,
            coverage_top300,
            baseline_rrf_rank_by_category,
        ),
        "oracle": {
            "baseline": {
                "hits": baseline_hits,
                "rate": baseline_rate,
            },
            "best_single_source_per_category": oracle_single,
            "best_two_source_rrf_per_category": oracle_pair,
            "best_category_conditioned_source_choice_ceiling": {
                **oracle_conditioned,
                "gate_threshold_delta_rate": CEILING_GATE,
                "gate_pass": oracle_conditioned["delta_rate"] >= CEILING_GATE,
            },
        },
        "asymmetry": {
            "low_categories": LOW_CATS,
            "high_categories": HIGH_CATS,
            "by_category": asymmetry_by_category,
            "ck_source_specific_advantage_present": ck_advantage_present,
            "c_source_asymmetry": asymmetry_by_category["C"]["source_specific_advantage"],
            "k_source_asymmetry": asymmetry_by_category["K"]["source_specific_advantage"],
        },
        "lr_bury_analysis": lr_bury,
        "verdict": {
            "decision": verdict,
            "reason": verdict_reason,
            "proceed_requires": {
                "best_conditioned_delta_rate_at_least": CEILING_GATE,
                "ck_source_specific_advantage_present": True,
            },
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(data), encoding="utf-8")

    print(f"{ts()} Wrote {OUT_JSON}", flush=True)
    print(f"{ts()} Wrote {OUT_MD}", flush=True)
    print(
        f"{verdict}: oracle ceiling={signed(oracle_conditioned['delta_rate'])}, "
        f"C asymmetry={'Y' if data['asymmetry']['c_source_asymmetry'] else 'N'}, "
        f"K asymmetry={'Y' if data['asymmetry']['k_source_asymmetry'] else 'N'}",
        flush=True,
    )


def build_consistency(top50_info: list[dict[str, Any]], baseline_hit_by_case: list[bool]) -> dict[str, Any]:
    mismatches = []
    for idx, baseline_hit in enumerate(baseline_hit_by_case):
        top50_pool = bool(top50_info[idx]["gt_in_pool"])
        if top50_pool != baseline_hit:
            mismatches.append(idx)
    return {
        "baseline_vs_top50_gt_in_pool_mismatches": len(mismatches),
        "mismatch_case_idx_sample": mismatches[:20],
        "top50_gt_in_pool_hits": sum(1 for row in top50_info if row["gt_in_pool"]),
        "baseline_recomputed_hits": sum(1 for hit in baseline_hit_by_case if hit),
    }


def build_per_category_output(
    n_by_category: dict[str, int],
    coverage_cached: dict[str, defaultdict[str, int]],
    coverage_top300: dict[str, defaultdict[str, int]],
    baseline_rrf_rank_by_category: dict[str, list[int]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    columns = SOURCE_NAMES + ["source-union", "weighted_rrf@300"]
    for category in CATEGORIES:
        n = n_by_category[category]
        cached_counts = {column: int(coverage_cached[category][column]) for column in columns}
        top300_counts = {column: int(coverage_top300[category][column]) for column in columns}
        out[category] = {
            "n": n,
            "coverage_cached_depth_counts": cached_counts,
            "coverage_cached_depth_rates": {column: rate(cached_counts[column], n) for column in columns},
            "coverage_top300_counts": top300_counts,
            "coverage_top300_rates": {column: rate(top300_counts[column], n) for column in columns},
            "baseline_rrf_rank_for_union_hits": {
                "median": median_or_none(baseline_rrf_rank_by_category[category]),
                "min": min(baseline_rrf_rank_by_category[category]) if baseline_rrf_rank_by_category[category] else None,
                "max": max(baseline_rrf_rank_by_category[category]) if baseline_rrf_rank_by_category[category] else None,
            },
        }
    return out


def build_lr_bury_analysis(
    per_case: list[dict[str, Any]],
    top50_info: list[dict[str, Any]],
    option_hit_by_case: dict[str, list[bool]],
    best_single_by_category: dict[str, dict[str, Any]],
    best_pair_by_category: dict[str, dict[str, Any]],
    best_conditioned_by_category: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    categories = LOW_CATS + ["C+K"]
    rows_by_group: dict[str, list[dict[str, Any]]] = {
        "C": [row for row in per_case if row["category"] == "C"],
        "K": [row for row in per_case if row["category"] == "K"],
    }
    rows_by_group["C+K"] = rows_by_group["C"] + rows_by_group["K"]

    out: dict[str, Any] = {}
    for group in categories:
        rows = rows_by_group[group]
        status_counts: Counter[str] = Counter()
        miss_counts: Counter[str] = Counter()
        miss_rrf_ranks: list[int] = []
        top50_present_in_misses = 0
        recovered: dict[str, Any] = {}

        for row in rows:
            case_idx = int(row["case_idx"])
            gt_in_pool = bool(top50_info[case_idx]["gt_in_pool"])
            gt_lr_rank = int(top50_info[case_idx]["gt_lr_rank"])
            status_counts[lr_rank_bucket(gt_lr_rank, gt_in_pool)] += 1

            if not row["baseline_hit"]:
                if row["source_union_top300_hit"]:
                    miss_counts["baseline_miss_but_in_source_union_top300"] += 1
                elif row["source_union_full_hit"]:
                    miss_counts["baseline_miss_in_cached_union_only_beyond_top300"] += 1
                else:
                    miss_counts["baseline_miss_outside_cached_source_union"] += 1
                if int(row["baseline_rrf_rank"]) > 0:
                    miss_rrf_ranks.append(int(row["baseline_rrf_rank"]))
                if int(top50_info[case_idx]["gt_top50_rank"]) > 0:
                    top50_present_in_misses += 1

        for oracle_name, oracle_by_category in [
            ("best_single_source", best_single_by_category),
            ("best_two_source_rrf", best_pair_by_category),
            ("best_conditioned_source_choice", best_conditioned_by_category),
        ]:
            recovered_counts: Counter[str] = Counter()
            recovered_lr_buckets: Counter[str] = Counter()
            recovered_options: Counter[str] = Counter()
            for row in rows:
                case_idx = int(row["case_idx"])
                category = row["category"]
                option = str(oracle_by_category[category]["option"])
                if row["baseline_hit"] or not option_hit_by_case[option][case_idx]:
                    continue
                recovered_options[option] += 1
                recovered_counts["total_recovered_over_current_rrf300"] += 1
                if row["source_union_top300_hit"]:
                    recovered_counts["in_source_union_top300"] += 1
                if row["source_union_full_hit"]:
                    recovered_counts["in_cached_source_union"] += 1
                if int(top50_info[case_idx]["gt_top50_rank"]) > 0:
                    recovered_counts["appears_in_expR55_top50"] += 1
                if bool(top50_info[case_idx]["gt_in_pool"]):
                    recovered_counts["was_scored_in_current_rrf300_pool"] += 1
                else:
                    recovered_counts["unscored_by_current_expR55_pool"] += 1
                recovered_lr_buckets[
                    lr_rank_bucket(
                        int(top50_info[case_idx]["gt_lr_rank"]),
                        bool(top50_info[case_idx]["gt_in_pool"]),
                    )
                ] += 1
            recovered[oracle_name] = {
                "counts": dict(recovered_counts),
                "lr_rank_buckets": dict(recovered_lr_buckets),
                "options": dict(recovered_options),
            }

        out[group] = {
            "n": len(rows),
            "current_rrf300_lr_rank_buckets": dict(status_counts),
            "current_rrf300_misses": {
                "total": sum(miss_counts.values()),
                **dict(miss_counts),
                "appears_in_expR55_top50": top50_present_in_misses,
                "unscored_by_current_expR55_pool": sum(miss_counts.values()) - top50_present_in_misses,
                "rrf_full_rank_for_cached_union_misses": {
                    "median": median_or_none(miss_rrf_ranks),
                    "min": min(miss_rrf_ranks) if miss_rrf_ranks else None,
                    "max": max(miss_rrf_ranks) if miss_rrf_ranks else None,
                },
            },
            "recovered_over_current_rrf300": recovered,
        }
    return out


def decide_verdict(oracle_conditioned: dict[str, Any], ck_advantage_present: bool) -> tuple[str, str]:
    delta = float(oracle_conditioned["delta_rate"])
    ceiling_pass = delta >= CEILING_GATE
    if ceiling_pass and ck_advantage_present:
        return (
            "PROCEED_TO_SIMULATION_REVIEW",
            f"best category-conditioned source-choice ceiling clears {CEILING_GATE:+.3f} "
            "and C/K source-specific advantage is present",
        )
    reasons = []
    if not ceiling_pass:
        reasons.append(f"oracle ceiling {delta:+.4f} is below {CEILING_GATE:+.3f}")
    if not ck_advantage_present:
        reasons.append("C/K source-specific advantage is absent")
    return "ARCHIVE_G1", "; ".join(reasons)


def render_markdown(data: dict[str, Any]) -> str:
    oracle = data["oracle"]["best_category_conditioned_source_choice_ceiling"]
    baseline = data["oracle"]["baseline"]
    asym = data["asymmetry"]
    verdict = data["verdict"]
    ceiling_delta = float(oracle["delta_rate"])
    ceiling_gate = (
        f"{'PROCEED' if oracle['gate_pass'] else 'ARCHIVE'} "
        f"({'>=' if oracle['gate_pass'] else '<'} +0.010)"
    )
    ck_advantage = bool(asym["ck_source_specific_advantage_present"])
    ck_gate = "PROCEED (Yes)" if ck_advantage else "ARCHIVE (No)"

    lines = [
        "| Diagnostic | Value | Gate |",
        "|---|---:|---|",
        (
            "| Best category-conditioned source-choice ceiling on all-dev pool_hit | "
            f"{signed(ceiling_delta)} ({pct(baseline['rate'])} -> {pct(oracle['overall_rate'])}) | "
            f"{ceiling_gate} |"
        ),
        (
            "| C/K source-specific advantage present | "
            f"{'Yes' if ck_advantage else 'No'} "
            f"(C={'Y' if asym['c_source_asymmetry'] else 'N'}, K={'Y' if asym['k_source_asymmetry'] else 'N'}) | "
            f"{ck_gate} |"
        ),
        "",
        f"**Verdict**",
        "",
        f"{verdict['decision']}: {verdict['reason']}.",
        "",
        "**Coverage Matrix**",
        "",
        "Cached-depth source coverage counts GT membership in each cached list as-is; "
        "B/C have depth 500, R21/R54 depth 300, ALS/A/D/F max depth 200. "
        "The oracle sections below enforce pool@300 for source-choice comparisons.",
        "",
    ]
    matrix_cols = SOURCE_NAMES + ["source-union", "weighted_rrf@300"]
    lines.append("| category | n | " + " | ".join(matrix_cols) + " |")
    lines.append("|---|---:|" + "|".join(["---:" for _ in matrix_cols]) + "|")
    for category in CATEGORIES:
        row = data["per_category"][category]
        n = int(row["n"])
        counts = row["coverage_cached_depth_counts"]
        vals = [hit_cell(int(counts[col]), n) for col in matrix_cols]
        lines.append(f"| {category} | {n} | " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "**Oracle Ceiling**",
            "",
            render_oracle_table(
                "Best single source per category",
                data["oracle"]["best_single_source_per_category"],
                "source",
            ),
            "",
            render_oracle_table(
                "Best 2-source RRF per category",
                data["oracle"]["best_two_source_rrf_per_category"],
                "pair",
            ),
            "",
            render_oracle_table(
                "Best conditioned choice per category (baseline, single, or pair)",
                data["oracle"]["best_category_conditioned_source_choice_ceiling"],
                "option",
            ),
            "",
            "**C/K vs F/H Focus**",
            "",
            "| category | baseline | best single | delta | best pair | delta | best conditioned | delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category in LOW_CATS + HIGH_CATS:
        row = asym["by_category"][category]
        lines.append(
            f"| {category} | {pct(row['baseline_rate'])} | "
            f"{row['best_single_option']} {pct(row['best_single_rate'])} | {signed(row['best_single_delta_rate'])} | "
            f"{row['best_pair_option']} {pct(row['best_pair_rate'])} | {signed(row['best_pair_delta_rate'])} | "
            f"{row['best_conditioned_option']} {pct(row['best_conditioned_rate'])} | "
            f"{signed(row['best_conditioned_delta_rate'])} |"
        )

    lines.extend(
        [
            "",
            "**LR-Bury Analysis**",
            "",
            "For C/K current RRF@300 misses, `source_union_top300` means the GT is already in at least one existing source within the pool-comparable top-300 depth but was diluted out of the current all-source RRF@300. `cached_union_only` is mostly B/C depth beyond 300. `unscored` means absent from the current expR55 RRF@300 pool and therefore absent from its LR top-50 capture.",
            "",
            "| group | baseline misses | in source_union_top300 | cached_union_only | outside cached union | appears in expR55 top50 | unscored | best-conditioned recovered | recovered in expR55 top50 | recovered unscored |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ["C", "K", "C+K"]:
        row = data["lr_bury_analysis"][group]["current_rrf300_misses"]
        rec = data["lr_bury_analysis"][group]["recovered_over_current_rrf300"]["best_conditioned_source_choice"]["counts"]
        lines.append(
            f"| {group} | {row.get('total', 0)} | "
            f"{row.get('baseline_miss_but_in_source_union_top300', 0)} | "
            f"{row.get('baseline_miss_in_cached_union_only_beyond_top300', 0)} | "
            f"{row.get('baseline_miss_outside_cached_source_union', 0)} | "
            f"{row.get('appears_in_expR55_top50', 0)} | "
            f"{row.get('unscored_by_current_expR55_pool', 0)} | "
            f"{rec.get('total_recovered_over_current_rrf300', 0)} | "
            f"{rec.get('appears_in_expR55_top50', 0)} | "
            f"{rec.get('unscored_by_current_expR55_pool', 0)} |"
        )

    lines.extend(
        [
            "",
            "**Sanity**",
            "",
            f"- Recomputed weighted_rrf@300 pool_hit: {data['sanity']['baseline_hits']}/{data['meta']['n_cases']} ({pct(data['sanity']['baseline_rate'])}).",
            f"- Cached-depth source-union hit: {data['sanity']['source_union_cached_depth_hits']}/{data['meta']['n_cases']} ({pct(data['sanity']['source_union_cached_depth_rate'])}).",
            f"- Top50 `gt_in_pool` mismatches vs recomputed baseline: {data['sanity']['top50_consistency']['baseline_vs_top50_gt_in_pool_mismatches']}.",
            "",
            "**Verdict Line**",
            "",
            f"{verdict['decision']} with reason: {verdict['reason']}.",
            "",
        ]
    )
    return "\n".join(lines)


def render_oracle_table(title: str, oracle: dict[str, Any], option_label_name: str) -> str:
    lines = [
        f"{title}: overall {oracle['overall_hits']}/{oracle['n_cases']} "
        f"({pct(oracle['overall_rate'])}), delta {oracle['delta_hits']:+d} cases / {signed(oracle['delta_rate'])}.",
        "",
        f"| category | {option_label_name} | hits/n | rate | delta rate |",
        "|---|---|---:|---:|---:|",
    ]
    for category in CATEGORIES:
        row = oracle["by_category"][category]
        lines.append(
            f"| {category} | {row['option']} | {row['hits']}/{row['n']} | "
            f"{pct(row['rate'])} | {signed(row['delta_rate'])} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
