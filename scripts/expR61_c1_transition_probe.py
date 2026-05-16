#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R61 / C1 cheap transition-memory probe.

Count-only, train-split-only transition source. No neural model, no embedding
training, no LR retraining, no blind data.

Outputs:
  exp/eval/expR61_c1_transition_probe.json
  docs/r61_c1_transition_probe_result.md
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO / ".hf_cache" / "datasets"))
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]

import scripts.expR59_c3_pool_admission_diagnostic as c3
import scripts.expR59_c3_phase2_frozen_lr_conversion as frozen_lr

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
DECOMP_JSON = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"
OUT_JSON = REPO / "exp" / "eval" / "expR61_c1_transition_probe.json"
OUT_MD = REPO / "docs" / "r61_c1_transition_probe_result.md"

TRAIN_DATASET = "talkpl-ai/TalkPlayData-Challenge-Dataset"
POOL_K = 300
TOP_K = 20
RRF_K = 20
C1_SOURCE_NAMES = [
    "c1_last_track",
    "c1_last3_tracks_recency",
    "c1_last_artist",
    "c1_last3_artist_tag_backoff",
]
RECENCY_WEIGHTS = [1.0, 0.5, 0.25]
FUSION_WEIGHTS = [0.25, 0.5, 1.0]
GATE_H7_UNIQUE_OUTSIDE = 30
GATE_H7_NDCG_DELTA = 0.003
GATE_SAME_ARTIST_DELTA = -0.002
GATE_TOP1_CHURN_RATE = 0.015

COUNTER_PRUNE_EXACT = 1000
COUNTER_PRUNE_ARTIST = 1500
COUNTER_PRUNE_TAG = 250
MAX_TAGS_PER_TRACK = 32


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def find_train_arrow_path() -> str:
    candidates = sorted(
        (REPO / ".hf_cache").glob(
            "datasets/talkpl-ai___talk_play_data-challenge-dataset/**/"
            "talk_play_data-challenge-dataset-train.arrow"
        )
    )
    if candidates:
        return str(candidates[0])
    home_candidates = sorted(
        Path.home().glob(
            ".cache/huggingface/datasets/talkpl-ai___talk_play_data-challenge-dataset/**/"
            "talk_play_data-challenge-dataset-train.arrow"
        )
    )
    return str(home_candidates[0]) if home_candidates else ""


def load_payload_for_dev_sessions() -> tuple[dict[str, Any], set[str]]:
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    dev_session_ids = {str(case["session_id"]) for case in payload["cases"]}
    return payload, dev_session_ids


def first_scalar(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def normalize_tags(tags: Any) -> list[str]:
    if not tags:
        return []
    values = list(tags) if isinstance(tags, (set, list, tuple)) else [str(tags)]
    out = []
    seen = set()
    for tag in values:
        val = str(tag).strip().lower()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
        if len(out) >= MAX_TAGS_PER_TRACK:
            break
    return out


def music_events_with_user_text(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
    events = []
    previous_user_text = ""
    current_user_text = ""
    for row in conversations:
        role = str(row.get("role", "")).lower()
        content = str(row.get("content", "")).strip()
        if role == "user":
            previous_user_text = current_user_text
            current_user_text = content
        elif role == "music" and content:
            events.append(
                {
                    "track_id": content,
                    "turn_number": str(row.get("turn_number", "")),
                    "current_user_text": current_user_text,
                    "previous_user_text": previous_user_text,
                }
            )
    return events


def add_counter(counter_map: dict[str, Counter[str]], key: str, target: str, weight: float = 1.0) -> None:
    if key and target:
        counter_map[key][target] += weight


def prune_counter_map(
    counter_map: dict[str, Counter[str]],
    limit: int,
) -> dict[str, list[tuple[str, float]]]:
    pruned: dict[str, list[tuple[str, float]]] = {}
    for key, counter in counter_map.items():
        if not counter:
            continue
        items = sorted(counter.items(), key=lambda item: (-float(item[1]), item[0]))[:limit]
        pruned[key] = [(str(track_id), float(score)) for track_id, score in items]
    return pruned


def load_train_split() -> Any:
    print(f"{ts()} Loading official train split only from HF local cache...", flush=True)
    return load_dataset(
        TRAIN_DATASET,
        split="train",
        download_config=DownloadConfig(local_files_only=True),
    )


def build_transition_memory(
    train_ds: Any,
    dev_session_ids: set[str],
    track_artist: dict[str, str],
    track_tags: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_track_to_next: dict[str, Counter[str]] = defaultdict(Counter)
    last3_track_to_next: dict[str, Counter[str]] = defaultdict(Counter)
    last_artist_to_next: dict[str, Counter[str]] = defaultdict(Counter)
    recent_artist_to_next: dict[str, Counter[str]] = defaultdict(Counter)
    recent_tag_to_next: dict[str, Counter[str]] = defaultdict(Counter)

    train_session_ids = []
    excluded_overlap = 0
    music_turn_hist = Counter()
    transition_count = 0
    text_context_count = 0
    sessions_with_2plus_music = 0
    unknown_artist_refs = 0
    unknown_tag_refs = 0

    start = time.time()
    for row_idx, row in enumerate(train_ds):
        sid = str(row["session_id"])
        train_session_ids.append(sid)
        if sid in dev_session_ids:
            excluded_overlap += 1
            continue

        events = music_events_with_user_text(row["conversations"])
        music_turn_hist[len(events)] += 1
        if len(events) >= 2:
            sessions_with_2plus_music += 1

        played_so_far: list[str] = []
        for event in events:
            target = str(event["track_id"])
            if played_so_far:
                transition_count += 1
                if event["current_user_text"] or event["previous_user_text"]:
                    text_context_count += 1

                last_track = played_so_far[-1]
                add_counter(last_track_to_next, last_track, target)
                last_artist = track_artist.get(last_track, "")
                if last_artist:
                    add_counter(last_artist_to_next, last_artist, target)
                else:
                    unknown_artist_refs += 1

                recent_tracks = list(reversed(played_so_far[-3:]))
                for pos, prior_track in enumerate(recent_tracks):
                    recency_weight = RECENCY_WEIGHTS[pos]
                    add_counter(last3_track_to_next, prior_track, target, recency_weight)
                    prior_artist = track_artist.get(prior_track, "")
                    if prior_artist:
                        add_counter(recent_artist_to_next, prior_artist, target, recency_weight)
                    else:
                        unknown_artist_refs += 1
                    tags = normalize_tags(track_tags.get(prior_track, set()))
                    if tags:
                        tag_weight = recency_weight / len(tags)
                        for tag in tags:
                            add_counter(recent_tag_to_next, tag, target, tag_weight)
                    else:
                        unknown_tag_refs += 1
            played_so_far.append(target)

        if (row_idx + 1) % 3000 == 0:
            print(
                f"  scanned {row_idx + 1}/{len(train_ds)} train sessions "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )

    print(f"{ts()} Pruning transition counters for fast dev scoring...", flush=True)
    stats = {
        "last_track_to_next": prune_counter_map(last_track_to_next, COUNTER_PRUNE_EXACT),
        "last3_track_to_next": prune_counter_map(last3_track_to_next, COUNTER_PRUNE_EXACT),
        "last_artist_to_next": prune_counter_map(last_artist_to_next, COUNTER_PRUNE_ARTIST),
        "recent_artist_to_next": prune_counter_map(recent_artist_to_next, COUNTER_PRUNE_ARTIST),
        "recent_tag_to_next": prune_counter_map(recent_tag_to_next, COUNTER_PRUNE_TAG),
    }
    audit = {
        "dataset": TRAIN_DATASET,
        "split": "train",
        "train_arrow_path": find_train_arrow_path(),
        "train_rows": len(train_ds),
        "train_fields": list(train_ds.features.keys()),
        "session_id_sample": str(train_ds[0]["session_id"]) if len(train_ds) else "",
        "session_id_schema": "uuid-like string",
        "session_id_sample_uuid_match": bool(
            re.fullmatch(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                str(train_ds[0]["session_id"]) if len(train_ds) else "",
            )
        ),
        "train_unique_session_ids": len(set(train_session_ids)),
        "dev_unique_session_ids": len(dev_session_ids),
        "excluded_train_sessions_overlapping_dev": excluded_overlap,
        "included_train_sessions": len(train_ds) - excluded_overlap,
        "sessions_with_2plus_music": sessions_with_2plus_music,
        "transition_rows": transition_count,
        "transition_rows_with_user_text_context": text_context_count,
        "music_turn_histogram_after_exclusion": dict(sorted(music_turn_hist.items())),
        "unknown_artist_references": unknown_artist_refs,
        "unknown_tag_references": unknown_tag_refs,
        "counter_sizes": {name: len(value) for name, value in stats.items()},
        "counter_prune_limits": {
            "exact_track": COUNTER_PRUNE_EXACT,
            "artist": COUNTER_PRUNE_ARTIST,
            "tag": COUNTER_PRUNE_TAG,
        },
    }
    return stats, audit


def add_items(
    scores: dict[str, float],
    items: list[tuple[str, float]] | None,
    weight: float,
    played_set: set[str],
) -> None:
    if not items or weight == 0.0:
        return
    for track_id, score in items:
        if track_id in played_set:
            continue
        scores[track_id] = scores.get(track_id, 0.0) + weight * float(score)


def rank_scores(scores: dict[str, float], k: int = POOL_K) -> list[str]:
    if not scores:
        return []
    return [
        track_id
        for track_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]


def c1_sources_for_case(
    case: dict[str, Any],
    transition_stats: dict[str, Any],
    track_artist: dict[str, str],
    track_tags: dict[str, set[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    played = [str(track_id) for track_id in case["music_turns"]]
    played_set = set(played)
    source_scores: dict[str, dict[str, float]] = {name: {} for name in C1_SOURCE_NAMES}

    if played:
        last_track = played[-1]
        add_items(
            source_scores["c1_last_track"],
            transition_stats["last_track_to_next"].get(last_track),
            1.0,
            played_set,
        )

        last_artist = track_artist.get(last_track, "")
        if last_artist:
            add_items(
                source_scores["c1_last_artist"],
                transition_stats["last_artist_to_next"].get(last_artist),
                1.0,
                played_set,
            )

    recent_tracks = list(reversed(played[-3:]))
    for pos, prior_track in enumerate(recent_tracks):
        recency_weight = RECENCY_WEIGHTS[pos]
        add_items(
            source_scores["c1_last3_tracks_recency"],
            transition_stats["last3_track_to_next"].get(prior_track),
            recency_weight,
            played_set,
        )
        prior_artist = track_artist.get(prior_track, "")
        if prior_artist:
            add_items(
                source_scores["c1_last3_artist_tag_backoff"],
                transition_stats["recent_artist_to_next"].get(prior_artist),
                recency_weight * 0.70,
                played_set,
            )
        tags = normalize_tags(track_tags.get(prior_track, set()))
        if tags:
            per_tag_weight = recency_weight * 0.30 / len(tags)
            for tag in tags:
                add_items(
                    source_scores["c1_last3_artist_tag_backoff"],
                    transition_stats["recent_tag_to_next"].get(tag),
                    per_tag_weight,
                    played_set,
                )

    source_lists = {
        source_name: rank_scores(source_scores[source_name], POOL_K)
        for source_name in C1_SOURCE_NAMES
    }
    combined_scores: dict[str, float] = {}
    for source_name in C1_SOURCE_NAMES:
        for rank, track_id in enumerate(source_lists[source_name], start=1):
            combined_scores[track_id] = combined_scores.get(track_id, 0.0) + 1.0 / (RRF_K + rank)
    combined = rank_scores(combined_scores, POOL_K)
    return source_lists, combined


def build_c1_dev_lists(
    cases: list[dict[str, Any]],
    transition_stats: dict[str, Any],
    track_artist: dict[str, str],
    track_tags: dict[str, set[str]],
) -> tuple[list[dict[str, list[str]]], list[list[str]], dict[str, Any]]:
    source_lists_by_case = []
    combined_by_case = []
    nonempty_by_source = Counter()
    source_hit_counts = Counter()
    start = time.time()
    print(f"{ts()} Scoring dev cases with fixed C1 transition sources...", flush=True)
    for case_idx, case in enumerate(cases):
        source_lists, combined = c1_sources_for_case(case, transition_stats, track_artist, track_tags)
        source_lists_by_case.append(source_lists)
        combined_by_case.append(combined)
        gt = case["gt"]
        for source_name, ranked in source_lists.items():
            if ranked:
                nonempty_by_source[source_name] += 1
            if gt in set(ranked[:POOL_K]):
                source_hit_counts[source_name] += 1
        if (case_idx + 1) % 1000 == 0:
            print(
                f"  C1 scored {case_idx + 1}/{len(cases)} cases "
                f"({time.time() - start:.0f}s)",
                flush=True,
            )
    audit = {
        "nonempty_cases_by_source": dict(nonempty_by_source),
        "hit_at_300_count_by_source": dict(source_hit_counts),
        "combined_nonempty_cases": sum(1 for ranked in combined_by_case if ranked),
    }
    return source_lists_by_case, combined_by_case, audit


def same_artist_case(case: dict[str, Any], track_artist: dict[str, str]) -> bool:
    gt_artist = track_artist.get(case["gt"], "")
    played_artists = {track_artist.get(track_id, "") for track_id in case["music_turns"]} - {""}
    return bool(gt_artist and gt_artist in played_artists)


def build_splits(cases: list[dict[str, Any]], track_artist: dict[str, str]) -> dict[str, list[int]]:
    same_idx = []
    diff_idx = []
    h7_idx = []
    for idx, case in enumerate(cases):
        if int(case["n_prior_music"]) == 7:
            h7_idx.append(idx)
        if same_artist_case(case, track_artist):
            same_idx.append(idx)
        else:
            diff_idx.append(idx)
    return {
        "all_dev": list(range(len(cases))),
        "h7": h7_idx,
        "same_artist": same_idx,
        "diff_artist": diff_idx,
    }


def hit_metrics_for_indices(
    cases: list[dict[str, Any]],
    ranked_by_case: list[list[str]],
    indices: list[int],
) -> dict[str, Any]:
    if not indices:
        return {
            "n": 0,
            "hit_at_20": 0.0,
            "hit_at_100": 0.0,
            "hit_at_300": 0.0,
            "hit_count_at_20": 0,
            "hit_count_at_100": 0,
            "hit_count_at_300": 0,
        }
    h20 = h100 = h300 = 0
    for idx in indices:
        gt = cases[idx]["gt"]
        ranked = ranked_by_case[idx]
        if gt in set(ranked[:20]):
            h20 += 1
        if gt in set(ranked[:100]):
            h100 += 1
        if gt in set(ranked[:300]):
            h300 += 1
    n = len(indices)
    return {
        "n": n,
        "hit_at_20": h20 / n,
        "hit_at_100": h100 / n,
        "hit_at_300": h300 / n,
        "hit_count_at_20": h20,
        "hit_count_at_100": h100,
        "hit_count_at_300": h300,
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def standalone_evaluation(
    cases: list[dict[str, Any]],
    c1_ranked: list[list[str]],
    baseline_pools: list[list[str]],
    buckets: list[str],
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    hit_metrics = {
        split_name: hit_metrics_for_indices(cases, c1_ranked, indices)
        for split_name, indices in splits.items()
    }

    baseline_pool_sets = [set(pool[:POOL_K]) for pool in baseline_pools]
    c1_sets = [set(ranked[:POOL_K]) for ranked in c1_ranked]

    h7_outside_cases = [
        idx
        for idx in splits["h7"]
        if cases[idx]["gt"] in c1_sets[idx] and cases[idx]["gt"] not in baseline_pool_sets[idx]
    ]
    outside_by_split = {}
    for split_name, indices in splits.items():
        outside_by_split[split_name] = sum(
            1
            for idx in indices
            if cases[idx]["gt"] in c1_sets[idx] and cases[idx]["gt"] not in baseline_pool_sets[idx]
        )

    bucket_recoveries = {}
    for bucket in ["HIT", "DEMOTED", "POOL_MISS", "UNREACHABLE"]:
        idxs = [idx for idx, label in enumerate(buckets) if label == bucket]
        recovered = sum(1 for idx in idxs if cases[idx]["gt"] in c1_sets[idx])
        bucket_recoveries[bucket] = {
            "n": len(idxs),
            "c1_hit_at_300_count": recovered,
            "c1_hit_at_300": recovered / max(len(idxs), 1),
        }

    overlap_counts = [len(c1_sets[idx] & baseline_pool_sets[idx]) for idx in range(len(cases))]
    overlap_fracs = [count / POOL_K for count in overlap_counts]

    return {
        "hit_metrics": hit_metrics,
        "unique_h7_gt_hits_outside_rrf_pool_at_300": len(h7_outside_cases),
        "h7_gt_hit_cases_outside_rrf_pool_at_300": h7_outside_cases,
        "outside_pool_hit_counts_by_split": outside_by_split,
        "bucket_recoveries": bucket_recoveries,
        "top300_overlap_with_weighted_rrf": {
            "count": percentile_summary([float(v) for v in overlap_counts]),
            "fraction": percentile_summary(overlap_fracs),
        },
    }


def rrf_scores_local(source_lists: dict[str, list[str]], weights: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for source_name, ranked in source_lists.items():
        weight = weights.get(source_name, 0.0)
        if weight == 0.0:
            continue
        for rank, track_id in enumerate(ranked, start=1):
            scores[track_id] = scores.get(track_id, 0.0) + weight / (RRF_K + rank)
    return scores


def sorted_score_keys(scores: dict[str, float], k: int = POOL_K) -> list[str]:
    return [
        track_id
        for track_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]


def fuse_pool(source_lists: dict[str, list[str]], c1_ranked: list[str], c1_weight: float) -> list[str]:
    fused_sources = dict(source_lists)
    fused_sources["C1"] = c1_ranked
    weights = dict(c3.SW)
    weights["C1"] = c1_weight
    return sorted_score_keys(rrf_scores_local(fused_sources, weights), POOL_K)


def ndcg_from_rank(rank: int, cutoff: int = TOP_K) -> float:
    if 1 <= rank <= cutoff:
        return float(1.0 / np.log2(rank + 1))
    return 0.0


def mean_values(values: list[float], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.asarray([values[idx] for idx in indices], dtype=np.float64).mean())


def fusion_metric_block(
    cases: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    fused_rows: list[dict[str, Any]],
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    base_ndcg = [float(row["ndcg_at_20"]) for row in baseline_rows]
    fused_ndcg = [float(row["ndcg_at_20"]) for row in fused_rows]
    out = {}
    for split_name, indices in splits.items():
        base_val = mean_values(base_ndcg, indices)
        fused_val = mean_values(fused_ndcg, indices)
        top1_changed = sum(
            1
            for idx in indices
            if baseline_rows[idx]["top1"] != fused_rows[idx]["top1"]
        )
        top20_overlap = [
            len(set(baseline_rows[idx]["top20"]) & set(fused_rows[idx]["top20"]))
            for idx in indices
        ]
        base_pool_hit = sum(1 for idx in indices if baseline_rows[idx]["gt_in_pool"])
        fused_pool_hit = sum(1 for idx in indices if fused_rows[idx]["gt_in_pool"])
        n = len(indices)
        out[split_name] = {
            "n": n,
            "baseline_ndcg_at_20": base_val,
            "fused_ndcg_at_20": fused_val,
            "delta_ndcg_at_20": fused_val - base_val,
            "baseline_pool_hit_at_300": base_pool_hit / max(n, 1),
            "fused_pool_hit_at_300": fused_pool_hit / max(n, 1),
            "delta_pool_hit_at_300": (fused_pool_hit - base_pool_hit) / max(n, 1),
            "top1_changed": top1_changed,
            "top1_churn_rate": top1_changed / max(n, 1),
            "top1_churn_equiv_per_80": top1_changed / max(n, 1) * 80.0,
            "top20_overlap": percentile_summary([float(v) for v in top20_overlap]),
        }
    return out


def run_fusion_evaluation(
    payload: dict[str, Any],
    r21_source: list[list[str]],
    r54_source: list[list[str]],
    case_index: dict[str, Any],
    c1_ranked: list[list[str]],
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    cases = payload["cases"]
    baseline_reproduction = frozen_lr.verify_oof_baseline_reproduction(cases)
    als_factors, _als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = frozen_lr.load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    ranker = lgb.Booster(model_file=str(frozen_lr.LR_MODEL))
    if ranker.num_feature() != len(frozen_lr.FEAT_ALL):
        raise RuntimeError(
            f"Frozen LR feature count mismatch: "
            f"model={ranker.num_feature()} expected={len(frozen_lr.FEAT_ALL)}"
        )

    print(f"{ts()} Scoring baseline and C1-fused pools with frozen LR...", flush=True)
    baseline_rows: list[dict[str, Any]] = []
    fused_rows_by_weight: dict[str, list[dict[str, Any]]] = {str(w): [] for w in FUSION_WEIGHTS}
    pool_hits_by_weight = {str(w): 0 for w in FUSION_WEIGHTS}
    baseline_pool_hits = 0
    score_start = time.time()

    for case_idx, case in enumerate(cases):
        source_lists = c3.make_source_lists(
            payload,
            r21_source,
            r54_source,
            case_index["als_source"],
            case_idx,
        )
        baseline_pool = case_index["baseline_pools"][case_idx]
        als_vec = case_index["als_session_vecs"][case_idx]
        r54_score_map = case_index["r54_scores"][case_idx]
        baseline_result = frozen_lr.score_pool(
            ranker,
            baseline_pool,
            source_lists,
            case,
            maps,
            als_factors,
            als_to_idx,
            als_vec,
            track_pop,
            max_pop,
            track_album,
            r54_score_map,
            rrf_full_rank_map=None,
        )
        baseline_rows.append(baseline_result)
        if baseline_result["gt_in_pool"]:
            baseline_pool_hits += 1

        for weight in FUSION_WEIGHTS:
            fused_pool = fuse_pool(source_lists, c1_ranked[case_idx], weight)
            if case["gt"] in set(fused_pool):
                pool_hits_by_weight[str(weight)] += 1
            fused_result = frozen_lr.score_pool(
                ranker,
                fused_pool,
                source_lists,
                case,
                maps,
                als_factors,
                als_to_idx,
                als_vec,
                track_pop,
                max_pop,
                track_album,
                r54_score_map,
                rrf_full_rank_map=None,
            )
            fused_rows_by_weight[str(weight)].append(fused_result)

        if (case_idx + 1) % 1000 == 0:
            print(
                f"  frozen LR scored {case_idx + 1}/{len(cases)} cases "
                f"({time.time() - score_start:.0f}s)",
                flush=True,
            )

    metrics_by_weight = {}
    for weight in FUSION_WEIGHTS:
        key = str(weight)
        block = fusion_metric_block(cases, baseline_rows, fused_rows_by_weight[key], splits)
        metrics_by_weight[key] = {
            "weight": weight,
            "pool_hit_at_300": {
                "baseline_count": baseline_pool_hits,
                "fused_count": pool_hits_by_weight[key],
                "baseline": baseline_pool_hits / len(cases),
                "fused": pool_hits_by_weight[key] / len(cases),
                "delta": (pool_hits_by_weight[key] - baseline_pool_hits) / len(cases),
            },
            "metrics": block,
        }

    best_weight_key = max(
        metrics_by_weight,
        key=lambda key: metrics_by_weight[key]["metrics"]["h7"]["delta_ndcg_at_20"],
    )
    del maps, track_pop, track_album, ranker, als_factors
    gc.collect()
    return {
        "baseline_reproduction": baseline_reproduction,
        "fusion_weights": FUSION_WEIGHTS,
        "metrics_by_weight": metrics_by_weight,
        "best_weight_by_h7_delta": best_weight_key,
    }


def evaluate_gates(standalone: dict[str, Any], fusion: dict[str, Any]) -> dict[str, Any]:
    best_weight = fusion["best_weight_by_h7_delta"]
    best = fusion["metrics_by_weight"][best_weight]
    best_metrics = best["metrics"]
    h7_delta = best_metrics["h7"]["delta_ndcg_at_20"]
    all_delta = best_metrics["all_dev"]["delta_ndcg_at_20"]
    same_delta = best_metrics["same_artist"]["delta_ndcg_at_20"]
    h7_churn_rate = best_metrics["h7"]["top1_churn_rate"]
    unique_h7 = standalone["unique_h7_gt_hits_outside_rrf_pool_at_300"]
    novelty_or_lift = unique_h7 >= GATE_H7_UNIQUE_OUTSIDE or h7_delta >= GATE_H7_NDCG_DELTA
    checks = {
        "unique_h7_outside_pool_ge_30": unique_h7 >= GATE_H7_UNIQUE_OUTSIDE,
        "best_fused_h7_delta_ge_0.003": h7_delta >= GATE_H7_NDCG_DELTA,
        "novelty_or_h7_lift": novelty_or_lift,
        "best_fused_all_dev_delta_nonnegative": all_delta >= 0.0,
        "best_fused_same_artist_delta_ge_minus_0.002": same_delta >= GATE_SAME_ARTIST_DELTA,
        "best_fused_h7_top1_churn_lte_1.5pct": h7_churn_rate <= GATE_TOP1_CHURN_RATE,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {
        "verdict": verdict,
        "gate_weight": float(best_weight),
        "checks": checks,
        "values": {
            "unique_h7_outside_pool": unique_h7,
            "best_h7_delta": h7_delta,
            "best_all_dev_delta": all_delta,
            "best_same_artist_delta": same_delta,
            "best_h7_top1_churn_rate": h7_churn_rate,
            "best_h7_top1_churn_equiv_per_80": best_metrics["h7"]["top1_churn_equiv_per_80"],
        },
    }


def fmt_rate(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: dict[str, Any]) -> None:
    gate = report["gates"]
    standalone = report["standalone"]
    fusion = report["fusion"]
    train = report["train_audit"]
    bucket = standalone["bucket_recoveries"]
    best_weight = fusion["best_weight_by_h7_delta"]
    best = fusion["metrics_by_weight"][best_weight]
    best_metrics = best["metrics"]
    overlap = standalone["top300_overlap_with_weighted_rrf"]

    lines = [
        "# R61 C1 Transition-Memory Probe Result",
        "",
        f"Created: {report['created_at']}",
        "",
        "## Gate Table",
        "",
        f"Verdict: **{gate['verdict']}**",
        "",
        "| Criterion | Threshold | Observed | Pass |",
        "|---|---:|---:|---:|",
        (
            f"| unique h7 GT outside current RRF pool@300 | >= {GATE_H7_UNIQUE_OUTSIDE} | "
            f"{gate['values']['unique_h7_outside_pool']} | "
            f"{gate['checks']['unique_h7_outside_pool_ge_30']} |"
        ),
        (
            f"| best fused h7 nDCG@20 delta (w={best_weight}) | >= +{GATE_H7_NDCG_DELTA:.3f} | "
            f"{gate['values']['best_h7_delta']:+.5f} | "
            f"{gate['checks']['best_fused_h7_delta_ge_0.003']} |"
        ),
        (
            f"| novelty OR h7 lift | pass one above | "
            f"{gate['checks']['novelty_or_h7_lift']} | "
            f"{gate['checks']['novelty_or_h7_lift']} |"
        ),
        (
            "| best fused all-dev nDCG@20 delta | >= 0.00000 | "
            f"{gate['values']['best_all_dev_delta']:+.5f} | "
            f"{gate['checks']['best_fused_all_dev_delta_nonnegative']} |"
        ),
        (
            "| best fused same-artist nDCG@20 delta | >= -0.00200 | "
            f"{gate['values']['best_same_artist_delta']:+.5f} | "
            f"{gate['checks']['best_fused_same_artist_delta_ge_minus_0.002']} |"
        ),
        (
            "| best fused h7 top-1 churn | <= 1.5% (1.2/80 blind-eq) | "
            f"{gate['values']['best_h7_top1_churn_rate']:.2%} "
            f"({gate['values']['best_h7_top1_churn_equiv_per_80']:.2f}/80) | "
            f"{gate['checks']['best_fused_h7_top1_churn_lte_1.5pct']} |"
        ),
        "",
        "## Standalone C1 Retrieval",
        "",
        "| Split | n | hit@20 | hit@100 | hit@300 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in ["all_dev", "h7", "same_artist", "diff_artist"]:
        metrics = standalone["hit_metrics"][split_name]
        lines.append(
            f"| {split_name} | {metrics['n']} | "
            f"{metrics['hit_at_20']:.4f} ({metrics['hit_count_at_20']}) | "
            f"{metrics['hit_at_100']:.4f} ({metrics['hit_count_at_100']}) | "
            f"{metrics['hit_at_300']:.4f} ({metrics['hit_count_at_300']}) |"
        )
    lines.extend(
        [
            "",
            "## Novelty And Buckets",
            "",
            "| Metric | Value |",
            "|---|---:|",
            (
                "| unique h7 GT hits outside current RRF pool@300 | "
                f"{standalone['unique_h7_gt_hits_outside_rrf_pool_at_300']} |"
            ),
            (
                f"| POOL_MISS recovered by C1 top-300 | "
                f"{bucket['POOL_MISS']['c1_hit_at_300_count']} / {bucket['POOL_MISS']['n']} |"
            ),
            (
                f"| UNREACHABLE recovered by C1 top-300 | "
                f"{bucket['UNREACHABLE']['c1_hit_at_300_count']} / {bucket['UNREACHABLE']['n']} |"
            ),
            (
                f"| same-artist outside-pool hits | "
                f"{standalone['outside_pool_hit_counts_by_split']['same_artist']} |"
            ),
            (
                f"| diff-artist outside-pool hits | "
                f"{standalone['outside_pool_hit_counts_by_split']['diff_artist']} |"
            ),
            "",
            "## Top-300 Overlap With Current Weighted RRF",
            "",
            "| Statistic | Overlap count | Overlap fraction |",
            "|---|---:|---:|",
            f"| mean | {overlap['count']['mean']:.2f} | {overlap['fraction']['mean']:.4f} |",
            f"| median | {overlap['count']['median']:.2f} | {overlap['fraction']['median']:.4f} |",
            f"| p10 | {overlap['count']['p10']:.2f} | {overlap['fraction']['p10']:.4f} |",
            f"| p90 | {overlap['count']['p90']:.2f} | {overlap['fraction']['p90']:.4f} |",
            "",
            "## Frozen-LR Fusion Sanity",
            "",
            "Predeclared C1 source weights: `{0.25, 0.5, 1.0}`. LR was loaded from "
            "`cache/r54_phase3_lr_model.txt` and was not retrained.",
            "",
            "| C1 weight | pool_hit@300 | delta | h7 nDCG delta | all-dev nDCG delta | same-artist delta | diff-artist delta | h7 top1 churn |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for weight in [str(w) for w in FUSION_WEIGHTS]:
        item = fusion["metrics_by_weight"][weight]
        metrics = item["metrics"]
        lines.append(
            f"| {weight} | {item['pool_hit_at_300']['fused']:.5f} | "
            f"{item['pool_hit_at_300']['delta']:+.5f} | "
            f"{metrics['h7']['delta_ndcg_at_20']:+.5f} | "
            f"{metrics['all_dev']['delta_ndcg_at_20']:+.5f} | "
            f"{metrics['same_artist']['delta_ndcg_at_20']:+.5f} | "
            f"{metrics['diff_artist']['delta_ndcg_at_20']:+.5f} | "
            f"{metrics['h7']['top1_churn_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Train Split Audit",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| dataset | `{train['dataset']}` |",
            f"| split | `{train['split']}` |",
            f"| train arrow path | `{train['train_arrow_path']}` |",
            f"| train rows | {train['train_rows']} |",
            f"| train unique session_ids | {train['train_unique_session_ids']} |",
            f"| dev unique session_ids | {train['dev_unique_session_ids']} |",
            (
                "| excluded train sessions overlapping dev | "
                f"{train['excluded_train_sessions_overlapping_dev']} |"
            ),
            f"| included train sessions | {train['included_train_sessions']} |",
            f"| sessions with >=2 music turns | {train['sessions_with_2plus_music']} |",
            f"| transition rows emitted | {train['transition_rows']} |",
            (
                "| transition rows with current/previous user text context | "
                f"{train['transition_rows_with_user_text_context']} |"
            ),
            f"| session_id schema | {train['session_id_schema']} |",
            f"| sample UUID match | {train['session_id_sample_uuid_match']} |",
            "",
            "## Implementation Notes",
            "",
            "- Train source is official `train` split only; script requests `split=\"train\"` and never loads Blind-A.",
            "- Dev sessions found in train are excluded before counting transitions.",
            "- Candidate generator is count-only: last-track counts, last-3 recency counts, last-artist counts, and artist/tag metadata backoff.",
            "- Metadata backoff uses existing cached `track_artist` and `track_tags` maps from the R12 payload.",
            "- No cached metadata-neighbor NN index was found, so `c1_metadata_neighbor` was skipped.",
            "- Played tracks are excluded from C1 outputs before ranking.",
            "",
        ]
    )
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.time()
    print("R61 / C1 transition-memory probe")
    print("=" * 70)
    print("Boundary: count-only; no neural model; no LR retraining; no blind data.", flush=True)

    payload, dev_session_ids = load_payload_for_dev_sessions()
    cases = payload["cases"]
    train_ds = load_train_split()
    transition_stats, train_audit = build_transition_memory(
        train_ds,
        dev_session_ids,
        payload["track_artist"],
        payload["track_tags"],
    )
    del train_ds
    gc.collect()

    c1_source_lists, c1_ranked, c1_audit = build_c1_dev_lists(
        cases,
        transition_stats,
        payload["track_artist"],
        payload["track_tags"],
    )
    del transition_stats
    gc.collect()

    print(f"{ts()} Loading current RRF/R54c pool baseline context...", flush=True)
    payload2, r21_source, r54_source, r54_scores = c3.load_payloads()
    if len(payload2["cases"]) != len(cases):
        raise RuntimeError("Payload length changed between loads")
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    case_index = c3.build_case_index(
        payload2,
        r21_source,
        r54_source,
        r54_scores,
        als_factors,
        als_track_ids,
        als_to_idx,
    )
    buckets, bucket_details = c3.load_bucket_labels(
        len(cases),
        case_index["baseline_hit"],
        case_index["source_union_has_gt"],
    )
    del als_factors, als_track_ids
    gc.collect()

    with open(DECOMP_JSON) as f:
        decomp = json.load(f)
    splits = build_splits(cases, payload["track_artist"])
    standalone = standalone_evaluation(
        cases,
        c1_ranked,
        case_index["baseline_pools"],
        buckets,
        splits,
    )

    fusion = run_fusion_evaluation(
        payload2,
        r21_source,
        r54_source,
        case_index,
        c1_ranked,
        splits,
    )
    gates = evaluate_gates(standalone, fusion)

    report = {
        "experiment": "R61 C1 transition-memory probe",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - start_time,
        "dev_only": True,
        "blind_access": False,
        "neural_model": False,
        "embedding_training": False,
        "lr_retraining": False,
        "hard_negatives": False,
        "response_work": False,
        "train_audit": train_audit,
        "c1_audit": c1_audit,
        "candidate_sources": C1_SOURCE_NAMES,
        "metadata_neighbor": {
            "used": False,
            "reason": "No cached metadata nearest-neighbor index found on disk; building a new index is out of scope.",
        },
        "baseline": {
            "r55_decomp_pool_hit_at_300": decomp["pool_hit"],
            "computed_weighted_rrf_pool_hit_at_300": float(np.mean(case_index["baseline_hit"])),
            "bucket_counts": dict(Counter(buckets)),
            "bucket_details": bucket_details,
        },
        "standalone": standalone,
        "fusion": fusion,
        "gates": gates,
        "per_case_minimal": [
            {
                "case_idx": idx,
                "session_id": case["session_id"],
                "n_prior_music": int(case["n_prior_music"]),
                "bucket": buckets[idx],
                "same_artist": same_artist_case(case, payload["track_artist"]),
                "gt": case["gt"],
                "c1_hit_at_300": case["gt"] in set(c1_ranked[idx][:POOL_K]),
                "gt_in_weighted_rrf_pool_at_300": bool(case_index["baseline_hit"][idx]),
                "c1_rank": (c1_ranked[idx].index(case["gt"]) + 1) if case["gt"] in c1_ranked[idx] else -1,
            }
            for idx, case in enumerate(cases)
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=json_default)
    write_markdown(report)

    best_weight = fusion["best_weight_by_h7_delta"]
    best_metrics = fusion["metrics_by_weight"][best_weight]["metrics"]
    pool_miss = standalone["bucket_recoveries"]["POOL_MISS"]
    unreachable = standalone["bucket_recoveries"]["UNREACHABLE"]
    print(f"\n{ts()} Saved JSON: {OUT_JSON}", flush=True)
    print(f"{ts()} Saved markdown: {OUT_MD}", flush=True)
    print(
        f"{ts()} Verdict: {gates['verdict']}  "
        f"unique_h7_outside_pool={standalone['unique_h7_gt_hits_outside_rrf_pool_at_300']}  "
        f"h7_delta_best_w{best_weight}={best_metrics['h7']['delta_ndcg_at_20']:+.5f}  "
        f"POOL_MISS={pool_miss['c1_hit_at_300_count']}/{pool_miss['n']}  "
        f"UNREACHABLE={unreachable['c1_hit_at_300_count']}/{unreachable['n']}  "
        f"same_delta={best_metrics['same_artist']['delta_ndcg_at_20']:+.5f}",
        flush=True,
    )
    print(f"Elapsed: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
