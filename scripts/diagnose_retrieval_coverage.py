"""Offline retrieval coverage diagnostics for devset artifacts.

This is intentionally spend-safe: it sets MCRS_REQUIRE_LLM_CACHE=1 before any
optional QR/generative calls, so cached LLM outputs may be reused but cache
misses cannot trigger paid API calls.
"""
# ruff: noqa: T201
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from offline_retrieval_sweep import CachedBM25, load_track_metadata, meta_text, rrf
from mcrs.query_reformulation import QueryReformulator
from mcrs.retrieval_modules.generative import GenerativeRetriever
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever


DEPTHS = (20, 50, 100, 200, 500)


def norm_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def contains_phrase(haystack: str, phrase: str | None) -> bool:
    if not phrase:
        return False
    return phrase.lower() in haystack.lower()


def latest_arrow(pattern: str) -> str:
    matches = sorted(Path.home().glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cached Arrow file matches {pattern}")
    return str(matches[-1])


def load_devset() -> Dataset:
    path = cached_test_arrow_path()
    if not path:
        raise FileNotFoundError("Cached devset test arrow not found")
    return Dataset.from_file(path)


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["session_id"]), "" if row.get("user_id") is None else str(row.get("user_id"))


def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dev_index(ds: Dataset) -> dict[tuple[str, str], dict[str, Any]]:
    index = {}
    for item in ds:
        uid = "" if item.get("user_id") is None else str(item.get("user_id"))
        index[(str(item["session_id"]), uid)] = item
    return index


def conversation_parts(item: dict[str, Any], turn: int, metadata: dict[str, dict]) -> dict[str, Any]:
    rows = sorted(item["conversations"], key=lambda r: int(r["turn_number"]))
    before = [r for r in rows if int(r["turn_number"]) < turn]
    current_user = next(
        (str(r["content"]) for r in rows if r["role"] == "user" and int(r["turn_number"]) == turn),
        "",
    )
    user_turns = [str(r["content"]) for r in before if r["role"] == "user"]
    music_turns = [str(r["content"]).strip() for r in before if r["role"] == "music"]
    last_music = music_turns[-1] if music_turns else None
    prior_music_meta = " ".join(meta_text(tid, metadata) for tid in music_turns)
    full_text = " ".join(user_turns + [current_user, prior_music_meta])
    return {
        "before": before,
        "current_user": current_user,
        "user_text": " ".join(user_turns + [current_user]),
        "full_text": full_text,
        "music_turns": music_turns,
        "last_music": last_music,
    }


def rank_of(track_id: str | None, ranked: list[str], max_k: int | None = None) -> int | None:
    if not track_id:
        return None
    scan = ranked if max_k is None else ranked[:max_k]
    try:
        return scan.index(track_id) + 1
    except ValueError:
        return None


def update_depth_hits(counter: Counter[str], prefix: str, ranked: list[str], gt_id: str) -> None:
    for depth in DEPTHS:
        if rank_of(gt_id, ranked, depth) is not None:
            counter[f"{prefix}@{depth}"] += 1


def safe_cached_v23_components(
    parts: dict[str, Any],
    bm25: CachedBM25,
    track_sim: TrackSimilarityRetriever | None,
    topk: int,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Reconstruct v23 component candidates using cache-hit-only LLM calls."""
    meta: dict[str, Any] = {"cache_ok": True, "cache_errors": []}
    lists: dict[str, list[str]] = {}
    try:
        nlq = QueryReformulator(mode="nlq", fallback_on_error=False).reformulate(
            parts["before"], parts["current_user"]
        )
        entity = QueryReformulator(mode="entity", fallback_on_error=False).reformulate(
            parts["before"], parts["current_user"]
        )
        gen = GenerativeRetriever().get_suggestions(parts["before"], parts["current_user"])
    except Exception as exc:  # cache miss or unreadable cached output
        meta["cache_ok"] = False
        meta["cache_errors"].append(str(exc))
        return lists, meta

    lists["v23_nlq_bm25"] = bm25.retrieve(nlq, topk)
    lists["v23_entity_bm25"] = bm25.retrieve(entity, topk)
    gen_lists = []
    for gq in GenerativeRetriever().suggestions_to_queries(gen):
        gen_lists.append(bm25.retrieve(gq, 5))
    if gen_lists:
        lists["v23_generative_bm25_top5"] = rrf(gen_lists, topk=topk)
    if track_sim and parts["last_music"]:
        lists["v23_track_sim_30"] = track_sim.track_id_to_neighbors(parts["last_music"], topk=30)

    merge_inputs = [lists[name] for name in ("v23_nlq_bm25", "v23_entity_bm25") if name in lists]
    merge_inputs.extend(gen_lists)
    if "v23_track_sim_30" in lists:
        merge_inputs.append(lists["v23_track_sim_30"])
    if merge_inputs:
        lists["v23_rrf_100"] = rrf(merge_inputs, topk=100)

    meta["nlq_query"] = nlq
    meta["entity_query"] = entity
    meta["gen_suggestions"] = len(gen)
    return lists, meta


def bucket_pool_rank(rank: int | None) -> str:
    if rank is None:
        return "no_hit"
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    if rank <= 20:
        return "rank_6_20"
    if rank <= 50:
        return "rank_21_50"
    return "rank_51_plus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="exp/inference/devset/echo_v23_pool50_s200.json")
    parser.add_argument("--output_json", default="exp/eval/retrieval_coverage_diagnostic.json")
    parser.add_argument("--miss_csv", default="exp/eval/retrieval_coverage_misses.csv")
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--skip_cached_llm_components", action="store_true")
    args = parser.parse_args()

    os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"

    rows = load_cases(args.artifact)
    ds = load_devset()
    gt_map = build_ground_truth(ds)
    dev_index = build_dev_index(ds)
    metadata = load_track_metadata()
    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir="./cache")

    counters: Counter[str] = Counter()
    pool_hist: Counter[str] = Counter()
    component_hits: Counter[str] = Counter()
    miss_reason: Counter[str] = Counter()
    cache_errors: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    miss_rows: list[dict[str, Any]] = []

    for row in rows:
        sid, uid = case_key(row)
        turn = int(row["turn_number"])
        item = dev_index[(sid, uid)]
        gt_id = lookup_ground_truth(gt_map, sid, row.get("user_id"), turn)
        if not gt_id:
            counters["missing_gt"] += 1
            continue

        gt_meta = metadata.get(gt_id, {})
        parts = conversation_parts(item, turn, metadata)
        pool = row.get("candidate_pool_track_ids", [])
        pred = row.get("predicted_track_ids", [])
        pool_rank = rank_of(gt_id, pool, 50)
        pred_rank = rank_of(gt_id, pred, 20)
        pool_hist[bucket_pool_rank(pool_rank)] += 1
        if pred_rank:
            counters["pred_hit20"] += 1
        if pool_rank:
            counters["pool_hit50"] += 1

        text_all = parts["full_text"]
        text_user = parts["user_text"]
        gt_artist = str(gt_meta.get("artist_name") or "")
        gt_title = str(gt_meta.get("track_name") or "")
        gt_tags = gt_meta.get("tag_list") or []
        gt_tag_tokens = norm_tokens(" ".join(str(t) for t in gt_tags[:12]))
        full_tokens = norm_tokens(text_all)
        artist_in_user = contains_phrase(text_user, gt_artist)
        title_in_user = contains_phrase(text_user, gt_title)
        artist_in_full = contains_phrase(text_all, gt_artist)
        title_in_full = contains_phrase(text_all, gt_title)
        tag_overlap = len(gt_tag_tokens & full_tokens)

        exact_gt_query = bm25.retrieve(f"{gt_title} {gt_artist}", args.topk)
        current_bm25 = bm25.retrieve(parts["current_user"], args.topk)
        user_bm25 = bm25.retrieve(text_user, args.topk)
        full_bm25 = bm25.retrieve(text_all, args.topk)
        last_music_bm25 = (
            bm25.retrieve(meta_text(parts["last_music"], metadata), args.topk)
            if parts["last_music"] else []
        )
        neighbors = (
            track_sim.track_id_to_neighbors(parts["last_music"], topk=min(args.topk, 500))
            if parts["last_music"] else []
        )

        probes = {
            "exact_gt_metadata_bm25": exact_gt_query,
            "current_user_bm25": current_bm25,
            "all_user_bm25": user_bm25,
            "full_history_bm25": full_bm25,
            "last_music_meta_bm25": last_music_bm25,
            "last_track_neighbors": neighbors,
        }
        if not args.skip_cached_llm_components:
            component_lists, component_meta = safe_cached_v23_components(
                parts, bm25, track_sim, args.topk
            )
            probes.update(component_lists)
            if not component_meta["cache_ok"]:
                cache_errors[component_meta["cache_errors"][0]] += 1

        for name, ranked in probes.items():
            update_depth_hits(component_hits, name, ranked, gt_id)

        if pool_rank is None:
            if rank_of(gt_id, exact_gt_query, 20):
                miss_reason["index_can_exact_match_gt_top20"] += 1
            if artist_in_user or title_in_user:
                reason = "explicit_artist_or_title_in_user_text"
            elif artist_in_full or title_in_full:
                reason = "artist_or_title_only_in_history"
            elif tag_overlap >= 2:
                reason = "tag_overlap_only"
            else:
                reason = "no_obvious_textual_signal"
            miss_reason[reason] += 1

            row_out = {
                "session_id": sid,
                "user_id": uid,
                "turn_number": turn,
                "current_user": parts["current_user"],
                "gt_track_id": gt_id,
                "gt_track_name": gt_title,
                "gt_artist_name": gt_artist,
                "gt_tags": "|".join(str(t) for t in gt_tags[:8]) if isinstance(gt_tags, list) else str(gt_tags),
                "artist_in_user": artist_in_user,
                "title_in_user": title_in_user,
                "artist_in_full": artist_in_full,
                "title_in_full": title_in_full,
                "tag_overlap": tag_overlap,
                "exact_gt_bm25_rank": rank_of(gt_id, exact_gt_query, args.topk),
                "current_user_bm25_rank": rank_of(gt_id, current_bm25, args.topk),
                "all_user_bm25_rank": rank_of(gt_id, user_bm25, args.topk),
                "full_history_bm25_rank": rank_of(gt_id, full_bm25, args.topk),
                "last_music_meta_bm25_rank": rank_of(gt_id, last_music_bm25, args.topk),
                "last_track_neighbor_rank": rank_of(gt_id, neighbors, min(args.topk, 500)),
            }
            miss_rows.append(row_out)
            if len(examples[reason]) < 5:
                examples[reason].append(row_out)

    n = len(rows) - counters["missing_gt"]
    summary = {
        "artifact": args.artifact,
        "n": n,
        "pool_rank_hist": dict(pool_hist),
        "pred_hit20": counters["pred_hit20"],
        "pool_hit50": counters["pool_hit50"],
        "pred_hit20_rate": counters["pred_hit20"] / n if n else 0.0,
        "pool_hit50_rate": counters["pool_hit50"] / n if n else 0.0,
        "component_hits": dict(component_hits),
        "component_hit_rates": {
            key: val / n for key, val in sorted(component_hits.items())
        },
        "miss_reason_counts": dict(miss_reason),
        "cache_error_counts": dict(cache_errors),
        "examples": examples,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if miss_rows:
        os.makedirs(os.path.dirname(args.miss_csv), exist_ok=True)
        with open(args.miss_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(miss_rows[0].keys()))
            writer.writeheader()
            writer.writerows(miss_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved miss CSV: {args.miss_csv}")


if __name__ == "__main__":
    main()
