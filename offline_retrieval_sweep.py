"""Fast no-LLM retrieval sweeps on devset.

This script is deliberately local/offline: it loads cached Arrow files and the
cached BM25 index, then evaluates simple retrieval/fusion strategies. Use it to
screen ideas before spending on LLM reranking or response generation.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import bm25s
from datasets import Dataset

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth, ndcg_at_k
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever


DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"
DEFAULT_BM25_INDEX = "cache/bm25/track_name_artist_name_album_name_tag_list"


def latest_arrow(pattern: str) -> str:
    matches = sorted(DEFAULT_HF_CACHE.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cached Arrow file matches {pattern}")
    return str(matches[-1])


def load_devset() -> Dataset:
    path = cached_test_arrow_path()
    if not path:
        raise FileNotFoundError("Cached devset test arrow not found")
    return Dataset.from_file(path)


def load_track_metadata() -> dict[str, dict]:
    path = latest_arrow(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"
    )
    ds = Dataset.from_file(path)
    return {str(item["track_id"]): item for item in ds}


class CachedBM25:
    def __init__(self, index_dir: str = DEFAULT_BM25_INDEX):
        self.model = bm25s.BM25.load(index_dir, load_corpus=True)
        with open(os.path.join(index_dir, "track_ids.json"), encoding="utf-8") as f:
            self.track_ids = json.load(f)

    def retrieve(self, query: str, topk: int) -> list[str]:
        tokens = bm25s.tokenize([query.lower()])
        results = self.model.retrieve(tokens, k=topk, return_as="tuple")
        return [self.track_ids[item["id"]] for item in results.documents[0]]

    def retrieve_batch(self, queries: list[str], topk: int) -> list[list[str]]:
        if not queries:
            return []
        tokens = bm25s.tokenize([q.lower() for q in queries])
        results = self.model.retrieve(tokens, k=topk, return_as="tuple")
        return [
            [self.track_ids[item["id"]] for item in results.documents[i]]
            for i in range(len(queries))
        ]


def meta_text(track_id: str, metadata: dict[str, dict], include_track_name: bool = True) -> str:
    meta = metadata.get(str(track_id), {})
    parts = []
    if include_track_name:
        parts.append(str(meta.get("track_name", "")))
    parts.append(str(meta.get("artist_name", "")))
    parts.append(str(meta.get("album_name", "")))
    tags = meta.get("tag_list", [])
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags[:8])
    elif tags:
        parts.append(str(tags))
    return " ".join(p for p in parts if p)


def last_turn(item: dict) -> tuple[int, str, list[dict], str | None]:
    rows = sorted(item["conversations"], key=lambda r: int(r["turn_number"]))
    user_rows = [r for r in rows if r["role"] == "user"]
    turn = int(user_rows[-1]["turn_number"])
    query = user_rows[-1]["content"]
    history = [r for r in rows if int(r["turn_number"]) < turn]
    gt = next(
        (r["content"].strip() for r in rows if r["role"] == "music" and int(r["turn_number"]) == turn),
        None,
    )
    return turn, query, history, gt


def query_parts(history: list[dict], user_query: str, metadata: dict[str, dict], mode: str) -> list[str]:
    parts = []
    user_turns = [h["content"] for h in history if h["role"] == "user"]
    music_turns = [h["content"].strip() for h in history if h["role"] == "music"]

    if mode in ("current", "recency", "all_user", "full"):
        if mode == "current":
            parts.extend([user_query] * 3)
        elif mode == "all_user":
            parts.extend(user_turns + [user_query] * 2)
        else:
            for i, turn in enumerate(user_turns):
                parts.append(turn)
                if i == len(user_turns) - 1:
                    parts.append(turn)
            parts.extend([user_query] * 3)

    if mode in ("music_meta", "full", "last_music_meta"):
        selected = music_turns[-1:] if mode == "last_music_meta" else music_turns
        for tid in selected:
            parts.append(meta_text(tid, metadata, include_track_name=(mode == "last_music_meta")))

    return [p for p in parts if p]


def rrf(lists: list[list[str]], k: int = 60, topk: int = 20) -> list[str]:
    scores = {}
    for ranked in lists:
        for rank, tid in enumerate(ranked):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]


def bm25_modes_for_strategy(strategy: str) -> list[str]:
    """Map strategy names to one or more BM25 query builders."""
    combo_modes = {
        "current_plus_last_music": ["current", "last_music_meta"],
        "recency_plus_last_music": ["recency", "last_music_meta"],
        "full_plus_last_music": ["full", "last_music_meta"],
        "current_full_last_music": ["current", "full", "last_music_meta"],
    }
    if strategy in combo_modes:
        return combo_modes[strategy]
    if strategy in ("track_neighbors", "full_plus_neighbors", "user_only_plus_neighbors"):
        return []
    return [strategy]


def score_strategy(
    ds,
    bm25: CachedBM25,
    metadata: dict[str, dict],
    track_sim: TrackSimilarityRetriever | None,
    strategy: str,
    candidate_k: int,
    session_ids: set[str] | None,
    gt_map,
) -> tuple[list[float], list[dict]]:
    cases = []
    queries = []
    for item in ds:
        sid = str(item["session_id"])
        if session_ids and sid not in session_ids:
            continue
        turn, user_query, history, _gt = last_turn(item)
        gt = lookup_ground_truth(gt_map, sid, item.get("user_id"), turn)
        if not gt:
            continue

        music_turns = [h["content"].strip() for h in history if h["role"] == "music"]
        q_list = []
        if strategy == "full_plus_neighbors":
            q_list = [" ".join(query_parts(history, user_query, metadata, "full"))]
        elif strategy == "user_only_plus_neighbors":
            q_list = [" ".join(query_parts(history, user_query, metadata, "recency"))]
        elif strategy != "track_neighbors":
            q_list = [
                " ".join(query_parts(history, user_query, metadata, mode))
                for mode in bm25_modes_for_strategy(strategy)
            ]
        q_list = [q for q in q_list if q]

        cases.append((sid, item.get("user_id"), turn, gt, music_turns, q_list))
        queries.extend(q_list)

    bm25_results = iter(bm25.retrieve_batch(queries, topk=candidate_k))
    scores = []
    rows = []
    for sid, user_id, turn, gt, music_turns, q_list in cases:
        lists = []
        for _ in q_list:
            lists.append(next(bm25_results))
        if strategy in ("track_neighbors", "full_plus_neighbors", "user_only_plus_neighbors"):
            if track_sim and music_turns:
                lists.append(track_sim.track_id_to_neighbors(music_turns[-1], topk=candidate_k))

        pred = rrf(lists, topk=20) if len(lists) > 1 else (lists[0][:20] if lists else [])
        score = ndcg_at_k(pred, gt)
        scores.append(score)
        rows.append(
            {
                "session_id": sid,
                "user_id": user_id,
                "turn_number": turn,
                "predicted_track_ids": pred,
                "predicted_response": "",
            }
        )
    return scores, rows


def summarize(scores: list[float]) -> dict:
    n = len(scores)
    avg = sum(scores) / n if n else 0.0
    hits = sum(1 for x in scores if x > 0)
    if n > 1:
        var = sum((x - avg) ** 2 for x in scores) / (n - 1)
        stderr = math.sqrt(var / n)
    else:
        stderr = 0.0
    return {"n": n, "ndcg20": avg, "stderr": stderr, "hits20": hits, "hit_rate20": hits / n if n else 0.0}


def load_session_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        # Subset files may be either one session id per line or TSV
        # session_id/user_id pairs. Only the first column is the session key.
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_ids", default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--candidate_k", type=int, default=100)
    parser.add_argument("--write_best_tid", default=None)
    parser.add_argument("--no_track_sim", action="store_true")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names to run. Defaults to all strategies.",
    )
    args = parser.parse_args()

    ds = load_devset()
    session_ids = load_session_ids(args.session_ids)
    if args.sample_size:
        all_sessions = [str(item["session_id"]) for item in ds]
        sampled = set(random.Random(args.sample_seed).sample(all_sessions, min(args.sample_size, len(all_sessions))))
        session_ids = sampled if session_ids is None else session_ids & sampled

    metadata = load_track_metadata()
    gt_map = build_ground_truth(ds)
    bm25 = CachedBM25()
    track_sim = None if args.no_track_sim else TrackSimilarityRetriever(cache_dir="./cache")

    default_strategies = [
        "current",
        "all_user",
        "recency",
        "music_meta",
        "last_music_meta",
        "full",
        "current_plus_last_music",
        "recency_plus_last_music",
        "full_plus_last_music",
        "current_full_last_music",
        "track_neighbors",
        "user_only_plus_neighbors",
        "full_plus_neighbors",
    ]
    strategies = (
        [s.strip() for s in args.strategies.split(",") if s.strip()]
        if args.strategies
        else default_strategies
    )
    unknown = sorted(set(strategies) - set(default_strategies))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")

    results = {}
    best_name, best_rows, best_ndcg = None, [], -1.0
    for strategy in strategies:
        scores, rows = score_strategy(
            ds, bm25, metadata, track_sim, strategy, args.candidate_k, session_ids, gt_map
        )
        summary = summarize(scores)
        results[strategy] = summary
        print(
            f"{strategy:24s} n={summary['n']:4d} "
            f"nDCG={summary['ndcg20']:.4f} stderr={summary['stderr']:.4f} "
            f"hit={summary['hits20']}/{summary['n']}"
        )
        if summary["ndcg20"] > best_ndcg:
            best_name, best_rows, best_ndcg = strategy, rows, summary["ndcg20"]

    os.makedirs("exp/eval", exist_ok=True)
    with open("exp/eval/offline_retrieval_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.write_best_tid and best_name:
        os.makedirs("exp/inference/devset", exist_ok=True)
        out = f"exp/inference/devset/{args.write_best_tid}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(best_rows, f, ensure_ascii=False, indent=2)
        print(f"Best strategy: {best_name} ({best_ndcg:.4f}); wrote {out}")


if __name__ == "__main__":
    main()
