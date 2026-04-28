"""Offline post-rerank sweep for existing inference top-20 lists.

This is a zero-LLM-cost diagnostic: it reorders an existing prediction file
using deterministic metadata/history features, then evaluates nDCG@20.
"""
import argparse
import json
import math
import os
import random
import re
from pathlib import Path

from datasets import Dataset

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth, ndcg_at_k


DEFAULT_HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "want", "need",
    "looking", "like", "similar", "something", "some", "give", "play",
    "song", "songs", "track", "tracks", "music", "artist", "artists",
    "band", "bands", "more", "another", "please",
}


def latest_arrow(pattern: str) -> str:
    matches = sorted(DEFAULT_HF_CACHE.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No cached Arrow file matches {pattern}")
    return str(matches[-1])


def tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def metadata_tokens(meta: dict) -> set[str]:
    parts = [meta.get("track_name", ""), meta.get("artist_name", ""), meta.get("album_name", "")]
    tags = meta.get("tag_list") or []
    if isinstance(tags, list):
        parts.extend(tags[:12])
    elif tags:
        parts.append(tags)
    return tokens(" ".join(str(p) for p in parts if p))


def summarize(scores: list[float]) -> dict:
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    stderr = 0.0
    if n > 1:
        var = sum((x - mean) ** 2 for x in scores) / (n - 1)
        stderr = math.sqrt(var / n)
    return {
        "n": n,
        "ndcg20": mean,
        "stderr": stderr,
        "hits20": sum(x > 0 for x in scores),
    }


def load_cases(tid: str, sample_size: int | None, sample_seed: int) -> tuple[list[dict], dict]:
    with open(f"exp/inference/devset/{tid}.json", encoding="utf-8") as f:
        rows = json.load(f)
    if sample_size:
        cases = sorted({(str(r["session_id"]), str(r.get("user_id", ""))) for r in rows})
        sampled = set(random.Random(sample_seed).sample(cases, min(sample_size, len(cases))))
        rows = [
            r for r in rows
            if (str(r["session_id"]), str(r.get("user_id", ""))) in sampled
        ]
    return rows, {(
        str(r["session_id"]), str(r.get("user_id", "")), int(r["turn_number"])
    ): r for r in rows}


def build_feature_rows(rows: list[dict], ds: Dataset, metadata: dict[str, dict]) -> list[dict]:
    gt_map = build_ground_truth(ds)
    dev_cases = {(str(x["session_id"]), str(x.get("user_id", ""))): x for x in ds}
    feature_rows = []
    for r in rows:
        sid = str(r["session_id"])
        uid = str(r.get("user_id", ""))
        turn = int(r["turn_number"])
        item = dev_cases.get((sid, uid))
        if not item:
            continue
        gt_id = lookup_ground_truth(gt_map, sid, uid, turn)
        if not gt_id:
            continue
        conv = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        history = [c for c in conv if int(c["turn_number"]) < turn]
        user_now = next(
            c["content"] for c in conv
            if c["role"] == "user" and int(c["turn_number"]) == turn
        )
        user_all = " ".join(c["content"] for c in history if c["role"] == "user")
        music_hist = [c["content"].strip() for c in history if c["role"] == "music"]
        last_meta = metadata.get(music_hist[-1], {}) if music_hist else {}
        last_artist = str(last_meta.get("artist_name", "")).lower()
        last_tags = set(str(t).lower() for t in (last_meta.get("tag_list") or []))
        now_tokens = tokens(user_now)
        all_tokens = tokens(f"{user_all} {user_now}")

        candidates = []
        for rank, tid in enumerate(r["predicted_track_ids"][:20], 1):
            meta = metadata.get(str(tid), {})
            artist = str(meta.get("artist_name", "")).lower()
            tags = set(str(t).lower() for t in (meta.get("tag_list") or []))
            mtoks = metadata_tokens(meta)
            candidates.append({
                "tid": tid,
                "orig": 1.0 / rank,
                "now_overlap": len(now_tokens & mtoks),
                "all_overlap": len(all_tokens & mtoks),
                "same_artist": 1.0 if artist and artist == last_artist else 0.0,
                "tag_overlap": float(len(tags & last_tags)),
            })
        feature_rows.append({"row": r, "gt_id": gt_id, "candidates": candidates})
    return feature_rows


def rerank_ids(candidates: list[dict], weights: tuple[float, float, float, float, float]) -> list[str]:
    w_orig, w_now, w_all, w_artist, w_tag = weights
    ranked = sorted(
        candidates,
        key=lambda x: (
            w_orig * x["orig"]
            + w_now * x["now_overlap"]
            + w_all * x["all_overlap"]
            + w_artist * x["same_artist"]
            + w_tag * x["tag_overlap"]
        ),
        reverse=True,
    )
    return [x["tid"] for x in ranked]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--output_tid", default=None)
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated weights: orig,now_overlap,all_overlap,same_artist,tag_overlap",
    )
    args = parser.parse_args()

    rows, _ = load_cases(args.tid, args.sample_size, args.sample_seed)
    ds = Dataset.from_file(cached_test_arrow_path())
    track_ds = Dataset.from_file(latest_arrow(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"
    ))
    metadata = {str(x["track_id"]): x for x in track_ds}
    feature_rows = build_feature_rows(rows, ds, metadata)

    base_scores = [
        ndcg_at_k(fr["row"]["predicted_track_ids"], fr["gt_id"])
        for fr in feature_rows
    ]
    base_summary = summarize(base_scores)
    print(
        f"base n={base_summary['n']} nDCG={base_summary['ndcg20']:.4f} "
        f"hits={base_summary['hits20']}"
    )

    if args.weights:
        weights = tuple(float(x) for x in args.weights.split(","))
        if len(weights) != 5:
            raise ValueError("--weights must contain exactly five comma-separated floats")
        scores = [
            ndcg_at_k(rerank_ids(fr["candidates"], weights), fr["gt_id"])
            for fr in feature_rows
        ]
        best = (sum(scores) / len(scores) if scores else 0.0, weights, scores)
    else:
        grids = [
            [0.25, 0.5, 1.0, 2.0, 4.0],
            [0.0, 0.5, 1.0, 2.0, 4.0],
            [0.0, 0.25, 0.5, 1.0],
            [0.0, 0.5, 1.0, 2.0, 4.0],
            [0.0, 0.05, 0.1, 0.25, 0.5],
        ]
        best = (base_summary["ndcg20"], (1.0, 0.0, 0.0, 0.0, 0.0), base_scores)
        for w_orig in grids[0]:
            for w_now in grids[1]:
                for w_all in grids[2]:
                    for w_artist in grids[3]:
                        for w_tag in grids[4]:
                            weights = (w_orig, w_now, w_all, w_artist, w_tag)
                            scores = [
                                ndcg_at_k(rerank_ids(fr["candidates"], weights), fr["gt_id"])
                                for fr in feature_rows
                            ]
                            avg = sum(scores) / len(scores) if scores else 0.0
                            if avg > best[0]:
                                best = (avg, weights, scores)

    best_summary = summarize(best[2])
    print(
        f"best weights={best[1]} nDCG={best_summary['ndcg20']:.4f} "
        f"stderr={best_summary['stderr']:.4f} hits={best_summary['hits20']}"
    )

    if args.output_tid:
        out_rows = []
        for fr in feature_rows:
            row = dict(fr["row"])
            row["predicted_track_ids"] = rerank_ids(fr["candidates"], best[1])
            out_rows.append(row)
        os.makedirs("exp/inference/devset", exist_ok=True)
        out = f"exp/inference/devset/{args.output_tid}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(out_rows, f, ensure_ascii=False, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
