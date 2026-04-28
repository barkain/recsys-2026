#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""400-session deterministic R3 confirmation.

No LLM/API calls. Builds a fresh devset slice excluding the clean v23 200 rows
where possible, retrieves deterministic B/C/D sources, and evaluates the same
8-feature Powell postranker over a weighted-RRF top-50 pool.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, STOPWORDS

N_CONFIRM = 400
BM25_K = 500
TRACK_K = 200
POOL_K = 50
OUT_FILE = "exp/eval/r3_confirm_400_deterministic.json"
CACHE_FILE = "exp/eval/_r3_confirm_400_deterministic.pkl"
V23_ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
TOKEN_RE = re.compile(r"[a-z0-9']+")


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(str(text).lower()) if len(t) > 2 and t not in STOPWORDS}


def dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        tid = str(item)
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def latest_user_turn(item: dict[str, Any]) -> tuple[int, str, list[dict[str, Any]]]:
    rows = sorted(item["conversations"], key=lambda r: int(r["turn_number"]))
    user_rows = [r for r in rows if r["role"] == "user"]
    turn = int(user_rows[-1]["turn_number"])
    query = str(user_rows[-1]["content"])
    history = [r for r in rows if int(r["turn_number"]) < turn]
    return turn, query, history


def load_cases() -> list[dict[str, Any]]:
    arrow = cached_test_arrow_path()
    if not arrow:
        raise FileNotFoundError("Cached devset test arrow not found")
    ds = Dataset.from_file(arrow)
    gt_map = build_ground_truth(ds)
    exclude: set[str] = set()
    artifact = Path(V23_ARTIFACT)
    if artifact.exists():
        with open(artifact, encoding="utf-8") as f:
            exclude = {str(row["session_id"]) for row in json.load(f)}

    cases = []
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        turn, user_query, history = latest_user_turn(item)
        gt = lookup_ground_truth(gt_map, sid, uid, turn)
        if not gt:
            continue
        music_turns = [str(row["content"]).strip() for row in history if row["role"] == "music"]
        cases.append({
            "session_id": sid,
            "user_id": uid,
            "turn_number": turn,
            "user_query": user_query,
            "history": history,
            "music_turns": music_turns,
            "gt": str(gt),
            "excluded_from_v23_200": sid in exclude,
        })

    fresh = [c for c in cases if not c["excluded_from_v23_200"]]
    source = fresh if len(fresh) >= N_CONFIRM else cases
    selected = sorted(source, key=lambda c: stable_hash(c["session_id"]))[:N_CONFIRM]
    if len(selected) < N_CONFIRM:
        raise RuntimeError(f"Only found {len(selected)} eligible cases")
    return selected


def build_or_load_payload() -> dict[str, Any]:
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        print(f"loading cached payload {CACHE_FILE}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)  # noqa: S301 - local cache produced by this script

    t0 = time.time()
    print("loading cases + metadata", flush=True)
    cases = load_cases()
    metadata = load_track_metadata()

    queries_b = []
    queries_c = []
    for c in cases:
        q_b = " ".join(query_parts(c["history"], c["user_query"], metadata, "last_music_meta"))
        q_c = " ".join(query_parts(c["history"], c["user_query"], metadata, "full"))
        queries_b.append(q_b or c["user_query"])
        queries_c.append(q_c or c["user_query"])

    print("loading BM25 cache", flush=True)
    bm25 = CachedBM25()
    print(f"retrieving B last_music_meta@{BM25_K}", flush=True)
    src_b = bm25.retrieve_batch(queries_b, topk=BM25_K)
    print(f"retrieving C full_history@{BM25_K}", flush=True)
    src_c = bm25.retrieve_batch(queries_c, topk=BM25_K)

    print(f"retrieving D track_neighbors@{TRACK_K}", flush=True)
    track_sim = TrackSimilarityRetriever(cache_dir="./cache")
    src_d = []
    for i, c in enumerate(cases):
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        src_d.append(dedupe(track_sim.track_id_to_neighbors(anchor, topk=TRACK_K)) if anchor else [])
        if (i + 1) % 100 == 0:
            print(f"  neighbors {i + 1}/{len(cases)}", flush=True)

    print("precomputing metadata feature maps", flush=True)
    all_tracks: set[str] = set()
    for i, c in enumerate(cases):
        all_tracks.update(src_b[i])
        all_tracks.update(src_c[i])
        all_tracks.update(src_d[i])
        all_tracks.update(c["music_turns"])
    track_artist: dict[str, str] = {}
    track_tags: dict[str, set[str]] = {}
    track_title_toks: dict[str, set[str]] = {}
    track_artist_toks: dict[str, set[str]] = {}
    track_meta_toks: dict[str, set[str]] = {}
    for tid in all_tracks:
        meta = metadata.get(str(tid), {})
        artist = str(meta.get("artist_name", "")).lower().strip()
        raw_tags = meta.get("tag_list") or []
        tags = {str(t).lower().strip() for t in raw_tags if str(t).strip()} if isinstance(raw_tags, list) else set()
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        meta_parts = [title, str(meta.get("artist_name", "")), album]
        if isinstance(raw_tags, list):
            meta_parts.extend(str(t) for t in raw_tags[:12])
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = tokens(title)
        track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
        track_meta_toks[tid] = tokens(" ".join(meta_parts))

    payload = {
        "cases": cases,
        "src_b": [dedupe(x) for x in src_b],
        "src_c": [dedupe(x) for x in src_c],
        "src_d": src_d,
        "track_artist": track_artist,
        "track_tags": track_tags,
        "track_title_toks": track_title_toks,
        "track_artist_toks": track_artist_toks,
        "track_meta_toks": track_meta_toks,
        "elapsed_build_sec": time.time() - t0,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def build_pool_features(payload: dict[str, Any], source_weights: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cases = payload["cases"]
    src_b = payload["src_b"]
    src_c = payload["src_c"]
    src_d = payload["src_d"]
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    n = len(cases)
    f_count = len(FEATURE_NAMES)
    X = np.zeros((n, POOL_K, f_count), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(cases):
        scores: dict[str, float] = {}
        for seq, weight in ((src_b[i], source_weights[0]), (src_c[i], source_weights[1]), (src_d[i], source_weights[2])):
            if weight == 0:
                continue
            for rank, tid in enumerate(seq, start=1):
                scores[tid] = scores.get(tid, 0.0) + weight / (60.0 + rank)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:POOL_K]
        sizes[i] = len(ranked)

        user_messages = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tokens = tokens(user_messages[-1]) if user_messages else set()
        all_user_tokens = tokens(" ".join(user_messages))
        played_set = set(played)
        last_artist = track_artist.get(played[-1], "") if played else ""
        last_tags = track_tags.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), track_artist.get(tid, ""), track_tags.get(tid, set())) for j, tid in enumerate(reversed(played))]

        for rank, tid in enumerate(ranked, start=1):
            cand_artist = track_artist.get(tid, "")
            cand_tags = track_tags.get(tid, set())
            row = X[i, rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
            if cand_tags or last_tags:
                row[2] = len(cand_tags & last_tags) / len(cand_tags | last_tags)
            row[3] = float(len(track_artist_toks.get(tid, set()) & now_tokens))
            row[4] = float(len(track_title_toks.get(tid, set()) & now_tokens))
            row[5] = float(len(track_meta_toks.get(tid, set()) & all_user_tokens))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for weight, p_artist, p_tags in prior:
                artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                tag_match = len(cand_tags & p_tags) / len(cand_tags | p_tags) if (cand_tags or p_tags) else 0.0
                rec += weight * (artist_match + tag_match)
            row[7] = rec
        if c["gt"] in ranked:
            gt_idx[i] = ranked.index(c["gt"])
    return X, gt_idx, sizes


def split_order(sessions: list[str], seed: int) -> list[int]:
    return sorted(range(len(sessions)), key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))


def cv_folds(sessions: list[str], seed: int, k: int = 5) -> list[np.ndarray]:
    order = split_order(sessions, seed)
    folds = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(idx)
    return [np.asarray(f, dtype=np.int64) for f in folds]


def vec_ndcg(X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray, weights: np.ndarray, idx: np.ndarray) -> float:
    pool_axis = np.arange(X.shape[1])[None, :]
    valid_pool = pool_axis < sizes[idx, None]
    scores = X[idx] @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    gt = gt_idx[idx]
    has_gt = gt >= 0
    safe_gt = np.where(has_gt, gt, 0)
    gt_scores = scores[np.arange(len(idx)), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_weights(X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, float]:
    init = np.asarray([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)

    def objective(w: np.ndarray) -> float:
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)

    res = minimize(objective, init, method="Powell", options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def eval_config(payload: dict[str, Any], source_weights: tuple[float, float, float], seeds: list[int]) -> dict[str, Any]:
    print(f"building pool features for B/C/D={source_weights}", flush=True)
    X, gt_idx, sizes = build_pool_features(payload, source_weights)
    sessions = [c["session_id"] for c in payload["cases"]]
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([i for i in range(len(sessions)) if i not in held], dtype=np.int64)
            weights, train_score = fit_weights(X, gt_idx, sizes, train)
            fold_scores.append(vec_ndcg(X, gt_idx, sizes, weights, fold))
        per_seed.append({
            "seed": seed,
            "cv5_mean": float(np.mean(fold_scores)),
            "cv5_std": float(np.std(fold_scores, ddof=1)),
            "cv5_per_fold": fold_scores,
        })
        print(f"  seed {seed}: {per_seed[-1]['cv5_mean']:.4f} ± {per_seed[-1]['cv5_std']:.4f}", flush=True)
    means = [r["cv5_mean"] for r in per_seed]
    return {
        "source_weights_BCD": source_weights,
        "pool_hit_rate": float(np.mean(gt_idx >= 0)),
        "per_seed": per_seed,
        "aggregate": {
            "mean_cv5": float(np.mean(means)),
            "std_of_cv5_means": float(np.std(means, ddof=1)),
            "min_cv5": float(np.min(means)),
            "max_cv5": float(np.max(means)),
        },
    }


def main() -> None:
    t0 = time.time()
    payload = build_or_load_payload()
    seeds = [0, 1, 2]
    configs = {
        "det_A0_B2_C05_D1": (2.0, 0.5, 1.0),
        "det_A0_B2_C0_D1": (2.0, 0.0, 1.0),
        "det_A0_B4_C0_D1": (4.0, 0.0, 1.0),
    }
    results = {name: eval_config(payload, weights, seeds) for name, weights in configs.items()}
    out = {
        "meta": {
            "n": len(payload["cases"]),
            "seeds": seeds,
            "source_definition": "A removed. B=last_music_meta_bm25@500, C=full_history_bm25@500, D=track_neighbors@200.",
            "slice": "stable 400-session devset slice excluding echo_v23_pool50_s200 sessions where possible",
            "elapsed_sec": time.time() - t0,
            "no_llm_calls": True,
        },
        "results": results,
    }
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
