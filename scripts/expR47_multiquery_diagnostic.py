#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R47: Multi-query diagnostic for r21_deep cases.

Tests whether alternative query formulations help the 61 r21_deep cases
(from R45) where GT is seen-in-training, diff-artist, and R21 ranks poorly.

Hypothesis: R21's query representation may be wrong for these cases, not the
encoder weights. We test multiple query views with the SAME R21 model.

No lightgbm — retrieval diagnostics only.  build_als / als_session_vector are
inlined to avoid importing modules that pull in lightgbm (loky/torch segfault).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_MODEL_DIR = REPO / "cache" / "r21_production" / "model"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_TRACK_IDS = REPO / "cache" / "r21_production" / "track_ids.json"

QUERY_VIEWS = [
    "full_conv",           # R21 default: last 3 user messages
    "last_query_only",     # Just the current user query
    "all_user_messages",   # ALL user messages concatenated
    "music_history_summary",  # "Previously played: track by artist ..."
    "intent_query",        # Intent-based query
    "genre_mood_query",    # Tags/genres from played tracks
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ── Inlined from expS2_lambdarank to avoid lightgbm import ──────────


def build_als():
    """Train ALS and return factors + track mapping."""
    from datasets import DownloadConfig, load_dataset
    from implicit.als import AlternatingLeastSquares
    from scipy import sparse

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]
    track_set: set[str] = set()
    session_tracks: list[list[str]] = []
    for item in train:
        tracks: list[str] = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                track_set.add(tid)
        session_tracks.append(tracks)

    track_ids = sorted(track_set)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    rows, cols, vals = [], [], []
    for si, tracks in enumerate(session_tracks):
        for tid in tracks:
            rows.append(si)
            cols.append(track_to_idx[tid])
            vals.append(1.0)
    matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(session_tracks), len(track_ids)),
        dtype=np.float32,
    )

    model = AlternatingLeastSquares(
        factors=128, alpha=100, regularization=0.05,
        iterations=20, random_state=42, use_gpu=False,
    )
    model.fit(matrix)
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    elif not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    return factors, track_ids, track_to_idx


def als_session_vector(played, track_to_idx, factors, decay=0.8):
    """Compute ALS session vector from played tracks."""
    anchors = [track_to_idx[t] for t in played if t in track_to_idx]
    if not anchors:
        return None
    n = len(anchors)
    w = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    w /= w.sum()
    v = np.zeros(factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        v += w[j] * factors[idx]
    return v


# ── Data loaders ─────────────────────────────────────────────────────


def load_training_track_set():
    """Get all track_ids that appear in training sessions."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]
    seen: set[str] = set()
    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                seen.add(str(c["content"]).strip())
    return seen


def load_full_track_metadata():
    """Load track_id -> {track_name, artist_name, tags} mapping."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    for item in ds:
        tid = str(item["track_id"])
        tn = item.get("track_name", [])
        if isinstance(tn, list) and tn:
            track_name = str(tn[0])
        else:
            track_name = str(tn) if tn else ""
        an = item.get("artist_name", [])
        if isinstance(an, list) and an:
            artist_name = str(an[0])
        else:
            artist_name = str(an) if an else ""
        tags = item.get("tag_list", [])
        if isinstance(tags, list):
            tag_list = [str(t).lower().strip() for t in tags if t]
        else:
            tag_list = []
        meta[tid] = {
            "track_name": track_name,
            "artist_name": artist_name,
            "tags": tag_list,
        }
    return meta


# ── Query view builder ───────────────────────────────────────────────


def build_query_views(case, track_meta):
    """Build all 6 query views for a single case."""
    history = case["history"]
    user_query = case["user_query"]
    played = case["music_turns"]

    user_parts = [str(h["content"]) for h in history if h["role"] == "user"]
    user_parts.append(user_query)

    views = {}

    # 1. full_conv (R21 default): last 3 user messages
    views["full_conv"] = " ".join(user_parts[-3:])

    # 2. last_query_only: just the current user query
    views["last_query_only"] = user_query

    # 3. all_user_messages: ALL user messages concatenated
    views["all_user_messages"] = " ".join(user_parts)

    # 4. music_history_summary
    history_parts = []
    for tid in played:
        meta = track_meta.get(tid, {})
        tn = meta.get("track_name", "unknown")
        an = meta.get("artist_name", "unknown")
        history_parts.append(f"{tn} by {an}")
    if history_parts:
        views["music_history_summary"] = "Previously played: " + ", ".join(history_parts)
    else:
        views["music_history_summary"] = "No prior plays"

    # 5. intent_query
    if played:
        last_meta = track_meta.get(played[-1], {})
        last_artist = last_meta.get("artist_name", "")
        last_track = last_meta.get("track_name", "")
        intent_prefix = f"Recommend music similar to {last_track} by {last_artist}."
    else:
        intent_prefix = "Recommend music."
    views["intent_query"] = f"{intent_prefix} The user wants: {user_query}"

    # 6. genre_mood_query: tags from played tracks, top-10 by frequency
    tag_counter: Counter[str] = Counter()
    for tid in played:
        meta = track_meta.get(tid, {})
        for tag in meta.get("tags", []):
            if tag:
                tag_counter[tag] += 1
    top_tags = [t for t, _ in tag_counter.most_common(10)]
    if top_tags:
        views["genre_mood_query"] = "music with tags: " + ", ".join(top_tags)
    else:
        views["genre_mood_query"] = user_query  # fallback

    return views


# ── Main ─────────────────────────────────────────────────────────────


def main():
    t0 = time.time()
    print(f"{ts()} R47: Multi-Query Diagnostic for r21_deep Cases", flush=True)
    print("=" * 70, flush=True)

    # ── Load data ────────────────────────────────────────────────────
    print(f"{ts()} Loading payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"{ts()} Loading training track set...", flush=True)
    training_tracks = load_training_track_set()
    print(f"  {len(training_tracks)} unique tracks seen in training", flush=True)

    print(f"{ts()} Loading track metadata...", flush=True)
    track_meta = load_full_track_metadata()
    print(f"  {len(track_meta)} tracks with metadata", flush=True)

    # ── Step 1: Identify r21_deep cases ──────────────────────────────
    print(f"\n{ts()} Step 1: Identifying r21_deep cases...", flush=True)

    als_factors, als_track_ids_als, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids_als[j] for j in top])
        else:
            als_source.append([])

    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"  {len(h7)} hist_7 cases", flush=True)

    dense_recoverable = []
    for i in h7:
        c = cases[i]
        gt = c["gt"]
        played = c["music_turns"]

        src_union: set[str] = set()
        src_union.update(payload["src_a"][i])
        src_union.update(payload["src_b"][i])
        src_union.update(payload["src_c"][i])
        src_union.update(payload["src_d"][i])
        src_union.update(payload["src_f"][i])
        src_union.update(als_source[i])
        src_union.update(r21_source[i])

        if gt in src_union:
            continue
        if gt not in training_tracks:
            continue

        played_artists: set[str] = set()
        for t in played:
            a = ta.get(t, "")
            if isinstance(a, list):
                for aa in a:
                    if aa:
                        played_artists.add(str(aa).lower().strip())
            elif a:
                played_artists.add(str(a).lower().strip())

        gt_artist_raw = ta.get(gt, "")
        if isinstance(gt_artist_raw, list):
            gt_artist = gt_artist_raw[0].lower().strip() if gt_artist_raw else ""
        else:
            gt_artist = str(gt_artist_raw).lower().strip() if gt_artist_raw else ""

        if gt_artist and gt_artist in played_artists:
            continue

        dense_recoverable.append(i)

    print(f"  {len(dense_recoverable)} dense-recoverable cases", flush=True)

    # Load R21 embeddings for ranking
    print(f"{ts()} Loading R21 track embeddings...", flush=True)
    r21_track_embs = np.load(R21_TRACK_EMBS)
    with open(R21_TRACK_IDS) as f:
        r21_track_ids = json.load(f)
    r21_tid_to_idx = {tid: i for i, tid in enumerate(r21_track_ids)}
    dev_query_embs = np.load(REPO / "cache" / "r21_production" / "dev_query_embeddings.npy")
    print(f"  Track embs: {r21_track_embs.shape}, Query embs: {dev_query_embs.shape}", flush=True)

    # Classify dense_recoverable to find r21_deep
    r21_deep_cases: list[tuple[int, int]] = []
    for i in dense_recoverable:
        c = cases[i]
        gt = c["gt"]

        gt_r21_idx = r21_tid_to_idx.get(gt)
        if gt_r21_idx is not None and i < len(dev_query_embs):
            q_emb = dev_query_embs[i]
            scores = r21_track_embs @ q_emb
            gt_score = scores[gt_r21_idx]
            r21_rank = int(np.sum(scores > gt_score)) + 1
        else:
            r21_rank = -1

        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        gt_als_idx = als_track_to_idx.get(gt)
        if sv is not None and gt_als_idx is not None:
            sc_als = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc_als[als_track_to_idx[t]] = -np.inf
            als_rank = int(np.sum(sc_als > sc_als[gt_als_idx])) + 1
        else:
            als_rank = -1

        if 0 < r21_rank <= 500:
            cls = "r21_shallow"
        elif 0 < als_rank <= 500:
            cls = "als_shallow"
        elif 0 < r21_rank <= 5000:
            cls = "r21_deep"
        else:
            cls = "other"

        if cls == "r21_deep":
            r21_deep_cases.append((i, r21_rank))

    print(f"  {len(r21_deep_cases)} r21_deep cases identified", flush=True)
    if r21_deep_cases:
        baseline_ranks = [r for _, r in r21_deep_cases]
        print(f"  Baseline R21 rank: median={np.median(baseline_ranks):.0f}, "
              f"min={min(baseline_ranks)}, max={max(baseline_ranks)}", flush=True)

    # ── Step 2-3: Encode query views and score ───────────────────────
    print(f"\n{ts()} Step 2-3: Encoding query views and scoring...", flush=True)
    print("=" * 70, flush=True)

    from sentence_transformers import SentenceTransformer
    print(f"{ts()} Loading R21 SentenceTransformer model...", flush=True)
    r21_model = SentenceTransformer(str(R21_MODEL_DIR), device="cpu")

    n_deep = len(r21_deep_cases)
    n_views = len(QUERY_VIEWS)

    gt_ranks = np.zeros((n_deep, n_views), dtype=np.int32)
    query_texts: dict[tuple[int, str], str] = {}

    for ci, (case_idx, _baseline_rank) in enumerate(r21_deep_cases):
        if ci % 10 == 0:
            print(f"{ts()}   Case {ci+1}/{n_deep} (case_idx={case_idx})...", flush=True)

        c = cases[case_idx]
        gt = c["gt"]
        played = set(c["music_turns"])

        gt_r21_idx = r21_tid_to_idx.get(gt)
        if gt_r21_idx is None:
            gt_ranks[ci, :] = -1
            continue

        views = build_query_views(c, track_meta)
        view_texts = [views[vn] for vn in QUERY_VIEWS]
        view_embs = r21_model.encode(view_texts, normalize_embeddings=True,
                                      show_progress_bar=False).astype(np.float32)

        for vi, vn in enumerate(QUERY_VIEWS):
            query_texts[(ci, vn)] = view_texts[vi]
            q_emb = view_embs[vi]
            scores = r21_track_embs @ q_emb

            for tid in played:
                idx = r21_tid_to_idx.get(tid)
                if idx is not None:
                    scores[idx] = -np.inf

            gt_score = scores[gt_r21_idx]
            rank = int(np.sum(scores > gt_score)) + 1
            gt_ranks[ci, vi] = rank

    # ── Step 4: Report ───────────────────────────────────────────────
    print(f"\n{ts()} Step 4: Results", flush=True)
    print("=" * 70, flush=True)

    baseline_idx = QUERY_VIEWS.index("full_conv")
    baseline_ranks_arr = gt_ranks[:, baseline_idx]

    header = (f"{'Query view':<25s} | {'improved':>8s} | {'median rank':>11s} | "
              f"{'top-300':>7s} | {'top-500':>7s} | {'top-1000':>8s}")
    print(f"\n{header}", flush=True)
    print("-" * 80, flush=True)

    view_results = {}
    for vi, vn in enumerate(QUERY_VIEWS):
        ranks = gt_ranks[:, vi]
        valid_ranks = ranks[ranks > 0]

        improved = 0
        for ci in range(n_deep):
            if ranks[ci] > 0 and baseline_ranks_arr[ci] > 0:
                if ranks[ci] < baseline_ranks_arr[ci]:
                    improved += 1

        median_rank = int(np.median(valid_ranks)) if len(valid_ranks) > 0 else -1
        top300 = int(np.sum(valid_ranks <= 300))
        top500 = int(np.sum(valid_ranks <= 500))
        top1000 = int(np.sum(valid_ranks <= 1000))

        print(f"{vn:<25s} | {improved:>8d} | {median_rank:>11d} | "
              f"{top300:>7d} | {top500:>7d} | {top1000:>8d}", flush=True)

        view_results[vn] = {
            "improved_over_baseline": improved,
            "median_rank": median_rank,
            "mean_rank": float(np.mean(valid_ranks)) if len(valid_ranks) > 0 else -1,
            "top_300": top300,
            "top_500": top500,
            "top_1000": top1000,
        }

    # Per-case best view
    print(f"\n{'='*70}", flush=True)
    print("Per-case best view:", flush=True)
    best_view_counter: Counter[str] = Counter()
    per_case_results = []

    any_top300 = 0
    any_top500 = 0
    any_top1000 = 0

    for ci in range(n_deep):
        case_idx, baseline_rank = r21_deep_cases[ci]
        case_ranks = {}
        best_rank = 999999
        best_view = "none"

        for vi, vn in enumerate(QUERY_VIEWS):
            r = int(gt_ranks[ci, vi])
            case_ranks[vn] = r
            if 0 < r < best_rank:
                best_rank = r
                best_view = vn

        best_view_counter[best_view] += 1

        all_valid_ranks = [r for r in case_ranks.values() if r > 0]
        min_rank = min(all_valid_ranks) if all_valid_ranks else 999999
        if min_rank <= 300:
            any_top300 += 1
        if min_rank <= 500:
            any_top500 += 1
        if min_rank <= 1000:
            any_top1000 += 1

        per_case_results.append({
            "case_idx": case_idx,
            "gt": cases[case_idx]["gt"],
            "baseline_r21_rank": baseline_rank,
            "ranks_by_view": case_ranks,
            "best_view": best_view,
            "best_rank": best_rank,
            "improvement": baseline_rank - best_rank if best_rank < 999999 else 0,
        })

    print("\nBest view distribution:", flush=True)
    for vn in QUERY_VIEWS:
        cnt = best_view_counter.get(vn, 0)
        print(f"  {vn:<25s}: {cnt:>4d}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"{any_top1000} of {n_deep} cases have at least one query view "
          "bringing GT into top-1000", flush=True)
    print(f"{any_top500} of {n_deep} cases have at least one query view "
          "bringing GT into top-500", flush=True)
    print(f"{any_top300} of {n_deep} cases have at least one query view "
          "bringing GT into top-300", flush=True)

    # Top improvements
    per_case_sorted = sorted(per_case_results, key=lambda x: -x["improvement"])
    print("\nTop 15 cases by rank improvement (baseline -> best):", flush=True)
    for pc in per_case_sorted[:15]:
        print(f"  case {pc['case_idx']:>5d}: {pc['baseline_r21_rank']:>5d} -> "
              f"{pc['best_rank']:>5d} (delta={pc['improvement']:>+5d}) "
              f"via {pc['best_view']}", flush=True)

    # Example query texts for top improved cases
    print(f"\n{'='*70}", flush=True)
    print("Example query views for top-3 improved cases:", flush=True)
    for pc in per_case_sorted[:3]:
        ci_idx = None
        for ci, (cidx, _) in enumerate(r21_deep_cases):
            if cidx == pc["case_idx"]:
                ci_idx = ci
                break
        if ci_idx is None:
            continue
        print(f"\n--- Case {pc['case_idx']} (baseline={pc['baseline_r21_rank']}, "
              f"best={pc['best_rank']} via {pc['best_view']}) ---", flush=True)
        for vn in QUERY_VIEWS:
            txt = query_texts.get((ci_idx, vn), "")
            rank = pc["ranks_by_view"].get(vn, -1)
            txt_short = txt[:150] + "..." if len(txt) > 150 else txt
            print(f"  [{vn:<25s}] rank={rank:>5d} | {txt_short}", flush=True)

    # ── Save results ─────────────────────────────────────────────────
    out_data = {
        "n_r21_deep_cases": n_deep,
        "query_views": QUERY_VIEWS,
        "view_results": view_results,
        "any_top_300": any_top300,
        "any_top_500": any_top500,
        "any_top_1000": any_top1000,
        "best_view_distribution": dict(best_view_counter),
        "per_case": per_case_results,
    }

    out_path = REPO / "exp" / "eval" / "expR47_multiquery_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
