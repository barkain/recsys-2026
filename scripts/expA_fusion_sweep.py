#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Experiment A fusion sweep — staged, no-API, deterministic.

Step 1: cheap pool-only sweep (vary RRF k, source depths, fusion weights, plus a
quota-fusion variant). Persist all results to exp/eval/expA_fusion_pool_sweep.json.

Step 2: filter to top configs by combined metric (pool_hit primary,
median_rank tiebreak, Jaccard-overlap-with-baseline tertiary).

Step 3 (gated): Powell CV5 on the top 3 (max), reusing the same protocol as
scripts/r3_confirm_400_deterministic.py. Persist to
exp/eval/expA_fusion_powell_top3.json.

Strict constraints:
  - No LLM calls. No Sonnet/Haiku. No openai/anthropic imports.
  - Reuse cached payload (exp/eval/_r3_confirm_400_deterministic.pkl) and
    cached A' lists (exp/eval/_r3_a_prime_400.pkl) and qwen3 vectors.
  - For deeper A' (>50), recompute on-the-fly from the qwen3 cache (still
    no model download, no API).
  - For B/C deeper than 500: cache only has 500; we cap at 500 and note this
    in the artifact's meta.
  - For D deeper than 200: recompute from the qwen3 cache directly.

Baseline anchor:
  A'=A_max_recent_5 @ depth 50, B@500, C@500, D@200, k=60,
  weights (A',B,C,D)=(2.0,2.0,0.5,1.0)
  → pool_hit@50 = 0.325, median GT rank when hit = 5
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_a_prime import (  # noqa: E402
    A_PRIME_CACHE,
)
from scripts.r3_confirm_400_deterministic import (  # noqa: E402
    CACHE_FILE,
    POOL_K,
    cv_folds,
    fit_weights,
    vec_ndcg,
)

POOL_SWEEP_OUT = REPO_ROOT / "exp" / "eval" / "expA_fusion_pool_sweep.json"
POWELL_OUT = REPO_ROOT / "exp" / "eval" / "expA_fusion_powell_top3.json"

QWEN_VECS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy"
QWEN_IDS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"


def fast_load_track_metadata_for(target_tids: set[str]) -> dict[str, dict]:
    """Fast targeted catalog lookup using pyarrow columnar API.

    Avoids the slow `Dataset.from_file → {item['track_id']: item for item in ds}`
    pattern, which calls Array.to_pylist + scalar wrapping for every row.

    Returns: tid → {artist_name, track_name, album_name, tag_list}
    """
    import pyarrow as pa  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415

    from offline_retrieval_sweep import latest_arrow  # noqa: PLC0415

    path = latest_arrow(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"
    )
    ds = Dataset.from_file(path)
    tbl: pa.Table = ds.data.table  # underlying arrow table
    # Pull columns once.
    tids_col = tbl.column("track_id").to_pylist()
    if not target_tids:
        return {}
    target = {str(t) for t in target_tids}
    # Build position map
    positions = [i for i, t in enumerate(tids_col) if t in target]
    if not positions:
        return {}
    sub = tbl.take(positions)
    sub_tids = sub.column("track_id").to_pylist()
    sub_artist = sub.column("artist_name").to_pylist()
    sub_track = sub.column("track_name").to_pylist()
    sub_album = sub.column("album_name").to_pylist()
    sub_tags = sub.column("tag_list").to_pylist()
    out: dict[str, dict] = {}
    for tid, art, trk, alb, tg in zip(sub_tids, sub_artist, sub_track, sub_album, sub_tags, strict=True):
        out[str(tid)] = {
            "artist_name": art,
            "track_name": trk,
            "album_name": alb,
            "tag_list": tg,
        }
    return out


def fast_augment_metadata_for_a(
    payload: dict[str, Any],
    a_variants_or_lists,
) -> int:
    """Fast replacement for augment_metadata_for_a.

    Accepts either a dict {variant: list-of-lists} or a single list-of-lists.
    Returns count of new tids augmented.
    """
    from scripts.r3_confirm_400_deterministic import tokens  # noqa: PLC0415

    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    if isinstance(a_variants_or_lists, dict):
        iters = list(a_variants_or_lists.values())
    else:
        iters = [a_variants_or_lists]

    new_tids: set[str] = set()
    for v in iters:
        for sub in v:
            for tid in sub:
                if tid not in track_artist:
                    new_tids.add(tid)
    if not new_tids:
        return 0
    print(f"  fast-augment: looking up metadata for {len(new_tids)} new tids", flush=True)
    metadata = fast_load_track_metadata_for(new_tids)
    for tid in new_tids:
        meta = metadata.get(str(tid), {})
        artist = str(meta.get("artist_name", "")).lower().strip()
        raw_tags = meta.get("tag_list") or []
        tags = (
            {str(t).lower().strip() for t in raw_tags if str(t).strip()}
            if isinstance(raw_tags, list) else set()
        )
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
    return len(new_tids)

# ---------------------------------------------------------------------------
# Baseline and gates
BASELINE = {
    "name": "baseline_v25",
    "params": {
        "variant": "A_max_recent_5",
        "depths": {"A": 50, "B": 500, "C": 500, "D": 200},
        "k": 60,
        "weights": {"A": 2.0, "B": 2.0, "C": 0.5, "D": 1.0},
        "fusion": "rrf",
    },
}

# Gates (from spec)
GATE_INTERMEDIATE_POOL = 0.34
GATE_INTERMEDIATE_CV = 0.160
GATE_SUBMISSION_CV = 0.167
GATE_REJECT_DELTA_POOL = 0.01
GATE_WEAK_CV = 0.155


# ---------------------------------------------------------------------------
# A' deeper recomputation helper (cache only has top-50 per variant per session)
def load_qwen_index() -> tuple[np.ndarray, dict[str, int], list[str]]:
    vecs = np.load(QWEN_VECS).astype(np.float32, copy=False)
    with open(QWEN_IDS) as f:
        ids = json.load(f)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    return vecs, id_to_idx, ids


def a_prime_deep(
    vecs: np.ndarray,
    id_to_idx: dict[str, int],
    track_ids: list[str],
    music_turns: list[str],
    *,
    n: int,
    k: int,
) -> list[str]:
    """A_max_recent_N at depth k via cached qwen3 vectors. No API."""
    played_idxs = [id_to_idx[t] for t in music_turns if t in id_to_idx]
    if not played_idxs:
        return []
    window = played_idxs[-n:]
    sims = vecs @ vecs[window].T
    agg = sims.max(axis=1)
    exclude = set(played_idxs)
    for j in exclude:
        agg[j] = -1.0
    nn = min(k + 1, len(agg))
    cand = np.argpartition(-agg, nn - 1)[:nn]
    cand = cand[np.argsort(-agg[cand])]
    return [track_ids[i] for i in cand if i not in exclude][:k]


def a_prime_deep_batch(
    vecs: np.ndarray,
    id_to_idx: dict[str, int],
    track_ids: list[str],
    cases: list[dict[str, Any]],
    *,
    n: int,
    k: int,
) -> list[list[str]]:
    """Vectorised A_max_recent_N for ALL cases.

    Stack each session's window vectors into one big (totalrows, D) matrix and
    do a single BLAS call: scores = vecs @ stacked.T → (V, totalrows). Then
    per-session split + max + topk. This is ~10× faster than the per-session
    Python loop because of BLAS GEMM overhead amortisation.
    """
    sess_indices: list[list[int]] = []  # per-session played idxs
    sess_windows: list[list[int]] = []  # per-session window slice
    for c in cases:
        played_idxs = [id_to_idx[t] for t in c["music_turns"] if t in id_to_idx]
        sess_indices.append(played_idxs)
        sess_windows.append(played_idxs[-n:] if played_idxs else [])

    flat = []
    offsets = [0]
    for win in sess_windows:
        flat.extend(win)
        offsets.append(offsets[-1] + len(win))
    if not flat:
        return [[] for _ in cases]
    flat_arr = np.asarray(flat, dtype=np.int64)
    stacked = vecs[flat_arr]  # (totalrows, D)
    sims = vecs @ stacked.T  # (V, totalrows)
    out: list[list[str]] = []
    for i in range(len(cases)):
        a, b = offsets[i], offsets[i + 1]
        if a == b:
            out.append([])
            continue
        agg = sims[:, a:b].max(axis=1)
        exclude = set(sess_indices[i])
        agg = agg.copy()
        for j in exclude:
            agg[j] = -1.0
        nn = min(k + len(exclude) + 1, len(agg))
        cand = np.argpartition(-agg, nn - 1)[:nn]
        cand = cand[np.argsort(-agg[cand])]
        out.append([track_ids[idx] for idx in cand if idx not in exclude][:k])
    return out


def d_deep(
    vecs: np.ndarray,
    id_to_idx: dict[str, int],
    track_ids: list[str],
    anchor_tid: str | None,
    k: int,
) -> list[str]:
    if anchor_tid is None:
        return []
    idx = id_to_idx.get(str(anchor_tid))
    if idx is None:
        return []
    query_vec = vecs[idx]
    scores = vecs @ query_vec
    n = min(k + 1, len(scores))
    cand = np.argpartition(-scores, n - 1)[:n]
    cand = cand[np.argsort(-scores[cand])]
    return [track_ids[i] for i in cand if i != idx][:k]


def d_deep_batch(
    vecs: np.ndarray,
    id_to_idx: dict[str, int],
    track_ids: list[str],
    cases: list[dict[str, Any]],
    *,
    k: int,
) -> list[list[str]]:
    """Vectorised D@k (last-track neighbours) over ALL cases.

    Single GEMM: vecs @ stacked.T where stacked are the anchor vectors.
    """
    anchors = []
    valid_mask = []
    for c in cases:
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        idx = id_to_idx.get(str(anchor)) if anchor else None
        if idx is None:
            anchors.append(0)
            valid_mask.append(-1)
        else:
            anchors.append(idx)
            valid_mask.append(idx)
    arr = np.asarray(anchors, dtype=np.int64)
    stacked = vecs[arr]
    sims = vecs @ stacked.T  # (V, n_cases)
    out: list[list[str]] = []
    for i in range(len(cases)):
        if valid_mask[i] < 0:
            out.append([])
            continue
        scores = sims[:, i].copy()
        idx = valid_mask[i]
        scores[idx] = -1.0
        nn = min(k + 1, len(scores))
        cand = np.argpartition(-scores, nn - 1)[:nn]
        cand = cand[np.argsort(-scores[cand])]
        out.append([track_ids[j] for j in cand if j != idx][:k])
    return out


# ---------------------------------------------------------------------------
def rrf_fuse(
    seqs_with_weights: list[tuple[list[str], float]],
    k: int,
    pool_size: int = POOL_K,
) -> list[str]:
    scores: dict[str, float] = {}
    for seq, w in seqs_with_weights:
        if w == 0:
            continue
        for r, tid in enumerate(seq, start=1):
            scores[tid] = scores.get(tid, 0.0) + w / (k + r)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:pool_size]


def quota_fuse(
    sources: dict[str, list[str]],
    quotas: dict[str, int],
    leftover_order: list[str],
    pool_size: int = POOL_K,
) -> list[str]:
    """Take qX from each source, dedupe, then fill from a leftover ranked list."""
    out: list[str] = []
    seen: set[str] = set()
    for src in ("A", "B", "C", "D"):
        q = quotas.get(src, 0)
        if q <= 0:
            continue
        for tid in sources.get(src, []):
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
                q -= 1
                if q == 0:
                    break
    if len(out) < pool_size:
        for tid in leftover_order:
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
                if len(out) >= pool_size:
                    break
    return out[:pool_size]


# ---------------------------------------------------------------------------
def evaluate_pool_metrics(
    cases: list[dict[str, Any]],
    pools: list[list[str]],
    baseline_pools: list[list[str]] | None,
) -> dict[str, float]:
    """Compute pool_hit@50, median GT rank when hit, Jaccard overlap vs baseline."""
    n = len(cases)
    hits = 0
    ranks: list[int] = []
    overlaps: list[float] = []
    for i, c in enumerate(cases):
        gt = c["gt"]
        pool = pools[i]
        if gt in pool:
            hits += 1
            ranks.append(pool.index(gt))
        if baseline_pools is not None:
            base = baseline_pools[i]
            if not base and not pool:
                overlaps.append(1.0)
                continue
            sa = set(pool)
            sb = set(base)
            union = sa | sb
            if union:
                overlaps.append(len(sa & sb) / len(union))
    return {
        "pool_hit": hits / n if n else 0.0,
        "median_rank_when_hit": float(np.median(ranks)) if ranks else float("nan"),
        "median_rank_p25": float(np.percentile(ranks, 25)) if ranks else float("nan"),
        "median_rank_p75": float(np.percentile(ranks, 75)) if ranks else float("nan"),
        "overlap_with_baseline": float(np.mean(overlaps)) if overlaps else float("nan"),
    }


# ---------------------------------------------------------------------------
def compute_baseline_pools(
    cases: list[dict[str, Any]],
    src_a: list[list[str]],
    src_b: list[list[str]],
    src_c: list[list[str]],
    src_d: list[list[str]],
) -> list[list[str]]:
    """Reproduce the spec baseline (k=60, A50/B500/C500/D200, w=2/2/0.5/1)."""
    pools = []
    for i in range(len(cases)):
        seqs = [
            (src_a[i][:50], 2.0),
            (src_b[i][:500], 2.0),
            (src_c[i][:500], 0.5),
            (src_d[i][:200], 1.0),
        ]
        pools.append(rrf_fuse(seqs, k=60))
    return pools


def precompute_session_score_matrices(
    cases: list[dict[str, Any]],
    src_a: list[list[str]],
    src_b: list[list[str]],
    src_c: list[list[str]],
    src_d: list[list[str]],
    *,
    depths: dict[str, int],
    k: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Per session, return (M, tid_to_idx_array, gt_idx_in_session_or_-1).

    M is (n_tids_in_session, 4) where M[i, s] = 1/(k + rank) if tid_i is in
    source s at the requested depth. Sources order: A, B, C, D.

    Used by the vectorised weight-grid sweep so we don't rebuild the score
    dict for every weight combo.
    """
    Ms: list[np.ndarray] = []
    tid_arrays: list[np.ndarray] = []
    gt_idxs: list[int] = []
    for i, c in enumerate(cases):
        seqs = (
            src_a[i][: depths.get("A", 0)] if depths.get("A", 0) > 0 else [],
            src_b[i][: depths.get("B", 0)] if depths.get("B", 0) > 0 else [],
            src_c[i][: depths.get("C", 0)] if depths.get("C", 0) > 0 else [],
            src_d[i][: depths.get("D", 0)] if depths.get("D", 0) > 0 else [],
        )
        # union of unique tids
        seen: dict[str, int] = {}
        for seq in seqs:
            for tid in seq:
                if tid not in seen:
                    seen[tid] = len(seen)
        n_tids = len(seen)
        if n_tids == 0:
            Ms.append(np.zeros((0, 4), dtype=np.float64))
            tid_arrays.append(np.empty(0, dtype=object))
            gt_idxs.append(-1)
            continue
        M = np.zeros((n_tids, 4), dtype=np.float64)
        tid_arr = np.empty(n_tids, dtype=object)
        for tid, idx in seen.items():
            tid_arr[idx] = tid
        for s, seq in enumerate(seqs):
            for r, tid in enumerate(seq, start=1):
                idx = seen[tid]
                # Only set if not already set (handle duplicates within a list)
                if M[idx, s] == 0.0:
                    M[idx, s] = 1.0 / (k + r)
        gt = c["gt"]
        gt_idxs.append(seen.get(gt, -1))
        Ms.append(M)
        tid_arrays.append(tid_arr)
    return Ms, tid_arrays, gt_idxs


def evaluate_weight_grid_fast(
    cases: list[dict[str, Any]],
    Ms: list[np.ndarray],
    tid_arrays: list[np.ndarray],
    gt_idxs: list[int],
    weight_vectors: np.ndarray,
    *,
    pool_size: int = POOL_K,
    baseline_pools: list[list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate W weight vectors in one shot per session using matrix multiply.

    Returns: list of W dicts, each with pool_hit, median_rank, overlap.
    """
    n_sessions = len(cases)
    n_weights = weight_vectors.shape[0]
    # For each weight: count hits + ranks + overlap
    hit_counts = np.zeros(n_weights, dtype=np.int64)
    rank_lists: list[list[int]] = [[] for _ in range(n_weights)]
    overlap_lists: list[list[float]] = [[] for _ in range(n_weights)]
    # Per-session pools cached just for overlap reference
    for i in range(n_sessions):
        M = Ms[i]
        gt_idx = gt_idxs[i]
        tid_arr = tid_arrays[i]
        if M.shape[0] == 0:
            continue
        scores_all = M @ weight_vectors.T  # (n_tids, W)
        # We need per-W top-`pool_size` by score. Use argpartition.
        # For efficiency, iterate W (W is 192).
        n_tids = M.shape[0]
        actual_pool = min(pool_size, n_tids)
        for w in range(n_weights):
            scores = scores_all[:, w]
            if n_tids <= pool_size:
                cand = np.argsort(-scores)
            else:
                cand = np.argpartition(-scores, actual_pool - 1)[:actual_pool]
                cand = cand[np.argsort(-scores[cand])]
            top = cand[:pool_size]
            if gt_idx >= 0:
                # Find GT rank in top
                pos = np.where(top == gt_idx)[0]
                if pos.size > 0:
                    hit_counts[w] += 1
                    rank_lists[w].append(int(pos[0]))
            if baseline_pools is not None:
                base = set(baseline_pools[i])
                pool_set = set(tid_arr[top].tolist())
                union = pool_set | base
                if union:
                    overlap_lists[w].append(len(pool_set & base) / len(union))
    out = []
    for w in range(n_weights):
        ranks = rank_lists[w]
        ovs = overlap_lists[w]
        out.append({
            "pool_hit": hit_counts[w] / n_sessions,
            "median_rank_when_hit": float(np.median(ranks)) if ranks else float("nan"),
            "median_rank_p25": float(np.percentile(ranks, 25)) if ranks else float("nan"),
            "median_rank_p75": float(np.percentile(ranks, 75)) if ranks else float("nan"),
            "overlap_with_baseline": float(np.mean(ovs)) if ovs else float("nan"),
        })
    return out


def compute_pools_rrf(
    cases: list[dict[str, Any]],
    src_a: list[list[str]],
    src_b: list[list[str]],
    src_c: list[list[str]],
    src_d: list[list[str]],
    *,
    depths: dict[str, int],
    weights: dict[str, float],
    k: int,
) -> list[list[str]]:
    pools = []
    dA, dB, dC, dD = depths["A"], depths["B"], depths["C"], depths["D"]
    wA, wB, wC, wD = weights["A"], weights["B"], weights["C"], weights["D"]
    for i in range(len(cases)):
        seqs = [
            (src_a[i][:dA] if dA > 0 else [], wA),
            (src_b[i][:dB] if dB > 0 else [], wB),
            (src_c[i][:dC] if dC > 0 else [], wC),
            (src_d[i][:dD] if dD > 0 else [], wD),
        ]
        pools.append(rrf_fuse(seqs, k=k))
    return pools


# ---------------------------------------------------------------------------
def stage1_pool_sweep(payload: dict[str, Any], a_variants: dict[str, list[list[str]]]) -> dict[str, Any]:
    """Big-but-bounded pool-only sweep, returning all configs with metrics."""
    cases = payload["cases"]
    src_b_all = payload["src_b"]  # depth 500
    src_c_all = payload["src_c"]  # depth 500
    src_d_all = payload["src_d"]  # depth 200
    a_lists_50 = a_variants["A_max_recent_5"]  # depth 50

    # Cache deeper sources lazily.
    print("loading qwen3 index for deeper A'/D recomputation", flush=True)
    vecs, id_to_idx, track_ids = load_qwen_index()

    deeper_A: dict[int, list[list[str]]] = {50: a_lists_50}
    deeper_D: dict[int, list[list[str]]] = {200: src_d_all}

    def get_a(depth: int) -> list[list[str]]:
        if depth in deeper_A:
            return deeper_A[depth]
        print(f"  recomputing A'@{depth} for {len(cases)} sessions (batched)...", flush=True)
        t0 = time.time()
        out = a_prime_deep_batch(vecs, id_to_idx, track_ids, cases, n=5, k=depth)
        print(f"    A'@{depth} done in {time.time() - t0:.1f}s", flush=True)
        deeper_A[depth] = out
        return out

    def get_d(depth: int) -> list[list[str]]:
        if depth in deeper_D:
            return deeper_D[depth]
        print(f"  recomputing D@{depth} for {len(cases)} sessions (batched)...", flush=True)
        t0 = time.time()
        out = d_deep_batch(vecs, id_to_idx, track_ids, cases, k=depth)
        print(f"    D@{depth} done in {time.time() - t0:.1f}s", flush=True)
        deeper_D[depth] = out
        return out

    # ---- Baseline pools for overlap reference ----
    print("computing baseline pools for overlap reference", flush=True)
    baseline_pools = compute_baseline_pools(cases, a_lists_50, src_b_all, src_c_all, src_d_all)
    baseline_metrics = evaluate_pool_metrics(cases, baseline_pools, baseline_pools)

    # ---- Phase A: anchor-centred 1-D sweeps (k, depths) ----
    anchor = {
        "depths": {"A": 50, "B": 500, "C": 500, "D": 200},
        "weights": {"A": 2.0, "B": 2.0, "C": 0.5, "D": 1.0},
        "k": 60,
    }

    configs: list[dict[str, Any]] = []
    cfg_id = 0

    def eval_rrf(*, depths: dict[str, int], weights: dict[str, float], k: int, tag: str) -> dict[str, Any]:
        nonlocal cfg_id
        cfg_id += 1
        # Resolve A'/D source depths (recompute as needed).
        src_a = get_a(depths["A"]) if depths["A"] > 0 else [[] for _ in cases]
        src_d = get_d(depths["D"]) if depths["D"] > 0 else [[] for _ in cases]
        # B/C cache caps at 500.
        src_b = src_b_all
        src_c = src_c_all
        pools = compute_pools_rrf(
            cases, src_a, src_b, src_c, src_d,
            depths=depths, weights=weights, k=k,
        )
        m = evaluate_pool_metrics(cases, pools, baseline_pools)
        return {
            "id": f"cfg{cfg_id:04d}_{tag}",
            "fusion": "rrf",
            "params": {"depths": depths.copy(), "weights": weights.copy(), "k": k, "variant": "A_max_recent_5"},
            "pool_hit": m["pool_hit"],
            "median_rank_when_hit": m["median_rank_when_hit"],
            "median_rank_p25": m["median_rank_p25"],
            "median_rank_p75": m["median_rank_p75"],
            "overlap_with_baseline": m["overlap_with_baseline"],
        }

    # k sweep with anchor depths/weights
    print("\n[Phase A.1] RRF k sweep at anchor depths/weights", flush=True)
    for k_val in (10, 20, 40, 60, 100):
        cfg = eval_rrf(depths=anchor["depths"], weights=anchor["weights"], k=k_val, tag=f"k{k_val}")
        configs.append(cfg)
        print(f"  {cfg['id']}: pool_hit={cfg['pool_hit']:.4f}  med_rank={cfg['median_rank_when_hit']}", flush=True)

    # A' depth sweep (with anchor weights/k)
    print("\n[Phase A.2] A' depth sweep", flush=True)
    for dA in (50, 100, 200):
        depths = {**anchor["depths"], "A": dA}
        cfg = eval_rrf(depths=depths, weights=anchor["weights"], k=anchor["k"], tag=f"A{dA}")
        configs.append(cfg)
        print(f"  {cfg['id']}: pool_hit={cfg['pool_hit']:.4f}  med_rank={cfg['median_rank_when_hit']}", flush=True)

    # B depth sweep (cache caps at 500, so {500} only — 1000 unavailable in cache).
    # We still eval depth=200 as a contraction.
    print("\n[Phase A.3] B depth sweep (capped at 500 by cache)", flush=True)
    for dB in (200, 500):
        depths = {**anchor["depths"], "B": dB}
        cfg = eval_rrf(depths=depths, weights=anchor["weights"], k=anchor["k"], tag=f"B{dB}")
        configs.append(cfg)
        print(f"  {cfg['id']}: pool_hit={cfg['pool_hit']:.4f}  med_rank={cfg['median_rank_when_hit']}", flush=True)

    # C depth sweep (0 = drop entirely)
    print("\n[Phase A.4] C depth sweep (incl. drop=0)", flush=True)
    for dC in (0, 100, 500):
        depths = {**anchor["depths"], "C": dC}
        weights = anchor["weights"].copy()
        if dC == 0:
            weights = {**weights, "C": 0.0}
        cfg = eval_rrf(depths=depths, weights=weights, k=anchor["k"], tag=f"C{dC}")
        configs.append(cfg)
        print(f"  {cfg['id']}: pool_hit={cfg['pool_hit']:.4f}  med_rank={cfg['median_rank_when_hit']}", flush=True)

    # D depth sweep
    print("\n[Phase A.5] D depth sweep", flush=True)
    for dD in (100, 200, 500):
        depths = {**anchor["depths"], "D": dD}
        cfg = eval_rrf(depths=depths, weights=anchor["weights"], k=anchor["k"], tag=f"D{dD}")
        configs.append(cfg)
        print(f"  {cfg['id']}: pool_hit={cfg['pool_hit']:.4f}  med_rank={cfg['median_rank_when_hit']}", flush=True)

    # ---- Phase B: weight grid at best depths from Phase A ----
    # Pick best depths from Phase A by pool_hit (tiebreak: lower median_rank).
    def sort_key(c: dict[str, Any]) -> tuple[float, float]:
        mrk = c["median_rank_when_hit"]
        return (-c["pool_hit"], mrk if mrk == mrk else 1e9)

    best_depths_cfg = sorted(configs, key=sort_key)[0]
    best_depths = best_depths_cfg["params"]["depths"]
    best_k_cfg = sorted([c for c in configs if c["id"].endswith(("_k10", "_k20", "_k40", "_k60", "_k100"))], key=sort_key)[0]
    best_k = best_k_cfg["params"]["k"]
    print(f"\n[Phase B] best depths so far: {best_depths}  best_k: {best_k}", flush=True)

    # Weight grid: A' ∈ {1,2,3,4}, B ∈ {1,2,3,4}, C ∈ {0,0.25,0.5,1}, D ∈ {0.5,1,2}
    # = 4×4×4×3 = 192 configs.
    # FAST PATH: precompute per-session source-rank matrix once at best depths
    # and best_k, then evaluate all 192 weight vectors in a single matmul per
    # session. ~50× faster than the per-config dict rebuild.
    print("\n[Phase B.1] full weight grid at best depths/k (192 configs, vectorised)", flush=True)
    weight_grid = list(product([1, 2, 3, 4], [1, 2, 3, 4], [0, 0.25, 0.5, 1], [0.5, 1, 2]))
    t_start_b = time.time()
    src_a_best = get_a(best_depths["A"]) if best_depths["A"] > 0 else [[] for _ in cases]
    src_d_best = get_d(best_depths["D"]) if best_depths["D"] > 0 else [[] for _ in cases]
    Ms, tid_arrays, gt_idxs = precompute_session_score_matrices(
        cases, src_a_best, src_b_all, src_c_all, src_d_best,
        depths=best_depths, k=best_k,
    )
    print(f"  precomputed score matrices in {time.time() - t_start_b:.1f}s", flush=True)
    t1 = time.time()
    weight_arr = np.asarray([[wA, wB, wC, wD] for (wA, wB, wC, wD) in weight_grid], dtype=np.float64)
    metric_rows = evaluate_weight_grid_fast(
        cases, Ms, tid_arrays, gt_idxs, weight_arr,
        pool_size=POOL_K, baseline_pools=baseline_pools,
    )
    print(f"  weight grid eval in {time.time() - t1:.1f}s", flush=True)
    for (wA, wB, wC, wD), m in zip(weight_grid, metric_rows, strict=True):
        cfg_id += 1
        depths_used = best_depths.copy()
        # When wC == 0, treat as drop (depth set to 0 conceptually)
        if wC == 0:
            depths_used = {**depths_used, "C": 0}
        configs.append({
            "id": f"cfg{cfg_id:04d}_w{wA}-{wB}-{wC}-{wD}",
            "fusion": "rrf",
            "params": {
                "depths": depths_used,
                "weights": {"A": float(wA), "B": float(wB), "C": float(wC), "D": float(wD)},
                "k": best_k,
                "variant": "A_max_recent_5",
            },
            "pool_hit": m["pool_hit"],
            "median_rank_when_hit": m["median_rank_when_hit"],
            "median_rank_p25": m["median_rank_p25"],
            "median_rank_p75": m["median_rank_p75"],
            "overlap_with_baseline": m["overlap_with_baseline"],
        })
    print(f"  Phase B total: {time.time() - t_start_b:.1f}s", flush=True)

    # ---- Phase C: targeted near-best k re-sweep around the top-3 weight configs ----
    # Skip if no improvement vs baseline; otherwise run k ∈ {20, 40, 60, 100} for top 3.
    sorted_phase_b = sorted(
        [c for c in configs if "_w" in c["id"]],
        key=sort_key,
    )
    top3_weight = sorted_phase_b[:3]
    print("\n[Phase C] k re-sweep around top-3 weight configs", flush=True)
    for tcfg in top3_weight:
        for k_val in (20, 40, 60, 100):
            if k_val == best_k:
                continue
            cfg = eval_rrf(
                depths=tcfg["params"]["depths"],
                weights=tcfg["params"]["weights"],
                k=k_val,
                tag=f"{tcfg['id'].split('_', 1)[1]}_k{k_val}",
            )
            configs.append(cfg)

    # ---- Phase D: quota-fusion variant ----
    # qA ∈ {10,15,20}, qB ∈ {15,20,25}, qC ∈ {0,5,10}, qD ∈ {10,15,20}
    # = 3×3×3×3 = 81 configs.
    print("\n[Phase D] quota fusion grid (81 configs)", flush=True)
    # For leftover-fill we use the ANCHOR RRF pool ranking (i.e. the same source
    # pool the postranker already trusts) so that the quota-fusion stays
    # comparable to RRF.
    src_a_anchor = a_lists_50
    src_d_anchor = src_d_all
    src_b_anchor = src_b_all
    src_c_anchor = src_c_all
    # Top-200 fallback list (from the same RRF) for quota-fusion leftover-fill:
    leftover_top200 = []
    for i in range(len(cases)):
        scores: dict[str, float] = {}
        for seq, w in (
            (src_a_anchor[i][:50], 2.0),
            (src_b_anchor[i][:500], 2.0),
            (src_c_anchor[i][:500], 0.5),
            (src_d_anchor[i][:200], 1.0),
        ):
            for r, tid in enumerate(seq, start=1):
                scores[tid] = scores.get(tid, 0.0) + w / (60.0 + r)
        leftover_top200.append(sorted(scores, key=scores.__getitem__, reverse=True)[:200])

    quota_grid = list(product([10, 15, 20], [15, 20, 25], [0, 5, 10], [10, 15, 20]))
    for qA, qB, qC, qD in quota_grid:
        cfg_id += 1
        pools = []
        for i in range(len(cases)):
            sources = {
                "A": src_a_anchor[i][:50],
                "B": src_b_anchor[i][:500],
                "C": src_c_anchor[i][:500],
                "D": src_d_anchor[i][:200],
            }
            quotas = {"A": qA, "B": qB, "C": qC, "D": qD}
            pool = quota_fuse(sources, quotas, leftover_top200[i], pool_size=POOL_K)
            pools.append(pool)
        m = evaluate_pool_metrics(cases, pools, baseline_pools)
        configs.append({
            "id": f"cfg{cfg_id:04d}_quota_{qA}-{qB}-{qC}-{qD}",
            "fusion": "quota",
            "params": {"quotas": {"A": qA, "B": qB, "C": qC, "D": qD}, "fallback": "anchor_rrf_top200"},
            "pool_hit": m["pool_hit"],
            "median_rank_when_hit": m["median_rank_when_hit"],
            "median_rank_p25": m["median_rank_p25"],
            "median_rank_p75": m["median_rank_p75"],
            "overlap_with_baseline": m["overlap_with_baseline"],
        })

    return {
        "baseline": {**BASELINE, **baseline_metrics},
        "configs": configs,
        "n_cases": len(cases),
        "best_phaseA_depths": best_depths,
        "best_phaseA_k": best_k,
        "depth_cap_note": "B/C source caches are size 500; B@1000/C@1000 not evaluated (would require BM25 retrieval rebuild for ZERO API benefit).",
    }


# ---------------------------------------------------------------------------
def filter_top(stage1: dict[str, Any], min_delta: float, max_picks: int = 5) -> list[dict[str, Any]]:
    base_pool = stage1["baseline"]["pool_hit"]
    candidates = [c for c in stage1["configs"] if c["pool_hit"] >= base_pool + min_delta]
    # Sort: pool_hit DESC, median_rank ASC, overlap ASC
    def key(c):
        mrk = c["median_rank_when_hit"] if c["median_rank_when_hit"] == c["median_rank_when_hit"] else 1e9
        ov = c["overlap_with_baseline"] if c["overlap_with_baseline"] == c["overlap_with_baseline"] else 1.0
        return (-c["pool_hit"], mrk, ov)
    return sorted(candidates, key=key)[:max_picks]


# ---------------------------------------------------------------------------
def stage2_filter(stage1: dict[str, Any]) -> dict[str, Any]:
    """Apply the spec's filtering rules and decide whether to run Powell."""
    base_pool = stage1["baseline"]["pool_hit"]
    # First: hard reject if NO config beats baseline by ≥+0.01
    strong_lift = [c for c in stage1["configs"] if c["pool_hit"] >= base_pool + GATE_REJECT_DELTA_POOL]
    if len(strong_lift) == 0:
        return {
            "decision": "REJECT",
            "reason": f"No config improves pool_hit by ≥+{GATE_REJECT_DELTA_POOL} over baseline {base_pool:.4f}",
            "top10": filter_top(stage1, min_delta=-1.0, max_picks=10),  # show top regardless
            "selected_for_powell": [],
        }
    # Need ≥5 configs that lift pool_hit by ≥+0.01 to proceed
    if len(strong_lift) < 5:
        return {
            "decision": "NO_FUSION_LIFT",
            "reason": (
                f"Only {len(strong_lift)} configs improve pool_hit by ≥+{GATE_REJECT_DELTA_POOL} "
                f"over baseline {base_pool:.4f}; need ≥5 to proceed."
            ),
            "top10": filter_top(stage1, min_delta=-1.0, max_picks=10),
            "strong_lift_configs": strong_lift,
            "selected_for_powell": [],
        }
    # Filter by ≥+0.005 then take top 5
    top5 = filter_top(stage1, min_delta=0.005, max_picks=5)
    top3_for_powell = top5[:3]  # cap per spec
    return {
        "decision": "PROCEED_TO_POWELL",
        "reason": f"{len(strong_lift)} configs lift pool_hit by ≥+{GATE_REJECT_DELTA_POOL}; running Powell on top 3.",
        "top10": filter_top(stage1, min_delta=-1.0, max_picks=10),
        "top5": top5,
        "selected_for_powell": top3_for_powell,
    }


# ---------------------------------------------------------------------------
def stage3_powell(
    payload: dict[str, Any],
    a_variants: dict[str, list[list[str]]],
    selected: list[dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    """Run the full Powell CV5 protocol on each selected config."""
    # Lazy/fast metadata augment — only for tids actually present in any A' list
    # we may consult during Stage 3.
    fast_augment_metadata_for_a(payload, a_variants["A_max_recent_5"])
    results: dict[str, Any] = {}

    # We need deeper A'/D for some configs.
    print("loading qwen3 index for Powell A'/D depth recomputation", flush=True)
    vecs, id_to_idx, track_ids = load_qwen_index()

    cache_a: dict[int, list[list[str]]] = {50: a_variants["A_max_recent_5"]}
    cache_d: dict[int, list[list[str]]] = {200: payload["src_d"]}

    def get_a(depth: int) -> list[list[str]]:
        if depth in cache_a:
            return cache_a[depth]
        out = a_prime_deep_batch(vecs, id_to_idx, track_ids, payload["cases"], n=5, k=depth)
        cache_a[depth] = out
        return out

    def get_d(depth: int) -> list[list[str]]:
        if depth in cache_d:
            return cache_d[depth]
        out = d_deep_batch(vecs, id_to_idx, track_ids, payload["cases"], k=depth)
        cache_d[depth] = out
        return out

    for cfg in selected:
        if cfg["fusion"] != "rrf":
            print(f"[{cfg['id']}] non-rrf config; skipping Powell (would need its own pipeline)", flush=True)
            results[cfg["id"]] = {"skipped": True, "reason": "non-rrf"}
            continue
        depths = cfg["params"]["depths"]
        weights = cfg["params"]["weights"]
        k = cfg["params"]["k"]
        # Recompute per-source feature lists at the chosen depths.
        a_lists_full = get_a(depths["A"]) if depths["A"] > 0 else [[] for _ in payload["cases"]]
        # Inject overrides via temporary payload
        tmp_payload = dict(payload)
        tmp_payload["src_b"] = [s[: depths["B"]] for s in payload["src_b"]] if depths["B"] > 0 else [[] for _ in payload["cases"]]
        tmp_payload["src_c"] = [s[: depths["C"]] for s in payload["src_c"]] if depths["C"] > 0 else [[] for _ in payload["cases"]]
        if depths["D"] != 200:
            tmp_payload["src_d"] = get_d(depths["D"])
        else:
            tmp_payload["src_d"] = payload["src_d"]

        print(f"\n[{cfg['id']}] Powell CV5 with k={k}, depths={depths}, weights={weights}", flush=True)
        # eval_one in r3_confirm_400_a_prime hardcodes RRF k=60. Need to call
        # build_pool_features_with_a + the same Powell loop, but with custom k.
        res = _eval_powell_with_k(
            tmp_payload, a_lists_full, weights, k=k, seeds=seeds,
        )
        agg = res["aggregate"]
        results[cfg["id"]] = {
            "config": cfg,
            **res,
        }
        print(
            f"[{cfg['id']}] mean_cv5={agg['mean_cv5']:.4f}  "
            f"std_of_means={agg['std_of_cv5_means']:.4f}  "
            f"pool_hit={res['pool_hit_rate']:.4f}",
            flush=True,
        )

    return {
        "seeds": seeds,
        "results": results,
    }


def _eval_powell_with_k(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    weights: dict[str, float],
    *,
    k: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Custom Powell eval that supports a non-default RRF k.

    Mirrors r3_confirm_400_a_prime.eval_one but rebuilds pool features with the
    requested k.
    """
    # Adapter: build_pool_features_with_a uses fixed k=60 in r3_confirm_400_a_prime.
    # We replicate the function inline here so we can vary k.
    from scripts.r3_confirm_400_deterministic import tokens  # noqa: PLC0415
    from scripts.tune_postrank_v23 import FEATURE_NAMES  # noqa: PLC0415

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
    wA, wB, wC, wD = weights["A"], weights["B"], weights["C"], weights["D"]
    for i, c in enumerate(cases):
        scores: dict[str, float] = {}
        for seq, w in (
            (a_lists[i], wA),
            (src_b[i], wB),
            (src_c[i], wC),
            (src_d[i], wD),
        ):
            if w == 0:
                continue
            for r, tid in enumerate(seq, start=1):
                scores[tid] = scores.get(tid, 0.0) + w / (k + r)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:POOL_K]
        sizes[i] = len(ranked)

        user_messages = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tokens = tokens(user_messages[-1]) if user_messages else set()
        all_user_tokens = tokens(" ".join(user_messages))
        played_set = set(played)
        last_artist = track_artist.get(played[-1], "") if played else ""
        last_tags = track_tags.get(played[-1], set()) if played else set()
        prior = [
            (1.0 / (j + 1), track_artist.get(tid, ""), track_tags.get(tid, set()))
            for j, tid in enumerate(reversed(played))
        ]

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
            for w_p, p_artist, p_tags in prior:
                artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                tag_match = (
                    len(cand_tags & p_tags) / len(cand_tags | p_tags)
                    if (cand_tags or p_tags) else 0.0
                )
                rec += w_p * (artist_match + tag_match)
            row[7] = rec
        if c["gt"] in ranked:
            gt_idx[i] = ranked.index(c["gt"])

    sessions = [c["session_id"] for c in cases]
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores: list[float] = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray(
                [i for i in range(len(sessions)) if i not in held],
                dtype=np.int64,
            )
            wts, _ = fit_weights(X, gt_idx, sizes, train)
            fold_scores.append(vec_ndcg(X, gt_idx, sizes, wts, fold))
        per_seed.append({
            "seed": seed,
            "cv5_mean": float(np.mean(fold_scores)),
            "cv5_std": float(np.std(fold_scores, ddof=1)),
            "cv5_per_fold": fold_scores,
        })
    means = [r["cv5_mean"] for r in per_seed]
    return {
        "weights": weights,
        "k": k,
        "pool_hit_rate": float(np.mean(gt_idx >= 0)),
        "per_seed": per_seed,
        "aggregate": {
            "mean_cv5": float(np.mean(means)),
            "std_of_cv5_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "min_cv5": float(np.min(means)),
            "max_cv5": float(np.max(means)),
        },
    }


# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    print("loading cached payload + A' variants", flush=True)
    with open(CACHE_FILE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    with open(A_PRIME_CACHE, "rb") as f:
        a_variants = pickle.load(f)  # noqa: S301
    print(f"  {len(payload['cases'])} cases loaded", flush=True)
    # NOTE: skip augment_metadata_for_a here — Stage 1 (pool-only) does not need
    # metadata at all (only RRF+pool_hit). Stage 3 will augment lazily, and only
    # for tids actually appearing in the selected configs' pools.

    # ---- Stage 1 ----
    print("\n=========================================")
    print("STAGE 1: pool-only sweep")
    print("=========================================", flush=True)
    stage1 = stage1_pool_sweep(payload, a_variants)
    print(f"\nStage 1 evaluated {len(stage1['configs'])} configs", flush=True)
    POOL_SWEEP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_SWEEP_OUT, "w") as f:
        json.dump(stage1, f, indent=2, default=float)

    # ---- Stage 2 ----
    print("\n=========================================")
    print("STAGE 2: filter + decide")
    print("=========================================", flush=True)
    stage2 = stage2_filter(stage1)
    print(f"  Decision: {stage2['decision']}", flush=True)
    print(f"  Reason: {stage2['reason']}", flush=True)
    # Append Stage 2 decision to artifact for transparency.
    stage1["stage2_decision"] = stage2
    with open(POOL_SWEEP_OUT, "w") as f:
        json.dump(stage1, f, indent=2, default=float)

    if stage2["decision"] != "PROCEED_TO_POWELL":
        print("\nStopping per spec gates. Final report follows main.", flush=True)
        # Save an empty Powell artifact so the structure exists.
        stub = {
            "skipped": True,
            "reason": stage2["reason"],
            "decision": stage2["decision"],
            "stage1_baseline": stage1["baseline"],
            "elapsed_sec": time.time() - t0,
        }
        with open(POWELL_OUT, "w") as f:
            json.dump(stub, f, indent=2, default=float)
        return

    # ---- Stage 3 ----
    print("\n=========================================")
    print("STAGE 3: Powell CV5 on top 3 (capped)")
    print("=========================================", flush=True)
    seeds = [0, 1, 2]
    stage3 = stage3_powell(payload, a_variants, stage2["selected_for_powell"], seeds)
    out = {
        "meta": {
            "n": len(payload["cases"]),
            "seeds": seeds,
            "baseline_pool_hit": stage1["baseline"]["pool_hit"],
            "baseline_mean_cv5": 0.1474,  # from r3_a_prime_400_top3_results.json
            "no_llm_calls": True,
            "elapsed_sec_total": time.time() - t0,
        },
        "stage2_decision": stage2,
        "powell": stage3,
    }
    with open(POWELL_OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {POOL_SWEEP_OUT}", flush=True)
    print(f"Wrote {POWELL_OUT}", flush=True)


if __name__ == "__main__":
    main()
