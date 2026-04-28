#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Deterministic A' substitute experiment on the 400-row R3 slice.

Tests whether deterministic A' candidate sources (built from cached qwen3 track
embeddings) can recover the pool_hit_rate boost that the v23 LLM rerank
provides on the 200-row slice.

Sources:
  A_last:               top-50 cosine to LAST played track (sanity: overlaps D).
  A_centroid_recent_N:  recency-weighted centroid of last-N played; N in {3,5,all}.
                        Two weight schemes: uniform mean and exponential decay 0.7^k.
  A_max_recent_N:       top-50 by max(cosine_sim(cand, played_i)) for last N; N in {3,5}.

A_query (free-text user-query embedding) is SKIPPED — the qwen3 track vectors
are an opaque cached dataset; no compatible local text encoder ships with this
repo, and using a foreign embedding space (e.g. bge/mpnet) against qwen3 would
produce nonsense cosines. See cache verification preamble for details.

For each A' variant × weight in {0.5, 1, 2}, runs the same 5-fold × 3-seed
Powell CV5 protocol as scripts/r3_confirm_400_deterministic.py over the
weighted-RRF top-50 pool with B=2, C=0.5, D=1 fixed.

ZERO API CALLS. NO MODEL DOWNLOADS. Uses only:
  - cache/track_sim/metadata-qwen3_embedding_0.6b/{vectors.npy, track_ids.json}
  - exp/eval/_r3_confirm_400_deterministic.pkl (preloaded B/C/D + features)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (  # noqa: E402
    CACHE_FILE,
    POOL_K,
    cv_folds,
    fit_weights,
    tokens,
    vec_ndcg,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES  # noqa: E402

A_PRIME_TOPK = 50
TRACK_VECS = "cache/track_sim/metadata-qwen3_embedding_0.6b/vectors.npy"
TRACK_IDS = "cache/track_sim/metadata-qwen3_embedding_0.6b/track_ids.json"
A_PRIME_CACHE = "exp/eval/_r3_a_prime_400.pkl"
RESULTS_FILE = "exp/eval/r3_a_prime_400_results.json"

# Fixed B/C/D source weights from the R3 baseline (det_A0_B2_C05_D1).
W_B, W_C, W_D = 2.0, 0.5, 1.0
A_WEIGHTS = [0.5, 1.0, 2.0]


def load_track_index() -> tuple[np.ndarray, dict[str, int]]:
    if not (os.path.exists(TRACK_VECS) and os.path.exists(TRACK_IDS)):
        raise FileNotFoundError("Missing cached qwen3 track-sim index")
    vecs = np.load(TRACK_VECS).astype(np.float32, copy=False)
    with open(TRACK_IDS) as f:
        ids = json.load(f)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    return vecs, id_to_idx


def topk_by_query(vecs: np.ndarray, query_vec: np.ndarray, exclude_idxs: set[int], k: int) -> list[int]:
    """Return indices of top-k tracks by cosine similarity to *query_vec*.

    Vectors are pre-normalized, so cosine == inner product.  *query_vec* must
    also be unit-norm.  Excluded indices are removed from results.
    """
    scores = vecs @ query_vec
    # Fetch a few extra so we can drop excludes without going short.
    n = min(k + len(exclude_idxs) + 1, len(scores))
    cand = np.argpartition(-scores, n - 1)[:n]
    cand = cand[np.argsort(-scores[cand])]
    return [int(i) for i in cand if i not in exclude_idxs][:k]


def dense_topk_for_session(
    vecs: np.ndarray,
    id_to_idx: dict[str, int],
    music_turns: list[str],
    *,
    variant: str,
    n: int | None,
    decay: float | None,
    k: int = A_PRIME_TOPK,
) -> list[str]:
    """Build a single A' candidate list of size <=k for one session.

    variant ∈ {'last','centroid','max'}.
    For 'centroid': n = window length, decay = None for uniform, otherwise base
        of exponential weight (e.g. 0.7).
    For 'max': n = window length.
    """
    played_idxs = [id_to_idx[t] for t in music_turns if t in id_to_idx]
    if not played_idxs:
        return []
    exclude = set(played_idxs)

    if variant == "last":
        last_vec = vecs[played_idxs[-1]]
        idxs = topk_by_query(vecs, last_vec, exclude, k)
    elif variant == "centroid":
        window = played_idxs if n is None else played_idxs[-n:]
        m = len(window)
        if decay is None:
            weights = np.ones(m, dtype=np.float32)
        else:
            # weight i = decay^(m-1-i) so the most recent gets weight 1.
            weights = np.array([decay ** (m - 1 - i) for i in range(m)], dtype=np.float32)
        weights = weights / weights.sum()
        cent = (weights[:, None] * vecs[window]).sum(axis=0)
        norm = float(np.linalg.norm(cent))
        if norm == 0.0:
            return []
        cent = cent / norm
        idxs = topk_by_query(vecs, cent.astype(np.float32, copy=False), exclude, k)
    elif variant == "max":
        window = played_idxs if n is None else played_idxs[-n:]
        sims = vecs @ vecs[window].T  # (V, |window|)
        agg = sims.max(axis=1)
        for j in exclude:
            agg[j] = -1.0
        nn = min(k + 1, len(agg))
        cand = np.argpartition(-agg, nn - 1)[:nn]
        cand = cand[np.argsort(-agg[cand])]
        idxs = [int(i) for i in cand if i not in exclude][:k]
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return [id_to_idx_inverse[idx] for idx in idxs]  # filled by caller via closure


# ----------------------------------------------------------------------------
# A' variant registry — name → (variant, n, decay)
A_PRIME_VARIANTS: dict[str, tuple[str, int | None, float | None]] = {
    "A_last": ("last", None, None),
    "A_centroid_recent_3_mean": ("centroid", 3, None),
    "A_centroid_recent_3_decay": ("centroid", 3, 0.7),
    "A_centroid_recent_5_mean": ("centroid", 5, None),
    "A_centroid_recent_5_decay": ("centroid", 5, 0.7),
    "A_centroid_recent_all_mean": ("centroid", None, None),
    "A_centroid_recent_all_decay": ("centroid", None, 0.7),
    "A_max_recent_3": ("max", 3, None),
    "A_max_recent_5": ("max", 5, None),
}


def build_a_prime_lists(payload: dict[str, Any]) -> dict[str, list[list[str]]]:
    """Build (and cache) per-session A' top-50 lists for every registered variant."""
    cache_path = Path(A_PRIME_CACHE)
    if cache_path.exists():
        print(f"loading A' cache {A_PRIME_CACHE}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)  # noqa: S301

    print("loading qwen3 track index", flush=True)
    vecs, id_to_idx = load_track_index()
    track_ids_list: list[str] = [""] * len(id_to_idx)
    for tid, i in id_to_idx.items():
        track_ids_list[i] = tid

    # Inject the inverse mapping into the helper module-globally (needed by closure).
    global id_to_idx_inverse  # noqa: PLW0603
    id_to_idx_inverse = track_ids_list

    cases = payload["cases"]
    out: dict[str, list[list[str]]] = {name: [] for name in A_PRIME_VARIANTS}
    t0 = time.time()
    for i, c in enumerate(cases):
        music_turns = c["music_turns"]
        for name, (variant, n, decay) in A_PRIME_VARIANTS.items():
            out[name].append(
                dense_topk_for_session(
                    vecs, id_to_idx, music_turns,
                    variant=variant, n=n, decay=decay, k=A_PRIME_TOPK,
                )
            )
        if (i + 1) % 50 == 0:
            print(f"  A' build {i + 1}/{len(cases)}  ({time.time() - t0:.1f}s)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ----------------------------------------------------------------------------
def build_pool_features_with_a(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    weights_abcd: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as r3_confirm_400_deterministic.build_pool_features but adds A' source."""
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
    w_a, w_b, w_c, w_d = weights_abcd
    for i, c in enumerate(cases):
        scores: dict[str, float] = {}
        for seq, weight in (
            (a_lists[i], w_a),
            (src_b[i], w_b),
            (src_c[i], w_c),
            (src_d[i], w_d),
        ):
            if weight == 0:
                continue
            for rank, tid in enumerate(seq, start=1):
                scores[tid] = scores.get(tid, 0.0) + weight / (60.0 + rank)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:POOL_K]
        sizes[i] = len(ranked)

        # Need feature maps for ANY tid that appears in pool — build on the fly
        # from existing payload when available; missing tids get default zeros.
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
            for weight, p_artist, p_tags in prior:
                artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                tag_match = (
                    len(cand_tags & p_tags) / len(cand_tags | p_tags)
                    if (cand_tags or p_tags) else 0.0
                )
                rec += weight * (artist_match + tag_match)
            row[7] = rec
        if c["gt"] in ranked:
            gt_idx[i] = ranked.index(c["gt"])
    return X, gt_idx, sizes


def augment_metadata_for_a(payload: dict[str, Any], a_variants: dict[str, list[list[str]]]) -> None:
    """Make sure new tids introduced by A' have entries in the metadata feature maps."""
    from offline_retrieval_sweep import load_track_metadata  # noqa: PLC0415

    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    new_tids: set[str] = set()
    for lst in a_variants.values():
        for sub in lst:
            for tid in sub:
                if tid not in track_artist:
                    new_tids.add(tid)
    if not new_tids:
        return
    metadata = load_track_metadata()
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
    print(f"augmented metadata for {len(new_tids)} additional A' tids", flush=True)


def eval_one(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    w_a: float,
    seeds: list[int],
) -> dict[str, Any]:
    X, gt_idx, sizes = build_pool_features_with_a(
        payload, a_lists, weights_abcd=(w_a, W_B, W_C, W_D),
    )
    sessions = [c["session_id"] for c in payload["cases"]]
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
            weights, _ = fit_weights(X, gt_idx, sizes, train)
            fold_scores.append(vec_ndcg(X, gt_idx, sizes, weights, fold))
        per_seed.append({
            "seed": seed,
            "cv5_mean": float(np.mean(fold_scores)),
            "cv5_std": float(np.std(fold_scores, ddof=1)),
            "cv5_per_fold": fold_scores,
        })
    means = [r["cv5_mean"] for r in per_seed]
    return {
        "weights_ABCD": [w_a, W_B, W_C, W_D],
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
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing 400-row payload at {CACHE_FILE}")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    print(f"loaded {len(payload['cases'])} cases", flush=True)

    a_variants = build_a_prime_lists(payload)
    augment_metadata_for_a(payload, a_variants)

    seeds = [0, 1, 2]
    out: dict[str, Any] = {
        "meta": {
            "n": len(payload["cases"]),
            "seeds": seeds,
            "fixed_BCD_weights": [W_B, W_C, W_D],
            "a_weights": A_WEIGHTS,
            "variants": list(a_variants.keys()),
            "a_query_skipped_reason": (
                "no local text-embedding model compatible with cached qwen3 vectors"
            ),
            "no_llm_calls": True,
        },
        "results": {},
    }

    for v_name in a_variants:
        for w_a in A_WEIGHTS:
            key = f"{v_name}__wA{w_a}"
            print(f"\n[{key}] eval start", flush=True)
            res = eval_one(payload, a_variants[v_name], w_a, seeds)
            out["results"][key] = {"variant": v_name, **res}
            agg = res["aggregate"]
            print(
                f"[{key}] mean_cv5={agg['mean_cv5']:.4f}  "
                f"std={agg['std_of_cv5_means']:.4f}  "
                f"pool_hit={res['pool_hit_rate']:.4f}",
                flush=True,
            )
            # Persist incrementally so partial results survive interruption.
            Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
            out["meta"]["elapsed_sec"] = time.time() - t0
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    out["meta"]["elapsed_sec"] = time.time() - t0
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_FILE}  elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
