#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Experiment R1 — query→track semantic retrieval as a 5th source E.

ZERO-API. ZERO downloads (BGE-base-en-v1.5 already cached locally).
Reuses cached payload (exp/eval/_r3_confirm_400_deterministic.pkl) and cached
A' lists (exp/eval/_r3_a_prime_400.pkl). Reuses qwen3 vectors only if needed
(none of E's pipeline depends on them — A'/D recomputation in fusion uses them
through expA helpers).

Stages 2-6 of the original spec, all in one script:
  Stage 2: build E (BGE catalog index + per-session query embeddings) and
           retrieve top-1000 per session per variant. Persist.
  Stage 3: diagnostic — recall@50/100/500, unique-vs-cfg0209-union, top-50
           overlap, median GT rank when E hits.
  Stage 4: fusion sweep over (variant, depth, weight) on top of cfg0209.
  Stage 5: Powell CV5 (3 seeds, top 3 configs).
  Stage 6: gate decision + best-config robustness extension if PROMISING.

Hard caps: top 3 Powell, no auto-extension beyond seeds 0..4 (1 config only).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expA_fusion_sweep import (
    a_prime_deep_batch,
    fast_augment_metadata_for_a,
    load_qwen_index,
)
from scripts.r3_confirm_400_deterministic import (
    CACHE_FILE as R3_CACHE_FILE,
    POOL_K,
    cv_folds,
    fit_weights,
    tokens,
    vec_ndcg,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES

# ---------------------------------------------------------------------------
OUT_E_RETRIEVALS = REPO_ROOT / "exp" / "eval" / "_expR1_E_retrievals.pkl"
OUT_DIAG = REPO_ROOT / "exp" / "eval" / "expR1_diagnostic.json"
OUT_FUSION = REPO_ROOT / "exp" / "eval" / "expR1_fusion_sweep.json"
OUT_POWELL = REPO_ROOT / "exp" / "eval" / "expR1_powell_cv5.json"

A_PRIME_CACHE = REPO_ROOT / "exp" / "eval" / "_r3_a_prime_400.pkl"
META_CACHE = REPO_ROOT / "cache" / "metadata" / "track_metadata_all_tracks.json"

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_CACHE_DIR = REPO_ROOT / "cache" / "dense" / "bge-base-en-v1.5"
BGE_VECS_PATH = BGE_CACHE_DIR / "embeddings.npy"
BGE_IDS_PATH = BGE_CACHE_DIR / "track_ids.json"
# BGE official query prefix — no prefix on passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# cfg0209 spec (same source A as expB)
CFG0209 = {
    "depths": {"A": 200, "B": 500, "C": 500, "D": 200},
    "weights": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
    "k": 20,
    "variant": "A_max_recent_5",
}

# E retrieval depth (over-retrieve once, slice later for sweeps)
E_TOPK = 1000


# ---------------------------------------------------------------------------
def stringify_first(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return str(v)


def stringify_track_for_passage(meta: dict[str, Any]) -> str:
    """Compose a compact passage representation of a track for BGE encoding.

    Format follows the dense.py convention but tuned for BGE (no "passage:"
    prefix, just clean key:value text).
    """
    title = stringify_first(meta.get("track_name"))
    artist = stringify_first(meta.get("artist_name"))
    album = stringify_first(meta.get("album_name"))
    raw_tags = meta.get("tag_list") or []
    if isinstance(raw_tags, list):
        tag_str = ", ".join(str(t) for t in raw_tags[:25] if t)
    else:
        tag_str = ""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if artist:
        parts.append(f"Artist: {artist}")
    if album:
        parts.append(f"Album: {album}")
    if tag_str:
        parts.append(f"Tags: {tag_str}")
    return ". ".join(parts)


def stringify_history_track_short(meta: dict[str, Any]) -> str:
    """Shorter representation for query-side history concatenation."""
    title = stringify_first(meta.get("track_name"))
    artist = stringify_first(meta.get("artist_name"))
    raw_tags = meta.get("tag_list") or []
    if isinstance(raw_tags, list):
        tag_str = " ".join(str(t) for t in raw_tags[:6] if t)
    else:
        tag_str = ""
    bits = [b for b in [artist, title, tag_str] if b]
    return " ".join(bits)


# ---------------------------------------------------------------------------
def build_or_load_bge_catalog(
    metadata: dict[str, dict[str, Any]],
    *,
    batch_size: int = 64,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Encode the full catalog with BGE-base-en-v1.5 and cache to disk.

    Catalog text follows BGE convention: passages are bare metadata text,
    queries get a special prefix (handled in encode_queries).
    """
    BGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and BGE_VECS_PATH.exists() and BGE_IDS_PATH.exists():
        print(f"loading cached BGE catalog from {BGE_CACHE_DIR}", flush=True)
        vecs = np.load(BGE_VECS_PATH)
        with open(BGE_IDS_PATH) as f:
            ids = json.load(f)
        print(f"  catalog: {len(ids)} tracks, dim={vecs.shape[1]}", flush=True)
        return vecs, ids

    print(f"building BGE catalog index → {BGE_CACHE_DIR}", flush=True)
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  device: {device}", flush=True)
    encoder = SentenceTransformer(BGE_MODEL, device=device)

    # Sort track_ids deterministically for reproducibility
    track_ids = sorted(metadata.keys())
    print(f"  {len(track_ids)} tracks to encode", flush=True)
    passages = [stringify_track_for_passage(metadata[tid]) for tid in track_ids]

    t0 = time.time()
    embeddings = encoder.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.time() - t0
    print(f"  encoded in {elapsed:.1f}s ({len(track_ids) / max(elapsed, 1e-6):.1f} tracks/s)",
          flush=True)
    print(f"  shape: {embeddings.shape}", flush=True)

    np.save(BGE_VECS_PATH, embeddings)
    with open(BGE_IDS_PATH, "w") as f:
        json.dump(track_ids, f)
    return embeddings, track_ids


def encode_queries(queries: list[str], *, batch_size: int = 32) -> np.ndarray:
    """Encode queries with BGE query prefix. Normalised."""
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    encoder = SentenceTransformer(BGE_MODEL, device=device)
    prefixed = [BGE_QUERY_PREFIX + q for q in queries]
    vecs = encoder.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return vecs


def topk_search(catalog: np.ndarray, query_vecs: np.ndarray, topk: int) -> np.ndarray:
    """Vectorised top-k cosine search via inner product (vectors normalised).

    catalog:  (N, D)
    queries:  (Q, D)
    returns:  (Q, topk) integer index array (rows sorted by descending score).
    """
    # scores: (Q, N)
    # For 47k×768×400 the score matrix is ~120 MB float32 — fine in RAM.
    scores = query_vecs @ catalog.T
    # argpartition then sort
    idx_part = np.argpartition(-scores, topk - 1, axis=1)[:, :topk]
    # sort each row's topk by actual score descending
    out = np.empty_like(idx_part)
    for i in range(scores.shape[0]):
        row = idx_part[i]
        order = np.argsort(-scores[i, row])
        out[i] = row[order]
    return out


# ---------------------------------------------------------------------------
def build_E_retrievals(
    payload: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Stage 2: build BGE catalog, encode 2 query variants, retrieve top-1000.

    Persists to OUT_E_RETRIEVALS so Stage 3+ can reuse without re-encoding.
    """
    if OUT_E_RETRIEVALS.exists():
        print(f"loading cached E retrievals from {OUT_E_RETRIEVALS}", flush=True)
        with open(OUT_E_RETRIEVALS, "rb") as f:
            return pickle.load(f)  # noqa: S301

    cases = payload["cases"]
    n = len(cases)
    print(f"\n--- Stage 2: encoding catalog + {n} sessions × 2 variants ---",
          flush=True)
    t_total = time.time()

    catalog_vecs, catalog_ids = build_or_load_bge_catalog(metadata)

    # Build query strings for both variants
    queries_only: list[str] = []
    queries_plus: list[str] = []
    for c in cases:
        q = (c.get("user_query") or "").strip()
        queries_only.append(q if q else "music")
        # E_query_plus_history: query + last-5 played track metadata.
        played = c.get("music_turns") or []
        last5 = played[-5:]
        hist_strs = []
        for tid in last5:
            tmeta = metadata.get(str(tid))
            if tmeta:
                s = stringify_history_track_short(tmeta)
                if s:
                    hist_strs.append(s)
        if hist_strs:
            combined = q + " || Recent listens: " + "; ".join(hist_strs)
        else:
            combined = q if q else "music"
        queries_plus.append(combined)

    print(f"  encoding query variant E_query_only ({n})", flush=True)
    t0 = time.time()
    qo_vecs = encode_queries(queries_only)
    print(f"    done in {time.time() - t0:.1f}s, shape={qo_vecs.shape}", flush=True)

    print(f"  encoding query variant E_query_plus_history ({n})", flush=True)
    t0 = time.time()
    qp_vecs = encode_queries(queries_plus)
    print(f"    done in {time.time() - t0:.1f}s, shape={qp_vecs.shape}", flush=True)

    print(f"  searching catalog top-{E_TOPK} for both variants", flush=True)
    t0 = time.time()
    qo_idx = topk_search(catalog_vecs, qo_vecs, E_TOPK)
    qp_idx = topk_search(catalog_vecs, qp_vecs, E_TOPK)
    print(f"    search done in {time.time() - t0:.1f}s", flush=True)

    E_query_only = [[catalog_ids[i] for i in row] for row in qo_idx]
    E_query_plus_history = [[catalog_ids[i] for i in row] for row in qp_idx]

    out = {
        "encoder": BGE_MODEL,
        "topk": E_TOPK,
        "n_sessions": n,
        "catalog_size": len(catalog_ids),
        "query_strings_only": queries_only,
        "query_strings_plus": queries_plus,
        "E_query_only": E_query_only,
        "E_query_plus_history": E_query_plus_history,
        "elapsed_total_sec": time.time() - t_total,
    }
    OUT_E_RETRIEVALS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_E_RETRIEVALS, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_E_RETRIEVALS}", flush=True)
    print(f"  Stage 2 total: {out['elapsed_total_sec']:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
def rrf_fuse(seqs_with_weights: list[tuple[list[str], float]], k: int, pool_size: int) -> list[str]:
    scores: dict[str, float] = {}
    for seq, w in seqs_with_weights:
        if w == 0:
            continue
        for r, tid in enumerate(seq, start=1):
            scores[tid] = scores.get(tid, 0.0) + w / (k + r)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:pool_size]


def build_cfg0209_pools_and_unions(
    cases: list[dict[str, Any]],
    a_lists: list[list[str]],
    src_b: list[list[str]],
    src_c: list[list[str]],
    src_d: list[list[str]],
) -> tuple[list[list[str]], list[set[str]], dict[str, float]]:
    """Compute cfg0209 RRF pool@50 and union(A',B,C,D) for diagnostic.

    Returns: (pools, unions, baseline_metrics).
    """
    depths = CFG0209["depths"]
    weights = CFG0209["weights"]
    k = CFG0209["k"]
    pools = []
    unions = []
    hits = 0
    ranks: list[int] = []
    for i, c in enumerate(cases):
        a_l = a_lists[i][: depths["A"]]
        b_l = src_b[i][: depths["B"]]
        c_l = src_c[i][: depths["C"]]
        d_l = src_d[i][: depths["D"]]
        seqs = [
            (a_l, weights["A"]),
            (b_l, weights["B"]),
            (c_l, weights["C"]),
            (d_l, weights["D"]),
        ]
        pool = rrf_fuse(seqs, k=k, pool_size=POOL_K)
        pools.append(pool)
        union = set(a_l) | set(b_l) | set(c_l) | set(d_l)
        unions.append(union)
        gt = c["gt"]
        if gt in pool:
            hits += 1
            ranks.append(pool.index(gt))
    metrics = {
        "pool_hit_at_50": hits / len(cases) if cases else 0.0,
        "median_rank_when_hit": float(np.median(ranks)) if ranks else float("nan"),
    }
    return pools, unions, metrics


def diagnostic_for_E(
    cases: list[dict[str, Any]],
    E: list[list[str]],
    cfg_pools: list[list[str]],
    cfg_unions: list[set[str]],
) -> dict[str, Any]:
    """Compute the Stage 3 diagnostic metrics for one E variant."""
    n = len(cases)
    hit50 = hit100 = hit500 = 0
    median_ranks: list[int] = []
    unique_vs_union = 0  # GTs found in E top-500 but NOT in cfg0209 union(A',B,C,D)
    overlap_top50: list[float] = []
    e_in_pool_only = 0  # diag: how often does E top-50 contain GT but cfg0209 pool doesn't
    cfg_in_pool_only = 0
    both_have = 0
    for i, c in enumerate(cases):
        gt = c["gt"]
        e_top500 = E[i][:500]
        e_top100 = E[i][:100]
        e_top50 = E[i][:50]
        if gt in e_top50:
            hit50 += 1
        if gt in e_top100:
            hit100 += 1
        if gt in e_top500:
            hit500 += 1
            median_ranks.append(e_top500.index(gt))
        if gt in e_top500 and gt not in cfg_unions[i]:
            unique_vs_union += 1
        # overlap E top-50 with cfg0209 pool top-50
        sa = set(e_top50)
        sb = set(cfg_pools[i])
        union_ab = sa | sb
        if union_ab:
            overlap_top50.append(len(sa & sb) / len(union_ab))
        # GT-in-pool stats
        in_e = gt in e_top50
        in_c = gt in cfg_pools[i]
        if in_e and not in_c:
            e_in_pool_only += 1
        if in_c and not in_e:
            cfg_in_pool_only += 1
        if in_e and in_c:
            both_have += 1
    return {
        "n": n,
        "recall_at_50": hit50 / n,
        "recall_at_100": hit100 / n,
        "recall_at_500": hit500 / n,
        "unique_vs_cfg0209_union_count": unique_vs_union,
        "unique_vs_cfg0209_union_rate": unique_vs_union / n,
        "median_rank_when_hit_in_E_top500": (
            float(np.median(median_ranks)) if median_ranks else float("nan")
        ),
        "overlap_E50_with_cfg0209_pool50": float(np.mean(overlap_top50)) if overlap_top50 else 0.0,
        "e_top50_only_gt_count": e_in_pool_only,
        "cfg_pool50_only_gt_count": cfg_in_pool_only,
        "both_pools_have_gt_count": both_have,
    }


# ---------------------------------------------------------------------------
def fusion_sweep(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    E_retrievals: dict[str, Any],
    cfg_baseline_pools: list[list[str]],
    *,
    e_depths: list[int],
    e_weights: list[float],
    variants: list[str],
) -> dict[str, Any]:
    """Stage 4: add E as 5th source. Sweep (variant, E depth, E weight).

    cfg0209 (A,B,C,D) sources/depths/weights are FIXED. RRF k=20, pool=50.
    Compute pool_hit, median_rank_when_hit, overlap_with_cfg0209_baseline.
    """
    cases = payload["cases"]
    src_b = payload["src_b"]
    src_c = payload["src_c"]
    src_d = payload["src_d"]
    depths = CFG0209["depths"]
    weights = CFG0209["weights"]
    k = CFG0209["k"]
    n = len(cases)

    configs: list[dict[str, Any]] = []
    for variant in variants:
        E_full = E_retrievals[variant]
        for ed in e_depths:
            for ew in e_weights:
                hits = 0
                ranks: list[int] = []
                overlaps: list[float] = []
                for i in range(n):
                    a_l = a_lists[i][: depths["A"]]
                    b_l = src_b[i][: depths["B"]]
                    c_l = src_c[i][: depths["C"]]
                    d_l = src_d[i][: depths["D"]]
                    e_l = E_full[i][:ed]
                    seqs = [
                        (a_l, weights["A"]),
                        (b_l, weights["B"]),
                        (c_l, weights["C"]),
                        (d_l, weights["D"]),
                        (e_l, ew),
                    ]
                    pool = rrf_fuse(seqs, k=k, pool_size=POOL_K)
                    gt = cases[i]["gt"]
                    if gt in pool:
                        hits += 1
                        ranks.append(pool.index(gt))
                    base = set(cfg_baseline_pools[i])
                    pool_set = set(pool)
                    union_ab = pool_set | base
                    if union_ab:
                        overlaps.append(len(pool_set & base) / len(union_ab))
                configs.append({
                    "variant": variant,
                    "E_depth": ed,
                    "E_weight": ew,
                    "pool_hit": hits / n,
                    "median_rank_when_hit": float(np.median(ranks)) if ranks else float("nan"),
                    "median_rank_p25": float(np.percentile(ranks, 25)) if ranks else float("nan"),
                    "median_rank_p75": float(np.percentile(ranks, 75)) if ranks else float("nan"),
                    "overlap_with_cfg0209_pool": float(np.mean(overlaps)) if overlaps else 0.0,
                })
    return {
        "fixed": {"depths_ABCD": depths, "weights_ABCD": weights, "k": k, "pool_size": POOL_K},
        "configs": configs,
    }


# ---------------------------------------------------------------------------
def build_pool_features_5src(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    e_lists: list[list[str]],
    *,
    weights_5: tuple[float, float, float, float, float],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replica of build_pool_features (in r3_confirm) extended for 5 sources A,B,C,D,E.

    Uses the canonical 8 postrank features (orig_rank, etc.).
    """
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
    wA, wB, wC, wD, wE = weights_5
    depths = CFG0209["depths"]
    for i, c in enumerate(cases):
        scores: dict[str, float] = {}
        for seq, w in (
            (a_lists[i][: depths["A"]], wA),
            (src_b[i][: depths["B"]], wB),
            (src_c[i][: depths["C"]], wC),
            (src_d[i][: depths["D"]], wD),
            (e_lists[i], wE),
        ):
            if w == 0:
                continue
            for rank, tid in enumerate(seq, start=1):
                scores[tid] = scores.get(tid, 0.0) + w / (k + rank)
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
    return X, gt_idx, sizes


def powell_cv5(
    payload: dict[str, Any],
    a_lists: list[list[str]],
    e_lists: list[list[str]],
    *,
    weights_5: tuple[float, float, float, float, float],
    k: int,
    seeds: list[int],
) -> dict[str, Any]:
    X, gt_idx, sizes = build_pool_features_5src(payload, a_lists, e_lists,
                                                weights_5=weights_5, k=k)
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
            wts, _ = fit_weights(X, gt_idx, sizes, train)
            fold_scores.append(vec_ndcg(X, gt_idx, sizes, wts, fold))
        per_seed.append({
            "seed": seed,
            "cv5_mean": float(np.mean(fold_scores)),
            "cv5_std": float(np.std(fold_scores, ddof=1)),
            "cv5_per_fold": fold_scores,
        })
        print(f"      seed {seed}: cv5={per_seed[-1]['cv5_mean']:.4f} ± {per_seed[-1]['cv5_std']:.4f}",
              flush=True)
    means = [r["cv5_mean"] for r in per_seed]
    return {
        "weights_5": list(weights_5),
        "k": k,
        "pool_hit": float(np.mean(gt_idx >= 0)),
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
    t_total = time.time()
    print("=" * 80)
    print("Experiment R1: query→track semantic retrieval (BGE-base) as 5th source E")
    print("=" * 80, flush=True)

    # ---- Load payload + A' (cfg0209 uses A_max_recent_5 @ 200) ----
    with open(R3_CACHE_FILE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    print(f"loaded payload: {len(cases)} cases", flush=True)

    # NOTE: A_PRIME_CACHE has @50 lists only; cfg0209 needs @200 → recompute.
    print("recomputing A_max_recent_5 @ depth 200 (cfg0209 spec)", flush=True)
    qvecs, qid_to_idx, qids = load_qwen_index()
    a_lists_200 = a_prime_deep_batch(qvecs, qid_to_idx, qids, cases, n=5, k=200)
    print(f"  A' lists ready ({len(a_lists_200)} sessions)", flush=True)
    # Augment metadata for any new tids in A_lists (Powell needs feature dicts)
    fast_augment_metadata_for_a(payload, a_lists_200)

    # ---- Load track metadata (for catalog encoding & history strings) ----
    print(f"loading track metadata cache from {META_CACHE}", flush=True)
    with open(META_CACHE) as f:
        metadata = json.load(f)
    print(f"  {len(metadata)} tracks", flush=True)

    # ---- Stage 2: build E retrievals (cached) ----
    E_data = build_E_retrievals(payload, metadata)

    # ---- Stage 3: diagnostic ----
    print("\n" + "=" * 80)
    print("Stage 3: diagnostic")
    print("=" * 80, flush=True)
    cfg_pools, cfg_unions, cfg_metrics = build_cfg0209_pools_and_unions(
        cases, a_lists_200, payload["src_b"], payload["src_c"], payload["src_d"]
    )
    print(f"cfg0209 baseline pool_hit@50 = {cfg_metrics['pool_hit_at_50']:.4f}",
          flush=True)
    print(f"cfg0209 baseline median GT rank when hit = {cfg_metrics['median_rank_when_hit']}",
          flush=True)

    diag_results: dict[str, dict[str, Any]] = {}
    for variant in ("E_query_only", "E_query_plus_history"):
        d = diagnostic_for_E(cases, E_data[variant], cfg_pools, cfg_unions)
        diag_results[variant] = d
        print(f"\n[{variant}]", flush=True)
        for k_, v_ in d.items():
            print(f"  {k_}: {v_}", flush=True)

    # Apply gate: redundant?
    gate_redundant = []
    for variant, d in diag_results.items():
        if (d["unique_vs_cfg0209_union_rate"] < 0.03
                and d["overlap_E50_with_cfg0209_pool50"] > 0.6):
            gate_redundant.append(variant)
    promising_variants = [
        v for v, d in diag_results.items()
        if not (d["unique_vs_cfg0209_union_rate"] < 0.03
                and d["overlap_E50_with_cfg0209_pool50"] > 0.6)
    ]
    diag_out = {
        "cfg0209_baseline": cfg_metrics,
        "diagnostic": diag_results,
        "gate_redundant_variants": gate_redundant,
        "promising_variants": promising_variants,
    }
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIAG, "w") as f:
        json.dump(diag_out, f, indent=2, default=float)
    print(f"\nwrote {OUT_DIAG}", flush=True)

    if not promising_variants:
        print("\nGATE: BOTH variants are REDUNDANT vs cfg0209. Stopping at Stage 3.",
              flush=True)
        # Still write empty downstream artifacts so report has known paths
        with open(OUT_FUSION, "w") as f:
            json.dump({"skipped": True, "reason": "Both variants gated as redundant",
                       "diagnostic": diag_results}, f, indent=2)
        with open(OUT_POWELL, "w") as f:
            json.dump({"skipped": True, "reason": "Both variants gated as redundant"},
                      f, indent=2)
        print(f"\nTotal: {time.time() - t_total:.1f}s", flush=True)
        return

    # ---- Stage 4: fusion sweep ----
    print("\n" + "=" * 80)
    print("Stage 4: fusion sweep on top of cfg0209")
    print("=" * 80, flush=True)
    e_depths = [100, 200, 500, 1000]
    e_weights_grid = [0.25, 0.5, 1.0, 2.0, 4.0]
    print(f"  variants={promising_variants}  depths={e_depths}  weights={e_weights_grid}",
          flush=True)
    sweep = fusion_sweep(
        payload, a_lists_200, E_data, cfg_pools,
        e_depths=e_depths, e_weights=e_weights_grid, variants=promising_variants,
    )

    base_pool = cfg_metrics["pool_hit_at_50"]
    base_med = cfg_metrics["median_rank_when_hit"]
    # Sort: pool_hit DESC primary, median_rank ASC tiebreak, overlap ASC tertiary
    def sort_key(c):
        mrk = c["median_rank_when_hit"] if c["median_rank_when_hit"] == c["median_rank_when_hit"] else 1e9
        ov = c["overlap_with_cfg0209_pool"] if c["overlap_with_cfg0209_pool"] == c["overlap_with_cfg0209_pool"] else 1.0
        return (-c["pool_hit"], mrk, ov)
    sweep["configs"].sort(key=sort_key)
    top5 = sweep["configs"][:5]
    print(f"\n  cfg0209 baseline pool_hit@50 = {base_pool:.4f}", flush=True)
    print(f"  cfg0209 baseline median_rank_when_hit = {base_med}", flush=True)
    print("  Top 5 configs by combined metric:", flush=True)
    for c in top5:
        delta = c["pool_hit"] - base_pool
        print(f"    {c['variant']} | E_d={c['E_depth']:>4} | E_w={c['E_weight']:>4} | "
              f"pool_hit={c['pool_hit']:.4f} (Δ={delta:+.4f}) | "
              f"med_rank={c['median_rank_when_hit']} | "
              f"overlap={c['overlap_with_cfg0209_pool']:.3f}", flush=True)
    sweep["top5"] = top5
    sweep["cfg0209_baseline"] = cfg_metrics
    with open(OUT_FUSION, "w") as f:
        json.dump(sweep, f, indent=2, default=float)
    print(f"\nwrote {OUT_FUSION}", flush=True)

    # Pre-Powell guard: if NO config beats baseline pool_hit by ≥ 0.005, skip Powell.
    strong = [c for c in sweep["configs"] if c["pool_hit"] >= base_pool + 0.005]
    if not strong:
        print("\nGATE: No config improves pool_hit@50 by ≥+0.005 over cfg0209. "
              "Skipping Powell — would not improve nDCG.", flush=True)
        with open(OUT_POWELL, "w") as f:
            json.dump({
                "skipped": True,
                "reason": (f"No fusion config beats cfg0209 pool_hit@50 "
                           f"({base_pool:.4f}) by ≥+0.005."),
                "top5_pool_only": top5,
            }, f, indent=2, default=float)
        print(f"\nTotal: {time.time() - t_total:.1f}s", flush=True)
        return

    # ---- Stage 5: Powell CV5 on top 3 ----
    print("\n" + "=" * 80)
    print("Stage 5: Powell CV5 (3 seeds × top 3 configs)")
    print("=" * 80, flush=True)
    seeds = [0, 1, 2]
    top3_for_powell = top5[:3]
    powell_results: dict[str, dict[str, Any]] = {}
    for cfg in top3_for_powell:
        variant = cfg["variant"]
        ed = cfg["E_depth"]
        ew = cfg["E_weight"]
        e_lists = [e[:ed] for e in E_data[variant]]
        weights_5 = (
            CFG0209["weights"]["A"], CFG0209["weights"]["B"],
            CFG0209["weights"]["C"], CFG0209["weights"]["D"], ew,
        )
        cfg_id = f"{variant}_d{ed}_w{ew}"
        print(f"\n  [{cfg_id}] running Powell CV5", flush=True)
        res = powell_cv5(payload, a_lists_200, e_lists,
                         weights_5=weights_5, k=CFG0209["k"], seeds=seeds)
        agg = res["aggregate"]
        delta = agg["mean_cv5"] - 0.1602
        print(f"    mean_cv5={agg['mean_cv5']:.4f} ± {agg['std_of_cv5_means']:.4f} "
              f"(Δ vs cfg0209 0.1602 = {delta:+.4f})  pool_hit={res['pool_hit']:.4f}",
              flush=True)
        powell_results[cfg_id] = {"sweep_config": cfg, **res}

    # Pick best by mean_cv5
    best_id = max(powell_results, key=lambda c: powell_results[c]["aggregate"]["mean_cv5"])
    best = powell_results[best_id]
    best_mean = best["aggregate"]["mean_cv5"]
    best_std = best["aggregate"]["std_of_cv5_means"]
    best_pool = best["pool_hit"]
    print(f"\n  best: {best_id}  mean_cv5={best_mean:.4f} ± {best_std:.4f}  pool_hit={best_pool:.4f}",
          flush=True)

    # ---- Stage 5b: optionally extend best to seeds 0..4 ----
    extended_seeds_used = list(seeds)
    promising_threshold = 0.170
    pool_promising = 0.4175
    if best_mean >= promising_threshold or best_pool >= pool_promising:
        extra_seeds = [3, 4]
        cfg = best["sweep_config"]
        variant = cfg["variant"]
        ed = cfg["E_depth"]
        ew = cfg["E_weight"]
        e_lists = [e[:ed] for e in E_data[variant]]
        weights_5 = (
            CFG0209["weights"]["A"], CFG0209["weights"]["B"],
            CFG0209["weights"]["C"], CFG0209["weights"]["D"], ew,
        )
        print(f"\n  extending best ({best_id}) to seeds {extra_seeds}", flush=True)
        ext = powell_cv5(payload, a_lists_200, e_lists,
                         weights_5=weights_5, k=CFG0209["k"], seeds=extra_seeds)
        extended_seeds_used = list(seeds) + extra_seeds
        all_means = (
            [s["cv5_mean"] for s in best["per_seed"]]
            + [s["cv5_mean"] for s in ext["per_seed"]]
        )
        ext_summary = {
            "extended_to_seeds": extended_seeds_used,
            "all_means": all_means,
            "mean_cv5_5seeds": float(np.mean(all_means)),
            "std_of_cv5_means_5seeds": float(np.std(all_means, ddof=1)) if len(all_means) > 1 else 0.0,
            "min_cv5_5seeds": float(np.min(all_means)),
            "max_cv5_5seeds": float(np.max(all_means)),
        }
        print(f"    5-seed mean_cv5={ext_summary['mean_cv5_5seeds']:.4f} "
              f"± {ext_summary['std_of_cv5_means_5seeds']:.4f}", flush=True)
        powell_results[best_id]["extended_5seeds"] = {
            "extra_seeds_results": ext,
            "summary": ext_summary,
        }

    out_powell = {
        "meta": {
            "n": len(cases),
            "seeds": seeds,
            "extended_seeds_used": extended_seeds_used,
            "encoder": BGE_MODEL,
            "cfg0209_baseline_mean_cv5": 0.1602,
            "cfg0209_baseline_pool_hit": base_pool,
            "no_llm_calls": True,
            "elapsed_total_sec": time.time() - t_total,
        },
        "best_config_id": best_id,
        "results": powell_results,
    }
    with open(OUT_POWELL, "w") as f:
        json.dump(out_powell, f, indent=2, default=float)
    print(f"\nwrote {OUT_POWELL}", flush=True)
    print(f"\nTotal: {time.time() - t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
