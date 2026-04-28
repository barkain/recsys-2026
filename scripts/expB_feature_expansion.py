#!/usr/bin/env python3
# ruff: noqa: E402,T201
# pyright: reportMissingImports=false
"""Experiment B — postrank feature expansion on the cfg0209 fusion baseline.

ZERO-API. No openai/anthropic imports. No LLM calls. No blind code.

Stages:
  1. Build the cfg0209 pool features (base 8 + candidate features). Cache to
     exp/eval/_expB_features.pkl.
  2. Univariate diagnostics (top-decile lift) → exp/eval/expB_univariate_diagnostics.json.
  3. Powell CV5 (3 seeds) on grouped feature models M0..M5 → exp/eval/expB_powell_models.json.
  4. Leave-one-group-out ablation on M5 → exp/eval/expB_ablation.json.
  5. Robustness check: M5 with seeds 5..9 (5 extra seeds, 8 total).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expA_fusion_sweep import (
    a_prime_deep_batch,
    fast_augment_metadata_for_a,
    load_qwen_index,
    rrf_fuse,
)
from scripts.r3_confirm_400_deterministic import (
    CACHE_FILE,
    POOL_K,
    cv_folds,
    tokens,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES as BASE_FEATURE_NAMES
from scripts.tune_postrank_v23 import INIT_WEIGHTS as BASE_INIT_WEIGHTS

OUT_FEATURES = REPO_ROOT / "exp" / "eval" / "_expB_features.pkl"
OUT_DIAG = REPO_ROOT / "exp" / "eval" / "expB_univariate_diagnostics.json"
OUT_MODELS = REPO_ROOT / "exp" / "eval" / "expB_powell_models.json"
OUT_ABLATION = REPO_ROOT / "exp" / "eval" / "expB_ablation.json"
OUT_FINAL = REPO_ROOT / "exp" / "eval" / "expB_final_report.json"

QWEN_VECS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy"
QWEN_IDS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"

# cfg0209 specification (from expA_fusion_powell_top3.json)
CFG0209 = {
    "depths": {"A": 200, "B": 500, "C": 500, "D": 200},
    "weights": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
    "k": 20,
    "variant": "A_max_recent_5",
}


# ---------------------------------------------------------------------------
# Feature group definitions
GROUP1_FEATURES = [
    "in_Aprime", "in_B", "in_C", "in_D",
    "rr_Aprime", "rr_B", "rr_C", "rr_D",
    "source_count",
]
GROUP2_FEATURES = [
    "max_sim_recent5", "mean_sim_recent5", "sim_last", "sim_centroid_recent5", "sim_margin",
]
GROUP3_FEATURES = [
    "already_artist_played_count", "already_tag_overlap_max",
    "candidate_artist_recent_frequency", "novelty_vs_recent_tracks",
]
GROUP4_FEATURES = [
    "tag_token_overlap_query", "album_token_overlap_query",
    "artist_or_title_exact_from_query",
]
GROUP5_FEATURES = [
    "artist_frequency_in_catalog", "tag_frequency_score", "track_popularity_proxy",
]
GROUP6_FEATURES = [
    "release_year_distance_to_history_median", "decade_match_flag",
]

ALL_CANDIDATE_FEATURES = (
    GROUP1_FEATURES + GROUP2_FEATURES + GROUP3_FEATURES
    + GROUP4_FEATURES + GROUP5_FEATURES + GROUP6_FEATURES
)


# ---------------------------------------------------------------------------
def build_cfg0209_a_lists(payload: dict[str, Any]) -> list[list[str]]:
    """Recompute A_max_recent_5 @ depth 200 from qwen3 vectors. No API."""
    print("loading qwen3 index for A'@200 recomputation", flush=True)
    vecs, id_to_idx, track_ids = load_qwen_index()
    a_lists = a_prime_deep_batch(
        vecs, id_to_idx, track_ids, payload["cases"], n=5, k=200
    )
    print(f"  computed A'@200 for {len(a_lists)} sessions", flush=True)
    return a_lists


def build_cfg0209_pool(
    payload: dict[str, Any], a_lists: list[list[str]]
) -> tuple[list[list[str]], list[dict[str, dict[str, int]]]]:
    """RRF-fuse cfg0209 sources, return pools and per-pool source-rank maps."""
    cases = payload["cases"]
    src_b_all = payload["src_b"]
    src_c_all = payload["src_c"]
    src_d_all = payload["src_d"]
    depths = CFG0209["depths"]
    weights = CFG0209["weights"]
    k = CFG0209["k"]

    pools: list[list[str]] = []
    rank_maps: list[dict[str, dict[str, int]]] = []
    for i in range(len(cases)):
        # Truncate at depth
        a_list = a_lists[i][: depths["A"]] if depths["A"] > 0 else []
        b_list = src_b_all[i][: depths["B"]]
        c_list = src_c_all[i][: depths["C"]]
        d_list = src_d_all[i][: depths["D"]]

        seqs = [
            (a_list, weights["A"]),
            (b_list, weights["B"]),
            (c_list, weights["C"]),
            (d_list, weights["D"]),
        ]
        pool = rrf_fuse(seqs, k=k, pool_size=POOL_K)
        pools.append(pool)

        # Source rank lookup (1-indexed; absent → not in source)
        rank_a = {tid: r for r, tid in enumerate(a_list, start=1)}
        rank_b = {tid: r for r, tid in enumerate(b_list, start=1)}
        rank_c = {tid: r for r, tid in enumerate(c_list, start=1)}
        rank_d = {tid: r for r, tid in enumerate(d_list, start=1)}
        rank_maps.append({"A": rank_a, "B": rank_b, "C": rank_c, "D": rank_d})
    return pools, rank_maps


# ---------------------------------------------------------------------------
def parse_year(release_date: str | None) -> int | None:
    if not release_date:
        return None
    s = str(release_date).strip()
    if len(s) < 4 or not s[:4].isdigit():
        return None
    y = int(s[:4])
    if y < 1900 or y > 2030:
        return None
    return y


def build_extended_metadata(payload: dict[str, Any], pools: list[list[str]]) -> dict[str, Any]:
    """Augment payload with album_toks, release_year, popularity, artist_id, etc.

    Only fetch metadata for tids appearing in at least one pool or session history.
    """
    print("collecting tids needing extended metadata", flush=True)
    needed: set[str] = set()
    for pool in pools:
        needed.update(pool)
    for c in payload["cases"]:
        needed.update(c["music_turns"])
    print(f"  total unique tids needing metadata: {len(needed)}", flush=True)

    print("loading extended metadata via fast_load", flush=True)
    # We need release_date, popularity, artist_id, album_id beyond the basic 4 fields.
    # Use a custom loader that pulls the full row.
    ext_meta = _fast_load_extended(needed)
    print(f"  loaded extended metadata for {len(ext_meta)} tids", flush=True)

    # Build per-tid maps
    track_album_toks: dict[str, set[str]] = {}
    track_year: dict[str, int | None] = {}
    track_pop: dict[str, float | None] = {}
    track_artist_id: dict[str, str] = {}
    track_album_id: dict[str, str] = {}

    for tid, meta in ext_meta.items():
        album = meta.get("album_name")
        if isinstance(album, list):
            album_str = " ".join(str(x) for x in album)
        else:
            album_str = str(album) if album else ""
        track_album_toks[tid] = tokens(album_str)
        track_year[tid] = parse_year(meta.get("release_date"))
        pop = meta.get("popularity")
        try:
            track_pop[tid] = float(pop) if pop is not None and not (isinstance(pop, float) and np.isnan(pop)) else None
        except (TypeError, ValueError):
            track_pop[tid] = None
        # artist_id is a list of UUID strings
        a_id = meta.get("artist_id")
        if isinstance(a_id, list) and a_id:
            track_artist_id[tid] = str(a_id[0])
        elif a_id:
            track_artist_id[tid] = str(a_id)
        else:
            track_artist_id[tid] = ""
        b_id = meta.get("album_id")
        if isinstance(b_id, list) and b_id:
            track_album_id[tid] = str(b_id[0])
        elif b_id:
            track_album_id[tid] = str(b_id)
        else:
            track_album_id[tid] = ""

    # Fill defaults for tids in the payload's existing maps that we didn't touch.
    for tid in payload["track_artist"]:
        track_album_toks.setdefault(tid, set())
        track_year.setdefault(tid, None)
        track_pop.setdefault(tid, None)
        track_artist_id.setdefault(tid, "")
        track_album_id.setdefault(tid, "")

    # Catalog-level frequencies (computed over the FULL catalog, not just pools)
    print("computing catalog-level artist/tag frequencies", flush=True)
    artist_freq, tag_freq = _catalog_frequencies()

    return {
        "track_album_toks": track_album_toks,
        "track_year": track_year,
        "track_pop": track_pop,
        "track_artist_id": track_artist_id,
        "track_album_id": track_album_id,
        "catalog_artist_freq": artist_freq,
        "catalog_tag_freq": tag_freq,
    }


def _fast_load_extended(target_tids: set[str]) -> dict[str, dict]:
    """Load full metadata rows including release_date, popularity, artist_id."""
    import pyarrow as pa
    from datasets import Dataset

    from offline_retrieval_sweep import latest_arrow

    path = latest_arrow(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"
    )
    ds = Dataset.from_file(path)
    tbl: pa.Table = ds.data.table
    tids_col = tbl.column("track_id").to_pylist()
    if not target_tids:
        return {}
    target = {str(t) for t in target_tids}
    positions = [i for i, t in enumerate(tids_col) if t in target]
    if not positions:
        return {}
    sub = tbl.take(positions)
    cols_to_take = ["track_id", "track_name", "artist_name", "album_name",
                    "tag_list", "release_date", "popularity", "artist_id", "album_id"]
    extracted = {col: sub.column(col).to_pylist() for col in cols_to_take if col in tbl.column_names}
    out: dict[str, dict] = {}
    n = len(extracted["track_id"])
    for i in range(n):
        tid = str(extracted["track_id"][i])
        out[tid] = {col: extracted[col][i] for col in extracted if col != "track_id"}
    return out


def _catalog_frequencies() -> tuple[dict[str, int], dict[str, int]]:
    """Per-artist track count and per-tag track count across the full catalog."""
    import pyarrow as pa
    from datasets import Dataset

    from offline_retrieval_sweep import latest_arrow

    path = latest_arrow(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"
    )
    ds = Dataset.from_file(path)
    tbl: pa.Table = ds.data.table
    artists = tbl.column("artist_name").to_pylist()
    tags = tbl.column("tag_list").to_pylist()
    artist_freq: dict[str, int] = {}
    tag_freq: dict[str, int] = {}
    # Important: payload["track_artist"] stores artist as the lowercased str()
    # of the raw arrow field (which is a list, e.g. "['helios']"). To make
    # cand_artist match catalog_artist_freq we use the same normalisation here.
    for a in artists:
        key = str(a).lower().strip() if a is not None else ""
        if key:
            artist_freq[key] = artist_freq.get(key, 0) + 1
    for tag_list in tags:
        if not isinstance(tag_list, list):
            continue
        for t in tag_list:
            k = str(t).lower().strip()
            if k:
                tag_freq[k] = tag_freq.get(k, 0) + 1
    return artist_freq, tag_freq


# ---------------------------------------------------------------------------
def build_full_feature_matrix(
    payload: dict[str, Any],
    pools: list[list[str]],
    rank_maps: list[dict[str, dict[str, int]]],
    ext: dict[str, Any],
    qwen_vecs: np.ndarray,
    qwen_id_to_idx: dict[str, int],
    selected_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, gt_idx, sizes, feature_names) — all features in one matrix.

    Feature ordering: BASE 8 first (from r3_confirm_400_deterministic), then candidates.
    """
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]
    track_album_toks = ext["track_album_toks"]
    track_year = ext["track_year"]
    track_pop = ext["track_pop"]
    catalog_artist_freq = ext["catalog_artist_freq"]
    catalog_tag_freq = ext["catalog_tag_freq"]

    feature_names = list(BASE_FEATURE_NAMES) + list(selected_features)
    f_count = len(feature_names)
    n = len(cases)
    X = np.zeros((n, POOL_K, f_count), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    # Index of each candidate feature
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    for i, c in enumerate(cases):
        ranked = pools[i]
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

        # Group 2 prep: dense similarity to last 5 played
        recent5 = played[-5:] if played else []
        recent5_idx = [qwen_id_to_idx.get(t) for t in recent5]
        recent5_idx = [j for j in recent5_idx if j is not None]
        # Pad up to 5 valid; if none, all sim features fall to 0
        recent5_vecs = qwen_vecs[recent5_idx] if recent5_idx else None
        last_idx_q = qwen_id_to_idx.get(played[-1]) if played else None
        last_vec_q = qwen_vecs[last_idx_q] if last_idx_q is not None else None
        # centroid over the recent5
        if recent5_vecs is not None and len(recent5_vecs) > 0:
            centroid_vec = recent5_vecs.mean(axis=0)
            cn = np.linalg.norm(centroid_vec)
            if cn > 0:
                centroid_vec = centroid_vec / cn
        else:
            centroid_vec = None

        # Group 3 prep: history artist counts
        history_artist_count: dict[str, int] = {}
        for tid in played:
            a = track_artist.get(tid, "")
            if a:
                history_artist_count[a] = history_artist_count.get(a, 0) + 1
        recent5_artist_count: dict[str, int] = {}
        for tid in played[-5:]:
            a = track_artist.get(tid, "")
            if a:
                recent5_artist_count[a] = recent5_artist_count.get(a, 0) + 1
        recent5_size = max(1, len(played[-5:]))

        # Group 6 prep: median played decade
        played_years = [track_year.get(tid) for tid in played]
        played_years = [y for y in played_years if y is not None]
        median_year = int(np.median(played_years)) if played_years else None
        median_decade = (median_year // 10) * 10 if median_year is not None else None

        # Group 4 prep: query tokens (now_tokens already prepared)

        for rank0, tid in enumerate(ranked):
            cand_artist = track_artist.get(tid, "")
            cand_tags = track_tags.get(tid, set())
            row = X[i, rank0]
            # ---------- Base 8 features ----------
            row[feat_idx["orig_rank"]] = 1.0 / (rank0 + 1)
            row[feat_idx["same_artist_last_played"]] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
            if cand_tags or last_tags:
                row[feat_idx["tag_overlap_last_played"]] = (
                    len(cand_tags & last_tags) / len(cand_tags | last_tags)
                )
            row[feat_idx["artist_token_overlap_query"]] = float(
                len(track_artist_toks.get(tid, set()) & now_tokens)
            )
            row[feat_idx["title_token_overlap_query"]] = float(
                len(track_title_toks.get(tid, set()) & now_tokens)
            )
            row[feat_idx["all_history_user_token_overlap"]] = float(
                len(track_meta_toks.get(tid, set()) & all_user_tokens)
            )
            row[feat_idx["already_played"]] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for w_p, p_artist, p_tags in prior:
                artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                tag_match = (
                    len(cand_tags & p_tags) / len(cand_tags | p_tags)
                    if (cand_tags or p_tags) else 0.0
                )
                rec += w_p * (artist_match + tag_match)
            row[feat_idx["recency_weighted_metadata"]] = rec

            # ---------- Group 1: source membership/rank ----------
            ra = rank_maps[i]["A"].get(tid)
            rb = rank_maps[i]["B"].get(tid)
            rc = rank_maps[i]["C"].get(tid)
            rd = rank_maps[i]["D"].get(tid)
            in_A = 1.0 if ra is not None else 0.0
            in_B = 1.0 if rb is not None else 0.0
            in_C = 1.0 if rc is not None else 0.0
            in_D = 1.0 if rd is not None else 0.0
            if "in_Aprime" in feat_idx:
                row[feat_idx["in_Aprime"]] = in_A
            if "in_B" in feat_idx:
                row[feat_idx["in_B"]] = in_B
            if "in_C" in feat_idx:
                row[feat_idx["in_C"]] = in_C
            if "in_D" in feat_idx:
                row[feat_idx["in_D"]] = in_D
            if "rr_Aprime" in feat_idx:
                row[feat_idx["rr_Aprime"]] = (1.0 / (ra + 1)) if ra is not None else 0.0
            if "rr_B" in feat_idx:
                row[feat_idx["rr_B"]] = (1.0 / (rb + 1)) if rb is not None else 0.0
            if "rr_C" in feat_idx:
                row[feat_idx["rr_C"]] = (1.0 / (rc + 1)) if rc is not None else 0.0
            if "rr_D" in feat_idx:
                row[feat_idx["rr_D"]] = (1.0 / (rd + 1)) if rd is not None else 0.0
            if "source_count" in feat_idx:
                row[feat_idx["source_count"]] = in_A + in_B + in_C + in_D

            # ---------- Group 2: dense similarity ----------
            cand_q_idx = qwen_id_to_idx.get(tid)
            max_sim = 0.0
            mean_sim = 0.0
            sim_last = 0.0
            sim_centroid = 0.0
            if cand_q_idx is not None and recent5_vecs is not None and len(recent5_vecs) > 0:
                cand_vec = qwen_vecs[cand_q_idx]
                sims = recent5_vecs @ cand_vec
                max_sim = float(sims.max())
                mean_sim = float(sims.mean())
                if last_vec_q is not None:
                    sim_last = float(cand_vec @ last_vec_q)
                if centroid_vec is not None:
                    sim_centroid = float(cand_vec @ centroid_vec)
            if "max_sim_recent5" in feat_idx:
                row[feat_idx["max_sim_recent5"]] = max_sim
            if "mean_sim_recent5" in feat_idx:
                row[feat_idx["mean_sim_recent5"]] = mean_sim
            if "sim_last" in feat_idx:
                row[feat_idx["sim_last"]] = sim_last
            if "sim_centroid_recent5" in feat_idx:
                row[feat_idx["sim_centroid_recent5"]] = sim_centroid
            if "sim_margin" in feat_idx:
                row[feat_idx["sim_margin"]] = max_sim - sim_last

            # ---------- Group 3: history diversity / novelty ----------
            if "already_artist_played_count" in feat_idx:
                row[feat_idx["already_artist_played_count"]] = float(
                    history_artist_count.get(cand_artist, 0)
                ) if cand_artist else 0.0
            if "already_tag_overlap_max" in feat_idx:
                best_jacc = 0.0
                for tid2 in played:
                    p_tags = track_tags.get(tid2, set())
                    if cand_tags or p_tags:
                        jacc = (
                            len(cand_tags & p_tags) / len(cand_tags | p_tags)
                            if (cand_tags or p_tags) else 0.0
                        )
                        if jacc > best_jacc:
                            best_jacc = jacc
                row[feat_idx["already_tag_overlap_max"]] = best_jacc
            if "candidate_artist_recent_frequency" in feat_idx:
                row[feat_idx["candidate_artist_recent_frequency"]] = (
                    recent5_artist_count.get(cand_artist, 0) / recent5_size
                    if cand_artist else 0.0
                )
            if "novelty_vs_recent_tracks" in feat_idx:
                row[feat_idx["novelty_vs_recent_tracks"]] = 1.0 - max_sim

            # ---------- Group 4: query/meta interaction ----------
            if "tag_token_overlap_query" in feat_idx:
                tag_set_lower = {t.lower() for t in cand_tags}
                # Jaccard of candidate tag tokens vs query tokens
                if tag_set_lower or now_tokens:
                    inter = len(tag_set_lower & now_tokens)
                    union = len(tag_set_lower | now_tokens)
                    row[feat_idx["tag_token_overlap_query"]] = inter / union if union else 0.0
                else:
                    row[feat_idx["tag_token_overlap_query"]] = 0.0
            if "album_token_overlap_query" in feat_idx:
                alb_toks = track_album_toks.get(tid, set())
                if alb_toks or now_tokens:
                    inter = len(alb_toks & now_tokens)
                    union = len(alb_toks | now_tokens)
                    row[feat_idx["album_token_overlap_query"]] = inter / union if union else 0.0
            if "artist_or_title_exact_from_query" in feat_idx:
                # Treat "exact from query" as: the candidate's artist tokens or
                # title tokens are a subset of the query tokens AND non-empty.
                # This is more meaningful than substring match against the
                # list-repr "['name']" stored in track_artist.
                hit = 0.0
                a_toks = track_artist_toks.get(tid, set())
                t_toks = track_title_toks.get(tid, set())
                # Strip stopwords already done by tokens(). Require min 1-token coverage.
                # Multi-token artist/title coverage in query → stronger signal.
                if a_toks and a_toks.issubset(now_tokens):
                    hit = 1.0
                elif t_toks and t_toks.issubset(now_tokens):
                    hit = 1.0
                row[feat_idx["artist_or_title_exact_from_query"]] = hit

            # ---------- Group 5: popularity ----------
            if "artist_frequency_in_catalog" in feat_idx:
                row[feat_idx["artist_frequency_in_catalog"]] = float(
                    catalog_artist_freq.get(cand_artist, 0)
                )
            if "tag_frequency_score" in feat_idx:
                tag_score = sum(catalog_tag_freq.get(t.lower(), 0) for t in cand_tags)
                row[feat_idx["tag_frequency_score"]] = float(tag_score)
            if "track_popularity_proxy" in feat_idx:
                p = track_pop.get(tid)
                row[feat_idx["track_popularity_proxy"]] = float(p) if p is not None else 0.0

            # ---------- Group 6: temporal ----------
            cand_year = track_year.get(tid)
            if "release_year_distance_to_history_median" in feat_idx:
                if cand_year is not None and median_year is not None:
                    row[feat_idx["release_year_distance_to_history_median"]] = (
                        float(abs(cand_year - median_year))
                    )
                else:
                    row[feat_idx["release_year_distance_to_history_median"]] = 0.0  # impute zero
            if "decade_match_flag" in feat_idx:
                if cand_year is not None and median_decade is not None:
                    cand_dec = (cand_year // 10) * 10
                    row[feat_idx["decade_match_flag"]] = 1.0 if cand_dec == median_decade else 0.0
                else:
                    row[feat_idx["decade_match_flag"]] = 0.0

        if c["gt"] in ranked:
            gt_idx[i] = ranked.index(c["gt"])

    return X, gt_idx, sizes, feature_names


# ---------------------------------------------------------------------------
def vec_ndcg_pre(
    Xs: np.ndarray, gt_idx_sub: np.ndarray, sizes_sub: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Vectorised nDCG given pre-sliced (rows, cols) tensor.

    Xs: (n_eval, pool, F_sel) — already sliced to the eval rows + feature cols.
    gt_idx_sub: (n_eval,) — gt indices for those rows.
    sizes_sub: (n_eval,) — pool sizes for those rows.
    """
    pool_axis = np.arange(Xs.shape[1])[None, :]
    valid_pool = pool_axis < sizes_sub[:, None]
    scores = Xs @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_idx_sub >= 0
    safe_gt = np.where(has_gt, gt_idx_sub, 0)
    gt_scores = scores[np.arange(Xs.shape[0]), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_weights_pre(
    X_train: np.ndarray, gt_train: np.ndarray, sizes_train: np.ndarray,
    init: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Powell using pre-sliced training tensor."""
    def objective(w: np.ndarray) -> float:
        return -vec_ndcg_pre(X_train, gt_train, sizes_train, w)

    res = minimize(
        objective, init, method="Powell",
        options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500},
    )
    return res.x, -float(res.fun)


def cv5_evaluate_subset(
    X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray,
    sessions: list[str], feat_select_idx: list[int], feat_names: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """5-fold CV across multiple seeds; return aggregate stats and learned weights.

    Speedup: pre-slice the feature columns once, so each Powell call only
    indexes rows (no triple fancy index inside the inner loop).
    """
    feat_select = np.asarray(feat_select_idx, dtype=np.int64)
    sel_names = [feat_names[i] for i in feat_select_idx]
    init = np.asarray(
        [BASE_INIT_WEIGHTS.get(fn, 0.0) for fn in sel_names],
        dtype=np.float64,
    )
    # Pre-slice features once: shape (n_sessions, K, F_sel) — contiguous memory.
    X_sel = np.ascontiguousarray(X[:, :, feat_select])

    per_seed: list[dict[str, Any]] = []
    fold_weights: list[np.ndarray] = []
    n_sessions = len(sessions)
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores: list[float] = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray(
                [i for i in range(n_sessions) if i not in held],
                dtype=np.int64,
            )
            X_train = X_sel[train]
            gt_train = gt_idx[train]
            sizes_train = sizes[train]
            wts, _ = fit_weights_pre(X_train, gt_train, sizes_train, init)
            X_held = X_sel[fold]
            fold_scores.append(vec_ndcg_pre(X_held, gt_idx[fold], sizes[fold], wts))
            fold_weights.append(wts.copy())
        per_seed.append({
            "seed": seed,
            "cv5_mean": float(np.mean(fold_scores)),
            "cv5_std": float(np.std(fold_scores, ddof=1)),
            "cv5_per_fold": [float(x) for x in fold_scores],
        })
    means = [r["cv5_mean"] for r in per_seed]
    # Average weights across folds for reporting
    avg_weights = np.stack(fold_weights, axis=0).mean(axis=0)
    return {
        "n_features": len(feat_select_idx),
        "feature_names": sel_names,
        "per_seed": per_seed,
        "aggregate": {
            "mean_cv5": float(np.mean(means)),
            "std_of_cv5_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "min_cv5": float(np.min(means)),
            "max_cv5": float(np.max(means)),
        },
        "avg_weights": dict(zip(sel_names, [float(x) for x in avg_weights], strict=True)),
    }


# ---------------------------------------------------------------------------
def stage2_diagnostics(
    X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray, feat_names: list[str],
) -> dict[str, Any]:
    """Univariate diagnostics: top-decile lift, mean GT vs non-GT, missing rate."""
    n_sess = X.shape[0]
    pool_K = X.shape[1]
    # Build (n_sess*pool, F) flat with mask of valid pool entries
    pool_axis = np.arange(pool_K)[None, :]
    valid = (pool_axis < sizes[:, None]).flatten()  # (n_sess*pool,)
    Xf = X.reshape(n_sess * pool_K, -1)  # (n_sess*pool, F)
    is_gt = np.zeros(n_sess * pool_K, dtype=bool)
    has_gt = gt_idx >= 0
    for i in np.where(has_gt)[0]:
        is_gt[i * pool_K + gt_idx[i]] = True

    Xf_valid = Xf[valid]
    is_gt_valid = is_gt[valid]
    base_rate = is_gt_valid.mean() if len(is_gt_valid) > 0 else 0.0

    diagnostics = []
    for j, fname in enumerate(feat_names):
        col = Xf_valid[:, j]
        gt_mean = float(col[is_gt_valid].mean()) if is_gt_valid.any() else float("nan")
        non_gt_mean = float(col[~is_gt_valid].mean()) if (~is_gt_valid).any() else float("nan")
        # Missing rate: count of zeros in feature (proxy)
        missing_rate = float((col == 0.0).mean())
        # Top-decile lift: rate of GT among top 10% of candidates by this feature value
        n_valid = len(col)
        if n_valid == 0:
            lift = float("nan")
            verdict = "no_data"
        else:
            top_k = max(1, int(0.1 * n_valid))
            top_idx_global = np.argpartition(-col, top_k - 1)[:top_k]
            top_gt_rate = is_gt_valid[top_idx_global].mean()
            lift = float(top_gt_rate / base_rate) if base_rate > 0 else float("nan")
            if lift != lift or lift is None:
                verdict = "no_signal"
            elif lift > 1.5:
                verdict = "strong"
            elif lift > 1.2:
                verdict = "moderate"
            else:
                verdict = "weak"
            if missing_rate > 0.95:
                verdict = "exclude_high_missing"
        # Inverse direction (low values matter): also compute bottom-decile lift
        if n_valid > 0:
            bot_k = max(1, int(0.1 * n_valid))
            bot_idx_global = np.argpartition(col, bot_k - 1)[:bot_k]
            bot_gt_rate = is_gt_valid[bot_idx_global].mean()
            inv_lift = float(bot_gt_rate / base_rate) if base_rate > 0 else float("nan")
        else:
            inv_lift = float("nan")
        diagnostics.append({
            "feature": fname,
            "mean_GT": gt_mean,
            "mean_nonGT": non_gt_mean,
            "missing_rate": missing_rate,
            "top_decile_lift": lift,
            "bottom_decile_lift": inv_lift,
            "verdict": verdict,
        })
    return {
        "base_rate": float(base_rate),
        "n_pool_entries": int(valid.sum()),
        "features": diagnostics,
    }


# ---------------------------------------------------------------------------
def main() -> None:  # noqa: C901
    t0 = time.time()
    print("=" * 60)
    print("EXPERIMENT B — feature expansion on cfg0209 fusion baseline")
    print("=" * 60)

    # ---- Stage 1: build pool + features ----
    if OUT_FEATURES.exists():
        print(f"\n[Stage 1] loading cached features {OUT_FEATURES}", flush=True)
        with open(OUT_FEATURES, "rb") as f:
            stage1_cache = pickle.load(f)  # noqa: S301
        X = stage1_cache["X"]
        gt_idx = stage1_cache["gt_idx"]
        sizes = stage1_cache["sizes"]
        feat_names = stage1_cache["feat_names"]
        sessions = stage1_cache["sessions"]
        skipped = stage1_cache.get("skipped_features", [])
    else:
        print("\n[Stage 1] loading payload", flush=True)
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        n = len(payload["cases"])
        print(f"  loaded {n} cases", flush=True)

        # Build A'@200
        a_lists = build_cfg0209_a_lists(payload)
        # Augment payload metadata for any new tids in A'@200
        new_count = fast_augment_metadata_for_a(payload, a_lists)
        print(f"  augmented metadata for {new_count} new tids from A'@200", flush=True)

        # Build cfg0209 pool
        print("\n[Stage 1] building cfg0209 pools", flush=True)
        pools, rank_maps = build_cfg0209_pool(payload, a_lists)
        pool_hits = sum(1 for i, c in enumerate(payload["cases"]) if c["gt"] in pools[i])
        pool_hit_rate = pool_hits / n
        print(f"  cfg0209 pool_hit@50 = {pool_hit_rate:.4f} (expected ~0.3775)", flush=True)

        # Augment payload metadata for any tids in pool that we didn't see yet
        # (this ensures track_meta_toks etc. cover the pool)
        all_tids_in_pool = set()
        for pool in pools:
            all_tids_in_pool.update(pool)
        missing_tids = all_tids_in_pool - set(payload["track_artist"].keys())
        if missing_tids:
            # Use the helper to augment
            print(f"  augmenting {len(missing_tids)} pool tids missing metadata", flush=True)
            missing_lists = [list(missing_tids)]
            fast_augment_metadata_for_a(payload, missing_lists)

        # Extended metadata for new feature groups
        ext = build_extended_metadata(payload, pools)

        # Determine which candidate features survive based on field availability
        # Compute availability over POOL TIDS ONLY (not the full augmented payload,
        # which has many tids defaulted to None for fields we never loaded).
        pool_tids = set()
        for pool in pools:
            pool_tids.update(pool)
        skipped: list[dict[str, str]] = []
        rel_avail_pool = sum(1 for tid in pool_tids if ext["track_year"].get(tid) is not None)
        rel_pct = rel_avail_pool / max(1, len(pool_tids))
        print(f"  release_year availability (pool): {rel_avail_pool}/{len(pool_tids)} = {rel_pct:.2%}", flush=True)
        pop_avail_pool = sum(1 for tid in pool_tids if ext["track_pop"].get(tid) is not None)
        pop_pct = pop_avail_pool / max(1, len(pool_tids))
        print(f"  popularity availability (pool): {pop_avail_pool}/{len(pool_tids)} = {pop_pct:.2%}", flush=True)

        candidate_features = list(ALL_CANDIDATE_FEATURES)
        # If popularity essentially all zero/missing, skip track_popularity_proxy.
        if pop_pct < 0.5:
            for f in ("track_popularity_proxy",):
                if f in candidate_features:
                    candidate_features.remove(f)
                    skipped.append({"feature": f, "reason": f"popularity availability {pop_pct:.2%} < 50%"})
        # If release_year missing > 50%, skip year features
        if rel_pct < 0.5:
            for f in GROUP6_FEATURES:
                if f in candidate_features:
                    candidate_features.remove(f)
                    skipped.append({"feature": f, "reason": f"release_year availability {rel_pct:.2%} < 50%"})

        # Load qwen vectors (mmap; only needed sessions touch their rows)
        print("\n[Stage 1] loading qwen3 vectors for dense sim features", flush=True)
        qwen_vecs, qwen_id_to_idx, _ = load_qwen_index()

        # Build the FULL feature matrix (all available features at once)
        print(f"\n[Stage 1] building feature matrix ({len(BASE_FEATURE_NAMES)} base + {len(candidate_features)} candidates)", flush=True)
        t_b = time.time()
        X, gt_idx, sizes, feat_names = build_full_feature_matrix(
            payload, pools, rank_maps, ext,
            qwen_vecs, qwen_id_to_idx, candidate_features,
        )
        print(f"  feature matrix built in {time.time() - t_b:.1f}s; shape = {X.shape}", flush=True)
        sessions = [c["session_id"] for c in payload["cases"]]

        # Save
        OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FEATURES, "wb") as f:
            pickle.dump({
                "X": X, "gt_idx": gt_idx, "sizes": sizes,
                "feat_names": feat_names, "sessions": sessions,
                "skipped_features": skipped,
                "pool_hit_rate": pool_hit_rate,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  cached → {OUT_FEATURES}", flush=True)

    print(f"\n[Stage 1] feature matrix: {X.shape}; sessions: {len(sessions)}", flush=True)
    print(f"  skipped features: {[s['feature'] for s in skipped]}", flush=True)

    # ---- Stage 2: univariate diagnostics ----
    print("\n[Stage 2] univariate diagnostics", flush=True)
    diag = stage2_diagnostics(X, gt_idx, sizes, feat_names)
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIAG, "w", encoding="utf-8") as f:
        json.dump({
            "n_pool_entries": diag["n_pool_entries"],
            "base_rate": diag["base_rate"],
            "skipped_features": skipped,
            "features": diag["features"],
        }, f, indent=2)
    print(f"  base GT rate: {diag['base_rate']:.4f}", flush=True)
    for d in diag["features"]:
        if d["feature"] in BASE_FEATURE_NAMES:
            continue
        print(f"  {d['feature']:<42}  lift={d['top_decile_lift']:.2f}  inv={d['bottom_decile_lift']:.2f}  miss={d['missing_rate']:.2f}  → {d['verdict']}", flush=True)

    # Pick which Group 3/5/6 features survive for M4
    surviving = []
    for d in diag["features"]:
        f = d["feature"]
        if f in BASE_FEATURE_NAMES:
            continue
        if f not in (GROUP3_FEATURES + GROUP5_FEATURES + GROUP6_FEATURES):
            continue
        # Keep features that have some signal (lift>1.05 or inv lift>1.2 since some features are inversely-related)
        keep = False
        if d["top_decile_lift"] is not None and d["top_decile_lift"] == d["top_decile_lift"]:
            if d["top_decile_lift"] > 1.05 or d["bottom_decile_lift"] > 1.2:
                keep = True
        if d["missing_rate"] > 0.95:
            keep = False
        if keep:
            surviving.append(f)
    print(f"\n[Stage 2] surviving Group 3/5/6 features for M4: {surviving}", flush=True)

    # ---- Stage 3: Powell CV5 with feature group models ----
    print("\n[Stage 3] Powell CV5 — group models", flush=True)
    seeds = [0, 1, 2]

    base_idx = [feat_names.index(f) for f in BASE_FEATURE_NAMES]
    g1_idx = [feat_names.index(f) for f in GROUP1_FEATURES if f in feat_names]
    g2_idx = [feat_names.index(f) for f in GROUP2_FEATURES if f in feat_names]
    g3_idx = [feat_names.index(f) for f in GROUP3_FEATURES if f in feat_names]
    g4_idx = [feat_names.index(f) for f in GROUP4_FEATURES if f in feat_names]
    g5_idx = [feat_names.index(f) for f in GROUP5_FEATURES if f in feat_names]
    g6_idx = [feat_names.index(f) for f in GROUP6_FEATURES if f in feat_names]
    surv_idx = [feat_names.index(f) for f in surviving if f in feat_names]

    models_def = {
        "M0_base": base_idx,
        "M1_base_plus_source": base_idx + g1_idx,
        "M2_base_plus_dense": base_idx + g2_idx,
        "M3_base_plus_query": base_idx + g4_idx,
        "M4_base_plus_rest_surv": base_idx + surv_idx,
        "M5_base_plus_all": sorted(set(base_idx + g1_idx + g2_idx + g3_idx + g4_idx + g5_idx + g6_idx)),
    }

    model_results: dict[str, Any] = {}
    for mname, sel in models_def.items():
        print(f"\n  [{mname}] features={len(sel)}", flush=True)
        t_m = time.time()
        res = cv5_evaluate_subset(X, gt_idx, sizes, sessions, sel, feat_names, seeds)
        agg = res["aggregate"]
        delta = agg["mean_cv5"] - 0.1602
        print(f"    mean_cv5 = {agg['mean_cv5']:.4f}  std-of-means = {agg['std_of_cv5_means']:.4f}  Δ vs M0 = {delta:+.4f}  ({time.time() - t_m:.1f}s)", flush=True)
        model_results[mname] = res

    OUT_MODELS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODELS, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds,
            "models": model_results,
            "models_def_indices": models_def,
            "feat_names": feat_names,
        }, f, indent=2, default=float)

    # ---- Stage 4: leave-one-group-out ablation on M5 ----
    print("\n[Stage 4] leave-one-group-out ablation on M5", flush=True)
    m5_idx_set = set(models_def["M5_base_plus_all"])
    group_idx_map = {
        "G1_source": set(g1_idx),
        "G2_dense": set(g2_idx),
        "G3_history": set(g3_idx),
        "G4_query": set(g4_idx),
        "G5_popularity": set(g5_idx),
        "G6_temporal": set(g6_idx),
    }
    ablation_results: dict[str, Any] = {}
    m5_mean = model_results["M5_base_plus_all"]["aggregate"]["mean_cv5"]
    for gname, gidx in group_idx_map.items():
        if not gidx:
            ablation_results[gname] = {"skipped": True, "reason": "group empty (no features available)"}
            continue
        sel = sorted(m5_idx_set - gidx)
        if not sel:
            ablation_results[gname] = {"skipped": True, "reason": "no features remain"}
            continue
        print(f"  drop {gname} ({len(gidx)} feats) → {len(sel)} feats", flush=True)
        res = cv5_evaluate_subset(X, gt_idx, sizes, sessions, sel, feat_names, seeds)
        d = res["aggregate"]["mean_cv5"] - m5_mean
        print(f"    mean_cv5 = {res['aggregate']['mean_cv5']:.4f}  Δ vs M5 = {d:+.4f}", flush=True)
        ablation_results[gname] = {**res, "delta_vs_M5": float(d)}

    with open(OUT_ABLATION, "w", encoding="utf-8") as f:
        json.dump({
            "seeds": seeds,
            "M5_mean_cv5": m5_mean,
            "ablation": ablation_results,
        }, f, indent=2, default=float)

    # ---- Stage 5: robustness on M5 (5 extra seeds) ----
    print("\n[Stage 5] robustness — M5 with extra seeds 5..9", flush=True)
    extra_seeds = [5, 6, 7, 8, 9]
    res_extra = cv5_evaluate_subset(
        X, gt_idx, sizes, sessions, models_def["M5_base_plus_all"], feat_names, extra_seeds,
    )
    # Merge with seeds 0..2 already in M5
    m5_seeds = model_results["M5_base_plus_all"]["per_seed"]
    all_seeds = m5_seeds + res_extra["per_seed"]
    means = [r["cv5_mean"] for r in all_seeds]
    robustness = {
        "seeds": [0, 1, 2] + extra_seeds,
        "per_seed_means": means,
        "mean_across_8": float(np.mean(means)),
        "std_across_8": float(np.std(means, ddof=1)),
        "min_across_8": float(np.min(means)),
        "max_across_8": float(np.max(means)),
    }
    print(f"  M5 across 8 seeds: mean={robustness['mean_across_8']:.4f}  std={robustness['std_across_8']:.4f}", flush=True)

    # ---- Verdict ----
    best_model_name = max(model_results, key=lambda k: model_results[k]["aggregate"]["mean_cv5"])
    best_mean = model_results[best_model_name]["aggregate"]["mean_cv5"]
    best_std = model_results[best_model_name]["aggregate"]["std_of_cv5_means"]
    if best_mean >= 0.167 and best_std <= 0.003:
        verdict = "SUBMISSION-READY"
    elif best_mean >= 0.163:
        verdict = "MARGINAL"
    else:
        verdict = "REJECT"
    print(f"\n=== Best model: {best_model_name}  mean_cv5={best_mean:.4f}  std={best_std:.4f}  → {verdict} ===", flush=True)

    final = {
        "elapsed_sec": time.time() - t0,
        "no_llm_calls": True,
        "best_model": best_model_name,
        "best_mean_cv5": best_mean,
        "best_std_of_means": best_std,
        "verdict": verdict,
        "robustness_M5": robustness,
        "skipped_features": skipped,
    }
    with open(OUT_FINAL, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=float)
    print(f"\nWrote {OUT_FINAL}", flush=True)


if __name__ == "__main__":
    main()
