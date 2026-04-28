#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""F1 experiment: CF-BPR item-item retrieval as new candidate source.

Uses pre-computed cf-bpr (128-dim) embeddings from
talkpl-ai/TalkPlayData-Challenge-Track-Embeddings to retrieve collaborative
nearest neighbors of session-played tracks.

Phase 1: source-alone diagnostics (recall, unique GT hits vs cfg0209)
Phase 2: fusion with cfg0209 weighted-RRF + Powell CV5

No API calls. Reuses the same 400-session devset slice as r3_confirm_400.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (
    build_or_load_payload,
    build_pool_features,
    cv_folds,
    fit_weights,
    vec_ndcg,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS

# --- CF-BPR index --------------------------------------------------------- #

CFBPR_CACHE_DIR = REPO_ROOT / "cache" / "track_sim" / "cf-bpr"


def build_cfbpr_index() -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Load cf-bpr embeddings, filter empty vectors, L2-normalize, return
    (track_ids, vectors, id_to_idx)."""
    ids_path = CFBPR_CACHE_DIR / "track_ids.json"
    vecs_path = CFBPR_CACHE_DIR / "vectors.npy"

    if ids_path.exists() and vecs_path.exists():
        print(f"loading cached cf-bpr index from {CFBPR_CACHE_DIR}", flush=True)
        with open(ids_path) as f:
            track_ids = json.load(f)
        vectors = np.load(vecs_path)
    else:
        print("building cf-bpr index from HF cache...", flush=True)
        from datasets import DownloadConfig, load_dataset
        ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            download_config=DownloadConfig(local_files_only=True),
        )
        combined = ds["all_tracks"]  # all_tracks already includes test_tracks
        all_ids = [str(t) for t in combined["track_id"]]
        raw_vecs = []
        valid_ids = []
        for i in range(len(combined)):
            v = combined[i]["cf-bpr"]
            if v is not None and len(v) == 128:
                raw_vecs.append(v)
                valid_ids.append(all_ids[i])
        vectors = np.array(raw_vecs, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors /= norms
        track_ids = valid_ids
        CFBPR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(vecs_path, vectors)
        with open(ids_path, "w") as f:
            json.dump(track_ids, f)
        print(f"  cached {len(track_ids)} tracks ({len(all_ids) - len(track_ids)} empty removed)")

    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"cf-bpr index: {len(track_ids)} tracks, dim={vectors.shape[1]}", flush=True)
    return track_ids, vectors, id_to_idx


# --- Retrieval variants --------------------------------------------------- #

def cfbpr_last1(played: list[str], vectors: np.ndarray, id_to_idx: dict,
                track_ids: list[str], topn: int) -> list[str]:
    """Nearest neighbors of last played track."""
    anchor = played[-1] if played else None
    if anchor is None or anchor not in id_to_idx:
        return []
    idx = id_to_idx[anchor]
    scores = vectors @ vectors[idx]
    exclude = {id_to_idx[t] for t in played if t in id_to_idx}
    cap = min(len(scores), topn + len(exclude))
    cand = np.argpartition(-scores, cap - 1)[:cap]
    cand = cand[np.argsort(-scores[cand])]
    out = []
    for i in cand:
        if int(i) in exclude:
            continue
        out.append(track_ids[int(i)])
        if len(out) >= topn:
            break
    return out


def cfbpr_max_recent(played: list[str], vectors: np.ndarray, id_to_idx: dict,
                     track_ids: list[str], recent_k: int, topn: int) -> list[str]:
    """Max cosine over last K played tracks (same as A' but in cf-bpr space)."""
    anchors_idx = [id_to_idx[t] for t in played[-recent_k:] if t in id_to_idx]
    if not anchors_idx:
        return []
    anchor_vecs = vectors[anchors_idx]  # (k, D)
    sims = vectors @ anchor_vecs.T      # (N, k)
    scores = sims.max(axis=1)           # (N,)
    exclude = {id_to_idx[t] for t in played if t in id_to_idx}
    cap = min(len(scores), topn + len(exclude))
    cand = np.argpartition(-scores, cap - 1)[:cap]
    cand = cand[np.argsort(-scores[cand])]
    out = []
    for i in cand:
        if int(i) in exclude:
            continue
        out.append(track_ids[int(i)])
        if len(out) >= topn:
            break
    return out


def cfbpr_mean_recent(played: list[str], vectors: np.ndarray, id_to_idx: dict,
                      track_ids: list[str], recent_k: int, topn: int) -> list[str]:
    """Mean-pooled user vector from last K played tracks."""
    anchors_idx = [id_to_idx[t] for t in played[-recent_k:] if t in id_to_idx]
    if not anchors_idx:
        return []
    user_vec = vectors[anchors_idx].mean(axis=0)
    user_vec /= np.linalg.norm(user_vec) + 1e-9
    scores = vectors @ user_vec
    exclude = {id_to_idx[t] for t in played if t in id_to_idx}
    cap = min(len(scores), topn + len(exclude))
    cand = np.argpartition(-scores, cap - 1)[:cap]
    cand = cand[np.argsort(-scores[cand])]
    out = []
    for i in cand:
        if int(i) in exclude:
            continue
        out.append(track_ids[int(i)])
        if len(out) >= topn:
            break
    return out


# --- Weighted RRF (same as r3_det) ---------------------------------------- #

def weighted_rrf(sources: dict[str, list[str]], weights: dict[str, float],
                 topk: int, k: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for name, ranked in sources.items():
        w = weights.get(name, 0.0)
        if w == 0 or not ranked:
            continue
        for rank, tid in enumerate(ranked, start=1):
            scores[tid] = scores.get(tid, 0.0) + w / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]


# --- Build pool features with F source ------------------------------------ #

def build_pool_features_with_f(
    payload: dict, src_f: list[list[str]],
    source_weights: dict[str, float], pool_k: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as build_pool_features but adds source F to RRF fusion."""
    cases = payload["cases"]
    src_b = payload["src_b"]
    src_c = payload["src_c"]
    src_d = payload["src_d"]
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]
    from scripts.tune_postrank_v23 import tokens

    n = len(cases)
    f_count = len(FEATURE_NAMES)
    rrf_k = 20
    X = np.zeros((n, pool_k, f_count), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        sources_dict = {"B": src_b[i], "C": src_c[i], "D": src_d[i], "F": src_f[i]}
        pool = weighted_rrf(sources_dict, source_weights, topk=pool_k, k=rrf_k)
        sizes[i] = len(pool)

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

        for rank, tid in enumerate(pool, start=1):
            if rank > pool_k:
                break
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
            for w_dec, p_artist, p_tags in prior:
                artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                tag_match = len(cand_tags & p_tags) / len(cand_tags | p_tags) if (cand_tags or p_tags) else 0.0
                rec += w_dec * (artist_match + tag_match)
            row[7] = rec

        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

    return X, gt_idx, sizes


# --- Main ----------------------------------------------------------------- #

def main() -> None:
    t0 = time.time()

    # Load cf-bpr index
    track_ids, vectors, id_to_idx = build_cfbpr_index()

    # Load the same 400-session devset payload
    print("\nloading 400-session devset payload...", flush=True)
    payload = build_or_load_payload()
    cases = payload["cases"]
    n = len(cases)
    print(f"loaded {n} cases", flush=True)

    # Ensure metadata maps cover F candidates
    from offline_retrieval_sweep import load_track_metadata
    metadata = load_track_metadata()
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]
    from scripts.tune_postrank_v23 import tokens as tok_fn

    def ensure_meta(tids: list[str]):
        for tid in tids:
            if tid in track_artist:
                continue
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
            track_title_toks[tid] = tok_fn(title)
            track_artist_toks[tid] = tok_fn(meta.get("artist_name", ""))
            track_meta_toks[tid] = tok_fn(" ".join(meta_parts))

    # =====================================================================
    # SANITY CHECK: validate cf-bpr index integrity before any metrics
    # =====================================================================
    print("\n" + "=" * 70)
    print("SANITY CHECK: CF-BPR index integrity")
    print("=" * 70)

    # Build set of empty-vector track IDs (the 1232 we filtered)
    # We can verify by checking that no returned ID maps to an index outside our vectors
    assert len(track_ids) == vectors.shape[0], (
        f"track_ids/vectors mismatch: {len(track_ids)} vs {vectors.shape[0]}"
    )
    assert len(id_to_idx) == len(track_ids), (
        f"id_to_idx/track_ids mismatch: {len(id_to_idx)} vs {len(track_ids)}"
    )
    # Verify id_to_idx is consistent with track_ids ordering
    for check_i in range(0, len(track_ids), len(track_ids) // 20):
        assert id_to_idx[track_ids[check_i]] == check_i, (
            f"id_to_idx inconsistent at {check_i}: "
            f"track_ids[{check_i}]={track_ids[check_i]}, "
            f"id_to_idx gives {id_to_idx[track_ids[check_i]]}"
        )
    print(f"  index consistency: OK ({len(track_ids)} tracks, dim={vectors.shape[1]})")

    # 5 random anchor tracks — full retrieval validation
    import random
    rng = random.Random(42)
    valid_anchors = [c["music_turns"][-1] for c in cases if c["music_turns"] and c["music_turns"][-1] in id_to_idx]
    sample_anchors = rng.sample(valid_anchors, min(5, len(valid_anchors)))
    print(f"  testing 5 anchor tracks:")
    for anchor_tid in sample_anchors:
        neighbors = cfbpr_last1([anchor_tid], vectors, id_to_idx, track_ids, 10)
        # anchor excluded
        assert anchor_tid not in neighbors, f"anchor {anchor_tid} found in its own neighbors"
        # every neighbor maps back to a valid index
        for nb in neighbors:
            assert nb in id_to_idx, f"neighbor {nb} not in id_to_idx"
            nb_idx = id_to_idx[nb]
            assert 0 <= nb_idx < vectors.shape[0], f"neighbor {nb} index {nb_idx} out of range"
            assert track_ids[nb_idx] == nb, f"round-trip failed: track_ids[{nb_idx}]={track_ids[nb_idx]} != {nb}"
        print(f"    anchor={anchor_tid}  neighbors={neighbors[:5]}  (all valid, anchor excluded)")
    print("  anchor retrieval checks: PASS")

    # Verify no returned neighbor has a zero-norm vector (would indicate empty-vec leak)
    for anchor_tid in sample_anchors:
        neighbors = cfbpr_last1([anchor_tid], vectors, id_to_idx, track_ids, 50)
        for nb in neighbors:
            nb_norm = float(np.linalg.norm(vectors[id_to_idx[nb]]))
            assert nb_norm > 0.1, f"neighbor {nb} has near-zero norm {nb_norm} — empty-vector leak"
    print("  zero-vector leak check: PASS")

    # Full variant validation helper
    cfbpr_id_set = set(track_ids)

    # Anchor policies: how many recent played tracks each variant uses
    ANCHOR_POLICIES = {
        "F_last1": 1,
        "F_max_recent3": 3,
        "F_max_recent5": 5,
        "F_mean_recent5": 5,
    }

    def validate_f_source(vname: str, src_lists: list[list[str]], cases_list: list[dict],
                          catalog_ids: set[str]) -> None:
        """Validate every F source list. Anchor coverage is variant-aware."""
        anchor_k = ANCHOR_POLICIES.get(vname, 5)
        errors = []
        n_no_valid_anchor = 0
        n_short_warned = 0
        for i, (f_list, c) in enumerate(zip(src_lists, cases_list)):
            played = c["music_turns"]
            played_set = set(played)
            # Variant-specific anchors
            variant_anchors = played[-anchor_k:]
            has_valid_anchor = any(t in cfbpr_id_set for t in variant_anchors)
            # No duplicates
            if len(f_list) != len(set(f_list)):
                errors.append(f"  case {i}: {len(f_list) - len(set(f_list))} duplicate IDs")
            # No played tracks returned
            leaked = set(f_list) & played_set
            if leaked:
                errors.append(f"  case {i}: {len(leaked)} played tracks leaked: {list(leaked)[:3]}")
            # All IDs are valid catalog IDs
            invalid = [tid for tid in f_list if tid not in catalog_ids]
            if invalid:
                errors.append(f"  case {i}: {len(invalid)} invalid catalog IDs: {invalid[:3]}")
            # Empty-list check: only an error if variant-specific anchors exist in cf-bpr
            if not has_valid_anchor:
                n_no_valid_anchor += 1
            elif played and len(f_list) == 0:
                errors.append(f"  case {i}: empty list despite valid anchors (last {anchor_k})")
            # Short non-empty list: warning only
            elif 0 < len(f_list) < 50:
                n_short_warned += 1
        if errors:
            print(f"  {vname}: VALIDATION FAILED")
            for e in errors[:10]:
                print(e)
            raise AssertionError(f"{vname} validation failed with {len(errors)} errors")
        parts = [f"{len(src_lists)} lists"]
        parts.append(f"{n_no_valid_anchor} no valid anchor (last {anchor_k})")
        if n_short_warned:
            parts.append(f"{n_short_warned} short (<50) warned")
        print(f"  {vname}: PASS ({', '.join(parts)})")

    # Build catalog ID set for validation
    all_catalog_ids = set(track_ids)  # cf-bpr tracks are a subset of catalog
    # Also include tracks from metadata (which covers all catalog tracks)
    all_catalog_ids.update(metadata.keys())

    print("  sanity checks complete\n")

    # =====================================================================
    # PHASE 1: Source-alone diagnostics
    # =====================================================================
    print("=" * 70)
    print("PHASE 1: CF-BPR source-alone diagnostics")
    print("=" * 70)

    F_DEPTH = 200
    variants = {
        "F_last1": lambda played: cfbpr_last1(played, vectors, id_to_idx, track_ids, F_DEPTH),
        "F_max_recent3": lambda played: cfbpr_max_recent(played, vectors, id_to_idx, track_ids, 3, F_DEPTH),
        "F_max_recent5": lambda played: cfbpr_max_recent(played, vectors, id_to_idx, track_ids, 5, F_DEPTH),
        "F_mean_recent5": lambda played: cfbpr_mean_recent(played, vectors, id_to_idx, track_ids, 5, F_DEPTH),
    }

    # Also build cfg0209 baseline pool for comparison
    cfg0209_weights = {"B": 1.0, "C": 1.0, "D": 0.5}
    cfg0209_pools = []
    for i, c in enumerate(cases):
        sources_dict = {"B": payload["src_b"][i], "C": payload["src_c"][i], "D": payload["src_d"][i]}
        pool = weighted_rrf(sources_dict, cfg0209_weights, topk=50, k=20)
        cfg0209_pools.append(set(pool))

    # Also get A' source for overlap analysis
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    print("computing A' (max_recent_5 qwen3) for overlap analysis...", flush=True)
    src_a_prime = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = qwen_sim.vectors[a_idxs]
            sims = qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims.max(axis=1)
            exclude_a = {qwen_sim._id_to_idx[t] for t in played if t in qwen_sim._id_to_idx}
            cap_a = min(len(scores_a), 200 + len(exclude_a))
            cand_a = np.argpartition(-scores_a, cap_a - 1)[:cap_a]
            cand_a = cand_a[np.argsort(-scores_a[cand_a])]
            out_a = []
            for ii in cand_a:
                if int(ii) in exclude_a:
                    continue
                out_a.append(qwen_sim.track_ids[int(ii)])
                if len(out_a) >= 200:
                    break
            src_a_prime.append(out_a)
        else:
            src_a_prime.append([])

    # cfg0209 with A' (the actual submission config)
    cfg0209_full_pools = []
    for i, c in enumerate(cases):
        sources_dict = {"A": src_a_prime[i], "B": payload["src_b"][i], "C": payload["src_c"][i], "D": payload["src_d"][i]}
        w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
        pool = weighted_rrf(sources_dict, w, topk=50, k=20)
        cfg0209_full_pools.append(set(pool))

    phase1_results = {}
    for vname, retriever_fn in variants.items():
        print(f"\n--- {vname} @ {F_DEPTH} ---")
        src_f_lists = []
        for c in cases:
            played = c["music_turns"]
            result = retriever_fn(played) if played else []
            ensure_meta(result)
            src_f_lists.append(result)

        # Validate this variant's output (variant-aware anchor policy)
        validate_f_source(vname, src_f_lists, cases, all_catalog_ids)

        # Recall metrics
        gt_hits = {k: 0 for k in [10, 20, 50, 100, 200]}
        gt_ranks = []
        unique_gt_vs_cfg0209 = 0
        unique_gt_vs_cfg0209_full = 0
        overlap_with_d = []
        overlap_with_a = []
        n_with_history = 0

        for i, c in enumerate(cases):
            gt = c["gt"]
            f_list = src_f_lists[i]
            d_list = payload["src_d"][i]
            a_list = src_a_prime[i]

            if not c["music_turns"]:
                continue
            n_with_history += 1

            # Recall at various depths
            for k in gt_hits:
                if gt in f_list[:k]:
                    gt_hits[k] += 1

            # GT rank
            if gt in f_list:
                gt_ranks.append(f_list.index(gt) + 1)

            # Unique GT hits vs cfg0209 pool (BCD only, no A')
            if gt in f_list[:200] and gt not in cfg0209_pools[i]:
                unique_gt_vs_cfg0209 += 1

            # Unique GT hits vs cfg0209 full pool (ABCD)
            if gt in f_list[:200] and gt not in cfg0209_full_pools[i]:
                unique_gt_vs_cfg0209_full += 1

            # Overlap with D (qwen3 track neighbors)
            f_set = set(f_list[:200])
            d_set = set(d_list[:200])
            a_set = set(a_list[:200])
            if f_set:
                overlap_with_d.append(len(f_set & d_set) / len(f_set))
                overlap_with_a.append(len(f_set & a_set) / len(f_set))

        result = {
            "n_with_history": n_with_history,
            "recall": {f"@{k}": gt_hits[k] / n_with_history for k in gt_hits},
            "recall_raw": {f"@{k}": gt_hits[k] for k in gt_hits},
            "gt_rank_when_hit": {
                "n_hits": len(gt_ranks),
                "median": float(np.median(gt_ranks)) if gt_ranks else None,
                "mean": float(np.mean(gt_ranks)) if gt_ranks else None,
                "p25": float(np.percentile(gt_ranks, 25)) if gt_ranks else None,
                "p75": float(np.percentile(gt_ranks, 75)) if gt_ranks else None,
            },
            "unique_gt_hits_vs_cfg0209_BCD_pool50": unique_gt_vs_cfg0209,
            "unique_gt_hits_vs_cfg0209_ABCD_pool50": unique_gt_vs_cfg0209_full,
            "overlap_with_D_qwen3": {
                "mean": float(np.mean(overlap_with_d)) if overlap_with_d else None,
                "median": float(np.median(overlap_with_d)) if overlap_with_d else None,
            },
            "overlap_with_A_qwen3": {
                "mean": float(np.mean(overlap_with_a)) if overlap_with_a else None,
                "median": float(np.median(overlap_with_a)) if overlap_with_a else None,
            },
        }
        phase1_results[vname] = result
        print(f"  recall: " + ", ".join(f"@{k}={v:.4f}" for k, v in result["recall"].items()))
        print(f"  GT rank (when hit): median={result['gt_rank_when_hit']['median']}, n={result['gt_rank_when_hit']['n_hits']}")
        print(f"  unique GT vs cfg0209 BCD pool50: {unique_gt_vs_cfg0209}")
        print(f"  unique GT vs cfg0209 ABCD pool50: {unique_gt_vs_cfg0209_full}")
        print(f"  overlap with D(qwen3): mean={result['overlap_with_D_qwen3']['mean']:.3f}")
        print(f"  overlap with A'(qwen3): mean={result['overlap_with_A_qwen3']['mean']:.3f}")

    # Pick best variant for Phase 2
    best_var = max(phase1_results, key=lambda v: phase1_results[v]["unique_gt_hits_vs_cfg0209_ABCD_pool50"])
    print(f"\nBest F variant by unique GT hits vs ABCD: {best_var}")

    # =====================================================================
    # PHASE 2: Fusion with cfg0209
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"PHASE 2: Fusion eval — cfg0209 + {best_var}")
    print("=" * 70)

    # Rebuild F source for best variant at multiple depths
    best_retriever = variants[best_var]
    src_f_full = []
    for c in cases:
        played = c["music_turns"]
        result = best_retriever(played) if played else []
        ensure_meta(result)
        src_f_full.append(result)

    # Also rebuild with all 4 variants at depth 500 for thoroughness
    # (some variants might be better at fusion even if worse standalone)

    # Fusion grid: vary F weight and depth
    f_depths = [50, 100, 200]
    f_weights_grid = [0.25, 0.5, 1.0, 2.0]
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}

    # cfg0209 baseline (A+B+C+D, no F)
    print("\nBaseline cfg0209 (ABCD, no F):")
    baseline_X, baseline_gt, baseline_sizes = build_pool_features(payload, (1.0, 1.0, 0.5))
    # Wait — build_pool_features uses BCD only, not A'. Let me compute baseline with A' properly.
    # Actually build_pool_features from r3_confirm uses only BCD. The full cfg0209 uses A+B+C+D.
    # Let me build the full ABCD baseline and the F-augmented version using the same function.

    # Build ABCD baseline pools and features
    src_f_empty = [[] for _ in cases]
    baseline_sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 0.0}

    # We need to add A' source to the payload for the fusion function
    # Extend payload with src_a_prime
    payload["src_a"] = src_a_prime

    def build_features_abcdf(src_f_lists, source_weights, pool_k=50):
        """Build pool features for A+B+C+D+F fusion."""
        from scripts.tune_postrank_v23 import tokens
        rrf_k = 20
        n = len(cases)
        f_count = len(FEATURE_NAMES)
        X = np.zeros((n, pool_k, f_count), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            sources_dict = {
                "A": src_a_prime[i],
                "B": payload["src_b"][i],
                "C": payload["src_c"][i],
                "D": payload["src_d"][i],
                "F": src_f_lists[i],
            }
            pool = weighted_rrf(sources_dict, source_weights, topk=pool_k, k=rrf_k)
            sizes[i] = len(pool)

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

            for rank, tid in enumerate(pool, start=1):
                if rank > pool_k:
                    break
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
                for w_dec, p_artist, p_tags in prior:
                    artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
                    tag_match = len(cand_tags & p_tags) / len(cand_tags | p_tags) if (cand_tags or p_tags) else 0.0
                    rec += w_dec * (artist_match + tag_match)
                row[7] = rec

            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])

        return X, gt_idx, sizes

    # Baseline: ABCD no F
    print("building ABCD baseline features...", flush=True)
    bl_sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 0.0}
    bl_X, bl_gt, bl_sizes = build_features_abcdf(src_f_empty, bl_sw)
    bl_pool_hit = float(np.mean(bl_gt >= 0))
    bl_median_rank = float(np.median(bl_gt[bl_gt >= 0] + 1)) if (bl_gt >= 0).any() else 999
    print(f"  pool_hit@50: {bl_pool_hit:.4f} ({(bl_gt >= 0).sum()}/{n})")
    print(f"  median GT rank: {bl_median_rank:.1f}")

    # CV5 for baseline
    sessions = [c["session_id"] for c in cases]
    seeds = [0, 1, 2]
    bl_cv5_per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([i for i in range(n) if i not in held], dtype=np.int64)
            weights, _ = fit_weights(bl_X, bl_gt, bl_sizes, train)
            fold_scores.append(vec_ndcg(bl_X, bl_gt, bl_sizes, weights, fold))
        bl_cv5_per_seed.append(float(np.mean(fold_scores)))
    bl_cv5 = float(np.mean(bl_cv5_per_seed))
    print(f"  CV5 nDCG: {bl_cv5:.4f} (per-seed: {bl_cv5_per_seed})")

    # Fusion grid
    print(f"\n--- Fusion grid: {best_var} ---")
    fusion_results = []
    for f_depth in f_depths:
        src_f_truncated = [fl[:f_depth] for fl in src_f_full]
        for f_weight in f_weights_grid:
            sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": f_weight}
            X, gt_idx, sizes = build_features_abcdf(src_f_truncated, sw)
            pool_hit = float(np.mean(gt_idx >= 0))
            med_rank = float(np.median(gt_idx[gt_idx >= 0] + 1)) if (gt_idx >= 0).any() else 999
            fusion_results.append({
                "f_depth": f_depth,
                "f_weight": f_weight,
                "pool_hit_50": pool_hit,
                "pool_hit_raw": int((gt_idx >= 0).sum()),
                "median_gt_rank": med_rank,
                "X": X,
                "gt_idx": gt_idx,
                "sizes": sizes,
            })
            delta_hit = pool_hit - bl_pool_hit
            print(f"  depth={f_depth:3d} w_F={f_weight:.2f}  pool_hit={pool_hit:.4f} (Δ={delta_hit:+.4f})  med_rank={med_rank:.1f}")

    # Select top 3 configs by combined metric: pool_hit - 0.01 * median_rank
    fusion_results.sort(key=lambda r: r["pool_hit_50"] - 0.01 * r["median_gt_rank"], reverse=True)
    top3 = fusion_results[:3]

    print(f"\n--- Powell CV5 on top 3 fusion configs ---")
    cv5_results = []
    for cfg in top3:
        X, gt_idx, sizes = cfg["X"], cfg["gt_idx"], cfg["sizes"]
        cv5_per_seed = []
        for seed in seeds:
            folds = cv_folds(sessions, seed)
            fold_scores = []
            for fold in folds:
                held = set(fold.tolist())
                train_idx = np.asarray([i for i in range(n) if i not in held], dtype=np.int64)
                w, _ = fit_weights(X, gt_idx, sizes, train_idx)
                fold_scores.append(vec_ndcg(X, gt_idx, sizes, w, fold))
            cv5_per_seed.append(float(np.mean(fold_scores)))
        cv5_mean = float(np.mean(cv5_per_seed))
        entry = {
            "f_depth": cfg["f_depth"],
            "f_weight": cfg["f_weight"],
            "pool_hit_50": cfg["pool_hit_50"],
            "median_gt_rank": cfg["median_gt_rank"],
            "cv5_mean": cv5_mean,
            "cv5_per_seed": cv5_per_seed,
            "cv5_delta_vs_baseline": cv5_mean - bl_cv5,
        }
        cv5_results.append(entry)
        print(f"  depth={cfg['f_depth']:3d} w_F={cfg['f_weight']:.2f}  "
              f"CV5={cv5_mean:.4f} (Δ={cv5_mean - bl_cv5:+.4f})  "
              f"pool_hit={cfg['pool_hit_50']:.4f}  med_rank={cfg['median_gt_rank']:.1f}")

    # =====================================================================
    # Also run all 4 F variants at fixed depth=200 w=1.0 for comparison
    # =====================================================================
    print(f"\n--- All variants @ depth=200 w=1.0 ---")
    variant_cv5 = {}
    for vname, retriever_fn in variants.items():
        src_f_v = []
        for c in cases:
            played = c["music_turns"]
            result = retriever_fn(played)[:200] if played else []
            ensure_meta(result)
            src_f_v.append(result)
        sw_fixed = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        X_v, gt_v, sz_v = build_features_abcdf(src_f_v, sw_fixed)
        ph_v = float(np.mean(gt_v >= 0))
        cv5_seeds_v = []
        for seed in seeds:
            folds = cv_folds(sessions, seed)
            fold_sc = []
            for fold in folds:
                held = set(fold.tolist())
                train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
                w, _ = fit_weights(X_v, gt_v, sz_v, train_idx)
                fold_sc.append(vec_ndcg(X_v, gt_v, sz_v, w, fold))
            cv5_seeds_v.append(float(np.mean(fold_sc)))
        cv5_v = float(np.mean(cv5_seeds_v))
        variant_cv5[vname] = {"cv5": cv5_v, "pool_hit": ph_v, "cv5_per_seed": cv5_seeds_v}
        print(f"  {vname:20s}  CV5={cv5_v:.4f} (Δ={cv5_v - bl_cv5:+.4f})  pool_hit={ph_v:.4f}")

    # =====================================================================
    # Summary & gate
    # =====================================================================
    elapsed = time.time() - t0
    best_cv5_entry = max(cv5_results, key=lambda x: x["cv5_mean"])
    best_cv5 = best_cv5_entry["cv5_mean"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline cfg0209 ABCD:  CV5={bl_cv5:.4f}  pool_hit@50={bl_pool_hit:.4f}")
    print(f"Best fusion config:     CV5={best_cv5:.4f} (Δ={best_cv5 - bl_cv5:+.4f})"
          f"  depth={best_cv5_entry['f_depth']} w_F={best_cv5_entry['f_weight']}")
    print(f"Best F variant at d=200 w=1.0: {max(variant_cv5, key=lambda v: variant_cv5[v]['cv5'])}")

    if best_cv5 >= 0.178:
        verdict = "PASS"
    elif best_cv5 >= 0.170 or (best_cv5_entry["pool_hit_50"] - bl_pool_hit) >= 0.04:
        verdict = "PROMISING"
    elif best_cv5 >= 0.165:
        verdict = "WEAK"
    else:
        unique_lift = max(
            phase1_results[v]["unique_gt_hits_vs_cfg0209_ABCD_pool50"] / max(phase1_results[v]["n_with_history"], 1)
            for v in phase1_results
        )
        if unique_lift < 0.03:
            verdict = "FAIL"
        else:
            verdict = "WEAK"

    print(f"\nGATE VERDICT: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Save artifact
    out_path = REPO_ROOT / "exp" / "eval" / "expF1_cfbpr_retrieval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {
            "n_cases": n,
            "cfbpr_tracks": len(track_ids),
            "cfbpr_dim": int(vectors.shape[1]),
            "elapsed_sec": elapsed,
            "seeds": seeds,
        },
        "phase1": phase1_results,
        "baseline_cfg0209_ABCD": {
            "pool_hit_50": bl_pool_hit,
            "median_gt_rank": bl_median_rank,
            "cv5_mean": bl_cv5,
            "cv5_per_seed": bl_cv5_per_seed,
        },
        "fusion_grid": [
            {k: v for k, v in r.items() if k not in ("X", "gt_idx", "sizes")}
            for r in fusion_results
        ],
        "top3_cv5": cv5_results,
        "all_variants_d200_w1": variant_cv5,
        "gate_verdict": verdict,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
