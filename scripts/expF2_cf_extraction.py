#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""F2 experiment: CF-aware extraction — reserved-slot fusion + CF postrank features.

Builds on F1 winner (F_max_recent5, depth=200, w_F=1.0, CV5=0.1721).

Part A: Reserved-slot pool construction variants.
Part B: CF-aware postrank features (13 features = 8 base + 5 CF).
Part C: Robustness (5-fold CV × 5 seeds on top-2 configs).

No API calls. Same 400-session devset slice.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (
    build_or_load_payload,
    cv_folds,
    fit_weights as fit_weights_8f,
    vec_ndcg as vec_ndcg_base,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from offline_retrieval_sweep import load_track_metadata

# --- Constants -------------------------------------------------------------- #

POOL_K = 50
RRF_K = 20
F_DEPTH = 200
F_RECENT_K = 5

# 13-feature names: 8 original + 5 CF
FEATURE_NAMES_13 = list(FEATURE_NAMES) + [
    "cf_rank_recip",          # 1/(cf_rank+1), 0 if not from CF
    "cf_max_sim_recent5",     # max cosine sim to recent-5 anchors in cf-bpr space
    "cf_mean_sim_recent5",    # mean cosine sim to recent-5 anchors
    "cf_anchor_hit_count",    # how many of the recent-5 anchors retrieve this in top200
    "cf_in_source",           # 1 if candidate appears in CF source, else 0
]

INIT_WEIGHTS_13 = dict(INIT_WEIGHTS)
INIT_WEIGHTS_13.update({
    "cf_rank_recip": 1.0,
    "cf_max_sim_recent5": 0.5,
    "cf_mean_sim_recent5": 0.3,
    "cf_anchor_hit_count": 0.3,
    "cf_in_source": 0.5,
})


# --- CF scored retrieval --------------------------------------------------- #

def cfbpr_max_recent_scored(
    played: list[str], vectors: np.ndarray, id_to_idx: dict,
    track_ids: list[str], recent_k: int, topn: int,
) -> list[tuple[str, float, int]]:
    """Like cfbpr_max_recent but returns (track_id, max_sim, n_anchors_hitting).

    n_anchors_hitting: how many of the recent-K anchor tracks have this candidate
    in their individual top-200 neighbors.
    """
    anchors_idx = [id_to_idx[t] for t in played[-recent_k:] if t in id_to_idx]
    if not anchors_idx:
        return []
    anchor_vecs = vectors[anchors_idx]  # (k, D)
    sims = vectors @ anchor_vecs.T      # (N, k)
    max_scores = sims.max(axis=1)       # (N,)
    mean_scores = sims.mean(axis=1)     # (N,)

    # Per-anchor top-200 membership
    anchor_hits = np.zeros(len(vectors), dtype=np.int32)
    for a in range(len(anchors_idx)):
        col = sims[:, a]
        cap = min(len(col), topn + len(anchors_idx))
        top_idx = np.argpartition(-col, cap - 1)[:cap]
        anchor_hits[top_idx] += 1

    exclude = {id_to_idx[t] for t in played if t in id_to_idx}
    cap = min(len(max_scores), topn + len(exclude))
    cand = np.argpartition(-max_scores, cap - 1)[:cap]
    cand = cand[np.argsort(-max_scores[cand])]
    out = []
    for i in cand:
        if int(i) in exclude:
            continue
        out.append((
            track_ids[int(i)],
            float(max_scores[int(i)]),
            float(mean_scores[int(i)]),
            int(anchor_hits[int(i)]),
        ))
        if len(out) >= topn:
            break
    return out


# --- Pool construction variants -------------------------------------------- #

def build_rrf_pool(src_a, src_b, src_c, src_d, topk=50):
    """Standard cfg0209 ABCD RRF pool."""
    sources = {"A": src_a, "B": src_b, "C": src_c, "D": src_d}
    weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
    return weighted_rrf(sources, weights, topk=topk, k=RRF_K)


def build_rrf_pool_with_f(src_a, src_b, src_c, src_d, src_f_ids, topk=50):
    """cfg0209 + CF-BPR via standard weighted RRF (F1 winner config)."""
    sources = {"A": src_a, "B": src_b, "C": src_c, "D": src_d, "F": src_f_ids}
    weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
    return weighted_rrf(sources, weights, topk=topk, k=RRF_K)


def build_reserved_slot_pool(src_a, src_b, src_c, src_d, src_f_ids,
                             base_slots: int, cf_slots: int):
    """Base top-N via ABCD RRF + top-M CF candidates not already in base."""
    base = build_rrf_pool(src_a, src_b, src_c, src_d, topk=base_slots)
    base_set = set(base)
    cf_fill = [tid for tid in src_f_ids if tid not in base_set][:cf_slots]
    return base + cf_fill


def build_interleaved_pool(src_a, src_b, src_c, src_d, src_f_ids,
                           cf_every: int, max_cf: int, total: int = 50):
    """ABCD RRF base with CF candidates interleaved every N slots."""
    base = build_rrf_pool(src_a, src_b, src_c, src_d, topk=total + max_cf)
    base_set = set()
    cf_queue = [tid for tid in src_f_ids if tid not in set(base[:total])]
    cf_iter = iter(cf_queue)
    cf_inserted = 0
    pool = []
    base_idx = 0
    for slot in range(total):
        if (slot + 1) % cf_every == 0 and cf_inserted < max_cf:
            try:
                cf_tid = next(cf_iter)
                pool.append(cf_tid)
                cf_inserted += 1
                continue
            except StopIteration:
                pass
        # Fill from base, skipping any already added
        while base_idx < len(base) and base[base_idx] in set(pool):
            base_idx += 1
        if base_idx < len(base):
            pool.append(base[base_idx])
            base_idx += 1
    return pool[:total]


# --- Feature building ------------------------------------------------------ #

def build_13f_features(
    pool: list[str],
    user_messages: list[str],
    music_turns: list[str],
    cf_scored: dict[str, tuple[float, float, int]],
    cf_rank_map: dict[str, int],
    track_artist: dict, track_tags: dict,
    track_title_toks: dict, track_artist_toks: dict, track_meta_toks: dict,
    pool_k: int = POOL_K,
) -> np.ndarray:
    """Build 13-feature matrix for one case: 8 base + 5 CF features."""
    F = len(FEATURE_NAMES_13)
    X = np.zeros((pool_k, F), dtype=np.float64)
    if not pool:
        return X

    now_tokens = tokens(user_messages[-1]) if user_messages else set()
    all_user_tokens = tokens(" ".join(user_messages)) if user_messages else set()
    played_set = set(music_turns)
    last_artist = track_artist.get(music_turns[-1], "") if music_turns else ""
    last_tags = track_tags.get(music_turns[-1], set()) if music_turns else set()
    prior = [
        (1.0 / (j + 1), track_artist.get(tid, ""), track_tags.get(tid, set()))
        for j, tid in enumerate(reversed(music_turns))
    ]

    for rank, tid in enumerate(pool[:pool_k], start=1):
        cand_artist = track_artist.get(tid, "")
        cand_tags = track_tags.get(tid, set())
        row = X[rank - 1]
        # 8 base features (same as Powell)
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

        # 5 CF features
        cf_r = cf_rank_map.get(tid)
        row[8] = 1.0 / (cf_r + 1) if cf_r is not None else 0.0
        cf_info = cf_scored.get(tid)
        if cf_info:
            row[9] = cf_info[0]   # max_sim
            row[10] = cf_info[1]  # mean_sim
            row[11] = float(cf_info[2])  # anchor_hit_count
            row[12] = 1.0
        # else: all zeros (not from CF)

    return X


# --- Vectorized nDCG + fit for 13 features -------------------------------- #

def vec_ndcg_13(X, gt_idx, sizes, weights, idx):
    """Same as vec_ndcg_base but for arbitrary feature count."""
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


def fit_weights_13(X, gt_idx, sizes, train_idx):
    """Powell fit for 13-feature model."""
    init = np.array([INIT_WEIGHTS_13[name] for name in FEATURE_NAMES_13], dtype=np.float64)
    def objective(w):
        return -vec_ndcg_13(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


# --- Evaluation helpers ---------------------------------------------------- #

def eval_cv5(X, gt_idx, sizes, sessions, seeds, fit_fn, ndcg_fn):
    """Run 5-fold CV across multiple seeds. Returns per-seed means."""
    n = len(sessions)
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_scores = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([i for i in range(n) if i not in held], dtype=np.int64)
            w, _ = fit_fn(X, gt_idx, sizes, train)
            fold_scores.append(ndcg_fn(X, gt_idx, sizes, w, fold))
        per_seed.append(float(np.mean(fold_scores)))
    return per_seed


# --- Main ------------------------------------------------------------------ #

def main():
    t0 = time.time()

    # Load CF-BPR index
    track_ids_cf, vectors_cf, id_to_idx_cf = build_cfbpr_index()
    cfbpr_id_set = set(track_ids_cf)

    # Load payload
    print("\nloading 400-session devset payload...", flush=True)
    payload = build_or_load_payload()
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"loaded {n} cases", flush=True)

    # Load metadata for feature building
    metadata = load_track_metadata()
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    def ensure_meta(tids):
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
            track_title_toks[tid] = tokens(title)
            track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
            track_meta_toks[tid] = tokens(" ".join(meta_parts))

    # Build A' source (reuse from F1 approach)
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    print("computing A' (max_recent_5 qwen3)...", flush=True)
    src_a_prime = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = qwen_sim.vectors[a_idxs]
            sims_a = qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims_a.max(axis=1)
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

    # Build CF-BPR scored source for all cases
    print("computing CF-BPR scored source (F_max_recent5 @ 200)...", flush=True)
    cf_scored_per_case = []  # list of list[(tid, max_sim, mean_sim, n_anchors)]
    for c in cases:
        played = c["music_turns"]
        if played:
            scored = cfbpr_max_recent_scored(
                played, vectors_cf, id_to_idx_cf, track_ids_cf,
                recent_k=F_RECENT_K, topn=F_DEPTH,
            )
        else:
            scored = []
        cf_scored_per_case.append(scored)
        # Ensure metadata for CF candidates
        ensure_meta([s[0] for s in scored])

    # Build per-case CF lookup dicts
    cf_id_lists = []      # just track_ids in CF rank order
    cf_scored_maps = []   # {tid: (max_sim, mean_sim, n_anchors)}
    cf_rank_maps = []     # {tid: 0-based rank}
    for scored in cf_scored_per_case:
        ids = [s[0] for s in scored]
        cf_id_lists.append(ids)
        cf_scored_maps.append({s[0]: (s[1], s[2], s[3]) for s in scored})
        cf_rank_maps.append({s[0]: r for r, s in enumerate(scored)})

    seeds_quick = [0, 1, 2]
    seeds_robust = [0, 1, 2, 3, 4]

    # =====================================================================
    # BASELINE: F1 naive RRF (for comparison)
    # =====================================================================
    print("\n" + "=" * 70)
    print("BASELINE: F1 naive RRF (ABCD+F w=1.0) with 8-feature Powell")
    print("=" * 70)

    bl_pools_8f = []
    bl_X_8f = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
    bl_gt_8f = np.full(n, -1, dtype=np.int64)
    bl_sizes_8f = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        pool = build_rrf_pool_with_f(
            src_a_prime[i], payload["src_b"][i], payload["src_c"][i],
            payload["src_d"][i], cf_id_lists[i],
        )
        bl_pools_8f.append(pool)
        bl_sizes_8f[i] = len(pool)
        if c["gt"] in pool:
            bl_gt_8f[i] = pool.index(c["gt"])

        # Build 8-feature row (reuse same logic)
        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = track_artist.get(played[-1], "") if played else ""
        l_tags = track_tags.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set())) for j,t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = track_artist.get(tid, "")
            ct = track_tags.get(tid, set())
            row = bl_X_8f[i, rank-1]
            row[0] = 1.0/rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
            row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
            row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

    bl_hit = float(np.mean(bl_gt_8f >= 0))
    bl_cv5 = eval_cv5(bl_X_8f, bl_gt_8f, bl_sizes_8f, sessions, seeds_quick,
                       fit_weights_8f, vec_ndcg_base)
    bl_cv5_mean = float(np.mean(bl_cv5))
    print(f"  pool_hit@50: {bl_hit:.4f} ({(bl_gt_8f >= 0).sum()}/{n})")
    print(f"  CV5 (8f): {bl_cv5_mean:.4f} (per-seed: {bl_cv5})")

    # =====================================================================
    # PART A: Reserved-slot fusion
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART A: Reserved-slot fusion variants")
    print("=" * 70)

    slot_configs = [
        ("rrf_base45_cf5", 45, 5),
        ("rrf_base40_cf10", 40, 10),
        ("rrf_base35_cf15", 35, 15),
        ("rrf_base30_cf20", 30, 20),
    ]

    partA_results = {}
    for cfg_name, base_slots, cf_slots in slot_configs:
        pools = []
        X_8f = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        n_cf_gt_admitted = 0

        for i, c in enumerate(cases):
            pool = build_reserved_slot_pool(
                src_a_prime[i], payload["src_b"][i], payload["src_c"][i],
                payload["src_d"][i], cf_id_lists[i],
                base_slots=base_slots, cf_slots=cf_slots,
            )
            pools.append(pool)
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])
                # Check if GT came via CF slot
                base_pool = build_rrf_pool(
                    src_a_prime[i], payload["src_b"][i], payload["src_c"][i],
                    payload["src_d"][i], topk=base_slots,
                )
                if c["gt"] not in base_pool and c["gt"] in cf_id_lists[i]:
                    n_cf_gt_admitted += 1

            # Build 8-feature row
            user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
            played = c["music_turns"]
            now_tok = tokens(user_msgs[-1]) if user_msgs else set()
            all_tok = tokens(" ".join(user_msgs))
            played_set = set(played)
            l_artist = track_artist.get(played[-1], "") if played else ""
            l_tags = track_tags.get(played[-1], set()) if played else set()
            prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set())) for j,t in enumerate(reversed(played))]
            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = track_artist.get(tid, "")
                ct = track_tags.get(tid, set())
                row = X_8f[i, rank-1]
                row[0] = 1.0/rank
                row[1] = 1.0 if ca and ca == l_artist else 0.0
                if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
                row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
                row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
                row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
                row[6] = 1.0 if tid in played_set else 0.0
                rec = 0.0
                for wd, pa, pt in prior:
                    am = 1.0 if ca and ca == pa else 0.0
                    tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                    rec += wd * (am + tm)
                row[7] = rec

        hit = float(np.mean(gt_idx >= 0))
        med_rank = float(np.median(gt_idx[gt_idx >= 0] + 1)) if (gt_idx >= 0).any() else 999
        cv5 = eval_cv5(X_8f, gt_idx, sizes, sessions, seeds_quick,
                        fit_weights_8f, vec_ndcg_base)
        cv5_mean = float(np.mean(cv5))
        partA_results[cfg_name] = {
            "base_slots": base_slots, "cf_slots": cf_slots,
            "pool_hit_50": hit, "pool_hit_raw": int((gt_idx >= 0).sum()),
            "median_gt_rank": med_rank,
            "cf_gt_admitted": n_cf_gt_admitted,
            "cv5_mean": cv5_mean, "cv5_per_seed": cv5,
            "cv5_delta_vs_f1": cv5_mean - bl_cv5_mean,
            "X": X_8f, "gt_idx": gt_idx, "sizes": sizes,
        }
        print(f"  {cfg_name:25s}  pool_hit={hit:.4f}  med_rank={med_rank:.1f}"
              f"  cf_gt_in={n_cf_gt_admitted}  CV5={cv5_mean:.4f} (Δ={cv5_mean - bl_cv5_mean:+.4f})")

    # Interleaved variant
    print("\n  Interleaved: every 5th slot from CF, max 10 CF")
    pools_il = []
    X_il = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
    gt_il = np.full(n, -1, dtype=np.int64)
    sz_il = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(cases):
        pool = build_interleaved_pool(
            src_a_prime[i], payload["src_b"][i], payload["src_c"][i],
            payload["src_d"][i], cf_id_lists[i],
            cf_every=5, max_cf=10,
        )
        pools_il.append(pool)
        sz_il[i] = len(pool)
        if c["gt"] in pool:
            gt_il[i] = pool.index(c["gt"])
        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = track_artist.get(played[-1], "") if played else ""
        l_tags = track_tags.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set())) for j,t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = track_artist.get(tid, "")
            ct = track_tags.get(tid, set())
            row = X_il[i, rank-1]
            row[0] = 1.0/rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
            row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
            row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

    hit_il = float(np.mean(gt_il >= 0))
    med_il = float(np.median(gt_il[gt_il >= 0] + 1)) if (gt_il >= 0).any() else 999
    cv5_il = eval_cv5(X_il, gt_il, sz_il, sessions, seeds_quick,
                       fit_weights_8f, vec_ndcg_base)
    cv5_il_mean = float(np.mean(cv5_il))
    partA_results["interleaved_5th_10cf"] = {
        "pool_hit_50": hit_il, "median_gt_rank": med_il,
        "cv5_mean": cv5_il_mean, "cv5_per_seed": cv5_il,
        "cv5_delta_vs_f1": cv5_il_mean - bl_cv5_mean,
        "X": X_il, "gt_idx": gt_il, "sizes": sz_il,
    }
    print(f"  {'interleaved_5th_10cf':25s}  pool_hit={hit_il:.4f}  med_rank={med_il:.1f}"
          f"  CV5={cv5_il_mean:.4f} (Δ={cv5_il_mean - bl_cv5_mean:+.4f})")

    # Best Part A config
    best_a_name = max(partA_results, key=lambda k: partA_results[k]["cv5_mean"])
    best_a = partA_results[best_a_name]
    print(f"\n  Best Part A: {best_a_name} CV5={best_a['cv5_mean']:.4f}")

    # =====================================================================
    # PART B: CF-aware postrank features (13 features)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART B: 13-feature Powell (8 base + 5 CF)")
    print("=" * 70)

    # Test on: (1) F1 naive RRF pool, (2) best Part A pool
    pool_configs_b = {
        "f1_naive_rrf": (bl_pools_8f, "F1 naive RRF"),
        f"partA_{best_a_name}": (None, f"Part A best: {best_a_name}"),
    }

    # Rebuild best Part A pools for 13f features
    if best_a_name == "interleaved_5th_10cf":
        best_a_pools = pools_il
    else:
        base_s = best_a["base_slots"]
        cf_s = best_a["cf_slots"]
        best_a_pools = []
        for i, c in enumerate(cases):
            pool = build_reserved_slot_pool(
                src_a_prime[i], payload["src_b"][i], payload["src_c"][i],
                payload["src_d"][i], cf_id_lists[i],
                base_slots=base_s, cf_slots=cf_s,
            )
            best_a_pools.append(pool)
    pool_configs_b[f"partA_{best_a_name}"] = (best_a_pools, f"Part A: {best_a_name}")

    partB_results = {}
    for pool_label, (pool_list, desc) in pool_configs_b.items():
        print(f"\n  --- {desc} with 13f Powell ---")
        X_13 = np.zeros((n, POOL_K, len(FEATURE_NAMES_13)), dtype=np.float64)
        gt_13 = np.full(n, -1, dtype=np.int64)
        sz_13 = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            pool = pool_list[i]
            sz_13[i] = len(pool)
            if c["gt"] in pool:
                gt_13[i] = pool.index(c["gt"])

            user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
            X_13[i] = build_13f_features(
                pool, user_msgs, c["music_turns"],
                cf_scored_maps[i], cf_rank_maps[i],
                track_artist, track_tags,
                track_title_toks, track_artist_toks, track_meta_toks,
            )

        hit = float(np.mean(gt_13 >= 0))
        med_rank = float(np.median(gt_13[gt_13 >= 0] + 1)) if (gt_13 >= 0).any() else 999
        cv5 = eval_cv5(X_13, gt_13, sz_13, sessions, seeds_quick,
                        fit_weights_13, vec_ndcg_13)
        cv5_mean = float(np.mean(cv5))
        partB_results[pool_label] = {
            "pool_hit_50": hit, "median_gt_rank": med_rank,
            "cv5_mean": cv5_mean, "cv5_per_seed": cv5,
            "cv5_delta_vs_f1": cv5_mean - bl_cv5_mean,
            "X": X_13, "gt_idx": gt_13, "sizes": sz_13,
        }
        print(f"    pool_hit={hit:.4f}  med_rank={med_rank:.1f}"
              f"  CV5(13f)={cv5_mean:.4f} (Δ vs F1 8f={cv5_mean - bl_cv5_mean:+.4f})")

    # Also test 13f on the pure ABCD+F naive RRF pool to isolate feature effect
    print(f"\n  --- F1 naive RRF with 8f vs 13f comparison ---")
    print(f"    8f CV5: {bl_cv5_mean:.4f}")
    print(f"   13f CV5: {partB_results['f1_naive_rrf']['cv5_mean']:.4f}"
          f"  (feature lift: {partB_results['f1_naive_rrf']['cv5_mean'] - bl_cv5_mean:+.4f})")

    # =====================================================================
    # PART C: Robustness (5 seeds on top 2 configs)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART C: Robustness — 5-seed CV5")
    print("=" * 70)

    # Collect all configs with their arrays
    all_configs = {}
    # F1 baseline 8f
    all_configs["f1_naive_8f"] = (bl_X_8f, bl_gt_8f, bl_sizes_8f, fit_weights_8f, vec_ndcg_base, "8f")
    # Part B configs (13f)
    for k, v in partB_results.items():
        all_configs[f"13f_{k}"] = (v["X"], v["gt_idx"], v["sizes"], fit_weights_13, vec_ndcg_13, "13f")
    # Best Part A 8f
    all_configs[f"8f_partA_{best_a_name}"] = (
        best_a["X"], best_a["gt_idx"], best_a["sizes"], fit_weights_8f, vec_ndcg_base, "8f"
    )

    # Sort by quick CV5 to pick top 2
    quick_scores = {}
    for k, (X, gt, sz, fit_fn, ndcg_fn, label) in all_configs.items():
        cv = eval_cv5(X, gt, sz, sessions, [0], fit_fn, ndcg_fn)
        quick_scores[k] = cv[0]
    ranked = sorted(quick_scores, key=quick_scores.__getitem__, reverse=True)
    top2 = ranked[:2]
    print(f"  Top 2 configs for robust eval: {top2}")
    print(f"  Quick scores: {[(k, f'{quick_scores[k]:.4f}') for k in ranked[:5]]}")

    partC_results = {}
    for k in top2:
        X, gt, sz, fit_fn, ndcg_fn, label = all_configs[k]
        cv5_robust = eval_cv5(X, gt, sz, sessions, seeds_robust, fit_fn, ndcg_fn)
        cv5_mean = float(np.mean(cv5_robust))
        cv5_std = float(np.std(cv5_robust, ddof=1))
        partC_results[k] = {
            "cv5_mean": cv5_mean,
            "cv5_std": cv5_std,
            "cv5_per_seed": cv5_robust,
            "features": label,
            "pool_hit": float(np.mean(gt >= 0)),
        }
        print(f"  {k:40s}  CV5={cv5_mean:.4f} ± {cv5_std:.4f}  ({cv5_robust})")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    best_config = max(partC_results, key=lambda k: partC_results[k]["cv5_mean"])
    best_cv5 = partC_results[best_config]["cv5_mean"]
    best_std = partC_results[best_config]["cv5_std"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"cfg0209 baseline:           CV5=0.1602")
    print(f"F1 naive RRF (8f):          CV5={bl_cv5_mean:.4f}")
    print(f"Best F2 config:             CV5={best_cv5:.4f} ± {best_std:.4f}  [{best_config}]")
    print(f"Total lift vs cfg0209:      {best_cv5 - 0.1602:+.4f}")
    print(f"Lift vs F1:                 {best_cv5 - bl_cv5_mean:+.4f}")

    if best_cv5 >= 0.178:
        verdict = "PASS"
    elif best_cv5 >= 0.175 and partC_results[best_config]["pool_hit"] >= 0.42:
        verdict = "STRONG PROMISING"
    elif best_cv5 >= 0.175:
        verdict = "PROMISING"
    elif best_cv5 < bl_cv5_mean:
        verdict = "FAIL"
    else:
        verdict = "WEAK"

    print(f"\nGATE VERDICT: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Save artifact
    out_path = REPO_ROOT / "exp" / "eval" / "expF2_cf_extraction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def clean(d):
        return {k: v for k, v in d.items() if k not in ("X", "gt_idx", "sizes")}

    artifact = {
        "meta": {"n_cases": n, "elapsed_sec": elapsed, "seeds_quick": seeds_quick, "seeds_robust": seeds_robust},
        "baseline_f1_naive_8f": {"cv5_mean": bl_cv5_mean, "cv5_per_seed": bl_cv5, "pool_hit_50": bl_hit},
        "partA_reserved_slots": {k: clean(v) for k, v in partA_results.items()},
        "partA_best": best_a_name,
        "partB_13f": {k: clean(v) for k, v in partB_results.items()},
        "partC_robustness": partC_results,
        "best_config": best_config,
        "gate_verdict": verdict,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
