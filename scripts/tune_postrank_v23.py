# ruff: noqa: T201
"""Offline supervised post-rerank tuner for v23 (no API calls, no new inference).

Loads the v23 200-row inference artifact, reconstructs per-row session history
and last-user-query context from the HuggingFace devset, computes deterministic
features per (row, candidate), and fits a linear scorer to maximize train
nDCG@20. Evaluates on a held-out split and reports bootstrap CIs + drop-one
feature ablation.

Usage:
  .venv/bin/python scripts/tune_postrank_v23.py --split 100_100 --seed 0
  .venv/bin/python scripts/tune_postrank_v23.py --cv 5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

# Make sibling modules importable (mirrors eval_inference.py's repo-rooted layout).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from eval_inference import (
    build_ground_truth,
    cached_test_arrow_path,
)
from mcrs.db_item.music_catalog import MusicCatalogDB


def _stable_hash(s: str) -> int:
    """Non-cryptographic deterministic hash for train/holdout split keys only."""
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your",
    "song", "songs", "track", "tracks", "music", "artist", "artists",
    "band", "bands", "more", "another", "please", "some", "have",
    "like", "want", "really", "something", "can", "could", "would",
    "from", "about", "into", "give", "play", "just", "any",
}


def tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9']+", str(text).lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def tag_set(meta: dict) -> set[str]:
    tags = meta.get("tag_list") or []
    if isinstance(tags, list):
        return {str(t).lower().strip() for t in tags if str(t).strip()}
    return set()


def artist_str(meta: dict) -> str:
    return str(meta.get("artist_name", "")).lower().strip()


def meta_token_set(meta: dict) -> set[str]:
    parts = [
        str(meta.get("track_name", "")),
        str(meta.get("artist_name", "")),
        str(meta.get("album_name", "")),
    ]
    tags = meta.get("tag_list") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags[:12])
    else:
        parts.append(str(tags))
    return tokens(" ".join(parts))


def session_key(session_id: str) -> int:
    """Deterministic integer from session_id for splitting."""
    return _stable_hash(session_id)


def reconstruct_context(convs: list[dict], turn_number: int) -> dict:
    """Return {user_messages, played_track_ids} for turns strictly before target."""
    user_messages = []
    played = []
    for c in convs:
        t = int(c["turn_number"])
        if t >= turn_number:
            continue
        if c["role"] == "user":
            user_messages.append(str(c["content"]))
        elif c["role"] == "music":
            played.append(str(c["content"]).strip())
    return {
        "user_messages": user_messages,
        "played": played,
    }


FEATURE_NAMES = [
    "orig_rank",                      # 1/(rank)  — baseline signal from v23
    "same_artist_last_played",        # artist matches last music turn's track
    "tag_overlap_last_played",        # Jaccard of tags vs last-played
    "artist_token_overlap_query",     # artist tokens ∩ last user query tokens
    "title_token_overlap_query",      # title tokens ∩ last user query tokens
    "all_history_user_token_overlap", # meta tokens ∩ all user-query tokens
    "already_played",                 # 1 if candidate is in session's played set (penalty)
    "recency_weighted_metadata",      # decay-weighted sum of artist+tag overlap vs prior plays
]

INIT_WEIGHTS = {
    "orig_rank": 3.0,
    "same_artist_last_played": 1.0,
    "tag_overlap_last_played": 0.3,
    "artist_token_overlap_query": 0.5,
    "title_token_overlap_query": 0.5,
    "all_history_user_token_overlap": 0.1,
    "already_played": -2.0,
    "recency_weighted_metadata": 0.5,
}


def build_row_features(
    candidates: list[str],
    ctx: dict,
    item_db: MusicCatalogDB,
) -> np.ndarray:
    """Return shape (K, F) feature matrix for candidates in a single row."""
    user_messages = ctx["user_messages"]
    played = ctx["played"]

    now_tokens = tokens(user_messages[-1]) if user_messages else set()
    all_user_tokens = tokens(" ".join(user_messages)) if user_messages else set()
    played_set = set(played)

    # Last-played artist/tags
    last_artist = ""
    last_tags: set[str] = set()
    if played:
        try:
            last_meta = item_db.id_to_full_metadata(played[-1])
            last_artist = artist_str(last_meta)
            last_tags = tag_set(last_meta)
        except KeyError:
            pass

    # Precompute recency context for prior plays (most recent has highest weight).
    # weight = 1 / (pos_from_end + 1); oldest gets smallest weight.
    prior_plays_ctx = []
    for idx_from_end, tid in enumerate(reversed(played)):
        try:
            meta = item_db.id_to_full_metadata(tid)
        except KeyError:
            continue
        prior_plays_ctx.append({
            "weight": 1.0 / (idx_from_end + 1),
            "artist": artist_str(meta),
            "tags": tag_set(meta),
        })

    K = len(candidates)
    F = len(FEATURE_NAMES)
    X = np.zeros((K, F), dtype=np.float64)

    for rank, tid in enumerate(candidates, start=1):
        try:
            meta = item_db.id_to_full_metadata(tid)
        except KeyError:
            meta = {}
        cand_artist = artist_str(meta)
        cand_tags = tag_set(meta)
        cand_title_tokens = tokens(meta.get("track_name", ""))
        cand_artist_tokens = tokens(meta.get("artist_name", ""))
        cand_meta_tokens = meta_token_set(meta)

        f = {}
        f["orig_rank"] = 1.0 / rank
        f["same_artist_last_played"] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
        if last_tags or cand_tags:
            inter = len(cand_tags & last_tags)
            union = len(cand_tags | last_tags)
            f["tag_overlap_last_played"] = inter / union if union else 0.0
        else:
            f["tag_overlap_last_played"] = 0.0
        f["artist_token_overlap_query"] = float(len(cand_artist_tokens & now_tokens))
        f["title_token_overlap_query"] = float(len(cand_title_tokens & now_tokens))
        f["all_history_user_token_overlap"] = float(len(cand_meta_tokens & all_user_tokens))
        f["already_played"] = 1.0 if tid in played_set else 0.0

        rec = 0.0
        for p in prior_plays_ctx:
            artist_match = 1.0 if cand_artist and cand_artist == p["artist"] else 0.0
            if cand_tags or p["tags"]:
                inter = len(cand_tags & p["tags"])
                union = len(cand_tags | p["tags"])
                jacc = inter / union if union else 0.0
            else:
                jacc = 0.0
            rec += p["weight"] * (artist_match + jacc)
        f["recency_weighted_metadata"] = rec

        for i, name in enumerate(FEATURE_NAMES):
            X[rank - 1, i] = f[name]

    return X


def row_score_to_ndcg(X: np.ndarray, weights: np.ndarray, gt_idx: int | None, k: int = 20) -> float:
    """gt_idx = index of GT within candidate list, or None if GT not in list (=> 0 nDCG)."""
    if gt_idx is None:
        return 0.0
    scores = X @ weights
    # Stable sort; higher score first. Tie-break by original rank (feature 0 desc already implies lower rank first since 1/rank).
    # Use np.lexsort with secondary key = original position ascending.
    order = np.argsort(-scores, kind="stable")
    for i, idx in enumerate(order[:k]):
        if idx == gt_idx:
            return 1.0 / math.log2(i + 2)
    return 0.0


def mean_ndcg(feat_matrices: list[np.ndarray], gt_indices: list[int | None], weights: np.ndarray) -> float:
    if not feat_matrices:
        return 0.0
    total = 0.0
    for X, gt in zip(feat_matrices, gt_indices):
        total += row_score_to_ndcg(X, weights, gt)
    return total / len(feat_matrices)


def hit_rate_at_20(feat_matrices: list[np.ndarray], gt_indices: list[int | None], weights: np.ndarray) -> float:
    if not feat_matrices:
        return 0.0
    hits = 0
    for X, gt in zip(feat_matrices, gt_indices):
        if gt is None:
            continue
        scores = X @ weights
        order = np.argsort(-scores, kind="stable")
        if gt in order[:20].tolist():
            hits += 1
    return hits / len(feat_matrices)


def bootstrap_ndcg_ci(feat_matrices, gt_indices, weights, n_boot=1000, seed=0, k=20):
    per_row = np.array([
        row_score_to_ndcg(X, weights, gt, k=k)
        for X, gt in zip(feat_matrices, gt_indices)
    ])
    rng = np.random.default_rng(seed)
    n = len(per_row)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = per_row[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(per_row.mean()), float(lo), float(hi)


def fit_weights(feat_matrices, gt_indices, init: np.ndarray, method: str = "Powell") -> tuple[np.ndarray, float]:
    def neg_ndcg(w):
        return -mean_ndcg(feat_matrices, gt_indices, w)

    res = minimize(neg_ndcg, init, method=method, options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 2000})
    return res.x, -res.fun


def build_data(artifact_path: str) -> tuple[list[dict], Dataset, MusicCatalogDB]:
    with open(artifact_path) as f:
        rows = json.load(f)
    ds = Dataset.from_file(cached_test_arrow_path())

    print("Initializing MusicCatalogDB (track metadata)...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )
    return rows, ds, item_db


def assemble_features(rows, ds, item_db, use_candidate_pool: bool = False):
    """For each row build feature matrix and locate GT index.

    When use_candidate_pool is True, uses `candidate_pool_track_ids` (the
    wider pool) as the re-rankable candidate set. Raises a clear error if
    any row lacks that field. When False (default), uses
    `predicted_track_ids` (top-20 emission).
    """
    gt_maps = build_ground_truth(ds)
    # Build session→conversations lookup
    conv_by_sid = {item["session_id"]: item["conversations"] for item in ds}

    feat_matrices = []
    gt_indices: list[int | None] = []
    row_sessions: list[str] = []
    gt_hits = 0

    for r in rows:
        sid = str(r["session_id"])
        uid = r.get("user_id")
        turn = int(r["turn_number"])
        if use_candidate_pool:
            pool = r.get("candidate_pool_track_ids")
            if pool is None:
                raise KeyError(
                    f"row {sid}/{uid}/turn{turn} lacks candidate_pool_track_ids; "
                    "artifact was produced before pool-save instrumentation. "
                    "Re-run inference with --save_rerank_pool_k to populate it."
                )
            candidates = list(pool)
        else:
            candidates = list(r["predicted_track_ids"])

        # ground truth
        gt_id = None
        if uid is not None:
            gt_id = gt_maps["session_user"].get((sid, str(uid)), {}).get(turn)
        if gt_id is None:
            gt_id = gt_maps["session"].get(sid, {}).get(turn)

        ctx = reconstruct_context(conv_by_sid.get(sid, []), turn)
        X = build_row_features(candidates, ctx, item_db)
        gt_idx = candidates.index(gt_id) if gt_id in candidates else None
        if gt_idx is not None:
            gt_hits += 1

        feat_matrices.append(X)
        gt_indices.append(gt_idx)
        row_sessions.append(sid)

    return feat_matrices, gt_indices, row_sessions, gt_hits


def split_indices(row_sessions: list[str], seed: int, train_n: int, holdout_n: int):
    # Deterministic hash-based split by session_id (salted by seed).
    order = sorted(
        range(len(row_sessions)),
        key=lambda i: _stable_hash(f"{row_sessions[i]}:{seed}"),
    )
    return order[:train_n], order[train_n:train_n + holdout_n]


def cv_folds(row_sessions: list[str], k: int, seed: int):
    """k-fold deterministic split on session_id."""
    n = len(row_sessions)
    order = sorted(
        range(n),
        key=lambda i: _stable_hash(f"{row_sessions[i]}:{seed}"),
    )
    folds = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(idx)
    return folds


def ablation(feat_matrices_h, gt_indices_h, weights, holdout_ndcg):
    """Drop-one ablation: zero out each feature's weight and recompute holdout nDCG."""
    results = []
    for i, name in enumerate(FEATURE_NAMES):
        w_ablate = weights.copy()
        w_ablate[i] = 0.0
        ndcg_ab = mean_ndcg(feat_matrices_h, gt_indices_h, w_ablate)
        results.append({
            "feature": name,
            "holdout_ndcg": ndcg_ab,
            "delta": ndcg_ab - holdout_ndcg,
        })
    return results


def gate_verdict(holdout_ndcg: float) -> tuple[str, str]:
    if holdout_ndcg < 0.14:
        return ("INSUFFICIENT", "ranking heuristics alone won't clear the gate; deterministic postrank topped out well below 0.18.")
    if holdout_ndcg < 0.18:
        return ("PROMISING", "refine features then retest on a larger v23 run or paired sample; do NOT scale yet.")
    return ("PASSES GATE", "clears 0.18 threshold on held-out; verify with a second split/seed before scaling.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="exp/inference/devset/echo_v23_devset.json")
    ap.add_argument("--split", default="100_100", help="<train>_<holdout>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", type=int, default=0, help="k-fold CV (overrides --split when >0)")
    ap.add_argument("--out_dir", default="exp/eval")
    ap.add_argument("--method", default="Powell")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--use_candidate_pool", action="store_true",
                    help="Use candidate_pool_track_ids as the re-rankable pool "
                         "(instead of predicted_track_ids). Requires the artifact "
                         "to include the pool field; errors clearly otherwise.")
    args = ap.parse_args()

    rows, ds, item_db = build_data(args.artifact)
    print(f"Loaded {len(rows)} rows from {args.artifact}")
    # Confirm pool depth
    pool_field = "candidate_pool_track_ids" if args.use_candidate_pool else "predicted_track_ids"
    try:
        pool_depths = {len(r[pool_field]) for r in rows}
    except KeyError as e:
        raise SystemExit(
            f"ERROR: --use_candidate_pool set but artifact lacks '{pool_field}': {e}. "
            "Re-run inference with pool-save instrumentation to populate it."
        ) from e
    print(f"Candidate pool depth(s) [{pool_field}]: {sorted(pool_depths)}")

    print("Building features for all rows...", flush=True)
    feat_matrices, gt_indices, row_sessions, gt_hits = assemble_features(
        rows, ds, item_db, use_candidate_pool=args.use_candidate_pool,
    )
    rate = gt_hits / len(rows) if rows else 0.0
    print(f"GT-in-top20 rate (pre-tune): {gt_hits}/{len(rows)} = {rate:.4f}")

    # Baseline nDCG from v23 (orig order) = nDCG using only w=[orig_rank=1, others=0]
    baseline_w = np.zeros(len(FEATURE_NAMES))
    baseline_w[FEATURE_NAMES.index("orig_rank")] = 1.0
    baseline_ndcg_all = mean_ndcg(feat_matrices, gt_indices, baseline_w)
    print(f"Baseline (v23 orig order) nDCG@20 on all 200 rows: {baseline_ndcg_all:.4f}")

    init_w = np.array([INIT_WEIGHTS[n] for n in FEATURE_NAMES], dtype=np.float64)

    summary: dict[str, Any] = {
        "artifact": args.artifact,
        "n_rows": len(rows),
        "pool_depth": sorted(pool_depths),
        "features": FEATURE_NAMES,
        "baseline_orig_order_ndcg": baseline_ndcg_all,
        "gt_in_top20_count": gt_hits,
        "seed": args.seed,
    }

    if args.cv and args.cv > 1:
        k = args.cv
        folds = cv_folds(row_sessions, k, args.seed)
        fold_results = []
        for i in range(k):
            holdout_idx = set(folds[i])
            train_idx = [j for j in range(len(rows)) if j not in holdout_idx]
            hold_idx = list(folds[i])
            X_tr = [feat_matrices[j] for j in train_idx]
            g_tr = [gt_indices[j] for j in train_idx]
            X_ho = [feat_matrices[j] for j in hold_idx]
            g_ho = [gt_indices[j] for j in hold_idx]

            w_opt, tr_ndcg = fit_weights(X_tr, g_tr, init_w, method=args.method)
            ho_ndcg = mean_ndcg(X_ho, g_ho, w_opt)
            ho_hit = hit_rate_at_20(X_ho, g_ho, w_opt)
            fold_results.append({
                "fold": i,
                "n_train": len(X_tr),
                "n_holdout": len(X_ho),
                "train_ndcg": tr_ndcg,
                "holdout_ndcg": ho_ndcg,
                "holdout_hit_rate20": ho_hit,
                "weights": dict(zip(FEATURE_NAMES, w_opt.tolist())),
            })
            print(f"Fold {i}: train={tr_ndcg:.4f}  holdout={ho_ndcg:.4f}  hit20={ho_hit:.4f}")
        cv_mean_ndcg = float(np.mean([f["holdout_ndcg"] for f in fold_results]))
        cv_mean_hit = float(np.mean([f["holdout_hit_rate20"] for f in fold_results]))
        cv_std_ndcg = float(np.std([f["holdout_ndcg"] for f in fold_results], ddof=1))
        summary["mode"] = f"cv{k}"
        summary["folds"] = fold_results
        summary["cv_mean_holdout_ndcg"] = cv_mean_ndcg
        summary["cv_std_holdout_ndcg"] = cv_std_ndcg
        summary["cv_mean_holdout_hit20"] = cv_mean_hit
        verdict, reason = gate_verdict(cv_mean_ndcg)
        summary["gate_verdict"] = verdict
        summary["gate_reason"] = reason

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = f"{args.out_dir}/postrank_tune_v23_cv{k}_seed{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCV mean holdout nDCG@20: {cv_mean_ndcg:.4f} (±{cv_std_ndcg:.4f})")
        print(f"CV mean holdout hit@20: {cv_mean_hit:.4f}")
        print(f"Verdict: {verdict} — {reason}")
        print(f"Saved: {out_path}")
        return

    # Regular train/holdout split
    try:
        train_n, hold_n = [int(x) for x in args.split.split("_")]
    except Exception:
        train_n, hold_n = 100, 100
    train_idx, hold_idx = split_indices(row_sessions, args.seed, train_n, hold_n)

    X_tr = [feat_matrices[i] for i in train_idx]
    g_tr = [gt_indices[i] for i in train_idx]
    X_ho = [feat_matrices[i] for i in hold_idx]
    g_ho = [gt_indices[i] for i in hold_idx]
    print(f"Train: {len(X_tr)}  Holdout: {len(X_ho)}")

    # Fit
    w_opt, tr_ndcg = fit_weights(X_tr, g_tr, init_w, method=args.method)
    ho_ndcg = mean_ndcg(X_ho, g_ho, w_opt)
    ho_hit = hit_rate_at_20(X_ho, g_ho, w_opt)
    ho_boot_mean, ho_lo, ho_hi = bootstrap_ndcg_ci(X_ho, g_ho, w_opt, n_boot=args.n_boot, seed=args.seed)
    abl = ablation(X_ho, g_ho, w_opt, ho_ndcg)

    # Baseline on holdout for reference
    baseline_ho = mean_ndcg(X_ho, g_ho, baseline_w)

    summary.update({
        "mode": "split",
        "train_n": len(X_tr),
        "holdout_n": len(X_ho),
        "train_ndcg": tr_ndcg,
        "holdout_ndcg": ho_ndcg,
        "holdout_hit_rate20": ho_hit,
        "holdout_ndcg_boot_mean": ho_boot_mean,
        "holdout_ndcg_ci95": [ho_lo, ho_hi],
        "holdout_baseline_orig_order_ndcg": baseline_ho,
        "weights": dict(zip(FEATURE_NAMES, w_opt.tolist())),
        "ablation": abl,
    })
    verdict, reason = gate_verdict(ho_ndcg)
    summary["gate_verdict"] = verdict
    summary["gate_reason"] = reason

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = f"{args.out_dir}/postrank_tune_v23_{args.split}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Train nDCG@20   : {tr_ndcg:.4f}")
    print(f"Holdout nDCG@20 : {ho_ndcg:.4f}   95% CI: [{ho_lo:.4f}, {ho_hi:.4f}]")
    print(f"Holdout hit@20  : {ho_hit:.4f}")
    print(f"Baseline (orig) : holdout {baseline_ho:.4f}")
    print("-" * 60)
    print("Weights:")
    for name, w in zip(FEATURE_NAMES, w_opt):
        print(f"  {name:40s}  {w:+.4f}")
    print("-" * 60)
    print("Ablation (drop one, holdout):")
    for a in abl:
        print(f"  -{a['feature']:38s}  {a['holdout_ndcg']:.4f}  Δ {a['delta']:+.4f}")
    print("-" * 60)
    print(f"Verdict: {verdict} — {reason}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
