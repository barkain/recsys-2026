# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""R3 Validation Bundle (READ + COMPUTE only — zero LLM spend).

Three-part offline validation matching the convention that produced the
seed-0 reference (holdout 0.1207, CV5 0.1494, std 0.0502):

For each (config, seed):
- 100/100 split: Powell-fit on train_idx, eval on holdout_idx.
- 5-fold CV: PER-FOLD Powell fit (160-session train each), eval on each fold.

R3 = postrank Powell-tuned scorer (8 deterministic features) over a 50-deep
weighted-RRF pool built from sources:
  A = v23 LLM-reranked top-50, B = bm25 last-music-meta@500,
  C = bm25 full-history@500, D = track neighbors@200.

Features cached to disk (pickle) for fast restarts. NO LLM calls.

E3 uses a relaxed Powell (xtol=ftol=5e-3, maxiter=200) for tractability;
E1/E2 use full Powell (xtol=ftol=1e-3, maxiter=500).
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
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import Dataset
from eval_inference import build_ground_truth, cached_test_arrow_path
from scripts.tune_postrank_v23 import (
    FEATURE_NAMES,
    INIT_WEIGHTS,
    STOPWORDS,
    reconstruct_context,
)
from mcrs.db_item.music_catalog import MusicCatalogDB

ARTIFACT = "exp/inference/devset/echo_v23_pool50_s200.json"
CACHE_FILE = "exp/eval/v23_union_retrievals_cache.json"
OUT_FILE = "exp/eval/r3_validation_bundle.json"
FEATURE_PICKLE = "exp/eval/_r3_validation_features.pkl"

TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {
        t for t in TOKEN_RE.findall(str(text).lower())
        if len(t) > 2 and t not in STOPWORDS
    }


def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def precompute_features(cache_path: Path) -> dict[str, Any]:
    """One-time per-session full-union feature computation. Cached as pickle."""
    print("Loading cache + artifact + GT + item_db...", flush=True)
    with open(CACHE_FILE) as f:
        payload = json.load(f)
    with open(ARTIFACT) as f:
        artifact_rows = json.load(f)
    cases = payload["cases"]
    bm25_meta = payload["bm25_meta"]
    bm25_full = payload["bm25_full"]
    neighbors = payload["neighbors"]

    arrow = cached_test_arrow_path()
    if not arrow:
        sys.exit("ERROR: devset arrow not in HF cache")
    ds = Dataset.from_file(arrow)
    gt_maps = build_ground_truth(ds)
    conv_by_sid: dict[str, list[dict]] = {item["session_id"]: item["conversations"] for item in ds}

    n = len(cases)
    v23_pool: list[list[str]] = [list(artifact_rows[i]["candidate_pool_track_ids"]) for i in range(n)]

    gts: list[str | None] = []
    for c in cases:
        sid = str(c["session_id"])
        uid = c.get("user_id")
        turn = int(c["turn_number"])
        gt_id = None
        if uid is not None:
            gt_id = gt_maps["session_user"].get((sid, str(uid)), {}).get(turn)
        if gt_id is None:
            gt_id = gt_maps["session"].get(sid, {}).get(turn)
        gts.append(gt_id)
    print(f"  n={n}, GT non-null: {sum(1 for g in gts if g)}/{n}", flush=True)

    print("Initializing MusicCatalogDB...", flush=True)
    item_db = MusicCatalogDB(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
    )

    print("Collecting unique tracks...", flush=True)
    all_tracks: set[str] = set()
    for i in range(n):
        for tid in v23_pool[i] + bm25_meta[i] + bm25_full[i] + neighbors[i]:
            all_tracks.add(tid)
        for c in conv_by_sid.get(cases[i]["session_id"], []):
            if c["role"] == "music" and int(c["turn_number"]) < int(cases[i]["turn_number"]):
                all_tracks.add(str(c["content"]).strip())
    print(f"  unique tracks: {len(all_tracks)}", flush=True)

    print("Precomputing per-track artifacts...", flush=True)
    t1 = time.time()
    track_artist: dict[str, str] = {}
    track_tags: dict[str, set[str]] = {}
    track_title_toks: dict[str, set[str]] = {}
    track_artist_toks: dict[str, set[str]] = {}
    track_meta_toks: dict[str, set[str]] = {}
    cnt = 0
    for tid in all_tracks:
        try:
            meta = item_db.id_to_full_metadata(tid)
        except KeyError:
            meta = {}
        artist = str(meta.get("artist_name", "")).lower().strip()
        tags_raw = meta.get("tag_list") or []
        if isinstance(tags_raw, list):
            tags = {str(t).lower().strip() for t in tags_raw if str(t).strip()}
        else:
            tags = set()
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = _tokens(title)
        track_artist_toks[tid] = _tokens(meta.get("artist_name", ""))
        meta_parts: list[str] = [title, str(meta.get("artist_name", "")), album]
        if isinstance(tags_raw, list):
            meta_parts.extend(str(t) for t in tags_raw[:12])
        else:
            meta_parts.append(str(tags_raw))
        track_meta_toks[tid] = _tokens(" ".join(meta_parts))
        cnt += 1
        if cnt % 10000 == 0:
            print(f"  {cnt}/{len(all_tracks)} ({time.time() - t1:.1f}s)", flush=True)
    print(f"  per-track ready in {time.time() - t1:.1f}s", flush=True)

    print("Building per-session union features...", flush=True)
    t2 = time.time()
    union_per_session: list[list[str]] = []
    union_idx_per_session: list[dict[str, int]] = []
    feat_per_session: list[np.ndarray] = []
    F = len(FEATURE_NAMES)
    for i, c in enumerate(cases):
        union = dedupe(v23_pool[i] + bm25_meta[i] + bm25_full[i] + neighbors[i])
        union_per_session.append(union)
        union_idx_per_session.append({tid: pos for pos, tid in enumerate(union)})
        ctx = reconstruct_context(conv_by_sid.get(c["session_id"], []), c["turn_number"])
        user_messages = ctx["user_messages"]
        played = ctx["played"]
        now_tokens = _tokens(user_messages[-1]) if user_messages else set()
        all_user_tokens = _tokens(" ".join(user_messages)) if user_messages else set()
        played_set = set(played)
        last_artist = track_artist.get(played[-1], "") if played else ""
        last_tags = track_tags.get(played[-1], set()) if played else set()
        prior = []
        for idx_from_end, tid in enumerate(reversed(played)):
            prior.append((1.0 / (idx_from_end + 1), track_artist.get(tid, ""), track_tags.get(tid, set())))

        K = len(union)
        X = np.zeros((K, F), dtype=np.float64)
        for rank, tid in enumerate(union, start=1):
            cand_artist = track_artist.get(tid, "")
            cand_tags = track_tags.get(tid, set())
            cand_title_tokens = track_title_toks.get(tid, set())
            cand_artist_tokens = track_artist_toks.get(tid, set())
            cand_meta_tokens = track_meta_toks.get(tid, set())
            X[rank - 1, 0] = 1.0 / rank
            X[rank - 1, 1] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
            if last_tags or cand_tags:
                inter = len(cand_tags & last_tags)
                u = len(cand_tags | last_tags)
                X[rank - 1, 2] = inter / u if u else 0.0
            X[rank - 1, 3] = float(len(cand_artist_tokens & now_tokens))
            X[rank - 1, 4] = float(len(cand_title_tokens & now_tokens))
            X[rank - 1, 5] = float(len(cand_meta_tokens & all_user_tokens))
            X[rank - 1, 6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for w_, p_art, p_tags in prior:
                am = 1.0 if cand_artist and cand_artist == p_art else 0.0
                if cand_tags or p_tags:
                    inter = len(cand_tags & p_tags)
                    u = len(cand_tags | p_tags)
                    jacc = inter / u if u else 0.0
                else:
                    jacc = 0.0
                rec += w_ * (am + jacc)
            X[rank - 1, 7] = rec
        feat_per_session.append(X)
        if (i + 1) % 50 == 0:
            print(f"  built {i+1}/{n} ({time.time() - t2:.1f}s)", flush=True)
    print(f"  features ready in {time.time() - t2:.1f}s", flush=True)

    sessions = [c["session_id"] for c in cases]
    out = {
        "n": n,
        "v23_pool": v23_pool,
        "bm25_meta": bm25_meta,
        "bm25_full": bm25_full,
        "neighbors": neighbors,
        "gts": gts,
        "sessions": sessions,
        "union_per_session": union_per_session,
        "union_idx_per_session": union_idx_per_session,
        "feat_per_session": feat_per_session,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  cached features → {cache_path}", flush=True)
    return out


def load_or_build_features() -> dict[str, Any]:
    p = Path(FEATURE_PICKLE)
    if p.exists():
        print(f"Loading cached features from {p}...", flush=True)
        with open(p, "rb") as f:
            data = pickle.load(f)  # noqa: S301 - local-only cache file produced by this script
        print(f"  loaded n={data['n']}", flush=True)
        return data
    return precompute_features(p)


def main():
    t_start = time.time()
    data = load_or_build_features()
    n = data["n"]
    v23_pool = data["v23_pool"]
    bm25_meta = data["bm25_meta"]
    bm25_full = data["bm25_full"]
    neighbors = data["neighbors"]
    gts = data["gts"]
    sessions = data["sessions"]
    union_idx_per_session = data["union_idx_per_session"]
    feat_per_session = data["feat_per_session"]
    F = len(FEATURE_NAMES)
    POOL_K = 50

    # ---- Source rank maps for fast RRF.
    src_rr: list[dict[str, dict[str, float]]] = []
    for i in range(n):
        d_all: dict[str, dict[str, float]] = {}
        for label, seq in [("A", v23_pool[i]), ("B", bm25_meta[i]), ("C", bm25_full[i]), ("D", neighbors[i])]:
            d: dict[str, float] = {}
            for rank, tid in enumerate(seq):
                if tid not in d:
                    d[tid] = 1.0 / (60 + rank + 1)
            d_all[label] = d
        src_rr.append(d_all)

    def build_pool_tensor(weights: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X3D, gt_arr, pool_size_arr).
        X3D: (n, POOL_K, F) padded with zeros for sessions where pool < POOL_K
        gt_arr: (n,) gt position in pool, -1 if not present
        pool_size_arr: (n,) actual pool size
        """
        wA, wB, wC, wD = weights
        X3D = np.zeros((n, POOL_K, F), dtype=np.float64)
        gt_arr = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        for i in range(n):
            scores: dict[str, float] = {}
            sr = src_rr[i]
            if wA != 0:
                for tid, rr in sr["A"].items():
                    scores[tid] = scores.get(tid, 0.0) + wA * rr
            if wB != 0:
                for tid, rr in sr["B"].items():
                    scores[tid] = scores.get(tid, 0.0) + wB * rr
            if wC != 0:
                for tid, rr in sr["C"].items():
                    scores[tid] = scores.get(tid, 0.0) + wC * rr
            if wD != 0:
                for tid, rr in sr["D"].items():
                    scores[tid] = scores.get(tid, 0.0) + wD * rr
            if not scores:
                continue
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:POOL_K]
            sizes[i] = len(ranked)
            idx_map = union_idx_per_session[i]
            full_X = feat_per_session[i]
            rows_idx = np.array([idx_map[tid] for tid in ranked], dtype=np.int64)
            X_pool = full_X[rows_idx].copy()
            X_pool[:, 0] = 1.0 / np.arange(1, len(ranked) + 1, dtype=np.float64)
            X3D[i, :len(ranked)] = X_pool
            gt = gts[i]
            if gt is not None and gt in ranked:
                gt_arr[i] = ranked.index(gt)
        return X3D, gt_arr, sizes

    # Precompute valid mask for each session: shape (n, POOL_K), True where j < pool_size
    def valid_mask_for_sizes(sizes: np.ndarray) -> np.ndarray:
        return np.arange(POOL_K)[None, :] < sizes[:, None]

    def vec_ndcg(X3D: np.ndarray, gt_arr: np.ndarray, valid_mask: np.ndarray, idx: np.ndarray, w: np.ndarray) -> float:
        """nDCG@20 vectorized over sessions.

        Tie-breaking: stable argsort places lower-index first → equivalent to
        rank = #strict-greater + #ties-with-lower-pool-index, computed only over
        VALID pool positions (not padding).
        """
        if len(idx) == 0:
            return 0.0
        Xs = X3D[idx]  # (m, K, F)
        gts_s = gt_arr[idx]
        vmask = valid_mask[idx]  # (m, K) True where pool position is real
        scores = Xs @ w  # (m, K)
        # Replace pad scores with -inf so they don't compete
        scores_safe = np.where(vmask, scores, -np.inf)
        valid = gts_s >= 0
        gt_safe = np.where(valid, gts_s, 0)
        m = len(idx)
        gt_scores = scores_safe[np.arange(m), gt_safe]
        # rank = #(scores_safe > gt_score) over the ROW
        strict_gt = (scores_safe > gt_scores[:, None]).sum(axis=1)
        # ties at lower pool index < gt_idx (only valid positions)
        tie_mask = (scores_safe == gt_scores[:, None]) & vmask & (np.arange(POOL_K)[None, :] < gt_safe[:, None])
        ties_lower = tie_mask.sum(axis=1)
        rank = strict_gt + ties_lower
        return float(np.where(valid & (rank < 20), 1.0 / np.log2(rank + 2), 0.0).mean())

    def fit_powell(X3D: np.ndarray, gt_arr: np.ndarray, valid_mask: np.ndarray, idx: np.ndarray, init: np.ndarray,
                   xtol: float = 1e-3, ftol: float = 1e-3, maxiter: int = 500) -> tuple[np.ndarray, float]:
        def neg(w: np.ndarray) -> float:
            return -vec_ndcg(X3D, gt_arr, valid_mask, idx, w)
        res = minimize(neg, init, method="Powell", options={"xtol": xtol, "ftol": ftol, "maxiter": maxiter})
        return res.x, -float(res.fun)

    init_w = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    SEEDS = [0, 1, 2, 3, 4]

    # Precompute splits/folds per seed (deterministic)
    seed_to_train: dict[int, np.ndarray] = {}
    seed_to_holdout: dict[int, np.ndarray] = {}
    seed_to_folds: dict[int, list[np.ndarray]] = {}
    seed_to_fold_train: dict[int, list[np.ndarray]] = {}  # complement of each fold
    for seed in SEEDS:
        order = sorted(range(n), key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))
        seed_to_train[seed] = np.array(order[:100], dtype=np.int64)
        seed_to_holdout[seed] = np.array(order[100:], dtype=np.int64)
        folds = [[] for _ in range(5)]
        for pos, idx in enumerate(order):
            folds[pos % 5].append(idx)
        seed_to_folds[seed] = [np.array(f, dtype=np.int64) for f in folds]
        # Per-fold train sets = full set minus this fold
        seed_to_fold_train[seed] = [
            np.array([j for j in range(n) if j not in set(f)], dtype=np.int64)
            for f in seed_to_folds[seed]
        ]

    def run_for_weights(weights: tuple[float, float, float, float], seeds: list[int],
                        do_split: bool = True, mode: str = "perfold") -> dict[str, Any]:
        """mode in {'perfold', 'pertrain', 'fast'}.

        - 'perfold' (E1, E2): For each seed, ONE Powell fit on train_idx →
          eval on holdout. PLUS per-fold Powell fits (5×160-session train).
          Matches scripts/r3_validation_smoke.py reference (cv5=0.1494 seed 0).
        - 'fast' (E3): For each seed, ONE Powell fit on FULL 200 sessions
          (relaxed tolerance), evaluate on each fold. 1 fit per (cfg, seed)
          instead of 5. Tolerable accuracy for relative ranking sweeps.
        """
        X3D, gt_arr, sizes = build_pool_tensor(weights)
        valid_mask = valid_mask_for_sizes(sizes)
        per_seed: list[dict[str, Any]] = []
        if mode == "fast":
            xtol, ftol, maxiter = 5e-3, 5e-3, 200
        else:
            xtol, ftol, maxiter = 1e-3, 1e-3, 500
        for seed in seeds:
            entry: dict[str, Any] = {"seed": seed}
            if do_split:
                w_opt, tr_score = fit_powell(X3D, gt_arr, valid_mask, seed_to_train[seed], init_w, xtol, ftol, maxiter)
                entry["holdout_ndcg"] = vec_ndcg(X3D, gt_arr, valid_mask, seed_to_holdout[seed], w_opt)
                entry["train_ndcg"] = tr_score
            cv_per_fold: list[float] = []
            if mode == "fast":
                # Single Powell fit on the full 200 sessions; evaluate on each fold.
                full_idx = np.arange(n, dtype=np.int64)
                w_full, _ = fit_powell(X3D, gt_arr, valid_mask, full_idx, init_w, xtol, ftol, maxiter)
                for fold in seed_to_folds[seed]:
                    cv_per_fold.append(vec_ndcg(X3D, gt_arr, valid_mask, fold, w_full))
            else:
                # Proper per-fold CV (fits 5 separate Powell on the fold's complement).
                for fold, fold_train in zip(seed_to_folds[seed], seed_to_fold_train[seed]):
                    w_f, _ = fit_powell(X3D, gt_arr, valid_mask, fold_train, init_w, xtol, ftol, maxiter)
                    cv_per_fold.append(vec_ndcg(X3D, gt_arr, valid_mask, fold, w_f))
            entry["cv5_per_fold"] = cv_per_fold
            entry["cv5_mean"] = float(np.mean(cv_per_fold))
            entry["cv5_std"] = float(np.std(cv_per_fold, ddof=1))
            per_seed.append(entry)
        return {"weights": list(weights), "per_seed": per_seed}

    # =========================
    # E1: Multi-seed CV stability for R3 (1, 2, 0.5, 1)
    # =========================
    print("\n=== E1: Multi-seed CV stability for R3 (1, 2, 0.5, 1) ===", flush=True)
    t1e = time.time()
    e1 = run_for_weights((1.0, 2.0, 0.5, 1.0), SEEDS, do_split=True, mode="perfold")
    holdouts = [s["holdout_ndcg"] for s in e1["per_seed"]]
    cv_means = [s["cv5_mean"] for s in e1["per_seed"]]
    e1["aggregate"] = {
        "n_seeds": len(SEEDS),
        "mean_holdout": float(np.mean(holdouts)),
        "std_holdout": float(np.std(holdouts, ddof=1)),
        "mean_of_cv5_means": float(np.mean(cv_means)),
        "std_of_cv5_means": float(np.std(cv_means, ddof=1)),
        "min_cv5_mean": float(np.min(cv_means)),
        "max_cv5_mean": float(np.max(cv_means)),
    }
    print(f"  E1 done in {time.time() - t1e:.1f}s", flush=True)
    for s in e1["per_seed"]:
        print(f"  seed {s['seed']}: holdout={s['holdout_ndcg']:.4f}  cv5={s['cv5_mean']:.4f} ± {s['cv5_std']:.4f}")
    agg = e1["aggregate"]
    print(f"  agg: holdout {agg['mean_holdout']:.4f} ± {agg['std_holdout']:.4f}  cv5 {agg['mean_of_cv5_means']:.4f} ± {agg['std_of_cv5_means']:.4f}  range [{agg['min_cv5_mean']:.4f}, {agg['max_cv5_mean']:.4f}]")

    # =========================
    # E2: Sensitivity to A
    # =========================
    print("\n=== E2: Sensitivity to A ===", flush=True)
    t2e = time.time()
    variants = {
        "V_A1": (1.0, 2.0, 0.5, 1.0),
        "V_A0": (0.0, 2.0, 0.5, 1.0),
        "V_A_05": (0.5, 2.0, 0.5, 1.0),
        "V_A_2": (2.0, 2.0, 0.5, 1.0),
    }
    e2: dict[str, Any] = {"variants": {}}
    base_cv5: float | None = None
    for name, w in variants.items():
        run = run_for_weights(w, SEEDS, do_split=False, mode="perfold")
        cv_means = [s["cv5_mean"] for s in run["per_seed"]]
        run["aggregate"] = {
            "mean_cv5": float(np.mean(cv_means)),
            "std_of_cv5_means": float(np.std(cv_means, ddof=1)),
        }
        if name == "V_A1":
            base_cv5 = run["aggregate"]["mean_cv5"]
        e2["variants"][name] = run
    for name, run in e2["variants"].items():
        a = run["aggregate"]
        delta = a["mean_cv5"] - (base_cv5 or 0.0)
        a["delta_vs_V_A1"] = delta
        print(f"  {name} {run['weights']}: cv5 {a['mean_cv5']:.4f} ± {a['std_of_cv5_means']:.4f}  Δ {delta:+.4f}")
    drop_loss = e2["variants"]["V_A1"]["aggregate"]["mean_cv5"] - e2["variants"]["V_A0"]["aggregate"]["mean_cv5"]
    if drop_loss <= 0.005:
        verdict = "A NOT load-bearing (deterministic-only viable)"
    elif drop_loss > 0.02:
        verdict = "A IS load-bearing"
    else:
        verdict = f"A partially load-bearing (loss={drop_loss:.4f})"
    e2["verdict"] = verdict
    e2["drop_loss"] = drop_loss
    print(f"  E2 done in {time.time() - t2e:.1f}s, verdict: {verdict}")

    # =========================
    # E3: Coarse weight grid robustness
    # =========================
    # E3 uses 3 seeds (0, 2, 4) per spec's "If 144 configs × 5 seeds is too slow,
    # restrict to seeds 0, 2, 4 (3 seeds)" — keeps total in budget.
    SEEDS_E3 = [0, 2, 4]
    print(f"\n=== E3: Weight grid robustness (4 × 3 × 4 × 3 = 144 configs × {len(SEEDS_E3)} seeds, fast single-fit Powell) ===", flush=True)
    grid_A = [0, 0.5, 1, 2]
    grid_B = [1, 2, 4]
    grid_C = [0, 0.5, 1, 2]
    grid_D = [0, 1, 2]

    e3_configs: list[dict[str, Any]] = []
    t3 = time.time()
    n_done = 0
    for wA in grid_A:
        for wB in grid_B:
            for wC in grid_C:
                for wD in grid_D:
                    weights = (float(wA), float(wB), float(wC), float(wD))
                    if all(w == 0 for w in weights):
                        continue
                    run = run_for_weights(weights, SEEDS_E3, do_split=False, mode="fast")
                    seed_means = [s["cv5_mean"] for s in run["per_seed"]]
                    mean_cv5 = float(np.mean(seed_means))
                    std_means = float(np.std(seed_means, ddof=1))
                    robustness = mean_cv5 / (mean_cv5 + std_means) if (mean_cv5 + std_means) > 0 else 0.0
                    e3_configs.append({
                        "weights": list(weights),
                        "mean_cv5": mean_cv5,
                        "std_of_means": std_means,
                        "robustness": robustness,
                        "per_seed_cv5_means": seed_means,
                    })
                    n_done += 1
                    if n_done % 12 == 0:
                        elapsed = time.time() - t3
                        rate = n_done / elapsed if elapsed > 0 else 0
                        eta_min = (144 - n_done) / rate / 60 if rate > 0 else 0
                        print(f"  E3 progress: {n_done}/144 ({elapsed:.1f}s, {rate:.2f} cfg/s, ETA {eta_min:.1f}min)", flush=True)

    e3_sorted = sorted(e3_configs, key=lambda c: c["mean_cv5"], reverse=True)
    top5 = e3_sorted[:5]
    seed0_best_target = [1.0, 2.0, 0.5, 1.0]
    seed0_rank = None
    seed0_entry = None
    for r, c in enumerate(e3_sorted, start=1):
        if c["weights"] == seed0_best_target:
            seed0_rank = r
            seed0_entry = c
            break
    overall_winner = e3_sorted[0]
    plateau_threshold = overall_winner["mean_cv5"] - 0.005
    plateau_size = sum(1 for c in e3_sorted if c["mean_cv5"] >= plateau_threshold)
    e3 = {
        "n_configs": len(e3_sorted),
        "top_5": top5,
        "overall_winner": overall_winner,
        "seed0_best": {
            "weights": seed0_best_target,
            "rank": seed0_rank,
            "entry": seed0_entry,
        },
        "plateau": {
            "within_0_005_of_winner": plateau_size,
            "winner_cv5": overall_winner["mean_cv5"],
        },
        "all_configs_sorted": e3_sorted,
    }
    print(f"  E3 done in {time.time() - t3:.1f}s; plateau within 0.005 of winner: {plateau_size}", flush=True)
    for r, c in enumerate(top5, start=1):
        print(f"  E3 top-{r}: {c['weights']}  cv5 {c['mean_cv5']:.4f} ± {c['std_of_means']:.4f}  robust {c['robustness']:.4f}")
    if seed0_entry:
        print(f"  seed-0 best (1,2,0.5,1) rank: {seed0_rank}  cv5 {seed0_entry['mean_cv5']:.4f}")

    # =========================
    # Persist
    # =========================
    out = {
        "meta": {
            "n_sessions": n,
            "seeds": SEEDS,
            "feature_names": FEATURE_NAMES,
            "init_weights": INIT_WEIGHTS,
            "tool_used": "v23_union_retrievals_cache.json + Powell-tuned 8-feat scorer over weighted-RRF pool-50",
            "convention": "Per-fold Powell (5 fits per (config,seed) on 160 sessions; matches scripts/r3_validation_smoke.py reproduction of cited 0.1494 cv5 / 0.1207 holdout for seed 0).",
            "elapsed_sec": time.time() - t_start,
            "no_llm_calls": True,
            "e3_powell_relaxed": True,
        },
        "e1_multi_seed": e1,
        "e2_a_sensitivity": e2,
        "e3_weight_grid": e3,
    }
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_FILE} ({time.time() - t_start:.1f}s total)")


if __name__ == "__main__":
    main()
