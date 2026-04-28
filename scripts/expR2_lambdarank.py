# ruff: noqa: T201
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Experiment R2 — LightGBM LambdaRank postranker on the cfg0209 pool.

ZERO API calls. Reuses cached features (_expB_features.pkl, cfg0209 pool) plus
BGE retrievals (_expR1_E_retrievals.pkl) and the BGE catalog index. Adds 4 new
BGE features per (session, candidate), assembles into _expR2_features.pkl, then
runs 5-fold grouped CV × 5 seeds for:

  B0  cfg0209 raw rank (no postrank)
  B1  Powell linear (refit per fold, on full 34 + new features) — sanity
  B2  LightGBM pointwise (regression on is_gt 0/1)
  B3  LightGBM LambdaRank (the actual target model)
  B4  LightGBM LambdaRank with `bge_*` ablation (drop newly-added group)

Reports mean_cv5 ± std-of-means per model, top-15 importances for B3, and
error-case analysis (top wins/losses vs B1).
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
EXPB_FEATURES = REPO_ROOT / "exp" / "eval" / "_expB_features.pkl"
EXPR1_E_RETRIEVALS = REPO_ROOT / "exp" / "eval" / "_expR1_E_retrievals.pkl"
PAYLOAD_PKL = REPO_ROOT / "exp" / "eval" / "_r3_confirm_400_deterministic.pkl"
APRIME50_PKL = REPO_ROOT / "exp" / "eval" / "_r3_a_prime_400.pkl"

QWEN_VECS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy"
QWEN_IDS = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"
BGE_VECS = REPO_ROOT / "cache" / "dense" / "bge-base-en-v1.5" / "embeddings.npy"
BGE_IDS = REPO_ROOT / "cache" / "dense" / "bge-base-en-v1.5" / "track_ids.json"

OUT_FEATURES = REPO_ROOT / "exp" / "eval" / "_expR2_features.pkl"
OUT_RESULTS = REPO_ROOT / "exp" / "eval" / "expR2_lambdarank_results.json"
OUT_ERRORS = REPO_ROOT / "exp" / "eval" / "expR2_error_cases.json"

CFG0209_DEPTHS = {"A": 200, "B": 500, "C": 500, "D": 200}
CFG0209_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
RRF_K = 20
POOL_K = 50
NDCG_K = 20
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5

# If False, skip the qwen3 pool rebuild and BGE feature add. Use only the
# 34 features from _expB_features.pkl. Fastest path to B3 vs B1 signal.
ADD_BGE_FEATURES = False

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

POWELL_INIT = {
    "orig_rank": 3.0,
    "same_artist_last_played": 1.0,
    "tag_overlap_last_played": 0.3,
    "artist_token_overlap_query": 0.5,
    "title_token_overlap_query": 0.5,
    "all_history_user_token_overlap": 0.1,
    "already_played": -2.0,
    "recency_weighted_metadata": 0.5,
}


# ---------------------------------------------------------------------------
def stable_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def split_order(sessions: list[str], seed: int) -> list[int]:
    return sorted(range(len(sessions)),
                  key=lambda i: stable_hash(f"{sessions[i]}:{seed}"))


def cv_folds(sessions: list[str], seed: int, k: int = N_FOLDS) -> list[np.ndarray]:
    order = split_order(sessions, seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for pos, idx in enumerate(order):
        folds[pos % k].append(idx)
    return [np.asarray(f, dtype=np.int64) for f in folds]


# ---------------------------------------------------------------------------
def load_qwen_index() -> tuple[np.ndarray, dict[str, int], list[str]]:
    vecs = np.load(QWEN_VECS).astype(np.float32, copy=False)
    with open(QWEN_IDS) as f:
        ids = json.load(f)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    return vecs, id_to_idx, ids


def load_bge_index() -> tuple[np.ndarray, dict[str, int], list[str]]:
    vecs = np.load(BGE_VECS).astype(np.float32, copy=False)
    with open(BGE_IDS) as f:
        ids = json.load(f)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    return vecs, id_to_idx, ids


def a_prime_deep_batch(vecs, id_to_idx, track_ids, cases, *, n: int, k: int):
    """Vectorised A_max_recent_n at depth k. Mirrors expA implementation."""
    sess_indices: list[list[int]] = []
    sess_windows: list[list[int]] = []
    for c in cases:
        played_idxs = [id_to_idx[t] for t in c["music_turns"] if t in id_to_idx]
        sess_indices.append(played_idxs)
        sess_windows.append(played_idxs[-n:] if played_idxs else [])
    flat: list[int] = []
    offsets = [0]
    for win in sess_windows:
        flat.extend(win)
        offsets.append(offsets[-1] + len(win))
    if not flat:
        return [[] for _ in cases]
    flat_arr = np.asarray(flat, dtype=np.int64)
    stacked = vecs[flat_arr]
    sims = vecs @ stacked.T
    out: list[list[str]] = []
    for i in range(len(cases)):
        a, b = offsets[i], offsets[i + 1]
        if a == b:
            out.append([])
            continue
        agg = sims[:, a:b].max(axis=1).copy()
        exclude = set(sess_indices[i])
        for j in exclude:
            agg[j] = -1.0
        nn = min(k + len(exclude) + 1, len(agg))
        cand = np.argpartition(-agg, nn - 1)[:nn]
        cand = cand[np.argsort(-agg[cand])]
        out.append([track_ids[idx] for idx in cand if idx not in exclude][:k])
    return out


def rrf_fuse(seqs_with_weights, k: int, pool_size: int) -> list[str]:
    scores: dict[str, float] = {}
    for seq, w in seqs_with_weights:
        if w == 0:
            continue
        for r, tid in enumerate(seq, start=1):
            scores[tid] = scores.get(tid, 0.0) + w / (k + r)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:pool_size]


def build_cfg0209_pools(payload: dict[str, Any], a_lists200: list[list[str]]) -> list[list[str]]:
    cases = payload["cases"]
    src_b = payload["src_b"]
    src_c = payload["src_c"]
    src_d = payload["src_d"]
    pools: list[list[str]] = []
    for i in range(len(cases)):
        a = a_lists200[i][: CFG0209_DEPTHS["A"]] if CFG0209_DEPTHS["A"] > 0 else []
        b = src_b[i][: CFG0209_DEPTHS["B"]]
        c = src_c[i][: CFG0209_DEPTHS["C"]]
        d = src_d[i][: CFG0209_DEPTHS["D"]]
        seqs = [
            (a, CFG0209_WEIGHTS["A"]),
            (b, CFG0209_WEIGHTS["B"]),
            (c, CFG0209_WEIGHTS["C"]),
            (d, CFG0209_WEIGHTS["D"]),
        ]
        pool = rrf_fuse(seqs, k=RRF_K, pool_size=POOL_K)
        pools.append(pool)
    return pools


# ---------------------------------------------------------------------------
def encode_bge_queries(queries: list[str]) -> np.ndarray:
    """Encode strings with BGE-base-en-v1.5 + the query prefix. L2-normalised."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  loading BGE encoder on device={device}", flush=True)
    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    prefixed = [BGE_QUERY_PREFIX + q for q in queries]
    print(f"  encoding {len(queries)} queries", flush=True)
    vecs = encoder.encode(
        prefixed,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return vecs


# ---------------------------------------------------------------------------
def build_expR2_features() -> dict[str, Any]:
    """Stage 1. Augment _expB_features.pkl with 4 BGE features. Cached."""
    if OUT_FEATURES.exists():
        print(f"[Stage 1] loading cached {OUT_FEATURES}", flush=True)
        with open(OUT_FEATURES, "rb") as f:
            return pickle.load(f)  # noqa: S301

    print("[Stage 1] loading expB feature cache", flush=True)
    with open(EXPB_FEATURES, "rb") as f:
        expB = pickle.load(f)  # noqa: S301
    X_old: np.ndarray = expB["X"]
    gt_idx: np.ndarray = expB["gt_idx"]
    sizes: np.ndarray = expB["sizes"]
    feat_names: list[str] = list(expB["feat_names"])
    sessions: list[str] = list(expB["sessions"])
    pool_hit_rate = expB.get("pool_hit_rate")
    print(f"  expB pool_hit_rate = {pool_hit_rate} (cfg0209 = 0.3775 expected)",
          flush=True)
    if abs(pool_hit_rate - 0.3775) > 1e-6:
        raise RuntimeError("expB features not on cfg0209 pool — rebuild required.")

    print("[Stage 1] loading payload", flush=True)
    with open(PAYLOAD_PKL, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    if [c["session_id"] for c in cases] != sessions:
        raise RuntimeError("payload cases vs expB sessions misaligned.")

    if not ADD_BGE_FEATURES:
        # Fast path: use the 34 base features only. Pools are not needed for
        # training; they're only required for error-case analysis (we'll fill
        # them with placeholders).
        print("[Stage 1] ADD_BGE_FEATURES=False — skipping pool rebuild + BGE encoding",
              flush=True)
        out = {
            "X": X_old,
            "gt_idx": gt_idx,
            "sizes": sizes,
            "feat_names": feat_names,
            "sessions": sessions,
            "pool_hit_rate": pool_hit_rate,
            "pools": [["__placeholder__"] * 50 for _ in sessions],
        }
        OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_FEATURES, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {OUT_FEATURES} (shape={X_old.shape})", flush=True)
        return out

    print("[Stage 1] loading R1 BGE retrievals", flush=True)
    with open(EXPR1_E_RETRIEVALS, "rb") as f:
        r1 = pickle.load(f)  # noqa: S301
    queries_only = r1["query_strings_only"]
    queries_plus = r1["query_strings_plus"]
    E_only = r1["E_query_only"]
    E_plus = r1["E_query_plus_history"]
    if len(E_only) != len(sessions):
        raise RuntimeError("R1 retrievals misaligned.")

    print("[Stage 1] rebuilding cfg0209 pools (n=5, k=200 for A)", flush=True)
    t0 = time.time()
    qvecs, qid_to_idx, qids = load_qwen_index()
    a_lists200 = a_prime_deep_batch(qvecs, qid_to_idx, qids, cases, n=5, k=200)
    pools = build_cfg0209_pools(payload, a_lists200)
    pool_hits = sum(1 for i, c in enumerate(cases) if c["gt"] in pools[i])
    rebuilt_hr = pool_hits / len(cases)
    print(f"  rebuilt pool_hit = {rebuilt_hr:.4f} (built in {time.time()-t0:.1f}s)",
          flush=True)
    if abs(rebuilt_hr - 0.3775) > 1e-6:
        raise RuntimeError(f"rebuilt pool_hit {rebuilt_hr} != cached 0.3775")

    # Sanity: rebuilt pool ordering should match what expB used (we'll verify
    # by checking that gt_idx aligns with rebuilt pool index for sessions where
    # expB recorded gt_idx >= 0).
    mismatches = 0
    for i, c in enumerate(cases):
        gt = c["gt"]
        try:
            r_pos = pools[i].index(gt) if gt in pools[i] else -1
        except ValueError:
            r_pos = -1
        if r_pos != int(gt_idx[i]):
            mismatches += 1
    if mismatches > 0:
        raise RuntimeError(
            f"rebuilt pool order mismatch with expB cache for {mismatches} sessions"
        )
    print(f"  pool order matches expB cache for all {len(cases)} sessions",
          flush=True)
    del qvecs, qid_to_idx, qids

    # ---------- Encode BGE queries ----------
    print("[Stage 1] encoding BGE queries", flush=True)
    t0 = time.time()
    bge_q_only = encode_bge_queries(queries_only)
    bge_q_plus = encode_bge_queries(queries_plus)
    print(f"  encoded in {time.time()-t0:.1f}s", flush=True)

    # Load BGE catalog
    print("[Stage 1] loading BGE catalog", flush=True)
    bge_cat, bge_cat_idx, _ = load_bge_index()

    # ---------- Build new features ----------
    print("[Stage 1] adding 4 BGE features", flush=True)
    new_feats = [
        "bge_query_track_sim_query_only",
        "bge_query_track_sim_query_plus_history",
        "bge_rank_in_E_query_only",
        "bge_rank_in_E_query_plus_history",
    ]
    F_old = X_old.shape[2]
    F_new = F_old + len(new_feats)
    X_new = np.zeros((X_old.shape[0], X_old.shape[1], F_new), dtype=np.float64)
    X_new[:, :, :F_old] = X_old

    # E rank lookup per session
    rank_only_maps = [
        {tid: r for r, tid in enumerate(seq, start=1)} for seq in E_only
    ]
    rank_plus_maps = [
        {tid: r for r, tid in enumerate(seq, start=1)} for seq in E_plus
    ]

    n_sess = X_old.shape[0]
    for i in range(n_sess):
        pool = pools[i]
        size = int(sizes[i])
        # candidate BGE indices
        cand_bge_idx = [bge_cat_idx.get(tid) for tid in pool[:size]]
        valid_mask = np.array([j is not None for j in cand_bge_idx], dtype=bool)
        valid_idx = np.array([j for j in cand_bge_idx if j is not None],
                             dtype=np.int64)
        # Fill features (default zeros if no valid)
        sim_only_full = np.zeros(size, dtype=np.float64)
        sim_plus_full = np.zeros(size, dtype=np.float64)
        if len(valid_idx) > 0:
            cand_vecs = bge_cat[valid_idx]  # (n_valid, D)
            sim_only = cand_vecs @ bge_q_only[i]   # (n_valid,)
            sim_plus = cand_vecs @ bge_q_plus[i]
            sim_only_full[valid_mask] = sim_only
            sim_plus_full[valid_mask] = sim_plus
        # Rank in E (1/(rank+1); 0 if not in top-1000)
        rank_only_full = np.zeros(size, dtype=np.float64)
        rank_plus_full = np.zeros(size, dtype=np.float64)
        for j, tid in enumerate(pool[:size]):
            r1_only = rank_only_maps[i].get(tid)
            r1_plus = rank_plus_maps[i].get(tid)
            if r1_only is not None:
                rank_only_full[j] = 1.0 / (r1_only + 1)
            if r1_plus is not None:
                rank_plus_full[j] = 1.0 / (r1_plus + 1)
        X_new[i, :size, F_old + 0] = sim_only_full
        X_new[i, :size, F_old + 1] = sim_plus_full
        X_new[i, :size, F_old + 2] = rank_only_full
        X_new[i, :size, F_old + 3] = rank_plus_full

    feat_names_new = feat_names + new_feats

    # ---------- Persist ----------
    out = {
        "X": X_new,
        "gt_idx": gt_idx,
        "sizes": sizes,
        "feat_names": feat_names_new,
        "sessions": sessions,
        "pool_hit_rate": pool_hit_rate,
        "pools": pools,  # save for error-case analysis
    }
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FEATURES, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {OUT_FEATURES} (shape={X_new.shape})", flush=True)
    return out


# ---------------------------------------------------------------------------
# Evaluation primitives
def session_ndcg_at_k(scores: np.ndarray, gt_pos: int, valid_n: int, k: int = NDCG_K) -> float:
    """Compute nDCG@k for a single session given per-candidate scores.

    scores: (pool_K,) — higher is better; we mask invalid (>= valid_n) to -inf
    gt_pos: the GT index in pool, or -1 if GT not in pool
    """
    if gt_pos < 0:
        return 0.0
    s = scores.copy()
    s[valid_n:] = -np.inf
    gt_score = s[gt_pos]
    strict = int((s > gt_score).sum())
    tie_before = int(((s == gt_score) & (np.arange(len(s)) < gt_pos)).sum())
    rank0 = strict + tie_before
    if rank0 >= k:
        return 0.0
    return 1.0 / np.log2(rank0 + 2)


def vec_ndcg(X_sub: np.ndarray, gt_sub: np.ndarray, sizes_sub: np.ndarray, weights: np.ndarray) -> float:
    """Vectorised nDCG@20 with linear weights. Same as expB's vec_ndcg_pre."""
    pool_axis = np.arange(X_sub.shape[1])[None, :]
    valid_pool = pool_axis < sizes_sub[:, None]
    scores = X_sub @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_sub >= 0
    safe_gt = np.where(has_gt, gt_sub, 0)
    gt_scores = scores[np.arange(X_sub.shape[0]), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def vec_ndcg_per_session(X_sub: np.ndarray, gt_sub: np.ndarray, sizes_sub: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    pool_axis = np.arange(X_sub.shape[1])[None, :]
    valid_pool = pool_axis < sizes_sub[:, None]
    scores = X_sub @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_sub >= 0
    safe_gt = np.where(has_gt, gt_sub, 0)
    gt_scores = scores[np.arange(X_sub.shape[0]), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0)
    return vals.astype(np.float64)


def eval_per_session_from_scores(scores: np.ndarray, gt_idx: np.ndarray,
                                 sizes: np.ndarray) -> np.ndarray:
    """Generic: scores shape (N, K). Returns per-session nDCG@20 (N,)."""
    pool_axis = np.arange(scores.shape[1])[None, :]
    valid_pool = pool_axis < sizes[:, None]
    s = np.where(valid_pool, scores, -np.inf)
    has_gt = gt_idx >= 0
    safe_gt = np.where(has_gt, gt_idx, 0)
    gt_scores = s[np.arange(scores.shape[0]), safe_gt]
    strict_gt = (s > gt_scores[:, None]).sum(axis=1)
    tie_before = ((s == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    return np.where(has_gt & (rank0 < NDCG_K), 1.0 / np.log2(rank0 + 2), 0.0).astype(np.float64)


# ---------------------------------------------------------------------------
def fit_powell(X_train: np.ndarray, gt_train: np.ndarray, sizes_train: np.ndarray,
               feat_names: list[str]) -> np.ndarray:
    """Powell on linear weights. Init from POWELL_INIT (0 for unknown)."""
    from scipy.optimize import minimize  # type: ignore
    init = np.asarray(
        [POWELL_INIT.get(fn, 0.0) for fn in feat_names], dtype=np.float64
    )

    def objective(w: np.ndarray) -> float:
        return -vec_ndcg(X_train, gt_train, sizes_train, w)

    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return np.asarray(res.x, dtype=np.float64)


# ---------------------------------------------------------------------------
def lgbm_train_lambdarank(X_tr: np.ndarray, y_tr: np.ndarray, group_tr: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray,
                           seed: int, params: dict):
    import lightgbm as lgb  # type: ignore
    train_set = lgb.Dataset(X_tr, label=y_tr, group=group_tr)
    valid_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)
    p = dict(params)
    p["objective"] = "lambdarank"
    p["metric"] = "ndcg"
    p["ndcg_eval_at"] = [NDCG_K]
    p["lambdarank_truncation_level"] = NDCG_K
    p["seed"] = seed
    p["verbosity"] = -1
    p["num_threads"] = 0  # let it use all cores
    booster = lgb.train(
        params=p,
        train_set=train_set,
        num_boost_round=p.pop("n_estimators", 200),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return booster


def lgbm_train_pointwise(X_tr: np.ndarray, y_tr: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          seed: int, params: dict):
    import lightgbm as lgb  # type: ignore
    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    p = dict(params)
    p["objective"] = "regression"
    p["metric"] = "rmse"
    p["seed"] = seed
    p["verbosity"] = -1
    p["num_threads"] = 0
    booster = lgb.train(
        params=p,
        train_set=train_set,
        num_boost_round=p.pop("n_estimators", 200),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return booster


# ---------------------------------------------------------------------------
def reshape_for_lgbm(X: np.ndarray, gt_idx: np.ndarray, sizes: np.ndarray,
                     idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (n_sess_sub, K, F) → (n_sess_sub*K_valid, F) with per-session group sizes.

    NOTE: in our setup all sizes == POOL_K == 50. We still respect sizes for safety.
    """
    K = X.shape[1]
    F = X.shape[2]
    Xs = X[idx].reshape(-1, F)
    # Build labels: 1.0 if (i, j == gt_idx), else 0.0
    rows = len(idx)
    labels = np.zeros(rows * K, dtype=np.float64)
    for k_, i in enumerate(idx):
        gp = gt_idx[i]
        if gp >= 0:
            labels[k_ * K + gp] = 1.0
    # Group sizes
    groups = np.full(rows, K, dtype=np.int64)
    # If sizes < K, we'd need to slice — not the case here.
    if (sizes[idx] != K).any():
        raise RuntimeError("variable group sizes — unsupported in this layout")
    return Xs, labels, groups


def predict_lgbm_per_session(booster, X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Returns scores shape (len(idx), K). Higher is better."""
    K = X.shape[1]
    F = X.shape[2]
    flat = X[idx].reshape(-1, F)
    preds = booster.predict(flat)
    return preds.reshape(len(idx), K)


# ---------------------------------------------------------------------------
def run_baselines(features: dict[str, Any]):
    """Stages 3-4: full evaluation across baselines.

    Returns (results_dict, b1_per_session_avg, b3_per_session_avg, delta_avg).
    """
    X = features["X"]
    gt_idx = features["gt_idx"]
    sizes = features["sizes"]
    feat_names: list[str] = list(features["feat_names"])
    sessions = features["sessions"]
    n_sess = X.shape[0]
    F = X.shape[2]

    # Identify B4 ablation feature columns (drop bge_*)
    bge_cols = [i for i, fn in enumerate(feat_names) if fn.startswith("bge_")]
    non_bge_cols = [i for i in range(F) if i not in set(bge_cols)]
    print(f"feature counts: total={F}, BGE_new={len(bge_cols)}, "
          f"non_BGE={len(non_bge_cols)}", flush=True)

    # B0: orig_rank only (raw rank, no postrank). orig_rank already encodes
    # 1/(rank0+1). Higher is better.
    or_idx = feat_names.index("orig_rank")
    # B0 score = orig_rank value directly.

    # Shared LGBM hyperparameters
    lgbm_hp_grid = [
        # (label, params)
        ("hp31_lr05", {"num_leaves": 31, "learning_rate": 0.05,
                       "min_child_samples": 20, "n_estimators": 250,
                       "feature_fraction": 0.9, "bagging_fraction": 0.9,
                       "bagging_freq": 1, "min_gain_to_split": 0.0}),
    ]

    results: dict[str, Any] = {
        "B0_raw_rank": {"per_seed": []},
        "B1_powell_linear": {"per_seed": []},
        "B2_lgbm_pointwise": {"per_seed": [], "hp": lgbm_hp_grid[0][0]},
        "B3_lgbm_lambdarank": {"per_seed": [], "hp": lgbm_hp_grid[0][0]},
        "B4_lgbm_lambdarank_no_bge": {"per_seed": [], "hp": lgbm_hp_grid[0][0]},
    }

    # Per-session score matrices for B3 deep-dive (needed for error-case analysis)
    B1_per_session_all: list[np.ndarray] = []   # length n_seeds, each (n_sess,) of nDCG
    B3_per_session_all: list[np.ndarray] = []
    B3_importances: list[dict[str, float]] = []  # per seed, gain importance

    t_total = time.time()

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===", flush=True)
        folds = cv_folds(sessions, seed)
        # Per-session nDCG arrays for this seed
        b0_perfold = []
        b1_perfold = []
        b2_perfold = []
        b3_perfold = []
        b4_perfold = []
        b1_persess = np.zeros(n_sess, dtype=np.float64)
        b3_persess = np.zeros(n_sess, dtype=np.float64)

        # Aggregate importances across folds
        b3_imp_sum = np.zeros(F, dtype=np.float64)

        for fi, fold in enumerate(folds):
            held = set(fold.tolist())
            train = np.asarray(
                [i for i in range(n_sess) if i not in held], dtype=np.int64
            )
            X_tr_full = X[train]
            X_held_full = X[fold]

            # ---- B0: raw rank ----
            scores_held = X_held_full[:, :, or_idx]  # (n_held, K)
            persess = eval_per_session_from_scores(scores_held, gt_idx[fold], sizes[fold])
            b0_perfold.append(float(persess.mean()))

            # ---- B1: Powell linear (refit per fold) ----
            wts = fit_powell(X_tr_full, gt_idx[train], sizes[train], feat_names)
            persess_b1 = vec_ndcg_per_session(X_held_full, gt_idx[fold],
                                              sizes[fold], wts)
            b1_perfold.append(float(persess_b1.mean()))
            b1_persess[fold] = persess_b1

            # Build LGBM tensors. We use 80/20 inside training for early stopping val.
            # Note: this is a SECONDARY split inside train, NOT the held-out fold.
            rng = np.random.default_rng(seed * 100 + fi)
            shuffled = rng.permutation(train)
            split = int(0.85 * len(shuffled))
            inner_tr = shuffled[:split]
            inner_va = shuffled[split:]

            Xtr_flat, ytr_flat, gtr = reshape_for_lgbm(X, gt_idx, sizes, inner_tr)
            Xva_flat, yva_flat, gva = reshape_for_lgbm(X, gt_idx, sizes, inner_va)

            # ---- B2: LGBM pointwise ----
            hp = lgbm_hp_grid[0][1]
            booster_pw = lgbm_train_pointwise(
                Xtr_flat, ytr_flat, Xva_flat, yva_flat, seed, dict(hp)
            )
            scores_b2 = predict_lgbm_per_session(booster_pw, X, fold)
            persess_b2 = eval_per_session_from_scores(scores_b2, gt_idx[fold], sizes[fold])
            b2_perfold.append(float(persess_b2.mean()))

            # ---- B3: LGBM LambdaRank ----
            booster_lr = lgbm_train_lambdarank(
                Xtr_flat, ytr_flat, gtr, Xva_flat, yva_flat, gva, seed, dict(hp)
            )
            scores_b3 = predict_lgbm_per_session(booster_lr, X, fold)
            persess_b3 = eval_per_session_from_scores(scores_b3, gt_idx[fold], sizes[fold])
            b3_perfold.append(float(persess_b3.mean()))
            b3_persess[fold] = persess_b3

            # B3 importances (gain)
            try:
                gains = booster_lr.feature_importance(importance_type="gain")
                b3_imp_sum += np.asarray(gains, dtype=np.float64)
            except Exception as e:
                print(f"  warn: importance unavailable seed{seed} fold{fi}: {e}", flush=True)

            # ---- B4: LGBM LambdaRank without bge_* features ----
            non_bge_arr = np.asarray(non_bge_cols, dtype=np.int64)
            X_nb = X[:, :, non_bge_arr]
            Xtr_flat_nb, _, _ = reshape_for_lgbm(X_nb, gt_idx, sizes, inner_tr)
            Xva_flat_nb, _, _ = reshape_for_lgbm(X_nb, gt_idx, sizes, inner_va)
            booster_b4 = lgbm_train_lambdarank(
                Xtr_flat_nb, ytr_flat, gtr, Xva_flat_nb, yva_flat, gva, seed, dict(hp)
            )
            scores_b4 = predict_lgbm_per_session(booster_b4, X_nb, fold)
            persess_b4 = eval_per_session_from_scores(scores_b4, gt_idx[fold], sizes[fold])
            b4_perfold.append(float(persess_b4.mean()))

            print(f"  seed{seed} fold{fi}: B0={b0_perfold[-1]:.4f} "
                  f"B1={b1_perfold[-1]:.4f} B2={b2_perfold[-1]:.4f} "
                  f"B3={b3_perfold[-1]:.4f} B4={b4_perfold[-1]:.4f}",
                  flush=True)

        results["B0_raw_rank"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b0_perfold)),
            "cv5_std": float(np.std(b0_perfold, ddof=1)),
            "per_fold": [float(x) for x in b0_perfold],
        })
        results["B1_powell_linear"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b1_perfold)),
            "cv5_std": float(np.std(b1_perfold, ddof=1)),
            "per_fold": [float(x) for x in b1_perfold],
        })
        results["B2_lgbm_pointwise"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b2_perfold)),
            "cv5_std": float(np.std(b2_perfold, ddof=1)),
            "per_fold": [float(x) for x in b2_perfold],
        })
        results["B3_lgbm_lambdarank"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b3_perfold)),
            "cv5_std": float(np.std(b3_perfold, ddof=1)),
            "per_fold": [float(x) for x in b3_perfold],
        })
        results["B4_lgbm_lambdarank_no_bge"]["per_seed"].append({
            "seed": seed,
            "cv5_mean": float(np.mean(b4_perfold)),
            "cv5_std": float(np.std(b4_perfold, ddof=1)),
            "per_fold": [float(x) for x in b4_perfold],
        })
        B1_per_session_all.append(b1_persess)
        B3_per_session_all.append(b3_persess)
        # Average importance across folds
        imp_avg = b3_imp_sum / len(folds)
        B3_importances.append(dict(zip(feat_names, [float(v) for v in imp_avg], strict=True)))

    # Aggregates
    for key in results:
        means = [s["cv5_mean"] for s in results[key]["per_seed"]]
        results[key]["aggregate"] = {
            "mean_cv5": float(np.mean(means)),
            "std_of_means": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
            "min": float(np.min(means)),
            "max": float(np.max(means)),
        }

    # B3 importance averaged across seeds (gain)
    imp_table = {fn: 0.0 for fn in feat_names}
    for d in B3_importances:
        for k, v in d.items():
            imp_table[k] += v
    for fn in imp_table:
        imp_table[fn] /= len(B3_importances)
    sorted_imp = sorted(imp_table.items(), key=lambda x: -x[1])
    results["B3_lgbm_lambdarank"]["top_15_importance"] = sorted_imp[:15]
    results["B3_lgbm_lambdarank"]["all_importance"] = sorted_imp

    # B3 vs B1 per-session deltas (averaged across seeds)
    b1_avg = np.mean(np.stack(B1_per_session_all, axis=0), axis=0)
    b3_avg = np.mean(np.stack(B3_per_session_all, axis=0), axis=0)
    delta = b3_avg - b1_avg

    print(f"\n[Done] total time: {time.time() - t_total:.1f}s", flush=True)
    return results, b1_avg, b3_avg, delta


# ---------------------------------------------------------------------------
def error_case_analysis(features, results, b1_avg, b3_avg, delta) -> dict[str, Any]:
    """Top wins and losses for B3 vs B1, with feature deltas."""
    sessions = features["sessions"]
    pools = features["pools"]
    gt_idx = features["gt_idx"]
    feat_names = features["feat_names"]
    X = features["X"]
    n_sess = len(sessions)

    order = np.argsort(-delta)  # desc: biggest wins first
    top_wins = order[:20].tolist()
    top_losses = order[-20:][::-1].tolist()  # losses (most negative)

    def session_summary(i):
        gp = int(gt_idx[i])
        gt_track = pools[i][gp] if gp >= 0 and gp < len(pools[i]) else None
        gt_features: dict[str, float] = {}
        if gp >= 0:
            for j, fn in enumerate(feat_names):
                gt_features[fn] = float(X[i, gp, j])
        return {
            "session_id": sessions[i],
            "gt_idx": gp,
            "gt_track_id": gt_track,
            "B1_ndcg": float(b1_avg[i]),
            "B3_ndcg": float(b3_avg[i]),
            "delta": float(delta[i]),
            "gt_features": gt_features,
        }

    return {
        "top_wins": [session_summary(i) for i in top_wins],
        "top_losses": [session_summary(i) for i in top_losses],
        "n_sessions_with_gt": int((gt_idx >= 0).sum()),
        "n_sessions_total": n_sess,
        "delta_summary": {
            "mean_delta": float(delta.mean()),
            "std_delta": float(delta.std(ddof=1)),
            "median_delta": float(np.median(delta)),
            "p_b3_gt_b1": float((delta > 0).mean()),
        },
    }


# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT R2 — LightGBM LambdaRank on cfg0209 pool", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    features = build_expR2_features()
    print(f"\n[Stage 1 done] X shape={features['X'].shape}, "
          f"feat_count={len(features['feat_names'])}", flush=True)

    print("\n[Stage 2] checking model availability", flush=True)
    try:
        import lightgbm as lgb  # type: ignore
        print(f"  lightgbm: {lgb.__version__}", flush=True)
    except ImportError:
        raise RuntimeError("lightgbm not installed")

    print("\n[Stages 3-4] running baselines + LambdaRank (5 seeds × 5 folds)", flush=True)
    results, b1_avg, b3_avg, delta = run_baselines(features)

    print("\n[Stage 5] error case analysis", flush=True)
    error_cases = error_case_analysis(features, results, b1_avg, b3_avg, delta)

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w") as f:
        json.dump({
            "meta": {
                "total_time_sec": time.time() - t_start,
                "n_sessions": features["X"].shape[0],
                "n_features": features["X"].shape[2],
                "feat_names": features["feat_names"],
                "seeds": SEEDS,
                "n_folds": N_FOLDS,
                "ndcg_k": NDCG_K,
                "pool_hit_rate": features["pool_hit_rate"],
            },
            "results": results,
        }, f, indent=2)
    print(f"  wrote {OUT_RESULTS}", flush=True)

    with open(OUT_ERRORS, "w") as f:
        json.dump(error_cases, f, indent=2)
    print(f"  wrote {OUT_ERRORS}", flush=True)

    # Verdict
    b3_mean = results["B3_lgbm_lambdarank"]["aggregate"]["mean_cv5"]
    b3_std = results["B3_lgbm_lambdarank"]["aggregate"]["std_of_means"]
    b1_mean = results["B1_powell_linear"]["aggregate"]["mean_cv5"]
    b0_mean = results["B0_raw_rank"]["aggregate"]["mean_cv5"]
    if b3_mean >= 0.178 and b3_std <= 0.003:
        verdict = "PASS"
    elif b3_mean >= 0.170:
        verdict = "PROMISING"
    elif b3_mean >= 0.165:
        verdict = "WEAK"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 70, flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"  B0 raw rank mean_cv5     = {b0_mean:.4f}", flush=True)
    print(f"  B1 Powell mean_cv5       = {b1_mean:.4f}", flush=True)
    print(f"  B3 LambdaRank mean_cv5   = {b3_mean:.4f} ± {b3_std:.4f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
