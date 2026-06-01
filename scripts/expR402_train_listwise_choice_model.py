#!/usr/bin/env python3
"""R400 Model B (GO_RANKING_ONLY) — per-user taste prior as RANKER FEATURES, 5-fold OOF.

Tests whether a NEW per-user taste prior, added as ranker features over the EXISTING
production pool (c3 R54-stacked RRF, the R84c/R92 family pool), converts to OOF nDCG@20.

Production proxy = R106 A-clean (blind nDCG@20 0.5073; dev OOF production-proxy ~0.3159),
which is exactly arm A here: c3 pool + 37 base features (FEAT_R39_ALL + r84 rank_inv/
presence/cosine) under a 5-fold OOF sibling LightGBM LambdaRank.

The taste prior is genuinely untested: R104 used only within-conversation played tracks;
here we use each user's FULL train-session history (cache/r400/user_train_history.json;
74.2% of dev users are in train). Leak-free: history is from TRAIN sessions, dev is held
out OOF, and the GT-in-own-prefix rate was verified 0.

Two OOF sibling-LR arms (per held-out fold, LR trained on the other 4 folds, c3 pool):
  A  base       : 37 base feats (production proxy / R106 A-clean analog).
  D  base+userh : 37 base feats + 5 NEW user-history feature columns.

The 5 user-history features (per (case, candidate) over the c3 pool):
  1. direct_in_history   : 1.0 if cand in user_train_history[user_id] else 0.0.
  2. user_taste_cfbpr_cos: cos(cf-bpr[cand], L2-norm mean of cf-bpr over user's train hist).
  3. user_taste_meta_cos : same with metadata-qwen3 (1024-d) centroid.
  4. played_nn_cfbpr_cos : cos(cf-bpr[cand], recency-weighted (0.85) mean over case's played).
  5. user_in_train       : case-level 1.0 if user_id in train (broadcast to all candidates).

Reports arm D vs arm A: nDCG@20 (all + h7 + same-artist + diff-artist + n_prior buckets),
churn (top-1 changed /80-equiv, top-20 overlap), recovered/lost top-20 GTs, and the
LightGBM gain-importance of the 5 user-history features in arm D.

Output: exp/eval/expR403_oof_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_R39_ALL, _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5
DECAY = 0.85

W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
USER_HIST = REPO / "cache" / "r400" / "user_train_history.json"
OUT_JSON = REPO / "exp" / "eval" / "expR403_oof_results.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
USERH_FEATS = ["direct_in_history", "user_taste_cfbpr_cos", "user_taste_meta_cos",
               "played_nn_cfbpr_cos", "user_in_train"]
FEAT_NAMES_D = FEAT_NAMES_R84 + USERH_FEATS


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    return "no-git" if g is None else subprocess.check_output(
        [g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ndcg_at_k(gt_rank: int, k: int) -> float:
    return 0.0 if (gt_rank <= 0 or gt_rank > k) else 1.0 / math.log2(gt_rank + 1)


def load_r84_lists_for_fold(fold: int) -> dict[int, dict]:
    path = (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json" if fold == 0
            else REPO / "cache" / "r84" / f"phase1_fold{fold}" / "oof_r84_lists.json")
    if not path.exists():
        raise FileNotFoundError(f"R84 OOF lists missing for fold {fold}: {path}")
    raw = json.load(open(path))
    out = {}
    for ci_str, pairs in raw.items():
        tids = [t for t, _ in pairs]
        out[int(ci_str)] = {"ranks": {t: r + 1 for r, t in enumerate(tids)},
                            "scores": {t: float(s) for t, s in pairs}}
    return out


def overwrite_r84(feats: np.ndarray, pool: list[str],
                  r84_ranks: dict[str, int], r84_scores: dict[str, float]) -> np.ndarray:
    for i, tid in enumerate(pool):
        feats[i, N_R39 + 0] = (1.0 / r84_ranks[tid]) if tid in r84_ranks else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in r84_ranks else 0.0
        feats[i, N_R39 + 2] = r84_scores.get(tid, 0.0)
    return feats


# ----------------------------------------------------------------------------
# Official Track-Embeddings: cf-bpr (128) + metadata-qwen3 (1024), L2-normalized.
# Build matrices row-by-row, skip missing / len-mismatch (cold tracks).
# ----------------------------------------------------------------------------
def load_embeddings():
    from datasets import load_dataset
    print(f"{ts()} Loading official Track-Embeddings (cf-bpr 128, metadata-qwen3 1024) ...",
          flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
    ds = ds.select_columns(["track_id", "cf-bpr", "metadata-qwen3_embedding_0.6b"])
    ids = [str(t) for t in ds["track_id"]]
    tid2idx = {t: i for i, t in enumerate(ids)}

    def build(col, dim):
        raw = ds[col]
        M = np.zeros((len(ids), dim), dtype=np.float32)
        miss = 0
        for i, v in enumerate(raw):
            if v is not None and len(v) == dim:
                M[i] = v
            else:
                miss += 1
        norms = np.linalg.norm(M, axis=1)
        Mn = M / np.clip(norms[:, None], 1e-8, None)
        print(f"  {col}: {M.shape} missing/cold (no vec): {miss} ({miss/len(ids):.4f})",
              flush=True)
        return Mn, norms

    cf_n, cf_norm = build("cf-bpr", 128)
    meta_n, meta_norm = build("metadata-qwen3_embedding_0.6b", 1024)
    return ids, tid2idx, cf_n, cf_norm, meta_n, meta_norm


def centroid(track_ids, Mn, norms, tid2idx, recency_weighted=False, decay=DECAY):
    """L2-normalized (recency-weighted) mean of warm embedding rows. None if empty."""
    idxs = [tid2idx[t] for t in track_ids if t in tid2idx and norms[tid2idx[t]] > 1e-8]
    if not idxs:
        return None
    if recency_weighted:
        m = len(idxs)
        w = np.array([decay ** (m - 1 - j) for j in range(m)], dtype=np.float32)
        v = (w[:, None] * Mn[idxs]).sum(0)
    else:
        v = Mn[idxs].mean(0)
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return None
    return (v / nv).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()
    out_json = Path(args.out)

    t0 = time.time()
    print(f"{ts()} R400 Model B — per-user taste prior as ranker features, 5-fold OOF")
    print(f"  pool: c3 R54-stacked RRF (production), K={POOL_K}")
    print(f"  arms: A=base(37, R106 A-clean proxy)  D=base+5 user-history feats(42)")
    print("=" * 72)

    print(f"\n{ts()} Loading payload + R21/R54/ALS + maps ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    user_hist = json.load(open(USER_HIST))
    print(f"  user_train_history: {len(user_hist)} users", flush=True)
    cases_in_train = sum(1 for c in cases if c.get("user_id") in user_hist)
    print(f"  dev cases with user_id in train: {cases_in_train}/{n} "
          f"({cases_in_train/n:.4f})", flush=True)

    ids, tid2idx, cf_n, cf_norm, meta_n, meta_norm = load_embeddings()

    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_to_idx: dict[int, list[int]] = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)

    print(f"{ts()} Loading R84 OOF (5 folds) ...", flush=True)
    r84_per_fold = {k: load_r84_lists_for_fold(k) for k in range(N_FOLDS)}

    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx)

    def featurize37(i, src_lists, pool):
        case = cases[i]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album).astype(np.float32)

    # ------------------------------------------------------------------
    # Per-case user-history feature columns (5 cols) over the c3 pool.
    # ------------------------------------------------------------------
    def userh_cols(i, pool):
        case = cases[i]
        uid = case.get("user_id")
        in_train = uid in user_hist
        hist = user_hist.get(uid, [])
        hist_set = set(hist)

        # case-level centroids (computed once per case)
        u_cf = centroid(hist, cf_n, cf_norm, tid2idx) if in_train and hist else None
        u_meta = centroid(hist, meta_n, meta_norm, tid2idx) if in_train and hist else None
        played = case["music_turns"]
        p_cf = centroid(played, cf_n, cf_norm, tid2idx, recency_weighted=True) if played else None

        g = np.zeros((len(pool), 5), dtype=np.float32)
        for r, tid in enumerate(pool):
            ti = tid2idx.get(tid, -1)
            warm_cf = ti >= 0 and cf_norm[ti] > 1e-8
            warm_meta = ti >= 0 and meta_norm[ti] > 1e-8
            g[r, 0] = 1.0 if tid in hist_set else 0.0
            if u_cf is not None and warm_cf:
                g[r, 1] = float(cf_n[ti] @ u_cf)
            if u_meta is not None and warm_meta:
                g[r, 2] = float(meta_n[ti] @ u_meta)
            if p_cf is not None and warm_cf:
                g[r, 3] = float(cf_n[ti] @ p_cf)
            g[r, 4] = 1.0 if in_train else 0.0
        return g

    # --- Pre-build per-case pools + feature matrices ---
    print(f"\n{ts()} === Pre-building c3 pools + features (8000 cases) ===", flush=True)
    cf_store: dict[int, dict] = {}
    t_feat = time.time()
    hit_base = 0
    direct_hits = 0  # GT directly in user's train history (sanity vs leakage_audit 423)
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)

        gt = cases[i]["gt"]
        gp = pool.index(gt) if gt in pool else -1
        hit_base += gp >= 0

        uid = cases[i].get("user_id")
        if uid in user_hist and gt in set(user_hist[uid]):
            direct_hits += 1

        owner = case_fold[i]
        r84m = r84_per_fold[owner][i]

        feats37 = overwrite_r84(featurize37(i, src_lists, pool), pool,
                                r84m["ranks"], r84m["scores"])
        uh = userh_cols(i, pool)
        feats_D = np.concatenate([feats37, uh], axis=1)

        cf_store[i] = {
            "pool": pool, "feats_D": feats_D,   # A = feats_D[:, :37]
            "gp": gp, "gt": gt,
        }
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t_feat:.0f}s) pool_hit={hit_base} "
                  f"gt_in_userhist={direct_hits}", flush=True)
    print(f"  pool_hit (GT in c3 pool): {hit_base}/{n} ({hit_base/n:.4f})", flush=True)
    print(f"  GT directly in user train-history: {direct_hits}/{n} "
          f"(expect ~423 per leakage_audit)", flush=True)
    print(f"  feat-build elapsed: {time.time() - t_feat:.0f}s", flush=True)

    # --- 5-fold OOF: train 2 sibling LRs (A, D), score held-out fold ---
    results: list[dict] = []
    userh_imp_accum = defaultdict(float)
    all_imp_accum = defaultdict(float)
    for fk in range(N_FOLDS):
        print(f"\n{ts()} === FOLD {fk} held out ===", flush=True)
        eval_idx = fold_to_idx[fk]
        train_idx = [i for i in range(n) if case_fold[i] != fk]

        def stack(ncol):
            X, y, grp = [], [], []
            for i in train_idx:
                c = cf_store[i]
                P = c["pool"]; F = c["feats_D"]; gp = c["gp"]
                for r in range(len(P)):
                    X.append(F[r, :ncol]); y.append(1.0 if r == gp else 0.0)
                grp.append(len(P))
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), grp

        # Arm A (37)
        print("  train A (base 37) ...", flush=True)
        XA, yA, gA = stack(37)
        lrA = lgb.train(LR_PARAMS, lgb.Dataset(XA, label=yA, group=gA,
                        feature_name=FEAT_NAMES_R84), num_boost_round=LR_NUM_BOOST_ROUND)
        del XA, yA
        # Arm D (42)
        print("  train D (base 37 + 5 user-history) ...", flush=True)
        XD, yD, gD = stack(42)
        lrD = lgb.train(LR_PARAMS, lgb.Dataset(XD, label=yD, group=gD,
                        feature_name=FEAT_NAMES_D), num_boost_round=LR_NUM_BOOST_ROUND)
        del XD, yD
        for fn, im in zip(FEAT_NAMES_D, lrD.feature_importance(importance_type="gain")):
            all_imp_accum[fn] += float(im)
            if fn in USERH_FEATS:
                userh_imp_accum[fn] += float(im)

        print(f"  scoring fold {fk} ({len(eval_idx)} cases) ...", flush=True)
        for i in eval_idx:
            c = cf_store[i]
            pool = c["pool"]; gp = c["gp"]
            sA = lrA.predict(c["feats_D"][:, :37])
            sD = lrD.predict(c["feats_D"])
            oA = np.argsort(-sA, kind="mergesort")
            oD = np.argsort(-sD, kind="mergesort")

            def rank_of(order, gp):
                if gp < 0:
                    return -1
                w = np.where(order == gp)[0]
                return int(w[0]) + 1 if len(w) else -1

            rA = rank_of(oA, gp); rD = rank_of(oD, gp)
            topA = [pool[int(j)] for j in oA[:TOP_K]]
            topD = [pool[int(j)] for j in oD[:TOP_K]]

            uid = cases[i].get("user_id")
            results.append({
                "case_idx": i, "fold": fk,
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "user_in_train": int(uid in user_hist),
                "n_prior_history": len(user_hist.get(uid, [])),
                "rA": rA, "rD": rD,
                "ndA": ndcg_at_k(rA, TOP_K), "ndD": ndcg_at_k(rD, TOP_K),
                "A_in20": 0 < rA <= TOP_K, "D_in20": 0 < rD <= TOP_K,
                "overlap_AD": len(set(topA) & set(topD)),
                "top1_AD_eq": int(bool(topA) and bool(topD) and topA[0] == topD[0]),
            })
        del lrA, lrD

    # --- Aggregate ---
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7 = [r for r in results if r["n_prior_music"] == 7]
    same = [r for r in results if r["same_artist"]]
    diff = [r for r in results if not r["same_artist"]]
    in_train_rows = [r for r in results if r["user_in_train"]]
    not_in_train_rows = [r for r in results if not r["user_in_train"]]

    metrics = {}
    subsets = [("all", results), ("h7", h7), ("same_artist", same),
               ("diff_artist", diff), ("user_in_train", in_train_rows),
               ("user_not_in_train", not_in_train_rows)]
    for name, rows in subsets:
        A, D = avg(rows, "ndA"), avg(rows, "ndD")
        metrics[name] = {"n": len(rows), "A_base": A, "D_userh": D, "dD_vs_A": D - A}

    # n_prior (history-size) buckets — does the prior help users with more train history?
    def hbucket(npri):
        if npri == 0:
            return "0"
        if npri <= 8:
            return "1-8"
        if npri <= 32:
            return "9-32"
        if npri <= 64:
            return "33-64"
        return "65+"
    nprior_buckets = {}
    bk = defaultdict(list)
    for r in results:
        bk[hbucket(r["n_prior_history"])].append(r)
    for name in ["0", "1-8", "9-32", "33-64", "65+"]:
        rows = bk.get(name, [])
        A, D = avg(rows, "ndA"), avg(rows, "ndD")
        nprior_buckets[name] = {"n": len(rows), "A_base": A, "D_userh": D, "dD_vs_A": D - A}

    rec_all = sum(1 for r in results if r["D_in20"] and not r["A_in20"])
    lost_all = sum(1 for r in results if r["A_in20"] and not r["D_in20"])
    h7_rec = sum(1 for r in h7 if r["D_in20"] and not r["A_in20"])
    h7_lost = sum(1 for r in h7 if r["A_in20"] and not r["D_in20"])
    overlap_AD = avg(results, "overlap_AD")
    churn = (1 - avg(results, "top1_AD_eq")) * 80

    by_fold_h7 = {}
    for k in range(N_FOLDS):
        rows = [r for r in h7 if r["fold"] == k]
        if rows:
            by_fold_h7[k] = {"n": len(rows), "A": avg(rows, "ndA"), "D": avg(rows, "ndD"),
                             "dD_vs_A": avg(rows, "ndD") - avg(rows, "ndA")}

    userh_imp = {k: userh_imp_accum[k] for k in USERH_FEATS}
    tot_imp = sum(all_imp_accum.values()) or 1.0
    userh_imp_frac = sum(userh_imp.values()) / tot_imp

    h7_d = metrics["h7"]["dD_vs_A"]
    sa_d = metrics["same_artist"]["dD_vs_A"]
    diff_d = metrics["diff_artist"]["dD_vs_A"]
    all_d = metrics["all"]["dD_vs_A"]
    A1 = h7_d >= 0.005
    A2 = h7_rec > h7_lost
    B1 = sa_d >= -0.005
    B2 = diff_d >= -0.005
    B3 = overlap_AD >= 8.0
    userh_used = userh_imp_frac > 0.01

    if (A1 or A2) and B1 and B2 and B3 and userh_used:
        verdict = "PROCEED_TO_BLIND"
    elif not B1 or not B2:
        verdict = "ARCHIVE_SPRINT"
    elif not userh_used:
        verdict = "ARCHIVE_USERH_IGNORED"
    elif h7_d < 0.002 and not A2:
        verdict = "ARCHIVE_NO_LIFT"
    else:
        verdict = "INVESTIGATE"

    # --- Report ---
    print(f"\n{ts()} === AGGREGATE (5-fold OOF) ===")
    print(f"  {'subset':18} {'n':>5} {'A_base':>8} {'D_userh':>8} {'ΔD-A':>9}")
    for name, m in metrics.items():
        print(f"  {name:18} {m['n']:5d} {m['A_base']:8.4f} {m['D_userh']:8.4f} "
              f"{m['dD_vs_A']:+9.4f}")
    print(f"\n  nDCG@20 by user train-history size bucket:")
    print(f"  {'bucket':18} {'n':>5} {'A_base':>8} {'D_userh':>8} {'ΔD-A':>9}")
    for name, m in nprior_buckets.items():
        print(f"  {name:18} {m['n']:5d} {m['A_base']:8.4f} {m['D_userh']:8.4f} "
              f"{m['dD_vs_A']:+9.4f}")
    print(f"\n  recovered top-20 (D vs A): all {rec_all} / lost {lost_all} "
          f"(net {rec_all - lost_all:+d})")
    print(f"  h7 recovered: {h7_rec}   lost: {h7_lost}   net: {h7_rec - h7_lost:+d}")
    print(f"  top-20 overlap A vs D: {overlap_AD:.2f}/20   top-1 churn: {churn:.1f}/80")
    print(f"\n  arm-D LR gain importance (5 user-history feats = "
          f"{userh_imp_frac*100:.2f}% of total):")
    for k in USERH_FEATS:
        print(f"    {k:22} {userh_imp[k]:12.1f}")
    print(f"\n  per-fold h7 ΔD-A:")
    for k, m in by_fold_h7.items():
        print(f"    fold {k}: n={m['n']} A={m['A']:.4f} D={m['D']:.4f} Δ={m['dD_vs_A']:+.4f}")
    print(f"\n  Gates:")
    print(f"    A1 h7 Δ≥+0.005:          {A1}  ({h7_d:+.4f})")
    print(f"    A2 h7 rec>lost:          {A2}  ({h7_rec}>{h7_lost})")
    print(f"    B1 same-artist Δ≥-0.005: {B1}  ({sa_d:+.4f})")
    print(f"    B2 diff-artist Δ≥-0.005: {B2}  ({diff_d:+.4f})")
    print(f"    B3 overlap≥8/20:         {B3}  ({overlap_AD:.2f})")
    print(f"    userh feats used (>1%):  {userh_used}  ({userh_imp_frac*100:.2f}%)")
    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R400 Model B — per-user taste prior as ranker features, 5-fold OOF",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "verdict": verdict,
        "feature_stacks": {"A": FEAT_NAMES_R84, "D": FEAT_NAMES_D},
        "userh_features": USERH_FEATS,
        "pool_hit": {"base": hit_base, "n": n, "frac": hit_base / n},
        "gt_in_user_train_history": direct_hits,
        "cases_user_in_train": cases_in_train,
        "metrics": metrics,
        "metrics_by_history_bucket": nprior_buckets,
        "metrics_per_fold_h7": by_fold_h7,
        "recovery_D_vs_A": {"recovered_all": rec_all, "lost_all": lost_all,
                            "net_all": rec_all - lost_all,
                            "recovered_h7": h7_rec, "lost_h7": h7_lost,
                            "net_h7": h7_rec - h7_lost},
        "churn": {"overlap_AD": overlap_AD, "top1_churn_per_80": churn},
        "userh_feature_importance_gain": userh_imp,
        "userh_feature_importance_frac": userh_imp_frac,
        "all_feature_importance_gain": dict(all_imp_accum),
        "deltas": {"all": all_d, "h7": h7_d, "same_artist": sa_d, "diff_artist": diff_d},
        "gates": {"A1": [h7_d, A1], "A2": [[h7_rec, h7_lost], A2], "B1": [sa_d, B1],
                  "B2": [diff_d, B2], "B3": [overlap_AD, B3],
                  "userh_used": [userh_imp_frac, userh_used]},
    }

    def _ser(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return float(o)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=2, default=_ser)
    print(f"\n{ts()} Saved {out_json}")


if __name__ == "__main__":
    main()
