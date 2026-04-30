#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Evaluate S2 GRU model on B1 8000-case benchmark.

Standalone metrics, unique hits vs ABCDF@200, fusion sweep with CV5.
No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.r3_confirm_400_deterministic import cv_folds
from scripts.train_s2_gru import S2GRU
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
UTT_NPY = REPO_ROOT / "cache" / "seq_model" / "utt_embeddings.npy"
UTT_INDEX = REPO_ROOT / "cache" / "seq_model" / "utt_embedding_index.json"
POOL_K = 50
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = S2GRU(
        n_tracks=config["n_tracks"],
        emb_dim=config.get("emb_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 2),
        dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def retrieve_all_cases(model, cases, track_to_idx, track_ids, utt_embs,
                       utt_index, device, topk=200):
    """Retrieve top-k tracks for each eval case using S2 model."""
    n_tracks = model.n_tracks
    item_emb = F.normalize(model.get_item_embeddings().to(device), dim=-1)
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    no_history_id = n_tracks

    results = []
    for i, c in enumerate(cases):
        sid = c["session_id"]
        turn = c["turn_number"]
        played = c["music_turns"]

        # Build track sequence
        seq = [track_to_idx[t] for t in played if t in track_to_idx]
        if not seq:
            seq = [no_history_id]
        seq_t = torch.tensor([seq], dtype=torch.long, device=device)
        seq_len = torch.tensor([len(seq)], dtype=torch.long, device=device)

        # Utterance
        utt_key = f"{sid}:{turn}"
        utt_idx = utt_index.get(utt_key)
        if utt_idx is not None and utt_idx < len(utt_embs):
            utt = utt_embs[utt_idx]
        else:
            utt = np.zeros(384, dtype=np.float32)
        utt_t = torch.from_numpy(utt).unsqueeze(0).to(device)

        # Forward
        output = model(seq_t, utt_t, seq_len)  # (1, D)
        output_norm = F.normalize(output, dim=-1)
        scores = (output_norm @ item_emb.T).squeeze(0) * logit_scale  # (n_tracks,)

        # Exclude played tracks
        played_idx = [track_to_idx[t] for t in played if t in track_to_idx]
        if played_idx:
            scores[played_idx] = -torch.inf

        # Top-k
        k = min(topk, n_tracks)
        _, top_idx = scores.topk(k)
        top_list = [track_ids[j] for j in top_idx.cpu().tolist()]
        results.append(top_list)

        if (i + 1) % 2000 == 0:
            print(f"  {ts()} {i+1}/{len(cases)}", flush=True)

    return results


def eval_standalone(s2_results, cases, abcdf_pools):
    """Standalone metrics and unique hits."""
    n = len(cases)
    hit20 = hit50 = hit200 = 0
    unique_hits = 0
    unique_ranks = []
    gt_ranks = []

    hist_buckets = defaultdict(lambda: {"n": 0, "hit20": 0, "hit200": 0, "unique": 0})

    for i, c in enumerate(cases):
        gt = c["gt"]
        n_hist = len(c["music_turns"])
        bk = f"hist_{min(n_hist, 7)}"
        hist_buckets[bk]["n"] += 1

        top_list = s2_results[i]
        top_set = set(top_list[:200])

        if gt in top_list[:20]:
            hit20 += 1
            hist_buckets[bk]["hit20"] += 1
        if gt in top_list[:50]:
            hit50 += 1
        if gt in top_set:
            hit200 += 1
            hist_buckets[bk]["hit200"] += 1
            rank = top_list.index(gt) + 1
            gt_ranks.append(rank)
            if gt not in abcdf_pools[i]:
                unique_hits += 1
                unique_ranks.append(rank)
                hist_buckets[bk]["unique"] += 1

    return {
        "n": n,
        "hit20": hit20, "hit50": hit50, "hit200": hit200,
        "hit20_rate": hit20 / n, "hit50_rate": hit50 / n, "hit200_rate": hit200 / n,
        "unique_hits": unique_hits,
        "unique_median_rank": float(np.median(unique_ranks)) if unique_ranks else None,
        "gt_median_rank": float(np.median(gt_ranks)) if gt_ranks else None,
        "hist_buckets": {k: dict(v) for k, v in sorted(hist_buckets.items())},
    }


# ============ Fusion + CV5 ============

def vec_ndcg(X, gt_idx, sizes, weights, idx):
    pool_axis = np.arange(X.shape[1])[None, :]
    valid_pool = pool_axis < sizes[idx, None]
    scores = X[idx] @ weights
    scores = np.where(valid_pool, scores, -np.inf)
    gt = gt_idx[idx]
    has_gt = gt >= 0
    safe_gt = np.where(has_gt, gt, 0)
    gt_scores = scores[np.arange(len(idx)), safe_gt]
    strict_gt = (scores > gt_scores[:, None]).sum(axis=1)
    tie_before = ((scores == gt_scores[:, None]) & valid_pool
                  & (pool_axis < safe_gt[:, None])).sum(axis=1)
    rank0 = strict_gt + tie_before
    vals = np.where(has_gt & (rank0 < 20), 1.0 / np.log2(rank0 + 2), 0.0)
    return float(vals.mean())


def fit_powell(X, gt_idx, sizes, train_idx):
    init = np.array([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    def objective(w):
        return -vec_ndcg(X, gt_idx, sizes, w, train_idx)
    res = minimize(objective, init, method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500})
    return res.x, -float(res.fun)


def eval_cv5(X, gt_idx, sizes, sessions, seeds):
    n = len(sessions)
    per_seed = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_powell(X, gt_idx, sizes, train)
            fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
        per_seed.append(float(np.mean(fold_sc)))
    return per_seed


def build_features_with_s2(payload, s2_results, source_weights, pool_k=50, rrf_k=20):
    """Build Powell features with S2 as an additional source."""
    cases = payload["cases"]
    n = len(cases)
    X = np.zeros((n, pool_k, len(FEATURE_NAMES)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        sources = {}
        if source_weights.get("A", 0) > 0:
            sources["A"] = payload["src_a"][i]
        if source_weights.get("B", 0) > 0:
            sources["B"] = payload["src_b"][i]
        if source_weights.get("C", 0) > 0:
            sources["C"] = payload["src_c"][i]
        if source_weights.get("D", 0) > 0:
            sources["D"] = payload["src_d"][i]
        if source_weights.get("F", 0) > 0:
            sources["F"] = payload["src_f"][i]
        if source_weights.get("S2", 0) > 0:
            sources["S2"] = s2_results[i]

        pool = weighted_rrf(sources, source_weights, topk=pool_k, k=rrf_k)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:pool_k], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

    return X, gt_idx, sizes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to S2 training run directory")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint filename (default: best.pt)")
    args = parser.parse_args()

    t0 = time.time()
    run_dir = Path(args.run_dir)
    device = torch.device("cpu")  # CPU for deterministic eval

    # --- Load model ---
    ckpt_path = run_dir / args.checkpoint
    print(f"{ts()} Loading model from {ckpt_path}", flush=True)
    model, config = load_model(ckpt_path, device)
    n_tracks = config["n_tracks"]

    # Load track vocabulary
    with open(run_dir / "track_ids.json") as f:
        track_ids = json.load(f)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    # --- Load eval payload ---
    print(f"{ts()} Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    # Check GT coverage
    gt_in_vocab = sum(1 for c in cases if c["gt"] in track_to_idx)
    print(f"  {n} cases, GT coverage: {gt_in_vocab}/{n} ({gt_in_vocab/n:.1%})")

    # --- Load utterance cache ---
    utt_embs = np.load(UTT_NPY)
    with open(UTT_INDEX) as f:
        utt_index = json.load(f)

    # --- Build ABCDF@200 pools ---
    print(f"{ts()} Building ABCDF@200 pools...", flush=True)
    abcdf_pools = []
    for i in range(n):
        sources = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i],
        }
        weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources, weights, topk=200, k=RRF_K)
        abcdf_pools.append(set(pool))

    # --- Retrieve ---
    print(f"{ts()} Retrieving top-200 for {n} cases...", flush=True)
    s2_results = retrieve_all_cases(
        model, cases, track_to_idx, track_ids,
        utt_embs, utt_index, device, topk=200,
    )

    # --- Standalone eval ---
    standalone = eval_standalone(s2_results, cases, abcdf_pools)
    print(f"\n{ts()} STANDALONE METRICS")
    print(f"  hit@20:  {standalone['hit20']}/{n} ({standalone['hit20_rate']:.1%})")
    print(f"  hit@50:  {standalone['hit50']}/{n} ({standalone['hit50_rate']:.1%})")
    print(f"  hit@200: {standalone['hit200']}/{n} ({standalone['hit200_rate']:.1%})")
    print(f"  unique hits vs ABCDF@200: {standalone['unique_hits']}")
    if standalone["unique_median_rank"]:
        print(f"  unique median rank: {standalone['unique_median_rank']:.0f}")
    if standalone["gt_median_rank"]:
        print(f"  GT median rank: {standalone['gt_median_rank']:.0f}")

    print(f"\n  {'bucket':10s} {'n':>5s} {'hit@20':>7s} {'hit@200':>8s} {'unique':>7s}")
    for bk, d in sorted(standalone["hist_buckets"].items()):
        bn = d["n"]
        print(f"  {bk:10s} {bn:5d} {d['hit20']/bn:7.1%} {d['hit200']/bn:8.1%} {d['unique']:7d}")

    # --- Gate check ---
    unique = standalone["unique_hits"]
    if unique < 200:
        print(f"\n  GATE FAIL: unique_hits ({unique}) < 200. Stopping.")
        elapsed = time.time() - t0
        print(f"\n{ts()} Elapsed: {elapsed:.1f}s")
        return

    # --- Fusion sweep ---
    print(f"\n{ts()} Running fusion sweep...", flush=True)
    seeds = [0, 1, 2, 3, 4]

    # Baseline ABCDF
    from scripts.expA1_ablation_cv5 import build_features as build_base_features
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
    X_base, gt_base, sizes_base = build_base_features(payload, base_weights)
    cv5_base = eval_cv5(X_base, gt_base, sizes_base, sessions, seeds)
    cv5_base_mean = float(np.mean(cv5_base))
    pool_hit_base = float(np.mean(gt_base >= 0))
    print(f"  Baseline ABCDF: CV5={cv5_base_mean:.4f}, pool_hit@50={pool_hit_base:.4f}")

    fusion_results = {}
    for w_s2 in [0.25, 0.5, 1.0, 2.0]:
        fname = f"w_S2={w_s2}"
        print(f"\n  {fname}...", flush=True)
        fw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "S2": w_s2}
        X_f, gt_f, sizes_f = build_features_with_s2(payload, s2_results, fw,
                                                     pool_k=POOL_K, rrf_k=RRF_K)
        pool_hit_f = float(np.mean(gt_f >= 0))
        cv5_f = eval_cv5(X_f, gt_f, sizes_f, sessions, seeds)
        cv5_f_mean = float(np.mean(cv5_f))
        cv5_f_std = float(np.std(cv5_f, ddof=1))
        delta = cv5_f_mean - cv5_base_mean
        print(f"    pool_hit@50={pool_hit_f:.4f} (Δ={pool_hit_f - pool_hit_base:+.4f})")
        print(f"    CV5={cv5_f_mean:.4f} ± {cv5_f_std:.4f} (Δ={delta:+.4f})")
        fusion_results[fname] = {
            "w_s2": w_s2, "pool_hit50": pool_hit_f,
            "cv5": cv5_f_mean, "cv5_std": cv5_f_std,
            "delta_cv5": delta, "delta_pool_hit": pool_hit_f - pool_hit_base,
        }

    # --- Final report ---
    best_fusion = max(fusion_results.values(), key=lambda x: x["cv5"])
    cv5_lift = best_fusion["cv5"] - cv5_base_mean

    print(f"\n{ts()} {'='*60}")
    print(f"FINAL REPORT")
    print(f"  Unique hits vs ABCDF@200: {unique}")
    print(f"  Best fusion CV5: {best_fusion['cv5']:.4f} (Δ={cv5_lift:+.4f})")
    print(f"  GATE unique >= 500: {'PASS' if unique >= 500 else 'FAIL'} ({unique})")
    print(f"  GATE CV5 lift >= +0.015: {'PASS' if cv5_lift >= 0.015 else 'FAIL'} ({cv5_lift:+.4f})")

    elapsed = time.time() - t0
    print(f"\n{ts()} Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expS2_neural_candidate.json"
    out = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "standalone": standalone,
        "baseline_cv5": cv5_base_mean,
        "fusion_results": fusion_results,
        "best_fusion_cv5": best_fusion["cv5"],
        "cv5_lift": cv5_lift,
        "gate_unique_500": unique >= 500,
        "gate_cv5_015": cv5_lift >= 0.015,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
