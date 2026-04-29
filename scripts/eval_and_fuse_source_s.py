#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Evaluate best Source S checkpoint + ABCDF+S fusion sweep.

Part 1: Epoch 8 comprehensive eval (standalone + unique hits + overlap + hist breakdown)
Part 2: ABCDF+S fusion sweep (S_depth × w_S grid, CV5 with Powell)
"""
import json, math, pickle, sys, time
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(".").resolve()))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from datasets import Dataset
from mcrs.retrieval_modules.seq_model import SequenceRecommender
from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.r3_confirm_400_deterministic import cv_folds
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens

RUN_DIR = Path("cache/seq_model/runs/20260428_200502_fdddfe")
BEST_CKPT = RUN_DIR / "epoch_8.pt"
UTT_NPY = "cache/seq_model/utt_embeddings.npy"
UTT_INDEX = "cache/seq_model/utt_embedding_index.json"
TRACK_IDS_PATH = "cache/track_sim/metadata-qwen3_embedding_0.6b/track_ids.json"
TRACK_VECS_PATH = "cache/track_sim/metadata-qwen3_embedding_0.6b/vectors.npy"
R12_CACHE = "exp/eval/_R12_all_turns_payload.pkl"
RRF_K = 20
POOL_K = 50


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
    tie_before = ((scores == gt_scores[:, None]) & valid_pool & (pool_axis < safe_gt[:, None])).sum(axis=1)
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


def main():
    t0 = time.time()
    device = torch.device("cpu")

    print(f"[{datetime.now():%H:%M:%S}] Loading resources...", flush=True)
    utt_embs = np.load(UTT_NPY)
    utt_index = json.load(open(UTT_INDEX))
    catalog_ids = json.load(open(TRACK_IDS_PATH))
    catalog_matrix = np.load(TRACK_VECS_PATH)
    catalog_t = torch.from_numpy(catalog_matrix).float()
    track_emb_dict = {tid: catalog_matrix[i] for i, tid in enumerate(catalog_ids)}
    catalog_id_to_idx = {tid: i for i, tid in enumerate(catalog_ids)}

    # Load model
    print(f"[{datetime.now():%H:%M:%S}] Loading epoch 8 checkpoint...", flush=True)
    checkpoint = torch.load(BEST_CKPT, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    model = SequenceRecommender(
        track_emb_dim=config.get("track_emb_dim", 1024),
        utt_emb_dim=config.get("utt_emb_dim", 384),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 4),
        output_dim=config.get("output_dim", 1024),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load R12 payload
    print(f"[{datetime.now():%H:%M:%S}] Loading R12 payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    def get_utt(sid, turn):
        key = f"{sid}:{turn}"
        idx = utt_index.get(key)
        if idx is not None and idx < len(utt_embs):
            return utt_embs[idx]
        return np.zeros(384, dtype=np.float32)

    # =====================================================================
    # PART 1: Compute S top-200 for all 8000 cases
    # =====================================================================
    print(f"\n[{datetime.now():%H:%M:%S}] {'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] PART 1: Source S epoch 8 — full eval")
    print(f"[{datetime.now():%H:%M:%S}] {'='*60}")

    s_top200_all = []  # list of lists, per case
    s_ranks = []  # GT rank in S top-200, or None

    with torch.no_grad():
        for i, c in enumerate(cases):
            history = [(int(h["turn_number"]), str(h["content"]).strip())
                       for h in c["history"] if h["role"] == "music"]
            if not history:
                s_top200_all.append([])
                s_ranks.append(None)
                if (i+1) % 2000 == 0:
                    print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{n}", flush=True)
                continue

            T_hist = len(history)
            T = T_hist + 1
            te = np.zeros((T_hist, 1024), dtype=np.float32)
            ue = np.zeros((T, 384), dtype=np.float32)
            for j, (t, tid) in enumerate(history):
                emb = track_emb_dict.get(tid)
                if emb is not None: te[j] = emb
                ue[j] = get_utt(c["session_id"], t)
            ue[T_hist] = get_utt(c["session_id"], c["turn_number"])
            ac = np.ones(T_hist, dtype=np.int64)
            ti = np.zeros(T, dtype=np.int64)
            for j, (t, _) in enumerate(history): ti[j] = min(t-1, 7)
            ti[T_hist] = min(c["turn_number"]-1, 7)

            target = model(
                torch.from_numpy(te).unsqueeze(0), torch.from_numpy(ue).unsqueeze(0),
                torch.from_numpy(ac).unsqueeze(0), torch.from_numpy(ti).unsqueeze(0),
                torch.tensor([T]),
            )
            scores = (target @ catalog_t.T).squeeze(0).numpy()
            played_set = {tid for _, tid in history}
            for tid in played_set:
                idx = catalog_id_to_idx.get(tid)
                if idx is not None: scores[idx] = -np.inf

            top200_idx = np.argpartition(-scores, 200)[:200]
            top200_idx = top200_idx[np.argsort(-scores[top200_idx])]
            s_top200 = [catalog_ids[j] for j in top200_idx]
            s_top200_all.append(s_top200)

            gt = c["gt"]
            rank = None
            for r, tid in enumerate(s_top200):
                if tid == gt:
                    rank = r + 1
                    break
            s_ranks.append(rank)

            if (i+1) % 2000 == 0:
                print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{n}", flush=True)

    # Build pipeline pools
    print(f"[{datetime.now():%H:%M:%S}] Building pipeline pools...", flush=True)
    pipe_pools_200 = []
    pipe_a_200 = []
    pipe_d_200 = []
    pipe_f_200 = []
    for i in range(n):
        sources = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                    "C": payload["src_c"][i], "D": payload["src_d"][i],
                    "F": payload["src_f"][i]}
        weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources, weights, topk=200, k=RRF_K)
        pipe_pools_200.append(set(pool))
        pipe_a_200.append(set(payload["src_a"][i][:200]))
        pipe_d_200.append(set(payload["src_d"][i][:200]))
        pipe_f_200.append(set(payload["src_f"][i][:200]))

    # Compute all Part 1 metrics
    s_hit_20 = sum(1 for r in s_ranks if r is not None and r <= 20)
    s_hit_50 = sum(1 for r in s_ranks if r is not None and r <= 50)
    s_hit_200 = sum(1 for r in s_ranks if r is not None)
    s_unique = 0
    s_unique_ranks = []
    overlap_a = []
    overlap_d = []
    overlap_f = []
    collapse_count = 0
    n_with_hist = 0
    hist_data = defaultdict(lambda: {"n": 0, "hit20": 0, "hit50": 0, "hit200": 0, "unique": 0})

    for i, c in enumerate(cases):
        gt = c["gt"]
        s200 = set(s_top200_all[i])
        history = [(int(h["turn_number"]), str(h["content"]).strip())
                   for h in c["history"] if h["role"] == "music"]
        n_hist = len(history)
        bk = f"hist_{min(n_hist, 6)}" if n_hist <= 6 else "hist_7"

        hist_data[bk]["n"] += 1
        if s_ranks[i] is not None and s_ranks[i] <= 20: hist_data[bk]["hit20"] += 1
        if s_ranks[i] is not None and s_ranks[i] <= 50: hist_data[bk]["hit50"] += 1
        if s_ranks[i] is not None: hist_data[bk]["hit200"] += 1

        if gt in s200 and gt not in pipe_pools_200[i]:
            s_unique += 1
            hist_data[bk]["unique"] += 1
            if s_ranks[i]: s_unique_ranks.append(s_ranks[i])

        if s200:
            overlap_a.append(len(s200 & pipe_a_200[i]) / len(s200))
            overlap_d.append(len(s200 & pipe_d_200[i]) / len(s200))
            overlap_f.append(len(s200 & pipe_f_200[i]) / len(s200))

        if history:
            n_with_hist += 1
            if s_top200_all[i] and s_top200_all[i][0] == history[-1][1]:
                collapse_count += 1

    print(f"\n  Standalone metrics:")
    print(f"    nDCG@20:    (see val: 0.0638)")
    print(f"    hit@20:     {s_hit_20}/{n} ({s_hit_20/n:.1%})")
    print(f"    hit@50:     {s_hit_50}/{n} ({s_hit_50/n:.1%})")
    print(f"    hit@200:    {s_hit_200}/{n} ({s_hit_200/n:.1%})")
    print(f"    collapse:   {collapse_count}/{n_with_hist} ({collapse_count/n_with_hist:.1%})")

    print(f"\n  Unique hits vs ABCDF@200:")
    print(f"    S unique:   {s_unique}")
    if s_unique_ranks:
        print(f"    Median rank: {np.median(s_unique_ranks):.0f}")
        print(f"    P75 rank:    {np.percentile(s_unique_ranks, 75):.0f}")

    print(f"\n  Overlap (mean % of S top-200):")
    print(f"    A' (qwen3): {np.mean(overlap_a):.3f}")
    print(f"    D (nbrs):   {np.mean(overlap_d):.3f}")
    print(f"    F (CF-BPR): {np.mean(overlap_f):.3f}")

    print(f"\n  History-depth breakdown:")
    print(f"    {'bucket':8s} {'n':>5s} {'hit@20':>7s} {'hit@50':>7s} {'hit@200':>8s} {'unique':>7s}")
    for bk in sorted(hist_data.keys()):
        d = hist_data[bk]
        bn = d["n"]
        if bn == 0: continue
        print(f"    {bk:8s} {bn:5d} {d['hit20']/bn:7.1%} {d['hit50']/bn:7.1%} "
              f"{d['hit200']/bn:8.1%} {d['unique']:7d}")

    # =====================================================================
    # PART 2: ABCDF+S fusion sweep
    # =====================================================================
    print(f"\n[{datetime.now():%H:%M:%S}] {'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] PART 2: ABCDF+S fusion sweep")
    print(f"[{datetime.now():%H:%M:%S}] {'='*60}")

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    seeds = [0, 1, 2, 3, 4]

    def build_features_with_s(s_lists, source_weights):
        X = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        n_s_unique_in_pool = 0

        for i, c in enumerate(cases):
            sources = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                        "C": payload["src_c"][i], "D": payload["src_d"][i],
                        "F": payload["src_f"][i], "S": s_lists[i]}
            pool = weighted_rrf(sources, source_weights, topk=POOL_K, k=RRF_K)
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])
                # Check if GT was S-unique
                pipe_pool = pipe_pools_200[i]
                if c["gt"] not in pipe_pool and c["gt"] in set(s_lists[i]):
                    n_s_unique_in_pool += 1

            user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
            played = c["music_turns"]
            now_tok = tokens(user_msgs[-1]) if user_msgs else set()
            all_tok = tokens(" ".join(user_msgs))
            played_set = set(played)
            l_artist = ta.get(played[-1], "") if played else ""
            l_tags = tt.get(played[-1], set()) if played else set()
            prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j,t in enumerate(reversed(played))]
            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank-1]
                row[0] = 1.0/rank
                row[1] = 1.0 if ca and ca == l_artist else 0.0
                if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
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

        return X, gt_idx, sizes, n_s_unique_in_pool

    # Baseline: ABCDF (no S)
    print(f"\n[{datetime.now():%H:%M:%S}] Building ABCDF baseline...", flush=True)
    bl_w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "S": 0.0}
    bl_X, bl_gt, bl_sz, _ = build_features_with_s([[] for _ in cases], bl_w)
    bl_hit = float(np.mean(bl_gt >= 0))
    bl_cv5_seeds = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_powell(bl_X, bl_gt, bl_sz, train_idx)
            fold_sc.append(vec_ndcg(bl_X, bl_gt, bl_sz, w, fold))
        bl_cv5_seeds.append(float(np.mean(fold_sc)))
    bl_cv5 = float(np.mean(bl_cv5_seeds))
    print(f"  ABCDF baseline: pool_hit@50={bl_hit:.4f}  CV5={bl_cv5:.4f}")

    # Fusion grid
    s_depths = [50, 100, 200]
    s_weights = [0.25, 0.5, 1.0, 1.5, 2.0]

    print(f"\n  {'depth':>5s} {'w_S':>5s} {'pool_hit':>9s} {'Δhit':>6s} {'CV5':>7s} {'ΔCV5':>7s} {'S_uniq_in':>10s}")
    best_cv5 = bl_cv5
    best_config = None

    for depth in s_depths:
        s_truncated = [sl[:depth] for sl in s_top200_all]
        for w_s in s_weights:
            sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "S": w_s}
            X, gt_idx, sizes, n_s_uniq = build_features_with_s(s_truncated, sw)
            pool_hit = float(np.mean(gt_idx >= 0))

            cv5_seeds = []
            for seed in seeds:
                folds = cv_folds(sessions, seed)
                fold_sc = []
                for fold in folds:
                    held = set(fold.tolist())
                    train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
                    w, _ = fit_powell(X, gt_idx, sizes, train_idx)
                    fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
                cv5_seeds.append(float(np.mean(fold_sc)))
            cv5 = float(np.mean(cv5_seeds))

            d_hit = pool_hit - bl_hit
            d_cv5 = cv5 - bl_cv5
            print(f"  {depth:5d} {w_s:5.2f} {pool_hit:9.4f} {d_hit:+6.4f} {cv5:7.4f} {d_cv5:+7.4f} {n_s_uniq:10d}")

            if cv5 > best_cv5:
                best_cv5 = cv5
                best_config = {"depth": depth, "w_s": w_s, "cv5": cv5, "pool_hit": pool_hit,
                               "s_unique_admitted": n_s_uniq}

    # Summary
    print(f"\n[{datetime.now():%H:%M:%S}] {'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] SUMMARY")
    print(f"[{datetime.now():%H:%M:%S}] {'='*60}")
    print(f"  ABCDF baseline:  CV5={bl_cv5:.4f}  pool_hit={bl_hit:.4f}")
    if best_config:
        print(f"  Best ABCDF+S:    CV5={best_config['cv5']:.4f} (Δ={best_config['cv5']-bl_cv5:+.4f})  "
              f"depth={best_config['depth']} w_S={best_config['w_s']}")
        print(f"                   pool_hit={best_config['pool_hit']:.4f}  S_unique_admitted={best_config['s_unique_admitted']}")
        delta = best_config['cv5'] - bl_cv5
    else:
        print(f"  No config beat baseline")
        delta = 0

    if delta >= 0.010:
        verdict = "STRONG — submission candidate"
    elif delta >= 0.005:
        verdict = "PASS — prepare blind driver"
    elif delta > 0:
        verdict = "WEAK — tune fusion/ranker"
    else:
        verdict = "FAIL — S does not improve fusion"

    print(f"\n  GATE: {verdict}")
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    out_path = Path("exp/eval/eval_source_s_fusion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "epoch": 8, "val_ndcg": 0.0638,
            "s_unique_hits": s_unique,
            "baseline_cv5": bl_cv5, "baseline_pool_hit": bl_hit,
            "best_config": best_config,
            "verdict": verdict,
        }, f, indent=2)
    print(f"  Artifact: {out_path}")


if __name__ == "__main__":
    main()
