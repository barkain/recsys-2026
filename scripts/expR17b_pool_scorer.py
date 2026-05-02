#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R17b: Learned pool scorer — train a session→track model to rerank V3 pool.

Unlike R17 (full-catalog retrieval, failed to generalize), this scores only
the ~200 candidates already in the V3 pool. Training signal is dense:
for each case where GT is in pool, learn to score GT above other candidates.

Evaluation:
  1. Standalone reranker over V3 pool
  2. As LambdaRank feature (r17_score, r17_rank_inv)
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import math
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import lightgbm as lgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
QWEN_DIR = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"
R13_QUERY_DIR = REPO_ROOT / "cache" / "r13_query_emb"
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

FEATURE_NAMES_R17 = FEATURE_NAMES_V2 + ["r17_score", "r17_rank_inv"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def l2_norm(x, eps=1e-8):
    d = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(d, eps)


def _get_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F

class PoolScorer:
    """Wrapper that defers torch import to avoid implicit/OpenMP conflicts."""
    @staticmethod
    def create(emb_dim=1024, scalar_dim=5, hidden=256, out_dim=128):
        torch, nn, F = _get_torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                session_in = emb_dim * 3 + 4 + emb_dim
                cand_in = emb_dim + scalar_dim
                self.session_tower = nn.Sequential(
                    nn.Linear(session_in, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, out_dim),
                )
                self.cand_tower = nn.Sequential(
                    nn.Linear(cand_in, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden, out_dim),
                )
                self.temperature = nn.Parameter(torch.tensor(0.07, dtype=torch.float32).log())

            def forward(self, session_feats, cand_embs, cand_scalars):
                s = F.normalize(self.session_tower(session_feats), dim=-1)
                cand_in = torch.cat([cand_embs, cand_scalars], dim=-1)
                B, K, _ = cand_in.shape
                c = F.normalize(self.cand_tower(cand_in.view(B * K, -1)), dim=-1).view(B, K, -1)
                temp = self.temperature.exp().clamp(min=0.01, max=1.0)
                return (s.unsqueeze(1) * c).sum(dim=-1) / temp
        return _Model()


def load_data():
    """Load everything needed for pool scoring."""
    print(f"{ts()} Loading payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    print(f"{ts()} Loading Qwen catalog...", flush=True)
    track_ids = json.load(open(QWEN_DIR / "track_ids.json"))
    track_matrix = l2_norm(np.load(QWEN_DIR / "vectors.npy").astype(np.float32))
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    print(f"{ts()} Loading query embeddings...", flush=True)
    q_current = l2_norm(np.load(R13_QUERY_DIR / "emb_current.npy").astype(np.float32))
    q_context = l2_norm(np.load(R13_QUERY_DIR / "emb_context.npy").astype(np.float32))

    print(f"{ts()} Loading track scalars...", flush=True)
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    # Build track scalar features (pop, year, duration, tag_count, train_freq)
    n_scalars = 5
    track_scalars = np.zeros((len(track_ids), n_scalars), dtype=np.float32)
    from datasets import Dataset, concatenate_datasets
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            ds = Dataset.from_file(str(matches[-1]))
            cols = ds.to_dict()
            for i, tid in enumerate(cols["track_id"]):
                idx = track_to_idx.get(str(tid))
                if idx is None: continue
                track_scalars[idx, 0] = float(cols["popularity"][i] or 0) / 100.0
                try:
                    track_scalars[idx, 1] = (int(str(cols["release_date"][i])[:4]) - 1990) / 40.0
                except Exception:
                    pass
                track_scalars[idx, 2] = min(float(cols["duration"][i] or 0) / 600000, 2.0)
                track_scalars[idx, 3] = min(len(cols["tag_list"][i] or []), 50) / 50.0
    for tid, cnt in track_pop.items():
        idx = track_to_idx.get(tid)
        if idx is not None:
            track_scalars[idx, 4] = math.log1p(cnt) / math.log1p(max_pop)

    # Build ALS
    print(f"{ts()} Training ALS...", flush=True)
    from scripts.expS2_lambdarank import build_als
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx: scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    # Build V3 pools
    print(f"{ts()} Building V3 pools...", flush=True)
    pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        pools.append(weighted_rrf(sl, SOURCE_WEIGHTS, topk=200, k=RRF_K))

    # Build session features
    print(f"{ts()} Building session features...", flush=True)
    dim = track_matrix.shape[1]
    played_mean = np.zeros((n, dim), dtype=np.float32)
    last_track = np.zeros((n, dim), dtype=np.float32)
    session_scalars = np.zeros((n, 4), dtype=np.float32)

    for i, c in enumerate(cases):
        played = c["music_turns"]
        valid = [track_to_idx[t] for t in played if t in track_to_idx]
        if valid:
            recent = valid[-5:]
            m = len(recent)
            w = np.array([0.8 ** (m-1-j) for j in range(m)], dtype=np.float32)
            w /= w.sum()
            played_mean[i] = (track_matrix[recent] * w[:, None]).sum(0)
            last_track[i] = track_matrix[recent[-1]]
        session_scalars[i, 0] = min(len(played), 8) / 8.0
        session_scalars[i, 1] = 1.0 if len(played) == 0 else 0.0
        session_scalars[i, 2] = 1.0 if len(played) == 1 else 0.0
        session_scalars[i, 3] = min(len(str(c.get("user_query", "")).split()), 40) / 40.0

    return {
        "cases": cases, "sessions": sessions, "n": n,
        "pools": pools, "track_ids": track_ids, "track_matrix": track_matrix,
        "track_scalars": track_scalars, "track_to_idx": track_to_idx,
        "played_mean": played_mean, "last_track": last_track,
        "q_current": q_current, "q_context": q_context,
        "session_scalars": session_scalars,
        "als_source": als_source, "als_vecs": als_vecs,
        "als_factors": als_factors, "als_track_to_idx": als_track_to_idx,
        "track_pop": track_pop, "max_pop": max_pop,
        "payload": payload,
    }


def build_pool_tensors(data, case_indices, pool_k=200):
    """Build tensors for pool scoring: session features + candidate features + labels."""
    cases = data["cases"]
    pools = data["pools"]
    tm = data["track_matrix"]
    ts_arr = data["track_scalars"]
    t2i = data["track_to_idx"]
    dim = tm.shape[1]
    n_scalars = ts_arr.shape[1]

    n = len(case_indices)

    session_feats = np.zeros((n, dim * 3 + 4 + dim), dtype=np.float32)
    cand_embs = np.zeros((n, pool_k, dim), dtype=np.float32)
    cand_scalars_arr = np.zeros((n, pool_k, n_scalars), dtype=np.float32)
    labels = np.full(n, -1, dtype=np.int64)
    valid_mask = np.ones(n, dtype=bool)

    for j, ci in enumerate(case_indices):
        c = cases[ci]
        pool = pools[ci][:pool_k]

        # Session features
        session_feats[j, :dim] = data["played_mean"][ci]
        session_feats[j, dim:2*dim] = data["last_track"][ci]
        session_feats[j, 2*dim:3*dim] = data["q_current"][ci]
        session_feats[j, 3*dim:3*dim+4] = data["session_scalars"][ci]
        session_feats[j, 3*dim+4:] = data["q_context"][ci]

        # Candidate features
        gt = c["gt"]
        gt_in_pool = False
        for k, tid in enumerate(pool):
            idx = t2i.get(tid)
            if idx is not None:
                cand_embs[j, k] = tm[idx]
                cand_scalars_arr[j, k] = ts_arr[idx]
            if tid == gt:
                labels[j] = k
                gt_in_pool = True

        if not gt_in_pool:
            valid_mask[j] = False

    return session_feats, cand_embs, cand_scalars_arr, labels, valid_mask


def train_pool_scorer(data, train_idx, val_idx, pool_k=200, epochs=30, lr=1e-3,
                      batch_size=128, patience=5, device="cpu"):
    """Train the pool scorer model."""
    torch, nn, F = _get_torch()

    print(f"{ts()} Building train/val tensors...", flush=True)
    tr_sf, tr_ce, tr_cs, tr_labels, tr_mask = build_pool_tensors(data, train_idx, pool_k)
    va_sf, va_ce, va_cs, va_labels, va_mask = build_pool_tensors(data, val_idx, pool_k)

    tr_valid = np.where(tr_mask & (tr_labels >= 0))[0]
    va_valid = np.where(va_mask & (va_labels >= 0))[0]
    print(f"  Train: {len(tr_valid)}/{len(train_idx)} with GT in pool")
    print(f"  Val:   {len(va_valid)}/{len(val_idx)} with GT in pool")

    tr_sf_t = torch.from_numpy(tr_sf[tr_valid]).to(device)
    tr_ce_t = torch.from_numpy(tr_ce[tr_valid]).to(device)
    tr_cs_t = torch.from_numpy(tr_cs[tr_valid]).to(device)
    tr_lab_t = torch.from_numpy(tr_labels[tr_valid]).to(device)

    va_sf_t = torch.from_numpy(va_sf[va_valid]).to(device)
    va_ce_t = torch.from_numpy(va_ce[va_valid]).to(device)
    va_cs_t = torch.from_numpy(va_cs[va_valid]).to(device)
    va_lab_t = torch.from_numpy(va_labels[va_valid]).to(device)

    dim = data["track_matrix"].shape[1]
    model = PoolScorer.create(emb_dim=dim, scalar_dim=data["track_scalars"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_valid), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(tr_valid), batch_size):
            idx = perm[start:start+batch_size]
            logits = model(tr_sf_t[idx], tr_ce_t[idx], tr_cs_t[idx])  # (B, K)
            loss = torch.nn.functional.cross_entropy(logits, tr_lab_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = []
            for start in range(0, len(va_valid), batch_size):
                vl = model(va_sf_t[start:start+batch_size],
                           va_ce_t[start:start+batch_size],
                           va_cs_t[start:start+batch_size])
                val_logits.append(vl)
            val_logits = torch.cat(val_logits, dim=0)
            val_loss = torch.nn.functional.cross_entropy(val_logits, va_lab_t).item()

            # Val nDCG@20
            val_ranks = []
            for j in range(len(va_valid)):
                gt_pos = va_lab_t[j].item()
                sc = val_logits[j]
                gt_score = sc[gt_pos]
                rank = int((sc > gt_score).sum()) + int(((sc == gt_score) & (torch.arange(len(sc), device=device) < gt_pos)).sum())
                val_ranks.append(rank)
            val_ndcg = np.mean([1.0/np.log2(r+2) if r < 20 else 0.0 for r in val_ranks])

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_nDCG@20={val_ndcg:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model


def score_all_pools(model, data, case_indices, pool_k=200, batch_size=256, device="cpu"):
    """Score all pool candidates for given cases."""
    torch, _, _ = _get_torch()
    sf, ce, cs, labels, mask = build_pool_tensors(data, case_indices, pool_k)
    all_scores = np.zeros((len(case_indices), pool_k), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(case_indices), batch_size):
            end = min(start + batch_size, len(case_indices))
            logits = model(
                torch.from_numpy(sf[start:end]).to(device),
                torch.from_numpy(ce[start:end]).to(device),
                torch.from_numpy(cs[start:end]).to(device),
            )
            all_scores[start:end] = logits.cpu().numpy()

    return all_scores, labels


def main():
    t0 = time.time()
    device = "cpu"  # MPS can be flaky with this model size

    data = load_data()
    cases = data["cases"]
    n = data["n"]
    sessions = data["sessions"]
    ta = data["payload"]["track_artist"]
    tt = data["payload"]["track_tags"]
    track_pop = data["track_pop"]
    max_pop = data["max_pop"]
    pools = data["pools"]
    payload = data["payload"]

    # ===== OVERFIT SMOKE TEST =====
    print(f"\n{ts()} === OVERFIT SMOKE (200 in-pool cases) ===", flush=True)
    in_pool_cases = [i for i in range(n) if cases[i]["gt"] in pools[i]]
    smoke_idx = np.array(in_pool_cases[:200])
    smoke_model = train_pool_scorer(data, smoke_idx, smoke_idx, pool_k=200,
                                     epochs=50, lr=1e-3, batch_size=64, patience=50, device=device)
    smoke_scores, smoke_labels = score_all_pools(smoke_model, data, smoke_idx, pool_k=200, device=device)
    smoke_ranks = []
    for j in range(len(smoke_idx)):
        if smoke_labels[j] < 0: continue
        sc = smoke_scores[j]
        gt_score = sc[smoke_labels[j]]
        rank = int(np.sum(sc > gt_score))
        smoke_ranks.append(rank)
    print(f"  Overfit smoke: median_rank={np.median(smoke_ranks):.0f}  "
          f"hit@20={np.mean([r < 20 for r in smoke_ranks]):.1%}  "
          f"nDCG@20={np.mean([1/np.log2(r+2) if r < 20 else 0 for r in smoke_ranks]):.4f}")
    if np.median(smoke_ranks) > 5:
        print("  ⚠ Overfit smoke failed — model can't memorize pool. Check architecture.")
    del smoke_model

    # ===== GROUPED-SESSION CV5 =====
    print(f"\n{ts()} === GROUPED-SESSION CV5 ===", flush=True)

    seeds = [0, 1, 2]
    pool_k = 200

    # Collect per-case R17 scores across folds
    r17_scores_all = np.zeros((n, pool_k), dtype=np.float32)

    standalone_ndcgs = []
    standalone_lt_ndcgs = []
    v3_rrf_ndcgs = []
    v3_rrf_lt_ndcgs = []

    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        print(f"\n{ts()} Seed {seed}:", flush=True)

        for fold_i, fold in enumerate(folds):
            held = set(fold.tolist())
            train_cases = np.array([j for j in range(n) if j not in held])
            val_cases = fold.tolist()

            print(f"  Fold {fold_i}: train={len(train_cases)} val={len(val_cases)}", flush=True)

            model = train_pool_scorer(
                data, train_cases, val_cases, pool_k=pool_k,
                epochs=30, lr=1e-3, batch_size=128, patience=7, device=device)

            val_scores, val_labels = score_all_pools(model, data, val_cases, pool_k=pool_k, device=device)

            # Store scores for LambdaRank feature
            for j_local, j_global in enumerate(val_cases):
                r17_scores_all[j_global] = val_scores[j_local]

            # Standalone rerank evaluation
            for j_local, j_global in enumerate(val_cases):
                gt_pos = val_labels[j_local]
                sc = val_scores[j_local]
                if gt_pos >= 0:
                    gt_score = sc[gt_pos]
                    rank = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(pool_k) < gt_pos)))
                    ndcg = 1.0/np.log2(rank+2) if rank < 20 else 0.0
                else:
                    ndcg = 0.0
                standalone_ndcgs.append(ndcg)
                if len(cases[j_global]["music_turns"]) == 7:
                    standalone_lt_ndcgs.append(ndcg)

                # V3 RRF baseline (pool order = rank order)
                pool = pools[j_global]
                gt = cases[j_global]["gt"]
                if gt in pool:
                    rrf_rank = pool.index(gt)
                    rrf_ndcg = 1.0/np.log2(rrf_rank+2) if rrf_rank < 20 else 0.0
                else:
                    rrf_ndcg = 0.0
                v3_rrf_ndcgs.append(rrf_ndcg)
                if len(cases[j_global]["music_turns"]) == 7:
                    v3_rrf_lt_ndcgs.append(rrf_ndcg)

            del model

    print(f"\n{ts()} === STANDALONE RERANK RESULTS ===")
    print(f"  R17b pool rerank:  CV={np.mean(standalone_ndcgs):.4f}  last_turn={np.mean(standalone_lt_ndcgs):.4f}")
    print(f"  V3 RRF ordering:   CV={np.mean(v3_rrf_ndcgs):.4f}  last_turn={np.mean(v3_rrf_lt_ndcgs):.4f}")

    # ===== LAMBDARANK WITH R17 FEATURES =====
    print(f"\n{ts()} === LAMBDARANK + R17 FEATURES ===", flush=True)

    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    als_source = data["als_source"]
    als_vecs = data["als_vecs"]
    als_factors = data["als_factors"]
    als_track_to_idx = data["als_track_to_idx"]

    # Build R17 rank features from scores
    r17_rank_inv = np.zeros((n, pool_k), dtype=np.float32)
    for i in range(n):
        sc = r17_scores_all[i]
        order = np.argsort(-sc)
        for r, idx in enumerate(order):
            r17_rank_inv[i, idx] = 1.0 / (r + 1)

    n_feat_v2 = len(FEATURE_NAMES_V2)
    n_feat_r17 = len(FEATURE_NAMES_R17)

    # Build feature matrices for both V3 baseline and V3+R17
    for config_name, use_r17 in [("v3_baseline", False), ("v3+r17_features", True)]:
        nf = n_feat_r17 if use_r17 else n_feat_v2
        feat_names = FEATURE_NAMES_R17 if use_r17 else FEATURE_NAMES_V2

        print(f"\n{ts()} {config_name}: building {nf}-feature matrix...", flush=True)
        X = np.zeros((n, pool_k, nf), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            pool = pools[i][:pool_k]
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])

            src_lists = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                         "C": payload["src_c"][i], "D": payload["src_d"][i],
                         "F": payload["src_f"][i], "ALS": als_source[i]}
            src_rank = {sn: {tid: r+1 for r, tid in enumerate(sl)} for sn, sl in src_lists.items()}
            user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                         + [c["user_query"]])
            played = c["music_turns"]
            n_hist = len(played)
            now_tok = tokens(user_msgs[-1]) if user_msgs else set()
            all_tok = tokens(" ".join(user_msgs))
            played_set = set(played)
            l_artist = ta.get(played[-1], "") if played else ""
            l_tags = tt.get(played[-1], set()) if played else set()
            prior = [(1.0/(j+1), ta.get(t,""), tt.get(t,set())) for j, t in enumerate(reversed(played))]
            sv = als_vecs[i]
            pool_artists = [ta.get(tid, "") for tid in pool]
            artist_counts = Counter(a for a in pool_artists if a)

            for rank, tid in enumerate(pool, start=1):
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
                for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                    sr = src_rank[sn].get(tid)
                    row[8+fi] = 1.0/sr if sr else 0.0
                for fi, sn in enumerate(["A","B","C","D","F","ALS"]):
                    row[14+fi] = 1.0 if tid in src_rank[sn] else 0.0
                row[20] = sum(1 for sn in src_lists if tid in src_rank[sn])
                if sv is not None:
                    aidx = als_track_to_idx.get(tid)
                    if aidx is not None: row[21] = float(np.dot(sv, als_factors[aidx]))
                row[22] = float(n_hist)
                pop = track_pop.get(tid, 0)
                row[23] = pop / max_pop
                row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
                row[25] = float(artist_counts.get(ca, 0)) if ca else 0
                row[26] = row[20]

                if use_r17:
                    row[27] = float(r17_scores_all[i, rank-1])
                    row[28] = float(r17_rank_inv[i, rank-1])

        # LambdaRank CV5
        lgb_params = {
            "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
            "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
            "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
            "random_state": 42, "force_col_wise": True,
        }

        X_flat = X.reshape(-1, nf)
        labels = np.zeros(n * pool_k, dtype=np.float32)
        for i in range(n):
            if gt_idx[i] >= 0:
                labels[i * pool_k + gt_idx[i]] = 1.0

        cv5_seeds, lt_seeds = [], []
        sa_vals, da_vals, h0_vals, p0_vals = [], [], [], []

        for seed in seeds:
            folds = grouped_session_folds(sessions, seed)
            fold_ndcgs = []
            for fold in folds:
                held = set(fold.tolist())
                train_c = [j for j in range(n) if j not in held]
                val_c = fold.tolist()
                train_flat = [j*pool_k+k for j in train_c for k in range(int(sizes[j]))]
                val_flat = [j*pool_k+k for j in val_c for k in range(int(sizes[j]))]
                g_train = np.array([int(sizes[j]) for j in train_c], dtype=np.int32)
                g_val = np.array([int(sizes[j]) for j in val_c], dtype=np.int32)
                dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat],
                                     group=g_train, feature_name=feat_names, free_raw_data=False)
                dval = lgb.Dataset(X_flat[val_flat], labels[val_flat],
                                   group=g_val, reference=dtrain, free_raw_data=False)
                mdl = lgb.train(lgb_params, dtrain, num_boost_round=300,
                                valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
                val_scores_lgb = mdl.predict(X_flat[val_flat])
                offset = 0
                case_ndcgs = []
                for j in val_c:
                    sz = int(sizes[j])
                    if sz == 0: case_ndcgs.append(0.0); continue
                    sc = val_scores_lgb[offset:offset+sz]
                    gt = gt_idx[j]
                    if gt >= 0:
                        gt_score = sc[gt]
                        rank0 = int(np.sum(sc > gt_score) + np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                        ndcg = 1.0/np.log2(rank0+2) if rank0 < 20 else 0.0
                    else:
                        ndcg = 0.0
                    case_ndcgs.append(ndcg)
                    played = cases[j]["music_turns"]
                    gt_tid = cases[j]["gt"]
                    if len(played) == 7: lt_seeds.append(ndcg)
                    if played:
                        la = ta.get(played[-1], "")
                        ga = ta.get(gt_tid, "")
                        if isinstance(la, list): la = la[0] if la else ""
                        if isinstance(ga, list): ga = ga[0] if ga else ""
                        if ga and la:
                            if ga == la: sa_vals.append(ndcg)
                            else: da_vals.append(ndcg)
                    if len(played) == 0: h0_vals.append(ndcg)
                    if track_pop.get(gt_tid, 0) == 0: p0_vals.append(ndcg)
                    offset += sz
                fold_ndcgs.append(float(np.mean(case_ndcgs)))
            cv5_seeds.append(float(np.mean(fold_ndcgs)))

        cv5 = float(np.mean(cv5_seeds))
        lt = float(np.mean(lt_seeds)) if lt_seeds else 0
        sa = float(np.mean(sa_vals)) if sa_vals else 0
        da = float(np.mean(da_vals)) if da_vals else 0
        h0 = float(np.mean(h0_vals)) if h0_vals else 0
        p0 = float(np.mean(p0_vals)) if p0_vals else 0

        print(f"  {config_name}: CV5={cv5:.4f}  last_turn={lt:.4f}")
        print(f"    same_artist={sa:.4f}  diff_artist={da:.4f}  hist_0={h0:.4f}  pop_0={p0:.4f}")

        # Reset slice accumulators
        lt_seeds, sa_vals, da_vals, h0_vals, p0_vals = [], [], [], [], []

    elapsed = time.time() - t0
    print(f"\n{ts()} R17b complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
