#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R33c-clean: Neural MLP pool reranker with clean fold-0 protocol.

Fixes from exploratory R33c:
- Clean fold-0 R21 encoder for query + track embeddings
- Batched training [B, pool_size, features] with CE loss
- Proper inference mode via submodule iteration
- Normalized interaction features
- Checkpoint saving

Phases (exec restart between build and train):
  build   Encode queries/tracks, build feature tensors, save to disk
  train   Train MLP + evaluate on fold-0 hist_7
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
CLEAN_R21 = REPO / "cache" / "r30" / "clean_fold0" / "r21_fold0_model"
R33_DIR = REPO / "cache" / "r33c_clean"

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEAT_NAMES = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
N_LR_FEAT = len(FEAT_NAMES)
N_INTERACT = 3
TAU = 0.5


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


class PoolMLP(nn.Module):
    def __init__(self, emb_dim=768, lr_dim=29, inter_dim=3, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + emb_dim + lr_dim + inter_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_query_text(case):
    parts = [str(r["content"]) for r in case["history"] if r["role"] == "user"]
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


# ---------------------------------------------------------------------------
# Phase: build
# ---------------------------------------------------------------------------

def phase_build():
    print(f"{ts()} Phase: build")
    R33_DIR.mkdir(parents=True, exist_ok=True)

    tensors_path = R33_DIR / "tensors.npz"
    meta_path = R33_DIR / "meta.json"

    if tensors_path.exists() and meta_path.exists():
        print("  Artifacts exist, skipping build")
        return

    # Load payload + build pools/features (needs heavy imports)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids[j] for j in top])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    folds = grouped_session_folds(sessions, seed=0)
    fold0_val = set(folds[0].tolist())

    # Encode queries + tracks with CLEAN fold-0 R21 model
    print(f"{ts()} Encoding with clean fold-0 R21 model...")
    from sentence_transformers import SentenceTransformer
    clean_model = SentenceTransformer(str(CLEAN_R21), device="cpu")

    query_texts = [build_query_text(c) for c in cases]
    print(f"  Encoding {len(query_texts)} queries...")
    query_embs = clean_model.encode(query_texts, batch_size=64,
                                     normalize_embeddings=True,
                                     show_progress_bar=True).astype(np.float32)

    # Load track metadata for text encoding
    from scripts.expR30_deep_history_retriever import load_catalog, build_track_text
    meta_catalog, all_track_ids = load_catalog()
    track_texts = [build_track_text(tid, meta_catalog) for tid in all_track_ids]
    tid_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

    print(f"  Encoding {len(track_texts)} tracks...")
    track_embs = clean_model.encode(track_texts, batch_size=128,
                                     normalize_embeddings=True,
                                     show_progress_bar=True).astype(np.float32)
    del clean_model
    gc.collect()

    # Build per-case feature tensors
    n = len(cases)
    print(f"\n{ts()} Building feature tensors for {n} cases...")

    all_lr_feat = np.zeros((n, POOL_K, N_LR_FEAT), dtype=np.float32)
    all_interact = np.zeros((n, POOL_K, N_INTERACT), dtype=np.float32)
    all_cand_idx = np.full((n, POOL_K), -1, dtype=np.int32)
    gt_positions = np.full(n, -1, dtype=np.int32)
    pool_sizes = np.zeros(n, dtype=np.int32)
    is_fold0 = np.zeros(n, dtype=bool)
    hist_depth = np.zeros(n, dtype=np.int32)
    gt_same_artist = np.zeros(n, dtype=bool)

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pool_sizes[i] = len(pool)
        is_fold0[i] = i in fold0_val
        hist_depth[i] = c["n_prior_music"]

        gt = c["gt"]
        if gt in pool:
            gt_positions[i] = pool.index(gt)

        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in c["music_turns"]} - {""}
        gt_same_artist[i] = bool(gt_artist and gt_artist in played_artists)

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists_list = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists_list if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

        q_emb = query_embs[i]

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            j = rank - 1
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = all_lr_feat[i, j]

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
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sn, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

            cand_catalog_idx = tid_to_idx.get(tid, -1)
            all_cand_idx[i, j] = cand_catalog_idx

            if cand_catalog_idx >= 0:
                c_emb = track_embs[cand_catalog_idx]
                all_interact[i, j, 0] = float(np.dot(q_emb, c_emb))
            all_interact[i, j, 1] = 1.0 / rank
            all_interact[i, j, 2] = rank / POOL_K

    np.savez_compressed(tensors_path,
                        query_embs=query_embs, track_embs=track_embs,
                        lr_feat=all_lr_feat, interact=all_interact,
                        cand_idx=all_cand_idx, gt_pos=gt_positions,
                        pool_sizes=pool_sizes, is_fold0=is_fold0,
                        hist_depth=hist_depth, gt_same_artist=gt_same_artist)

    meta_out = {"n_cases": n, "n_tracks": len(all_track_ids),
                "track_ids": all_track_ids,
                "pool_k": POOL_K, "n_lr_feat": N_LR_FEAT}
    with open(meta_path, "w") as f:
        json.dump(meta_out, f)

    print(f"  Saved tensors + meta to {R33_DIR}")
    print(f"{ts()} build complete. Restarting for train phase...")
    print("  Build complete. Run again with --phase train")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Phase: train
# ---------------------------------------------------------------------------

def phase_train(epochs=20, lr=1e-3, batch_size=16):
    print(f"{ts()} Phase: train (epochs={epochs}, lr={lr}, bs={batch_size})")

    data = np.load(R33_DIR / "tensors.npz")
    query_embs = data["query_embs"]
    track_embs = data["track_embs"]
    lr_feat = data["lr_feat"]
    interact = data["interact"]
    cand_idx = data["cand_idx"]
    gt_pos = data["gt_pos"]
    pool_sizes = data["pool_sizes"]
    is_fold0 = data["is_fold0"]
    hist_depth = data["hist_depth"]
    gt_same_artist = data["gt_same_artist"]

    emb_dim = query_embs.shape[1]
    print(f"  {len(query_embs)} cases, emb_dim={emb_dim}")

    # Train: hist_5+, not fold-0, GT in pool
    train_idx = [i for i in range(len(query_embs))
                 if hist_depth[i] >= 5 and not is_fold0[i] and gt_pos[i] >= 0]
    val_idx = [i for i in range(len(query_embs))
               if hist_depth[i] == 7 and is_fold0[i]]

    print(f"  train: {len(train_idx)} (hist>=5, non-fold0, GT in pool)")
    print(f"  val: {len(val_idx)} (hist_7, fold-0)")

    model = PoolMLP(emb_dim=emb_dim, lr_dim=N_LR_FEAT, inter_dim=N_INTERACT)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")

    track_embs_t = torch.from_numpy(track_embs)

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(train_idx), batch_size):
            b_cases = train_idx[b_start:b_start + batch_size]
            actual_bs = len(b_cases)

            q_batch = torch.from_numpy(query_embs[b_cases])
            q_expanded = q_batch.unsqueeze(1).expand(-1, POOL_K, -1)

            cidx = cand_idx[b_cases]
            valid_mask = cidx >= 0
            cidx_safe = np.where(valid_mask, cidx, 0)
            c_batch = track_embs_t[cidx_safe.flatten()].reshape(actual_bs, POOL_K, emb_dim)

            lr_batch = torch.from_numpy(lr_feat[b_cases])
            inter_batch = torch.from_numpy(interact[b_cases])

            x = torch.cat([q_expanded, c_batch, lr_batch, inter_batch], dim=-1)
            scores = model(x.reshape(actual_bs * POOL_K, -1)).reshape(actual_bs, POOL_K)

            pad_mask = torch.from_numpy(
                np.arange(POOL_K)[None, :] >= pool_sizes[b_cases][:, None])
            scores = scores.masked_fill(pad_mask, -1e9)

            targets = torch.from_numpy(gt_pos[b_cases]).long()
            loss = F.cross_entropy(scores / TAU, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Val eval
        for m in model.modules():
            m.training = False

        val_ndcg = 0.0
        n_val = 0
        with torch.no_grad():
            for vi in val_idx:
                ps = int(pool_sizes[vi])
                if ps == 0:
                    n_val += 1
                    continue

                q = torch.from_numpy(query_embs[vi:vi+1]).expand(ps, -1)
                ci = cand_idx[vi, :ps]
                valid = ci >= 0
                ci_safe = np.where(valid, ci, 0)
                c = track_embs_t[ci_safe]
                lr_f = torch.from_numpy(lr_feat[vi, :ps])
                inter_f = torch.from_numpy(interact[vi, :ps])

                x = torch.cat([q, c, lr_f, inter_f], dim=-1)
                sc = model(x).numpy()
                ranked = np.argsort(-sc)

                gp = gt_pos[vi]
                if gp >= 0 and gp < ps:
                    gt_rank = np.where(ranked == gp)[0]
                    if len(gt_rank) > 0 and gt_rank[0] < 20:
                        val_ndcg += 1.0 / np.log2(gt_rank[0] + 2)
                n_val += 1

        val_h7 = val_ndcg / max(n_val, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}  val_h7={val_h7:.5f}", flush=True)

        # Save checkpoint each epoch
        torch.save({"model_state": model.state_dict(), "epoch": epoch,
                     "val_h7": val_h7, "loss": avg_loss},
                    R33_DIR / f"model_ep{epoch}.pt")

    # Final full eval with diagnostics
    print(f"\n{ts()} Final evaluation...")
    for m in model.modules():
        m.training = False

    # Get LambdaRank OOF baseline scores
    import lightgbm as lgb
    from scripts.expS2_lambdarank_grouped import grouped_session_folds

    n = len(query_embs)
    with open(R12_CACHE, "rb") as f:
        cases_raw = pickle.load(f)["cases"]
    sessions = [c["session_id"] for c in cases_raw]
    del cases_raw
    gc.collect()

    folds = grouped_session_folds(sessions, seed=0)
    lr_scores: list[np.ndarray | None] = [None] * n

    print("  Training CV5 LambdaRank baseline...")
    for fi in range(5):
        val_set = set(folds[fi].tolist())
        tr = [j for j in range(n) if j not in val_set]
        va = sorted(val_set)

        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in tr:
            s = int(pool_sizes[idx])
            for k in range(s):
                X_tr.append(lr_feat[idx, k])
                y_tr.append(1.0 if k == gt_pos[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(pool_sizes[idx])
            for k in range(s):
                X_va.append(lr_feat[idx, k])
                y_va.append(1.0 if k == gt_pos[idx] else 0.0)
            g_va.append(s)

        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(FEAT_NAMES))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        lgb_model = lgb.train(params, ds_tr, num_boost_round=300,
                              valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = lgb_model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(pool_sizes[idx])
            lr_scores[idx] = preds[offset:offset + s].copy()
            offset += s

    # Compare MLP vs LambdaRank on fold-0 hist_7
    h7_val = [i for i in range(n) if hist_depth[i] == 7 and is_fold0[i]]
    lr_ndcg = 0.0
    mlp_ndcg = 0.0
    same_lr = 0.0
    same_mlp = 0.0
    diff_lr = 0.0
    diff_mlp = 0.0
    n_same = 0
    n_diff = 0
    recovered = 0
    lost = 0
    lr_gt_ranks = []
    mlp_gt_ranks = []

    with torch.no_grad():
        for i in h7_val:
            ps = int(pool_sizes[i])
            gp = gt_pos[i]
            is_same = bool(gt_same_artist[i])

            # LambdaRank ranking
            sc_lr = lr_scores[i]
            if sc_lr is not None and len(sc_lr) > 0:
                lr_ranked = np.argsort(-sc_lr)
            else:
                lr_ranked = np.arange(ps)

            # MLP ranking
            if ps > 0:
                q = torch.from_numpy(query_embs[i:i+1]).expand(ps, -1)
                ci = cand_idx[i, :ps]
                ci_safe = np.where(ci >= 0, ci, 0)
                c = track_embs_t[ci_safe]
                x = torch.cat([q, c, torch.from_numpy(lr_feat[i, :ps]),
                               torch.from_numpy(interact[i, :ps])], dim=-1)
                mlp_sc = model(x).numpy()
                mlp_ranked = np.argsort(-mlp_sc)
            else:
                mlp_ranked = np.array([])

            # nDCG
            lr_v = 0.0
            if gp >= 0 and gp < ps:
                lr_pos = np.where(lr_ranked == gp)[0]
                if len(lr_pos) > 0 and lr_pos[0] < 20:
                    lr_v = 1.0 / np.log2(lr_pos[0] + 2)
                lr_gt_ranks.append(int(lr_pos[0]) + 1 if len(lr_pos) > 0 else ps + 1)

                mlp_pos = np.where(mlp_ranked == gp)[0]
                mlp_v_val = 0.0
                if len(mlp_pos) > 0 and mlp_pos[0] < 20:
                    mlp_v_val = 1.0 / np.log2(mlp_pos[0] + 2)
                mlp_gt_ranks.append(int(mlp_pos[0]) + 1 if len(mlp_pos) > 0 else ps + 1)
            else:
                mlp_v_val = 0.0

            lr_ndcg += lr_v
            mlp_ndcg += mlp_v_val

            if is_same:
                same_lr += lr_v
                same_mlp += mlp_v_val
                n_same += 1
            else:
                diff_lr += lr_v
                diff_mlp += mlp_v_val
                n_diff += 1

            lr_in_top20 = gp >= 0 and gp < ps and len(np.where(lr_ranked == gp)[0]) > 0 and np.where(lr_ranked == gp)[0][0] < 20
            mlp_in_top20 = gp >= 0 and ps > 0 and len(np.where(mlp_ranked == gp)[0]) > 0 and np.where(mlp_ranked == gp)[0][0] < 20
            if not lr_in_top20 and mlp_in_top20:
                recovered += 1
            if lr_in_top20 and not mlp_in_top20:
                lost += 1

    n_val = len(h7_val)
    n_in_pool = sum(1 for i in h7_val if gt_pos[i] >= 0)
    lr_h7 = lr_ndcg / max(n_val, 1)
    mlp_h7 = mlp_ndcg / max(n_val, 1)
    dh7 = mlp_h7 - lr_h7

    lr_h7_pool = lr_ndcg / max(n_in_pool, 1)
    mlp_h7_pool = mlp_ndcg / max(n_in_pool, 1)

    sep = "=" * 70
    print(f"\n{sep}")
    print("R33c-CLEAN FULL-POOL EVALUATION (fold-0 hist_7)")
    print(sep)
    print(f"  Cases: {n_val} total, {n_in_pool} GT in pool ({n_in_pool/n_val*100:.1f}%)")
    print(f"\n  h7_all (/{n_val}):")
    print(f"    LambdaRank:  {lr_h7:.5f}")
    print(f"    MLP:         {mlp_h7:.5f} ({dh7:+.5f})")
    print(f"\n  h7_in_pool (/{n_in_pool}):")
    print(f"    LambdaRank:  {lr_h7_pool:.5f}")
    print(f"    MLP:         {mlp_h7_pool:.5f} ({mlp_h7_pool-lr_h7_pool:+.5f})")
    if n_same > 0:
        print(f"  same_artist: LR={same_lr/n_same:.5f}  MLP={same_mlp/n_same:.5f}")
    if n_diff > 0:
        print(f"  diff_artist: LR={diff_lr/n_diff:.5f}  MLP={diff_mlp/n_diff:.5f}")
    print(f"  recovered: {recovered}  lost: {lost}  net: {recovered-lost:+d}")
    if lr_gt_ranks:
        print(f"  LR median GT rank:  {np.median(lr_gt_ranks):.0f}")
    if mlp_gt_ranks:
        print(f"  MLP median GT rank: {np.median(mlp_gt_ranks):.0f}")

    print(f"\n{sep}")
    print("GATE CHECK")
    g = dh7 >= 0.005
    print(f"  Δh7 >= +0.005: {dh7:+.5f} {'PASS' if g else 'FAIL'}")

    out_path = REPO / "exp" / "eval" / "expR33c_clean_fullpool.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"lr_h7": lr_h7, "mlp_h7": mlp_h7, "delta": dh7,
                    "recovered": recovered, "lost": lost,
                    "n_same": n_same, "n_diff": n_diff,
                    "lr_med_rank": float(np.median(lr_gt_ranks)) if lr_gt_ranks else -1,
                    "mlp_med_rank": float(np.median(mlp_gt_ranks)) if mlp_gt_ranks else -1},
                   f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["build", "train"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R33c-clean: Neural MLP Pool Reranker — {args.phase}")
    print("=" * 70)

    if args.phase == "build":
        phase_build()
    elif args.phase == "train":
        phase_train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
