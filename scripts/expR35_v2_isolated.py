#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R35-v2: Targeted correction reranker — process-isolated.

Uses cached LR scores from R34 (no LightGBM import).
Uses cached tensors from R33c clean (no datasets/loky import).
PyTorch-only: no segfault risk.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn

# Guard: no lightgbm/loky/joblib in this process
for forbidden in ["lightgbm", "joblib", "loky"]:
    if forbidden in sys.modules:
        raise RuntimeError(f"{forbidden} already imported — will segfault with torch autograd")

REPO = Path(__file__).resolve().parent.parent

R33_TENSORS = REPO / "cache" / "r33c_clean" / "tensors.npz"
R34_LR_SCORES = REPO / "cache" / "r34_residual" / "lr_scores.npy"
R35_DIR = REPO / "cache" / "r35"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


class CorrectionMLP(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--max-neg", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R35-v2: Targeted Correction (process-isolated)")
    print("=" * 70)

    # Load cached data (no LightGBM needed)
    print(f"{ts()} Loading cached tensors + LR scores...")
    data = np.load(R33_TENSORS)
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

    lr_scores = np.load(R34_LR_SCORES)
    n = len(query_embs)
    print(f"  {n} cases, LR scores {lr_scores.shape}")

    R35_DIR.mkdir(parents=True, exist_ok=True)

    # Identify LR failure cases (non-fold0, hist>=5, GT in pool, LR ranks >20)
    train_miss = []
    for i in range(n):
        if is_fold0[i] or hist_depth[i] < 5 or gt_pos[i] < 0:
            continue
        sc = lr_scores[i, :int(pool_sizes[i])]
        if len(sc) == 0:
            continue
        ranked = np.argsort(-sc)
        gt_rank = np.where(ranked == gt_pos[i])[0]
        if len(gt_rank) > 0 and gt_rank[0] >= 20:
            train_miss.append(i)

    print(f"  LR miss cases: {len(train_miss)}")

    # Build pairwise data: GT vs LR's top-k wrong candidates
    pair_gt, pair_neg = [], []
    for i in train_miss:
        ps = int(pool_sizes[i])
        sc = lr_scores[i, :ps]
        ranked = np.argsort(-sc)
        top_wrong = [ranked[j] for j in range(min(20, ps)) if ranked[j] != gt_pos[i]]

        q = query_embs[i]
        gj = gt_pos[i]
        gc = cand_idx[i, gj]
        if gc < 0:
            continue
        gt_feat = np.concatenate([q, track_embs[gc], lr_feat[i, gj], interact[i, gj]])

        for nj in top_wrong[:args.max_neg]:
            nc = cand_idx[i, nj]
            if nc < 0:
                continue
            neg_feat = np.concatenate([q, track_embs[nc], lr_feat[i, nj], interact[i, nj]])
            pair_gt.append(gt_feat)
            pair_neg.append(neg_feat)

    n_pairs = len(pair_gt)
    print(f"  Training pairs: {n_pairs}")

    gt_t = torch.from_numpy(np.stack(pair_gt).astype(np.float32))
    neg_t = torch.from_numpy(np.stack(pair_neg).astype(np.float32))
    feat_dim = gt_t.shape[1]
    print(f"  feat_dim={feat_dim}, gt={gt_t.shape}, neg={neg_t.shape}", flush=True)

    # Train
    model = CorrectionMLP(feat_dim, hidden=128, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e3:.1f}K params", flush=True)

    model.train()
    batch_size = 256
    for epoch in range(args.epochs):
        perm = torch.randperm(n_pairs)
        epoch_loss = 0.0
        nb = 0
        for bs in range(0, n_pairs, batch_size):
            bi = perm[bs:bs + batch_size]
            gs = model(gt_t[bi])
            ns = model(neg_t[bi])
            loss = torch.clamp(args.margin - (gs - ns), min=0).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch}: loss={epoch_loss/nb:.4f}", flush=True)

    # Save model
    torch.save(model.state_dict(), R35_DIR / "correction_mlp.pt")

    # Eval on fold-0 hist_7
    print(f"\n{ts()} Evaluating fold-0 hist_7...")
    for m in model.modules():
        m.training = False

    h7_val = [i for i in range(n) if hist_depth[i] == 7 and is_fold0[i]]
    n_in_pool = sum(1 for i in h7_val if gt_pos[i] >= 0)

    # Identify LR miss-in-pool on fold-0
    lr_miss_pool = set()
    for i in h7_val:
        ps = int(pool_sizes[i])
        if gt_pos[i] < 0 or ps == 0:
            continue
        sc = lr_scores[i, :ps]
        ranked = np.argsort(-sc)
        gt_rank = np.where(ranked == gt_pos[i])[0]
        if len(gt_rank) > 0 and gt_rank[0] >= 20:
            lr_miss_pool.add(i)

    print(f"  {len(h7_val)} eval cases, {n_in_pool} GT in pool, "
          f"{len(lr_miss_pool)} LR miss-in-pool")

    # Baseline LR nDCG
    lr_ndcg = 0.0
    for i in h7_val:
        ps = int(pool_sizes[i])
        if gt_pos[i] < 0 or ps == 0:
            continue
        sc = lr_scores[i, :ps]
        ranked = np.argsort(-sc)
        gt_rank = np.where(ranked == gt_pos[i])[0]
        if len(gt_rank) > 0 and gt_rank[0] < 20:
            lr_ndcg += 1.0 / np.log2(gt_rank[0] + 2)
    lr_h7 = lr_ndcg / len(h7_val)

    # Beta sweep
    betas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    print(f"\n  {'beta':<8} {'h7_all':>10} {'delta':>10} {'rec':>5} {'lost':>5} "
          f"{'net':>5} {'miss_h7':>10} {'same':>10} {'diff':>10}")
    print(f"  {'-'*80}")

    best_beta = 0.0
    best_h7 = lr_h7

    for beta in betas:
        mlp_ndcg = 0.0
        miss_ndcg = 0.0
        same_ndcg = 0.0
        diff_ndcg = 0.0
        n_same = 0
        n_diff = 0
        recovered = 0
        lost = 0

        with torch.no_grad():
            for i in h7_val:
                ps = int(pool_sizes[i])
                if ps == 0:
                    is_same = bool(gt_same_artist[i])
                    if is_same:
                        n_same += 1
                    else:
                        n_diff += 1
                    continue

                sc = lr_scores[i, :ps]
                lr_z = (sc - sc.mean()) / (sc.std() + 1e-8)
                gp = gt_pos[i]
                lr_ranked = np.argsort(-sc)

                if beta > 0:
                    q = query_embs[i]
                    ci = cand_idx[i, :ps]
                    ci_safe = np.where(ci >= 0, ci, 0)
                    feats = np.concatenate([
                        np.broadcast_to(q, (ps, query_embs.shape[1])),
                        track_embs[ci_safe],
                        lr_feat[i, :ps],
                        interact[i, :ps],
                    ], axis=1).astype(np.float32)
                    corr = model(torch.from_numpy(feats)).numpy()

                    final = lr_z.copy()
                    for j in range(ps):
                        rj = np.where(lr_ranked == j)[0]
                        if len(rj) > 0 and rj[0] >= 20:
                            final[j] = lr_z[j] + beta * corr[j]
                else:
                    final = lr_z

                ranked = np.argsort(-final)

                lr_in = (gp >= 0 and len(np.where(lr_ranked == gp)[0]) > 0
                         and np.where(lr_ranked == gp)[0][0] < 20)
                mlp_in = (gp >= 0 and len(np.where(ranked == gp)[0]) > 0
                          and np.where(ranked == gp)[0][0] < 20)

                mlp_v = 0.0
                if gp >= 0 and mlp_in:
                    pos = np.where(ranked == gp)[0][0]
                    mlp_v = 1.0 / np.log2(pos + 2)
                mlp_ndcg += mlp_v

                if i in lr_miss_pool:
                    miss_ndcg += mlp_v

                is_same = bool(gt_same_artist[i])
                if is_same:
                    same_ndcg += mlp_v
                    n_same += 1
                else:
                    diff_ndcg += mlp_v
                    n_diff += 1

                if not lr_in and mlp_in:
                    recovered += 1
                if lr_in and not mlp_in:
                    lost += 1

        h7 = mlp_ndcg / len(h7_val)
        miss_h7 = miss_ndcg / max(len(lr_miss_pool), 1)
        same_h7 = same_ndcg / max(n_same, 1)
        diff_h7 = diff_ndcg / max(n_diff, 1)
        dh7 = h7 - lr_h7

        print(f"  {beta:<8.3f} {h7:>10.5f} {dh7:>+10.5f} {recovered:>5} {lost:>5} "
              f"{recovered-lost:>+5} {miss_h7:>10.5f} {same_h7:>10.5f} {diff_h7:>10.5f}")

        if h7 > best_h7:
            best_h7 = h7
            best_beta = beta

    # Gate
    sep = "=" * 70
    print(f"\n{sep}")
    print("GATE CHECK")
    print(f"  LR baseline h7: {lr_h7:.5f}")
    print(f"  Best beta={best_beta}: h7={best_h7:.5f} ({best_h7-lr_h7:+.5f})")
    g = (best_h7 - lr_h7) >= 0.005
    print(f"  Δh7 >= +0.005: {'PASS' if g else 'FAIL'}")

    out = {"lr_h7": lr_h7, "best_h7": best_h7, "best_beta": best_beta,
           "delta": best_h7 - lr_h7, "n_val": len(h7_val),
           "n_in_pool": n_in_pool, "n_lr_miss": len(lr_miss_pool)}
    with open(REPO / "exp" / "eval" / "expR35_v2_targeted.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved. Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
