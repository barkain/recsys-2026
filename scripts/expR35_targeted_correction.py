#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R35: Targeted correction for LambdaRank failure cases.

Train specifically on cases where GT is in pool@300 but LambdaRank
ranks it below top-20. Force the model to promote the buried GT
above the current top-20 wrong candidates.

Phases:
  build   Build tensors with clean fold-0 embeddings (reuses R33c cache if available)
  train   Train pairwise correction model + evaluate fold-0 hist_7
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
CLEAN_R21 = REPO / "cache" / "r30" / "clean_fold0" / "r21_fold0_model"
R33_CACHE = REPO / "cache" / "r33c_clean"
R35_DIR = REPO / "cache" / "r35"

from scripts.expS2_lr_v2 import FEATURE_NAMES_V2

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEAT_NAMES = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
N_LR_FEAT = len(FEAT_NAMES)


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


def build_query_text(case):
    parts = [str(r["content"]) for r in case["history"] if r["role"] == "user"]
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def phase_build():
    """Build or reuse tensors from R33c clean cache."""
    print(f"{ts()} Phase: build")
    R35_DIR.mkdir(parents=True, exist_ok=True)

    # Check if R33c clean tensors exist
    r33_tensors = R33_CACHE / "tensors.npz"
    if r33_tensors.exists():
        print(f"  Reusing R33c clean tensors from {r33_tensors}")
        return

    # Otherwise build from scratch (same as R33c build)
    print("  R33c tensors not found. Run R33c --phase build first.")
    sys.exit(1)


def phase_train(epochs=30, lr=1e-3, beta_values=None):
    """Train targeted correction model."""
    if beta_values is None:
        beta_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    print(f"{ts()} Phase: train")

    # Load tensors
    data = np.load(R33_CACHE / "tensors.npz")
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
    n = len(query_embs)

    # Build OOF LambdaRank scores
    print(f"{ts()} Training CV5 LambdaRank...")
    with open(R12_CACHE, "rb") as f:
        sessions = [c["session_id"] for c in pickle.load(f)["cases"]]

    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)

    lr_scores_all: list[np.ndarray | None] = [None] * n
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
            lr_scores_all[idx] = preds[offset:offset + s].copy()
            offset += s

    # Identify LambdaRank failure cases for training
    fold0_val = set(folds[0].tolist())

    # Find cases where GT is in pool but LR ranks it below top-20
    train_miss_cases = []
    for i in range(n):
        if i in fold0_val:
            continue
        if hist_depth[i] < 5:
            continue
        if gt_pos[i] < 0:
            continue
        sc = lr_scores_all[i]
        if sc is None:
            continue
        lr_ranked = np.argsort(-sc)
        gt_lr_rank = np.where(lr_ranked == gt_pos[i])[0]
        if len(gt_lr_rank) > 0 and gt_lr_rank[0] >= 20:
            train_miss_cases.append(i)

    print(f"  LR miss cases (hist>=5, non-fold0, GT in pool, ranked >20): {len(train_miss_cases)}")

    # Build pairwise training data
    # For each miss case: GT vs each of LR's top-20 wrong candidates
    pair_gt_features = []
    pair_neg_features = []

    for i in train_miss_cases:
        ps = int(pool_sizes[i])
        sc = lr_scores_all[i]
        if sc is None:
            continue
        lr_ranked = np.argsort(-sc)
        top20_wrong = [lr_ranked[j] for j in range(min(20, ps)) if lr_ranked[j] != gt_pos[i]]

        q_emb = query_embs[i]
        gt_j = gt_pos[i]
        gt_cand = cand_idx[i, gt_j]
        if gt_cand < 0:
            continue
        gt_c_emb = track_embs[gt_cand]
        gt_lr = lr_feat[i, gt_j]
        gt_inter = interact[i, gt_j]
        gt_feat = np.concatenate([q_emb, gt_c_emb, gt_lr, gt_inter])

        for neg_j in top20_wrong[:10]:
            neg_cand = cand_idx[i, neg_j]
            if neg_cand < 0:
                continue
            neg_c_emb = track_embs[neg_cand]
            neg_lr = lr_feat[i, neg_j]
            neg_inter = interact[i, neg_j]
            neg_feat = np.concatenate([q_emb, neg_c_emb, neg_lr, neg_inter])

            pair_gt_features.append(gt_feat)
            pair_neg_features.append(neg_feat)

    n_pairs = len(pair_gt_features)
    print(f"  Training pairs: {n_pairs} (GT vs LR top-20 wrong)")

    if n_pairs == 0:
        print("  No training pairs! Cannot train.")
        return

    gt_feats = torch.from_numpy(np.stack(pair_gt_features).astype(np.float32))
    neg_feats = torch.from_numpy(np.stack(pair_neg_features).astype(np.float32))
    feat_dim = gt_feats.shape[1]

    # Train correction MLP
    model = CorrectionMLP(input_dim=feat_dim, hidden=128, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CorrectionMLP: {n_params/1e3:.1f}K params, feat_dim={feat_dim}", flush=True)
    print(f"  gt_feats: {gt_feats.shape}, neg_feats: {neg_feats.shape}", flush=True)

    batch_size = 256
    model.train()
    print("  Starting training loop...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    for epoch in range(epochs):
        if epoch == 0:
            print("    Entering epoch 0...", flush=True)
        perm = torch.randperm(n_pairs)
        epoch_loss = 0.0
        n_batches = 0
        for b_start in range(0, n_pairs, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            gt_b = gt_feats[b_idx]
            neg_b = neg_feats[b_idx]

            gt_sc = model(gt_b)
            neg_sc = model(neg_b)
            margin = 0.1
            loss = torch.clamp(margin - (gt_sc - neg_sc), min=0).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"    Epoch {epoch}: loss={avg_loss:.4f}", flush=True)

    # Evaluate: apply correction to fold-0 hist_7
    print(f"\n{ts()} Evaluating on fold-0 hist_7...")
    for m in model.modules():
        m.training = False

    h7_val = [i for i in range(n) if hist_depth[i] == 7 and is_fold0[i]]
    n_in_pool = sum(1 for i in h7_val if gt_pos[i] >= 0)
    n_val = len(h7_val)
    print(f"  {n_val} eval cases, {n_in_pool} GT in pool")

    # Baseline LambdaRank nDCG
    lr_ndcg = 0.0
    for i in h7_val:
        sc = lr_scores_all[i]
        if sc is None or pool_sizes[i] == 0:
            continue
        lr_ranked = np.argsort(-sc)
        gp = gt_pos[i]
        if gp >= 0:
            gt_rank = np.where(lr_ranked == gp)[0]
            if len(gt_rank) > 0 and gt_rank[0] < 20:
                lr_ndcg += 1.0 / np.log2(gt_rank[0] + 2)
    lr_h7 = lr_ndcg / n_val

    # Sweep beta values
    print(f"\n  {'beta':<8} {'h7_all':>10} {'h7_pool':>10} {'delta':>10} {'recov':>6} {'lost':>6} {'net':>6}")
    print(f"  {'-'*58}")
    print(f"  {'LR base':<8} {lr_h7:>10.5f} {lr_ndcg/max(n_in_pool,1):>10.5f}")

    best_beta = 0.0
    best_h7 = lr_h7

    # Identify LR miss-in-pool cases in fold-0 hist_7
    lr_miss_in_pool_val = set()
    for i in h7_val:
        sc = lr_scores_all[i]
        gp = gt_pos[i]
        if gp >= 0 and sc is not None:
            lr_ranked = np.argsort(-sc)
            gt_rank = np.where(lr_ranked == gp)[0]
            if len(gt_rank) > 0 and gt_rank[0] >= 20:
                lr_miss_in_pool_val.add(i)
    print(f"  LR miss-in-pool on fold0 hist_7: {len(lr_miss_in_pool_val)}")

    for beta in beta_values:
        mlp_ndcg = 0.0
        mlp_miss_ndcg = 0.0
        same_ndcg = 0.0
        diff_ndcg = 0.0
        n_same = 0
        n_diff = 0
        recovered = 0
        lost_count = 0

        with torch.no_grad():
            for i in h7_val:
                ps = int(pool_sizes[i])
                sc = lr_scores_all[i]
                if sc is None or ps == 0:
                    continue

                lr_ranked = np.argsort(-sc)
                gp = gt_pos[i]

                lr_z = (sc - sc.mean()) / (sc.std() + 1e-8)

                # Batch score all candidates
                q_exp = np.broadcast_to(query_embs[i], (ps, query_embs.shape[1]))
                ci = cand_idx[i, :ps]
                ci_safe = np.where(ci >= 0, ci, 0)
                c_embs = track_embs[ci_safe]
                feats = np.concatenate([q_exp, c_embs, lr_feat[i, :ps],
                                        interact[i, :ps]], axis=1).astype(np.float32)
                corrections = model(torch.from_numpy(feats)).detach().numpy()

                # Apply correction only to candidates ranked 21+
                final_scores = lr_z.copy()
                for j in range(ps):
                    lr_rank_j = np.where(lr_ranked == j)[0]
                    if len(lr_rank_j) > 0 and lr_rank_j[0] >= 20:
                        final_scores[j] = lr_z[j] + beta * corrections[j]
                mlp_ranked = np.argsort(-final_scores)

                lr_in_top20 = gp >= 0 and np.where(lr_ranked == gp)[0][0] < 20 if gp >= 0 and len(np.where(lr_ranked == gp)[0]) > 0 else False
                mlp_in_top20 = gp >= 0 and np.where(mlp_ranked == gp)[0][0] < 20 if gp >= 0 and len(np.where(mlp_ranked == gp)[0]) > 0 else False

                mlp_v = 0.0
                if gp >= 0 and mlp_in_top20:
                    pos = np.where(mlp_ranked == gp)[0][0]
                    mlp_v = 1.0 / np.log2(pos + 2)
                mlp_ndcg += mlp_v

                if i in lr_miss_in_pool_val:
                    mlp_miss_ndcg += mlp_v

                is_same = bool(gt_same_artist[i]) if gt_pos[i] >= 0 else False
                if is_same:
                    same_ndcg += mlp_v
                    n_same += 1
                else:
                    diff_ndcg += mlp_v
                    n_diff += 1

                if not lr_in_top20 and mlp_in_top20:
                    recovered += 1
                if lr_in_top20 and not mlp_in_top20:
                    lost_count += 1

        mlp_h7 = mlp_ndcg / n_val
        dh7 = mlp_h7 - lr_h7
        miss_h7 = mlp_miss_ndcg / max(len(lr_miss_in_pool_val), 1)
        same_h7 = same_ndcg / max(n_same, 1)
        diff_h7 = diff_ndcg / max(n_diff, 1)
        print(f"  {beta:<8.3f} {mlp_h7:>10.5f} {dh7:>+10.5f} {recovered:>6} {lost_count:>6} "
              f"{recovered-lost_count:>+6}  miss={miss_h7:.4f} same={same_h7:.4f} diff={diff_h7:.4f}")

        if mlp_h7 > best_h7:
            best_h7 = mlp_h7
            best_beta = beta

    # Gate check
    sep = "=" * 70
    print(f"\n{sep}")
    print("GATE CHECK")
    dh7_best = best_h7 - lr_h7
    g = dh7_best >= 0.005
    print(f"  Best beta={best_beta}: Δh7={dh7_best:+.5f} {'PASS' if g else 'FAIL'}")

    out_path = REPO / "exp" / "eval" / "expR35_targeted_correction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"lr_h7": lr_h7, "best_h7": best_h7,
                    "best_beta": best_beta, "delta": dh7_best}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["build", "train"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R35: Targeted Correction Reranker — {args.phase}")
    print("=" * 70)

    if args.phase == "build":
        phase_build()
    elif args.phase == "train":
        phase_train(epochs=args.epochs, lr=args.lr)

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
