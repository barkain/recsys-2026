#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R76 Phase 0B — Small MLP residual ranker, fold-0 inner CV.

Loads the Phase 0A dataset and trains a small MLP residual model:

    final_score(c) = zscore(oof_r54c_score) + beta * neural_delta(features)

Where neural_delta is a 3-layer MLP over per-candidate features.

Loss: listwise cross-entropy within each case (one-hot label = GT presence).
Cases without GT in top-30 are still trained on as zero-positive groups
(label = all zeros).

5-way inner CV within fold-0 (1600 cases → 1280 train / 320 test each fold).

Gate (predeclared):
  h7 nDCG Δ ≥ +0.005 vs OOF R54c baseline
  same-artist Δ ≥ -0.002
  recovered > lost
  top-20 overlap ≥ 14/20

CPU only (no MPS, no CUDA) for deterministic comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # type: ignore[reportMissingImports]
import torch.nn as nn  # type: ignore[reportMissingImports]
import torch.nn.functional as F  # type: ignore[reportMissingImports]

DATASET_PATH = REPO / "cache" / "r76" / "top30_fold0_dataset.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR76_phase0b_mlp_residual.json"
OUT_MD = REPO / "docs" / "r76_phase0b_result.md"

# Predeclared gates
GATE_H7_DELTA = 0.005
GATE_SAME_DELTA = -0.002
GATE_DIFF_DELTA = 0.0
GATE_OVERLAP_FLOOR = 14  # / 20
GATE_CHURN_CAP = 25  # / 80

# Inner CV setup
N_INNER_FOLDS = 5
TOP_K = 20
TOP_30 = 30

# Model hyperparams
HIDDEN = 64
N_LAYERS = 3
DROPOUT = 0.2
LR = 1e-3
N_EPOCHS = 30
BATCH_CASES = 64
WEIGHT_DECAY = 1e-4
SEED = 0


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


class ResidualMLP(nn.Module):
    """Small MLP that predicts a residual delta per candidate.

    The final ranking score is zscore(oof_r54c_score) + beta * neural_delta.
    We learn beta jointly.
    """

    def __init__(self, in_features: int, hidden: int = HIDDEN, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        layers = []
        dim_in = in_features
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim_in, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim_in = hidden
        layers.append(nn.Linear(dim_in, 1))
        self.mlp = nn.Sequential(*layers)
        # Learnable scalar blend coefficient
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, r54c_z: torch.Tensor) -> torch.Tensor:
        """
        x: (N, F) per-candidate features (excluding r54c_score itself)
        r54c_z: (N,) z-scored R54c score
        Returns: (N,) final score
        """
        delta = self.mlp(x).squeeze(-1)
        return r54c_z + self.beta * delta


def listwise_ce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Listwise CE within a group. scores/labels shape (n_cands,).

    If labels sum to 0 (no positive in this group), return zero loss.
    """
    if labels.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    log_probs = F.log_softmax(scores, dim=-1)
    targets = labels / labels.sum()
    return -(targets * log_probs).sum()


def build_inner_folds(case_indices: list[int], k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    shuf = list(case_indices)
    rng.shuffle(shuf)
    folds = [[] for _ in range(k)]
    for j, ci in enumerate(shuf):
        folds[j % k].append(ci)
    return folds


def prep_case_tensors(rows_per_case: list[dict], feature_names: list[str]):
    """For a single case, return (X, r54c_z, labels, candidate_ids)."""
    X = np.array([[r[f] for f in feature_names] for r in rows_per_case], dtype=np.float32)
    r54c = np.array([r["oof_r54c_score"] for r in rows_per_case], dtype=np.float32)
    labels = np.array([r["label"] for r in rows_per_case], dtype=np.float32)
    cand_ids = [r["candidate_track_id"] for r in rows_per_case]
    return X, r54c, labels, cand_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R76 Phase 0B — MLP residual ranker, fold-0 inner CV (CPU only)")
    print(f"  hidden={args.hidden} dropout={args.dropout} lr={args.lr} epochs={args.epochs}")
    print("=" * 70)

    # Reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")  # explicit: no MPS, no CUDA

    print(f"{ts()} Loading dataset ...", flush=True)
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)
    rows = data["rows"]
    feature_names_full = data["feature_names"]
    metadata = data["metadata"]
    print(f"  rows: {len(rows)}  features per row: {len(feature_names_full)}")
    print(f"  feature_names sample: {feature_names_full[:8]} ... (total {len(feature_names_full)})")

    # Group rows by case_idx (preserve order)
    by_case: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_case[r["case_idx"]].append(r)
    case_indices = sorted(by_case.keys())
    n_cases = len(case_indices)
    print(f"  cases: {n_cases}")

    # Exclude oof_r54c_score from MLP input (we add it back separately via r54c_z)
    # Keep all other features in the MLP.
    MLP_FEATURES = [f for f in feature_names_full if f != "oof_r54c_score"]
    print(f"  MLP input dim: {len(MLP_FEATURES)}")

    # Build inner folds
    inner_folds = build_inner_folds(case_indices, N_INNER_FOLDS, SEED)
    print(f"  inner-CV folds: {[len(f) for f in inner_folds]}")

    # Helper: featurize all cases once
    print(f"{ts()} Featurizing all cases ...", flush=True)
    case_tensors = {}
    for ci in case_indices:
        X, r54c, labels, cand_ids = prep_case_tensors(by_case[ci], MLP_FEATURES)
        # is_h7 and is_same_artist are global to the case
        case_tensors[ci] = {
            "X": X,
            "r54c": r54c,
            "labels": labels,
            "cand_ids": cand_ids,
            "is_h7": by_case[ci][0]["is_h7"],
            "is_same_artist": by_case[ci][0]["is_same_artist"],
            "case_idx": ci,
        }
    print(f"  done")

    # Feature normalization stats (computed across all cases — leak-safe since
    # this is per-fold inner CV, but the normalization is just per-feature
    # mean/std; the labels still drive learning)
    all_X = np.vstack([case_tensors[ci]["X"] for ci in case_indices])
    feat_mean = all_X.mean(axis=0)
    feat_std = all_X.std(axis=0) + 1e-6
    all_r54c = np.concatenate([case_tensors[ci]["r54c"] for ci in case_indices])
    r54c_mean = float(all_r54c.mean())
    r54c_std = float(all_r54c.std() + 1e-6)
    print(f"  feature stats computed: r54c mean={r54c_mean:.3f} std={r54c_std:.3f}")
    del all_X, all_r54c

    # Run 5-way inner CV
    per_case_predictions: dict[int, list[float]] = {}

    for fk in range(N_INNER_FOLDS):
        test_cases = inner_folds[fk]
        train_cases = [ci for fold in inner_folds[:fk] + inner_folds[fk+1:] for ci in fold]
        print(f"\n{ts()} === Inner fold {fk}/{N_INNER_FOLDS}: "
              f"train={len(train_cases)} test={len(test_cases)} ===", flush=True)

        model = ResidualMLP(in_features=len(MLP_FEATURES),
                            hidden=args.hidden, n_layers=N_LAYERS,
                            dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Train
        model.train()
        for epoch in range(args.epochs):
            random.shuffle(train_cases)
            total_loss = 0.0
            n_batches = 0
            for batch_start in range(0, len(train_cases), BATCH_CASES):
                batch_case_ids = train_cases[batch_start:batch_start + BATCH_CASES]
                opt.zero_grad()
                batch_loss = torch.tensor(0.0, requires_grad=True)
                for ci in batch_case_ids:
                    ct = case_tensors[ci]
                    X = torch.from_numpy((ct["X"] - feat_mean) / feat_std).to(device)
                    r54c_z = torch.from_numpy((ct["r54c"] - r54c_mean) / r54c_std).to(device)
                    labels = torch.from_numpy(ct["labels"]).to(device)
                    scores = model(X, r54c_z)
                    loss = listwise_ce_loss(scores, labels)
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / len(batch_case_ids)
                batch_loss.backward()
                opt.step()
                total_loss += float(batch_loss.item())
                n_batches += 1
            if (epoch + 1) % 5 == 0:
                print(f"    epoch {epoch+1}/{args.epochs}  avg_loss={total_loss/max(n_batches,1):.4f}  "
                      f"beta={float(model.beta.item()):.3f}",
                      flush=True)

        # Eval on test
        model.eval()
        with torch.no_grad():
            for ci in test_cases:
                ct = case_tensors[ci]
                X = torch.from_numpy((ct["X"] - feat_mean) / feat_std).to(device)
                r54c_z = torch.from_numpy((ct["r54c"] - r54c_mean) / r54c_std).to(device)
                scores = model(X, r54c_z).cpu().numpy()
                per_case_predictions[ci] = scores.tolist()

    # ---- Final eval: compare R76 top-20 vs OOF R54c top-20 ----
    print(f"\n{ts()} === Final evaluation: R76 vs OOF R54c baseline ===", flush=True)
    results = []
    for ci in case_indices:
        ct = case_tensors[ci]
        r54c_scores = ct["r54c"]
        r76_scores = np.array(per_case_predictions[ci])
        labels = ct["labels"]
        cand_ids = ct["cand_ids"]
        # GT candidate index (if in top-30)
        gt_pos_in_top30 = int(np.where(labels == 1.0)[0][0]) if labels.sum() > 0 else -1
        # Baseline top-20
        b_order = np.argsort(-r54c_scores, kind="mergesort")
        b_top20 = b_order[:TOP_K]
        b_gt_rank = -1
        if gt_pos_in_top30 >= 0:
            bp = np.where(b_top20 == gt_pos_in_top30)[0]
            if len(bp):
                b_gt_rank = int(bp[0]) + 1
        # R76 top-20
        r_order = np.argsort(-r76_scores, kind="mergesort")
        r_top20 = r_order[:TOP_K]
        r_gt_rank = -1
        if gt_pos_in_top30 >= 0:
            rp = np.where(r_top20 == gt_pos_in_top30)[0]
            if len(rp):
                r_gt_rank = int(rp[0]) + 1
        results.append({
            "case_idx": ci,
            "is_h7": ct["is_h7"],
            "is_same_artist": ct["is_same_artist"],
            "gt_in_top30": gt_pos_in_top30 >= 0,
            "b_gt_rank_top20": b_gt_rank,
            "r_gt_rank_top20": r_gt_rank,
            "b_ndcg": ndcg_at_k(b_gt_rank, TOP_K),
            "r_ndcg": ndcg_at_k(r_gt_rank, TOP_K),
            "b_in_top20": 0 < b_gt_rank <= TOP_K,
            "r_in_top20": 0 < r_gt_rank <= TOP_K,
            "top1_changed": 1 if b_order[0] != r_order[0] else 0,
            "top20_overlap": len(set(b_top20.tolist()) & set(r_top20.tolist())),
        })

    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7_rows = [r for r in results if r["is_h7"]]
    same_rows = [r for r in results if r["is_same_artist"]]
    diff_rows = [r for r in results if not r["is_same_artist"]]
    h7_same = [r for r in h7_rows if r["is_same_artist"]]
    h7_diff = [r for r in h7_rows if not r["is_same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", results), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows),
                       ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        b = avg(rows, "b_ndcg")
        r = avg(rows, "r_ndcg")
        metrics[name] = {"n": len(rows), "baseline": b, "r76": r, "delta": r - b}

    h7_rec = sum(1 for r in h7_rows if r["r_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["r_in_top20"])
    h7_net = h7_rec - h7_lost
    top1_churn = sum(r["top1_changed"] for r in results)
    churn_per_80 = top1_churn / len(results) * 80
    overlap_mean = avg(results, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]
    da_d = metrics["diff_artist"]["delta"]

    gate_h7 = h7_d >= GATE_H7_DELTA
    gate_sa = sa_d >= GATE_SAME_DELTA
    gate_da = da_d >= GATE_DIFF_DELTA
    gate_net = h7_net > 0
    gate_overlap = overlap_mean >= GATE_OVERLAP_FLOOR
    gate_churn = churn_per_80 <= GATE_CHURN_CAP
    all_pass = gate_h7 and gate_sa and gate_da and gate_net and gate_overlap and gate_churn

    if all_pass:
        verdict = "PROCEED_PHASE_1"
    elif h7_d >= 0 and sa_d >= -0.005:
        verdict = "PROCEED_EXPLORATORY"
    else:
        verdict = "ARCHIVE"

    print(f"\n{ts()} === Results ===", flush=True)
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  baseline={m['baseline']:.4f}  "
              f"r76={m['r76']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_rec}  lost={h7_lost}  net={h7_net:+d}",
          flush=True)
    print(f"  top1_changed={top1_churn}  churn_per_80={churn_per_80:.2f}  "
          f"overlap_mean={overlap_mean:.2f}/20", flush=True)
    print(f"\n  Gates:", flush=True)
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})", flush=True)
    print(f"    same-art Δ ≥ -0.002:   {gate_sa}  ({sa_d:+.4f})", flush=True)
    print(f"    diff-art Δ ≥ 0:        {gate_da}  ({da_d:+.4f})", flush=True)
    print(f"    h7 net > 0:           {gate_net}  ({h7_net:+d})", flush=True)
    print(f"    overlap ≥ 14:         {gate_overlap}  ({overlap_mean:.2f})", flush=True)
    print(f"    churn ≤ 25:           {gate_churn}  ({churn_per_80:.2f})", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R76 Phase 0B — MLP residual ranker (fold-0 inner CV)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "hyperparams": {"hidden": args.hidden, "n_layers": N_LAYERS,
                        "dropout": args.dropout, "lr": args.lr,
                        "epochs": args.epochs, "weight_decay": args.wd,
                        "batch_cases": BATCH_CASES, "seed": SEED},
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_rec, "lost": h7_lost, "net": h7_net},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "gates": {
            "h7_delta_>=_+0.005": {"value": h7_d, "pass": gate_h7},
            "same_artist_delta_>=_-0.002": {"value": sa_d, "pass": gate_sa},
            "diff_artist_delta_>=_0": {"value": da_d, "pass": gate_da},
            "h7_net_>_0": {"value": h7_net, "pass": gate_net},
            "top20_overlap_>=_14": {"value": overlap_mean, "pass": gate_overlap},
            "top1_churn_per_80_<=_25": {"value": churn_per_80, "pass": gate_churn},
        },
        "dataset": str(DATASET_PATH),
        "n_cases": n_cases,
        "feature_count_mlp": len(MLP_FEATURES),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved {OUT_JSON}")

    md = [
        "# R76 Phase 0B — MLP residual ranker (fold-0 inner CV, CPU)",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"## Hyperparams",
        "",
        f"hidden={args.hidden} layers={N_LAYERS} dropout={args.dropout} "
        f"lr={args.lr} epochs={args.epochs} wd={args.wd}",
        "",
        "## Metrics",
        "",
        "| Subset | n | OOF R54c top-20 | R76 top-20 | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r76']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_rec}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap mean = {overlap_mean:.2f}/20",
        "",
        "## Gates",
        f"- h7 Δ ≥ +0.005: **{gate_h7}** ({h7_d:+.4f})",
        f"- same-artist Δ ≥ -0.002: **{gate_sa}** ({sa_d:+.4f})",
        f"- diff-artist Δ ≥ 0: **{gate_da}** ({da_d:+.4f})",
        f"- h7 net > 0: **{gate_net}** ({h7_net:+d})",
        f"- top-20 overlap ≥ 14: **{gate_overlap}** ({overlap_mean:.2f})",
        f"- top-1 churn /80 ≤ 25: **{gate_churn}** ({churn_per_80:.2f})",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
