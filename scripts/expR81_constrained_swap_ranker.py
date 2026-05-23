#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R81 — Constrained-swap ranker on R80 dataset.

Hypothesis: R80 recovered 6 h7 cases but lost 17. The 17 losses came from
the unconstrained model demoting candidates R54c had correctly placed.
R81 tests whether a CONSTRAINED variant can keep the 6 recoveries while
blocking most of the 17 losses.

Two constraints:
1. **Training loss**: listwise CE + λ × anchor penalty
   - Anchor penalty: max(0, R54c_norm_score - R81_score) summed over the
     top-K_anchor candidates by R54c rank. Penalizes demoting strong R54c
     candidates below their R54c-normalized score.

2. **Inference**: conservative swap heuristic
   - Start with R54c top-20 as anchor
   - For each candidate at R54c rank 21..30, compute promotion confidence
     = R81_score - (worst R81_score among R54c top-20)
   - If confidence > THRESHOLD AND fewer than MAX_SWAPS swaps used:
     swap into top-20 at the position dictated by R81 rank
   - MAX_SWAPS = 2 per case

Loads same cache/r80/listwise_dataset_fold0.pkl.gz + catalog.

Trains 1 model (no inner CV — we apply the constraints uniformly).
Evaluates on fold-0 OOF against R54c baseline.

Gates:
- h7 recovered > h7 lost (the primary hypothesis)
- same-artist Δ ≥ -0.002
- top-1 churn /80 ≤ 25
- h7 nDCG Δ ≥ +0.005 (kept from R80 for consistency)
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
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
import torch.nn as nn  # type: ignore[reportMissingImports]
import torch.nn.functional as F  # type: ignore[reportMissingImports]

torch.use_deterministic_algorithms(True, warn_only=True)
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET = REPO / "cache" / "r80" / "listwise_dataset_fold0.pkl"
DATASET_GZ = REPO / "cache" / "r80" / "listwise_dataset_fold0.pkl.gz"
CATALOG_EMBS = REPO / "cache" / "r80" / "catalog_track_embs_fp16.npy"
CATALOG_IDS = REPO / "cache" / "r80" / "catalog_track_ids.json"

OUT_JSON = REPO / "exp" / "eval" / "expR81_constrained_swap.json"
OUT_DOC = REPO / "docs" / "r81_constrained_swap_result.md"

POOL_K = 300
TOP_20 = 20
TOP_30 = 30

# Predeclared gates (primary: recovered > lost)
GATE_RECOV_GT_LOST = True  # h7 recovered > lost
GATE_H7_DELTA = 0.005
GATE_SAME_DELTA = -0.002
GATE_CHURN_CAP = 25

# Model
N_NUMERIC = 47
EMB_DIM = 1024
HIDDEN = 256
N_LAYERS = 4
N_HEADS = 8
FF_DIM = 512
DROPOUT = 0.2
LR = 1e-4
WEIGHT_DECAY = 0.01
N_EPOCHS = 20
BATCH_CASES = 8
USE_BF16 = True

# Constraints
ANCHOR_K = 10            # penalize demoting candidates at R54c rank <= K
ANCHOR_LAMBDA = 0.5      # weight for anchor penalty in loss
MAX_SWAPS = 2            # max swaps from R54c top-20 to R54c[20:30]
SWAP_CONF_THRESHOLD = 0.0  # R81_score must beat worst R54c-top-20 R81_score by >= this


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


class ConstrainedRanker(nn.Module):
    def __init__(self, n_numeric=N_NUMERIC, emb_dim=EMB_DIM, hidden=HIDDEN,
                 n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        input_dim = n_numeric + emb_dim + emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.score_head = nn.Linear(hidden, 1)

    def forward(self, numeric, track_emb, query_emb):
        B, K, _ = numeric.shape
        q_b = query_emb.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([numeric, track_emb, q_b], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.score_head(x).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-cases", type=int, default=BATCH_CASES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--anchor-k", type=int, default=ANCHOR_K)
    parser.add_argument("--anchor-lambda", type=float, default=ANCHOR_LAMBDA)
    parser.add_argument("--max-swaps", type=int, default=MAX_SWAPS)
    parser.add_argument("--swap-threshold", type=float, default=SWAP_CONF_THRESHOLD)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R81 — constrained-swap ranker")
    print(f"  train: epochs={args.epochs} batch={args.batch_cases} lr={args.lr}")
    print(f"  loss: anchor_k={args.anchor_k} anchor_lambda={args.anchor_lambda}")
    print(f"  infer: max_swaps={args.max_swaps} threshold={args.swap_threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")

    # Load data
    print(f"\n{ts()} Loading dataset ...", flush=True)
    if DATASET.exists():
        with open(DATASET, "rb") as f:
            data = pickle.load(f)
    elif DATASET_GZ.exists():
        import gzip
        with gzip.open(DATASET_GZ, "rb") as f:
            data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither {DATASET} nor {DATASET_GZ}")
    cases = data["cases"]
    print(f"  cases: {len(cases)}")

    print(f"{ts()} Loading catalog ...", flush=True)
    catalog_fp16 = np.load(CATALOG_EMBS)
    catalog_t = torch.from_numpy(catalog_fp16).to(device)
    print(f"  catalog: {catalog_fp16.shape}")

    case_indices = list(range(len(cases)))

    # 5-way inner CV
    rng = random.Random(SEED)
    shuf = list(case_indices)
    rng.shuffle(shuf)
    N_INNER = 5
    inner_folds = [list(f) for f in np.array_split(shuf, N_INNER)]

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if USE_BF16 and device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))

    per_case_predictions = {}

    for fk in range(N_INNER):
        test_idx = set(inner_folds[fk])
        train_idx = [i for i in case_indices if i not in test_idx]
        print(f"\n{ts()} === Inner fold {fk}/{N_INNER}: "
              f"train={len(train_idx)} test={len(test_idx)} ===", flush=True)

        model = ConstrainedRanker().to(device)
        if fk == 0:
            print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        n_batches = (len(train_idx) + args.batch_cases - 1) // args.batch_cases
        total_steps = args.epochs * n_batches
        warmup = int(total_steps * 0.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: min(1.0, step / max(warmup, 1)) if step < warmup
            else max(0.0, 1.0 - (step - warmup) / max(total_steps - warmup, 1)),
        )

        model.train()
        global_step = 0
        for epoch in range(args.epochs):
            random.shuffle(train_idx)
            tot_ce = 0.0; tot_anc = 0.0; n_bd = 0
            for bs in range(0, len(train_idx), args.batch_cases):
                batch = train_idx[bs:bs + args.batch_cases]
                numeric = torch.stack([
                    torch.from_numpy(cases[i]["numeric_features"]).to(device)
                    for i in batch])
                track_idx = torch.stack([
                    torch.from_numpy(cases[i]["track_emb_idx"]).to(device).long().clamp(min=0)
                    for i in batch])
                track_emb = catalog_t[track_idx].float()
                query_emb = torch.stack([
                    torch.from_numpy(cases[i]["bge_query_emb"]).to(device)
                    for i in batch])
                gt_positions = torch.tensor(
                    [cases[i]["gt_pool_idx"] if cases[i]["gt_in_pool"] else 0
                     for i in batch], dtype=torch.long, device=device)
                mask = torch.tensor(
                    [cases[i]["gt_in_pool"] for i in batch],
                    dtype=torch.bool, device=device)
                # R54c scores per candidate (column 37 of numeric)
                r54c_scores = numeric[:, :, 37]  # (B, K)

                with autocast_ctx:
                    scores = model(numeric, track_emb, query_emb)
                    # listwise CE on cases with GT
                    if mask.sum() > 0:
                        ce_loss = F.cross_entropy(scores[mask], gt_positions[mask])
                    else:
                        ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    # Anchor penalty: for top-K_anchor R54c-ranked candidates,
                    # penalize when R81 score drops below the R54c normalized score.
                    # Pool is already sorted by R54c desc; positions 0..K_anchor-1
                    # are the top-K_anchor.
                    top_k_scores = scores[:, :args.anchor_k]  # (B, K_anchor)
                    top_k_r54c = r54c_scores[:, :args.anchor_k]
                    # Normalize R54c scores to be comparable to model scores
                    # (use per-case z-score)
                    r54c_mean = r54c_scores.mean(dim=1, keepdim=True)
                    r54c_std = r54c_scores.std(dim=1, keepdim=True) + 1e-6
                    r54c_norm = (top_k_r54c - r54c_mean) / r54c_std
                    score_mean = scores.mean(dim=1, keepdim=True)
                    score_std = scores.std(dim=1, keepdim=True) + 1e-6
                    score_norm = (top_k_scores - score_mean) / score_std
                    # Penalty: max(0, r54c_norm - score_norm) — penalize when
                    # model's z-score is below R54c's z-score for top candidates
                    anchor_pen = F.relu(r54c_norm - score_norm).mean()
                    loss = ce_loss + args.anchor_lambda * anchor_pen

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                tot_ce += float(ce_loss.item())
                tot_anc += float(anchor_pen.item())
                n_bd += 1
                global_step += 1

            if (epoch + 1) % 4 == 0:
                print(f"    epoch {epoch+1}/{args.epochs}  ce={tot_ce/n_bd:.4f} "
                      f"anc={tot_anc/n_bd:.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)

        # Eval on test set: collect scores
        model.eval()
        with torch.no_grad(), autocast_ctx:
            for i in test_idx:
                numeric = torch.from_numpy(cases[i]["numeric_features"]).to(device).unsqueeze(0)
                track_idx = torch.from_numpy(cases[i]["track_emb_idx"]).to(device).long().clamp(min=0).unsqueeze(0)
                track_emb = catalog_t[track_idx].float()
                query_emb = torch.from_numpy(cases[i]["bge_query_emb"]).to(device).unsqueeze(0)
                scores = model(numeric, track_emb, query_emb).squeeze(0).float().cpu().numpy()
                per_case_predictions[i] = scores

    # ---- Constrained-swap inference + eval ----
    print(f"\n{ts()} === Constrained-swap inference + eval ===")
    print(f"  max_swaps={args.max_swaps}  threshold={args.swap_threshold}")

    results = []
    n_swaps_done = 0
    swap_dist = [0, 0, 0]  # 0/1/2 swaps
    for i in case_indices:
        c = cases[i]
        r81_scores = per_case_predictions[i]  # (300,) — pool ALREADY sorted by R54c

        # Baseline = R54c top-20 = positions 0..19 in pool (since pool is R54c-sorted)
        baseline_top20 = list(range(TOP_20))
        b_gt_rank = c["gt_pool_idx"] + 1 if (c["gt_in_pool"] and c["gt_pool_idx"] < TOP_20) else -1

        # Constrained swap: for R54c positions 20..29, check R81 promotion confidence
        # Confidence = R81[pos] - worst_R81_in_top20
        r81_top20_min = r81_scores[:TOP_20].min()
        candidate_positions = list(range(TOP_20, min(TOP_30, POOL_K)))
        # Sort candidates by their R81 score descending
        candidates_with_scores = sorted(
            candidate_positions, key=lambda p: -r81_scores[p])

        # Apply up to MAX_SWAPS swaps
        r81_top20 = list(baseline_top20)
        # Drop the lowest-R81-score positions from top-20 to make room
        # but only if a candidate beats them by threshold
        used_swaps = 0
        for cand_pos in candidates_with_scores:
            if used_swaps >= args.max_swaps:
                break
            cand_score = r81_scores[cand_pos]
            # Find worst R81-scoring position in current top-20
            worst_idx_in_top20 = int(np.argmin([r81_scores[p] for p in r81_top20]))
            worst_pos = r81_top20[worst_idx_in_top20]
            worst_score = r81_scores[worst_pos]
            confidence = cand_score - worst_score
            if confidence >= args.swap_threshold:
                # Swap: replace worst_pos with cand_pos
                r81_top20[worst_idx_in_top20] = cand_pos
                used_swaps += 1
            # else: no more candidates beat threshold (since sorted desc)
            else:
                break

        swap_dist[used_swaps] += 1
        n_swaps_done += used_swaps

        # Compute R81 nDCG (need to know GT rank in R81 top-20 ordering)
        # R81 top-20 = r81_top20 (in some order). Sort by R81 score for the final ordering.
        r81_top20_sorted = sorted(r81_top20, key=lambda p: -r81_scores[p])
        r_gt_rank = -1
        if c["gt_in_pool"]:
            try:
                r_gt_rank = r81_top20_sorted.index(c["gt_pool_idx"]) + 1
            except ValueError:
                r_gt_rank = -1

        # Top-1 change
        top1_changed = 1 if r81_top20_sorted[0] != 0 else 0
        top20_overlap = len(set(r81_top20_sorted) & set(baseline_top20))

        results.append({
            "case_idx": i,
            "is_h7": c["is_h7"],
            "is_same_artist": c["is_same_artist"],
            "b_gt_rank": b_gt_rank,
            "r_gt_rank": r_gt_rank,
            "b_in_top20": 0 < b_gt_rank <= TOP_20,
            "r_in_top20": 0 < r_gt_rank <= TOP_20,
            "b_ndcg": ndcg_at_k(b_gt_rank, TOP_20),
            "r_ndcg": ndcg_at_k(r_gt_rank, TOP_20),
            "top1_changed": top1_changed,
            "top20_overlap": top20_overlap,
            "n_swaps": used_swaps,
        })

    print(f"  swap distribution: 0={swap_dist[0]} 1={swap_dist[1]} 2={swap_dist[2]}")
    print(f"  total swaps across all cases: {n_swaps_done}")

    # Metrics
    def avg(rs, key, where=None):
        if where is not None:
            rs = [r for r in rs if where(r)]
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    h7_rows = [r for r in results if r["is_h7"]]
    same_rows = [r for r in results if r["is_same_artist"]]
    diff_rows = [r for r in results if not r["is_same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", results), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows)]:
        b = avg(rows, "b_ndcg")
        r = avg(rows, "r_ndcg")
        metrics[name] = {"n": len(rows), "baseline": b, "r81": r, "delta": r - b}

    h7_rec = sum(1 for r in h7_rows if r["r_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["r_in_top20"])
    h7_net = h7_rec - h7_lost
    top1_churn = sum(r["top1_changed"] for r in results)
    churn_per_80 = top1_churn / len(results) * 80
    overlap_mean = avg(results, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]

    gate_h7 = h7_d >= GATE_H7_DELTA
    gate_sa = sa_d >= GATE_SAME_DELTA
    gate_recov_lost = h7_rec > h7_lost  # PRIMARY
    gate_churn = churn_per_80 <= GATE_CHURN_CAP

    # Primary gate: recovered > lost on h7. Secondary: same-artist + churn.
    all_pass = gate_recov_lost and gate_sa and gate_churn and gate_h7
    primary_pass = gate_recov_lost and gate_sa and gate_churn
    if all_pass:
        verdict = "PROCEED_PHASE_1"
    elif primary_pass and h7_d >= 0:
        verdict = "EXPLORATORY"
    else:
        verdict = "ARCHIVE"

    print(f"\n{ts()} === Results (constrained swap) ===")
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  baseline={m['baseline']:.4f}  "
              f"r81={m['r81']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_rec}  lost={h7_lost}  net={h7_net:+d}")
    print(f"  top1_churn={top1_churn}  per_80={churn_per_80:.2f}  overlap_mean={overlap_mean:.2f}/20")
    print(f"\n  Gates:")
    print(f"    h7 recovered > lost:  {gate_recov_lost}  ({h7_rec} > {h7_lost})  [PRIMARY]")
    print(f"    same-art Δ ≥ -0.002:   {gate_sa}  ({sa_d:+.4f})")
    print(f"    churn ≤ 25:           {gate_churn}  ({churn_per_80:.2f})")
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})")
    print(f"  VERDICT: {verdict}")

    out = {
        "experiment": "R81 — constrained-swap ranker",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "hyperparams": vars(args),
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_rec, "lost": h7_lost, "net": h7_net},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "swap_distribution": {"0_swaps": swap_dist[0], "1_swap": swap_dist[1], "2_swaps": swap_dist[2]},
        "total_swaps": n_swaps_done,
        "gates": {
            "primary_recovered_>_lost": {"value": [h7_rec, h7_lost], "pass": gate_recov_lost},
            "same_artist_delta_>=_-0.002": {"value": sa_d, "pass": gate_sa},
            "churn_<=_25": {"value": churn_per_80, "pass": gate_churn},
            "h7_delta_>=_+0.005": {"value": h7_d, "pass": gate_h7},
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved → {OUT_JSON}")

    md = [
        "# R81 — Constrained-swap ranker",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"## Hyperparams",
        "",
        f"anchor_k={args.anchor_k} anchor_lambda={args.anchor_lambda} "
        f"max_swaps={args.max_swaps} swap_threshold={args.swap_threshold}",
        "",
        "## Metrics",
        "",
        "| Subset | n | OOF R54c | R81 (constrained) | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r81']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_rec}, lost={h7_lost}, net={h7_net:+d}  **(PRIMARY GATE: rec > lost = {gate_recov_lost})**",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap = {overlap_mean:.2f}/20",
        f"- swap distribution: 0={swap_dist[0]}  1={swap_dist[1]}  2={swap_dist[2]}",
        "",
        "## Gates",
        f"- h7 recovered > lost: **{gate_recov_lost}** ({h7_rec} > {h7_lost}) [PRIMARY]",
        f"- same-artist Δ ≥ -0.002: **{gate_sa}** ({sa_d:+.4f})",
        f"- top-1 churn /80 ≤ 25: **{gate_churn}** ({churn_per_80:.2f})",
        f"- h7 nDCG Δ ≥ +0.005: **{gate_h7}** ({h7_d:+.4f})",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved → {OUT_DOC}")


if __name__ == "__main__":
    main()
