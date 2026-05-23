#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R80 Phase 0B — Train listwise SetTransformer over top-300, eval fold-0.

Designed for Colab A100. ~1-2 hours, ~$10.

Loads:
- cache/r80/listwise_dataset_fold0.pkl (compact: ~100 MB)
- cache/r80/catalog_track_embs_fp16.npy (96 MB)
- cache/r80/catalog_track_ids.json
- cache/r80/eval_baseline.json

Model:
- Per-candidate projection: 47 numeric + 1024 track BGE + 1024 query BGE (broadcast)
  → 256 dim (shared MLP)
- 4-layer Transformer encoder: d=256, heads=8, ff=512, dropout=0.2
- Listwise softmax CE on cases with GT in top-300 (60% of cases provide signal)

Inner-CV: 5-way within fold-0 (1280 train / 320 test per inner fold).

Gates:
- h7 nDCG Δ ≥ +0.005 vs OOF R54c baseline
- same-artist Δ ≥ -0.002
- recovered > lost on h7
- top-1 churn /80 ≤ 25
- top-20 overlap ≥ 14/20

bf16 autocast, deterministic algorithms where possible.
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

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
import torch.nn as nn  # type: ignore[reportMissingImports]
import torch.nn.functional as F  # type: ignore[reportMissingImports]

torch.use_deterministic_algorithms(True, warn_only=True)
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET = REPO / "cache" / "r80" / "listwise_dataset_fold0.pkl"
DATASET_GZ = REPO / "cache" / "r80" / "listwise_dataset_fold0.pkl.gz"
CATALOG_EMBS = REPO / "cache" / "r80" / "catalog_track_embs_fp16.npy"
CATALOG_IDS = REPO / "cache" / "r80" / "catalog_track_ids.json"
BASELINE = REPO / "cache" / "r80" / "eval_baseline.json"

OUT_DIR = REPO / "cache" / "r80" / "phase0b"
OUT_JSON = REPO / "exp" / "eval" / "expR80_phase0b_result.json"
OUT_DOC = REPO / "docs" / "r80_phase0b_result.md"

POOL_K = 300
TOP_20 = 20
N_INNER_FOLDS = 5
SEED = 0

# Predeclared gates
GATE_H7_DELTA = 0.005
GATE_SAME_DELTA = -0.002
GATE_OVERLAP_FLOOR = 14
GATE_CHURN_CAP = 25

# Model hyperparams
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


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


class ListwiseRanker(nn.Module):
    def __init__(self, n_numeric=N_NUMERIC, emb_dim=EMB_DIM, hidden=HIDDEN,
                 n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        input_dim = n_numeric + emb_dim + emb_dim  # numeric + track_emb + query_emb
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.score_head = nn.Linear(hidden, 1)

    def forward(self, numeric, track_emb, query_emb):
        """
        numeric:    (B, K, 47)
        track_emb:  (B, K, 1024)
        query_emb:  (B, 1024) — broadcast to (B, K, 1024)
        Returns:    (B, K) scores
        """
        B, K, _ = numeric.shape
        q_broadcast = query_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, 1024)
        x = torch.cat([numeric, track_emb, q_broadcast], dim=-1)  # (B, K, input_dim)
        x = self.input_proj(x)  # (B, K, hidden)
        x = self.encoder(x)  # (B, K, hidden)
        scores = self.score_head(x).squeeze(-1)  # (B, K)
        return scores


def listwise_ce(scores, gt_positions, mask):
    """
    scores: (B, K)
    gt_positions: (B,) long — index of GT in pool (POOL_K = no-gt sentinel)
    mask: (B,) bool — True if GT in pool
    """
    valid = scores[mask]  # (Nv, K)
    targets = gt_positions[mask]  # (Nv,)
    if valid.shape[0] == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return F.cross_entropy(valid, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-cases", type=int, default=BATCH_CASES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R80 Phase 0B — listwise SetTransformer over top-300")
    print(f"  epochs={args.epochs} batch_cases={args.batch_cases} lr={args.lr} "
          f"hidden={args.hidden} layers={args.n_layers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  VRAM free: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")

    print(f"\n{ts()} Loading dataset ...", flush=True)
    if DATASET.exists():
        with open(DATASET, "rb") as f:
            data = pickle.load(f)
    elif DATASET_GZ.exists():
        import gzip
        print(f"  decompressing {DATASET_GZ} ...")
        with gzip.open(DATASET_GZ, "rb") as f:
            data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Neither {DATASET} nor {DATASET_GZ} found")
    cases = data["cases"]
    metadata = data["metadata"]
    print(f"  cases: {len(cases)}  pool_k={metadata['pool_k']}  numeric_feats={metadata['n_numeric_features']}")

    print(f"{ts()} Loading catalog (fp16) ...", flush=True)
    catalog_fp16 = np.load(CATALOG_EMBS)
    print(f"  catalog: {catalog_fp16.shape}  dtype={catalog_fp16.dtype}")
    catalog_t = torch.from_numpy(catalog_fp16).to(device)  # keep on GPU as fp16

    # Materialize per-case tensors lazily during training
    # For each case: track_emb is catalog_t[case['track_emb_idx']] (POOL_K, 1024)
    # numeric_features (POOL_K, 47), bge_query_emb (1024,)

    case_indices = list(range(len(cases)))

    # Inner-CV
    rng = random.Random(SEED)
    shuf = list(case_indices)
    rng.shuffle(shuf)
    inner_folds = [list(f) for f in np.array_split(shuf, N_INNER_FOLDS)]
    print(f"  inner-CV folds: {[len(f) for f in inner_folds]}")

    per_case_predictions = {}

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if USE_BF16 and device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))

    for fk in range(N_INNER_FOLDS):
        test_idx = set(inner_folds[fk])
        train_idx = [i for i in case_indices if i not in test_idx]
        print(f"\n{ts()} === Inner fold {fk}/{N_INNER_FOLDS}: train={len(train_idx)} test={len(test_idx)} ===",
              flush=True)

        model = ListwiseRanker(hidden=args.hidden, n_layers=args.n_layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if fk == 0:
            print(f"  model params: {n_params/1e6:.2f}M")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=WEIGHT_DECAY)
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
            total_loss = 0.0
            n_batches_done = 0
            for bstart in range(0, len(train_idx), args.batch_cases):
                batch = train_idx[bstart:bstart + args.batch_cases]
                # Build tensors for this batch
                numeric = torch.stack([
                    torch.from_numpy(cases[i]["numeric_features"]).to(device)
                    for i in batch
                ])  # (B, K, 47)
                track_idx = torch.stack([
                    torch.from_numpy(cases[i]["track_emb_idx"]).to(device).long()
                    for i in batch
                ])  # (B, K) — int32 indices into catalog
                track_idx = track_idx.clamp(min=0)  # -1 → 0 (will be masked later)
                track_emb = catalog_t[track_idx].float()  # (B, K, 1024) cast fp16→fp32
                query_emb = torch.stack([
                    torch.from_numpy(cases[i]["bge_query_emb"]).to(device)
                    for i in batch
                ])  # (B, 1024)

                gt_positions = torch.tensor(
                    [cases[i]["gt_pool_idx"] if cases[i]["gt_in_pool"] else 0
                     for i in batch], dtype=torch.long, device=device)
                mask = torch.tensor(
                    [cases[i]["gt_in_pool"] for i in batch],
                    dtype=torch.bool, device=device)

                with autocast_ctx:
                    scores = model(numeric, track_emb, query_emb)  # (B, K)
                    loss = listwise_ce(scores, gt_positions, mask)

                if mask.sum() == 0:
                    continue

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

                total_loss += float(loss.item())
                n_batches_done += 1
                global_step += 1

            avg_loss = total_loss / max(n_batches_done, 1)
            if (epoch + 1) % 4 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"    epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}  "
                      f"step={global_step}  lr={lr_now:.2e}", flush=True)

        # Eval on inner test
        model.eval()
        with torch.no_grad(), autocast_ctx:
            for i in test_idx:
                numeric = torch.from_numpy(cases[i]["numeric_features"]).to(device).unsqueeze(0)
                track_idx = torch.from_numpy(cases[i]["track_emb_idx"]).to(device).long().clamp(min=0).unsqueeze(0)
                track_emb = catalog_t[track_idx].float()
                query_emb = torch.from_numpy(cases[i]["bge_query_emb"]).to(device).unsqueeze(0)
                scores = model(numeric, track_emb, query_emb).squeeze(0).float().cpu().numpy()
                per_case_predictions[i] = scores.tolist()

    # ---- Final eval ----
    print(f"\n{ts()} === Final eval R80 vs OOF R54c baseline ===", flush=True)
    results = []
    for i in case_indices:
        c = cases[i]
        r80_scores = np.array(per_case_predictions[i])
        order_r = np.argsort(-r80_scores, kind="mergesort")
        r_top20 = order_r[:TOP_20]
        # GT rank in R80
        r_gt_rank = -1
        if c["gt_in_pool"]:
            pos = np.where(r_top20 == c["gt_pool_idx"])[0]
            if len(pos):
                r_gt_rank = int(pos[0]) + 1
        # Baseline rank = OOF R54c's own ordering = pool is already sorted by R54c score
        # so baseline top-20 = positions 0..19, gt rank in baseline = gt_pool_idx + 1 if < 20
        b_gt_rank = c["gt_pool_idx"] + 1 if (c["gt_in_pool"] and c["gt_pool_idx"] < TOP_20) else -1
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
            "top1_changed": 1 if r_top20[0] != 0 else 0,  # baseline top-1 is pool[0]
            "top20_overlap": len(set(r_top20.tolist()) & set(range(TOP_20))),
        })

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
        metrics[name] = {"n": len(rows), "baseline": b, "r80": r, "delta": r - b}

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
    gate_net = h7_net > 0
    gate_churn = churn_per_80 <= GATE_CHURN_CAP
    gate_overlap = overlap_mean >= GATE_OVERLAP_FLOOR
    all_pass = gate_h7 and gate_sa and gate_net and gate_churn and gate_overlap

    verdict = "PROCEED_PHASE_1" if all_pass else "ARCHIVE"

    print(f"\n{ts()} === Results ===")
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  baseline={m['baseline']:.4f}  "
              f"r80={m['r80']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_rec}  lost={h7_lost}  net={h7_net:+d}")
    print(f"  top1_churn={top1_churn}  per_80={churn_per_80:.2f}  overlap_mean={overlap_mean:.2f}/20")
    print(f"\n  Gates:")
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})")
    print(f"    same-art Δ ≥ -0.002:   {gate_sa}  ({sa_d:+.4f})")
    print(f"    h7 net > 0:           {gate_net}  ({h7_net:+d})")
    print(f"    churn ≤ 25:           {gate_churn}  ({churn_per_80:.2f})")
    print(f"    overlap ≥ 14:         {gate_overlap}  ({overlap_mean:.2f})")
    print(f"  VERDICT: {verdict}")

    out = {
        "experiment": "R80 Phase 0B — listwise transformer fold-0",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "hyperparams": vars(args),
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_rec, "lost": h7_lost, "net": h7_net},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "gates": {
            "h7_delta_>=_+0.005": {"value": h7_d, "pass": gate_h7},
            "same_artist_delta_>=_-0.002": {"value": sa_d, "pass": gate_sa},
            "h7_net_>_0": {"value": h7_net, "pass": gate_net},
            "churn_<=_25": {"value": churn_per_80, "pass": gate_churn},
            "overlap_>=_14": {"value": overlap_mean, "pass": gate_overlap},
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved → {OUT_JSON}")

    md = [
        "# R80 Phase 0B — listwise transformer fold-0",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        f"## Verdict: **{verdict}**",
        "",
        "## Metrics",
        "",
        "| Subset | n | OOF R54c | R80 listwise | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r80']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_rec}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap = {overlap_mean:.2f}/20",
        "",
        "## Gates",
        f"- h7 Δ ≥ +0.005: **{gate_h7}** ({h7_d:+.4f})",
        f"- same-artist Δ ≥ -0.002: **{gate_sa}** ({sa_d:+.4f})",
        f"- h7 net > 0: **{gate_net}** ({h7_net:+d})",
        f"- top-1 churn /80 ≤ 25: **{gate_churn}** ({churn_per_80:.2f})",
        f"- top-20 overlap ≥ 14: **{gate_overlap}** ({overlap_mean:.2f})",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved → {OUT_DOC}")


if __name__ == "__main__":
    main()
