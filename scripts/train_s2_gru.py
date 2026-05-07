#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""S2 Phase 2: GRU next-track model with trainable ALS-initialized embeddings.

Architecture:
  track_seq → Embedding(ALS init) → GRU → hidden
  current_utt → E5(frozen, cached) → Linear → utt_proj
  [hidden ; utt_proj] → MLP → output (128-dim)
  score = output @ all_track_embs.T

Loss: InfoNCE with in-batch + BM25 hard negatives, learnable temperature.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import DownloadConfig, load_dataset
from implicit.als import AlternatingLeastSquares

CACHE_DIR = REPO_ROOT / "cache" / "seq_model"
UTT_NPY = CACHE_DIR / "utt_embeddings.npy"
UTT_INDEX = CACHE_DIR / "utt_embedding_index.json"
HN_PATH = CACHE_DIR / "hard_negatives.pkl"
RUN_DIR = CACHE_DIR / "runs_s2"

EMB_DIM = 128
UTT_DIM = 384
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 512
LR_MODEL = 1e-3
LR_EMB = 1e-4
EPOCHS = 15
PATIENCE = 3
N_HARD_NEG = 8
VAL_FRAC = 0.1
ALS_FACTORS = 128
ALS_ALPHA = 100
ALS_REG = 0.05
ALS_ITER = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ============ Model ============

class S2GRU(nn.Module):
    def __init__(self, n_tracks, emb_dim=EMB_DIM, utt_dim=UTT_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.n_tracks = n_tracks
        self.emb_dim = emb_dim
        # +1 for no-history token at index n_tracks
        self.track_emb = nn.Embedding(n_tracks + 1, emb_dim)
        self.utt_proj = nn.Linear(utt_dim, emb_dim)
        self.gru = nn.GRU(
            emb_dim, hidden_dim, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))

    def forward(self, track_seq, utt_emb, seq_lengths):
        """
        track_seq: (B, max_len) int64 — track indices, padded with 0
        utt_emb: (B, utt_dim) float32 — current utterance embedding
        seq_lengths: (B,) int64 — actual sequence lengths (>=1)
        Returns: (B, emb_dim) output vectors
        """
        te = self.track_emb(track_seq)  # (B, max_len, emb_dim)

        # Run GRU on padded sequences (unidirectional, so output at
        # position t only depends on 0..t — padding after doesn't affect it)
        gru_out, _ = self.gru(te)  # (B, max_len, hidden_dim)

        # Gather last valid position per example
        B = track_seq.shape[0]
        last_idx = (seq_lengths - 1).long()
        h = gru_out[torch.arange(B, device=gru_out.device), last_idx]  # (B, hidden_dim)

        u = self.utt_proj(utt_emb)  # (B, emb_dim)
        combined = torch.cat([h, u], dim=-1)  # (B, hidden_dim + emb_dim)
        return self.output_head(combined)  # (B, emb_dim)

    def get_item_embeddings(self):
        return self.track_emb.weight[:self.n_tracks]  # exclude no-history token


# ============ Dataset ============

class S2Dataset(Dataset):
    def __init__(self, examples, track_to_idx, utt_embs, utt_index,
                 hard_negatives, n_tracks):
        self.examples = examples
        self.track_to_idx = track_to_idx
        self.utt_embs = utt_embs
        self.utt_index = utt_index
        self.hard_negatives = hard_negatives
        self.n_tracks = n_tracks
        self.no_history_id = n_tracks

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sid, turn, track_history, target_tid = self.examples[idx]

        # Build track sequence
        if track_history:
            seq = [self.track_to_idx[t] for t in track_history
                   if t in self.track_to_idx]
        else:
            seq = []
        if not seq:
            seq = [self.no_history_id]

        # Utterance embedding
        utt_key = f"{sid}:{turn}"
        utt_idx = self.utt_index.get(utt_key)
        if utt_idx is not None and utt_idx < len(self.utt_embs):
            utt = self.utt_embs[utt_idx].copy()
        else:
            utt = np.zeros(UTT_DIM, dtype=np.float32)

        # Target track index
        target = self.track_to_idx.get(target_tid, -1)

        # Hard negatives
        hn_tids = self.hard_negatives.get((sid, turn), [])
        hn = [self.track_to_idx[t] for t in hn_tids
              if t in self.track_to_idx and self.track_to_idx[t] != target]
        hn = hn[:N_HARD_NEG]

        return {
            "track_seq": np.array(seq, dtype=np.int64),
            "utt": utt,
            "target": target,
            "hard_neg": np.array(hn, dtype=np.int64) if hn else np.array([], dtype=np.int64),
            "seq_len": len(seq),
        }


def collate_fn(batch):
    B = len(batch)
    max_seq = max(b["seq_len"] for b in batch)
    max_hn = max(len(b["hard_neg"]) for b in batch)
    max_hn = max(max_hn, 1)

    track_seq = torch.zeros(B, max_seq, dtype=torch.long)
    utt = torch.zeros(B, UTT_DIM, dtype=torch.float32)
    targets = torch.zeros(B, dtype=torch.long)
    seq_lens = torch.zeros(B, dtype=torch.long)
    hard_neg = torch.full((B, max_hn), -1, dtype=torch.long)

    for i, b in enumerate(batch):
        sl = b["seq_len"]
        track_seq[i, :sl] = torch.from_numpy(b["track_seq"])
        utt[i] = torch.from_numpy(b["utt"])
        targets[i] = b["target"]
        seq_lens[i] = sl
        hn = b["hard_neg"]
        if len(hn) > 0:
            hard_neg[i, :len(hn)] = torch.from_numpy(hn)

    return track_seq, utt, targets, seq_lens, hard_neg


# ============ Training ============

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for track_seq, utt, targets, seq_lens, hard_neg in loader:
        track_seq = track_seq.to(device)
        utt = utt.to(device)
        targets = targets.to(device)
        seq_lens = seq_lens.to(device)
        hard_neg = hard_neg.to(device)

        # Skip examples with invalid targets
        valid = targets >= 0
        if valid.sum() == 0:
            continue

        track_seq = track_seq[valid]
        utt = utt[valid]
        targets = targets[valid]
        seq_lens = seq_lens[valid]
        hard_neg = hard_neg[valid]

        B = track_seq.shape[0]
        output = model(track_seq, utt, seq_lens)  # (B, D)

        # Normalize output and embeddings for stable training
        output_norm = F.normalize(output, dim=-1)
        pos_emb = F.normalize(model.track_emb(targets), dim=-1)  # (B, D)

        logit_scale = model.logit_scale.exp().clamp(max=100.0)

        # In-batch scores: (B, B) — diagonal is positive
        in_batch = (output_norm @ pos_emb.T) * logit_scale

        # Hard negative scores
        hn_mask = hard_neg >= 0  # (B, max_hn)
        safe_hn = hard_neg.clamp(min=0)
        hn_emb = F.normalize(model.track_emb(safe_hn), dim=-1)  # (B, max_hn, D)
        hn_scores = (output_norm.unsqueeze(1) * hn_emb).sum(-1) * logit_scale  # (B, max_hn)
        hn_scores = hn_scores.masked_fill(~hn_mask, -1e9)

        # Concatenate: [in_batch | hard_neg]
        logits = torch.cat([in_batch, hn_scores], dim=1)  # (B, B + max_hn)

        # Labels: positive for example i is at position i (diagonal of in_batch)
        labels = torch.arange(B, device=device)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, n_tracks):
    model.eval()
    total_hit20 = 0
    total_ndcg = 0.0
    total = 0

    item_emb = F.normalize(model.get_item_embeddings(), dim=-1)  # (n_tracks, D)

    for track_seq, utt, targets, seq_lens, _ in loader:
        track_seq = track_seq.to(device)
        utt = utt.to(device)
        targets = targets.to(device)
        seq_lens = seq_lens.to(device)

        valid = targets >= 0
        if valid.sum() == 0:
            continue

        track_seq = track_seq[valid]
        utt = utt[valid]
        targets = targets[valid]
        seq_lens = seq_lens[valid]

        B = track_seq.shape[0]
        output = model(track_seq, utt, seq_lens)
        output_norm = F.normalize(output, dim=-1)

        logit_scale = model.logit_scale.exp().clamp(max=100.0)
        scores = (output_norm @ item_emb.T) * logit_scale  # (B, n_tracks)

        # Exclude played tracks
        for b_idx in range(B):
            sl = seq_lens[b_idx].item()
            played = track_seq[b_idx, :sl]
            played = played[played < n_tracks]  # exclude no-history token
            if len(played) > 0:
                scores[b_idx, played] = -torch.inf

        # Top-20 hit rate and nDCG
        _, top20 = scores.topk(20, dim=1)
        for b_idx in range(B):
            gt = targets[b_idx].item()
            ranking = top20[b_idx].tolist()
            if gt in ranking:
                total_hit20 += 1
                rank = ranking.index(gt)
                total_ndcg += 1.0 / np.log2(rank + 2)
            total += 1

    hit_rate = total_hit20 / max(total, 1)
    ndcg = total_ndcg / max(total, 1)
    return hit_rate, ndcg, total


# ============ Data Preparation ============

def build_als_factors(matrix):
    """Train ALS and return item factors as numpy array."""
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS, alpha=ALS_ALPHA, regularization=ALS_REG,
        iterations=ALS_ITER, random_state=42, use_gpu=False,
    )
    model.fit(matrix)
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    elif not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    return factors


def build_interaction_matrix():
    """Build session×track CSR matrix from train conversations."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]

    track_set = set()
    session_tracks = []

    for item in train:
        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                track_set.add(tid)
        session_tracks.append(tracks)

    track_ids = sorted(track_set)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    rows, cols, vals = [], [], []
    for si, tracks in enumerate(session_tracks):
        for tid in tracks:
            rows.append(si)
            cols.append(track_to_idx[tid])
            vals.append(1.0)

    matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(session_tracks), len(track_ids)),
        dtype=np.float32,
    )
    return matrix, track_ids, track_to_idx


def build_training_examples(track_to_idx):
    """Extract (session_id, turn, track_history, target) tuples from train data."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]

    examples = []
    for item in train:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))

        # Build mapping: turn_number → music track
        music_at_turn = {}
        for c in convs:
            if c["role"] == "music":
                music_at_turn[int(c["turn_number"])] = str(c["content"]).strip()

        # For each user turn, create a training example
        for c in convs:
            if c["role"] != "user":
                continue
            turn = int(c["turn_number"])
            gt = music_at_turn.get(turn)
            if gt is None or gt not in track_to_idx:
                continue

            # Collect played tracks before this turn
            history = [c2 for c2 in convs if int(c2["turn_number"]) < turn]
            played = [str(c2["content"]).strip() for c2 in history if c2["role"] == "music"]

            examples.append((sid, turn, played, gt))

    return examples


# ============ Main ============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze_emb", action="store_true",
                        help="Freeze track embeddings (train only GRU + output head)")
    parser.add_argument("--lr_emb", type=float, default=LR_EMB,
                        help=f"Embedding LR (default {LR_EMB})")
    parser.add_argument("--lr_model", type=float, default=LR_MODEL,
                        help=f"Model LR (default {LR_MODEL})")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    args = parser.parse_args()

    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{ts()} Device: {device}")
    if args.freeze_emb:
        print(f"  Mode: FROZEN embeddings")
    else:
        print(f"  Mode: trainable embeddings (lr_emb={args.lr_emb})")

    # --- Build ALS factors + expanded vocabulary ---
    print(f"{ts()} Building interaction matrix and ALS factors...", flush=True)
    matrix, als_track_ids, als_track_to_idx = build_interaction_matrix()
    n_als = len(als_track_ids)
    print(f"  ALS: {n_als} tracks, {matrix.shape[0]} sessions, {matrix.nnz} interactions")

    als_factors = build_als_factors(matrix)
    print(f"  ALS factors shape: {als_factors.shape}")

    # Expand vocabulary with CF-BPR tracks for better GT coverage
    cfbpr_ids_path = REPO_ROOT / "cache" / "track_sim" / "cf-bpr" / "track_ids.json"
    if cfbpr_ids_path.exists():
        with open(cfbpr_ids_path) as f:
            cfbpr_track_ids = json.load(f)
        cfbpr_vecs_path = REPO_ROOT / "cache" / "track_sim" / "cf-bpr" / "vectors.npy"
        cfbpr_vecs = np.load(cfbpr_vecs_path) if cfbpr_vecs_path.exists() else None

        # Add tracks not in ALS vocabulary
        extra_ids = [t for t in cfbpr_track_ids if t not in als_track_to_idx]
        track_ids = als_track_ids + sorted(extra_ids)
        track_to_idx = {t: i for i, t in enumerate(track_ids)}
        n_tracks = len(track_ids)

        # Build expanded factor matrix: ALS for known, CF-BPR projected for extra
        expanded_factors = np.zeros((n_tracks, ALS_FACTORS), dtype=np.float32)
        expanded_factors[:n_als] = als_factors

        if cfbpr_vecs is not None and len(extra_ids) > 0:
            cfbpr_to_idx = {t: i for i, t in enumerate(cfbpr_track_ids)}
            # CF-BPR is 128-dim, same as ALS — use directly as initialization
            for tid in extra_ids:
                ci = cfbpr_to_idx.get(tid)
                if ci is not None:
                    expanded_factors[track_to_idx[tid]] = cfbpr_vecs[ci]

        als_factors = expanded_factors
        print(f"  Expanded vocab: {n_tracks} tracks ({n_tracks - n_als} extra from CF-BPR)")
    else:
        track_ids = als_track_ids
        track_to_idx = als_track_to_idx
        n_tracks = n_als

    # --- Load caches ---
    print(f"{ts()} Loading caches...", flush=True)
    utt_embs = np.load(UTT_NPY)
    with open(UTT_INDEX) as f:
        utt_index = json.load(f)
    with open(HN_PATH, "rb") as f:
        hard_negatives = pickle.load(f)
    print(f"  Utt: {utt_embs.shape}, HN: {len(hard_negatives)} entries")

    # --- Build training examples ---
    print(f"{ts()} Building training examples...", flush=True)
    all_examples = build_training_examples(track_to_idx)
    print(f"  {len(all_examples)} total examples")

    # Train/val split by session
    rng = np.random.RandomState(42)
    all_sessions = sorted(set(ex[0] for ex in all_examples))
    n_val_sessions = max(1, int(len(all_sessions) * VAL_FRAC))
    val_sessions = set(rng.choice(all_sessions, n_val_sessions, replace=False))
    train_examples = [ex for ex in all_examples if ex[0] not in val_sessions]
    val_examples = [ex for ex in all_examples if ex[0] in val_sessions]
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)} "
          f"({len(val_sessions)} val sessions)")

    # Count hist_0 examples
    n_hist0_train = sum(1 for ex in train_examples if not ex[2])
    n_hist0_val = sum(1 for ex in val_examples if not ex[2])
    print(f"  hist_0: train={n_hist0_train}, val={n_hist0_val}")

    # --- Create datasets ---
    train_ds = S2Dataset(train_examples, track_to_idx, utt_embs, utt_index,
                         hard_negatives, n_tracks)
    val_ds = S2Dataset(val_examples, track_to_idx, utt_embs, utt_index,
                       hard_negatives, n_tracks)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # --- Create model ---
    model = S2GRU(n_tracks, dropout=args.dropout).to(device)

    # Initialize track embeddings from ALS/CF-BPR factors
    with torch.no_grad():
        model.track_emb.weight[:n_tracks] = torch.from_numpy(als_factors).float()
        nn.init.normal_(model.track_emb.weight[n_tracks:], std=0.01)

    if args.freeze_emb:
        model.track_emb.weight.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_params = model.track_emb.weight.numel()
    print(f"{ts()} Model: {total_params:,} params ({emb_params:,} embedding, "
          f"{total_params - emb_params:,} other), {trainable_params:,} trainable")

    # Optimizer: skip frozen parameters
    param_groups = []
    if not args.freeze_emb:
        param_groups.append({"params": model.track_emb.parameters(), "lr": args.lr_emb})
    param_groups.append({
        "params": [p for n, p in model.named_parameters()
                   if "track_emb" not in n and p.requires_grad],
        "lr": args.lr_model,
    })
    optimizer = torch.optim.Adam(param_groups)

    # --- Training loop ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "emb_dim": EMB_DIM, "hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS,
        "dropout": args.dropout, "batch_size": BATCH_SIZE,
        "lr_model": args.lr_model, "lr_emb": args.lr_emb,
        "freeze_emb": args.freeze_emb,
        "n_hard_neg": N_HARD_NEG, "n_tracks": n_tracks,
        "als_factors": ALS_FACTORS, "als_alpha": ALS_ALPHA, "als_reg": ALS_REG,
        "train_examples": len(train_examples), "val_examples": len(val_examples),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save track vocabulary for eval
    with open(run_dir / "track_ids.json", "w") as f:
        json.dump(track_ids, f)

    best_val_hit = -1.0
    best_epoch = -1
    no_improve = 0

    max_epochs = args.epochs
    patience = args.patience
    print(f"\n{ts()} Starting training ({max_epochs} epochs max, patience {patience})")
    print(f"  Run dir: {run_dir}")

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_time = time.time() - ep_t0

        val_hit, val_ndcg, val_n = validate(model, val_loader, device, n_tracks)
        val_time = time.time() - ep_t0 - train_time

        tau_eff = 1.0 / model.logit_scale.exp().item()

        print(f"{ts()} Epoch {epoch:2d}  loss={train_loss:.4f}  "
              f"val_hit@20={val_hit:.4f}  val_nDCG={val_ndcg:.4f}  "
              f"τ={tau_eff:.4f}  train={train_time:.0f}s  val={val_time:.0f}s",
              flush=True)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "train_loss": train_loss,
            "val_hit20": val_hit,
            "val_ndcg": val_ndcg,
        }, run_dir / f"epoch_{epoch}.pt")

        if val_hit > best_val_hit:
            best_val_hit = val_hit
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "train_loss": train_loss,
                "val_hit20": val_hit,
                "val_ndcg": val_ndcg,
            }, run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"{ts()} Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch}, val_hit@20={best_val_hit:.4f})")
                break

    elapsed = time.time() - t0
    print(f"\n{ts()} Training complete. Best epoch {best_epoch}, "
          f"val_hit@20={best_val_hit:.4f}")
    print(f"  Run dir: {run_dir}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
