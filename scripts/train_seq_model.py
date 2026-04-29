#!/usr/bin/env python3
# ruff: noqa: T201
"""B5: Train the sequence-aware target-embedding model on MPS.

Uses pre-cached utterance embeddings and hard negatives.
Trains with InfoNCE + anti-collapse loss.
Validates on 5% held-out train sessions every epoch.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcrs.retrieval_modules.seq_model import (
    SequenceRecommender,
    anti_collapse_loss,
    info_nce_loss,
)
from datasets import DownloadConfig, load_dataset

# Paths
CACHE_DIR = REPO_ROOT / "cache" / "seq_model"
UTT_NPY = CACHE_DIR / "utt_embeddings.npy"
UTT_INDEX = CACHE_DIR / "utt_embedding_index.json"
HN_PATH = CACHE_DIR / "hard_negatives.pkl"
TRACK_EMB_DIR = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"

# Hyperparameters
SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
TAU = 0.05
ANTI_COLLAPSE_LAM = 0.05
N_HARD_NEG = 16
N_RANDOM_NEG = 16
VAL_FRACTION = 0.05
PATIENCE = 2


class SeqRecDataset(Dataset):
    """Dataset for sequence model training."""

    def __init__(
        self,
        examples: list[dict],
        track_embs: dict[str, np.ndarray],
        utt_embs: np.ndarray,
        utt_index: dict[str, int],
        hard_negatives: dict[tuple, list[str]],
        catalog_track_ids: list[str],
        catalog_emb_matrix: np.ndarray,
    ):
        self.examples = examples
        self.track_embs = track_embs
        self.utt_embs = utt_embs
        self.utt_index = utt_index
        self.hard_negatives = hard_negatives
        self.catalog_track_ids = catalog_track_ids
        self.catalog_emb_matrix = catalog_emb_matrix
        self.n_catalog = len(catalog_track_ids)

    def __len__(self):
        return len(self.examples)

    def _get_utt(self, session_id, turn):
        key = f"{session_id}:{turn}"
        idx = self.utt_index.get(key)
        if idx is not None and idx < len(self.utt_embs):
            return self.utt_embs[idx]
        return np.zeros(384, dtype=np.float32)

    def _get_track(self, track_id):
        emb = self.track_embs.get(track_id)
        if emb is not None:
            return emb
        return np.zeros(1024, dtype=np.float32)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        sid = ex["session_id"]
        history = ex["history"]  # [(turn_num, track_id), ...]
        current_turn = ex["current_turn"]
        gt_track_id = ex["gt_track_id"]

        T_hist = len(history)
        T = T_hist + 1

        # Track embeddings (history only)
        track_embs = np.zeros((T_hist, 1024), dtype=np.float32)
        for i, (turn, tid) in enumerate(history):
            track_embs[i] = self._get_track(tid)

        # Utterance embeddings (history + current)
        utt_embs = np.zeros((T, 384), dtype=np.float32)
        for i, (turn, tid) in enumerate(history):
            utt_embs[i] = self._get_utt(sid, turn)
        utt_embs[T_hist] = self._get_utt(sid, current_turn)

        # Accept labels (all 1 for V0)
        accept = np.ones(T_hist, dtype=np.int64)

        # Turn indices
        turns = np.zeros(T, dtype=np.int64)
        for i, (turn, _) in enumerate(history):
            turns[i] = min(turn - 1, 7)
        turns[T_hist] = min(current_turn - 1, 7)

        # Positive: GT track embedding
        positive = self._get_track(gt_track_id)

        # Last track embedding (for anti-collapse)
        last_track = track_embs[-1] if T_hist > 0 else np.zeros(1024, dtype=np.float32)

        # Hard negatives from BM25
        hn_key = (sid, current_turn)
        hn_ids = self.hard_negatives.get(hn_key, [])[:N_HARD_NEG]
        hard_neg_embs = np.zeros((N_HARD_NEG, 1024), dtype=np.float32)
        for i, tid in enumerate(hn_ids):
            hard_neg_embs[i] = self._get_track(tid)

        # Random catalog negatives
        rand_neg_embs = np.zeros((N_RANDOM_NEG, 1024), dtype=np.float32)
        played_set = {tid for _, tid in history} | {gt_track_id}
        rand_indices = []
        while len(rand_indices) < N_RANDOM_NEG:
            ri = random.randint(0, self.n_catalog - 1)
            if self.catalog_track_ids[ri] not in played_set:
                rand_indices.append(ri)
        for i, ri in enumerate(rand_indices):
            rand_neg_embs[i] = self.catalog_emb_matrix[ri]

        return {
            "track_embs": torch.from_numpy(track_embs),
            "utt_embs": torch.from_numpy(utt_embs),
            "accept": torch.from_numpy(accept),
            "turns": torch.from_numpy(turns),
            "seq_len": T,
            "positive": torch.from_numpy(positive),
            "last_track": torch.from_numpy(last_track),
            "hard_neg": torch.from_numpy(hard_neg_embs),
            "rand_neg": torch.from_numpy(rand_neg_embs),
        }


def collate_fn(batch):
    """Pad variable-length sequences to max length in batch."""
    max_T = max(b["seq_len"] for b in batch)
    max_T_hist = max_T - 1
    B = len(batch)

    track_embs = torch.zeros(B, max_T_hist, 1024)
    utt_embs = torch.zeros(B, max_T, 384)
    accept = torch.zeros(B, max_T_hist, dtype=torch.long)
    turns = torch.zeros(B, max_T, dtype=torch.long)
    seq_lens = torch.zeros(B, dtype=torch.long)
    positives = torch.zeros(B, 1024)
    last_tracks = torch.zeros(B, 1024)
    hard_negs = torch.zeros(B, N_HARD_NEG, 1024)
    rand_negs = torch.zeros(B, N_RANDOM_NEG, 1024)

    for i, b in enumerate(batch):
        T = b["seq_len"]
        T_h = T - 1
        track_embs[i, :T_h] = b["track_embs"]
        utt_embs[i, :T] = b["utt_embs"]
        accept[i, :T_h] = b["accept"]
        turns[i, :T] = b["turns"]
        seq_lens[i] = T
        positives[i] = b["positive"]
        last_tracks[i] = b["last_track"]
        hard_negs[i] = b["hard_neg"]
        rand_negs[i] = b["rand_neg"]

    return {
        "track_embs": track_embs,
        "utt_embs": utt_embs,
        "accept": accept,
        "turns": turns,
        "seq_lens": seq_lens,
        "positives": positives,
        "last_tracks": last_tracks,
        "hard_negs": hard_negs,
        "rand_negs": rand_negs,
    }


def build_examples(train_data):
    """Build training examples from train conversations."""
    examples = []
    for item in train_data:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        music = [(int(c["turn_number"]), str(c["content"]).strip())
                 for c in convs if c["role"] == "music"]
        user = [(int(c["turn_number"]), str(c["content"]))
                for c in convs if c["role"] == "user"]

        for u_turn, u_text in user:
            # Find GT: the music response at this turn
            gt = None
            for m_turn, m_tid in music:
                if m_turn == u_turn:
                    gt = m_tid
                    break
            if gt is None:
                continue
            # History: music played before this turn
            history = [(t, tid) for t, tid in music if t < u_turn]
            if not history:
                continue  # skip t=1
            examples.append({
                "session_id": sid,
                "history": history,
                "current_turn": u_turn,
                "gt_track_id": gt,
            })
    return examples


def compute_val_ndcg(model, val_examples, track_embs_dict, utt_embs, utt_index,
                     catalog_ids, catalog_matrix, device, k=20):
    """Compute nDCG@k on validation examples."""
    model.eval()
    ndcgs = []
    catalog_t = torch.from_numpy(catalog_matrix).float().to(device)

    with torch.no_grad():
        for ex in val_examples:
            sid = ex["session_id"]
            history = ex["history"]
            gt = ex["gt_track_id"]
            T_hist = len(history)
            T = T_hist + 1

            te = np.zeros((T_hist, 1024), dtype=np.float32)
            ue = np.zeros((T, 384), dtype=np.float32)
            for i, (turn, tid) in enumerate(history):
                emb = track_embs_dict.get(tid)
                if emb is not None:
                    te[i] = emb
                key = f"{sid}:{turn}"
                idx = utt_index.get(key)
                if idx is not None:
                    ue[i] = utt_embs[idx]
            key = f"{sid}:{ex['current_turn']}"
            idx = utt_index.get(key)
            if idx is not None:
                ue[T_hist] = utt_embs[idx]

            ac = np.ones(T_hist, dtype=np.int64)
            ti = np.zeros(T, dtype=np.int64)
            for i, (turn, _) in enumerate(history):
                ti[i] = min(turn - 1, 7)
            ti[T_hist] = min(ex["current_turn"] - 1, 7)

            target = model(
                torch.from_numpy(te).unsqueeze(0).to(device),
                torch.from_numpy(ue).unsqueeze(0).to(device),
                torch.from_numpy(ac).unsqueeze(0).to(device),
                torch.from_numpy(ti).unsqueeze(0).to(device),
                torch.tensor([T], device=device),
            )  # (1, 1024)

            scores = (target @ catalog_t.T).squeeze(0).cpu().numpy()

            # Exclude already-played tracks (mirrors inference behavior)
            played_set = {tid for _, tid in ex["history"]}
            for j, tid in enumerate(catalog_ids):
                if tid in played_set:
                    scores[j] = -np.inf

            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
            predicted = [catalog_ids[i] for i in top_k_idx]

            ndcg = 0.0
            for rank, tid in enumerate(predicted):
                if tid == gt:
                    ndcg = 1.0 / math.log2(rank + 2)
                    break
            ndcgs.append(ndcg)

    model.train()
    return float(np.mean(ndcgs))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load caches
    print("Loading caches...", flush=True)
    utt_embs = np.load(UTT_NPY)
    utt_index = json.load(open(UTT_INDEX))
    with open(HN_PATH, "rb") as f:
        hard_negatives = pickle.load(f)
    print(f"  Utterances: {utt_embs.shape}")
    print(f"  Hard negatives: {len(hard_negatives)} entries")

    # Load catalog embeddings
    catalog_ids = json.load(open(TRACK_EMB_DIR / "track_ids.json"))
    catalog_matrix = np.load(TRACK_EMB_DIR / "vectors.npy")
    print(f"  Catalog: {len(catalog_ids)} tracks, dim={catalog_matrix.shape[1]}")

    # Build track embedding lookup
    track_embs_dict = {tid: catalog_matrix[i] for i, tid in enumerate(catalog_ids)}

    # Load train data and build examples
    print("Building training examples...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    all_examples = build_examples(ds["train"])
    print(f"  Total examples: {len(all_examples)}")

    # Split train/val by session
    all_sessions = list(set(ex["session_id"] for ex in all_examples))
    random.shuffle(all_sessions)
    n_val = max(1, int(len(all_sessions) * VAL_FRACTION))
    val_sessions = set(all_sessions[:n_val])
    train_examples = [ex for ex in all_examples if ex["session_id"] not in val_sessions]
    val_examples = [ex for ex in all_examples if ex["session_id"] in val_sessions]
    print(f"  Train: {len(train_examples)} examples ({len(all_sessions) - n_val} sessions)")
    print(f"  Val: {len(val_examples)} examples ({n_val} sessions)")

    # Dataset and loader
    train_dataset = SeqRecDataset(
        train_examples, track_embs_dict, utt_embs, utt_index,
        hard_negatives, catalog_ids, catalog_matrix,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    # Model
    model = SequenceRecommender().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * MAX_EPOCHS

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Run directory
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = CACHE_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "seed": SEED, "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "warmup_steps": WARMUP_STEPS,
        "tau": TAU, "anti_collapse_lam": ANTI_COLLAPSE_LAM,
        "n_hard_neg": N_HARD_NEG, "n_random_neg": N_RANDOM_NEG,
        "n_train": len(train_examples), "n_val": len(val_examples),
        "n_params": n_params, "device": str(device),
        "track_emb_dim": 1024, "utt_emb_dim": 384,
        "d_model": 256, "nhead": 4, "num_layers": 4, "output_dim": 1024,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nTraining — run_id: {run_id}")
    print(f"  Run dir: {run_dir}")
    best_val_ndcg = -1
    best_epoch = -1
    val_metrics = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch = time.time()
        model.train()
        total_loss = 0
        total_nce = 0
        total_ac = 0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            # Move to device
            track_e = batch["track_embs"].to(device)
            utt_e = batch["utt_embs"].to(device)
            accept_l = batch["accept"].to(device)
            turn_i = batch["turns"].to(device)
            seq_l = batch["seq_lens"].to(device)
            pos_e = batch["positives"].to(device)
            last_t = batch["last_tracks"].to(device)
            hard_n = batch["hard_negs"].to(device)
            rand_n = batch["rand_negs"].to(device)

            # Forward
            target = model(track_e, utt_e, accept_l, turn_i, seq_l)

            # Normalize embeddings
            pos_norm = F.normalize(pos_e, dim=-1)
            hard_norm = F.normalize(hard_n, dim=-1, p=2)
            rand_norm = F.normalize(rand_n, dim=-1, p=2)

            # In-batch negatives: use other examples' positives, excluding self
            B_cur = target.size(0)
            in_batch_parts = []
            for b_idx in range(B_cur):
                mask = torch.arange(B_cur, device=device) != b_idx
                in_batch_parts.append(pos_norm[mask])  # (B-1, D)
            in_batch_neg = torch.stack(in_batch_parts)  # (B, B-1, D)

            # Combine negatives
            all_neg = torch.cat([in_batch_neg, hard_norm, rand_norm], dim=1)

            # Losses
            nce = info_nce_loss(target, pos_norm, all_neg, tau=TAU)
            ac = anti_collapse_loss(target, F.normalize(last_t, dim=-1), lam=ANTI_COLLAPSE_LAM)
            loss = nce + ac

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_nce += nce.item()
            total_ac += ac.item()
            n_batches += 1

            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={total_loss/n_batches:.4f} nce={total_nce/n_batches:.4f} "
                      f"ac={total_ac/n_batches:.4f} lr={scheduler.get_last_lr()[0]:.6f}",
                      flush=True)

        avg_loss = total_loss / max(n_batches, 1)
        avg_nce = total_nce / max(n_batches, 1)
        avg_ac = total_ac / max(n_batches, 1)

        # Validation
        val_ndcg = compute_val_ndcg(
            model, val_examples, track_embs_dict, utt_embs, utt_index,
            catalog_ids, catalog_matrix, device,
        )

        # Check collapse: what fraction of top-1 predictions == played[-1]?
        model.eval()
        n_collapse = 0
        n_checked = min(500, len(val_examples))
        catalog_t = torch.from_numpy(catalog_matrix).float().to(device)
        with torch.no_grad():
            for ex in val_examples[:n_checked]:
                sid = ex["session_id"]
                history = ex["history"]
                T_hist = len(history)
                T = T_hist + 1
                te = np.zeros((T_hist, 1024), dtype=np.float32)
                ue = np.zeros((T, 384), dtype=np.float32)
                for i, (turn, tid) in enumerate(history):
                    emb = track_embs_dict.get(tid)
                    if emb is not None: te[i] = emb
                    key = f"{sid}:{turn}"
                    idx = utt_index.get(key)
                    if idx is not None: ue[i] = utt_embs[idx]
                key = f"{sid}:{ex['current_turn']}"
                idx = utt_index.get(key)
                if idx is not None: ue[T_hist] = utt_embs[idx]
                ac = np.ones(T_hist, dtype=np.int64)
                ti = np.zeros(T, dtype=np.int64)
                for i, (turn, _) in enumerate(history): ti[i] = min(turn - 1, 7)
                ti[T_hist] = min(ex["current_turn"] - 1, 7)
                target = model(
                    torch.from_numpy(te).unsqueeze(0).to(device),
                    torch.from_numpy(ue).unsqueeze(0).to(device),
                    torch.from_numpy(ac).unsqueeze(0).to(device),
                    torch.from_numpy(ti).unsqueeze(0).to(device),
                    torch.tensor([T], device=device),
                )
                scores = (target @ catalog_t.T).squeeze(0)
                top1_idx = scores.argmax().item()
                if catalog_ids[top1_idx] == history[-1][1]:
                    n_collapse += 1
        collapse_rate = n_collapse / n_checked
        model.train()

        epoch_time = time.time() - t_epoch
        entry = {
            "epoch": epoch, "loss": avg_loss, "nce": avg_nce, "ac": avg_ac,
            "val_ndcg20": val_ndcg, "collapse_rate": collapse_rate,
            "lr": scheduler.get_last_lr()[0], "epoch_sec": epoch_time,
        }
        val_metrics.append(entry)
        print(f"  EPOCH {epoch}: loss={avg_loss:.4f} val_nDCG@20={val_ndcg:.4f} "
              f"collapse={collapse_rate:.1%} time={epoch_time:.0f}s", flush=True)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_ndcg20": val_ndcg,
            "collapse_rate": collapse_rate,
        }, run_dir / f"epoch_{epoch}.pt")

        # Best model tracking
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_ndcg20": val_ndcg,
                "collapse_rate": collapse_rate,
            }, run_dir / "final.pt")
            print(f"  → New best: epoch {epoch}, val_nDCG@20={val_ndcg:.4f}")
        elif epoch - best_epoch >= PATIENCE:
            print(f"  Early stopping after {PATIENCE} epochs without improvement")
            break

    # Save metrics
    with open(run_dir / "val_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    print(f"\nTraining complete.")
    print(f"  Best epoch: {best_epoch}, val_nDCG@20: {best_val_ndcg:.4f}")
    print(f"  Run dir: {run_dir}")
    print(f"  Best checkpoint: {run_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
