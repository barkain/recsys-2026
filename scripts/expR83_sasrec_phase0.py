#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R83 Phase 0 — Behavior-native sequence model (SASRec) on A100.

The unexplored signal class: user/session BEHAVIOR. Earlier paths exhausted
query+catalog intent (R82) and feature-stack neural rerankers (R71/R76/R80/R81).
R83 attacks the question: given several tracks that match the query, which
one would THIS user/session actually pick next?

Architecture:
- Item embeddings: 47K tracks × 128 dim, trainable, init projection of
  BGE-large catalog (1024 → 128) — gives semantic prior
- Sequence encoder: 2-layer Transformer (d=128, heads=4, ff=256), causal
  masked self-attention over played track embeddings
- Utterance embedding: BGE query (1024) → 128 projection, prepended as
  context token at position 0
- Output: dot product with all 47K item embeddings → softmax

Training (Colab A100):
- Per case: history = played track IDs (variable length, max 10)
- Loss: sampled softmax with mixed negatives
  - K_in_batch random tracks from in-batch positives
  - K_random random catalog tracks
  - K_r54_fp R54c top-20 false positives from R79 training_pairs
- bf16 autocast, AdamW

Eval (fold-0 standalone):
- For each fold-0 case, score all 47K catalog tracks
- Take R83 top-300, then top-20
- Compare to OOF R54c top-20 baseline

Gates (predeclared, less conservative per user direction):
- h7 recovered > h7 lost, OR h7 Δ ≥ 0
- same-artist Δ ≥ -0.005
- top-20 overlap ≥ 10/20 (sanity bound)

If pass → 5-fold OOF. If pool_hit shows new candidates beyond top-30,
consider sharpness training before archive.

Reuses:
- cache/r79/training_pairs.pkl (per-case played_tracks + GT + hard_negs + fold)
- cache/r68/phase0_fold0/track_embeddings.npy (BGE-large catalog init)
- cache/r68/phase0_fold0/track_ids.json
- cache/r68/phase0_fold0/query_embeddings_dev.npy (fold-0 only)
- For train cases (folds 1-4), we need R68 query embeddings — but R68 has them
  only for fold-0 dev. We'll instead compute utterance-as-text → BGE on-the-fly
  for train cases. Actually simpler: use a pre-computed catalog of query
  texts and encode all of them once in Phase 0A.

Phase 0A (Mac): build cache/r83/training_data.pkl with:
- per case: played_tracks (list), user_query text, gt, fold, is_h7,
  hard_negs (R54c top-20), is_same_artist
- Also: pre-compute BGE-large for all 8000 dev query texts? No — Colab does that
  on the fly using the catalog text+BGE setup.

Actually, simpler: use the existing R79 training_pairs.pkl directly as input.
Encode utterances on Colab using the same BGE-large model used for the catalog
init (no fine-tune at first).
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

TRAINING_PAIRS = REPO / "cache" / "r79" / "training_pairs.pkl"
# Use R80's fp16 catalog (committed to git, 96 MB)
CATALOG_EMBS_R80_FP16 = REPO / "cache" / "r80" / "catalog_track_embs_fp16.npy"
CATALOG_IDS_R80 = REPO / "cache" / "r80" / "catalog_track_ids.json"
# Fallback (R68 fp32 catalog, not in git)
CATALOG_EMBS_R68 = REPO / "cache" / "r68" / "phase0_fold0" / "track_embeddings.npy"
CATALOG_IDS_R68 = REPO / "cache" / "r68" / "phase0_fold0" / "track_ids.json"

OUT_DIR = REPO / "cache" / "r83"
OUT_RESULT = REPO / "exp" / "eval" / "expR83_phase0.json"
OUT_DOC = REPO / "docs" / "r83_phase0_result.md"

# Model
ITEM_DIM = 128
SEQ_LEN = 10
N_LAYERS = 2
N_HEADS = 4
FF_DIM = 256
DROPOUT = 0.2

# Training
N_EPOCHS = 20
BATCH_CASES = 64
LR = 1e-3
WEIGHT_DECAY = 0.01
K_IN_BATCH = 64
K_RANDOM = 64
K_HARD = 16
USE_BF16 = True

# Eval
TOP_K = 20
TOP_30 = 30
TOP_300 = 300

# Gates
GATE_OVERLAP_FLOOR = 10  # sanity bound (looser than prior sprints)
GATE_SAME_DELTA = -0.005  # looser per user direction


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


class SASRec(nn.Module):
    """Causal SASRec-style sequence model.

    Input: list of item indices (history) + utterance embedding
    Output: per-position hidden states for next-item prediction
    """
    def __init__(self, n_items, dim=ITEM_DIM, seq_len=SEQ_LEN, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT,
                 utt_dim=1024):
        super().__init__()
        self.n_items = n_items
        self.dim = dim
        # Item embedding table (47K + 1 padding)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        # Positional embeddings (sequence length + 1 for utterance prefix)
        self.pos_emb = nn.Embedding(seq_len + 1, dim)
        # Utterance projection
        self.utt_proj = nn.Sequential(
            nn.Linear(utt_dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )
        # Encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def init_item_embeddings(self, init_tensor):
        """Initialize item_emb[1:] from projection of BGE embeddings.

        init_tensor: (n_items, init_dim) — will be projected to (n_items, dim).
        """
        assert init_tensor.shape[0] == self.n_items
        # Project init_dim -> dim via a learned linear (one-shot init)
        init_dim = init_tensor.shape[1]
        if init_dim != self.dim:
            # Linear projection (random init), one-shot apply
            with torch.no_grad():
                # Use a fixed seed projection to keep determinism
                g = torch.Generator(device='cpu').manual_seed(SEED)
                proj = torch.randn(init_dim, self.dim, generator=g) / math.sqrt(init_dim)
                projected = init_tensor @ proj.to(init_tensor.device)
                projected = F.normalize(projected, p=2, dim=-1) * math.sqrt(self.dim)
                # Skip padding row (index 0)
                self.item_emb.weight.data[1:] = projected
        else:
            with torch.no_grad():
                self.item_emb.weight.data[1:] = init_tensor

    def forward(self, history_ids, utt_emb):
        """
        history_ids: (B, seq_len) with 0 = padding, 1+i = item i
        utt_emb: (B, utt_dim) — raw BGE embedding
        Returns: (B, dim) — final hidden state for next-item prediction
        """
        B = history_ids.shape[0]
        # Embed history
        h_emb = self.item_emb(history_ids)  # (B, seq_len, dim)
        # Embed utterance
        u_emb = self.utt_proj(utt_emb).unsqueeze(1)  # (B, 1, dim)
        # Concat utterance at position 0
        x = torch.cat([u_emb, h_emb], dim=1)  # (B, seq_len+1, dim)
        # Add positional embeddings
        pos_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        x = self.dropout(x)
        # Causal mask
        seq_total = x.shape[1]
        causal = torch.triu(torch.ones(seq_total, seq_total, device=x.device),
                            diagonal=1).bool()
        # Padding mask (utterance never padded, history padding = id==0)
        pad_mask = torch.cat([
            torch.zeros(B, 1, device=x.device, dtype=torch.bool),
            (history_ids == 0),
        ], dim=1)
        # Encode
        out = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        # Take final non-padding position. We use the LAST history position
        # (or last non-pad) as the "next-item prediction" query
        # Find last non-pad position per case
        non_pad = (~pad_mask).float()  # (B, seq_total)
        last_idx = (non_pad.sum(dim=1) - 1).long().clamp(min=0)  # (B,)
        h_last = out[torch.arange(B, device=x.device), last_idx]  # (B, dim)
        return h_last

    def score_items(self, h, item_indices):
        """Score items against a query hidden state h.

        h: (B, dim)
        item_indices: (B, K) — items to score per case
        Returns: (B, K) scores (dot product)
        """
        item_embs = self.item_emb(item_indices)  # (B, K, dim)
        scores = (item_embs * h.unsqueeze(1)).sum(dim=-1)  # (B, K)
        return scores

    def score_all_items(self, h):
        """Score against all items (skip padding row).

        h: (B, dim)
        Returns: (B, n_items) scores
        """
        all_embs = self.item_emb.weight[1:]  # (n_items, dim) — skip pad
        scores = h @ all_embs.t()  # (B, n_items)
        return scores


def build_history_tensor(played_tracks, track_to_idx, seq_len):
    """Convert list of track_ids to seq_len-padded int tensor (1-indexed)."""
    out = np.zeros(seq_len, dtype=np.int64)
    truncated = played_tracks[-seq_len:]  # take most recent
    for i, t in enumerate(truncated):
        if t in track_to_idx:
            out[i] = track_to_idx[t] + 1  # 0 reserved for pad
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-cases", type=int, default=BATCH_CASES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--item-dim", type=int, default=ITEM_DIM)
    parser.add_argument("--k-random", type=int, default=K_RANDOM)
    parser.add_argument("--k-hard", type=int, default=K_HARD)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R83 Phase 0 — SASRec sequence model")
    print(f"  item_dim={args.item_dim} batch={args.batch_cases} epochs={args.epochs} lr={args.lr}")
    print(f"  negatives: in_batch={K_IN_BATCH} random={args.k_random} hard={args.k_hard}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n{ts()} Loading R79 training_pairs (re-using) ...", flush=True)
    with open(TRAINING_PAIRS, "rb") as f:
        data = pickle.load(f)
    pairs = data["training_pairs"]
    print(f"  {len(pairs)} cases total")

    print(f"{ts()} Loading catalog BGE-large embeddings ...", flush=True)
    if CATALOG_EMBS_R80_FP16.exists():
        catalog_embs = np.load(CATALOG_EMBS_R80_FP16).astype(np.float32)  # fp16→fp32
        catalog_ids = json.load(open(CATALOG_IDS_R80))
        print(f"  using R80 fp16 catalog (re-cast to fp32 for training)")
    elif CATALOG_EMBS_R68.exists():
        catalog_embs = np.load(CATALOG_EMBS_R68).astype(np.float32)
        catalog_ids = json.load(open(CATALOG_IDS_R68))
        print(f"  using R68 fp32 catalog")
    else:
        raise FileNotFoundError("Neither R80 fp16 nor R68 fp32 catalog found")
    track_to_idx = {tid: i for i, tid in enumerate(catalog_ids)}
    n_items = catalog_embs.shape[0]
    print(f"  catalog: {n_items} tracks × {catalog_embs.shape[1]} dim")

    # Encode utterances for all 8000 cases
    print(f"{ts()} Encoding utterances with BGE-large ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    bge_name = "BAAI/bge-large-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(bge_name)
    bge_model = AutoModel.from_pretrained(bge_name).to(device).eval()
    if device.type == "cuda":
        bge_model = bge_model.to(dtype=torch.bfloat16)
    utt_texts = []
    for p in pairs:
        utt = p.get("history_text") or p.get("user_query", "")
        utt_texts.append(str(utt)[:1024])

    utt_embs = np.zeros((len(pairs), 1024), dtype=np.float32)
    batch = 64
    t_enc = time.time()
    with torch.no_grad():
        for i in range(0, len(utt_texts), batch):
            chunk = utt_texts[i:i+batch]
            inp = tokenizer(chunk, padding=True, truncation=True,
                            max_length=192, return_tensors="pt").to(device)
            out = bge_model(**inp)
            emb = out.last_hidden_state[:, 0, :]  # CLS
            emb = F.normalize(emb.float(), p=2, dim=-1)
            utt_embs[i:i+batch] = emb.cpu().numpy()
            if (i // batch) % 10 == 0:
                print(f"    encoded {i+len(chunk)}/{len(utt_texts)} "
                      f"({time.time() - t_enc:.0f}s)", flush=True)
    del bge_model
    torch.cuda.empty_cache()
    print(f"  utt encoding done in {time.time() - t_enc:.0f}s")

    # Build per-case tensors
    print(f"{ts()} Building per-case history tensors ...", flush=True)
    case_histories = []
    case_gts = []
    case_hards = []
    case_folds = []
    case_h7 = []
    case_same = []
    case_played_artists = []
    track_artist_map = None
    # Try R12 payload first; if not available, build from HF catalog
    R12_PAYLOAD = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
    if R12_PAYLOAD.exists():
        with open(R12_PAYLOAD, "rb") as f:
            _p = pickle.load(f)
        track_artist_map = _p.get("track_artist", {})
        print(f"  track_artist from R12 payload: {len(track_artist_map)}")
    else:
        print(f"  R12 payload not found; building track_artist from HF catalog")
        from datasets import DownloadConfig, load_dataset
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                          download_config=DownloadConfig())["all_tracks"]
        track_artist_map = {}
        for item in ds:
            tid = str(item["track_id"])
            a = item.get("artist_name")
            if isinstance(a, list):
                a = a[0] if a else ""
            track_artist_map[tid] = a if a else ""
        print(f"  track_artist from HF: {len(track_artist_map)}")

    for p in pairs:
        played = p.get("played_tracks", [])
        gt = p["gt"]
        hist = build_history_tensor(played, track_to_idx, SEQ_LEN)
        case_histories.append(hist)
        # GT index (-1 if not in catalog)
        case_gts.append(track_to_idx.get(gt, -1))
        # Hard negs (R54c top-20 minus GT, in catalog)
        hards = [track_to_idx[h] for h in p.get("hard_negs", []) if h in track_to_idx]
        case_hards.append(hards[:args.k_hard])
        case_folds.append(p["fold"])
        case_h7.append(p.get("is_h7", False))
        case_same.append(p.get("is_same_artist", False))
        # Played artists for same-artist decoys
        artists = []
        for pt in played:
            a = track_artist_map.get(pt, "")
            if isinstance(a, list) and a:
                a = a[0]
            if isinstance(a, str) and a:
                artists.append(a)
        case_played_artists.append(artists[-3:])

    n_cases = len(pairs)
    fold0_indices = [i for i, f in enumerate(case_folds) if f == 0]
    train_indices = [i for i, f in enumerate(case_folds) if f != 0]
    h7_fold0_indices = [i for i in fold0_indices if case_h7[i]]
    print(f"  fold-0 cases: {len(fold0_indices)}  (h7: {len(h7_fold0_indices)})")
    print(f"  train cases (folds 1-4): {len(train_indices)}")

    # Pre-stack to numpy arrays
    history_arr = np.stack(case_histories)  # (N, seq_len)
    utt_arr = utt_embs  # (N, 1024)
    gt_arr = np.array(case_gts, dtype=np.int64)  # (N,)
    h7_arr = np.array(case_h7, dtype=np.bool_)
    same_arr = np.array(case_same, dtype=np.bool_)
    # Filter train cases to those with GT in catalog
    valid_train = [i for i in train_indices if gt_arr[i] >= 0]
    print(f"  train cases with GT in catalog: {len(valid_train)}/{len(train_indices)}")

    # Build artist -> list of track_idx mapping for same-artist decoys
    print(f"{ts()} Building artist -> track index ...", flush=True)
    artist_to_tracks = {}
    for tid, a in track_artist_map.items():
        if isinstance(a, list):
            a = a[0] if a else ""
        if not isinstance(a, str) or not a:
            continue
        if tid in track_to_idx:
            artist_to_tracks.setdefault(a, []).append(track_to_idx[tid])
    print(f"  artists indexed: {len(artist_to_tracks)}")

    # Build model
    print(f"\n{ts()} Building SASRec model ...", flush=True)
    model = SASRec(n_items=n_items, dim=args.item_dim, seq_len=SEQ_LEN,
                   n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=FF_DIM,
                   dropout=DROPOUT).to(device)
    # Init item embeddings from BGE-large catalog projection
    print(f"  initializing item embeddings from BGE-large projection")
    model.init_item_embeddings(torch.from_numpy(catalog_embs).to(device))
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    n_batches = (len(valid_train) + args.batch_cases - 1) // args.batch_cases
    total_steps = args.epochs * n_batches
    warmup = int(total_steps * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda step: min(1.0, step / max(warmup, 1)) if step < warmup
        else max(0.0, 1.0 - (step - warmup) / max(total_steps - warmup, 1)),
    )

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if USE_BF16 and device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))

    # ---- Training ----
    print(f"\n{ts()} === Training SASRec ({args.epochs} epochs, {len(valid_train)} cases) ===")
    rng = np.random.default_rng(SEED)
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        rng.shuffle(valid_train)
        tot_loss = 0.0
        n_bd = 0
        for bs in range(0, len(valid_train), args.batch_cases):
            batch_ids = valid_train[bs:bs + args.batch_cases]
            B = len(batch_ids)

            history_b = torch.from_numpy(history_arr[batch_ids]).to(device)
            utt_b = torch.from_numpy(utt_arr[batch_ids]).to(device)
            gts = torch.from_numpy(gt_arr[batch_ids]).to(device)  # (B,)

            # Build negative candidate sets per case
            # Positives: gts (B,)
            # In-batch negs: other cases' GTs (B-1 each but we sample K_IN_BATCH)
            # Random negs: K_RANDOM per case
            # Hard negs: case-specific R54c top-20 FPs (K_HARD per case)
            # Same-artist decoys: K_SAME per case (4)
            with autocast_ctx:
                h = model(history_b, utt_b)  # (B, dim)

                # Pos scores
                pos_emb = model.item_emb(gts + 1)  # 1-indexed (+1 for pad)
                pos_score = (h * pos_emb).sum(dim=-1, keepdim=True)  # (B, 1)

                # In-batch negs (from current batch's other GTs)
                # Sample K_IN_BATCH random other-case GTs
                in_batch_gts = gts.clone()
                k_ib = min(K_IN_BATCH, B - 1)
                in_batch_neg = in_batch_gts[torch.randperm(B, device=device)[:k_ib]].unsqueeze(0).expand(B, k_ib)
                # Random negs
                random_ids = torch.randint(0, n_items, (B, args.k_random), device=device)
                # Hard negs (from R54c top-20 FPs)
                hard_padded = np.zeros((B, args.k_hard), dtype=np.int64)
                for i, ci in enumerate(batch_ids):
                    hards = case_hards[ci]
                    for j, h_idx in enumerate(hards[:args.k_hard]):
                        hard_padded[i, j] = h_idx
                hard_t = torch.from_numpy(hard_padded).to(device)
                # Same-artist decoys (4 per case)
                K_SAME = 4
                same_padded = np.zeros((B, K_SAME), dtype=np.int64)
                for i, ci in enumerate(batch_ids):
                    artists = case_played_artists[ci]
                    pool = []
                    for a in artists:
                        pool.extend(artist_to_tracks.get(a, [])[:8])
                    pool = list(set(pool) - {gt_arr[ci]})
                    if pool:
                        chosen = rng.choice(pool, min(K_SAME, len(pool)), replace=False)
                        for j, c in enumerate(chosen):
                            same_padded[i, j] = c
                same_t = torch.from_numpy(same_padded).to(device)

                # Concat all neg sets
                neg_ids = torch.cat([in_batch_neg, random_ids, hard_t, same_t], dim=1)  # (B, K_total)
                # Score negs: 1-index for item_emb
                neg_emb = model.item_emb(neg_ids + 1)  # (B, K_total, dim)
                neg_scores = (h.unsqueeze(1) * neg_emb).sum(dim=-1)  # (B, K_total)

                # Listwise CE: positive index 0
                all_scores = torch.cat([pos_score, neg_scores], dim=1)  # (B, 1+K_total)
                temperature = 0.05
                logits = all_scores / temperature
                target = torch.zeros(B, dtype=torch.long, device=device)
                loss = F.cross_entropy(logits, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            tot_loss += float(loss.item())
            n_bd += 1
            global_step += 1

        if (epoch + 1) % 2 == 0:
            print(f"    epoch {epoch+1}/{args.epochs}  avg_loss={tot_loss/n_bd:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

    # ---- Eval on fold-0 ----
    print(f"\n{ts()} === Eval R83 standalone on fold-0 ===")
    model.eval()
    with torch.no_grad(), autocast_ctx:
        r83_top20_per_case = {}
        r83_top30_per_case = {}
        eval_t = time.time()
        for bi, bs in enumerate(range(0, len(fold0_indices), args.batch_cases)):
            batch_ids = fold0_indices[bs:bs + args.batch_cases]
            history_b = torch.from_numpy(history_arr[batch_ids]).to(device)
            utt_b = torch.from_numpy(utt_arr[batch_ids]).to(device)
            h = model(history_b, utt_b)
            scores = model.score_all_items(h)  # (B, n_items)
            # Mask played tracks (so we don't recommend already-played)
            for i, ci in enumerate(batch_ids):
                played = case_histories[ci]
                for pid in played:
                    if pid > 0:
                        scores[i, pid - 1] = -1e9
            # top-300
            top_300 = scores.topk(TOP_300, dim=-1).indices.cpu().numpy()
            for i, ci in enumerate(batch_ids):
                r83_top30_per_case[ci] = top_300[i, :TOP_30].tolist()
                r83_top20_per_case[ci] = top_300[i, :TOP_20].tolist()
            if (bs // args.batch_cases) % 5 == 0:
                print(f"    eval {bs+len(batch_ids)}/{len(fold0_indices)} "
                      f"({time.time() - eval_t:.0f}s)", flush=True)

    # Build R54c baseline top-20 from training pairs (oof_top20 already sorted)
    print(f"\n{ts()} Computing metrics ...")
    results = []
    for ci in fold0_indices:
        pair = pairs[ci]
        gt = pair["gt"]
        gt_idx = track_to_idx.get(gt, -1)
        r83_top20 = r83_top20_per_case[ci]
        r83_top30 = r83_top30_per_case[ci]
        r_in_top20 = gt_idx in r83_top20
        r_gt_rank = -1
        if r_in_top20:
            r_gt_rank = r83_top20.index(gt_idx) + 1
        # Baseline
        b_top20 = [track_to_idx.get(t, -1) for t in pair["oof_top20"][:TOP_20]]
        b_in_top20 = gt_idx in b_top20
        b_gt_rank = b_top20.index(gt_idx) + 1 if b_in_top20 else -1
        results.append({
            "case_idx": pair["case_idx"],
            "is_h7": pair["is_h7"],
            "is_same_artist": pair["is_same_artist"],
            "b_in_top20": b_in_top20,
            "r_in_top20": r_in_top20,
            "b_ndcg": ndcg_at_k(b_gt_rank, TOP_K),
            "r_ndcg": ndcg_at_k(r_gt_rank, TOP_K),
            "top20_overlap": len(set(r83_top20) & set(b_top20)),
            "top1_changed": 1 if r83_top20[0] != b_top20[0] else 0,
            "r_top30_has_gt": gt_idx in r83_top30,
        })

    def avg(rs, key, where=None):
        if where: rs = [r for r in rs if where(r)]
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    h7_rows = [r for r in results if r["is_h7"]]
    same_rows = [r for r in results if r["is_same_artist"]]
    diff_rows = [r for r in results if not r["is_same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", results), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows)]:
        b = avg(rows, "b_ndcg")
        r = avg(rows, "r_ndcg")
        metrics[name] = {"n": len(rows), "baseline": b, "r83": r, "delta": r - b}

    h7_rec = sum(1 for r in h7_rows if r["r_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["r_in_top20"])
    h7_net = h7_rec - h7_lost
    h7_top30_hits = sum(1 for r in h7_rows if r["r_top30_has_gt"])
    h7_top30_unique = sum(1 for r in h7_rows if r["r_top30_has_gt"] and not r["b_in_top20"])
    top1_churn = sum(r["top1_changed"] for r in results)
    churn_per_80 = top1_churn / len(results) * 80
    overlap_mean = avg(results, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]

    # Gates (looser per user direction)
    gate_recov_or_h7 = (h7_rec > h7_lost) or (h7_d >= 0)
    gate_same = sa_d >= GATE_SAME_DELTA
    gate_overlap = overlap_mean >= GATE_OVERLAP_FLOOR
    all_pass = gate_recov_or_h7 and gate_same and gate_overlap

    if all_pass:
        verdict = "PROCEED_5FOLD"
    elif h7_top30_unique >= 10:
        verdict = "SHARPNESS_NEEDED"  # has top-30 signal but not top-20
    else:
        verdict = "ARCHIVE"

    print(f"\n{ts()} === Results ===")
    for name, m in metrics.items():
        print(f"  {name:14}  n={m['n']:5d}  baseline={m['baseline']:.4f}  "
              f"r83={m['r83']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_rec}  lost={h7_lost}  net={h7_net:+d}")
    print(f"  h7 top-30 GT hits: {h7_top30_hits}/{len(h7_rows)}  "
          f"(unique vs R54c top-20: {h7_top30_unique})")
    print(f"  top1_churn={top1_churn}  per_80={churn_per_80:.2f}  overlap_mean={overlap_mean:.2f}/20")
    print(f"\n  Gates:")
    print(f"    h7 rec>lost OR h7 Δ≥0:   {gate_recov_or_h7}  ({h7_rec}>{h7_lost}, Δ={h7_d:+.4f})")
    print(f"    same-art Δ ≥ -0.005:     {gate_same}  ({sa_d:+.4f})")
    print(f"    overlap ≥ 10:            {gate_overlap}  ({overlap_mean:.2f})")
    print(f"  VERDICT: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "R83 Phase 0 — SASRec",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "hyperparams": vars(args),
        "metrics": metrics,
        "h7_recovery": {"recovered": h7_rec, "lost": h7_lost, "net": h7_net,
                        "top30_hits": h7_top30_hits, "top30_unique": h7_top30_unique},
        "churn": {"top1_changed": top1_churn, "churn_per_80": churn_per_80,
                  "top20_overlap_mean": overlap_mean},
        "gates": {
            "h7_recov_gt_lost_or_h7_delta_geq_0": {"value": [h7_rec, h7_lost, h7_d],
                                                    "pass": gate_recov_or_h7},
            "same_artist_delta_geq_neg0.005": {"value": sa_d, "pass": gate_same},
            "overlap_geq_10": {"value": overlap_mean, "pass": gate_overlap},
        },
    }
    OUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULT.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved → {OUT_RESULT}")

    md = [
        "# R83 Phase 0 — SASRec sequence model",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        f"## Verdict: **{verdict}**",
        "",
        "## Metrics",
        "",
        "| Subset | n | OOF R54c | R83 standalone | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r83']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_rec}, lost={h7_lost}, net={h7_net:+d}",
        f"- h7 top-30 GT hits: {h7_top30_hits}/{len(h7_rows)} (unique vs R54c top-20: {h7_top30_unique})",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap = {overlap_mean:.2f}/20",
        "",
        "## Gates (looser per user direction)",
        f"- h7 rec > lost OR h7 Δ ≥ 0: **{gate_recov_or_h7}** ({h7_rec}>{h7_lost}, Δ={h7_d:+.4f})",
        f"- same-artist Δ ≥ -0.005: **{gate_same}** ({sa_d:+.4f})",
        f"- top-20 overlap ≥ 10: **{gate_overlap}** ({overlap_mean:.2f})",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved → {OUT_DOC}")


if __name__ == "__main__":
    main()
