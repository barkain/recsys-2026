#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R31: SASRec sequential retriever — Stage 0 + Stage 1.

Phases (run with --phase):
  build_data      Build vocab, training pairs, fold-0 val (exec restarts)
  overfit_smoke   Overfit 256 examples to verify model works
  train_fold0     Train on all 15,199 train sessions
  eval_fold0      Evaluate fold-0 standalone vs R21 OOF
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import sys
import time
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
R31_DIR = REPO / "cache" / "r31"

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2
MAX_LEN = 8
TAU = 0.07


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SASRecV0(nn.Module):
    def __init__(self, n_items, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, seq_len):
        """
        seq: (B, L) int64 — track indices, 0=padding
        seq_len: (B,) int64 — actual sequence lengths
        Returns: (B, d_model) session vectors
        """
        bsz, slen = seq.shape
        positions = torch.arange(slen, device=seq.device).unsqueeze(0).expand(bsz, -1)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.drop(self.norm(x))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(slen, device=seq.device)
        pad_mask = seq == 0

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        idx = (seq_len - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
        out = x.gather(1, idx).squeeze(1)
        return F.normalize(out, dim=-1)

    def score_all(self, session_vec):
        """Score all items with L2-normalized embeddings. Returns (B, n_items+1)."""
        normed_emb = F.normalize(self.item_emb.weight, dim=-1)
        return session_vec @ normed_emb.T


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase: build_data
# ---------------------------------------------------------------------------

def phase_build_data():
    print(f"{ts()} Phase: build_data")
    R31_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = R31_DIR / "vocab.json"
    pairs_path = R31_DIR / "train_pairs.npz"
    val_path = R31_DIR / "fold0_val.json"
    counts_path = R31_DIR / "track_counts.json"

    if vocab_path.exists() and pairs_path.exists() and val_path.exists():
        print("  All data artifacts exist, skipping build_data")
        return

    from datasets import DownloadConfig, load_dataset

    print(f"{ts()} Loading train sessions...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))["train"]
    print(f"  {len(ds)} train sessions")

    all_tracks = set()
    track_counts: dict[str, int] = {}
    train_pairs_raw = []

    for item in ds:
        music = [str(c["content"]).strip() for c in item["conversations"]
                 if c["role"] == "music"]
        for t in music:
            all_tracks.add(t)
            track_counts[t] = track_counts.get(t, 0) + 1
        for k in range(1, len(music)):
            inp = music[max(0, k - MAX_LEN):k]
            target = music[k]
            train_pairs_raw.append((inp, target))

    sorted_tracks = sorted(all_tracks)
    vocab = {t: i + 1 for i, t in enumerate(sorted_tracks)}
    print(f"  Vocab: {len(vocab)} tracks, {len(train_pairs_raw)} training pairs")

    max_seq = max(len(p[0]) for p in train_pairs_raw)
    seqs = np.zeros((len(train_pairs_raw), max_seq), dtype=np.int32)
    seq_lens = np.zeros(len(train_pairs_raw), dtype=np.int32)
    targets = np.zeros(len(train_pairs_raw), dtype=np.int32)

    for i, (inp, tgt) in enumerate(train_pairs_raw):
        ids = [vocab[t] for t in inp if t in vocab]
        seqs[i, :len(ids)] = ids
        seq_lens[i] = len(ids)
        targets[i] = vocab.get(tgt, 0)

    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    np.savez_compressed(pairs_path, seqs=seqs, seq_lens=seq_lens, targets=targets)
    idx_counts = {str(vocab[t]): c for t, c in track_counts.items() if t in vocab}
    with open(counts_path, "w") as f:
        json.dump(idx_counts, f)
    print(f"  Saved vocab, {len(train_pairs_raw)} pairs, counts")

    del ds, train_pairs_raw
    gc.collect()

    print(f"{ts()} Loading dev payload for fold-0...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]

    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    fold0_val = sorted(folds[0].tolist())

    val_cases = []
    for j in fold0_val:
        c = cases[j]
        val_cases.append({
            "session_id": c["session_id"],
            "turn_number": c["turn_number"],
            "user_query": c["user_query"],
            "gt": c["gt"],
            "music_turns": c["music_turns"],
            "n_prior_music": c["n_prior_music"],
        })
    with open(val_path, "w") as f:
        json.dump(val_cases, f)

    with open(R21_OOF) as f:
        r21_all = json.load(f)
    r21_fold0 = [r21_all[j] for j in fold0_val]
    with open(R31_DIR / "fold0_r21_lists.json", "w") as f:
        json.dump(r21_fold0, f)

    print(f"  Saved fold-0 val ({len(val_cases)} cases) + R21 OOF lists")

    del payload, cases, r21_all
    gc.collect()

    print(f"{ts()} build_data complete. Restarting for next phase...")
    os.execv(sys.executable, [sys.executable] + sys.argv)  # noqa: S606


# ---------------------------------------------------------------------------
# Phase: overfit_smoke
# ---------------------------------------------------------------------------

def phase_overfit_smoke():
    print(f"{ts()} Phase: overfit_smoke")

    with open(R31_DIR / "vocab.json") as f:
        vocab = json.load(f)
    n_items = len(vocab)

    data = np.load(R31_DIR / "train_pairs.npz")
    seqs = data["seqs"][:256]
    seq_lens = data["seq_lens"][:256]
    targets = data["targets"][:256]

    model = SASRecV0(n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()

    n_smoke_epochs = 200
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  Smoke: 256 examples, {n_smoke_epochs} epochs")

    seq_t = torch.from_numpy(seqs).long()
    len_t = torch.from_numpy(seq_lens).long()
    tgt_t = torch.from_numpy(targets).long()
    session_vec = torch.zeros(len(targets), D_MODEL)

    for epoch in range(n_smoke_epochs):

        session_vec = model(seq_t, len_t)
        logits = model.score_all(session_vec) / TAU
        logits[:, 0] = -1e9
        loss = F.cross_entropy(logits, tgt_t)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                all_scores = model.score_all(session_vec)
                for i in range(len(targets)):
                    played = set(seqs[i][seqs[i] > 0].tolist())
                    for p in played:
                        all_scores[i, p] = -1e9
                    all_scores[i, 0] = -1e9
                ranks = (all_scores >= all_scores[torch.arange(len(targets)), tgt_t].unsqueeze(1)).sum(1)
                hit20 = (ranks <= 20).float().mean().item()
                med_rank = ranks.median().item()
            print(f"    Epoch {epoch+1}: loss={loss.item():.4f} hit@20={hit20:.3f} med_rank={med_rank:.0f}")

    with torch.no_grad():
        all_scores = model.score_all(session_vec)
        for i in range(len(targets)):
            played = set(seqs[i][seqs[i] > 0].tolist())
            for p in played:
                all_scores[i, p] = -1e9
            all_scores[i, 0] = -1e9
        ranks = (all_scores >= all_scores[torch.arange(len(targets)), tgt_t].unsqueeze(1)).sum(1)
        hit20 = (ranks <= 20).float().mean().item()
        med_rank = ranks.median().item()
    print(f"  Final: hit@20={hit20:.3f} med_rank={med_rank:.0f}")

    if hit20 < 0.90:
        print(f"\n  FAIL: train hit@20={hit20:.3f} < 0.90. Debug model/loss.")
        sys.exit(1)
    else:
        print(f"\n  PASS: train hit@20={hit20:.3f} >= 0.90")


# ---------------------------------------------------------------------------
# Phase: train_fold0
# ---------------------------------------------------------------------------

def phase_train_fold0(epochs=3):
    print(f"{ts()} Phase: train_fold0 (epochs={epochs})")

    with open(R31_DIR / "vocab.json") as f:
        vocab = json.load(f)
    n_items = len(vocab)

    data = np.load(R31_DIR / "train_pairs.npz")
    seqs = data["seqs"]
    seq_lens = data["seq_lens"]
    targets = data["targets"]
    n_train = len(seqs)
    print(f"  {n_train} training pairs, {n_items} items")

    model = SASRecV0(n_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    batch_size = 128
    n_batches = math.ceil(n_train / batch_size)
    total_steps = epochs * n_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=3e-4)

    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  batch_size={batch_size}, n_batches={n_batches}")

    model.train()
    step = 0
    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0
        epoch_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            b_seq = torch.from_numpy(seqs[b_idx]).long()
            b_len = torch.from_numpy(seq_lens[b_idx]).long()
            b_tgt = torch.from_numpy(targets[b_idx]).long()

            session_vec = model(b_seq, b_len)
            logits = model.score_all(session_vec) / TAU
            logits[:, 0] = -1e9
            loss = F.cross_entropy(logits, b_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_batches += 1
            step += 1

            if epoch_batches % 200 == 0:
                print(f"    batch {epoch_batches}/{n_batches}: loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}")

    model_path = R31_DIR / "model_fold0.pt"
    torch.save({
        "model_state": model.state_dict(),
        "n_items": n_items,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "max_len": MAX_LEN,
        "epochs": epochs,
    }, model_path)
    print(f"  Saved model to {model_path}")


# ---------------------------------------------------------------------------
# Phase: eval_fold0
# ---------------------------------------------------------------------------

def phase_eval_fold0():
    print(f"{ts()} Phase: eval_fold0")

    with open(R31_DIR / "vocab.json") as f:
        vocab = json.load(f)
    n_items = len(vocab)
    idx_to_track = {v: k for k, v in vocab.items()}

    checkpoint = torch.load(R31_DIR / "model_fold0.pt", weights_only=True)
    model = SASRecV0(checkpoint["n_items"], d_model=checkpoint["d_model"],
                     n_layers=checkpoint["n_layers"], n_heads=checkpoint["n_heads"],
                     max_len=checkpoint["max_len"])
    model.load_state_dict(checkpoint["model_state"])
    model.training = False

    with open(R31_DIR / "fold0_val.json") as f:
        val_cases = json.load(f)
    with open(R31_DIR / "fold0_r21_lists.json") as f:
        r21_lists = json.load(f)

    train_track_set = set(vocab.keys())
    topk = 300

    print(f"  {len(val_cases)} val cases, {n_items} items in vocab")

    r31_lists = []
    with torch.no_grad():
        item_embs = model.item_emb.weight.detach()

        for i, case in enumerate(val_cases):
            music = case["music_turns"]
            ids = [vocab[t] for t in music if t in vocab]
            if not ids:
                r31_lists.append([])
                continue

            ids = ids[-MAX_LEN:]
            seq = torch.zeros(1, len(ids), dtype=torch.long)
            seq[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            seq_len = torch.tensor([len(ids)], dtype=torch.long)

            session_vec = model(seq, seq_len)
            scores = (session_vec @ item_embs.T).squeeze(0).numpy()

            played_set = {vocab[t] for t in music if t in vocab}
            played_set.add(0)
            for p in played_set:
                scores[p] = -np.inf

            top_idx = np.argpartition(-scores, topk)[:topk]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            r31_lists.append([idx_to_track.get(int(j), "") for j in top_idx
                             if j in idx_to_track])

            if (i + 1) % 400 == 0:
                print(f"    {i+1}/{len(val_cases)} scored", flush=True)

    with open(R31_DIR / "fold0_r31_lists.json", "w") as f:
        json.dump(r31_lists, f)

    # Metrics
    def compute_metrics(cases, lists, label, ref_lists=None):
        metrics = {}
        for depth in range(8):
            idx = [i for i in range(len(cases)) if cases[i]["n_prior_music"] == depth]
            if not idx:
                continue
            for k in [20, 50, 100, 200, 300]:
                hit = sum(1 for i in idx if cases[i]["gt"] in set(lists[i][:k]))
                metrics[f"hist_{depth}_hit@{k}"] = hit / len(idx)

        hist57 = [i for i in range(len(cases)) if cases[i]["n_prior_music"] >= 5]
        for k in [20, 200]:
            hit = sum(1 for i in hist57 if cases[i]["gt"] in set(lists[i][:k]))
            metrics[f"hist57_hit@{k}"] = hit / len(hist57)

        all_200 = sum(1 for i in range(len(cases)) if cases[i]["gt"] in set(lists[i][:200]))
        metrics["all_hit@200"] = all_200 / len(cases)

        seen_idx = [i for i in range(len(cases)) if cases[i]["gt"] in train_track_set]
        unseen_idx = [i for i in range(len(cases)) if cases[i]["gt"] not in train_track_set]
        if seen_idx:
            metrics["seen_hit@200"] = sum(1 for i in seen_idx
                                          if cases[i]["gt"] in set(lists[i][:200])) / len(seen_idx)
        if unseen_idx:
            metrics["unseen_hit@200"] = sum(1 for i in unseen_idx
                                            if cases[i]["gt"] in set(lists[i][:200])) / len(unseen_idx)
        metrics["n_seen"] = len(seen_idx)
        metrics["n_unseen"] = len(unseen_idx)

        top1_repeat = sum(1 for i in range(len(cases))
                         if lists[i] and lists[i][0] in set(cases[i]["music_turns"]))
        metrics["top1_repeat_rate"] = top1_repeat / len(cases)

        if ref_lists:
            hist7 = [i for i in range(len(cases)) if cases[i]["n_prior_music"] == 7]
            unique_gt = sum(1 for i in hist7
                           if cases[i]["gt"] in set(lists[i][:200])
                           and cases[i]["gt"] not in set(ref_lists[i][:200]))
            metrics["unique_h7_gt_vs_ref"] = unique_gt
            lost_gt = sum(1 for i in hist7
                         if cases[i]["gt"] not in set(lists[i][:200])
                         and cases[i]["gt"] in set(ref_lists[i][:200]))
            metrics["lost_h7_gt_vs_ref"] = lost_gt

        h7_20 = metrics.get("hist_7_hit@20", 0)
        h7_200 = metrics.get("hist_7_hit@200", 0)
        h57_200 = metrics.get("hist57_hit@200", 0)
        a200 = metrics.get("all_hit@200", 0)
        print(f"  {label}: h7@20={h7_20:.4f}  h7@200={h7_200:.4f}  "
              f"h57@200={h57_200:.4f}  all@200={a200:.4f}")
        return metrics

    print(f"\n{ts()} Results:")
    r21_m = compute_metrics(val_cases, r21_lists, "R21_OOF")
    r31_m = compute_metrics(val_cases, r31_lists, "R31_V0", ref_lists=r21_lists)

    # Detailed report
    sep = "=" * 70
    print(f"\n{sep}")
    print("R31 FOLD-0 STANDALONE DIAGNOSTIC")
    print(sep)
    print(f"  {'Metric':<25} {'R21_OOF':>10} {'R31_V0':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for metric in ["hist_7_hit@20", "hist_7_hit@50", "hist_7_hit@100",
                    "hist_7_hit@200", "hist_7_hit@300",
                    "hist57_hit@20", "hist57_hit@200", "all_hit@200",
                    "seen_hit@200", "unseen_hit@200", "top1_repeat_rate"]:
        v1 = r21_m.get(metric, 0)
        v2 = r31_m.get(metric, 0)
        d = v2 - v1
        print(f"  {metric:<25} {v1:>10.4f} {v2:>10.4f} {d:>+10.4f}")

    unique = r31_m.get("unique_h7_gt_vs_ref", 0)
    lost = r31_m.get("lost_h7_gt_vs_ref", 0)
    print(f"\n  Unique h7 GTs vs R21: {unique}")
    print(f"  Lost h7 GTs vs R21: {lost}")
    print(f"  Seen/unseen: {r31_m.get('n_seen', 0)}/{r31_m.get('n_unseen', 0)}")

    # Gate check
    base_h7_200 = r21_m.get("hist_7_hit@200", 0)
    base_h7_20 = r21_m.get("hist_7_hit@20", 0)
    base_all = r21_m.get("all_hit@200", 0)
    r31_h7_200 = r31_m.get("hist_7_hit@200", 0)
    r31_h7_20 = r31_m.get("hist_7_hit@20", 0)
    r31_all = r31_m.get("all_hit@200", 0)

    g1 = (r31_h7_200 - base_h7_200) >= 0.05
    g2 = (r31_h7_20 - base_h7_20) >= -0.005
    g3 = unique >= 20
    g4 = (base_all - r31_all) <= 0.03

    print(f"\n{sep}")
    print("GATE CHECK")
    print(f"  h7@200 >= R21+0.05:    {r31_h7_200-base_h7_200:+.4f} {'PASS' if g1 else 'FAIL'}")
    print(f"  h7@20 >= R21-0.005:    {r31_h7_20-base_h7_20:+.4f} {'PASS' if g2 else 'FAIL'}")
    print(f"  unique h7 GTs >= 20:   {unique} {'PASS' if g3 else 'FAIL'}")
    print(f"  all@200 drop <= 0.03:  {base_all-r31_all:+.4f} {'PASS' if g4 else 'FAIL'}")
    status = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"
    print(f"  Overall: {status}")

    out_path = REPO / "exp" / "eval" / "expR31_stage1_standalone.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"r21": r21_m, "r31": r31_m, "status": status}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                        choices=["build_data", "overfit_smoke", "train_fold0", "eval_fold0"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R31 Sequential Retriever — {args.phase}")
    print("=" * 70)

    if args.phase == "build_data":
        phase_build_data()
    elif args.phase == "overfit_smoke":
        phase_overfit_smoke()
    elif args.phase == "train_fold0":
        phase_train_fold0(epochs=args.epochs)
    elif args.phase == "eval_fold0":
        phase_eval_fold0()

    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
