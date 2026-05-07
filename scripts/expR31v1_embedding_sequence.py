#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R31-V1: Embedding-sequence model over R21 track embeddings.

Instead of learned item IDs (V0 failed from sparsity), use fixed R21
production embeddings as item representations. The Transformer learns
to transform ordered track history into a next-track prediction vector
in R21's embedding space.

Phases (run with --phase):
  build_data      Build training pairs + fold-0 val (exec restarts)
  overfit_smoke   Overfit 256 examples
  train_fold0     Train on all train sessions
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
R21_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
V1_DIR = REPO / "cache" / "r31v1"

EMB_DIM = 768
D_MODEL = 256
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.2
MAX_LEN = 8
TAU = 0.05


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SeqTransformerV1(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, emb_dim)

    def forward(self, track_embs, seq_len):
        """
        track_embs: (B, L, emb_dim) — fixed R21 embeddings for input tracks
        seq_len: (B,) — actual sequence lengths
        Returns: (B, emb_dim) L2-normalized predicted vectors
        """
        bsz, slen, _ = track_embs.shape
        x = self.input_proj(track_embs)
        positions = torch.arange(slen, device=track_embs.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.pos_emb(positions)
        x = self.drop(self.norm(x))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(slen, device=track_embs.device)
        pad_mask = torch.arange(slen, device=track_embs.device).unsqueeze(0) >= seq_len.unsqueeze(1)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        idx = (seq_len - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
        out = x.gather(1, idx).squeeze(1)
        out = self.output_proj(out)
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Phase: build_data
# ---------------------------------------------------------------------------

def phase_build_data():
    print(f"{ts()} Phase: build_data")
    V1_DIR.mkdir(parents=True, exist_ok=True)

    pairs_path = V1_DIR / "train_pairs.json"
    val_path = V1_DIR / "fold0_val.json"

    if pairs_path.exists() and val_path.exists():
        print("  All data artifacts exist, skipping build_data")
        return

    from datasets import DownloadConfig, load_dataset

    print(f"{ts()} Loading R21 track index...")
    r21_ids = json.loads(Path(R21_IDS).read_text())
    r21_id_to_idx = {tid: i for i, tid in enumerate(r21_ids)}
    print(f"  {len(r21_ids)} tracks in R21 catalog")

    print(f"{ts()} Loading train sessions...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))["train"]
    print(f"  {len(ds)} train sessions")

    train_pairs = []
    for item in ds:
        music = [str(c["content"]).strip() for c in item["conversations"]
                 if c["role"] == "music"]
        for k in range(1, len(music)):
            inp = music[max(0, k - MAX_LEN):k]
            inp_idx = [r21_id_to_idx[t] for t in inp if t in r21_id_to_idx]
            tgt_idx = r21_id_to_idx.get(music[k])
            if inp_idx and tgt_idx is not None:
                train_pairs.append({"input_idx": inp_idx, "target_idx": tgt_idx})

    print(f"  {len(train_pairs)} training pairs (tracks in R21 catalog)")

    with open(pairs_path, "w") as f:
        json.dump(train_pairs, f)

    del ds
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
        music_idx = [r21_id_to_idx[t] for t in c["music_turns"] if t in r21_id_to_idx]
        gt_idx = r21_id_to_idx.get(c["gt"])
        val_cases.append({
            "session_id": c["session_id"],
            "turn_number": c["turn_number"],
            "gt": c["gt"],
            "gt_idx": gt_idx,
            "music_turns": c["music_turns"],
            "music_idx": music_idx,
            "n_prior_music": c["n_prior_music"],
        })
    with open(val_path, "w") as f:
        json.dump(val_cases, f)

    with open(R21_OOF) as f:
        r21_all = json.load(f)
    r21_fold0 = [r21_all[j] for j in fold0_val]
    with open(V1_DIR / "fold0_r21_lists.json", "w") as f:
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

    r21_embs = np.load(R21_EMBS).astype(np.float32)
    r21_embs_norm = r21_embs / (np.linalg.norm(r21_embs, axis=1, keepdims=True) + 1e-8)
    catalog_t = torch.from_numpy(r21_embs_norm)
    n_catalog = len(r21_embs)
    print(f"  Catalog: {n_catalog} tracks, {EMB_DIM}d")

    with open(V1_DIR / "train_pairs.json") as f:
        all_pairs = json.load(f)
    pairs = all_pairs[:256]

    model = SeqTransformerV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")
    print("  Smoke: 256 examples, 100 epochs")

    # Build batch once (reused every epoch)
    smoke_embs_list = []
    smoke_lens = []
    smoke_targets = []
    for p in pairs:
        inp_idx = p["input_idx"][-MAX_LEN:]
        smoke_embs_list.append(torch.from_numpy(r21_embs_norm[inp_idx]))
        smoke_lens.append(len(inp_idx))
        smoke_targets.append(p["target_idx"])

    smoke_max_len = max(smoke_lens)
    smoke_batch = torch.zeros(len(pairs), smoke_max_len, EMB_DIM)
    for i, embs in enumerate(smoke_embs_list):
        smoke_batch[i, :len(embs)] = embs
    batch_lens_t = torch.tensor(smoke_lens, dtype=torch.long)
    batch_targets_t = torch.tensor(smoke_targets, dtype=torch.long)
    pred_vec = torch.zeros(len(pairs), EMB_DIM)

    for epoch in range(100):
        pred_vec = model(smoke_batch, batch_lens_t)
        logits = pred_vec @ catalog_t.T / TAU
        loss = F.cross_entropy(logits, batch_targets_t)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                scores = pred_vec @ catalog_t.T
                for i, p in enumerate(pairs):
                    played = set(p["input_idx"])
                    for pi in played:
                        scores[i, pi] = -1e9
                ranks = (scores >= scores[torch.arange(len(pairs)), batch_targets_t].unsqueeze(1)).sum(1)
                hit20 = (ranks <= 20).float().mean().item()
                med_rank = ranks.median().item()
            print(f"    Epoch {epoch+1}: loss={loss.item():.4f} hit@20={hit20:.3f} med_rank={med_rank:.0f}")

    with torch.no_grad():
        scores = pred_vec @ catalog_t.T
        for i, p in enumerate(pairs):
            played = set(p["input_idx"])
            for pi in played:
                scores[i, pi] = -1e9
        ranks = (scores >= scores[torch.arange(len(pairs)), batch_targets_t].unsqueeze(1)).sum(1)
        hit20 = (ranks <= 20).float().mean().item()
        med_rank = ranks.median().item()
    print(f"  Final: hit@20={hit20:.3f} med_rank={med_rank:.0f}")

    if hit20 < 0.90:
        print(f"\n  FAIL: train hit@20={hit20:.3f} < 0.90")
        sys.exit(1)
    else:
        print(f"\n  PASS: train hit@20={hit20:.3f} >= 0.90")


# ---------------------------------------------------------------------------
# Phase: train_fold0
# ---------------------------------------------------------------------------

def phase_train_fold0(epochs=5):
    print(f"{ts()} Phase: train_fold0 (epochs={epochs})")

    r21_embs = np.load(R21_EMBS).astype(np.float32)
    r21_embs_norm = r21_embs / (np.linalg.norm(r21_embs, axis=1, keepdims=True) + 1e-8)
    catalog_t = torch.from_numpy(r21_embs_norm)
    n_catalog = len(r21_embs)

    with open(V1_DIR / "train_pairs.json") as f:
        all_pairs = json.load(f)
    n_train = len(all_pairs)
    print(f"  {n_train} training pairs, {n_catalog} catalog tracks")

    model = SeqTransformerV1()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    batch_size = 64
    n_batches = math.ceil(n_train / batch_size)
    total_steps = epochs * n_batches
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=2e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")
    print(f"  batch_size={batch_size}, n_batches={n_batches}")

    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            batch_pairs = [all_pairs[i] for i in b_idx]

            batch_embs_list = []
            batch_lens = []
            batch_targets = []
            for p in batch_pairs:
                inp_idx = p["input_idx"][-MAX_LEN:]
                embs = r21_embs_norm[inp_idx]
                batch_embs_list.append(embs)
                batch_lens.append(len(inp_idx))
                batch_targets.append(p["target_idx"])

            max_len = max(batch_lens)
            batch_embs = np.zeros((len(batch_pairs), max_len, EMB_DIM), dtype=np.float32)
            for i, embs in enumerate(batch_embs_list):
                batch_embs[i, :len(embs)] = embs

            batch_embs_t = torch.from_numpy(batch_embs)
            batch_lens_t = torch.tensor(batch_lens, dtype=torch.long)
            batch_targets_t = torch.tensor(batch_targets, dtype=torch.long)

            pred_vec = model(batch_embs_t, batch_lens_t)
            logits = pred_vec @ catalog_t.T / TAU
            loss = F.cross_entropy(logits, batch_targets_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_batches += 1

            if epoch_batches % 200 == 0:
                print(f"    batch {epoch_batches}/{n_batches}: loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_loss = epoch_loss / max(epoch_batches, 1)
        print(f"  Epoch {epoch}: avg_loss={avg_loss:.4f}")

        ckpt = {
            "model_state": model.state_dict(),
            "emb_dim": EMB_DIM, "d_model": D_MODEL,
            "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "max_len": MAX_LEN, "epoch": epoch, "avg_loss": avg_loss,
        }
        torch.save(ckpt, V1_DIR / f"model_fold0_ep{epoch}.pt")
        print(f"    checkpoint saved (epoch {epoch})", flush=True)

    model_path = V1_DIR / "model_fold0.pt"
    torch.save({
        "model_state": model.state_dict(),
        "emb_dim": EMB_DIM,
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

    r21_ids = json.loads(Path(R21_IDS).read_text())

    r21_embs = np.load(R21_EMBS).astype(np.float32)
    r21_embs_norm = r21_embs / (np.linalg.norm(r21_embs, axis=1, keepdims=True) + 1e-8)
    catalog_t = torch.from_numpy(r21_embs_norm)

    checkpoint = torch.load(V1_DIR / "model_fold0.pt", weights_only=True)
    model = SeqTransformerV1(emb_dim=checkpoint["emb_dim"], d_model=checkpoint["d_model"],
                              n_layers=checkpoint["n_layers"], n_heads=checkpoint["n_heads"],
                              max_len=checkpoint["max_len"])
    model.load_state_dict(checkpoint["model_state"])
    for m in model.modules():
        if hasattr(m, "training"):
            m.training = False

    with open(V1_DIR / "fold0_val.json") as f:
        val_cases = json.load(f)
    with open(V1_DIR / "fold0_r21_lists.json") as f:
        r21_lists = json.load(f)

    train_track_set = set()
    with open(V1_DIR / "train_pairs.json") as f:
        for p in json.load(f):
            for idx in p["input_idx"]:
                train_track_set.add(r21_ids[idx])
            train_track_set.add(r21_ids[p["target_idx"]])

    topk = 300
    r21_id_set = set(r21_ids)

    unknown_gt_count = sum(1 for c in val_cases if c["gt"] not in r21_id_set)
    empty_input_count = sum(1 for c in val_cases if not c["music_idx"])
    unknown_input_total = sum(
        len(c["music_turns"]) - len(c["music_idx"]) for c in val_cases)
    total_input_tracks = sum(len(c["music_turns"]) for c in val_cases)
    target_in_input_count = sum(
        1 for c in val_cases if c["gt_idx"] is not None and c["gt_idx"] in set(c["music_idx"]))

    print(f"  {len(val_cases)} val cases, {len(r21_ids)} catalog tracks")
    print(f"  unknown_gt: {unknown_gt_count}, empty_input: {empty_input_count}, "
          f"unknown_input_rate: {unknown_input_total}/{total_input_tracks}, "
          f"target_in_input: {target_in_input_count}")

    r31_lists = []
    with torch.no_grad():
        for i, case in enumerate(val_cases):
            music_idx = case["music_idx"]
            if not music_idx:
                r31_lists.append([])
                continue

            inp_idx = music_idx[-MAX_LEN:]
            track_embs = torch.from_numpy(r21_embs_norm[inp_idx]).unsqueeze(0)
            seq_len = torch.tensor([len(inp_idx)], dtype=torch.long)

            pred_vec = model(track_embs, seq_len)
            scores = (pred_vec @ catalog_t.T).squeeze(0).numpy()

            played_idx = set(music_idx)
            for pi in played_idx:
                scores[pi] = -np.inf

            top_idx = np.argpartition(-scores, topk)[:topk]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            r31_lists.append([r21_ids[j] for j in top_idx])

            if (i + 1) % 400 == 0:
                print(f"    {i+1}/{len(val_cases)} scored", flush=True)

    with open(V1_DIR / "fold0_r31v1_lists.json", "w") as f:
        json.dump(r31_lists, f)

    def compute_metrics(cases, lists, label, ref_lists=None):
        metrics: dict[str, float | int] = {}
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
    r31_m = compute_metrics(val_cases, r31_lists, "R31_V1", ref_lists=r21_lists)

    sep = "=" * 70
    print(f"\n{sep}")
    print("R31-V1 FOLD-0 STANDALONE DIAGNOSTIC")
    print(sep)
    print(f"  {'Metric':<25} {'R21_OOF':>10} {'R31_V1':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for metric in ["hist_7_hit@20", "hist_7_hit@50", "hist_7_hit@100",
                    "hist_7_hit@200", "hist_7_hit@300",
                    "hist57_hit@20", "hist57_hit@200", "all_hit@200",
                    "seen_hit@200", "unseen_hit@200", "top1_repeat_rate"]:
        v1 = r21_m.get(metric, 0)
        v2 = r31_m.get(metric, 0)
        print(f"  {metric:<25} {v1:>10.4f} {v2:>10.4f} {v2-v1:>+10.4f}")

    unique = r31_m.get("unique_h7_gt_vs_ref", 0)
    lost = r31_m.get("lost_h7_gt_vs_ref", 0)
    print(f"\n  Unique h7 GTs vs R21: {unique}")
    print(f"  Lost h7 GTs vs R21: {lost}")

    base_h7_200 = r21_m.get("hist_7_hit@200", 0)
    base_h7_20 = r21_m.get("hist_7_hit@20", 0)
    base_all = r21_m.get("all_hit@200", 0)
    r31_h7_200 = r31_m.get("hist_7_hit@200", 0)
    r31_h7_20 = r31_m.get("hist_7_hit@20", 0)
    r31_all = r31_m.get("all_hit@200", 0)

    g1 = (r31_h7_200 - base_h7_200) >= 0.05
    g2 = (r31_h7_20 - base_h7_20) >= -0.005
    g3 = int(unique) >= 20
    g4 = (base_all - r31_all) <= 0.03

    print(f"\n{sep}")
    print("GATE CHECK")
    print(f"  h7@200 >= R21+0.05:    {r31_h7_200-base_h7_200:+.4f} {'PASS' if g1 else 'FAIL'}")
    print(f"  h7@20 >= R21-0.005:    {r31_h7_20-base_h7_20:+.4f} {'PASS' if g2 else 'FAIL'}")
    print(f"  unique h7 GTs >= 20:   {unique} {'PASS' if g3 else 'FAIL'}")
    print(f"  all@200 drop <= 0.03:  {base_all-r31_all:+.4f} {'PASS' if g4 else 'FAIL'}")
    status = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"
    print(f"  Overall: {status}")

    out_path = REPO / "exp" / "eval" / "expR31v1_stage1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"r21": r21_m, "r31v1": r31_m, "status": status,
                    "diagnostics": {"unknown_gt": unknown_gt_count,
                                    "empty_input": empty_input_count,
                                    "unknown_input_rate": f"{unknown_input_total}/{total_input_tracks}",
                                    "target_in_input": target_in_input_count}}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                        choices=["build_data", "overfit_smoke", "train_fold0", "eval_fold0"])
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R31-V1 Embedding Sequence Model — {args.phase}")
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
