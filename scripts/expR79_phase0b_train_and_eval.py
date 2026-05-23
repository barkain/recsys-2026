#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R79 Phase 0B — Fine-tune BGE-large with hard negatives, eval fold-0 standalone.

Designed to run on Colab A100. Single script. ~$10, ~2 hours.

Inputs (downloaded automatically from public GitHub repo or HF):
- cache/r79/training_pairs.pkl (16 MB) — from Phase 0A
- cache/r79/eval_baseline.json — Phase 0B baseline metrics
- BAAI/bge-large-en-v1.5 — from HuggingFace
- TalkPlayData-Challenge-Track-Metadata — from HuggingFace (catalog)
- _R12 payload or similar — has fold-0 dev cases

Pipeline:
1. Load training_pairs (8000 cases with GT + hard_negs + query text + played tracks)
2. Construct query text for each case: structured concat of user_query + history
3. Construct track text for each track_id used: title + artist + album + tags
4. Fine-tune BGE-large with InfoNCE: per case, softmax over [GT, hard_negs]
5. Encode all 47K catalog tracks
6. Encode fold-0 dev queries
7. Cosine score, take top-20 per case
8. Compare to OOF R54c baseline on predeclared gates
9. Report PASS/FAIL

Determinism: torch.use_deterministic_algorithms(True, warn_only=True),
PYTHONHASHSEED=0, seed everywhere.
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

# Determinism
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
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

TRAINING_PAIRS = REPO / "cache" / "r79" / "training_pairs.pkl"
EVAL_BASELINE = REPO / "cache" / "r79" / "eval_baseline.json"
OUT_DIR = REPO / "cache" / "r79" / "phase0b"
OUT_JSON = REPO / "exp" / "eval" / "expR79_phase0b_result.json"
OUT_DOC = REPO / "docs" / "r79_phase0b_result.md"

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MAX_LEN_QUERY = 192
MAX_LEN_TRACK = 64
BATCH_CASES = 8
N_EPOCHS = 2
LR = 1e-5
USE_BF16 = True
HARD_NEGS_PER_CASE = 16  # cap from 19 to bound memory
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.05
CATALOG_BATCH = 128
QUERY_ENCODE_BATCH = 64

# Predeclared gates (fold-0 only)
GATE_H7_DELTA = 0.005
GATE_SAME_DELTA = -0.002
GATE_OVERLAP_FLOOR = 14  # / 20
GATE_CHURN_CAP = 25  # / 80

TOP_20 = 20
TOP_30 = 30
FOLD = 0


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def build_query_text(case_dict: dict) -> str:
    """Match R54-style structured query encoding."""
    user_q = case_dict.get("user_query", "")
    history = case_dict.get("history_text", "")
    played_summary = case_dict.get("history_summary", "")
    parts = [f"Query: {user_q}"]
    if played_summary:
        parts.append(f"Played artists: {played_summary}")
    if history:
        parts.append(f"Conversation: {history}")
    return " | ".join(parts)


def build_track_text(track_id: str, catalog: dict) -> str:
    m = catalog.get(track_id, {})
    title = (m.get("track_name") or ["unknown"])[0] if isinstance(m.get("track_name"), list) else m.get("track_name", "unknown")
    artist = (m.get("artist_name") or ["unknown"])[0] if isinstance(m.get("artist_name"), list) else m.get("artist_name", "unknown")
    album = ""
    if isinstance(m.get("album_name"), list) and m["album_name"]:
        album = m["album_name"][0]
    tags = m.get("tag_list") or []
    if not isinstance(tags, list):
        tags = []
    tag_str = ", ".join(tags[:5]) if tags else ""
    rel = m.get("release_date") or ""
    year = rel[:4] if rel and len(rel) >= 4 else ""
    parts = [f"Title: {title}", f"Artist: {artist}"]
    if album:
        parts.append(f"Album: {album}")
    if tag_str:
        parts.append(f"Genre: {tag_str}")
    if year:
        parts.append(f"Year: {year}")
    return " | ".join(parts)


def load_catalog() -> dict[str, dict]:
    """Load track catalog from HuggingFace."""
    from datasets import DownloadConfig, load_dataset
    print(f"{ts()} Loading catalog from HuggingFace ...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig())["all_tracks"]
    cat = {}
    for item in ds:
        tid = str(item["track_id"])
        cat[tid] = {
            "track_name": item.get("track_name"),
            "artist_name": item.get("artist_name"),
            "album_name": item.get("album_name"),
            "tag_list": item.get("tag_list") or item.get("tags") or [],
            "release_date": item.get("release_date") or "",
        }
    print(f"  loaded {len(cat)} catalog entries", flush=True)
    return cat


def fine_tune(
    model: Any,
    tokenizer: Any,
    training_pairs: list[dict],
    catalog: dict,
    device: torch.device,
    epochs: int = N_EPOCHS,
    batch_cases: int = BATCH_CASES,
    lr: float = LR,
):
    """Fine-tune BGE-large with InfoNCE on (query, GT, hard_negs)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    n = len(training_pairs)
    n_batches = (n + batch_cases - 1) // batch_cases
    total_steps = epochs * n_batches
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(warmup_steps, 1))
        if step < warmup_steps
        else max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1)),
    )

    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if USE_BF16 and device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))

    global_step = 0
    for epoch in range(epochs):
        random.shuffle(training_pairs)
        t_epoch = time.time()
        total_loss = 0.0
        n_skipped = 0
        for batch_start in range(0, n, batch_cases):
            batch = training_pairs[batch_start:batch_start + batch_cases]

            # Build text lists for this batch
            queries = []
            pos_texts = []
            all_neg_texts = []      # flat list of all negs across batch
            neg_counts = []         # per-case neg count for split
            for case in batch:
                queries.append(build_query_text(case))
                pos_texts.append(build_track_text(case["gt"], catalog))
                negs = case["hard_negs"][:HARD_NEGS_PER_CASE]
                neg_counts.append(len(negs))
                all_neg_texts.extend([build_track_text(neg, catalog) for neg in negs])

            n_total_negs = len(all_neg_texts)
            n_valid = sum(1 for c in neg_counts if c > 0)
            if n_valid == 0:
                n_skipped += len(batch)
                continue

            with autocast_ctx:
                # Encode queries
                q_inputs = tokenizer(queries, padding=True, truncation=True,
                                     max_length=MAX_LEN_QUERY, return_tensors="pt").to(device)
                q_emb = F.normalize(model(**q_inputs).last_hidden_state[:, 0, :], p=2, dim=-1)

                # Encode positives
                p_inputs = tokenizer(pos_texts, padding=True, truncation=True,
                                     max_length=MAX_LEN_TRACK, return_tensors="pt").to(device)
                p_emb = F.normalize(model(**p_inputs).last_hidden_state[:, 0, :], p=2, dim=-1)

                # Encode ALL negatives in one batch
                n_inputs = tokenizer(all_neg_texts, padding=True, truncation=True,
                                     max_length=MAX_LEN_TRACK, return_tensors="pt").to(device)
                n_emb_flat = F.normalize(model(**n_inputs).last_hidden_state[:, 0, :], p=2, dim=-1)

                # Compute losses per-case
                loss = torch.tensor(0.0, device=device)
                offset = 0
                for ci, k in enumerate(neg_counts):
                    if k == 0:
                        n_skipped += 1
                        continue
                    q_i = q_emb[ci]
                    pos_score = (q_i * p_emb[ci]).sum()  # cosine
                    n_i = n_emb_flat[offset:offset + k]
                    offset += k
                    neg_scores = n_i @ q_i
                    all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                    logits = all_scores * 20.0  # temperature 0.05
                    target = torch.tensor(0, device=device)
                    case_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                    loss = loss + case_loss
                loss = loss / n_valid

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            global_step += 1
            if global_step % 25 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"    epoch {epoch+1}/{epochs} step {global_step}/{total_steps} "
                      f"loss={float(loss.item()):.4f} lr={lr_now:.2e}", flush=True)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch+1} done in {time.time()-t_epoch:.0f}s "
              f"avg_loss={avg_loss:.4f} skipped={n_skipped}", flush=True)


def encode_texts(model, tokenizer, texts: list[str], device, batch: int,
                 max_len: int) -> np.ndarray:
    """Encode texts in batches, return (N, D) normalized."""
    model.eval()
    embs = []
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if USE_BF16 and device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))
    with torch.no_grad(), autocast_ctx:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            inp = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt").to(device)
            out = model(**inp)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=-1)
            embs.append(emb.float().cpu().numpy())  # cast bf16 → fp32 for numpy
            if (i // batch) % 20 == 0:
                print(f"    encoded {i+len(chunk)}/{len(texts)}", flush=True)
    return np.concatenate(embs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-cases", type=int, default=BATCH_CASES)
    parser.add_argument("--catalog-batch", type=int, default=CATALOG_BATCH)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{ts()} R79 Phase 0B — fine-tune BGE-large + standalone fold-0 eval")
    print(f"  model={args.model} epochs={args.epochs} lr={args.lr} batch_cases={args.batch_cases}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM free: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB")

    print(f"\n{ts()} Loading training pairs ...", flush=True)
    with open(TRAINING_PAIRS, "rb") as f:
        data = pickle.load(f)
    pairs_all = data["training_pairs"]
    print(f"  {len(pairs_all)} training pairs")

    # Split: train on all 8000 (the OOF top-20 hard negs are per-case OOF already)
    print(f"\n{ts()} Loading catalog from HuggingFace ...", flush=True)
    catalog = load_catalog()

    print(f"\n{ts()} Loading BGE-large model ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    print(f"  model loaded, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M",
          flush=True)

    # ---- Fine-tune ----
    print(f"\n{ts()} === Fine-tuning ===")
    t_train = time.time()
    fine_tune(model, tokenizer, pairs_all, catalog, device,
              epochs=args.epochs, batch_cases=args.batch_cases, lr=args.lr)
    print(f"  Fine-tuning done in {time.time()-t_train:.0f}s ({time.time()-t_train:.0f}s)")

    # ---- Save model ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUT_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"{ts()} Saved fine-tuned model → {model_dir}")

    # ---- Encode catalog ----
    print(f"\n{ts()} === Encoding full catalog ({len(catalog)} tracks) ===")
    track_ids = sorted(catalog.keys())
    track_texts = [build_track_text(tid, catalog) for tid in track_ids]
    t_enc = time.time()
    track_embs = encode_texts(model, tokenizer, track_texts, device,
                              batch=args.catalog_batch, max_len=MAX_LEN_TRACK)
    print(f"  encoded {len(track_ids)} tracks in {time.time()-t_enc:.0f}s "
          f"({track_embs.shape})")
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    np.save(OUT_DIR / "track_embs.npy", track_embs)
    with open(OUT_DIR / "track_ids.json", "w") as f:
        json.dump(track_ids, f)

    # ---- Encode fold-0 dev queries ----
    print(f"\n{ts()} === Encoding fold-0 dev queries ===")
    fold0_pairs = [p for p in pairs_all if p["fold"] == FOLD]
    print(f"  fold-0 cases: {len(fold0_pairs)}")
    queries = [build_query_text(p) for p in fold0_pairs]
    query_embs = encode_texts(model, tokenizer, queries, device,
                              batch=QUERY_ENCODE_BATCH, max_len=MAX_LEN_QUERY)
    print(f"  encoded {len(queries)} queries ({query_embs.shape})")

    # ---- Score: cosine top-20 per case ----
    print(f"\n{ts()} === Scoring + ranking ===")
    # Filter out played tracks per case before ranking
    cosine_scores = query_embs @ track_embs.T  # (N_cases, N_tracks)
    print(f"  cosine matrix: {cosine_scores.shape}")

    r79_results = []
    for ci, case in enumerate(fold0_pairs):
        played_set = set(case["played_tracks"])
        # Mask played tracks
        scores = cosine_scores[ci].copy()
        for pt in played_set:
            if pt in track_to_idx:
                scores[track_to_idx[pt]] = -1e9
        # Top-20
        order = np.argpartition(-scores, TOP_20)[:TOP_20]
        order = order[np.argsort(-scores[order])]
        r79_top20 = [track_ids[int(j)] for j in order]
        gt = case["gt"]
        r_gt_rank = -1
        for r, tid in enumerate(r79_top20):
            if tid == gt:
                r_gt_rank = r + 1
                break
        # Compare to baseline (OOF R54c top-20 already in case)
        base_top20 = case["oof_top20"][:TOP_20]
        b_gt_rank = -1
        for r, tid in enumerate(base_top20):
            if tid == gt:
                b_gt_rank = r + 1
                break
        r79_results.append({
            "case_idx": case["case_idx"],
            "is_h7": case["is_h7"],
            "is_same_artist": case["is_same_artist"],
            "b_gt_rank": b_gt_rank,
            "r_gt_rank": r_gt_rank,
            "b_in_top20": 0 < b_gt_rank <= TOP_20,
            "r_in_top20": 0 < r_gt_rank <= TOP_20,
            "b_ndcg": ndcg_at_k(b_gt_rank, TOP_20),
            "r_ndcg": ndcg_at_k(r_gt_rank, TOP_20),
            "top1_changed": 1 if (len(base_top20) and len(r79_top20)
                                  and base_top20[0] != r79_top20[0]) else 0,
            "top20_overlap": len(set(base_top20) & set(r79_top20)),
        })

    # ---- Compute metrics ----
    def avg(rs, key, where=None):
        if where is not None:
            rs = [r for r in rs if where(r)]
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    h7_rows = [r for r in r79_results if r["is_h7"]]
    same_rows = [r for r in r79_results if r["is_same_artist"]]
    diff_rows = [r for r in r79_results if not r["is_same_artist"]]
    h7_same = [r for r in h7_rows if r["is_same_artist"]]
    h7_diff = [r for r in h7_rows if not r["is_same_artist"]]

    metrics = {}
    for name, rows in [("all_fold0", r79_results), ("h7", h7_rows),
                       ("same_artist", same_rows), ("diff_artist", diff_rows),
                       ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        b = avg(rows, "b_ndcg")
        r = avg(rows, "r_ndcg")
        metrics[name] = {"n": len(rows), "baseline": b, "r79": r, "delta": r - b}

    h7_rec = sum(1 for r in h7_rows if r["r_in_top20"] and not r["b_in_top20"])
    h7_lost = sum(1 for r in h7_rows if r["b_in_top20"] and not r["r_in_top20"])
    h7_net = h7_rec - h7_lost
    top1_churn = sum(r["top1_changed"] for r in r79_results)
    churn_per_80 = top1_churn / len(r79_results) * 80
    overlap_mean = avg(r79_results, "top20_overlap")

    h7_d = metrics["h7"]["delta"]
    sa_d = metrics["same_artist"]["delta"]
    da_d = metrics["diff_artist"]["delta"]

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
              f"r79={m['r79']:.4f}  Δ={m['delta']:+.4f}", flush=True)
    print(f"  h7 recovered={h7_rec}  lost={h7_lost}  net={h7_net:+d}")
    print(f"  top1_churn={top1_churn}  per_80={churn_per_80:.2f}  "
          f"overlap_mean={overlap_mean:.2f}/20")
    print(f"\n  Gates:")
    print(f"    h7 Δ ≥ +0.005:        {gate_h7}  ({h7_d:+.4f})")
    print(f"    same-art Δ ≥ -0.002:   {gate_sa}  ({sa_d:+.4f})")
    print(f"    h7 net > 0:           {gate_net}  ({h7_net:+d})")
    print(f"    churn ≤ 25:           {gate_churn}  ({churn_per_80:.2f})")
    print(f"    overlap ≥ 14:         {gate_overlap}  ({overlap_mean:.2f})")
    print(f"  VERDICT: {verdict}")

    out = {
        "experiment": "R79 Phase 0B — fine-tune + fold-0 standalone eval",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "hyperparams": {
            "model": args.model,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_cases": args.batch_cases,
            "max_len_query": MAX_LEN_QUERY,
            "max_len_track": MAX_LEN_TRACK,
            "n_training_pairs": len(pairs_all),
            "n_fold0_eval": len(fold0_pairs),
        },
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
        "model_path": str(model_dir),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved → {OUT_JSON}")

    md = [
        "# R79 Phase 0B — fine-tune + standalone fold-0 eval",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s",
        f"Model: {args.model}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Metrics",
        "",
        "| Subset | n | OOF R54c | R79 standalone | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        md.append(f"| {name} | {m['n']} | {m['baseline']:.4f} | {m['r79']:.4f} | {m['delta']:+.4f} |")
    md += [
        "",
        f"- h7 recovered={h7_rec}, lost={h7_lost}, net={h7_net:+d}",
        f"- top-1 churn /80 = {churn_per_80:.2f}",
        f"- top-20 overlap mean = {overlap_mean:.2f}/20",
        "",
        "## Gates",
        f"- h7 Δ ≥ +0.005: **{gate_h7}** ({h7_d:+.4f})",
        f"- same-artist Δ ≥ -0.002: **{gate_sa}** ({sa_d:+.4f})",
        f"- h7 net > 0: **{gate_net}** ({h7_net:+d})",
        f"- top-1 churn ≤ 25: **{gate_churn}** ({churn_per_80:.2f})",
        f"- top-20 overlap ≥ 14: **{gate_overlap}** ({overlap_mean:.2f})",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved → {OUT_DOC}")


if __name__ == "__main__":
    main()
