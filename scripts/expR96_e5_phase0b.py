#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R96 Phase 0b: E5-large-v2 query-side retriever (A100), OOF 5-fold.

A genuinely new retrieval family (E5, distinct pretraining from the BGE/ST that
produced R54/R84/R90/R21) to recover GT absent from the complete union (R96
Phase 0a.2: 22.3% / 1784 dev cases). Same structured query as R54 (user spec),
E5 prefix convention (query:/passage:), in-batch random negatives ONLY (no hard
negatives, per feedback_no_hardneg_aux_first_run).

This is the R54 Phase-3 pipeline with MODEL_NAME=e5-large-v2 and E5 prefixes
applied at every encode site. Output OOF lists feed expR96_phase0b_eval.py.

Usage (Colab A100):
  uv run python scripts/expR96_e5_phase0b.py --fold 0 --device cuda \
      --output-dir /content/drive/MyDrive/r96_e5_phase0b      # fold-0 smoke
  uv run python scripts/expR96_e5_phase0b.py --device cuda ...  # all 5 folds
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54_phase3_full5fold_train import (  # noqa: E402
    build_query_structured_from_dev,
    build_track_text,
    build_train_split_sample,
    list_ckpts,
    load_catalog,
    save_checkpoint,
    ts,
)

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
DEFAULT_OUTPUT_DIR = REPO / "cache" / "r96_e5" / "phase0b"
MODEL_NAME = "intfloat/e5-large-v2"
Q_PREFIX = "query: "
P_PREFIX = "passage: "

EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
TAU = 0.05
MAX_SEQ_LEN = 256
TOPK = 300
CKPT_EVERY = 200
LOG_EVERY = 50


def q_text(case, meta):
    return Q_PREFIX + build_query_structured_from_dev(case, meta)


def p_text(tid, meta):
    return P_PREFIX + build_track_text(tid, meta)


def train_fold(fold_idx, all_pairs, model_dir, ckpt_dir, device):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  fold {fold_idx}: {len(all_pairs)} pairs, {total_batches} batches, device={device}", flush=True)
    np.random.seed(fold_idx)
    torch.manual_seed(fold_idx)
    if device == "cuda":
        torch.cuda.manual_seed_all(fold_idx)

    ckpts = list_ckpts(ckpt_dir)
    start_batch, perm = 0, None
    epoch_loss, n_seen = 0.0, 0
    model_state = optimizer_state = None
    if ckpts:
        with open(ckpts[-1], "rb") as f:
            ck = pickle.load(f)  # noqa: S301
        if ck["total_batches"] == total_batches:
            start_batch = ck["batch_idx"]; perm = np.array(ck["perm"], dtype=np.int64)
            epoch_loss = ck["epoch_loss"]; n_seen = ck["n_batches_seen"]
            model_state = ck["model_state_dict"]; optimizer_state = ck["optimizer_state_dict"]
            np.random.set_state(ck["numpy_rng_state"]); torch.set_rng_state(ck["torch_rng_state"])
            print(f"  resumed at batch {start_batch}/{total_batches}", flush=True)
    if perm is None:
        perm = np.random.permutation(len(all_pairs)).astype(np.int64)

    model = SentenceTransformer(MODEL_NAME, device=device)
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    model.train()
    tokenizer = model.tokenizer

    def enc(texts):
        e = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        e = {k: v.to(device) for k, v in e.items()}
        return F_t.normalize(model.forward(e)["sentence_embedding"], dim=-1)

    t_epoch = time.time()
    for bi in range(start_batch, total_batches):
        idx = perm[bi * BATCH_SIZE: bi * BATCH_SIZE + BATCH_SIZE]
        queries = [all_pairs[int(i)][0] for i in idx]
        positives = [all_pairs[int(i)][1] for i in idx]
        if not queries:
            continue
        q_emb, p_emb = enc(queries), enc(positives)
        sim = q_emb @ p_emb.T / TAU                       # in-batch (random) negatives only
        loss = F_t.cross_entropy(sim, torch.arange(len(queries), device=sim.device))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += float(loss.item()); n_seen += 1; done = bi + 1
        if done % LOG_EVERY == 0:
            el = time.time() - t_epoch; spb = el / max(done - start_batch, 1)
            print(f"      f{fold_idx} {done}/{total_batches} loss={float(loss.item()):.4f} "
                  f"avg={epoch_loss/max(n_seen,1):.4f} eta={(total_batches-done)*spb:.0f}s", flush=True)
        if done % CKPT_EVERY == 0:
            save_checkpoint(ckpt_dir, model, optimizer, done, perm, total_batches, epoch_loss, n_seen, device)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def encode_catalog(model, meta, all_track_ids, device):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks (passage:) on {device}...", flush=True)
    model.eval()
    texts = [p_text(tid, meta) for tid in all_track_ids]
    bs = 256 if device == "cuda" else 128
    return model.encode(texts, batch_size=bs, show_progress_bar=False,
                        normalize_embeddings=True).astype(np.float32)


def retrieve(model, queries, track_embs, track_ids, played_lists, device, topk=TOPK):
    print(f"{ts()} Encoding {len(queries)} queries (query:) + retrieving top-{topk}...", flush=True)
    bs = 256 if device == "cuda" else 64
    q = model.encode(queries, batch_size=bs, show_progress_bar=False,
                     normalize_embeddings=True).astype(np.float32)
    out = []
    for i in range(len(queries)):
        played = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        ranked = np.argsort(-(q[i] @ track_embs.T))
        top = []
        for j in ranked:
            tid = track_ids[j]
            if tid not in played:
                top.append((tid, float(q[i] @ track_embs[j])))
                if len(top) >= topk:
                    break
        out.append(top)
    return out


def run_one_fold(fold_i, folds, cases, meta, all_track_ids, train_split_pairs, output_dir, device):
    t0 = time.time()
    print(f"\n{ts()} === E5 FOLD {fold_i} (device={device}) ===")
    fold_dir = output_dir / f"fold_{fold_i}"
    model_dir, embs_path = fold_dir / "model", fold_dir / "track_embs.npy"
    lists_path, ckpt_dir = fold_dir / "oof_lists.json", fold_dir / "checkpoints"
    if lists_path.exists():
        print(f"  fold {fold_i}: lists exist — skip"); return

    val_idx = folds[fold_i].tolist()
    val_set = set(val_idx)
    val_cases = [cases[j] for j in val_idx]
    train_dev = [cases[j] for j in range(len(cases)) if j not in val_set]
    dev_pairs = [(q_text(c, meta), p_text(c["gt"], meta)) for c in train_dev if c["gt"] in meta]
    # train-split pairs are raw (q,t) -> apply E5 prefixes
    split_pairs = [(Q_PREFIX + q, P_PREFIX + t) for q, t in train_split_pairs]
    all_pairs = dev_pairs + split_pairs
    print(f"  fold {fold_i}: dev={len(dev_pairs)} split={len(split_pairs)} total={len(all_pairs)}", flush=True)

    if model_dir.exists() and (model_dir / "config.json").exists():
        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = train_fold(fold_i, all_pairs, model_dir, ckpt_dir, device)

    track_embs = np.load(embs_path) if embs_path.exists() else encode_catalog(model, meta, all_track_ids, device)
    if not embs_path.exists():
        np.save(embs_path, track_embs)

    val_q = [q_text(c, meta) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    fold_lists = retrieve(model, val_q, track_embs, all_track_ids, val_played, device)
    manifest = {"fold": fold_i, "model": MODEL_NAME, "n_pairs": len(all_pairs),
                "n_val": len(val_cases), "query_format": "structured+e5prefix",
                "negatives": "in-batch random only", "device": device,
                "elapsed_s": time.time() - t0, "created_at": datetime.now().isoformat()}
    json.dump({"lists": fold_lists, "manifest": manifest, "val_idx": val_idx}, open(lists_path, "w"))
    hits = sum(1 for k, c in enumerate(val_cases) if c["gt"] in [t for t, _ in fold_lists[k][:300]])
    print(f"  fold {fold_i} val hit@300: {hits}/{len(val_cases)} ({hits/len(val_cases):.4f}) "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


def aggregate(output_dir, n):
    oof = [None] * n
    mans = {}
    for fi in range(5):
        p = output_dir / f"fold_{fi}" / "oof_lists.json"
        if not p.exists():
            print(f"  fold {fi} missing — skip aggregate"); return
        d = json.load(open(p))
        for k, v in enumerate(d["val_idx"]):
            oof[v] = d["lists"][k]
        mans[fi] = d["manifest"]
    json.dump({"lists": oof, "fold_manifests": mans, "n_cases": n,
               "experiment": "R96 Phase0b E5-large-v2", "created_at": datetime.now().isoformat()},
              open(output_dir / "oof_e5_lists.json", "w"))
    print(f"  Aggregated -> {output_dir / 'oof_e5_lists.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()
    if args.device is None:
        try:
            import torch  # type: ignore[reportMissingImports]
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"{ts()} Device: {args.device} | model: {MODEL_NAME}")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    payload = pickle.load(open(R12_CACHE, "rb"))  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    meta, all_track_ids = load_catalog()
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    train_split_pairs = build_train_split_sample(meta, set(sessions))

    if args.fold is not None:
        run_one_fold(args.fold, folds, cases, meta, all_track_ids, train_split_pairs, output_dir, args.device)
    else:
        for fi in range(5):
            run_one_fold(fi, folds, cases, meta, all_track_ids, train_split_pairs, output_dir, args.device)
    if not args.no_aggregate:
        aggregate(output_dir, n)
    print(f"\n{ts()} Done.")


if __name__ == "__main__":
    main()
