#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Train R21 production model on all training data and cache model + track embeddings."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = REPO_ROOT / "cache" / "r21_production"
R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_track_text(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def build_query_text(history, user_query):
    parts = []
    for h in history:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(user_query)
    return " ".join(parts[-3:])


def main():
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer, InputExample

    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load all training data
    print(f"{ts()} Loading training data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]

    # Load catalog metadata
    from datasets import Dataset, concatenate_datasets
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    all_track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        all_track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    print(f"  {len(cases)} training cases, {len(all_track_ids)} catalog tracks")

    # Build training pairs
    print(f"{ts()} Building training pairs...", flush=True)
    examples = []
    for c in cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text(c["history"], c["user_query"])
        track_text = build_track_text(gt, meta)
        examples.append(InputExample(texts=[query_text, track_text]))
    print(f"  {len(examples)} positive pairs")

    # Train
    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    batch_size = 32
    epochs = 2

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt").to(device)
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    print(f"{ts()} Training on {device} ({len(examples)} pairs, {epochs} epochs, bs={batch_size})...", flush=True)
    model.train()
    for epoch in range(epochs):
        np.random.shuffle(examples)
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            queries = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]

            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)

            sim = q_emb @ p_emb.T / 0.05
            labels = torch.arange(len(batch), device=sim.device)
            loss = F_t.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"    batch {n_batches}: loss={loss.item():.4f}", flush=True)

        print(f"  Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.4f}", flush=True)

    # Save model
    model_dir = CACHE_DIR / "model"
    model.save(str(model_dir))
    print(f"{ts()} Model saved to {model_dir}")

    # Encode all catalog tracks
    print(f"{ts()} Encoding {len(all_track_ids)} catalog tracks...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)
    np.save(CACHE_DIR / "track_embeddings.npy", track_embs)
    with open(CACHE_DIR / "track_ids.json", "w") as f:
        json.dump(all_track_ids, f)
    print(f"  Saved track embeddings: {track_embs.shape}")

    # Also encode all dev queries for LambdaRank training
    print(f"{ts()} Encoding {len(cases)} dev queries...", flush=True)
    dev_queries = [build_query_text(c["history"], c["user_query"]) for c in cases]
    dev_embs = model.encode(dev_queries, batch_size=64, show_progress_bar=True,
                             normalize_embeddings=True).astype(np.float32)
    np.save(CACHE_DIR / "dev_query_embeddings.npy", dev_embs)
    print(f"  Saved dev query embeddings: {dev_embs.shape}")

    # Pre-compute R21 retrieval lists for all dev cases
    print(f"{ts()} Computing R21 retrieval for {len(cases)} dev cases...", flush=True)
    r21_lists = []
    for i in range(len(cases)):
        scores = track_embs @ dev_embs[i]
        played_set = set(cases[i]["music_turns"])
        for idx, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, 300)[:300]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        r21_lists.append([all_track_ids[j] for j in top_idx])

    with open(CACHE_DIR / "dev_r21_lists.json", "w") as f:
        json.dump(r21_lists, f)
    print(f"  Saved {len(r21_lists)} retrieval lists")

    # Quick sanity: how many dev GTs in top-200?
    hit200 = sum(1 for i, c in enumerate(cases) if c["gt"] in r21_lists[i][:200])
    print(f"\n  Dev hit@200: {hit200}/{len(cases)} ({hit200/len(cases):.1%})")

    elapsed = time.time() - t0
    print(f"\n{ts()} Production R21 training complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
