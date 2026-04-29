#!/usr/bin/env python3
# ruff: noqa: T201
"""B3.3: Build utterance embedding cache using E5-small-v2 (frozen).

Encodes all user utterances from train + dev conversations.
Output: cache/seq_model/utt_embeddings.npy + utt_embedding_index.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import DownloadConfig, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache" / "seq_model"
UTT_NPY = CACHE_DIR / "utt_embeddings.npy"
UTT_INDEX = CACHE_DIR / "utt_embedding_index.json"
MODEL_NAME = "intfloat/e5-small-v2"


def main():
    if UTT_NPY.exists() and UTT_INDEX.exists():
        idx = json.load(open(UTT_INDEX))
        print(f"Cache already exists: {len(idx)} utterances in {UTT_NPY}")
        return

    t0 = time.time()
    print(f"Loading sentence encoder: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL_NAME)
    print(f"  dim={encoder.get_sentence_embedding_dimension()}")

    print("Loading train + dev datasets...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))

    utterances = []  # (session_id, turn_number, text)
    index = {}       # "session_id:turn_number" -> row index

    for split in ["train", "test"]:
        for item in ds[split]:
            sid = str(item["session_id"])
            for conv in item["conversations"]:
                if conv["role"] == "user":
                    turn = int(conv["turn_number"])
                    text = str(conv["content"])
                    key = f"{sid}:{turn}"
                    if key not in index:
                        index[key] = len(utterances)
                        utterances.append(text)

    print(f"  {len(utterances)} unique utterances from {split} splits")

    # Also load blind-A utterances for inference
    try:
        blind = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A",
                             download_config=DownloadConfig(local_files_only=True), split="test")
        for item in blind:
            sid = str(item["session_id"])
            for conv in item["conversations"]:
                if conv["role"] == "user":
                    turn = int(conv["turn_number"])
                    text = str(conv["content"])
                    key = f"{sid}:{turn}"
                    if key not in index:
                        index[key] = len(utterances)
                        utterances.append(text)
        print(f"  {len(utterances)} total utterances (including blind-A)")
    except Exception as e:
        print(f"  Warning: could not load blind-A: {e}")

    # Encode in batches
    print(f"Encoding {len(utterances)} utterances...")
    # E5-small expects "query: " prefix for retrieval
    prefixed = [f"query: {text}" for text in utterances]
    embeddings = encoder.encode(prefixed, batch_size=128, show_progress_bar=True,
                                normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(UTT_NPY, embeddings)
    with open(UTT_INDEX, "w") as f:
        json.dump(index, f)

    elapsed = time.time() - t0
    print(f"Saved to {UTT_NPY} ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"Index: {UTT_INDEX} ({len(index)} entries)")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
