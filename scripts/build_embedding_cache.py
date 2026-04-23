"""One-time script to pre-cache Qwen3 track embeddings from HuggingFace.

Run this ONCE before v23 inference:
    uv run python3 scripts/build_embedding_cache.py

After it completes, v23 inference will load from disk in <1 second.

Optional args:
    --column   Embedding column to cache (default: metadata-qwen3_embedding_0.6b)
    --cache    Cache directory (default: ./cache)
"""
import argparse
import json
import os

import numpy as np


def main(column: str, cache_dir: str) -> None:
    dataset_name = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
    split_types = ["all_tracks"]

    out_dir = os.path.join(cache_dir, "track_sim", column.replace("/", "_"))
    ids_path = os.path.join(out_dir, "track_ids.json")
    vecs_path = os.path.join(out_dir, "vectors.npy")

    if os.path.exists(ids_path) and os.path.exists(vecs_path):
        import numpy as _np
        existing = _np.load(vecs_path, mmap_mode="r")
        print(f"Cache already exists at {out_dir}")
        print(f"  tracks: {existing.shape[0]}, dim: {existing.shape[1]}")
        print("Delete it to rebuild.")
        return

    print(f"Loading {dataset_name} [{column}] ...")
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(dataset_name)
    available = [s for s in split_types if s in ds]
    if not available:
        available = list(ds.keys())
        print(f"  Note: splits {split_types} not found, using: {available}")

    combined = concatenate_datasets([ds[s] for s in available])
    print(f"  Loaded {len(combined):,} tracks")

    track_ids = [str(t) for t in combined["track_id"]]
    vectors = np.array(combined[column], dtype=np.float32)
    print(f"  Vectors shape: {vectors.shape}")

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vectors /= norms

    os.makedirs(out_dir, exist_ok=True)
    np.save(vecs_path, vectors)
    with open(ids_path, "w") as f:
        json.dump(track_ids, f)

    size_mb = os.path.getsize(vecs_path) / 1e6
    print(f"Saved {len(track_ids):,} track IDs → {ids_path}")
    print(f"Saved vectors ({size_mb:.1f} MB) → {vecs_path}")
    print("Done. v23 inference is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--column", default="metadata-qwen3_embedding_0.6b")
    parser.add_argument("--cache", default="./cache")
    args = parser.parse_args()
    main(args.column, args.cache)
