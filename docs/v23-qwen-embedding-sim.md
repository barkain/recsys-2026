# v23: Qwen3 Track Embedding Similarity

## What this adds

v23 extends v22 (generative retrieval) with a third retrieval signal: **track-to-track embedding similarity**. Given the session's last recommended track, find the 30 most similar tracks in the full catalog using pre-computed Qwen3 embeddings, and merge them into the RRF candidate pool.

**Why**: BM25 retrieval and generative retrieval both work from the conversation text. This adds a complementary signal from the *music itself* — tracks that sound/feel similar to what the user just listened to, without needing the conversation to describe them.

---

## How it works (end-to-end)

```
Session memory: [..., assistant recommended "track_id=abc123", ...]
                              ↓
         _last_played_track_id(session_memory) → "abc123"
                              ↓
         FAISS lookup: vectors["abc123"] → top-30 nearest neighbors
                              ↓
         RRF merge with [dual-QR candidates, generative candidates, embedding neighbors]
                              ↓
         Sonnet reranker (window=50) → top 20
```

### Retrieval sources in v23 (all RRF-merged before reranking)

| Source | Count | Method |
|--------|-------|--------|
| NLQ query → BM25 | 100 | Haiku reformulates conversation to NLQ |
| Entity query → BM25 | 100 | Haiku extracts artist/genre/mood entities |
| Generative suggestions → BM25 | 25 × 3 = 75 | Haiku names 25 specific tracks; BM25 for each |
| Embedding neighbors | 30 | FAISS k-NN on last-played track's Qwen3 vector |

Total unique candidates before reranking: typically 150–250 (after RRF dedup).

---

## The embedding dataset

**HuggingFace**: `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`

Available columns (each row = one track):
- `track_id` — matches the track IDs in the main catalog
- `metadata-qwen3_embedding_0.6b` — 1024-dim float32, Qwen3-Embedding-0.6B on track metadata text
- `attributes-qwen3_embedding_0.6b` — 1024-dim, same model on attribute text
- `lyrics-qwen3_embedding_0.6b` — 1024-dim, same model on lyrics
- `audio-laion_clap` — 512-dim, LAION-CLAP audio embedding
- `image-siglip2` — 768-dim, SigLIP2 cover image embedding
- `cf-bpr` — 128-dim, collaborative filtering (BPR)

**v23 uses**: `metadata-qwen3_embedding_0.6b` — best for text-based recommendation context.

**Dataset size estimate**: ~50k–200k tracks × 1024 dims × 4 bytes = 200–800 MB for just the metadata embedding column. Plan accordingly.

---

## Why v23 failed (and how to fix it)

### The problem

`TrackSimilarityRetriever.__init__()` calls `load_dataset()` unconditionally at startup:

```python
# mcrs/retrieval_modules/track_sim.py, line 55
ds = load_dataset(track_embed_dataset)  # ← downloads 400MB+ at inference startup
```

This download is too large to complete within the inference container's timeout. The process dies silently before writing any output.

### The fix: pre-cache the embeddings once

The class already has cache logic built in:

```python
if os.path.exists(ids_path) and os.path.exists(vecs_path):
    # → fast path: load from disk (milliseconds)
else:
    # → slow path: download from HF, save to disk (minutes)
```

**You just need to run the slow path once**, deliberately, and save the result to:
```
recsys-work/cache/track_sim/metadata-qwen3_embedding_0.6b/
├── track_ids.json    (~2–5 MB, list of track_id strings)
└── vectors.npy       (~200–800 MB, float32 array shape [N, 1024])
```

After that, every v23 inference run hits the fast path.

---

## Implementation: cache-build script

Create `scripts/build_embedding_cache.py`:

```python
"""One-time script to pre-cache Qwen3 track embeddings from HuggingFace.

Run this ONCE before v23 inference:
    uv run python3 scripts/build_embedding_cache.py

After it completes, v23 inference will load from disk (<1 sec startup).
"""
import json
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets

DATASET = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings"
COLUMN = "metadata-qwen3_embedding_0.6b"
SPLIT_TYPES = ["all_tracks"]
CACHE_DIR = "./cache"

def main():
    out_dir = os.path.join(CACHE_DIR, "track_sim", COLUMN.replace("/", "_"))
    ids_path = os.path.join(out_dir, "track_ids.json")
    vecs_path = os.path.join(out_dir, "vectors.npy")

    if os.path.exists(ids_path) and os.path.exists(vecs_path):
        print(f"Cache already exists at {out_dir}. Delete it to rebuild.")
        return

    print(f"Loading {DATASET} [{COLUMN}] from HuggingFace...")
    ds = load_dataset(DATASET)
    available = [s for s in SPLIT_TYPES if s in ds]
    if not available:
        available = list(ds.keys())
        print(f"Splits {SPLIT_TYPES} not found, using: {available}")
    combined = concatenate_datasets([ds[s] for s in available])
    print(f"Loaded {len(combined)} tracks")

    track_ids = [str(t) for t in combined["track_id"]]
    vectors = np.array(combined[COLUMN], dtype=np.float32)
    print(f"Vectors shape: {vectors.shape}")

    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vectors /= norms

    os.makedirs(out_dir, exist_ok=True)
    np.save(vecs_path, vectors)
    with open(ids_path, "w") as f:
        json.dump(track_ids, f)

    print(f"Saved {len(track_ids)} track IDs to {ids_path}")
    print(f"Saved vectors ({vectors.nbytes / 1e6:.1f} MB) to {vecs_path}")
    print("Cache build complete. v23 inference is now ready.")

if __name__ == "__main__":
    main()
```

Run once:
```bash
cd recsys-work
uv run python3 scripts/build_embedding_cache.py
```

Expected output:
```
Loading talkpl-ai/TalkPlayData-Challenge-Track-Embeddings [metadata-qwen3_embedding_0.6b]...
Loaded 83412 tracks
Vectors shape: (83412, 1024)
Saved 83412 track IDs to ./cache/track_sim/metadata-qwen3_embedding_0.6b/track_ids.json
Saved vectors (341.3 MB) to ./cache/track_sim/metadata-qwen3_embedding_0.6b/vectors.npy
Cache build complete. v23 inference is now ready.
```

---

## What's already implemented

All code changes are committed to `v19-nlq-hint-wider-window` (commit `48d238d`):

### `mcrs/retrieval_modules/track_sim.py` (new file)

```python
class TrackSimilarityRetriever:
    def __init__(self, track_embed_dataset, embed_column, split_types, cache_dir):
        # Loads from cache if exists, else downloads from HF
        # Builds FAISS IndexFlatIP (cosine on L2-normalised vectors)

    def track_id_to_neighbors(self, track_id: str, topk: int = 20) -> list[str]:
        # O(1) lookup by track_id → FAISS search → list of similar track_ids
        # Returns [] if track_id not in index
```

### `mcrs/crs_system.py` (additions)

New constructor params:
```python
use_track_embedding_sim: bool = False
track_sim_embed_column: str = "metadata-qwen3_embedding_0.6b"
track_sim_topk: int = 30
```

New helper:
```python
def _last_played_track_id(self, session_memory) -> str | None:
    # Scans reversed session_memory for role="music"
    # Returns track_id string or None if no prior recommendation
```

Integration in `_retrieve()` (dual-QR path):
```python
if self.track_sim_retriever:
    last_tid = self._last_played_track_id(session_memory)
    if last_tid:
        embed_sim = self.track_sim_retriever.track_id_to_neighbors(
            last_tid, topk=self.track_sim_topk
        )
        if embed_sim:
            lists.append(embed_sim)
# embed_sim is RRF-merged with all other candidate lists
```

Same integration in `_batch_retrieve()`.

### `config/echo_v23_devset.yaml`

```yaml
use_generative_retrieval: true
generative_retrieval_model: "claude-haiku-4-5-20251001"
use_track_embedding_sim: true
track_sim_embed_column: "metadata-qwen3_embedding_0.6b"
track_sim_topk: 30
# ... (rest same as v22)
```

---

## Steps to run v23 inference

1. **Build cache** (one-time, ~10 min, requires internet):
   ```bash
   cd recsys-work
   uv run python3 scripts/build_embedding_cache.py
   ```

2. **Verify cache exists**:
   ```bash
   ls -lh cache/track_sim/metadata-qwen3_embedding_0.6b/
   # Should show track_ids.json (~few MB) and vectors.npy (~300-800MB)
   ```

3. **Run inference**:
   ```bash
   export ANTHROPIC_RECSYS_API_KEY=...
   uv run python3 run_inference_devset.py --tid echo_v23_devset --batch_size 8 --last_turn_only
   uv run python3 eval_devset.py --tid echo_v23_devset --last_turn_only
   cat exp/eval/echo_v23_devset_last_turn.json
   ```

4. **Update VERSIONS.md** with the result and commit.

---

## Expected behaviour and hypotheses

**Sessions with a prior recommendation** (~80% of last-turn sessions): the embedding neighbors add a semantically grounded candidate pool. If the ground truth track is similar to what was previously recommended, FAISS will surface it.

**Sessions without prior recommendation** (first-turn or cold-start): `_last_played_track_id()` returns None, embedding sim is skipped, falls back to v22 behaviour.

**Key question**: does track-to-track similarity actually help nDCG? The hypothesis is yes for sequential listening patterns (user is exploring a genre/artist). If the miss rate drops by even 5–10 sessions out of 100, nDCG improves by ~0.01–0.02.

**Risk**: embedding neighbors may add noise if the last recommendation was unrelated to the current request (e.g., session pivots to a different genre). The RRF merge weight is self-regulating — irrelevant tracks score low across other lists and don't make the top-20.

---

## Potential variations to test after v23

- `track_sim_embed_column: "audio-laion_clap"` — audio-based similarity (512-dim); might generalise better across genres
- `track_sim_topk: 50` — wider pool if 30 is too narrow
- `track_sim_topk: 15` — tighter pool if 30 adds noise
- Multi-track: average embeddings of last 3 played tracks instead of just the last one
