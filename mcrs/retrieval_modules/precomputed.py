"""Dense retrieval using pre-computed track embeddings from the challenge dataset.

Uses talkpl-ai/TalkPlayData-Challenge-Track-Embeddings (no local model needed).
Available embedding columns:
  - metadata-qwen3_embedding_0.6b  (1024-dim, best for text-query matching)
  - attributes-qwen3_embedding_0.6b (1024-dim)
  - lyrics-qwen3_embedding_0.6b    (1024-dim)
  - audio-laion_clap               (512-dim)
  - image-siglip2                  (768-dim)
  - cf-bpr                         (128-dim)
"""
import os
import json
import logging
import numpy as np
from datasets import load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)

# Query embedding uses sentence-transformers matching the stored embedding model
_QUERY_MODEL_FOR_COLUMN = {
    "metadata-qwen3_embedding_0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "attributes-qwen3_embedding_0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "lyrics-qwen3_embedding_0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


class PrecomputedEmbeddingRetriever:
    """Nearest-neighbour retrieval over pre-computed challenge track embeddings.

    Loads track vectors from HuggingFace (one-time, cached), builds a FAISS
    index, then embeds each query with a small local model for retrieval.
    """

    def __init__(
        self,
        track_embed_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        embed_column: str = "metadata-qwen3_embedding_0.6b",
        split_types: list[str] | None = None,
        cache_dir: str = "./cache",
        query_model: str | None = None,
    ):
        self.embed_column = embed_column
        self.cache_dir = cache_dir
        split_types = split_types or ["all_tracks"]

        index_cache = os.path.join(cache_dir, "precomputed", embed_column.replace("/", "_"))
        ids_path = os.path.join(index_cache, "track_ids.json")
        vecs_path = os.path.join(index_cache, "vectors.npy")

        if os.path.exists(ids_path) and os.path.exists(vecs_path):
            logger.info("Loading precomputed index from %s", index_cache)
            with open(ids_path) as f:
                self.track_ids = json.load(f)
            self.vectors = np.load(vecs_path)
        else:
            logger.info("Building precomputed index from %s [%s]", track_embed_dataset, embed_column)
            ds = load_dataset(track_embed_dataset)
            combined = concatenate_datasets([ds[s] for s in split_types if s in ds])
            self.track_ids = combined["track_id"]
            self.vectors = np.array(combined[embed_column], dtype=np.float32)
            # L2-normalise for cosine similarity via inner product
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self.vectors /= norms
            os.makedirs(index_cache, exist_ok=True)
            np.save(vecs_path, self.vectors)
            with open(ids_path, "w") as f:
                json.dump(list(self.track_ids), f)

        self._build_faiss_index()

        # Query encoder
        query_model_name = query_model or _QUERY_MODEL_FOR_COLUMN.get(
            embed_column, "intfloat/e5-base-v2"
        )
        logger.info("Loading query encoder: %s", query_model_name)
        from sentence_transformers import SentenceTransformer
        self.query_encoder = SentenceTransformer(query_model_name)

    def _build_faiss_index(self):
        import faiss
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalised vecs
        self.index.add(self.vectors)
        logger.info("FAISS index built: %d tracks, dim=%d", len(self.track_ids), dim)

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.query_encoder.encode([query], normalize_embeddings=True)
        return vec.astype(np.float32)

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        vec = self._embed_query(query)
        _, idxs = self.index.search(vec, topk)
        return [self.track_ids[i] for i in idxs[0] if i >= 0]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        vecs = self.query_encoder.encode(queries, normalize_embeddings=True).astype(np.float32)
        _, all_idxs = self.index.search(vecs, topk)
        return [
            [self.track_ids[i] for i in row if i >= 0]
            for row in all_idxs
        ]

    def scored_retrieval(self, query: str, topk: int) -> list[tuple[str, float]]:
        vec = self._embed_query(query)
        scores, idxs = self.index.search(vec, topk)
        return [
            (self.track_ids[i], float(scores[0][j]))
            for j, i in enumerate(idxs[0]) if i >= 0
        ]
