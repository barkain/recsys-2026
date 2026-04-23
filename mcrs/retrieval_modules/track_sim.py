"""Track-to-track similarity retrieval using pre-computed embeddings.

Looks up the embedding of a known track_id (no query encoder needed) and
finds the K nearest neighbour tracks via FAISS cosine similarity.

Requires only the pre-computed embedding dataset — no model download.
"""
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class TrackSimilarityRetriever:
    """Retrieves tracks similar to a given track_id using pre-computed embeddings.

    Loads track vectors from HuggingFace once, caches them locally, then
    performs fast FAISS nearest-neighbour lookup.  No query encoder needed —
    we look up the track's own embedding directly from the index.

    Args:
        track_embed_dataset: HuggingFace dataset with pre-computed embeddings.
        embed_column: Which embedding column to use.
        split_types: Dataset splits to load.
        cache_dir: Local directory for caching vectors + FAISS index.
    """

    def __init__(
        self,
        track_embed_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        embed_column: str = "metadata-qwen3_embedding_0.6b",
        split_types: list[str] | None = None,
        cache_dir: str = "./cache",
    ):
        self.embed_column = embed_column
        split_types = split_types or ["all_tracks"]

        index_cache = os.path.join(cache_dir, "track_sim", embed_column.replace("/", "_"))
        ids_path = os.path.join(index_cache, "track_ids.json")
        vecs_path = os.path.join(index_cache, "vectors.npy")

        if os.path.exists(ids_path) and os.path.exists(vecs_path):
            logger.info("Loading track-sim index from %s", index_cache)
            with open(ids_path) as f:
                self.track_ids: list[str] = json.load(f)
            self.vectors = np.load(vecs_path)
        else:
            logger.info("Building track-sim index from %s [%s]", track_embed_dataset, embed_column)
            from datasets import concatenate_datasets, load_dataset
            ds = load_dataset(track_embed_dataset)
            combined = concatenate_datasets([ds[s] for s in split_types if s in ds])
            self.track_ids = [str(t) for t in combined["track_id"]]
            self.vectors = np.array(combined[embed_column], dtype=np.float32)
            # L2-normalise for cosine similarity via inner product
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self.vectors /= norms
            os.makedirs(index_cache, exist_ok=True)
            np.save(vecs_path, self.vectors)
            with open(ids_path, "w") as f:
                json.dump(self.track_ids, f)

        # Map track_id → row index for O(1) lookup
        self._id_to_idx: dict[str, int] = {tid: i for i, tid in enumerate(self.track_ids)}

        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        import faiss  # noqa: PLC0415
        dim = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.vectors)
        logger.info(
            "TrackSimilarityRetriever: FAISS index built — %d tracks, dim=%d",
            len(self.track_ids),
            dim,
        )

    def track_id_to_neighbors(self, track_id: str, topk: int = 20) -> list[str]:
        """Return up to *topk* track_ids most similar to *track_id*.

        If *track_id* is not in the index, returns an empty list.
        The query track itself is excluded from results.
        """
        idx = self._id_to_idx.get(str(track_id))
        if idx is None:
            logger.debug("TrackSimilarityRetriever: track_id %r not in index", track_id)
            return []
        query_vec = self.vectors[idx : idx + 1]  # shape (1, dim)
        # Fetch topk+1 to exclude the query track itself
        _, idxs = self.index.search(query_vec, topk + 1)
        return [
            self.track_ids[i]
            for i in idxs[0]
            if i >= 0 and i != idx
        ][:topk]

    def batch_track_id_to_neighbors(
        self, track_ids: list[str], topk: int = 20
    ) -> list[list[str]]:
        """Batch version of *track_id_to_neighbors*."""
        results = []
        for tid in track_ids:
            results.append(self.track_id_to_neighbors(tid, topk=topk))
        return results
