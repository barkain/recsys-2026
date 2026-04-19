"""Dense retrieval using sentence-transformers (E5, BGE, etc.) over track metadata."""
import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """Dense retriever using sentence-transformers with FAISS-style matrix search."""

    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 64,
    ) -> None:
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_slug = model_name.replace("/", "_")

        self.metadata_dict = self._load_corpus()
        self.encoder = SentenceTransformer(model_name)

        index_dir = os.path.join(self.cache_dir, "dense", self.model_slug, self.corpus_name)
        emb_path = os.path.join(index_dir, "embeddings.pt")
        ids_path = os.path.join(index_dir, "track_ids.json")

        if os.path.exists(emb_path) and os.path.exists(ids_path):
            self.embeddings = torch.load(emb_path, map_location="cpu")
            with open(ids_path) as f:
                self.track_ids = json.load(f)
        else:
            self.embeddings, self.track_ids = self._build_index(index_dir)

    def _load_corpus(self) -> dict[str, dict]:
        ds = load_dataset(self.dataset_name)
        combined = concatenate_datasets([ds[s] for s in self.split_types])
        return {item["track_id"]: item for item in combined}

    def _stringify(self, metadata: dict) -> str:
        parts = []
        for field in self.corpus_types:
            if field not in metadata:
                continue
            val = metadata[field]
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            parts.append(f"{field}: {val}")
        # E5 models expect "passage: " prefix
        return "passage: " + "; ".join(parts)

    def _build_index(self, index_dir: str):
        track_ids = list(self.metadata_dict.keys())
        corpus = [self._stringify(self.metadata_dict[tid]) for tid in track_ids]
        embeddings = self.encoder.encode(
            corpus,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_tensor=True,
        ).cpu()
        os.makedirs(index_dir, exist_ok=True)
        torch.save(embeddings, os.path.join(index_dir, "embeddings.pt"))
        with open(os.path.join(index_dir, "track_ids.json"), "w") as f:
            json.dump(track_ids, f)
        return embeddings, track_ids

    def _encode_query(self, query: str) -> torch.Tensor:
        return self.encoder.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_tensor=True,
        ).cpu()

    def _encode_queries(self, queries: list[str]) -> torch.Tensor:
        return self.encoder.encode(
            ["query: " + q for q in queries],
            normalize_embeddings=True,
            convert_to_tensor=True,
        ).cpu()

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        q_emb = self._encode_query(query)
        scores = self.embeddings @ q_emb
        indices = torch.topk(scores, k=min(topk, len(self.track_ids))).indices.tolist()
        return [self.track_ids[i] for i in indices]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        q_embs = self._encode_queries(queries)  # [B, D]
        scores = self.embeddings @ q_embs.T  # [N, B]
        k = min(topk, len(self.track_ids))
        results = []
        for i in range(len(queries)):
            indices = torch.topk(scores[:, i], k=k).indices.tolist()
            results.append([self.track_ids[idx] for idx in indices])
        return results

    def scored_retrieval(self, query: str, topk: int) -> list[tuple[str, float]]:
        """Return (track_id, score) pairs for RRF fusion."""
        q_emb = self._encode_query(query)
        scores = self.embeddings @ q_emb
        k = min(topk, len(self.track_ids))
        top = torch.topk(scores, k=k)
        return [(self.track_ids[i], float(s)) for i, s in zip(top.indices.tolist(), top.values.tolist())]
