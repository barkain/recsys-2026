"""BM25 retrieval over enriched track metadata (name + artist + album + tags)."""
import os
import json
import bm25s
from datasets import load_dataset, concatenate_datasets


class BM25Retriever:
    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
    ) -> None:
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.metadata_dict = self._load_corpus()
        index_path = os.path.join(self.cache_dir, "bm25", self.corpus_name)
        if os.path.exists(index_path):
            self.bm25_model, self.track_ids = self._load_index(index_path)
        else:
            self.bm25_model, self.track_ids = self._build_and_save(index_path)

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
                val = " ".join(str(v) for v in val)
            parts.append(str(val))
        return " ".join(parts)

    def _build_and_save(self, index_path: str):
        track_ids = list(self.metadata_dict.keys())
        corpus = [self._stringify(self.metadata_dict[tid]) for tid in track_ids]
        corpus_tokens = bm25s.tokenize(corpus)
        model = bm25s.BM25()
        model.index(corpus_tokens)
        os.makedirs(index_path, exist_ok=True)
        model.save(index_path, corpus=corpus)
        with open(os.path.join(index_path, "track_ids.json"), "w") as f:
            json.dump(track_ids, f)
        return model, track_ids

    def _load_index(self, index_path: str):
        model = bm25s.BM25.load(index_path, load_corpus=True)
        with open(os.path.join(index_path, "track_ids.json")) as f:
            track_ids = json.load(f)
        return model, track_ids

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        tokens = bm25s.tokenize([query.lower()])
        results = self.bm25_model.retrieve(tokens, k=topk, return_as="tuple")
        return [self.track_ids[item["id"]] for item in results.documents[0]]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        tokens = bm25s.tokenize([q.lower() for q in queries])
        results = self.bm25_model.retrieve(tokens, k=topk, return_as="tuple")
        return [
            [self.track_ids[item["id"]] for item in results.documents[i]]
            for i in range(len(queries))
        ]

    def scored_retrieval(self, query: str, topk: int) -> list[tuple[str, float]]:
        """Return (track_id, score) pairs for RRF fusion."""
        tokens = bm25s.tokenize([query.lower()])
        results = self.bm25_model.retrieve(tokens, k=topk, return_as="tuple")
        return [
            (self.track_ids[item["id"]], float(results.scores[0][idx]))
            for idx, item in enumerate(results.documents[0])
        ]
