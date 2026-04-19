"""Hybrid retrieval: BM25 + dense with Reciprocal Rank Fusion (RRF)."""
from mcrs.retrieval_modules.bm25 import BM25Retriever
from mcrs.retrieval_modules.dense import DenseRetriever


def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: Each list is a ranked sequence of track_ids (best first).
        k: RRF constant (default 60, from the original paper).

    Returns:
        Merged list of track_ids sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, track_id in enumerate(ranked):
            scores[track_id] = scores.get(track_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


class HybridRetriever:
    """Two-stage hybrid retriever: BM25 + dense, fused via RRF."""

    def __init__(
        self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        dense_model: str = "intfloat/e5-base-v2",
        bm25_weight: float = 0.5,
        candidate_k: int = 100,
    ) -> None:
        self.candidate_k = candidate_k
        self.bm25 = BM25Retriever(dataset_name, split_types, corpus_types, cache_dir)
        self.dense = DenseRetriever(
            dataset_name, split_types, corpus_types, cache_dir, model_name=dense_model
        )

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        bm25_ranked = self.bm25.text_to_item_retrieval(query, topk=self.candidate_k)
        dense_ranked = self.dense.text_to_item_retrieval(query, topk=self.candidate_k)
        fused = rrf_fusion([bm25_ranked, dense_ranked])
        return fused[:topk]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        bm25_results = self.bm25.batch_text_to_item_retrieval(queries, topk=self.candidate_k)
        dense_results = self.dense.batch_text_to_item_retrieval(queries, topk=self.candidate_k)
        return [
            rrf_fusion([bm25_results[i], dense_results[i]])[:topk]
            for i in range(len(queries))
        ]
