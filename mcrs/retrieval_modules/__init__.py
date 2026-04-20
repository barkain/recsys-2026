from mcrs.retrieval_modules.bm25 import BM25Retriever
from mcrs.retrieval_modules.dense import DenseRetriever
from mcrs.retrieval_modules.hybrid import HybridRetriever
from mcrs.retrieval_modules.multi_query import MultiQueryRetriever


def load_retrieval_module(
    retrieval_type: str,
    dataset_name: str,
    split_types: list[str],
    corpus_types: list[str],
    cache_dir: str = "./cache",
    **kwargs,
):
    if retrieval_type == "bm25":
        return BM25Retriever(dataset_name, split_types, corpus_types, cache_dir)
    elif retrieval_type == "dense":
        model_name = kwargs.get("dense_model", "intfloat/e5-base-v2")
        return DenseRetriever(dataset_name, split_types, corpus_types, cache_dir, model_name=model_name)
    elif retrieval_type in ("hybrid", "multi_query"):
        dense_model = kwargs.get("dense_model", "intfloat/e5-base-v2")
        bm25_weight = kwargs.get("bm25_weight", 0.5)
        candidate_k = kwargs.get("candidate_k", 100)
        base = HybridRetriever(
            dataset_name, split_types, corpus_types, cache_dir,
            dense_model=dense_model, bm25_weight=bm25_weight,
            candidate_k=candidate_k,
        )
        if retrieval_type == "multi_query":
            model = kwargs.get("query_reformulation_model", "claude-haiku-4-5-20251001")
            n_queries = kwargs.get("n_queries", 3)
            per_query_k = kwargs.get("per_query_k", candidate_k)
            return MultiQueryRetriever(
                base_retriever=base,
                model=model,
                n_queries=n_queries,
                per_query_k=per_query_k,
            )
        return base
    else:
        raise ValueError(f"Unknown retrieval_type: {retrieval_type}")


__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "load_retrieval_module",
]
