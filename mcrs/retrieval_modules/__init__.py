from mcrs.retrieval_modules.bm25 import BM25Retriever
from mcrs.retrieval_modules.dense import DenseRetriever
from mcrs.retrieval_modules.hybrid import HybridRetriever
from mcrs.retrieval_modules.multi_query import MultiQueryRetriever
from mcrs.retrieval_modules.precomputed import PrecomputedEmbeddingRetriever


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
    elif retrieval_type == "hybrid":
        dense_model = kwargs.get("dense_model", "intfloat/e5-base-v2")
        bm25_weight = kwargs.get("bm25_weight", 0.5)
        candidate_k = kwargs.get("candidate_k", 100)
        return HybridRetriever(
            dataset_name, split_types, corpus_types, cache_dir,
            dense_model=dense_model, bm25_weight=bm25_weight,
            candidate_k=candidate_k,
        )
    elif retrieval_type == "precomputed":
        embed_col = kwargs.get("embed_column", "metadata-qwen3_embedding_0.6b")
        embed_ds = kwargs.get("embed_dataset", "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
        return PrecomputedEmbeddingRetriever(
            track_embed_dataset=embed_ds,
            embed_column=embed_col,
            split_types=split_types,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown retrieval_type: {retrieval_type}")


__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "MultiQueryRetriever",
    "PrecomputedEmbeddingRetriever",
    "load_retrieval_module",
]
