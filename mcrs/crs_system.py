"""Echo's enhanced CRS system for RecSys Challenge 2026."""
import os
from typing import Any

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module, MultiQueryRetriever
from mcrs.retrieval_modules.hybrid import HybridRetriever


class CRS_SYSTEM:
    """Enhanced Conversational Recommender System.

    Two-stage pipeline:
      1. Retrieval: BM25 | dense | hybrid (BM25 + E5, RRF fusion) | multi_query
      2. Generation: Claude API | Llama local

    Optional stages:
      3. Reranking: user-profile embedding similarity
      4. LLM listwise reranking (nDCG@20-optimised)

    Optional pre-retrieval:
      - Query reformulation via Claude (entity extraction)
      - Multi-query expansion (N diverse queries, RRF-fused)
    """

    def __init__(
        self,
        lm_type: str = "claude-haiku-4-5-20251001",
        retrieval_type: str = "hybrid",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"],
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name", "tag_list"],
        cache_dir: str = "./cache",
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype=None,
        dense_model: str = "intfloat/e5-base-v2",
        use_reranker: bool = False,
        reranker_alpha: float = 0.3,
        candidate_k: int = 50,
        # Query reformulation
        use_query_reformulation: bool = False,
        query_reformulation_model: str = "claude-haiku-4-5-20251001",
        # LLM listwise reranker
        use_llm_reranker: bool = False,
        llm_reranker_model: str = "claude-haiku-4-5-20251001",
        llm_reranker_window: int = 50,
        # Multi-query retrieval
        n_queries: int = 3,
        per_query_k: int | None = None,
        # Precomputed embedding retrieval
        embed_column: str = "metadata-qwen3_embedding_0.6b",
        embed_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        # Ignored pass-through params (e.g. test_dataset_name from run script)
        **_ignored,
    ):
        self.cache_dir = cache_dir
        self.candidate_k = candidate_k
        self.use_query_reformulation = use_query_reformulation
        self.retrieval_type = retrieval_type

        # Build base retriever
        base_retriever = load_retrieval_module(
            "hybrid" if retrieval_type == "multi_query" else retrieval_type,
            item_db_name,
            track_split_types,
            corpus_types,
            cache_dir,
            dense_model=dense_model,
            candidate_k=candidate_k,
            embed_column=embed_column,
            embed_dataset=embed_dataset,
        )

        # Wrap with multi-query if requested
        if retrieval_type == "multi_query":
            self.retrieval = MultiQueryRetriever(
                base_retriever=base_retriever,
                model=query_reformulation_model,
                n_queries=n_queries,
                per_query_k=per_query_k or candidate_k,
            )
        else:
            self.retrieval = base_retriever

        self.item_db = MusicCatalogDB(item_db_name, track_split_types)
        self.user_db = UserProfileDB(user_db_name, user_split_types)
        self.lm = load_lm_module(
            lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype
        )

        # Optional query reformulator (entity extraction → enriched query)
        self.query_reformulator = None
        if use_query_reformulation and retrieval_type != "multi_query":
            from mcrs.query_reformulation import QueryReformulator
            self.query_reformulator = QueryReformulator(model=query_reformulation_model)

        # Optional user-profile reranker
        self.reranker = None
        if use_reranker:
            from mcrs.reranking_modules import UserProfileReranker
            self.reranker = UserProfileReranker(cache_dir=cache_dir, alpha=reranker_alpha)

        # Optional LLM listwise reranker
        self.llm_reranker = None
        if use_llm_reranker:
            from mcrs.reranking_modules.llm_reranker import LLMListwiseReranker
            self.llm_reranker = LLMListwiseReranker(
                model=llm_reranker_model,
                topk=20,
                window_size=llm_reranker_window if llm_reranker_window else None,
            )

        prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(f"{prompts_dir}/roleplay.txt", encoding="utf-8").read(),
            "response_generation": open(
                f"{prompts_dir}/response_generation.txt", encoding="utf-8"
            ).read(),
            "personalization": open(
                f"{prompts_dir}/personalization.txt", encoding="utf-8"
            ).read(),
        }

    def _get_system_prompt(self, user_id: str | None = None) -> str:
        prompt = self.role_prompt["role_play"] + "\n" + self.role_prompt["response_generation"]
        if user_id:
            profile_str = self.user_db.id_to_profile_str(user_id)
            if profile_str:
                prompt += "\n" + self.role_prompt["personalization"] + "\n" + profile_str
        return prompt

    def _retrieve(self, session_memory: list[dict], user_query: str) -> list[str]:
        """Run retrieval, using multi-query or reformulation if configured."""
        if self.retrieval_type == "multi_query":
            # MultiQueryRetriever needs session_memory + user_query directly
            return self.retrieval.retrieve(
                session_memory, user_query, topk=self.candidate_k
            )

        if self.query_reformulator:
            query = self.query_reformulator.reformulate(session_memory, user_query)
        else:
            query = "\n".join(f"{m['role']}: {m['content']}" for m in session_memory)
            if user_query:
                query += f"\nUser: {user_query}"

        return self.retrieval.text_to_item_retrieval(query, topk=self.candidate_k)

    def _batch_retrieve(
        self,
        session_memories: list[list[dict]],
        user_queries: list[str],
    ) -> list[list[str]]:
        """Batch retrieval with optional multi-query or reformulation."""
        if self.retrieval_type == "multi_query":
            return self.retrieval.batch_retrieve(
                list(zip(session_memories, user_queries)), topk=self.candidate_k
            )

        if self.query_reformulator:
            queries = self.query_reformulator.batch_reformulate(
                list(zip(session_memories, user_queries))
            )
        else:
            queries = []
            for mem, uq in zip(session_memories, user_queries):
                q = "\n".join(f"{m['role']}: {m['content']}" for m in mem)
                if uq:
                    q += f"\nUser: {uq}"
                queries.append(q)

        if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            return self.retrieval.batch_text_to_item_retrieval(queries, topk=self.candidate_k)
        return [self.retrieval.text_to_item_retrieval(q, topk=self.candidate_k) for q in queries]

    def _rerank(
        self,
        candidates: list[str],
        user_id: str | None,
        session_memory: list[dict],
    ) -> list[str]:
        """Apply reranking stages in order: user-profile → LLM listwise."""
        if self.reranker:
            candidates = self.reranker.rerank(candidates, user_id, topk=len(candidates))
        if self.llm_reranker:
            candidates = self.llm_reranker.rerank(
                candidates, session_memory, self.item_db, topk=20
            )
        return candidates[:20]

    def chat(
        self,
        user_query: str,
        session_memory: list[dict],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        session_memory = list(session_memory)
        session_memory.append({"role": "user", "content": user_query})

        candidates = self._retrieve(session_memory[:-1], user_query)
        candidates = self._rerank(candidates, user_id, session_memory)

        sys_prompt = self._get_system_prompt(user_id)
        top_item = self.item_db.id_to_metadata(candidates[0])
        response = self.lm.response_generation(sys_prompt, session_memory, top_item)

        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": candidates,
            "recommend_item": top_item,
            "response": response,
        }

    def batch_chat(self, batch_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch of CRS turns."""
        session_memories_pre = []  # before appending current user query
        user_queries = []
        session_memories_full = []  # includes current user query
        user_ids = []

        for item in batch_data:
            user_query = item["user_query"]
            user_id = item.get("user_id")
            memory = list(item.get("session_memory", []))

            session_memories_pre.append(memory)
            user_queries.append(user_query)
            session_memories_full.append(memory + [{"role": "user", "content": user_query}])
            user_ids.append(user_id)

        batch_candidates = self._batch_retrieve(session_memories_pre, user_queries)

        # Parallelize reranking — LLM reranker calls are independent HTTP requests
        # and dominate latency; running them concurrently gives ~8x speedup.
        from concurrent.futures import ThreadPoolExecutor as _TPE
        def _rerank_one(args):
            i, candidates = args
            return self._rerank(candidates, user_ids[i], session_memories_full[i])

        with _TPE(max_workers=8) as pool:
            final_candidates = list(pool.map(_rerank_one, enumerate(batch_candidates)))

        top_items = [self.item_db.id_to_metadata(ranked[0]) for ranked in final_candidates]

        sys_prompts = [self._get_system_prompt(uid) for uid in user_ids]

        if hasattr(self.lm, "batch_response_generation"):
            responses = self.lm.batch_response_generation(
                sys_prompts, session_memories_full, top_items
            )
        else:
            responses = [
                self.lm.response_generation(
                    sys_prompts[i], session_memories_full[i], top_items[i]
                )
                for i in range(len(batch_data))
            ]

        return [
            {
                "user_id": user_ids[i],
                "user_query": batch_data[i]["user_query"],
                "retrieval_items": final_candidates[i],
                "recommend_item": top_items[i],
                "response": responses[i],
            }
            for i in range(len(batch_data))
        ]
