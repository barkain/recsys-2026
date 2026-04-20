"""Echo's enhanced CRS system for RecSys Challenge 2026."""
import os
from typing import Any

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module


class CRS_SYSTEM:
    """Enhanced Conversational Recommender System.

    Two-stage pipeline:
      1. Retrieval: BM25 | dense | hybrid (BM25 + E5, RRF fusion)
      2. Generation: Claude API | Llama local

    Optional stages:
      3. Query reformulation: LLM extracts music entities before retrieval
      4. Reranking: user-profile embedding similarity
      5. LLM listwise reranking (nDCG@20-optimised)
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
        use_query_reformulation: bool = False,
        query_reformulation_model: str = "claude-haiku-4-5-20251001",
        use_llm_reranker: bool = False,
        llm_reranker_model: str = "claude-haiku-4-5-20251001",
        llm_reranker_window: int | None = None,
    ):
        self.cache_dir = cache_dir
        self.candidate_k = candidate_k

        self.retrieval = load_retrieval_module(
            retrieval_type,
            item_db_name,
            track_split_types,
            corpus_types,
            cache_dir,
            dense_model=dense_model,
        )
        self.item_db = MusicCatalogDB(item_db_name, track_split_types)
        self.user_db = UserProfileDB(user_db_name, user_split_types)
        self.lm = load_lm_module(lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)

        self.reranker = None
        if use_reranker:
            from mcrs.reranking_modules import UserProfileReranker
            self.reranker = UserProfileReranker(cache_dir=cache_dir, alpha=reranker_alpha)

        self.llm_reranker = None
        if use_llm_reranker:
            from mcrs.reranking_modules import LLMListwiseReranker
            self.llm_reranker = LLMListwiseReranker(
                model=llm_reranker_model,
                window_size=llm_reranker_window,
            )

        self.query_reformulator = None
        if use_query_reformulation:
            from mcrs.query_reformulation import QueryReformulator
            self.query_reformulator = QueryReformulator(model=query_reformulation_model)

        prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(f"{prompts_dir}/roleplay.txt", encoding="utf-8").read(),
            "response_generation": open(f"{prompts_dir}/response_generation.txt", encoding="utf-8").read(),
            "personalization": open(f"{prompts_dir}/personalization.txt", encoding="utf-8").read(),
        }

    def _get_system_prompt(self, user_id: str | None = None) -> str:
        prompt = self.role_prompt["role_play"] + "\n" + self.role_prompt["response_generation"]
        if user_id:
            profile_str = self.user_db.id_to_profile_str(user_id)
            if profile_str:
                prompt += "\n" + self.role_prompt["personalization"] + "\n" + profile_str
        return prompt

    def _build_retrieval_query(self, session_memory: list[dict], user_query: str) -> str:
        """Build retrieval query, optionally via LLM reformulation."""
        if self.query_reformulator is not None:
            return self.query_reformulator.reformulate(session_memory, user_query)
        full_memory = list(session_memory) + [{"role": "user", "content": user_query}]
        return "\n".join(f"{m['role']}: {m['content']}" for m in full_memory)

    def _apply_rerankers(
        self,
        candidates: list[str],
        user_id: str | None,
        session_memory: list[dict],
    ) -> list[str]:
        """Apply reranking pipeline: user-profile reranker, then LLM reranker."""
        if self.reranker:
            # Pass more candidates to LLM reranker if it will run next
            k = self.candidate_k if self.llm_reranker else 20
            candidates = self.reranker.rerank(candidates, user_id, topk=k)

        if self.llm_reranker:
            candidates = self.llm_reranker.rerank(candidates, session_memory, self.item_db, topk=20)
        elif not self.reranker:
            candidates = candidates[:20]

        return candidates

    def chat(self, user_query: str, session_memory: list[dict], user_id: str | None = None) -> dict[str, Any]:
        session_memory = list(session_memory)
        retrieval_input = self._build_retrieval_query(session_memory, user_query)
        session_memory.append({"role": "user", "content": user_query})

        candidates = self.retrieval.text_to_item_retrieval(retrieval_input, topk=self.candidate_k)
        candidates = self._apply_rerankers(candidates, user_id, session_memory)

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
        sys_prompts, retrieval_inputs, session_memories, user_ids = [], [], [], []

        for item in batch_data:
            user_query = item["user_query"]
            user_id = item.get("user_id")
            memory = list(item.get("session_memory", []))

            retrieval_inputs.append(self._build_retrieval_query(memory, user_query))
            memory.append({"role": "user", "content": user_query})
            sys_prompts.append(self._get_system_prompt(user_id))
            session_memories.append(memory)
            user_ids.append(user_id)

        if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            batch_candidates = self.retrieval.batch_text_to_item_retrieval(retrieval_inputs, topk=self.candidate_k)
        else:
            batch_candidates = [self.retrieval.text_to_item_retrieval(q, topk=self.candidate_k) for q in retrieval_inputs]

        final_candidates = []
        top_items = []
        for i, candidates in enumerate(batch_candidates):
            ranked = self._apply_rerankers(candidates, user_ids[i], session_memories[i])
            final_candidates.append(ranked)
            top_items.append(self.item_db.id_to_metadata(ranked[0]))

        if hasattr(self.lm, "batch_response_generation"):
            responses = self.lm.batch_response_generation(sys_prompts, session_memories, top_items)
        else:
            responses = [
                self.lm.response_generation(sys_prompts[i], session_memories[i], top_items[i])
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
