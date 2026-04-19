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

    Optional third stage:
      3. Reranking: user-profile embedding similarity
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

    def chat(self, user_query: str, session_memory: list[dict], user_id: str | None = None) -> dict[str, Any]:
        session_memory = list(session_memory)
        session_memory.append({"role": "user", "content": user_query})

        retrieval_input = "\n".join(f"{m['role']}: {m['content']}" for m in session_memory)
        candidates = self.retrieval.text_to_item_retrieval(retrieval_input, topk=self.candidate_k)

        if self.reranker:
            candidates = self.reranker.rerank(candidates, user_id, topk=20)
        else:
            candidates = candidates[:20]

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
            memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(user_id))
            retrieval_inputs.append("\n".join(f"{m['role']}: {m['content']}" for m in memory))
            session_memories.append(memory)
            user_ids.append(user_id)

        if hasattr(self.retrieval, "batch_text_to_item_retrieval"):
            batch_candidates = self.retrieval.batch_text_to_item_retrieval(retrieval_inputs, topk=self.candidate_k)
        else:
            batch_candidates = [self.retrieval.text_to_item_retrieval(q, topk=self.candidate_k) for q in retrieval_inputs]

        # Rerank and get top items
        final_candidates = []
        top_items = []
        for i, candidates in enumerate(batch_candidates):
            if self.reranker:
                ranked = self.reranker.rerank(candidates, user_ids[i], topk=20)
            else:
                ranked = candidates[:20]
            final_candidates.append(ranked)
            top_items.append(self.item_db.id_to_metadata(ranked[0]))

        # Generate responses
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
