"""Echo's enhanced CRS system for RecSys Challenge 2026."""
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module


def _trace_init(message: str) -> None:
    if os.environ.get("MCRS_INIT_TRACE"):
        print(f"[mcrs-init] {message}", flush=True)  # noqa: T201


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
        query_reformulation_mode: str = "entity",
        # Dual-QR: run both NLQ and entity QR, merge candidates via RRF
        use_dual_qr: bool = False,
        # Track-similarity query: also retrieve using last music recommendation metadata
        use_track_sim_query: bool = False,
        # LLM listwise reranker
        use_llm_reranker: bool = False,
        llm_reranker_model: str = "claude-haiku-4-5-20251001",
        llm_reranker_window: int = 50,
        fail_fast_llm: bool = False,
        # Multi-query retrieval
        n_queries: int = 3,
        per_query_k: int | None = None,
        # Precomputed embedding retrieval
        embed_column: str = "metadata-qwen3_embedding_0.6b",
        embed_dataset: str = "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        # Generative retrieval: LLM suggests specific track/artist names → targeted BM25
        use_generative_retrieval: bool = False,
        generative_retrieval_model: str = "claude-haiku-4-5-20251001",
        # Track embedding similarity: find neighbors of last-played track via FAISS
        use_track_embedding_sim: bool = False,
        track_sim_embed_column: str = "metadata-qwen3_embedding_0.6b",
        track_sim_topk: int = 30,
        generate_responses: bool = True,
        use_history_postrank: bool = False,
        history_postrank_weights: list[float] | None = None,
        # Width of the candidate pool persisted alongside top-20 predictions.
        # Does NOT change top-20 emission — additive only. Default 50 matches
        # the standard llm_reranker_window so the pool is the full rerank input.
        save_rerank_pool_k: int = 50,
        # Ignored pass-through params (e.g. test_dataset_name from run script)
        **_ignored,
    ):
        self.cache_dir = cache_dir
        self.candidate_k = candidate_k
        self.use_query_reformulation = use_query_reformulation
        self.retrieval_type = retrieval_type
        self.use_track_sim_query = use_track_sim_query
        self.generate_responses = generate_responses
        self.use_history_postrank = use_history_postrank
        self.history_postrank_weights = history_postrank_weights or [1.0, 0.0, 0.0, 4.0, 0.5]
        self.save_rerank_pool_k = max(20, int(save_rerank_pool_k))

        # Build base retriever
        _base_type = retrieval_type
        if retrieval_type == "multi_query":
            _base_type = "hybrid"
        elif retrieval_type == "bm25_multi_query":
            _base_type = "bm25"

        _trace_init(f"loading base retriever: {_base_type}")
        base_retriever = load_retrieval_module(
            _base_type,
            item_db_name,
            track_split_types,
            corpus_types,
            cache_dir,
            dense_model=dense_model,
            candidate_k=candidate_k,
            embed_column=embed_column,
            embed_dataset=embed_dataset,
        )
        _trace_init("base retriever loaded")

        # Wrap with multi-query if requested
        if retrieval_type in ("multi_query", "bm25_multi_query"):
            from mcrs.retrieval_modules.multi_query import MultiQueryRetriever
            _trace_init("wrapping multi-query retriever")
            self.retrieval = MultiQueryRetriever(
                base_retriever=base_retriever,
                model=query_reformulation_model,
                n_queries=n_queries,
                per_query_k=per_query_k or candidate_k,
            )
        else:
            self.retrieval = base_retriever

        _trace_init("loading item catalog")
        self.item_db = MusicCatalogDB(item_db_name, track_split_types)
        _trace_init("item catalog loaded")
        _trace_init("loading user profiles")
        self.user_db = UserProfileDB(user_db_name, user_split_types)
        _trace_init("user profiles loaded")
        response_lm_type = lm_type if generate_responses else "null"
        _trace_init(f"loading LM module: {response_lm_type}")
        self.lm = load_lm_module(
            response_lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype
        )
        _trace_init("LM module loaded")

        # Optional query reformulator (entity extraction or NLQ synthesis → enriched query)
        self.query_reformulator = None
        self.query_reformulator_aux = None  # Second QR for dual-QR ensemble
        if use_dual_qr and retrieval_type not in ("multi_query", "bm25_multi_query"):
            _trace_init("loading dual query reformulators")
            from mcrs.query_reformulation import QueryReformulator
            # Primary: NLQ for semantic richness; auxiliary: entity for keyword precision
            self.query_reformulator = QueryReformulator(model=query_reformulation_model, mode="nlq")
            self.query_reformulator.fallback_on_error = not fail_fast_llm
            self.query_reformulator_aux = QueryReformulator(model=query_reformulation_model, mode="entity")
            self.query_reformulator_aux.fallback_on_error = not fail_fast_llm
        elif use_query_reformulation and retrieval_type not in ("multi_query", "bm25_multi_query"):
            _trace_init("loading query reformulator")
            from mcrs.query_reformulation import QueryReformulator
            self.query_reformulator = QueryReformulator(
                model=query_reformulation_model,
                mode=query_reformulation_mode,
                fallback_on_error=not fail_fast_llm,
            )

        # Optional track embedding similarity retriever
        self.track_sim_retriever = None
        self.track_sim_topk = track_sim_topk
        if use_track_embedding_sim:
            _trace_init("loading track embedding similarity retriever")
            from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
            self.track_sim_retriever = TrackSimilarityRetriever(
                embed_column=track_sim_embed_column,
                split_types=track_split_types,
                cache_dir=cache_dir,
            )
            _trace_init("track embedding similarity retriever loaded")

        # Optional generative retriever (LLM-guided track suggestion → targeted BM25)
        self.generative_retriever = None
        if use_generative_retrieval:
            _trace_init("loading generative retriever")
            from mcrs.retrieval_modules.generative import GenerativeRetriever
            self.generative_retriever = GenerativeRetriever(model=generative_retrieval_model)
            _trace_init("generative retriever loaded")

        # Optional user-profile reranker
        self.reranker = None
        if use_reranker:
            _trace_init("loading user-profile reranker")
            from mcrs.reranking_modules import UserProfileReranker
            self.reranker = UserProfileReranker(cache_dir=cache_dir, alpha=reranker_alpha)
            _trace_init("user-profile reranker loaded")

        # Optional LLM listwise reranker
        self.llm_reranker = None
        if use_llm_reranker:
            _trace_init("loading LLM listwise reranker")
            from mcrs.reranking_modules.llm_reranker import LLMListwiseReranker
            self.llm_reranker = LLMListwiseReranker(
                model=llm_reranker_model,
                topk=20,
                window_size=llm_reranker_window if llm_reranker_window else None,
                fallback_on_error=not fail_fast_llm,
            )
            _trace_init("LLM listwise reranker loaded")

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
        _trace_init("CRS init complete")

    def _get_system_prompt(self, user_id: str | None = None) -> str:
        prompt = self.role_prompt["role_play"] + "\n" + self.role_prompt["response_generation"]
        if user_id:
            profile_str = self.user_db.id_to_profile_str(user_id)
            if profile_str:
                prompt += "\n" + self.role_prompt["personalization"] + "\n" + profile_str
        return prompt

    def _last_played_track_id(self, session_memory: list[dict]) -> str | None:
        """Return the track_id of the most recent music recommendation in memory."""
        for msg in reversed(session_memory):
            if msg.get("role") not in ("assistant", "music"):
                continue
            content = msg.get("content")
            if isinstance(content, dict):
                tid = content.get("track_id") or content.get("id")
                if tid:
                    return str(tid)
            elif isinstance(content, str) and content:
                # Raw dataset history uses track ids for music turns. Parsed
                # inference history uses assistant metadata strings.
                if content.startswith("track_id:"):
                    return content.split("track_id:", 1)[1].split(",", 1)[0].strip()
                return content
        return None

    def _track_sim_candidates(self, session_memory: list[dict]) -> list[str] | None:
        """Return BM25 candidates using the last music recommendation's metadata as query.

        This boosts recall when the next recommendation is by the same artist or genre.
        """
        for msg in reversed(session_memory):
            content = msg.get("content")
            if isinstance(content, dict):
                parts = []
                artist = content.get("artist_name", "")
                track = content.get("track_name", "")
                tags = content.get("tag_list", [])
                if artist:
                    parts.append(artist)
                if track:
                    parts.append(track)
                if isinstance(tags, list):
                    parts.extend(str(t) for t in tags[:5])
                elif tags:
                    parts.append(str(tags))
                if parts:
                    query = " ".join(parts)
                    return self.retrieval.text_to_item_retrieval(query, topk=self.candidate_k)
        return None

    @staticmethod
    def _rrf_merge(lists: list[list[str]], k: int = 60, topk: int = 100) -> list[str]:
        """Reciprocal Rank Fusion over multiple ranked candidate lists."""
        scores: dict[str, float] = {}
        for ranked in lists:
            for rank, track_id in enumerate(ranked):
                scores[track_id] = scores.get(track_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]

    def _retrieve(self, session_memory: list[dict], user_query: str) -> tuple[list[str], str | None]:
        """Run retrieval, using multi-query or reformulation if configured.

        Returns:
            (candidates, reformulated_query) where reformulated_query is the
            NLQ/entity query string (or None if no reformulation was used).
            The reformulated_query is passed to the LLM reranker as an explicit signal.
        """
        if self.retrieval_type in ("multi_query", "bm25_multi_query"):
            return self.retrieval.retrieve(
                session_memory, user_query, topk=self.candidate_k
            ), None

        if self.query_reformulator_aux:
            # Dual-QR: run both reformulators in parallel, merge with RRF
            n_workers = 3 if self.generative_retriever else 2
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                f1 = pool.submit(self.query_reformulator.reformulate, session_memory, user_query)
                f2 = pool.submit(self.query_reformulator_aux.reformulate, session_memory, user_query)
                fg = pool.submit(self.generative_retriever.get_suggestions, session_memory, user_query) if self.generative_retriever else None
                q1, q2 = f1.result(), f2.result()
                gen_suggestions = fg.result() if fg else []
            cands1 = self.retrieval.text_to_item_retrieval(q1, topk=self.candidate_k)
            cands2 = self.retrieval.text_to_item_retrieval(q2, topk=self.candidate_k)
            lists = [cands1, cands2]
            # Generative retrieval: run targeted BM25 for each LLM-suggested track/artist
            if gen_suggestions:
                gen_queries = self.generative_retriever.suggestions_to_queries(gen_suggestions)
                gen_lists = [
                    self.retrieval.text_to_item_retrieval(gq, topk=5)
                    for gq in gen_queries
                ]
                lists.extend(gen_lists)
            if self.use_track_sim_query:
                sim_cands = self._track_sim_candidates(session_memory)
                if sim_cands:
                    lists.append(sim_cands)
            if self.track_sim_retriever:
                last_tid = self._last_played_track_id(session_memory)
                if last_tid:
                    embed_sim = self.track_sim_retriever.track_id_to_neighbors(
                        last_tid, topk=self.track_sim_topk
                    )
                    if embed_sim:
                        lists.append(embed_sim)
            return self._rrf_merge(lists, topk=self.candidate_k), q1  # NLQ as hint to reranker
        elif self.query_reformulator:
            query = self.query_reformulator.reformulate(session_memory, user_query)
        else:
            query = None

        raw_query = query or (
            "\n".join(f"{m['role']}: {m['content']}" for m in session_memory)
            + (f"\nUser: {user_query}" if user_query else "")
        )
        main_cands = self.retrieval.text_to_item_retrieval(raw_query, topk=self.candidate_k)
        if self.use_track_sim_query:
            sim_cands = self._track_sim_candidates(session_memory)
            if sim_cands:
                return self._rrf_merge([main_cands, sim_cands], topk=self.candidate_k), query
        return main_cands, query

    def _batch_retrieve(
        self,
        session_memories: list[list[dict]],
        user_queries: list[str],
    ) -> tuple[list[list[str]], list[str | None]]:
        """Batch retrieval with optional multi-query or reformulation.

        Returns:
            (list_of_candidates, list_of_reformulated_queries)
            reformulated_queries contains the NLQ/entity query for each session,
            or None where no reformulation was used.
        """
        if self.retrieval_type in ("multi_query", "bm25_multi_query"):
            cands = self.retrieval.batch_retrieve(
                list(zip(session_memories, user_queries)), topk=self.candidate_k
            )
            return cands, [None] * len(cands)

        if self.query_reformulator_aux:
            # Dual-QR: run both reformulators in parallel per batch, merge with RRF
            pairs = list(zip(session_memories, user_queries))
            n_workers = 3 if self.generative_retriever else 2
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                f1 = pool.submit(self.query_reformulator.batch_reformulate, pairs)
                f2 = pool.submit(self.query_reformulator_aux.batch_reformulate, pairs)
                fg = pool.submit(self.generative_retriever.batch_get_suggestions, pairs) if self.generative_retriever else None
                queries1, queries2 = f1.result(), f2.result()
                gen_suggestions_batch = fg.result() if fg else [[] for _ in pairs]
            results = []
            for (mem, _uq), q1, q2, gen_suggestions in zip(
                pairs, queries1, queries2, gen_suggestions_batch
            ):
                c1 = self.retrieval.text_to_item_retrieval(q1, topk=self.candidate_k)
                c2 = self.retrieval.text_to_item_retrieval(q2, topk=self.candidate_k)
                lists = [c1, c2]
                if gen_suggestions:
                    gen_queries = self.generative_retriever.suggestions_to_queries(gen_suggestions)
                    for gq in gen_queries:
                        lists.append(self.retrieval.text_to_item_retrieval(gq, topk=5))
                if self.use_track_sim_query:
                    sim_cands = self._track_sim_candidates(mem)
                    if sim_cands:
                        lists.append(sim_cands)
                if self.track_sim_retriever:
                    last_tid = self._last_played_track_id(mem)
                    if last_tid:
                        embed_sim = self.track_sim_retriever.track_id_to_neighbors(
                            last_tid, topk=self.track_sim_topk
                        )
                        if embed_sim:
                            lists.append(embed_sim)
                results.append(self._rrf_merge(lists, topk=self.candidate_k))
            return results, queries1  # pass NLQ queries as hints to reranker
        elif self.query_reformulator:
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
            main_results = self.retrieval.batch_text_to_item_retrieval(queries, topk=self.candidate_k)
        else:
            main_results = [self.retrieval.text_to_item_retrieval(q, topk=self.candidate_k) for q in queries]

        if self.use_track_sim_query:
            merged = []
            for i, mem in enumerate(session_memories):
                sim_cands = self._track_sim_candidates(mem)
                if sim_cands:
                    merged.append(self._rrf_merge([main_results[i], sim_cands], topk=self.candidate_k))
                else:
                    merged.append(main_results[i])
            return merged, queries if self.query_reformulator else [None] * len(queries)
        return main_results, queries if self.query_reformulator else [None] * len(queries)

    def _rerank(
        self,
        candidates: list[str],
        user_id: str | None,
        session_memory: list[dict],
        reformulated_query: str | None = None,
    ) -> list[str]:
        """Apply reranking stages in order: user-profile → LLM listwise."""
        if self.reranker:
            candidates = self.reranker.rerank(candidates, user_id, topk=len(candidates))
        if self.llm_reranker:
            candidates = self.llm_reranker.rerank(
                candidates, session_memory, self.item_db, topk=20,
                reformulated_query=reformulated_query,
            )
        if self.use_history_postrank:
            candidates = self._history_postrank(candidates, session_memory)
        return candidates[:20]

    def _rerank_with_pool(
        self,
        candidates: list[str],
        user_id: str | None,
        session_memory: list[dict],
        reformulated_query: str | None = None,
        pool_k: int = 50,
    ) -> tuple[list[str], list[str], dict]:
        """Like _rerank, but also returns the wider candidate pool and provenance.

        Returns:
            (top_20, pool, provenance)
              - top_20: identical to what _rerank would return (non-regression).
              - pool: combined candidate pool of length up to pool_k. Built as
                LLM-reranked prefix + remaining rerank-input-window candidates
                in retrieval order, deduped. If no LLM reranker is configured,
                `pool` is the upstream candidates list (pre-postrank) truncated.
              - provenance: dict describing pool construction. Honest about
                which ranks are LLM-ranked vs. upstream retrieval order.
        """
        if self.reranker:
            candidates = self.reranker.rerank(candidates, user_id, topk=len(candidates))

        provenance: dict = {
            "pool_k": pool_k,
            "reranker_class": None,
            "llm_reranked_prefix": 0,
            "pool_size": 0,
            "window_size": None,
            "remainder_source": "upstream_retrieval_order_no_llm_reranker",
            "llm_success": False,
            "history_postrank_applied": False,
        }

        if self.llm_reranker:
            top_k, pool, llm_prov = self.llm_reranker.rerank_with_pool(
                candidates, session_memory, self.item_db, topk=20,
                reformulated_query=reformulated_query,
            )
            provenance.update(llm_prov)
        else:
            # No LLM reranker — pool is just the upstream candidates, top_k is top 20
            pool = list(candidates)
            top_k = candidates[:20]
            provenance["reranker_class"] = None
            provenance["pool_size"] = len(pool)

        if self.use_history_postrank:
            # History postrank reorders top_k (the 20 that would be emitted).
            top_k_postranked = self._history_postrank(top_k, session_memory)
            # Keep the pool consistent: its prefix should match emitted top_k.
            # Replace the first len(top_k) entries of pool with the postranked
            # ordering, preserving tail (items 21+) in their original order.
            tail = [x for x in pool if x not in set(top_k_postranked)]
            pool = top_k_postranked + tail
            top_k = top_k_postranked
            provenance["history_postrank_applied"] = True
            provenance["remainder_source"] = (
                "history_postrank_applied_to_prefix; "
                + str(provenance.get("remainder_source", ""))
            )

        # Final truncation — top_k to 20, pool to pool_k
        top_k_final = top_k[:20]
        pool_final = pool[:pool_k]
        provenance["pool_size"] = len(pool_final)
        return top_k_final, pool_final, provenance

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "want", "need",
            "looking", "like", "similar", "something", "some", "give", "play",
            "song", "songs", "track", "tracks", "music", "artist", "artists",
            "band", "bands", "more", "another", "please",
        }
        return {
            t for t in re.findall(r"[a-z0-9']+", str(text).lower())
            if len(t) > 2 and t not in stopwords
        }

    def _history_postrank(self, candidates: list[str], session_memory: list[dict]) -> list[str]:
        """Deterministic post-ranker using last-played artist/tags as a light prior."""
        if not candidates:
            return []

        user_messages = [
            str(m.get("content", ""))
            for m in session_memory
            if m.get("role") == "user"
        ]
        now_tokens = self._tokens(user_messages[-1] if user_messages else "")
        all_tokens = self._tokens(" ".join(user_messages))

        last_artist = ""
        last_tags: set[str] = set()
        for msg in reversed(session_memory):
            if msg.get("role") not in ("assistant", "music"):
                continue
            content = msg.get("content")
            if isinstance(content, dict):
                last_artist = str(content.get("artist_name", "")).lower()
                tags = content.get("tag_list") or []
                if isinstance(tags, list):
                    last_tags = {str(t).lower() for t in tags}
                break
            if isinstance(content, str):
                artist_match = re.search(r"artist_name:\s*([^,]+)", content)
                if artist_match:
                    last_artist = artist_match.group(1).strip().lower()
                tags_match = re.search(r"tag_list:\s*(.+?)(?:,\s*[a-z_]+:|$)", content)
                if tags_match:
                    last_tags = {
                        t.strip().strip("'\"[]").lower()
                        for t in re.split(r"[|,]", tags_match.group(1))
                        if t.strip()
                    }
                if last_artist or last_tags:
                    break

        w_orig, w_now, w_all, w_artist, w_tag = self.history_postrank_weights

        def score(track_id: str, rank: int) -> float:
            try:
                meta = self.item_db.id_to_full_metadata(track_id)
            except KeyError:
                meta = {}
            parts = [meta.get("track_name", ""), meta.get("artist_name", ""), meta.get("album_name", "")]
            tags = meta.get("tag_list") or []
            if isinstance(tags, list):
                parts.extend(tags[:12])
                tag_set = {str(t).lower() for t in tags}
            else:
                parts.append(tags)
                tag_set = set()
            meta_tokens = self._tokens(" ".join(str(p) for p in parts if p))
            artist = str(meta.get("artist_name", "")).lower()
            return (
                w_orig * (1.0 / rank)
                + w_now * len(now_tokens & meta_tokens)
                + w_all * len(all_tokens & meta_tokens)
                + w_artist * (1.0 if artist and artist == last_artist else 0.0)
                + w_tag * len(tag_set & last_tags)
            )

        return [
            tid for rank, tid in sorted(
                enumerate(candidates, 1),
                key=lambda x: score(x[1], x[0]),
                reverse=True,
            )
        ]

    def chat(
        self,
        user_query: str,
        session_memory: list[dict],
        user_id: str | None = None,
    ) -> dict[str, Any]:
        session_memory = list(session_memory)
        session_memory.append({"role": "user", "content": user_query})

        candidates, reformulated_query = self._retrieve(session_memory[:-1], user_query)
        candidates = self._rerank(candidates, user_id, session_memory, reformulated_query)

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

        batch_candidates, batch_queries = self._batch_retrieve(session_memories_pre, user_queries)

        # Parallelize reranking — LLM reranker calls are independent HTTP requests
        # and dominate latency; running them concurrently gives ~8x speedup.
        # _rerank_with_pool emits the same top-20 as _rerank (non-regression)
        # plus the wider pool and honest provenance metadata.
        def _rerank_one(args):
            i, candidates = args
            return self._rerank_with_pool(
                candidates, user_ids[i], session_memories_full[i],
                reformulated_query=batch_queries[i] if batch_queries else None,
                pool_k=self.save_rerank_pool_k,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            rerank_results = list(pool.map(_rerank_one, enumerate(batch_candidates)))

        final_candidates = [r[0] for r in rerank_results]
        candidate_pools = [r[1] for r in rerank_results]
        pool_provenances = [r[2] for r in rerank_results]

        top_items = [
            self.item_db.id_to_metadata(ranked[0]) if ranked else ""
            for ranked in final_candidates
        ]

        if self.generate_responses:
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
        else:
            responses = [""] * len(batch_data)

        return [
            {
                "user_id": user_ids[i],
                "user_query": batch_data[i]["user_query"],
                "retrieval_items": final_candidates[i],
                "candidate_pool_items": candidate_pools[i],
                "candidate_pool_provenance": pool_provenances[i],
                "recommend_item": top_items[i],
                "response": responses[i],
            }
            for i in range(len(batch_data))
        ]
