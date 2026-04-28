"""Multi-query retrieval for music CRS.

Generates multiple diverse queries from the conversation (genre-focused,
artist-focused, mood/era-focused) and fuses their retrieval results via RRF.

This increases recall by surfacing tracks that match different facets of
what the user wants — e.g. "80s synth-pop" and "danceable electronic upbeat"
may retrieve different but equally relevant tracks.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import anthropic
from mcrs.retrieval_modules.hybrid import HybridRetriever, rrf_fusion

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a music retrieval query generator. Given a conversation, produce
EXACTLY 3 diverse retrieval queries that together cover all aspects
of what the user is looking for.

Each query should emphasise a different facet:
  q1 — genre/style and sonic texture (e.g. "mellow acoustic folk indie")
  q2 — artist names and similar artists
  q3 — mood, era, and thematic context (e.g. "nostalgic 80s upbeat summer")

Rules:
- Each query is a short phrase (5-15 words), optimised for BM25 + dense retrieval
- Capture the MOST RECENT user request first, supplemented by prior context
- Return ONLY valid JSON: {"q1": "...", "q2": "...", "q3": "..."}
- No explanations, no extra keys"""

_USER_TEMPLATE = """\
Conversation:
{conversation}

Generate 3 diverse retrieval queries as JSON."""


def _conversation_to_text(session_memory: list[dict], user_query: str) -> str:
    lines = []
    for msg in session_memory:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, dict):
            name = content.get("track_name") or "unknown track"
            artist = content.get("artist_name") or "unknown artist"
            content = f"{name} by {artist}"
        lines.append(f"{role}: {content}")
    lines.append(f"User: {user_query}")
    return "\n".join(lines)


class MultiQueryRetriever:
    """Hybrid retriever with LLM-generated multi-query expansion.

    Given a conversation, generates 3 diverse queries via Claude and fuses
    all retrieval results via RRF — increasing recall across facets.

    Args:
        base_retriever: Underlying HybridRetriever instance.
        model: Claude model for query generation.
        per_query_k: Candidates per query before fusion (default 100).
        fallback_on_error: If True, fall back to full conversation text on LLM failure.
    """

    def __init__(
        self,
        base_retriever: HybridRetriever,
        model: str = "claude-haiku-4-5-20251001",
        n_queries: int = 3,
        per_query_k: int = 100,
        fallback_on_error: bool = True,
    ):
        self.retriever = base_retriever
        self.model = model
        self.n_queries = n_queries
        self.per_query_k = per_query_k
        self.fallback_on_error = fallback_on_error
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_RECSYS_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        )

    def _generate_queries(self, session_memory: list[dict], user_query: str) -> list[str]:
        """Call Claude to generate 3 diverse queries. Returns list of query strings."""
        conversation_text = _conversation_to_text(session_memory, user_query)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _USER_TEMPLATE.format(conversation=conversation_text)}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        data = json.loads(raw)
        queries = [str(data[f"q{i+1}"]) for i in range(self.n_queries) if f"q{i+1}" in data]
        if len(queries) < self.n_queries:
            logger.warning(
                "MultiQueryRetriever: got %d/%d queries from LLM", len(queries), self.n_queries
            )
        return queries if queries else [_conversation_to_text(session_memory, user_query)]

    def retrieve(
        self,
        session_memory: list[dict],
        user_query: str,
        topk: int = 50,
    ) -> list[str]:
        """Retrieve candidates using multi-query fusion.

        Args:
            session_memory: Prior conversation turns.
            user_query: Current user message.
            topk: Number of candidates to return.

        Returns:
            Fused, RRF-ranked list of track_ids.
        """
        try:
            queries = self._generate_queries(session_memory, user_query)
        except Exception as e:
            if self.fallback_on_error:
                logger.warning("MultiQueryRetriever LLM call failed, using single query: %s", e)
                queries = [_conversation_to_text(session_memory, user_query)]
            else:
                raise

        ranked_lists = []
        for q in queries:
            ranked = self.retriever.text_to_item_retrieval(q, topk=self.per_query_k)
            ranked_lists.append(ranked)

        fused = rrf_fusion(ranked_lists)
        return fused[:topk]

    def batch_retrieve(
        self,
        batch: list[tuple[list[dict], str]],
        topk: int = 50,
    ) -> list[list[str]]:
        """Retrieve for a batch of (session_memory, user_query) pairs in parallel."""
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(self.retrieve, mem, q, topk) for mem, q in batch]
            return [f.result() for f in futures]
