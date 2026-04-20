"""Multi-query retrieval for music CRS.

Generates multiple diverse queries from the conversation (genre-focused,
artist-focused, mood/era-focused) and fuses their retrieval results via RRF.

This increases recall by surfacing tracks that match different facets of
what the user wants — e.g. "80s synth-pop" and "danceable electronic upbeat"
may retrieve different but equally relevant tracks.
"""
from __future__ import annotations

import json
import os
import anthropic
from mcrs.retrieval_modules.hybrid import HybridRetriever, rrf_fusion


_SYSTEM_PROMPT = """\
You are a music retrieval query generator. Given a conversation, produce
EXACTLY {n_queries} diverse retrieval queries that together cover all aspects
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

Generate {n_queries} diverse retrieval queries as JSON."""


def _conversation_to_text(session_memory: list[dict], user_query: str) -> str:
    lines = []
    for msg in session_memory:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, dict):
            content = content.get("track_name", "") + " by " + content.get("artist_name", "")
        lines.append(f"{role}: {content}")
    lines.append(f"User: {user_query}")
    return "\n".join(lines)


class MultiQueryRetriever:
    """Hybrid retriever with LLM-generated multi-query expansion.

    Given a conversation, generates N diverse queries via Claude and fuses
    all retrieval results via RRF — increasing recall across facets.

    Args:
        base_retriever: Underlying HybridRetriever instance.
        model: Claude model for query generation.
        n_queries: Number of diverse queries to generate (default 3).
        per_query_k: Candidates per query before fusion (default 100).
        fallback_on_error: If True, fall back to single-query on LLM failure.
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
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _generate_queries(self, session_memory: list[dict], user_query: str) -> list[str]:
        """Call Claude to generate N diverse queries. Returns list of query strings."""
        conversation_text = _conversation_to_text(session_memory, user_query)
        system = _SYSTEM_PROMPT.format(n_queries=self.n_queries)
        user_msg = _USER_TEMPLATE.format(
            conversation=conversation_text, n_queries=self.n_queries
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
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
        return queries if queries else [user_query]

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
        except Exception:
            if self.fallback_on_error:
                queries = [user_query]
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
        """Retrieve for a batch of (session_memory, user_query) pairs."""
        return [self.retrieve(mem, q, topk=topk) for mem, q in batch]
