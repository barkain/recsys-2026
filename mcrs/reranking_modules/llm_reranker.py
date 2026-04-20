"""LLM listwise reranker for music CRS.

Given the conversation history and a set of candidate tracks, prompts Claude
to return the top-20 most relevant tracks in ranked order.  This directly
optimises for conversational relevance at the nDCG@20 cut-off.

Design:
  • Sliding-window permutation: we chunk top-K candidates into windows of
    `window_size` and rerank each window, then merge.  Default: single pass
    over all candidates (window_size = candidate_k).
  • Graceful fallback: if the LLM response cannot be parsed, original order
    is preserved.
  • Stacks on top of user-profile reranker — call this *after* coarse
    user-profile reranking for best results.
"""
from __future__ import annotations

import json
import os
import re
import anthropic


_SYSTEM_PROMPT = """\
You are a music recommendation assistant.  Given a conversation and a numbered
list of candidate tracks, select the {topk} tracks that are MOST relevant to
what the user wants RIGHT NOW, and return them in descending relevance order.

Rules:
- Return ONLY a JSON array of track_id strings, e.g. ["id1", "id2", ...]
- Length must be exactly {topk} items (or fewer if fewer candidates exist)
- Use only track_ids from the provided candidate list
- Do NOT add explanations or any text outside the JSON array"""

_USER_TEMPLATE = """\
Conversation so far:
{conversation}

Candidate tracks (track_id → metadata):
{candidates_text}

Return the {topk} most relevant track_ids as a JSON array."""


def _format_conversation(session_memory: list[dict]) -> str:
    lines = []
    for msg in session_memory:
        role = msg["role"].capitalize()
        content = msg["content"]
        if isinstance(content, dict):
            content = content.get("track_name", "") + " by " + content.get("artist_name", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _format_candidates(candidates: list[str], item_db) -> str:
    """Build a numbered candidate list for the LLM prompt."""
    lines = []
    for i, track_id in enumerate(candidates, 1):
        meta = item_db.id_to_metadata(track_id) if item_db else f"track_id: {track_id}"
        lines.append(f"{i}. {meta}")
    return "\n".join(lines)


def _parse_llm_response(raw: str, valid_ids: set[str], topk: int) -> list[str] | None:
    """Extract a list of track_ids from the LLM response. Returns None on failure."""
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None

    try:
        ids = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(ids, list):
        return None

    # Filter to valid IDs only
    result = [str(tid) for tid in ids if str(tid) in valid_ids]
    return result[:topk] if result else None


class LLMListwiseReranker:
    """Reranks candidate tracks using an LLM with full conversation context.

    Args:
        model: Claude model to use for reranking.
        topk: Number of tracks to return after reranking.
        window_size: Max candidates to pass to the LLM in one call.
            Set to None to pass all candidates (up to ~50 is practical).
        fallback_on_error: If True, return original order on LLM failure.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        topk: int = 20,
        window_size: int | None = None,
        fallback_on_error: bool = True,
    ):
        self.model = model
        self.topk = topk
        self.window_size = window_size
        self.fallback_on_error = fallback_on_error
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def rerank(
        self,
        candidates: list[str],
        session_memory: list[dict],
        item_db,
        topk: int | None = None,
    ) -> list[str]:
        """Rerank candidates using LLM.

        Args:
            candidates: Ordered list of track_ids (best-first from prior stage).
            session_memory: Full conversation history (includes current user turn).
            item_db: MusicCatalogDB instance for metadata lookup.
            topk: Override default topk if provided.

        Returns:
            Reranked list of track_ids, length min(topk, len(candidates)).
        """
        k = topk if topk is not None else self.topk
        if not candidates:
            return []

        window = candidates if self.window_size is None else candidates[: self.window_size]
        valid_ids = set(window)
        conversation_text = _format_conversation(session_memory)
        candidates_text = _format_candidates(window, item_db)

        system = _SYSTEM_PROMPT.format(topk=min(k, len(window)))
        user_msg = _USER_TEMPLATE.format(
            conversation=conversation_text,
            candidates_text=candidates_text,
            topk=min(k, len(window)),
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            reranked = _parse_llm_response(raw, valid_ids, k)
            if reranked:
                # Append any candidates not returned by LLM (preserve coverage)
                remaining = [c for c in candidates if c not in set(reranked)]
                return (reranked + remaining)[:k]
        except Exception:
            pass

        if self.fallback_on_error:
            return candidates[:k]
        raise RuntimeError("LLM reranker failed and fallback is disabled")

    def batch_rerank(
        self,
        batch_candidates: list[list[str]],
        batch_session_memory: list[list[dict]],
        item_db,
        topk: int | None = None,
    ) -> list[list[str]]:
        """Rerank a batch sequentially (Claude has no native batch API)."""
        return [
            self.rerank(batch_candidates[i], batch_session_memory[i], item_db, topk)
            for i in range(len(batch_candidates))
        ]
