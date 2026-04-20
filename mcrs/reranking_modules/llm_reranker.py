"""LLM listwise reranker for music CRS.

Given the conversation history and a set of candidate tracks, prompts Claude
to return the top-20 most relevant tracks in ranked order.  This directly
optimises for conversational relevance at the nDCG@20 cut-off.

Design:
  • Single-pass over candidates up to window_size (default: all candidates).
  • Graceful fallback: if the LLM response cannot be parsed, original order
    is preserved.
  • Stacks on top of user-profile reranker — call this *after* coarse
    user-profile reranking for best results.
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
import anthropic

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a music recommendation expert.  Given a conversation and a numbered
list of candidate tracks (with genre tags and release year), select the {topk}
tracks that are MOST relevant to what the user wants RIGHT NOW.

Scoring criteria (in priority order):
1. Genre/style match — does it fit the genre, mood, or vibe the user described?
2. Artist match — if the user named an artist or asked for similar artists
3. Era/decade match — if the user mentioned a time period (e.g. "80s", "2000s")
4. Thematic fit — lyrical themes, energy level, tempo cues from conversation
5. Diversity — prefer varied results over 20 tracks by the same artist

Return them in DESCENDING relevance order (best match first).

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
            name = content.get("track_name") or "unknown track"
            artist = content.get("artist_name") or "unknown artist"
            content = f"{name} by {artist}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _format_candidates(candidates: list[str], item_db) -> str:
    """Build a numbered candidate list for the LLM prompt, including track_ids."""
    lines = []
    for i, track_id in enumerate(candidates, 1):
        if item_db:
            meta = item_db.id_to_metadata(track_id)
            if isinstance(meta, dict):
                meta_str = " | ".join(f"{k}: {v}" for k, v in meta.items() if v)
            else:
                meta_str = str(meta)
        else:
            meta_str = ""
        lines.append(f"{i}. [track_id: {track_id}] {meta_str}")
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

    # Try to find a JSON array anywhere in the response (greedy to capture full array)
    match = re.search(r"\[.*\]", text, re.DOTALL)
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
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text
            reranked = _parse_llm_response(raw, valid_ids, k)
            if reranked:
                # Append any candidates not returned by LLM (preserve coverage)
                remaining = [c for c in candidates if c not in set(reranked)]
                return (reranked + remaining)[:k]
        except Exception as e:
            logger.warning("LLMListwiseReranker failed: %s", e)

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
        """Rerank a batch in parallel using threads."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self.rerank, cands, mem, item_db, topk)
                for cands, mem in zip(batch_candidates, batch_session_memory)
            ]
            return [f.result() for f in futures]
