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

from mcrs.utils import call_claude_api

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a music recommendation expert. Given a conversation and a numbered list \
of candidate tracks, rank all {topk} tracks from most to least relevant to what \
the user wants RIGHT NOW.

Your #1 ranked track must be the single BEST match for the user's immediate request — \
the exact track they are most likely asking for at this moment. Be decisive: commit to \
the strongest match at rank 1, even if other tracks are plausible.

Scoring criteria (in strict priority order):
1. Explicit user request — if the user named a specific artist, band, or track, \
that MUST be rank 1, non-negotiable
2. Genre/style/sonic texture match — fits the genre, mood, or vibe the user described
3. Artist match — user asked for similar artists or named influences
4. Era/decade match — user mentioned a time period (e.g. "80s", "2000s")
5. Thematic/mood fit — lyrical themes, energy level, tempo cues from conversation

IMPORTANT: Tracks already recommended in the conversation (shown as "Music:" turns) \
have already been played. Do not place these at the top unless the user explicitly \
asks to hear them again.

Rules:
- Return ONLY a JSON array of track_id strings, e.g. ["id1", "id2", ...]
- Length must be exactly {topk} items (or fewer if fewer candidates exist)
- Use only track_ids from the provided candidate list — include ALL {topk} candidates
- Do NOT add explanations or any text outside the JSON array"""

_USER_TEMPLATE = """\
Current user request: {last_user_message}

Full conversation:
{conversation}

Candidate tracks (track_id → metadata):
{candidates_text}

Return the {topk} most relevant track_ids as a JSON array."""

_USER_TEMPLATE_WITH_QUERY = """\
Current user request: {last_user_message}

Full conversation:
{conversation}

Synthesized search query (what the user wants RIGHT NOW):
{reformulated_query}

Candidate tracks (track_id → metadata):
{candidates_text}

Return the {topk} most relevant track_ids as a JSON array."""


def _format_content(content) -> str:
    if isinstance(content, dict):
        name = content.get("track_name") or "unknown track"
        artist = content.get("artist_name") or "unknown artist"
        return f"{name} by {artist}"
    return str(content)


def _format_conversation(session_memory: list[dict]) -> str:
    lines = []
    for msg in session_memory:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {_format_content(msg['content'])}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _get_last_user_message(session_memory: list[dict]) -> str:
    """Return the most recent user message text."""
    for msg in reversed(session_memory):
        if msg["role"] == "user":
            return _format_content(msg["content"])
    return ""


_USEFUL_META_KEYS = {"track_name", "artist_name", "tag_list", "release_year", "album_name"}

_TRACK_ID_RE = re.compile(r"track_id:\s*(\S+?)(?:,|$)")


def _get_already_played(session_memory: list[dict]) -> set[str]:
    """Extract track IDs of tracks already recommended in the conversation.

    Prior music recommendations are stored in assistant turns as the metadata
    string from id_to_metadata (e.g. "track_id: 123, track_name: ...").
    """
    played = set()
    for msg in session_memory:
        if msg["role"] in ("assistant", "music"):
            content = str(msg.get("content", ""))
            m = _TRACK_ID_RE.search(content)
            if m:
                played.add(m.group(1).strip(",").strip())
    return played


def _format_candidates(candidates: list[str], item_db) -> str:
    """Build a numbered candidate list for the LLM prompt, including track_ids.

    item_db can be either a plain dict ({track_id: metadata_dict}) or an object
    with an id_to_metadata(track_id) method (MusicCatalogDB).
    """
    lines = []
    for i, track_id in enumerate(candidates, 1):
        meta_str = ""
        if item_db is not None:
            if isinstance(item_db, dict):
                meta = item_db.get(track_id, {})
                if isinstance(meta, dict):
                    meta_str = " | ".join(
                        f"{k}: {v}" for k, v in meta.items()
                        if k in _USEFUL_META_KEYS and v
                    )
            else:
                # MusicCatalogDB.id_to_metadata returns a formatted string directly
                meta_str = item_db.id_to_metadata(track_id) or ""
        lines.append(f"{i}. {meta_str}" if meta_str else f"{i}. [track_id: {track_id}]")
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
        logger.warning("LLM reranker JSON parse failed; raw snippet: %r", raw[:200])
        return None

    if not isinstance(ids, list):
        return None

    # Filter to valid IDs only, deduplicating while preserving order
    seen: set[str] = set()
    result = []
    for tid in ids:
        s = str(tid)
        if s in valid_ids and s not in seen:
            seen.add(s)
            result.append(s)
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

    def rerank(
        self,
        candidates: list[str],
        session_memory: list[dict],
        item_db,
        topk: int | None = None,
        reformulated_query: str | None = None,
    ) -> list[str]:
        """Rerank candidates using LLM.

        Args:
            candidates: Ordered list of track_ids (best-first from prior stage).
            session_memory: Full conversation history (includes current user turn).
            item_db: MusicCatalogDB instance for metadata lookup.
            topk: Override default topk if provided.
            reformulated_query: Optional NLQ query string synthesized from conversation.
                When provided, it is shown to the LLM as an explicit search signal,
                helping it focus on what the user wants right now.

        Returns:
            Reranked list of track_ids, length min(topk, len(candidates)).
        """
        k = topk if topk is not None else self.topk
        if not candidates:
            return []

        # Filter already-played tracks from the window (they appear in assistant turns)
        played = _get_already_played(session_memory)
        if played:
            # Move already-played tracks to end so new tracks fill the window
            fresh = [c for c in candidates if c not in played]
            stale = [c for c in candidates if c in played]
            candidates = fresh + stale

        window = candidates if self.window_size is None else candidates[: self.window_size]
        valid_ids = set(window)
        conversation_text = _format_conversation(session_memory)
        last_user_msg = _get_last_user_message(session_memory)
        candidates_text = _format_candidates(window, item_db)

        system = _SYSTEM_PROMPT.format(topk=min(k, len(window)))
        if reformulated_query:
            user_msg = _USER_TEMPLATE_WITH_QUERY.format(
                conversation=conversation_text,
                last_user_message=last_user_msg,
                reformulated_query=reformulated_query,
                candidates_text=candidates_text,
                topk=min(k, len(window)),
            )
        else:
            user_msg = _USER_TEMPLATE.format(
                conversation=conversation_text,
                last_user_message=last_user_msg,
                candidates_text=candidates_text,
                topk=min(k, len(window)),
            )

        try:
            raw = call_claude_api(system, user_msg, model=self.model, max_tokens=1024)
            if raw is None:
                raise RuntimeError("claude CLI returned no output")
            reranked = _parse_llm_response(raw, valid_ids, k)
            if reranked:
                # Append any candidates not returned by LLM (preserve coverage)
                reranked_set = set(reranked)
                remaining = [c for c in candidates if c not in reranked_set]
                combined = reranked + remaining
                # Final safety dedup (preserving order)
                seen: set[str] = set()
                combined = [x for x in combined if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
                return combined[:k]
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
        reformulated_queries: list[str] | None = None,
    ) -> list[list[str]]:
        """Rerank a batch in parallel using threads."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    self.rerank, cands, mem, item_db, topk,
                    reformulated_queries[i] if reformulated_queries else None,
                )
                for i, (cands, mem) in enumerate(zip(batch_candidates, batch_session_memory))
            ]
            return [f.result() for f in futures]
