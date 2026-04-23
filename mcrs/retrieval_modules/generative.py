"""LLM-guided generative retrieval.

Asks the LLM to suggest specific track/artist names given the conversation,
then searches the BM25 index for those specific tracks. This bypasses BM25's
keyword-matching weakness for sessions where the user describes a vibe or mood
rather than specific artists — the LLM knows music and can suggest real tracks.

Pipeline:
  1. LLM generates N specific track suggestions (track_name, artist_name pairs)
  2. Each suggestion becomes a targeted BM25 query
  3. Results merged with standard dual-QR BM25 via RRF
  4. Top-K passed to reranker as usual
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from mcrs.utils import call_llm_api

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a music expert. Given a music recommendation conversation, suggest \
specific tracks from your music knowledge that would be ideal recommendations.

Think broadly — cover:
- Tracks directly matching the explicit request
- The same artist's related work
- Similar artists in the same genre/subgenre
- Tracks with matching mood, energy, or era
- Hidden gems and well-known classics alike

Return ONLY a JSON array of up to 25 objects, each with "track" and "artist" keys.
No explanation, no markdown fences — raw JSON only."""

_USER_TEMPLATE = """\
Conversation:
{conversation}

Suggest 25 specific tracks (JSON array only):"""


def _conversation_to_text(session_memory: list[dict], user_query: str) -> str:
    lines = []
    for msg in session_memory:
        role = msg["role"].capitalize()
        content = msg.get("content", "")
        if isinstance(content, dict):
            name = content.get("track_name", "")
            artist = content.get("artist_name", "")
            content = f"{name} by {artist}" if name else str(content)
        lines.append(f"{role}: {content}")
    lines.append(f"User: {user_query}")
    return "\n".join(lines)


def _parse_suggestions(raw: str) -> list[dict]:
    """Parse LLM output into list of {track, artist} dicts."""
    if raw is None:
        return []
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [
                {"track": str(d.get("track", "")), "artist": str(d.get("artist", ""))}
                for d in items
                if isinstance(d, dict) and (d.get("track") or d.get("artist"))
            ]
    except json.JSONDecodeError:
        logger.warning("GenerativeRetriever: failed to parse LLM suggestions — raw: %r", raw[:300])
    return []


class GenerativeRetriever:
    """Generates BM25 queries from LLM-suggested track/artist names.

    Args:
        model: LLM model to use for suggestions.
        n_suggestions: Number of track suggestions to request from the LLM.
        bm25_topk_per_query: How many BM25 results to take per LLM suggestion.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        n_suggestions: int = 25,
        bm25_topk_per_query: int = 3,
    ):
        self.model = model
        self.n_suggestions = n_suggestions
        self.bm25_topk_per_query = bm25_topk_per_query

    def get_suggestions(self, session_memory: list[dict], user_query: str) -> list[dict]:
        """Return list of {track, artist} suggestions from the LLM."""
        conversation_text = _conversation_to_text(session_memory, user_query)
        raw = call_llm_api(
            _SYSTEM_PROMPT,
            _USER_TEMPLATE.format(conversation=conversation_text),
            model=self.model,
            max_tokens=1024,
        )
        suggestions = _parse_suggestions(raw)
        logger.debug("GenerativeRetriever: got %d suggestions", len(suggestions))
        return suggestions

    def suggestions_to_queries(self, suggestions: list[dict]) -> list[str]:
        """Convert suggestions to BM25 query strings."""
        queries = []
        for s in suggestions:
            track = s.get("track", "").strip()
            artist = s.get("artist", "").strip()
            if track and artist:
                queries.append(f"{track} {artist}")
            elif track:
                queries.append(track)
            elif artist:
                queries.append(artist)
        return queries

    def batch_get_suggestions(
        self, batch: list[tuple[list[dict], str]]
    ) -> list[list[dict]]:
        """Get suggestions for a batch of sessions in parallel."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.get_suggestions, mem, q) for mem, q in batch]
            return [f.result() for f in futures]
