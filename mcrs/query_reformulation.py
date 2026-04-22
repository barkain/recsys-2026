"""LLM-based query reformulation for music retrieval.

Extracts explicit music entities and preferences from conversation history
to build a focused, enriched retrieval query — improving BM25 and dense recall.
"""
import logging
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def _call_claude_cli(system: str, user: str, timeout: int = 20) -> str | None:
    """Call the claude CLI and return the response text, or None on failure."""
    prompt = f"{system}\n\n{user}"
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        logger.warning("claude CLI non-zero exit or empty output: %s", result.stderr[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("claude CLI timed out after %ss", timeout)
        return None
    except Exception as e:
        logger.warning("claude CLI failed: %s", e)
        return None

_SYSTEM_PROMPT = """\
You are a music entity extractor. Given a music recommendation conversation, \
extract the key search signals as a compact JSON object.

Fields (only include if mentioned or clearly implied):
- "artists": list of artist/band names
- "genres": list of music genres (e.g. "jazz", "indie pop", "90s hip-hop")
- "moods": list of moods/feelings (e.g. "upbeat", "melancholic", "energetic")
- "era": decade or period (e.g. "80s", "2010s", "classic")
- "instruments": list of featured instruments
- "themes": lyrical/thematic keywords
- "similar_to": tracks or artists the user wants something similar to
- "user_query": the user's most recent request, verbatim

Return ONLY valid JSON, no explanation."""

_USER_TEMPLATE = """\
Conversation:
{conversation}

Extract music search signals as JSON."""


def _build_enriched_query(entities: dict) -> str:
    """Convert extracted entities dict to a flat retrieval query string."""
    parts = []
    for key in ("artists", "genres", "moods", "era", "instruments", "themes", "similar_to"):
        val = entities.get(key)
        if not val:
            continue
        if isinstance(val, list):
            parts.append(", ".join(str(v) for v in val if v))
        else:
            parts.append(str(val))
    # Always include the raw user query
    user_q = entities.get("user_query", "")
    if user_q:
        parts.append(user_q)
    return " ".join(parts) if parts else user_q


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences from a string."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


class QueryReformulator:
    """Uses Claude to extract music search signals from conversation history."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        fallback_on_error: bool = True,
    ):
        self.model = model
        self.fallback_on_error = fallback_on_error

    def _conversation_to_text(self, session_memory: list[dict], user_query: str) -> str:
        lines = []
        for msg in session_memory:
            role = msg["role"].capitalize()
            content = msg["content"]
            # item metadata dict → summarise
            if isinstance(content, dict):
                content = content.get("track_name", "") + " by " + content.get("artist_name", "")
            lines.append(f"{role}: {content}")
        lines.append(f"User: {user_query}")
        return "\n".join(lines)

    def reformulate(self, session_memory: list[dict], user_query: str) -> str:
        """Return an enriched retrieval query string."""
        conversation_text = self._conversation_to_text(session_memory, user_query)
        try:
            raw = _call_claude_cli(
                _SYSTEM_PROMPT,
                _USER_TEMPLATE.format(conversation=conversation_text),
                timeout=20,
            )
            if raw is None:
                raise RuntimeError("claude CLI returned no output")
            raw = _strip_fences(raw)
            entities = json.loads(raw)
            enriched = _build_enriched_query(entities)
            return enriched if enriched.strip() else user_query
        except Exception as e:
            if self.fallback_on_error:
                logger.warning("QueryReformulator failed, falling back to raw query: %s", e)
                return user_query
            raise

    def batch_reformulate(self, batch: list[tuple[list[dict], str]]) -> list[str]:
        """Reformulate a batch of (session_memory, user_query) pairs in parallel."""
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(self.reformulate, mem, q) for mem, q in batch]
            return [f.result() for f in futures]
