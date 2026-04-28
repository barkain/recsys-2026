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
import re
from concurrent.futures import ThreadPoolExecutor

from mcrs.utils import LLMTruncationError, call_llm_api

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a music recommendation expert. Given a conversation and a numbered list \
of candidate tracks, select and rank the TOP {topk} most relevant tracks from the \
{n_candidates} candidates provided.

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
- Return exactly ONE JSON array of candidate numbers
- Length must be exactly {topk} items
- Each number must be an integer between 1 and {n_candidates}, referring to the
  numbered candidate list below
- Do NOT return track_id UUID strings; return candidate numbers only
- Do NOT invent candidates or use numbers outside 1..{n_candidates}
- Do NOT self-correct, include a second array, or write prose such as "Wait" or
  "Let me redo this"
- Do NOT add explanations or any text outside the JSON array"""

_USER_TEMPLATE = """\
Current user request: {last_user_message}

Full conversation:
{conversation}

Candidate tracks ({n_candidates} total, track_id → metadata):
{candidates_text}

Return exactly one JSON array containing the top {topk} candidate numbers.
Use integers 1..{n_candidates} only. Do not return UUIDs or any prose."""

_USER_TEMPLATE_WITH_QUERY = """\
Current user request: {last_user_message}

Full conversation:
{conversation}

Synthesized search query (what the user wants RIGHT NOW):
{reformulated_query}

Candidate tracks ({n_candidates} total, track_id → metadata):
{candidates_text}

Return exactly one JSON array containing the top {topk} candidate numbers.
Use integers 1..{n_candidates} only. Do not return UUIDs or any prose."""


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

_TRACK_ID_RE = re.compile(r"track_id:\s*(\S+?)(?:[,|\s]|$)")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UUID_LIKE_RE = re.compile(r"^[0-9a-fA-F-]{30,40}$")


def _get_already_played(session_memory: list[dict]) -> set[str]:
    """Extract track IDs of tracks already recommended in the conversation.

    Handles two formats:
    - dict content: raw track metadata dict with a "track_id" key
    - string content: id_to_metadata format "track_id: 123, track_name: ..."
    """
    played = set()
    for msg in session_memory:
        if msg["role"] in ("assistant", "music"):
            content = msg.get("content", "")
            if isinstance(content, dict):
                tid = content.get("track_id")
                if tid:
                    played.add(str(tid))
            elif isinstance(content, str):
                m = _TRACK_ID_RE.search(content)
                if m:
                    played.add(m.group(1).rstrip(",").strip())
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
                        [f"track_id: {track_id}"] +
                        [f"{k}: {v}" for k, v in meta.items()
                         if k in _USEFUL_META_KEYS and v]
                    )
            else:
                # MusicCatalogDB.id_to_metadata returns a formatted string;
                # it already starts with "track_id: <id>, ..." but we prepend
                # explicitly for safety in case the format ever changes.
                raw = item_db.id_to_metadata(track_id) or ""
                if raw and not raw.startswith("track_id:"):
                    meta_str = f"track_id: {track_id}, {raw}"
                else:
                    meta_str = raw or f"track_id: {track_id}"
        lines.append(f"{i}. {meta_str}")
    return "\n".join(lines)


def _parse_llm_response(raw: str, valid_ids: set[str], topk: int) -> list[str] | None:
    """Extract a list of track_ids from the LLM response. Returns None on failure."""
    result, _metadata = _parse_llm_response_with_repairs(raw, valid_ids, topk)
    return result


def _parse_candidate_index_response(
    raw: str,
    candidates: list[str],
    topk: int,
) -> tuple[list[str] | None, dict]:
    """Parse a JSON array of 1-based candidate positions and map to track_ids."""
    metadata: dict = {
        "response_format": "candidate_indices",
        "invalid_candidate_indices": [],
    }

    arrays = _extract_json_arrays(raw)
    for ids in reversed(arrays):
        seen: set[int] = set()
        result = []
        invalid = []
        for item in ids:
            if len(result) >= topk:
                break
            if isinstance(item, bool):
                invalid.append(item)
                continue
            if isinstance(item, int):
                idx = item
            elif isinstance(item, str) and item.isdigit():
                idx = int(item)
            else:
                invalid.append(item)
                continue
            if 1 <= idx <= len(candidates) and idx not in seen:
                seen.add(idx)
                result.append(candidates[idx - 1])
            else:
                invalid.append(item)
        if result:
            metadata["invalid_candidate_indices"] = invalid
            return result, metadata

    logger.warning("LLM reranker index JSON parse failed; raw snippet: %r", raw[:200])
    return None, metadata


def _edit_distance_at_most(a: str, b: str, max_distance: int) -> int | None:
    """Return Levenshtein distance if <= max_distance, else None."""
    if abs(len(a) - len(b)) > max_distance:
        return None

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        row_min = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr.append(val)
            row_min = min(row_min, val)
        if row_min > max_distance:
            return None
        prev = curr

    distance = prev[-1]
    return distance if distance <= max_distance else None


def _parse_llm_response_with_repairs(
    raw: str,
    valid_ids: set[str],
    topk: int,
) -> tuple[list[str] | None, dict]:
    """Extract track_ids and deterministic near-miss repair metadata."""
    metadata: dict = {
        "response_format": "track_ids",
        "near_miss_salvage_count": 0,
        "near_miss_salvages": [],
    }

    def _salvage_malformed_id(s: str, unavailable: set[str]) -> str | None:
        # Well-formed UUIDs that are not in the candidate set are hallucinations,
        # not transcription glitches. Do not repair them.
        if _UUID_RE.match(s) or not _UUID_LIKE_RE.match(s):
            return None

        matches: list[tuple[int, str]] = []
        for candidate in valid_ids:
            if candidate in unavailable:
                continue
            distance = _edit_distance_at_most(s.lower(), candidate.lower(), 2)
            if distance is not None:
                matches.append((distance, candidate))
        if len(matches) != 1:
            return None
        return matches[0][1]

    def _filtered_valid_ids(ids: list) -> tuple[list[str], list[dict]]:
        exact_valid = {str(tid) for tid in ids if str(tid) in valid_ids}
        unavailable = set(exact_valid)
        seen: set[str] = set()
        result = []
        salvages = []
        for tid in ids:
            if len(result) >= topk:
                break
            s = str(tid)
            if s in valid_ids:
                if s not in seen:
                    seen.add(s)
                    result.append(s)
                continue

            repaired = _salvage_malformed_id(s, unavailable | seen)
            if repaired and repaired not in seen:
                seen.add(repaired)
                unavailable.add(repaired)
                result.append(repaired)
                salvages.append({"original": s, "corrected": repaired})

        return result[:topk], salvages

    arrays = _extract_json_arrays(raw)

    for ids in reversed(arrays):
        result, salvages = _filtered_valid_ids(ids)
        if result:
            metadata["near_miss_salvage_count"] = len(salvages)
            metadata["near_miss_salvages"] = salvages
            return result, metadata

    logger.warning("LLM reranker JSON parse failed; raw snippet: %r", raw[:200])
    return None, metadata


def _extract_json_arrays(raw: str) -> list[list]:
    """Extract JSON arrays, with a narrow fallback for integer-index arrays.

    Claude occasionally emits candidate indices with leading zeroes (e.g. 05),
    which is not valid JSON but has an unambiguous integer interpretation.
    Only accept this fallback for bracketed comma-separated integers.
    """
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

    # Parse every well-formed JSON array and prefer the last one with valid
    # candidate IDs. This handles model self-corrections like:
    #   ["1", "2", ...] Wait, use track_ids. ["uuid", ...]
    decoder = json.JSONDecoder()
    parsed_arrays: list[list] = []
    for match in re.finditer(r"\[", text):
        try:
            parsed, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            parsed_arrays.append(parsed)
    if parsed_arrays:
        return parsed_arrays

    for match in re.finditer(r"\[([^\[\]]+)\]", text):
        body = match.group(1).strip()
        if not body:
            continue
        tokens = [tok.strip() for tok in body.split(",")]
        if all(re.fullmatch(r"[+-]?\d+", tok) for tok in tokens):
            parsed_arrays.append([int(tok) for tok in tokens])
    return parsed_arrays


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
        top_k, _pool, _prov = self.rerank_with_pool(
            candidates, session_memory, item_db,
            topk=topk, reformulated_query=reformulated_query,
        )
        return top_k

    def rerank_with_pool(
        self,
        candidates: list[str],
        session_memory: list[dict],
        item_db,
        topk: int | None = None,
        reformulated_query: str | None = None,
    ) -> tuple[list[str], list[str], dict]:
        """Rerank and also return the full combined candidate pool.

        Returns:
            (top_k, pool, provenance):
              - top_k: length min(topk, len(window)), same as rerank().
              - pool: full deduped combined list (LLM-reranked prefix +
                remaining rerank-input-window candidates in retrieval order).
                Length up to len(window).
              - provenance: dict describing how the pool was assembled.
                Honestly labels ranks 21+ as coming from the rerank-input
                window in retrieval order (not LLM-ranked).
        """
        k = topk if topk is not None else self.topk
        if not candidates:
            return [], [], {
                "reranker_class": type(self).__name__,
                "llm_reranked_prefix": 0,
                "pool_size": 0,
                "window_size": self.window_size,
                "remainder_source": "none",
                "llm_success": False,
            }

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

        out_topk = min(k, len(window))
        n_candidates = len(window)
        system = _SYSTEM_PROMPT.format(topk=out_topk, n_candidates=n_candidates)
        if reformulated_query:
            user_msg = _USER_TEMPLATE_WITH_QUERY.format(
                conversation=conversation_text,
                last_user_message=last_user_msg,
                reformulated_query=reformulated_query,
                candidates_text=candidates_text,
                topk=out_topk,
                n_candidates=n_candidates,
            )
        else:
            user_msg = _USER_TEMPLATE.format(
                conversation=conversation_text,
                last_user_message=last_user_msg,
                candidates_text=candidates_text,
                topk=out_topk,
                n_candidates=n_candidates,
            )

        # Under fail-fast mode (fallback_on_error=False), silent truncation
        # and short-list parses are catastrophic — they corrupt ranked JSON
        # without surfacing. Propagate a hard-error policy down to the API
        # layer and up through the length check below.
        strict = not self.fallback_on_error
        # Retry provenance (only set if retry is invoked).
        retry_prov: dict = {}
        try:
            raw = call_llm_api(
                system,
                user_msg,
                model=self.model,
                max_tokens=4096,
                strict_no_truncation=strict,
            )
            if raw is None:
                raise RuntimeError("claude CLI returned no output")
            reranked, repair_metadata = _parse_candidate_index_response(raw, window, k)
            # Compatibility fallback: older cached responses may still contain
            # track_id UUID arrays. New prompts ask for candidate indices.
            if reranked is None:
                reranked, repair_metadata = _parse_llm_response_with_repairs(raw, valid_ids, k)
            # Short-list detection: if the LLM returned fewer items than we
            # asked for (e.g., truncated mid-array, degenerate parse), treat
            # it as a hard error under fail-fast. The fallback path still
            # tolerates short lists so non-fail-fast callers are unchanged.
            if strict:
                if reranked is None:
                    raise RuntimeError(
                        "LLM reranker parse_failed under fail-fast "
                        "(no JSON array found or all ids invalid)"
                    )
                if len(reranked) < out_topk:
                    logger.warning(
                        "LLM reranker short-list under fail-fast: parsed_valid=%s "
                        "requested=%s valid_window=%s model=%s repairs=%s raw_prefix=%r",
                        len(reranked),
                        out_topk,
                        len(valid_ids),
                        self.model,
                        repair_metadata,
                        raw[:1000],
                    )
                    # Single retry-on-invalid-index (Option C). Only fires on
                    # candidate-index path with non-empty invalid indices.
                    invalid_indices = repair_metadata.get("invalid_candidate_indices", [])
                    if (
                        repair_metadata.get("response_format") == "candidate_indices"
                        and invalid_indices
                    ):
                        retry_system = (
                            system
                            + "\n\nRETRY CORRECTIVE FEEDBACK:\n"
                            f"Your previous response contained invalid candidate number(s): "
                            f"{invalid_indices}. "
                            f"Return exactly {out_topk} integers, each between 1 and "
                            f"{n_candidates}, no duplicates. Output only the JSON array; "
                            f"no prose."
                        )
                        retry_user = (
                            user_msg
                            + "\n\nYou previously returned invalid candidate number(s): "
                            f"{invalid_indices}. "
                            f"Return exactly {out_topk} integers, each between 1 and "
                            f"{n_candidates}, no duplicates. Output only the JSON array; "
                            f"no prose."
                        )
                        retry_raw = call_llm_api(
                            retry_system,
                            retry_user,
                            model=self.model,
                            max_tokens=4096,
                            strict_no_truncation=strict,
                        )
                        logger.info(
                            "LLM reranker retry after invalid indices: %s, new raw=%r",
                            invalid_indices,
                            (retry_raw or "")[:500],
                        )
                        retry_prov = {
                            "retry_invoked": True,
                            "retry_reason": list(invalid_indices),
                            "retry_succeeded": False,
                        }
                        if retry_raw is None:
                            raise RuntimeError(
                                f"LLM reranker retry returned no output; "
                                f"original parsed list length {len(reranked)} < "
                                f"requested top_k {out_topk} under fail-fast"
                            )
                        retry_reranked, retry_repair = _parse_candidate_index_response(
                            retry_raw, window, k
                        )
                        if (
                            retry_reranked is not None
                            and len(retry_reranked) == out_topk
                            and not retry_repair.get("invalid_candidate_indices")
                        ):
                            retry_prov["retry_succeeded"] = True
                            reranked = retry_reranked
                            repair_metadata = retry_repair
                        else:
                            raise RuntimeError(
                                f"LLM reranker retry also short-listed or failed "
                                f"(retry parsed length "
                                f"{len(retry_reranked) if retry_reranked else 0}, "
                                f"retry invalid="
                                f"{retry_repair.get('invalid_candidate_indices', [])}); "
                                f"original parsed list length {len(reranked)} < "
                                f"requested top_k {out_topk} under fail-fast"
                            )
                    else:
                        raise RuntimeError(
                            f"LLM reranker parsed list length {len(reranked)} < "
                            f"requested top_k {out_topk} under fail-fast"
                        )
            if reranked:
                # Append any candidates not returned by LLM (preserve coverage).
                # Remainder is in the ORIGINAL retrieval order of the rerank
                # input window (after already-played reordering). This is the
                # same behavior that existed before pool exposure was added.
                reranked_set = set(reranked)
                remaining = [c for c in candidates if c not in reranked_set]
                combined = reranked + remaining
                # Final safety dedup (preserving order)
                seen_final: set[str] = set()
                deduped: list[str] = []
                for x in combined:
                    if x not in seen_final:
                        seen_final.add(x)
                        deduped.append(x)
                provenance = {
                    "reranker_class": type(self).__name__,
                    "llm_reranked_prefix": len(reranked),
                    "pool_size": len(deduped),
                    "window_size": self.window_size,
                    "remainder_source": "rerank_input_window_retrieval_order",
                    "llm_success": True,
                    **repair_metadata,
                    **retry_prov,
                }
                return deduped[:k], deduped, provenance
        except LLMTruncationError:
            # Never swallow truncation under fail-fast — it indicates the
            # response was cut off mid-JSON and any partial parse is unsafe.
            if strict:
                raise
            logger.warning("LLMListwiseReranker truncated; using fallback order")
        except Exception as e:
            if strict:
                raise
            logger.warning("LLMListwiseReranker failed: %s", e)

        if self.fallback_on_error:
            # Fallback: no LLM ranking happened — pool is just the input window.
            fallback_pool = list(candidates)
            provenance = {
                "reranker_class": type(self).__name__,
                "llm_reranked_prefix": 0,
                "pool_size": len(fallback_pool),
                "window_size": self.window_size,
                "remainder_source": "fallback_retrieval_order_only",
                "llm_success": False,
                "near_miss_salvage_count": 0,
                "near_miss_salvages": [],
            }
            return candidates[:k], fallback_pool, provenance
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

    def batch_rerank_with_pool(
        self,
        batch_candidates: list[list[str]],
        batch_session_memory: list[list[dict]],
        item_db,
        topk: int | None = None,
        reformulated_queries: list[str] | None = None,
    ) -> list[tuple[list[str], list[str], dict]]:
        """Batched version of rerank_with_pool. Returns per-row (top_k, pool, provenance)."""
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    self.rerank_with_pool, cands, mem, item_db, topk,
                    reformulated_queries[i] if reformulated_queries else None,
                )
                for i, (cands, mem) in enumerate(zip(batch_candidates, batch_session_memory))
            ]
            return [f.result() for f in futures]
