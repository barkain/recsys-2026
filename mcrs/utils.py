"""Shared utilities for the Echo CRS system."""
import hashlib
import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

_ANTHROPIC_KEY_ENV = "ANTHROPIC_RECSYS_API_KEY"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"
_CACHE_ENV = "MCRS_LLM_CACHE_DIR"
_DISABLE_CACHE_ENV = "MCRS_DISABLE_LLM_CACHE"
# When set, raise instead of making a paid API call on cache miss. Used by
# smoke/regression runs that must not spend new tokens.
_REQUIRE_CACHE_ENV = "MCRS_REQUIRE_LLM_CACHE"


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "text-", "chatgpt"))


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("o1", "o3", "o4"))


# Floor for OpenAI reasoning models (o1/o3/o4): reasoning tokens are counted
# against max_completion_tokens and can consume several thousands of tokens
# before any visible output is emitted, especially on rerank-style prompts
# with ~20 candidates. We need enough headroom for reasoning + JSON completion.
# Keep non-reasoning callers untouched.
_REASONING_MAX_TOKENS_FLOOR = 16384


class LLMTruncationError(RuntimeError):
    """Raised when an LLM response was truncated (stop_reason=max_tokens /
    finish_reason=length) and the caller requested strict_no_truncation.

    Silent truncation is catastrophic for listwise rerank JSON — partial
    arrays unpickle as shorter-than-requested lists and silently corrupt
    ranked output. Callers under fail-fast mode must treat this as a hard
    error, not a fallback case.
    """


def call_llm_api(
    system: str,
    user,
    model: str,
    max_tokens: int = 1024,
    strict_no_truncation: bool = False,
) -> str | None:
    """Call either the Anthropic or OpenAI API based on the model name.

    Routes to OpenAI for models starting with gpt-/o1/o3/o4, otherwise Anthropic.
    API keys read from OPENAI_API_KEY and ANTHROPIC_RECSYS_API_KEY respectively.

    When `strict_no_truncation=True`, raises LLMTruncationError instead of
    returning a truncated string when the provider signals it ran out of
    output budget (stop_reason=max_tokens / finish_reason=length). Cached
    entries are NOT validated against this signal — callers that require
    clean output should combine this with cache invalidation when the cache
    predates the stricter policy.
    """
    if _is_reasoning_model(model):
        max_tokens = max(max_tokens, _REASONING_MAX_TOKENS_FLOOR)
    cache_path = _llm_cache_path(system, user, model, max_tokens)
    cached = _read_llm_cache(cache_path)
    if cached is not None:
        return cached

    # Cache-hit-only fence: when MCRS_REQUIRE_LLM_CACHE=1 is set, refuse to
    # make a paid API call. Smoke/regression runs that must not spend tokens
    # should set this env var. Raises a RuntimeError so fail-fast callers
    # surface the miss deterministically.
    if os.environ.get(_REQUIRE_CACHE_ENV):
        raise RuntimeError(
            f"LLM cache miss (model={model}, max_tokens={max_tokens}) while "
            f"{_REQUIRE_CACHE_ENV}=1 is set; refusing to call paid API"
        )

    if _is_openai_model(model):
        result = _call_openai_api(
            system, user, model, max_tokens,
            strict_no_truncation=strict_no_truncation,
        )
    else:
        result = call_claude_api(
            system, user, model, max_tokens,
            strict_no_truncation=strict_no_truncation,
        )

    if result is not None:
        _write_llm_cache(cache_path, result)
    return result


def _llm_cache_path(system: str, user, model: str, max_tokens: int) -> str | None:
    """Return the deterministic cache path for an LLM request, or None if disabled."""
    if os.environ.get(_DISABLE_CACHE_ENV):
        return None
    cache_dir = os.environ.get(_CACHE_ENV, os.path.join("cache", "llm_api"))
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "user": user,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return os.path.join(cache_dir, model.replace("/", "_"), f"{digest}.json")


def _read_llm_cache(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("text")
        return text if isinstance(text, str) else None
    except Exception as e:
        logger.debug("Ignoring unreadable LLM cache entry %s: %s", path, e)
        return None


def _write_llm_cache(path: str | None, text: str) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-", suffix=".json", dir=os.path.dirname(path)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.debug("Failed to write LLM cache entry %s: %s", path, e)


def _call_openai_api(
    system: str,
    user,
    model: str,
    max_tokens: int,
    strict_no_truncation: bool = False,
) -> str | None:
    api_key = (
        os.environ.get("OPENAI_API_KEY_RECSYS")
        or os.environ.get("OPENAI_RECSYS_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        logger.error("%s not set — cannot call OpenAI API", _OPENAI_KEY_ENV)
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # o1/o3/o4 reasoning models don't support system role — use developer role
        if model.startswith(("o1", "o3", "o4")):
            messages = [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ]
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
            choice = resp.choices[0]
            finish = getattr(choice, "finish_reason", None)
            content = choice.message.content
            if finish == "length" or not content:
                logger.warning(
                    "OpenAI reasoning model (%s) returned finish_reason=%s with %d-char content "
                    "at max_completion_tokens=%d — likely reasoning-budget exhaustion",
                    model, finish, len(content or ""), max_tokens,
                )
                if strict_no_truncation and finish == "length":
                    raise LLMTruncationError(
                        f"OpenAI reasoning model {model} truncated output "
                        f"(finish_reason=length, max_completion_tokens={max_tokens})"
                    )
            return content
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
            choice = resp.choices[0]
            finish = getattr(choice, "finish_reason", None)
            if strict_no_truncation and finish == "length":
                raise LLMTruncationError(
                    f"OpenAI model {model} truncated output "
                    f"(finish_reason=length, max_tokens={max_tokens})"
                )
            return choice.message.content
    except LLMTruncationError:
        raise
    except Exception as e:
        logger.warning("OpenAI API call failed (model=%s): %s", model, e)
        return None


def call_claude_api(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 1024,
    strict_no_truncation: bool = False,
) -> str | None:
    """Call the Anthropic API and return the response text, or None on failure.

    Reads the API key from the ANTHROPIC_RECSYS_API_KEY environment variable.

    When strict_no_truncation=True, raises LLMTruncationError if the response
    was truncated (stop_reason=max_tokens).
    """
    api_key = os.environ.get(_ANTHROPIC_KEY_ENV) or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("%s not set — cannot call Claude API", _ANTHROPIC_KEY_ENV)
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        messages = user if isinstance(user, list) else [{"role": "user", "content": user}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        text = message.content[0].text if message.content else None
        if message.stop_reason != "end_turn":
            logger.warning(
                "Claude API stop_reason=%s (model=%s) — response may be truncated",
                message.stop_reason,
                model,
            )
            if strict_no_truncation and message.stop_reason == "max_tokens":
                raise LLMTruncationError(
                    f"Claude model {model} truncated output "
                    f"(stop_reason=max_tokens, max_tokens={max_tokens})"
                )
        return text
    except LLMTruncationError:
        raise
    except Exception as e:
        logger.warning("Claude API call failed (model=%s): %s", model, e)
        return None
