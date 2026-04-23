"""Shared utilities for the Echo CRS system."""
import os
import logging

import anthropic

logger = logging.getLogger(__name__)

_ANTHROPIC_KEY_ENV = "ANTHROPIC_RECSYS_API_KEY"
_OPENAI_KEY_ENV = "OPENAI_API_KEY"


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "text-", "chatgpt"))


def call_llm_api(system: str, user: str, model: str, max_tokens: int = 1024) -> str | None:
    """Call either the Anthropic or OpenAI API based on the model name.

    Routes to OpenAI for models starting with gpt-/o1/o3/o4, otherwise Anthropic.
    API keys read from OPENAI_API_KEY and ANTHROPIC_RECSYS_API_KEY respectively.
    """
    if _is_openai_model(model):
        return _call_openai_api(system, user, model, max_tokens)
    return call_claude_api(system, user, model, max_tokens)


def _call_openai_api(system: str, user: str, model: str, max_tokens: int) -> str | None:
    api_key = os.environ.get(_OPENAI_KEY_ENV) or os.environ.get("OPENAI_RECSYS_API_KEY")
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
            resp = client.chat.completions.create(model=model, messages=messages)
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens
            )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning("OpenAI API call failed (model=%s): %s", model, e)
        return None


def call_claude_api(system: str, user: str, model: str, max_tokens: int = 1024) -> str | None:
    """Call the Anthropic API and return the response text, or None on failure.

    Reads the API key from the ANTHROPIC_RECSYS_API_KEY environment variable.
    """
    api_key = os.environ.get(_ANTHROPIC_KEY_ENV) or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("%s not set — cannot call Claude API", _ANTHROPIC_KEY_ENV)
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = message.content[0].text if message.content else None
        if message.stop_reason != "end_turn":
            logger.warning(
                "Claude API stop_reason=%s (model=%s) — response may be truncated",
                message.stop_reason,
                model,
            )
        return text
    except Exception as e:
        logger.warning("Claude API call failed (model=%s): %s", model, e)
        return None
