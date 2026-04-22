"""Shared utilities for the Echo CRS system."""
import os
import logging

import anthropic

logger = logging.getLogger(__name__)

_API_KEY_ENV = "ANTHROPIC_RECSYS_API_KEY"


def call_claude_api(system: str, user: str, model: str, max_tokens: int = 1024) -> str | None:
    """Call the Anthropic API and return the response text, or None on failure.

    Reads the API key from the ANTHROPIC_RECSYS_API_KEY environment variable.
    """
    api_key = os.environ.get(_API_KEY_ENV)
    if not api_key:
        logger.error("%s not set — cannot call Claude API", _API_KEY_ENV)
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
