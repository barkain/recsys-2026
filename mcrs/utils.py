"""Shared utilities for the Echo CRS system."""
import logging
import subprocess

logger = logging.getLogger(__name__)


def call_claude_cli(system: str, user: str, model: str, timeout: int = 30) -> str | None:
    """Call the claude CLI and return the response text, or None on failure."""
    prompt = f"{system}\n\n{user}"
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", model],
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
        logger.warning("claude CLI timed out after %ss (model=%s)", timeout, model)
        return None
    except Exception as e:
        logger.warning("claude CLI failed: %s", e)
        return None
