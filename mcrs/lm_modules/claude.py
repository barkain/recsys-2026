"""Claude API response generation module."""
import os

from mcrs.utils import call_llm_api


class ClaudeModule:
    """Response generation using Claude via the Anthropic API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        if api_key:
            os.environ.setdefault("ANTHROPIC_RECSYS_API_KEY", api_key)

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list[dict],
        recommend_item: str,
        max_new_tokens: int = 256,
    ) -> str:
        messages = list(chat_history)
        # Anthropic API rejects trailing whitespace on final assistant content.
        safe_item = recommend_item.rstrip() or "(none)"
        messages.append({"role": "assistant", "content": safe_item})
        raw = call_llm_api(
            sys_prompt,
            messages,
            model=self.model,
            max_tokens=max_new_tokens,
        )
        return raw or ""

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list[dict]],
        recommend_items: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        # Claude doesn't have true batching — process sequentially
        return [
            self.response_generation(sys_prompts[i], chat_histories[i], recommend_items[i], max_new_tokens)
            for i in range(len(sys_prompts))
        ]
