"""Claude API response generation module."""
import os
import anthropic


class ClaudeModule:
    """Response generation using Claude via the Anthropic API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
        self.client = anthropic.Anthropic(api_key=resolved_key)

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list[dict],
        recommend_item: str,
        max_new_tokens: int = 256,
    ) -> str:
        messages = list(chat_history)
        messages.append({"role": "assistant", "content": recommend_item})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            system=sys_prompt,
            messages=messages,
        )
        return response.content[0].text

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
