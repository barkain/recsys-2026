"""OpenAI API response generation module."""
import os
import openai


class OpenAIModule:
    """Response generation using OpenAI models (gpt-4o, o4-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_RECSYS_API_KEY")
        self.client = openai.OpenAI(api_key=resolved_key)
        # o1/o3/o4 reasoning models use developer role instead of system role
        self._is_reasoning = model.startswith(("o1", "o3", "o4"))

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list[dict],
        recommend_item: str,
        max_new_tokens: int = 256,
    ) -> str:
        messages = []
        if self._is_reasoning:
            messages.append({"role": "developer", "content": sys_prompt})
        else:
            messages.append({"role": "system", "content": sys_prompt})
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "assistant", "content": recommend_item})

        kwargs = {"model": self.model, "messages": messages}
        if not self._is_reasoning:
            kwargs["max_tokens"] = max_new_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list[dict]],
        recommend_items: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        return [
            self.response_generation(sys_prompts[i], chat_histories[i], recommend_items[i], max_new_tokens)
            for i in range(len(sys_prompts))
        ]
