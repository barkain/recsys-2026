import os
from mcrs.lm_modules.claude import ClaudeModule
from mcrs.lm_modules.llama import LlamaModule


class NullLM:
    """No-op LM for retrieval-only evaluation (no API key required)."""

    def response_generation(self, sys_prompt, chat_history, recommend_item, max_new_tokens=256):
        return ""

    def batch_response_generation(self, sys_prompts, chat_histories, recommend_items, max_new_tokens=256):
        return [""] * len(sys_prompts)


def load_lm_module(lm_type: str, **kwargs):
    if lm_type.startswith("claude") or lm_type.startswith("anthropic"):
        # Accept either ANTHROPIC_API_KEY or the project-specific ANTHROPIC_RECSYS_API_KEY
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
        if not api_key:
            import logging
            logging.getLogger(__name__).warning(
                "Neither ANTHROPIC_API_KEY nor ANTHROPIC_RECSYS_API_KEY set — "
                "using NullLM (retrieval-only, no response generation)"
            )
            return NullLM()
        return ClaudeModule(model=lm_type, api_key=api_key)
    else:
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", None)
        attn = kwargs.get("attn_implementation", "eager")
        return LlamaModule(model_name=lm_type, device=device, attn_implementation=attn, dtype=dtype)


__all__ = ["ClaudeModule", "LlamaModule", "NullLM", "load_lm_module"]
