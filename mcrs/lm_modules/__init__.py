import os
import logging
from mcrs.lm_modules.claude import ClaudeModule
from mcrs.lm_modules.llama import LlamaModule

_log = logging.getLogger(__name__)

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "text-", "chatgpt")


class NullLM:
    """No-op LM for retrieval-only evaluation (no API key required)."""

    def response_generation(self, sys_prompt, chat_history, recommend_item, max_new_tokens=256):
        return ""

    def batch_response_generation(self, sys_prompts, chat_histories, recommend_items, max_new_tokens=256):
        return [""] * len(sys_prompts)


def load_lm_module(lm_type: str, **kwargs):
    if lm_type in ("null", "none", "retrieval_only"):
        return NullLM()
    if lm_type.startswith("claude") or lm_type.startswith("anthropic"):
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
        if not api_key:
            _log.warning(
                "Neither ANTHROPIC_API_KEY nor ANTHROPIC_RECSYS_API_KEY set — "
                "using NullLM (retrieval-only, no response generation)"
            )
            return NullLM()
        return ClaudeModule(model=lm_type, api_key=api_key)
    if any(lm_type.startswith(p) for p in _OPENAI_PREFIXES):
        from mcrs.lm_modules.openai_module import OpenAIModule
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_RECSYS_API_KEY")
        if not api_key:
            _log.warning("OPENAI_API_KEY not set — using NullLM")
            return NullLM()
        return OpenAIModule(model=lm_type, api_key=api_key)
    device = kwargs.get("device", "cuda")
    dtype = kwargs.get("dtype", None)
    attn = kwargs.get("attn_implementation", "eager")
    return LlamaModule(model_name=lm_type, device=device, attn_implementation=attn, dtype=dtype)


__all__ = ["ClaudeModule", "LlamaModule", "NullLM", "load_lm_module"]
