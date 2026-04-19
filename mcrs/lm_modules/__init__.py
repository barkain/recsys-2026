from mcrs.lm_modules.claude import ClaudeModule
from mcrs.lm_modules.llama import LlamaModule


def load_lm_module(lm_type: str, **kwargs):
    if lm_type.startswith("claude") or lm_type.startswith("anthropic"):
        return ClaudeModule(model=lm_type)
    else:
        device = kwargs.get("device", "cuda")
        dtype = kwargs.get("dtype", None)
        attn = kwargs.get("attn_implementation", "eager")
        return LlamaModule(model_name=lm_type, device=device, attn_implementation=attn, dtype=dtype)


__all__ = ["ClaudeModule", "LlamaModule", "load_lm_module"]
