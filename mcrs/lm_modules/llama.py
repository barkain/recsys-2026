"""Llama-based response generation (local HuggingFace model)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaModule:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda",
        attn_implementation: str = "eager",
        dtype=None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self.attn_implementation = attn_implementation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            torch_dtype=self.dtype,
        )
        self.lm.eval()
        self.lm.to(self.device)

    def _format(self, sys_prompt: str, chat_history: list[dict], recommend_item: str) -> str:
        messages = [{"role": "system", "content": sys_prompt}]
        messages += chat_history
        messages += [{"role": "assistant", "content": recommend_item}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list[dict],
        recommend_item: str,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> str:
        text = self._format(sys_prompt, chat_history, recommend_item)
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attn_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.lm.generate(input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list[dict]],
        recommend_items: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        texts = [
            self._format(sys_prompts[i], chat_histories[i], recommend_items[i])
            for i in range(len(sys_prompts))
        ]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        attn_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.lm.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
