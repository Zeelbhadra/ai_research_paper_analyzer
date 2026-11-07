"""Local instruct-LLM helper (HF transformers, CPU ok) used for synthetic query generation/validation."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, checkpoint: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cpu", dtype: str = "float32"):
        self.dtype = dtype
        self.checkpoint, self.device = checkpoint, device
        self._tok = self._model = None

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.checkpoint)
            self._model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self

    def _prompt_ids(self, messages):
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tok(text, return_tensors="pt").to(self.device)

    def generate(self, messages: list[dict], max_new_tokens: int = 60) -> str:
        """Greedy decoding => deterministic output."""
        import torch
        self.load()
        enc = self._prompt_ids(messages)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def yes_probability(self, messages: list[dict]) -> float:
        """P('Yes') vs P('No') for the first generated token; a cheap calibrated-ish answerability score."""
        import torch
        self.load()
        enc = self._prompt_ids(messages)
        yes = self._tok.encode("Yes", add_special_tokens=False)[0]
        no = self._tok.encode("No", add_special_tokens=False)[0]
        with torch.no_grad():
            logits = self._model(**enc).logits[0, -1]
        p = torch.softmax(logits[[yes, no]], 0)
        return float(p[0])
