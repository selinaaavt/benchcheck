"""Real model backend backed by HuggingFace transformers.

Optional: importing this module does NOT require torch/transformers until you
actually construct an `HFModel`. That keeps the rest of the tool runnable with
only numpy/scipy installed (via the mock backend).

Install when you want real models:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers

Then point the CLI at e.g. `--model hf:gpt2` or any causal LM you can run.
"""
from __future__ import annotations

import math


class HFModel:
    """Implements Scorer and Generator over a causal LM.

    Loaded lazily so that a missing torch/transformers install produces a clear
    error only when someone actually asks for a real model.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "HFModel needs torch + transformers. Install them, or use the "
                "mock backend (--model mock) which needs neither."
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def logprob(self, text: str) -> float:  # pragma: no cover - needs real model
        torch = self._torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = self.model(ids).logits
        # Shift: predict token t from tokens < t.
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum().item())

    def complete(self, prefix: str, max_new_tokens: int = 64) -> str:  # pragma: no cover
        torch = self._torch
        ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
