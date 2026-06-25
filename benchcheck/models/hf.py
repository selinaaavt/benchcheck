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
import os


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
        # On CPU, torch sometimes defaults to a single thread; use all cores
        # unless the user pinned it via OMP_NUM_THREADS / torch threads already.
        if self.device == "cpu" and "OMP_NUM_THREADS" not in os.environ:
            try:
                ncpu = os.cpu_count() or 1
                if torch.get_num_threads() < ncpu:
                    torch.set_num_threads(ncpu)
            except Exception:
                pass
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Many causal LMs (e.g. gpt2) ship without a pad token; needed for
        # batched scoring. Left-pad so the final token positions stay aligned.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def logprob(self, text: str) -> float:  # pragma: no cover - needs real model
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        if ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask).logits
        # Shift: predict token t from tokens < t.
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_lp.sum().item())

    def logprob_batch(self, texts: list[str]) -> list[float]:  # pragma: no cover
        """Score many texts in ONE padded forward pass.

        This is the throughput lever: the shuffle signal scores ~16 rendered
        blocks per item, and doing them one-at-a-time wastes the GPU/CPU's batch
        parallelism. We left-pad, run a single batched forward, then mask out
        padding when summing per-token logprobs so padding contributes nothing.
        """
        torch = self._torch
        if not texts:
            return []
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        # Models with absolute position embeddings (e.g. gpt2) need position_ids
        # that ignore left padding -- otherwise real tokens get shifted position
        # embeddings and their logprobs diverge from the unpadded computation.
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask, position_ids=position_ids).logits
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # A predicted token counts only if BOTH it and the token it's predicted
        # from are real (non-pad). With left padding, the first real token is
        # predicted from a pad token, so it must be excluded -- matching the
        # single-sequence logprob(), where the first token has no prior.
        valid = (mask[:, 1:] * mask[:, :-1]).to(token_lp.dtype)
        summed = (token_lp * valid).sum(dim=1)
        return [float(x) for x in summed.tolist()]

    def complete(self, prefix: str, max_new_tokens: int = 64) -> str:  # pragma: no cover
        torch = self._torch
        enc = self.tokenizer(prefix, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
