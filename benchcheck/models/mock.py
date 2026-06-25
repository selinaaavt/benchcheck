"""A deterministic, dependency-free mock model.

Why this exists: the full pipeline -- including Tier A logprob signals and the
calibration experiment -- must run without downloading a multi-GB model. The
mock simulates the ONE behavior the whole tool is built to detect:
*memorization*. A real contaminated model assigns unusually high probability to
text it has seen verbatim, and will happily regurgitate it. The mock does the
same, in a controlled, reproducible way.

It is NOT a language model. It's a stand-in whose statistical behavior matches
the phenomenon under test, so we can validate the detector's machinery end to
end before pointing it at a real model.
"""
from __future__ import annotations

import hashlib
import math
import re

_WORD = re.compile(r"\w+")


def _stable_hash(text: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from text (no Math.random)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class MockModel:
    """Implements both Scorer and Generator.

    `memorized` is the set of strings the model has "seen during training".
    For those, it returns inflated logprob and verbatim completions -- exactly
    the contamination fingerprint the signals hunt for.

    `per_token_base` is the average log-prob per token for unseen text; a mild
    deterministic jitter keeps clean items from being suspiciously uniform.
    `memorization_boost` is how much logprob/token improves for seen text.
    """

    def __init__(
        self,
        memorized: list[str] | None = None,
        per_token_base: float = -2.5,
        memorization_boost: float = 1.8,
    ) -> None:
        self._memorized_norm = {_norm(m): m for m in (memorized or [])}
        self.per_token_base = per_token_base
        self.memorization_boost = memorization_boost

    # --- helpers -----------------------------------------------------------
    def _is_memorized(self, text: str) -> bool:
        n = _norm(text)
        if n in self._memorized_norm:
            return True
        # Substring match: a memorized item embedded in a larger prompt still
        # leaks. Mirrors real contamination where the question is part of a doc.
        return any(n and n in mem_norm or mem_norm in n for mem_norm in self._memorized_norm)

    # --- Scorer ------------------------------------------------------------
    def logprob(self, text: str) -> float:
        tokens = _WORD.findall(text)
        if not tokens:
            return 0.0
        jitter = (_stable_hash(text) - 0.5) * 0.1  # +/-0.05 nats/token, stable
        per_token = self.per_token_base + jitter
        if self._is_memorized(text):
            per_token += self.memorization_boost
        return per_token * len(tokens)

    # --- Generator ---------------------------------------------------------
    def complete(self, prefix: str, max_new_tokens: int = 64) -> str:
        n_prefix = _norm(prefix)
        # If the prefix is the start of something memorized, regurgitate the
        # remembered remainder verbatim -- the verbatim-completion tell.
        for mem_norm, mem_original in self._memorized_norm.items():
            if mem_norm.startswith(n_prefix) and len(n_prefix) < len(mem_norm):
                remainder = mem_original[len(prefix):]
                words = remainder.split()
                return " ".join(words[:max_new_tokens]) if words else remainder
        # Otherwise produce deterministic filler from sentinel tokens that never
        # appear in real prompts, so an unmemorized item shows ZERO verbatim
        # overlap (no accidental matches on common words like "the"/"of").
        out = [f"zzq{(int(_stable_hash(prefix) * 997) + i) % 97}" for i in range(8)]
        return " ".join(out[: min(8, max_new_tokens)])
