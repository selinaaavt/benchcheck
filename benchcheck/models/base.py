"""Model backend abstraction.

Signals shouldn't care whether they're talking to a 7B-parameter model on a GPU
or a deterministic stand-in used for tests. They depend only on this interface.

Two capabilities, deliberately separated so the tool can degrade gracefully:

  - `Scorer.logprob(text)` -> total log-probability the model assigns to `text`.
    This is the strong capability; only open-weight models you run yourself
    expose it. Tier A signals require it.

  - `Generator.complete(prefix, max_new_tokens)` -> the model's continuation.
    Almost any model (including hosted APIs) can do this. Tier B signals use it.

A backend may implement one or both. Signals check `supports()` and skip
themselves when their required capability is missing -- that's the graceful
degradation story.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """Can return the log-probability of a given string (teacher forcing)."""

    def logprob(self, text: str) -> float:
        """Sum of log p(token | preceding tokens) over `text`. Higher = the
        model finds `text` more probable / less surprising."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Can continue a prefix with greedy / low-temperature decoding."""

    def complete(self, prefix: str, max_new_tokens: int = 64) -> str:
        """Return the model's continuation of `prefix` (not including prefix)."""
        ...


class Capability:
    """String constants for capability checks, to avoid typos."""

    SCORE = "score"
    GENERATE = "generate"


def capabilities_of(model: object) -> set[str]:
    """Inspect which capabilities a backend actually provides."""
    caps: set[str] = set()
    if isinstance(model, Scorer):
        caps.add(Capability.SCORE)
    if isinstance(model, Generator):
        caps.add(Capability.GENERATE)
    return caps
