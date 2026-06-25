"""Signal interface + registry.

Every contamination signal is a class implementing `Signal`. The uniform
contract is what lets the aggregator combine them without special-casing:

  - `name`: stable identifier used in output and config.
  - `required_capability`: None, "score", or "generate" -- the model capability
    the signal needs. The pipeline skips signals whose capability is absent
    (graceful degradation), instead of crashing.
  - `score_item(item, ctx)` -> SignalResult with score where HIGHER = MORE
    SUSPICIOUS, normalized to roughly [0, 1] so signals are comparable.

`SignalContext` carries shared resources (the model, the reference corpus, etc.)
so signals stay stateless and easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from benchcheck.types import Item, SignalResult


@dataclass
class SignalContext:
    """Shared inputs handed to every signal during a run."""

    model: object | None = None
    model_capabilities: set[str] = field(default_factory=set)
    corpus_ngrams: set | None = None
    corpus_ngram_n: int = 8
    extra: dict = field(default_factory=dict)


class Signal(Protocol):
    name: str
    required_capability: str | None

    def score_item(self, item: Item, ctx: SignalContext) -> SignalResult: ...


_REGISTRY: dict[str, Signal] = {}


def register(signal: Signal) -> Signal:
    """Register a signal instance so the CLI can select it by name."""
    _REGISTRY[signal.name] = signal
    return signal


def get(name: str) -> Signal:
    return _REGISTRY[name]


def all_signals() -> list[Signal]:
    return list(_REGISTRY.values())


def available_for(capabilities: set[str]) -> list[Signal]:
    """Signals whose required capability is satisfied by the given model."""
    out = []
    for s in _REGISTRY.values():
        if s.required_capability is None or s.required_capability in capabilities:
            out.append(s)
    return out
