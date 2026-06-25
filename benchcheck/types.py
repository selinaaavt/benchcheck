"""Core data types shared across signals.

The whole tool is built around two ideas:
  - an `Item` is one benchmark question we want to judge.
  - a `SignalResult` is one weak piece of evidence about whether the model
    memorized that item. We collect several and combine them later.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    """One benchmark question.

    `id`       - stable identifier so we can join signals back together.
    `prompt`   - the question text as it appears in the benchmark.
    `answer`   - the canonical answer (optional; some signals don't need it).
    `choices`  - multiple-choice options, if any (used by the shuffle signal).
    """
    id: str
    prompt: str
    answer: str | None = None
    choices: list[str] = field(default_factory=list)


@dataclass
class SignalResult:
    """One signal's verdict on one item.

    `score` is normalized so that HIGHER = MORE SUSPICIOUS (more likely the
    item was memorized / contaminated). Each signal documents its own scale.
    `detail` holds raw numbers for debugging and the eventual writeup.
    """
    item_id: str
    signal: str
    score: float
    detail: dict = field(default_factory=dict)
