"""Perturbations: meaning-preserving rewrites used as statistical controls.

The central trick of the whole tool: a model that *understands* a question
treats a paraphrase about as well as the original. A model that *memorized* the
original is suspiciously attached to its exact wording. To measure that, we need
ways to rewrite an item that preserve meaning but break verbatim memorization.

These perturbations are intentionally simple, deterministic, and dependency-free
so results are reproducible (no Math.random / no LLM paraphraser in the loop).
Each is documented with what memorization fingerprint it's designed to expose.

A note on honesty: synthetic paraphrases are weaker than human ones. The tool
treats each original item's perturbations as that item's *control distribution*;
we never claim a single perturbation is a perfect meaning-preserving twin.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"\w+")

# Small, safe synonym swaps that preserve meaning in most contexts. Kept
# conservative on purpose -- a wrong swap would corrupt the control.
_SYNONYMS = {
    "leaves": "departs",
    "another": "a second",
    "when": "at what time",
    "how many": "what number of",
    "total": "altogether",
    "find": "determine",
    "compute": "calculate",
    "begins": "starts",
    "rapid": "fast",
    "large": "big",
    "small": "tiny",
}


def whitespace_normalize(text: str) -> str:
    """A near-identity perturbation: collapse/normalize whitespace only.

    This is the *negative control* -- it changes the surface string trivially
    without changing wording. A signal should NOT flag this as different from
    the original. Useful to detect signals that are over-sensitive to noise.
    """
    return " ".join(text.split())


def synonym_paraphrase(text: str) -> str:
    """Swap a few words for synonyms. Breaks exact-match memorization while
    preserving meaning. Exposes `perturbation_gap`: a memorizing model loses
    much more confidence here than an understanding model would."""
    out = text
    for phrase, repl in _SYNONYMS.items():
        out = re.sub(rf"\b{re.escape(phrase)}\b", repl, out, flags=re.IGNORECASE)
    return out


def clause_reorder(text: str) -> str:
    """Reorder around a comma when safe (e.g. 'If A, then B' -> 'B, given A').
    A coarse syntactic paraphrase; only applied when it yields a sane result."""
    if text.count(",") == 1:
        left, right = (s.strip() for s in text.split(","))
        if left and right:
            return f"{right}, given that {left[0].lower()}{left[1:]}"
    return text


def paraphrases(text: str) -> list[str]:
    """Return the control set for an item: several meaning-preserving rewrites.
    De-duplicated, and we never return one identical to the original (that would
    be a useless control)."""
    candidates = [
        synonym_paraphrase(text),
        clause_reorder(text),
        synonym_paraphrase(clause_reorder(text)),
    ]
    seen: set[str] = set()
    out: list[str] = []
    norm_orig = whitespace_normalize(text).lower()
    for c in candidates:
        c = whitespace_normalize(c)
        if c.lower() != norm_orig and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def shuffled_choice_orders(choices: list[str], max_orders: int = 3) -> list[list[int]]:
    """Deterministic permutations of multiple-choice option indices.

    Returns lists of indices (not the choices themselves) so callers can track
    where the correct answer moved. Used by `shuffle_sensitivity`: genuine
    understanding is invariant to option order; memorization is not.
    """
    n = len(choices)
    if n < 2:
        return []
    base = list(range(n))
    orders: list[list[int]] = []
    # Deterministic rotations + a reversal. No RNG -> reproducible.
    for shift in range(1, min(max_orders, n) + 1):
        orders.append([(i + shift) % n for i in base])
    rev = list(reversed(base))
    if rev not in orders:
        orders.append(rev)
    return orders[:max_orders]
