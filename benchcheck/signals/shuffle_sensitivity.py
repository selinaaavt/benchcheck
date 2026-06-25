"""Signal: multiple-choice shuffle sensitivity (Tier A -- needs token logprobs).

Idea: for a multiple-choice item, genuine understanding is invariant to the
ORDER the options are presented in -- the model should prefer the correct answer
regardless of whether it's listed first or last. A model that memorized the item
(e.g. "the answer is option B") is sensitive to the canonical ordering: shuffle
the options and its confidence in the correct answer wobbles.

We measure that wobble. For the original ordering and several deterministic
shuffles, we compute the model's logprob of each rendered choice block and look
at how much the *probability assigned to the correct answer* varies across
orderings. High variance = order-dependent = memorization-flavored.

Only applies to items that actually have choices + a known answer; otherwise the
signal abstains (returns score 0 with a reason).
"""
from __future__ import annotations

import math
import re

from benchcheck.models.base import Capability
from benchcheck.perturb import shuffled_choice_orders
from benchcheck.signals.base import SignalContext, register
from benchcheck.types import Item, SignalResult

_LETTERS = "ABCDEFGH"
_WORD = re.compile(r"\w+")


def _render(prompt: str, ordered_choices: list[str]) -> str:
    lines = [prompt]
    for letter, choice in zip(_LETTERS, ordered_choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def _correct_index(item: Item) -> int | None:
    if item.answer is None:
        return None
    # answer may be the letter ("B"), or the full choice text.
    ans = item.answer.strip()
    if len(ans) == 1 and ans.upper() in _LETTERS:
        idx = _LETTERS.index(ans.upper())
        return idx if idx < len(item.choices) else None
    for i, c in enumerate(item.choices):
        if c.strip().lower() == ans.lower():
            return i
    return None


def _score_blocks(model, blocks: list[str]) -> list[float]:
    """Score a list of rendered blocks, using the model's batched scorer when
    available (big throughput win) and falling back to per-block scoring."""
    batch = getattr(model, "logprob_batch", None)
    if callable(batch):
        return batch(blocks)
    return [model.logprob(b) for b in blocks]


def _softmax_prob(lps: list[float], pos: int) -> float:
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    total = sum(exps)
    return exps[pos] / total if total > 0 else 0.0


class ShuffleSensitivitySignal:
    name = "shuffle_sensitivity"
    required_capability = Capability.SCORE

    def score_item(self, item: Item, ctx: SignalContext) -> SignalResult:
        model = ctx.model
        if model is None:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_model"})
        if len(item.choices) < 2:
            return SignalResult(item.id, self.name, 0.0, {"reason": "not_mcq"})
        correct = _correct_index(item)
        if correct is None:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_answer"})

        orders = [list(range(len(item.choices)))] + shuffled_choice_orders(item.choices)

        # Build every rendered block across all orderings up front, then score
        # them in ONE batched pass. This is the throughput optimization: instead
        # of ~16 separate forward passes per item we do a single batch.
        blocks: list[str] = []
        layout: list[tuple[int, int]] = []  # (order_idx, correct_pos) per block
        for oi, order in enumerate(orders):
            ordered = [item.choices[i] for i in order]
            correct_pos = order.index(correct)
            rendered = _render(item.prompt, ordered)
            for i in range(len(ordered)):
                blocks.append(rendered + f"\nAnswer: {_LETTERS[i]}. {ordered[i]}")
                layout.append((oi, correct_pos))

        all_lps = _score_blocks(model, blocks)

        # Regroup per-ordering and compute the prob assigned to the correct option.
        probs = []
        for oi in range(len(orders)):
            lps = [all_lps[k] for k in range(len(blocks)) if layout[k][0] == oi]
            correct_pos = next(layout[k][1] for k in range(len(blocks)) if layout[k][0] == oi)
            probs.append(_softmax_prob(lps, correct_pos))

        mean_p = sum(probs) / len(probs)
        var = sum((p - mean_p) ** 2 for p in probs) / len(probs)
        std = math.sqrt(var)
        # Std of a probability is bounded by 0.5; scale to [0,1].
        score = min(1.0, std / 0.5)
        return SignalResult(
            item_id=item.id,
            signal=self.name,
            score=score,
            detail={
                "n_orders": len(orders),
                "prob_correct_per_order": [round(p, 4) for p in probs],
                "std": round(std, 4),
            },
        )


register(ShuffleSensitivitySignal())
