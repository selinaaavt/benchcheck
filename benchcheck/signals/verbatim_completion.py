"""Signal: verbatim completion (Tier B -- needs only text generation).

Idea: feed the model the first part of a benchmark item and let it continue. A
model that merely understands the topic will produce a plausible-but-different
continuation. A model that *memorized* the item will reproduce the rest of the
original wording word-for-word. We score by how much of the held-out remainder
the model reproduces verbatim.

Score = longest-common-subsequence-style word overlap between the model's
completion and the true remainder, as a fraction of the remainder length.
Higher = more verbatim regurgitation = more suspicious.
"""
from __future__ import annotations

import re

from benchcheck.models.base import Capability
from benchcheck.signals.base import SignalContext, register
from benchcheck.types import Item, SignalResult

_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# A contiguous match shorter than this is not evidence of regurgitation -- a
# couple of shared common words ("how many", "what is the") happens by chance.
MIN_VERBATIM_RUN = 3


def _contiguous_overlap(a: list[str], b: list[str]) -> int:
    """Length of the longest run of `a` that appears contiguously in `b`.
    Verbatim regurgitation shows up as a long contiguous run, which is much
    stronger evidence than scattered shared words. Runs below MIN_VERBATIM_RUN
    are treated as noise (returned as 0)."""
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best if best >= MIN_VERBATIM_RUN else 0


class VerbatimCompletionSignal:
    name = "verbatim_completion"
    required_capability = Capability.GENERATE
    prefix_fraction = 0.4  # show the model this much; hold out the rest

    def score_item(self, item: Item, ctx: SignalContext) -> SignalResult:
        model = ctx.model
        if model is None:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_model"})

        words = item.prompt.split()
        if len(words) < 5:
            return SignalResult(item.id, self.name, 0.0, {"reason": "too_short"})

        cut = max(1, int(len(words) * self.prefix_fraction))
        prefix = " ".join(words[:cut])
        remainder_words = _words(" ".join(words[cut:]))
        if not remainder_words:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_remainder"})

        completion = model.complete(prefix, max_new_tokens=len(remainder_words) + 8)
        comp_words = _words(completion)

        run = _contiguous_overlap(remainder_words, comp_words)
        score = run / len(remainder_words)
        return SignalResult(
            item_id=item.id,
            signal=self.name,
            score=min(1.0, score),
            detail={
                "prefix_words": cut,
                "remainder_words": len(remainder_words),
                "verbatim_run": run,
            },
        )


register(VerbatimCompletionSignal())
