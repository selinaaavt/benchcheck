"""Signal: perturbation likelihood gap (Tier A -- needs token logprobs).

This is the flagship signal. The reasoning:

  - A model assigns a log-probability to any string (how "unsurprising" it is).
  - For a question the model UNDERSTANDS, the original wording and a
    meaning-preserving paraphrase are about equally probable.
  - For a question the model MEMORIZED, the *exact original wording* is far more
    probable than any paraphrase -- the model has seen that precise string.

So we compare the model's per-token logprob on the original vs. its paraphrases
(the item's control distribution). A large positive gap = the original is
special to the model = memorization.

We normalize by token count (per-token logprob) because longer strings have
lower total logprob purely by length -- comparing raw totals would confound
length with memorization. We then map the gap through a squashing function to
[0, 1] so it's comparable with the other signals.
"""
from __future__ import annotations

import math
import re

from benchcheck.models.base import Capability
from benchcheck.perturb import paraphrases
from benchcheck.signals.base import SignalContext, register
from benchcheck.types import Item, SignalResult

_WORD = re.compile(r"\w+")


def _per_token_logprob(model, text: str) -> float:
    ntok = len(_WORD.findall(text))
    if ntok == 0:
        return 0.0
    return model.logprob(text) / ntok


def _squash(gap: float, scale: float = 0.5) -> float:
    """Map an unbounded logprob gap (nats/token) to [0, 1).
    A gap of ~`scale` nats/token lands around 0.46; large gaps approach 1.
    Negative gaps (original LESS probable than paraphrase) -> ~0, not suspicious.
    """
    if gap <= 0:
        return 0.0
    return 1.0 - math.exp(-gap / scale)


class PerturbationGapSignal:
    name = "perturbation_gap"
    required_capability = Capability.SCORE

    def score_item(self, item: Item, ctx: SignalContext) -> SignalResult:
        model = ctx.model
        if model is None:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_model"})

        controls = paraphrases(item.prompt)
        if not controls:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_controls"})

        orig_lp = _per_token_logprob(model, item.prompt)
        control_lps = [_per_token_logprob(model, c) for c in controls]
        mean_control = sum(control_lps) / len(control_lps)

        gap = orig_lp - mean_control  # positive => original is "special"
        score = _squash(gap)
        return SignalResult(
            item_id=item.id,
            signal=self.name,
            score=score,
            detail={
                "orig_logprob_per_tok": round(orig_lp, 4),
                "mean_control_logprob_per_tok": round(mean_control, 4),
                "gap": round(gap, 4),
                "n_controls": len(controls),
            },
        )


register(PerturbationGapSignal())
