"""Signal: n-gram overlap against a reference corpus.

Idea (the cheap fallback, needs NO model): if a benchmark question's exact
wording shows up in a corpus the model was plausibly trained on, that's
circumstantial evidence of contamination. We measure what fraction of the
question's distinct n-grams also appear in the reference corpus.

Weak alone -- common phrases overlap with everything -- so we use long n-grams
(default n=8), which rarely collide by chance. Higher score = more distinctive
wording found in the corpus = more suspicious.
"""
from __future__ import annotations

import re

from benchcheck.signals.base import SignalContext, register
from benchcheck.types import Item, SignalResult

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_corpus_ngrams(corpus_texts: list[str], n: int) -> set[tuple[str, ...]]:
    """Pre-compute the set of n-grams present in the reference corpus once."""
    seen: set[tuple[str, ...]] = set()
    for text in corpus_texts:
        seen |= _ngrams(_tokens(text), n)
    return seen


class NgramOverlapSignal:
    name = "ngram_overlap"
    required_capability = None  # no model needed

    def score_item(self, item: Item, ctx: SignalContext) -> SignalResult:
        n = ctx.corpus_ngram_n
        # Preferred path: a prebuilt corpus index (C++ or Python backend),
        # which exposes .overlap() directly.
        if getattr(ctx, "corpus_index", None) is not None:
            frac = ctx.corpus_index.overlap(item.prompt)
            return SignalResult(
                item_id=item.id,
                signal=self.name,
                score=frac,
                detail={"n": n, "backend": getattr(ctx.corpus_index, "backend", "?")},
            )
        # Legacy path: a raw set of corpus n-grams (kept for back-compat tests).
        corpus = ctx.corpus_ngrams
        if corpus is None:
            return SignalResult(item.id, self.name, 0.0, {"reason": "no_corpus"})
        item_ngrams = _ngrams(_tokens(item.prompt), n)
        if not item_ngrams:
            return SignalResult(item.id, self.name, 0.0, {"reason": "empty"})
        matched = sum(1 for g in item_ngrams if g in corpus)
        frac = matched / len(item_ngrams)
        return SignalResult(
            item_id=item.id,
            signal=self.name,
            score=frac,
            detail={"n": n, "matched": matched, "total": len(item_ngrams)},
        )


# Backwards-compat helper used by the original demo/tests.
def score_item(item: Item, corpus_ngrams: set, n: int = 8) -> SignalResult:
    ctx = SignalContext(corpus_ngrams=corpus_ngrams, corpus_ngram_n=n)
    return NgramOverlapSignal().score_item(item, ctx)


register(NgramOverlapSignal())
