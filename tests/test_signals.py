"""Tests for individual contamination signals against the mock model."""
from benchcheck.models.mock import MockModel
from benchcheck.signals.base import SignalContext
from benchcheck.signals.ngram_overlap import NgramOverlapSignal, build_corpus_ngrams
from benchcheck.signals.perturbation_gap import PerturbationGapSignal
from benchcheck.signals.shuffle_sensitivity import ShuffleSensitivitySignal
from benchcheck.signals.verbatim_completion import VerbatimCompletionSignal
from benchcheck.types import Item

LEAKED = "If a train leaves Chicago at 60 mph and another leaves New York at 40 mph when do they meet"
CLEAN = "A gardener plants seven rows of tulips with nine tulips in each row how many total"


def _ctx(memorized=None, corpus=None):
    model = MockModel(memorized=memorized or [])
    grams = build_corpus_ngrams(corpus, 8) if corpus else None
    return SignalContext(model=model, corpus_ngrams=grams, corpus_ngram_n=8)


def test_ngram_fires_on_corpus_member():
    item = Item(id="x", prompt=LEAKED)
    res = NgramOverlapSignal().score_item(item, _ctx(corpus=[LEAKED]))
    assert res.score > 0.9


def test_ngram_quiet_on_absent_item():
    item = Item(id="x", prompt=CLEAN)
    res = NgramOverlapSignal().score_item(item, _ctx(corpus=[LEAKED]))
    assert res.score == 0.0


def test_perturbation_gap_higher_for_memorized():
    ctx = _ctx(memorized=[LEAKED])
    sig = PerturbationGapSignal()
    leaked = sig.score_item(Item(id="a", prompt=LEAKED), ctx)
    clean = sig.score_item(Item(id="b", prompt=CLEAN), ctx)
    assert leaked.score > clean.score


def test_verbatim_completion_detects_regurgitation():
    ctx = _ctx(memorized=[LEAKED])
    res = VerbatimCompletionSignal().score_item(Item(id="a", prompt=LEAKED), ctx)
    assert res.score > 0.3


def test_verbatim_completion_quiet_on_clean():
    ctx = _ctx(memorized=[LEAKED])
    res = VerbatimCompletionSignal().score_item(Item(id="b", prompt=CLEAN), ctx)
    assert res.score == 0.0


def test_shuffle_abstains_without_choices():
    ctx = _ctx(memorized=[LEAKED])
    res = ShuffleSensitivitySignal().score_item(Item(id="a", prompt=LEAKED), ctx)
    assert res.score == 0.0
    assert res.detail.get("reason") == "not_mcq"


def test_signals_abstain_without_model():
    ctx = SignalContext(model=None)
    for sig in (PerturbationGapSignal(), VerbatimCompletionSignal(), ShuffleSensitivitySignal()):
        res = sig.score_item(Item(id="a", prompt=LEAKED), ctx)
        assert res.score == 0.0
