"""Tests for the n-gram overlap signal.

Run: cd ~/personal/benchcheck && python -m pytest -q
"""
from benchcheck.signals import ngram_overlap
from benchcheck.types import Item


def test_verbatim_match_scores_high():
    corpus = ["the quick brown fox jumps over the lazy sleeping dog today"]
    item = Item(id="x", prompt="the quick brown fox jumps over the lazy sleeping dog today")
    grams = ngram_overlap.build_corpus_ngrams(corpus, n=8)
    res = ngram_overlap.score_item(item, grams, n=8)
    assert res.score == 1.0


def test_unrelated_text_scores_low():
    corpus = ["completely different content about photosynthesis and plants growing"]
    item = Item(id="y", prompt="how many tulips are planted across seven equal rows of nine")
    grams = ngram_overlap.build_corpus_ngrams(corpus, n=8)
    res = ngram_overlap.score_item(item, grams, n=8)
    assert res.score == 0.0


def test_short_prompt_does_not_crash():
    corpus = ["some reference text here for safety"]
    item = Item(id="z", prompt="too short")
    grams = ngram_overlap.build_corpus_ngrams(corpus, n=8)
    res = ngram_overlap.score_item(item, grams, n=8)
    assert 0.0 <= res.score <= 1.0
