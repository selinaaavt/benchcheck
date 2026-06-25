"""Corpus n-gram index with a fast C++ backend and a pure-Python fallback.

The n-gram overlap check scans a (potentially large) reference corpus for each
benchmark question's n-grams. The hot path is compiled in native/ngram_scan.cpp
and exposed as the `ngram_scan` extension; build it with `python native/build.py`.

If the extension isn't built, we fall back to an equivalent pure-Python index so
the tool always works -- the C++ backend is an optimization, not a requirement.
Both backends expose the same `.overlap(text)` API and produce identical scores.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_WORD = re.compile(r"\w+")


def _native_module():
    """Import the compiled extension if present (it lives in the repo root)."""
    try:
        import ngram_scan  # type: ignore

        return ngram_scan
    except ImportError:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import ngram_scan  # type: ignore

            return ngram_scan
        except ImportError:
            return None


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class _PyIndex:
    backend = "python"

    def __init__(self, corpus_texts: list[str], n: int):
        self.n = n
        self._set: set[tuple[str, ...]] = set()
        for text in corpus_texts:
            self._set |= _ngrams(_tokens(text), n)

    def overlap(self, text: str) -> float:
        g = _ngrams(_tokens(text), self.n)
        if not g:
            return 0.0
        return sum(1 for x in g if x in self._set) / len(g)

    def size(self) -> int:
        return len(self._set)


class _CppIndex:
    backend = "cpp"

    def __init__(self, corpus_texts: list[str], n: int, module):
        self.n = n
        self._idx = module.NgramIndex(corpus_texts, n)

    def overlap(self, text: str) -> float:
        return self._idx.overlap(text)

    def size(self) -> int:
        return self._idx.size()


def build_index(corpus_texts: list[str], n: int, prefer_native: bool = True):
    """Construct a corpus index, using the C++ backend when available."""
    if prefer_native:
        mod = _native_module()
        if mod is not None:
            return _CppIndex(corpus_texts, n, mod)
    return _PyIndex(corpus_texts, n)


def native_available() -> bool:
    return _native_module() is not None
