"""Throughput benchmark: C++ ngram scanner vs pure-Python baseline.

Builds a large synthetic corpus, indexes it, scans many queries, and reports
n-grams/second and the C++/Python speedup. This is the source of the
performance numbers in the README/FINDINGS.

    python native/bench_ngram.py --corpus-docs 20000 --queries 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The compiled extension lives in the repo root; ensure it's importable even
# when this script is run as native/bench_ngram.py (which puts native/ on path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ngram_scan
from benchcheck.signals.ngram_overlap import _ngrams, _tokens, build_corpus_ngrams

_WORDS = (
    "the a of to in and for on with as by at from photosynthesis cell energy "
    "planet gravity orbit mass force water carbon oxygen nitrogen rotation "
    "ecosystem producer consumer light reaction electron atom molecule organ "
    "system temperature pressure density volume mineral rock soil fossil ocean"
).split()


def synth_corpus(n_docs: int, words_per_doc: int, seed: int = 0) -> list[str]:
    # Deterministic pseudo-random docs (no RNG dependency surprises).
    docs = []
    state = seed * 2654435761 & 0xFFFFFFFF
    for _ in range(n_docs):
        toks = []
        for _ in range(words_per_doc):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            toks.append(_WORDS[state % len(_WORDS)])
        docs.append(" ".join(toks))
    return docs


def synth_queries(corpus: list[str], n: int) -> list[str]:
    # Half real substrings of the corpus (will hit), half novel (will miss).
    qs = []
    for i in range(n):
        if i % 2 == 0:
            doc = corpus[i % len(corpus)]
            words = doc.split()
            qs.append(" ".join(words[:12]))
        else:
            qs.append(f"novel query number {i} about unrelated topics entirely unseen")
    return qs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-docs", type=int, default=20000)
    p.add_argument("--words-per-doc", type=int, default=60)
    p.add_argument("--queries", type=int, default=2000)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    print(f"building synthetic corpus: {args.corpus_docs} docs x {args.words_per_doc} words")
    corpus = synth_corpus(args.corpus_docs, args.words_per_doc)
    queries = synth_queries(corpus, args.queries)

    # --- C++ ---
    t0 = time.perf_counter()
    idx = ngram_scan.NgramIndex(corpus, args.n)
    cpp_build = time.perf_counter() - t0
    res = idx.scan(queries)
    print("\n[C++]")
    print(f"  index size        : {idx.size():,} distinct n-grams")
    print(f"  build time        : {cpp_build:.3f} s "
          f"({idx.ngrams_processed()/idx.build_seconds():,.0f} n-grams/s)")
    print(f"  scan time         : {res['scan_seconds']*1000:.1f} ms for {len(queries)} queries")
    print(f"  scan throughput   : {res['ngrams_per_second']:,.0f} query-n-grams/s")

    # --- Python ---
    t0 = time.perf_counter()
    pg = build_corpus_ngrams(corpus, args.n)
    py_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    py_total = 0
    for q in queries:
        g = _ngrams(_tokens(q), args.n)
        py_total += len(g)
        if g:
            _ = sum(1 for x in g if x in pg) / len(g)
    py_scan = time.perf_counter() - t0
    print("\n[Python]")
    print(f"  build time        : {py_build:.3f} s")
    print(f"  scan time         : {py_scan*1000:.1f} ms")
    print(f"  scan throughput   : {py_total/py_scan:,.0f} query-n-grams/s")

    print("\n[speedup]")
    print(f"  build : {py_build/cpp_build:.1f}x")
    print(f"  scan  : {py_scan/res['scan_seconds']:.1f}x")


if __name__ == "__main__":
    main()
