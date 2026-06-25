"""Runnable demo: n-gram overlap on a hand-built contaminated vs. clean case.

No model download, no internet. Run it to feel out what the tool does:

    cd ~/personal/benchcheck
    python -m examples.demo

We fake a "training corpus" that happens to contain one benchmark question
verbatim (the contaminated one) but not the other (the clean one). A working
detector should light up on the first and stay quiet on the second.
"""
from __future__ import annotations

from benchcheck.signals import ngram_overlap
from benchcheck.types import Item

# Pretend this is text the model was trained on. Note the FIRST benchmark
# question appears here word-for-word; the second does not.
FAKE_TRAINING_CORPUS = [
    "Welcome to the study guide. A classic problem: If a train leaves Chicago "
    "at 60 mph and another leaves New York at 40 mph, when do they meet? "
    "The answer involves relative speed.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Common interview questions and their model answers are listed below.",
]

ITEMS = [
    Item(
        id="q_contaminated",
        prompt="If a train leaves Chicago at 60 mph and another leaves New "
        "York at 40 mph, when do they meet?",
        answer="...",
    ),
    Item(
        id="q_clean",
        prompt="A gardener plants 7 rows of tulips with 9 tulips in each row. "
        "How many tulips are there in total?",
        answer="63",
    ),
]


def main() -> None:
    n = 8
    corpus = ngram_overlap.build_corpus_ngrams(FAKE_TRAINING_CORPUS, n)

    print(f"{'item':<18}{'overlap':>10}   verdict")
    print("-" * 48)
    for item in ITEMS:
        res = ngram_overlap.score_item(item, corpus, n=n)
        verdict = "SUSPICIOUS" if res.score > 0.5 else "looks clean"
        print(f"{item.id:<18}{res.score:>9.0%}   {verdict}")
        print(f"  matched {res.detail['matched']}/{res.detail['total']} {n}-grams")


if __name__ == "__main__":
    main()
