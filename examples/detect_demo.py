"""End-to-end detection demo on the human-readable sample benchmark.

Unlike `examples/demo.py` (which shows only the no-model n-gram signal), this
runs the FULL pipeline -- all available signals + the stats layer -- on
`sample_benchmark.jsonl`, against a mock model that has "memorized" exactly the
three questions that also appear verbatim in `sample_corpus.txt`.

So we know the ground truth: geo1, sci1, lit1 are contaminated. A working tool
should flag those and leave the other 13 alone.

Run:
    cd ~/personal/benchcheck && python -m examples.detect_demo
"""
from __future__ import annotations

from pathlib import Path

from benchcheck.dataset import load_jsonl
from benchcheck.models.mock import MockModel
from benchcheck.pipeline import RunConfig, run

HERE = Path(__file__).resolve().parent

# The three questions that leaked into the corpus (our ground truth).
LEAKED_IDS = {"geo1", "sci1", "lit1"}


def main() -> None:
    items = load_jsonl(HERE / "sample_benchmark.jsonl")
    corpus_texts = [
        ln for ln in (HERE / "sample_corpus.txt").read_text().splitlines() if ln.strip()
    ]

    # Simulate a model trained on that corpus: it memorized the leaked prompts.
    leaked_prompts = [it.prompt for it in items if it.id in LEAKED_IDS]
    model = MockModel(memorized=leaked_prompts)

    out = run(model, items, RunConfig(corpus_texts=corpus_texts, seed=0))

    print("=== benchcheck end-to-end detection demo ===")
    print(out.report.summary())
    print("\nper-item verdicts (sorted by combined score):")
    print(f"  {'item':<8}{'flagged':>9}{'score':>9}{'truth':>12}")
    for v in sorted(out.report.items, key=lambda v: -v.combined_score):
        truth = "LEAKED" if v.item_id in LEAKED_IDS else "clean"
        mark = "FLAG" if v.flagged else "-"
        print(f"  {v.item_id:<8}{mark:>9}{v.combined_score:>9.2f}{truth:>12}")

    flagged = {v.item_id for v in out.report.items if v.flagged}
    print("\nground truth leaked:", sorted(LEAKED_IDS))
    print("detector flagged   :", sorted(flagged))
    print("correct" if flagged == LEAKED_IDS else "mismatch -- see above")


if __name__ == "__main__":
    main()
