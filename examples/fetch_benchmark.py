"""Fetch a public multiple-choice benchmark and convert it to benchcheck JSONL.

Supports several benchmarks whose schemas differ; each has a small adapter that
normalizes it to {id, prompt, answer (letter), choices}.

    python -m examples.fetch_benchmark --benchmark arc_easy --limit 500 --out examples/arc_easy.jsonl
    python -m examples.fetch_benchmark --benchmark openbookqa --limit 500 --out examples/obqa.jsonl

Benchmarks: arc_easy, arc_challenge, openbookqa, sciq
"""
from __future__ import annotations

import argparse

from benchcheck.dataset import save_jsonl
from benchcheck.types import Item

_LETTERS = "ABCDEFGH"


def _arc(split, cfg):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", cfg, split=split)
    for row in ds:
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        ak = row["answerKey"]
        ans = _LETTERS[labels.index(ak)] if ak in labels else None
        yield Item(id=row["id"], prompt=row["question"], answer=ans, choices=texts)


def _openbookqa(split, cfg):
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split=split)
    for row in ds:
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        ak = row["answerKey"]
        ans = _LETTERS[labels.index(ak)] if ak in labels else None
        yield Item(id=row["id"], prompt=row["question_stem"], answer=ans, choices=texts)


def _sciq(split, cfg):
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split=split)
    for i, row in enumerate(ds):
        # SciQ gives a correct answer + 3 distractors; assemble deterministic
        # choices with the correct answer always first (answer="A"). Order does
        # not matter for our checks (shuffle_sensitivity reshuffles anyway).
        choices = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        yield Item(id=f"sciq_{i}", prompt=row["question"], answer="A", choices=choices)


_ADAPTERS = {
    "arc_easy": (_arc, "ARC-Easy", "test"),
    "arc_challenge": (_arc, "ARC-Challenge", "test"),
    "openbookqa": (_openbookqa, "main", "test"),
    "sciq": (_sciq, None, "test"),
}


def fetch(benchmark: str, limit: int | None) -> list[Item]:
    fn, cfg, split = _ADAPTERS[benchmark]
    items = []
    for it in fn(split, cfg):
        items.append(it)
        if limit and len(items) >= limit:
            break
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=sorted(_ADAPTERS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    items = fetch(args.benchmark, args.limit)
    save_jsonl(items, args.out)
    print(f"wrote {len(items)} items from {args.benchmark} to {args.out}")


if __name__ == "__main__":
    main()
