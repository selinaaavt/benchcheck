"""Fetch the ARC-Easy benchmark and convert it to benchcheck JSONL format.

ARC (AI2 Reasoning Challenge) is a widely-used multiple-choice science QA
benchmark. It predates many current models and is heavily scraped, making it a
realistic target for contamination analysis.

    python -m examples.fetch_arc --split test --out examples/arc_easy_test.jsonl --limit 500
"""
from __future__ import annotations

import argparse
from pathlib import Path

from benchcheck.dataset import save_jsonl
from benchcheck.types import Item


def fetch(split: str, limit: int | None) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    items: list[Item] = []
    for row in ds:
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        ans_key = row["answerKey"]
        # Normalize the answer to a letter index aligned with our A/B/C/D order.
        if ans_key in labels:
            answer_letter = "ABCDEFGH"[labels.index(ans_key)]
        else:
            answer_letter = None
        items.append(
            Item(
                id=row["id"],
                prompt=row["question"],
                answer=answer_letter,
                choices=choices,
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="examples/arc_easy_test.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    items = fetch(args.split, args.limit)
    save_jsonl(items, args.out)
    print(f"wrote {len(items)} items to {args.out}")


if __name__ == "__main__":
    main()
