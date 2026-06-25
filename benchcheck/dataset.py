"""Dataset loading.

Benchmarks are stored as JSONL, one item per line:

    {"id": "q1", "prompt": "...", "answer": "B", "choices": ["..", "..", ".."]}

`answer` and `choices` are optional. We keep the format dead simple so you can
point the tool at any benchmark by converting it to this shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchcheck.types import Item


def load_jsonl(path: str | Path) -> list[Item]:
    items: list[Item] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "prompt" not in obj:
                raise ValueError(f"{path}:{lineno}: item needs 'id' and 'prompt'")
            items.append(
                Item(
                    id=str(obj["id"]),
                    prompt=obj["prompt"],
                    answer=obj.get("answer"),
                    choices=obj.get("choices", []),
                )
            )
    return items


def save_jsonl(items: list[Item], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            obj = {"id": it.id, "prompt": it.prompt}
            if it.answer is not None:
                obj["answer"] = it.answer
            if it.choices:
                obj["choices"] = it.choices
            f.write(json.dumps(obj) + "\n")
