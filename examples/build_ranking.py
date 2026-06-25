"""Build contamination ranking / comparison tables from result JSON reports.

Reads every result JSON in examples/results/ (and results/), groups by model,
and prints:
  1. a per-model ranking table, and
  2. a side-by-side model comparison across benchmarks (when >1 model present).

Each result file is named <model_tag>_<benchmark>.json.

    python -m examples.build_ranking
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Known benchmark slugs, so we can split "<model_tag>_<benchmark>" correctly
# even when the model tag itself contains underscores.
BENCHMARKS = ["arc_easy", "arc_challenge", "openbookqa", "sciq"]
BENCH_LABEL = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "openbookqa": "OpenBookQA",
    "sciq": "SciQ",
}


def _load(path: str) -> dict | None:
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def _split_name(stem: str) -> tuple[str, str] | None:
    """Split '<model_tag>_<benchmark>' using the known benchmark suffixes."""
    for b in BENCHMARKS:
        if stem.endswith("_" + b):
            return stem[: -(len(b) + 1)], b
    return None


def _collect() -> dict[str, dict[str, dict]]:
    """Return {model_tag: {benchmark: report}}."""
    out: dict[str, dict[str, dict]] = {}
    seen_paths = set()
    for d in [ROOT / "examples" / "results", ROOT / "results"]:
        for path in sorted(glob.glob(str(d / "*.json"))):
            stem = Path(path).stem
            split = _split_name(stem)
            if not split:
                continue
            model, bench = split
            payload = _load(path)
            if payload and (model, bench) not in seen_paths:
                seen_paths.add((model, bench))
                out.setdefault(model, {})[bench] = payload
    return out


def main() -> None:
    by_model = _collect()
    if not by_model:
        print("no result JSONs found (run a detect sweep first)")
        return

    for model, benches in by_model.items():
        print(f"\n### {model}\n")
        print("| Benchmark | Questions | Flagged | Rate | 95% CI |")
        print("|---|---|---|---|---|")
        for b, d in sorted(benches.items(), key=lambda kv: -kv[1]["contamination_rate"]):
            print(
                f"| {BENCH_LABEL.get(b, b)} | {d['n_items']} | {d['n_flagged']} | "
                f"{d['contamination_rate']:.1%} | {d['ci_low']:.1%}-{d['ci_high']:.1%} |"
            )

    if len(by_model) > 1:
        models = list(by_model.keys())
        print("\n### Model comparison (flagged rate)\n")
        header = "| Benchmark | " + " | ".join(models) + " |"
        print(header)
        print("|" + "---|" * (len(models) + 1))
        all_benches = sorted({b for m in by_model.values() for b in m})
        for b in all_benches:
            cells = []
            for m in models:
                d = by_model[m].get(b)
                cells.append(f"{d['contamination_rate']:.1%}" if d else "-")
            print(f"| {BENCH_LABEL.get(b, b)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
