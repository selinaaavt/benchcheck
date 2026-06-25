"""Build a contamination ranking table from per-benchmark JSON reports.

Reads results/gpt2_<benchmark>.json files and prints a Markdown table ranking
benchmarks by flagged contamination rate, plus the ARC-Easy 500 result. Intended
to be pasted into FINDINGS.md once the detection runs finish.

    python -m examples.build_ranking
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path: str) -> dict | None:
    try:
        return json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    rows = []

    # The earlier ARC-Easy 500 run, if present.
    arc = _load(str(ROOT / "examples" / "results_gpt2_arc500.json"))
    if arc:
        rows.append(("ARC-Easy", arc))

    for path in sorted(glob.glob(str(ROOT / "results" / "gpt2_*.json"))):
        name = Path(path).stem.replace("gpt2_", "")
        d = _load(path)
        if d:
            rows.append((name, d))

    if not rows:
        print("no result JSONs found yet (run examples/run_all_benchmarks.sh)")
        return

    rows.sort(key=lambda r: -r[1]["contamination_rate"])
    print("| Benchmark | Questions | Flagged | Rate | 95% CI |")
    print("|---|---|---|---|---|")
    for name, d in rows:
        print(
            f"| {name} | {d['n_items']} | {d['n_flagged']} | "
            f"{d['contamination_rate']:.1%} | "
            f"{d['ci_low']:.1%}–{d['ci_high']:.1%} |"
        )


if __name__ == "__main__":
    main()
