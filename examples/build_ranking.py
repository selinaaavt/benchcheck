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
    # Committed result JSONs live in examples/results/; fall back to results/.
    search_dirs = [ROOT / "examples" / "results", ROOT / "results"]
    seen = set()
    for d in search_dirs:
        for path in sorted(glob.glob(str(d / "gpt2_*.json"))):
            name = Path(path).stem.replace("gpt2_", "")
            if name in seen:
                continue
            payload = _load(path)
            if payload:
                seen.add(name)
                rows.append((name, payload))

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
