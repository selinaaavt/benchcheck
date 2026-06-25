"""Render the RESULTS_PLACEHOLDER blocks in README.md and FINDINGS.md from a
benchcheck JSON report. Keeps the docs honest: the numbers come straight from
the run, not hand-typed.

    python -m examples.render_findings examples/results_gpt2_arc500.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _findings_block(d: dict) -> str:
    t = d["timing"]
    lines = [
        f"- **Questions analyzed:** {d['n_items']}",
        f"- **Flagged as likely contaminated:** {d['n_flagged']} "
        f"({d['contamination_rate']:.1%}), 95% CI "
        f"{d['ci_low']:.1%}-{d['ci_high']:.1%}",
        f"- **Throughput:** {t['items_per_second']:.2f} questions/sec "
        f"({t['ms_per_item']:.0f} ms/question) on CPU, "
        f"{t['n_items']} questions in {t['wall_seconds']:.0f} s",
        "",
        "Mean score per check:",
        "",
        "| Check | Mean score |",
        "|---|---|",
    ]
    for name, val in sorted(d["per_signal_mean"].items()):
        lines.append(f"| `{name}` | {val:.3f} |")

    flagged = [i for i in d["items"] if i["flagged"]]
    flagged.sort(key=lambda x: -x["combined_score"])
    lines += ["", "Top flagged questions (and which checks fired):", ""]
    for it in flagged[:8]:
        sigs = ", ".join(
            f"{k} {v:.2f}" for k, v in sorted(it["signal_scores"].items(), key=lambda x: -x[1]) if v > 0
        )
        lines.append(f"- `{it['id']}` (score {it['combined_score']:+.2f}): {sigs}")
    return "\n".join(lines)


def _readme_block(d: dict) -> str:
    t = d["timing"]
    return (
        f"- Analyzed **{d['n_items']} questions** in {t['wall_seconds']:.0f} s "
        f"({t['items_per_second']:.2f} questions/sec on CPU).\n"
        f"- **{d['n_flagged']} questions ({d['contamination_rate']:.1%})** flagged "
        f"as showing memorization fingerprints (95% CI "
        f"{d['ci_low']:.1%}-{d['ci_high']:.1%})."
    )


def _patch(path: Path, block: str) -> None:
    text = path.read_text()
    marker = "<!-- RESULTS_PLACEHOLDER -->"
    if marker not in text:
        print(f"  (no placeholder in {path.name}; skipping)")
        return
    path.write_text(text.replace(marker, block))
    print(f"  patched {path.name}")


def main() -> None:
    report = json.load(open(sys.argv[1]))
    print("rendering findings from", sys.argv[1])
    _patch(ROOT / "FINDINGS.md", _findings_block(report))
    _patch(ROOT / "README.md", _readme_block(report))


if __name__ == "__main__":
    main()
