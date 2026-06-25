"""Command-line interface.

    python -m benchcheck detect --dataset data.jsonl --model mock
    python -m benchcheck detect --dataset data.jsonl --model hf:gpt2 --corpus corpus.txt
    python -m benchcheck signals          # list available signals
    python -m benchcheck calibrate        # run the validation experiment

Designed so the whole thing runs with `--model mock` and no heavy deps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchcheck import signals as signals_pkg
from benchcheck.dataset import load_jsonl
from benchcheck.models import load_model
from benchcheck.pipeline import RunConfig, run


def _read_corpus(path: str | None) -> list[str]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8")
    # Treat each non-empty line as a corpus document.
    return [ln for ln in text.splitlines() if ln.strip()]


def cmd_detect(args: argparse.Namespace) -> int:
    items = load_jsonl(args.dataset)
    model = load_model(args.model)
    config = RunConfig(
        corpus_texts=_read_corpus(args.corpus),
        corpus_ngram_n=args.ngram_n,
        flag_threshold_p=args.threshold,
        signal_names=args.signals.split(",") if args.signals else None,
        seed=args.seed,
    )
    out = run(model, items, config)

    # Persist a machine-readable report FIRST, before any printing, so a long
    # run's results survive regardless of stdout buffering or downstream pipes.
    if args.output:
        _write_report_json(args, out)
        print(f"[wrote report to {args.output}]")

    print(f"\n=== benchcheck report: {args.dataset} (model={args.model}) ===")
    print(out.report.summary())
    if args.timing:
        print("\n--- timing ---")
        print(out.timing.summary())
    if out.signals_skipped:
        print("\nskipped signals:")
        for name, reason in out.signals_skipped.items():
            print(f"    {name:<22} ({reason})")

    if args.show_flagged:
        flagged = [v for v in out.report.items if v.flagged]
        flagged.sort(key=lambda v: -v.combined_score)
        print(f"\nflagged items ({len(flagged)}):")
        for v in flagged:
            print(f"    {v.item_id:<18} score={v.combined_score:+.2f}  "
                  f"signals={ {k: round(s, 2) for k, s in v.signal_scores.items() if s > 0} }")
    return 0


def _write_report_json(args: argparse.Namespace, out) -> None:
    import json

    r = out.report
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "n_items": r.n_items,
        "n_flagged": r.n_flagged,
        "contamination_rate": r.contamination_rate,
        "ci_low": r.ci_low,
        "ci_high": r.ci_high,
        "per_signal_mean": r.per_signal_mean,
        "signals_run": out.signals_run,
        "signals_skipped": out.signals_skipped,
        "timing": {
            "n_items": out.timing.n_items,
            "wall_seconds": out.timing.wall_seconds,
            "items_per_second": out.timing.items_per_second,
            "ms_per_item": out.timing.ms_per_item,
            "per_signal_seconds": out.timing.per_signal_seconds,
        },
        "items": [
            {
                "id": v.item_id,
                "combined_score": v.combined_score,
                "flagged": v.flagged,
                "signal_scores": v.signal_scores,
            }
            for v in r.items
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def cmd_signals(args: argparse.Namespace) -> int:
    print("available signals:")
    for s in signals_pkg.all_signals():
        cap = s.required_capability or "none (no model needed)"
        print(f"    {s.name:<22} requires: {cap}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    # Imported lazily to keep the CLI import light.
    from benchcheck.calibration import run_calibration

    run_calibration(n_items=args.n_items, contaminate_frac=args.frac, seed=args.seed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchcheck", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="estimate contamination of a benchmark")
    d.add_argument("--dataset", required=True, help="JSONL benchmark file")
    d.add_argument("--model", default="mock", help="model spec (mock | hf:NAME)")
    d.add_argument("--corpus", default=None, help="reference corpus text file")
    d.add_argument("--signals", default=None, help="comma-separated signal names")
    d.add_argument("--ngram-n", type=int, default=8, dest="ngram_n")
    d.add_argument("--threshold", type=float, default=0.10, help="flag p-value cutoff")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--show-flagged", action="store_true")
    d.add_argument("--timing", action="store_true", help="report throughput/latency")
    d.add_argument("--output", default=None, help="write JSON report to this path")
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("signals", help="list available signals")
    s.set_defaults(func=cmd_signals)

    c = sub.add_parser("calibrate", help="run the validation experiment")
    c.add_argument("--n-items", type=int, default=200, dest="n_items")
    c.add_argument("--frac", type=float, default=0.3, help="fraction to contaminate")
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_calibrate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
