"""Run benchcheck across several benchmarks on a GPU (intended for Colab/Kaggle).

Usage (after cloning the repo and installing deps):

    python -m examples.colab_run --model Qwen/Qwen2.5-3B --limit 300

Writes results/<tag>_<benchmark>.json for each benchmark, then prints a ranking
table. Designed for a free T4 GPU, where a 3B model runs ~50-100x faster than
CPU, so all four benchmarks finish in minutes with all four checks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchcheck.dataset import load_jsonl
from benchcheck.models.hf import HFModel
from benchcheck.pipeline import RunConfig, run

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = ["openbookqa", "sciq", "arc_easy", "arc_challenge"]


def _tag(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").replace("-", "_").lower()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--corpus", default="examples/sample_corpus.txt")
    args = p.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = _tag(args.model)

    print(f"loading {args.model} ...")
    t0 = time.time()
    model = HFModel(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s on device={model.device}")

    corpus = [
        ln for ln in (ROOT / args.corpus).read_text().splitlines() if ln.strip()
    ]

    summary = []
    for b in BENCHMARKS:
        path = ROOT / "examples" / f"{b}.jsonl"
        if not path.exists():
            print(f"  (missing {b}.jsonl -- fetch it first; skipping)")
            continue
        items = load_jsonl(path)[: args.limit]
        print(f"\n=== {b}: {len(items)} questions ===")
        t0 = time.time()
        res = run(model, items, RunConfig(corpus_texts=corpus, seed=0))
        dt = time.time() - t0
        rep = res.report
        out_path = out_dir / f"{tag}_{b}.json"
        _write_json(out_path, args.model, b, res)
        print(
            f"  flagged {rep.n_flagged}/{rep.n_items} "
            f"({rep.contamination_rate:.1%}, CI {rep.ci_low:.1%}-{rep.ci_high:.1%}) "
            f"in {dt:.0f}s ({len(items)/dt:.1f} q/s)"
        )
        summary.append((b, rep.n_items, rep.n_flagged, rep.contamination_rate,
                        rep.ci_low, rep.ci_high))

    print(f"\n=== ranking ({args.model}) ===")
    print("| Benchmark | Questions | Flagged | Rate | 95% CI |")
    print("|---|---|---|---|---|")
    for b, n, f, rate, lo, hi in sorted(summary, key=lambda r: -r[3]):
        print(f"| {b} | {n} | {f} | {rate:.1%} | {lo:.1%}-{hi:.1%} |")


def _write_json(path: Path, model: str, benchmark: str, res) -> None:
    r = res.report
    payload = {
        "dataset": benchmark,
        "model": model,
        "n_items": r.n_items,
        "n_flagged": r.n_flagged,
        "contamination_rate": r.contamination_rate,
        "ci_low": r.ci_low,
        "ci_high": r.ci_high,
        "per_signal_mean": r.per_signal_mean,
        "signals_run": res.signals_run,
        "timing": {
            "n_items": res.timing.n_items,
            "wall_seconds": res.timing.wall_seconds,
            "items_per_second": res.timing.items_per_second,
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
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
