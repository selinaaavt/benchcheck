"""Pipeline: run selected signals over a dataset and produce a report.

Glues the pieces together:
  model + dataset -> SignalContext -> run each applicable signal per item
  -> stats.analyze -> BenchmarkReport.

The pipeline is responsible for the graceful-degradation policy: it only runs
signals whose required model capability is available, and logs which signals
were skipped and why.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchcheck import signals as signals_pkg
from benchcheck.corpus_index import build_index
from benchcheck.models.base import capabilities_of
from benchcheck.signals.base import SignalContext, available_for
from benchcheck.stats import BenchmarkReport, analyze
from benchcheck.types import Item, SignalResult


@dataclass
class RunConfig:
    corpus_texts: list[str] = field(default_factory=list)
    corpus_ngram_n: int = 8
    flag_threshold_p: float = 0.10
    signal_names: list[str] | None = None  # None = all available
    weights: dict[str, float] | None = None
    seed: int = 0


@dataclass
class Timing:
    """Throughput/latency instrumentation for a run."""

    n_items: int = 0
    wall_seconds: float = 0.0
    per_signal_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def items_per_second(self) -> float:
        return self.n_items / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def ms_per_item(self) -> float:
        return 1000.0 * self.wall_seconds / self.n_items if self.n_items else 0.0

    def summary(self) -> str:
        lines = [
            f"items             : {self.n_items}",
            f"wall time         : {self.wall_seconds:.2f} s",
            f"throughput        : {self.items_per_second:.1f} items/s "
            f"({self.ms_per_item:.1f} ms/item)",
            "per-signal time:",
        ]
        for name, secs in sorted(self.per_signal_seconds.items(), key=lambda x: -x[1]):
            lines.append(f"    {name:<22} {secs:.2f} s")
        return "\n".join(lines)


@dataclass
class RunOutput:
    report: BenchmarkReport
    signals_run: list[str]
    signals_skipped: dict[str, str]  # name -> reason
    timing: Timing = field(default_factory=Timing)


def run(model, items: list[Item], config: RunConfig | None = None) -> RunOutput:
    config = config or RunConfig()

    caps = capabilities_of(model) if model is not None else set()
    corpus_index = (
        build_index(config.corpus_texts, config.corpus_ngram_n)
        if config.corpus_texts
        else None
    )
    ctx = SignalContext(
        model=model,
        model_capabilities=caps,
        corpus_index=corpus_index,
        corpus_ngram_n=config.corpus_ngram_n,
    )

    # Decide which signals to run.
    candidates = available_for(caps)
    if config.signal_names is not None:
        wanted = set(config.signal_names)
        candidates = [s for s in candidates if s.name in wanted]

    signals_run = [s.name for s in candidates]
    signals_skipped: dict[str, str] = {}
    for s in signals_pkg.all_signals():
        if s.name not in signals_run:
            if s.required_capability and s.required_capability not in caps:
                signals_skipped[s.name] = f"needs '{s.required_capability}' capability"
            elif config.signal_names is not None and s.name not in config.signal_names:
                signals_skipped[s.name] = "not selected"

    # Run, timing each signal so we can report throughput per signal.
    results_by_item: dict[str, list[SignalResult]] = {}
    per_signal_seconds: dict[str, float] = {s.name: 0.0 for s in candidates}
    wall_start = time.perf_counter()
    for item in items:
        item_results = []
        for s in candidates:
            t0 = time.perf_counter()
            item_results.append(s.score_item(item, ctx))
            per_signal_seconds[s.name] += time.perf_counter() - t0
        results_by_item[item.id] = item_results
    wall_seconds = time.perf_counter() - wall_start

    report = analyze(
        results_by_item,
        flag_threshold_p=config.flag_threshold_p,
        weights=config.weights,
        seed=config.seed,
    )
    timing = Timing(
        n_items=len(items),
        wall_seconds=wall_seconds,
        per_signal_seconds=per_signal_seconds,
    )
    return RunOutput(
        report=report,
        signals_run=signals_run,
        signals_skipped=signals_skipped,
        timing=timing,
    )
