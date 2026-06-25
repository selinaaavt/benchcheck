"""Pipeline: run selected signals over a dataset and produce a report.

Glues the pieces together:
  model + dataset -> SignalContext -> run each applicable signal per item
  -> stats.analyze -> BenchmarkReport.

The pipeline is responsible for the graceful-degradation policy: it only runs
signals whose required model capability is available, and logs which signals
were skipped and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from benchcheck import signals as signals_pkg
from benchcheck.models.base import capabilities_of
from benchcheck.signals.base import SignalContext, available_for
from benchcheck.signals.ngram_overlap import build_corpus_ngrams
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
class RunOutput:
    report: BenchmarkReport
    signals_run: list[str]
    signals_skipped: dict[str, str]  # name -> reason


def run(model, items: list[Item], config: RunConfig | None = None) -> RunOutput:
    config = config or RunConfig()

    caps = capabilities_of(model) if model is not None else set()
    ctx = SignalContext(
        model=model,
        model_capabilities=caps,
        corpus_ngrams=build_corpus_ngrams(config.corpus_texts, config.corpus_ngram_n)
        if config.corpus_texts
        else None,
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

    # Run.
    results_by_item: dict[str, list[SignalResult]] = {}
    for item in items:
        results_by_item[item.id] = [s.score_item(item, ctx) for s in candidates]

    report = analyze(
        results_by_item,
        flag_threshold_p=config.flag_threshold_p,
        weights=config.weights,
        seed=config.seed,
    )
    return RunOutput(report=report, signals_run=signals_run, signals_skipped=signals_skipped)
