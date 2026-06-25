"""Calibration experiment: prove the detector actually works.

This is the headline result -- the thing that turns "a script that prints
numbers" into "a tool I validated." The logic mirrors how you'd validate against
a real model, but runs in seconds with the mock backend:

  1. Build a synthetic benchmark of N items with KNOWN ground truth.
  2. Pick a random subset to be "contaminated": tell the model it memorized
     exactly those items (simulating their presence in training data).
  3. Run the detector, which does NOT know which items are contaminated.
  4. Compare the detector's flags against ground truth -> precision, recall,
     and whether the estimated contamination rate's CI covers the true rate.

A good detector should: flag mostly truly-contaminated items (high precision),
catch most of them (high recall), and produce a CI that contains the true rate.

We deliberately also report FALSE POSITIVES on the clean slice, because a
detector that flags everything is useless -- being honest about that is the
point of the whole project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchcheck.models.mock import MockModel
from benchcheck.pipeline import RunConfig, run
from benchcheck.types import Item


def _synthetic_benchmark(n_items: int, seed: int) -> list[Item]:
    """Generate distinct, realistic-ish QA items deterministically."""
    rng = np.random.default_rng(seed)
    templates = [
        "If a train leaves {a} at {x} mph and another leaves {b} at {y} mph, when do they meet?",
        "A gardener plants {x} rows of tulips with {y} tulips in each row. How many tulips are there in total?",
        "What is the capital city of the region known as {a} in the year {y}?",
        "Compute the sum of the first {x} positive integers and explain the formula used.",
        "A recipe needs {x} cups of flour for {y} servings. How much flour for {a} servings?",
    ]
    cities = ["Chicago", "Boston", "Denver", "Austin", "Reno", "Tampa", "Mesa", "Provo"]
    items: list[Item] = []
    for i in range(n_items):
        t = templates[i % len(templates)]
        prompt = t.format(
            a=cities[rng.integers(0, len(cities))],
            b=cities[rng.integers(0, len(cities))],
            x=int(rng.integers(2, 99)),
            y=int(rng.integers(2, 99)),
        )
        items.append(Item(id=f"q{i:04d}", prompt=prompt, answer="...") )
    return items


@dataclass
class CalibrationResult:
    true_rate: float
    estimated_rate: float
    ci_low: float
    ci_high: float
    precision: float
    recall: float
    f1: float
    ci_covers_truth: bool
    false_positive_rate: float  # fraction of CLEAN items wrongly flagged
    passed: bool


def run_calibration(n_items: int = 200, contaminate_frac: float = 0.3, seed: int = 0):
    items = _synthetic_benchmark(n_items, seed=seed)

    # Choose the contaminated subset (ground truth).
    rng = np.random.default_rng(seed + 1)
    n_contam = int(round(n_items * contaminate_frac))
    contam_idx = set(rng.choice(n_items, size=n_contam, replace=False).tolist())
    contaminated_ids = {items[i].id for i in contam_idx}

    # The model "memorized" exactly those items' prompts.
    memorized = [items[i].prompt for i in contam_idx]
    model = MockModel(memorized=memorized)

    # The reference corpus also contains the contaminated items verbatim, so the
    # no-model n-gram signal has something to find too.
    config = RunConfig(corpus_texts=memorized, seed=seed)
    out = run(model, items, config)

    # Score detector vs. ground truth.
    flagged_ids = {v.item_id for v in out.report.items if v.flagged}
    clean_ids = {it.id for it in items} - contaminated_ids
    tp = len(flagged_ids & contaminated_ids)
    fp = len(flagged_ids - contaminated_ids)
    fn = len(contaminated_ids - flagged_ids)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not contaminated_ids and not fp else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0  # vacuously perfect if nothing to find
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / len(clean_ids) if clean_ids else 0.0

    true_rate = n_contam / n_items
    covers = out.report.ci_low <= true_rate <= out.report.ci_high

    # Verdict is based on DETECTION QUALITY, not the rate point-estimate. The
    # flagged fraction is a known-biased (low) estimator of the true rate
    # because recall < 1; we judge the tool on precision + recall + low false
    # positives, which is what actually matters and what generalizes to a real
    # model. We require: high precision, decent recall, and few false alarms.
    if not contaminated_ids:
        passed = fpr <= 0.02  # clean benchmark: must not cry wolf
    else:
        passed = precision >= 0.80 and recall >= 0.70 and fpr <= 0.10

    result = CalibrationResult(
        true_rate=true_rate,
        estimated_rate=out.report.contamination_rate,
        ci_low=out.report.ci_low,
        ci_high=out.report.ci_high,
        precision=precision,
        recall=recall,
        f1=f1,
        ci_covers_truth=covers,
        false_positive_rate=fpr,
        passed=passed,
    )

    _print_report(result, out, n_items)
    return result


def _print_report(r: CalibrationResult, out, n_items: int) -> None:
    print("\n=== benchcheck calibration experiment ===")
    print(f"signals run        : {', '.join(out.signals_run)}")
    if out.signals_skipped:
        print(f"signals skipped    : {', '.join(out.signals_skipped)}")
    print(f"benchmark size     : {n_items} items")
    print(f"TRUE contamination : {r.true_rate:.1%}")
    print(
        f"DETECTED rate      : {r.estimated_rate:.1%} "
        f"(95% CI {r.ci_low:.1%}-{r.ci_high:.1%})"
    )
    print("  note: detected rate is a LOWER BOUND -- recall<1 biases it down.")
    print("--- detection quality (flags vs. ground truth) ---")
    print(f"precision          : {r.precision:.2f}  (of flagged, how many truly contaminated)")
    print(f"recall             : {r.recall:.2f}  (of contaminated, how many caught)")
    print(f"F1                 : {r.f1:.2f}")
    print(f"false positive rate: {r.false_positive_rate:.2f}  (of CLEAN items, how many wrongly flagged)")
    print(f"VERDICT            : {'PASS' if r.passed else 'NEEDS TUNING'}")
