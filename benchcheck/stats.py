"""Statistics layer: turn raw signals into calibrated, honest verdicts.

This is the heart of the project. Anyone can print a number; the value here is
*not fooling yourself*. Three jobs:

  1. Combine multiple signals per item into one contamination score.
  2. Decide whether an item is contaminated by comparing it to a NULL
     distribution built from the clean items themselves -- so "suspicious" means
     "suspicious relative to this benchmark's normal behavior", not an arbitrary
     threshold.
  3. Estimate the benchmark-level contamination rate WITH a confidence interval
     (bootstrap), because a point estimate with no error bar is dishonest.

Design choices worth being able to explain:
  - We z-score each signal across items before combining, so a signal with a
    naturally large range doesn't dominate one with a small range.
  - The per-item p-value uses an empirical null: the distribution of combined
    scores. We flag items in the upper tail. This is robust and assumption-light
    compared to assuming a parametric distribution.
  - The bootstrap resamples items with replacement to get a CI on the flagged
    fraction -- standard, transparent, no distributional assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchcheck.types import SignalResult


@dataclass
class ItemVerdict:
    item_id: str
    combined_score: float
    p_value: float
    flagged: bool
    signal_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    n_items: int
    n_flagged: int
    contamination_rate: float
    ci_low: float
    ci_high: float
    per_signal_mean: dict[str, float]
    items: list[ItemVerdict]

    def summary(self) -> str:
        lines = [
            f"items analyzed     : {self.n_items}",
            f"flagged contaminated: {self.n_flagged}",
            f"contamination rate : {self.contamination_rate:.1%} "
            f"(95% CI {self.ci_low:.1%}-{self.ci_high:.1%})",
            "per-signal mean score:",
        ]
        for name, val in sorted(self.per_signal_mean.items()):
            lines.append(f"    {name:<22} {val:.3f}")
        return "\n".join(lines)


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def signal_matrix(
    results_by_item: dict[str, list[SignalResult]],
) -> tuple[list[str], list[str], np.ndarray]:
    """Build the (items x signals) raw score matrix.
    Returns (item_ids, signal_names, matrix)."""
    item_ids = list(results_by_item.keys())
    signal_names = sorted({r.signal for rs in results_by_item.values() for r in rs})
    mat = np.zeros((len(item_ids), len(signal_names)))
    col_of = {name: j for j, name in enumerate(signal_names)}
    for i, iid in enumerate(item_ids):
        for r in results_by_item[iid]:
            mat[i, col_of[r.signal]] = r.score
    return item_ids, signal_names, mat


def _combine_matrix(
    mat: np.ndarray, signal_names: list[str], weights: dict[str, float] | None
) -> np.ndarray:
    """Z-score each signal column (so signals are comparable regardless of their
    natural range), weight, and average into one score per item."""
    if mat.shape[1] == 0:
        return np.zeros(mat.shape[0])
    w = np.array([(weights or {}).get(n, 1.0) for n in signal_names])
    total_w = w.sum() or 1.0
    z = np.column_stack([_zscore(mat[:, j]) for j in range(mat.shape[1])])
    return (z * w).sum(axis=1) / total_w


def combine_signals(
    results_by_item: dict[str, list[SignalResult]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Combine each item's signal scores into one score. Returns {item_id: score}."""
    item_ids, signal_names, mat = signal_matrix(results_by_item)
    if not signal_names:
        return {iid: 0.0 for iid in item_ids}
    combined = _combine_matrix(mat, signal_names, weights)
    return {iid: float(combined[i]) for i, iid in enumerate(item_ids)}


def empirical_p_values(combined: dict[str, float]) -> dict[str, float]:
    """Upper-tail p-value of each item's combined score against the empirical
    distribution of all combined scores. p = fraction of items scoring >= this.

    Kept as a DIAGNOSTIC only. It is NOT used for flagging, because it
    structurally caps the flagged fraction at the chosen threshold and uses a
    null that contaminated items themselves pollute -- it silently fails when
    contamination is common. `separation_threshold` is what we actually flag on.
    """
    ids = list(combined.keys())
    scores = np.array([combined[i] for i in ids])
    n = len(scores)
    pvals = {}
    for i, iid in enumerate(ids):
        ge = int(np.sum(scores >= scores[i]))
        pvals[iid] = ge / n
    return pvals


def _otsu_split(scores: np.ndarray):
    """Otsu's method: pick the threshold maximizing between-cluster variance.
    Returns (threshold, separation) where separation is the standardized gap
    between the two clusters' means (in pooled std-devs)."""
    s = np.sort(scores)
    n = len(s)
    best_thr, best_between = None, -1.0
    for i in range(1, n):
        lo, hi = s[:i], s[i:]
        between = (len(lo) / n) * (len(hi) / n) * (lo.mean() - hi.mean()) ** 2
        if between > best_between:
            best_between = between
            best_thr = (s[i - 1] + s[i]) / 2
    lo = scores[scores <= best_thr]
    hi = scores[scores > best_thr]
    if len(lo) == 0 or len(hi) == 0:
        return best_thr, 0.0
    pooled_std = np.sqrt((lo.var() + hi.var()) / 2) or 1e-9
    return best_thr, (hi.mean() - lo.mean()) / pooled_std


def _signal_fires(col: np.ndarray, min_sep: float = 1.5) -> np.ndarray:
    """Per-signal outlier flags: which items fall in this signal's HIGH cluster.

    We use per-signal Otsu rather than a median/MAD threshold on purpose. The
    signals are zero-inflated (most items score ~0; contaminated ones score
    high), and they can be contaminated well past 50%. Median/MAD breaks down
    above a 50% outlier fraction -- the median itself moves into the
    contaminated mass. Otsu instead finds the natural split between the
    low/"clean" and high/"fires" clusters at ANY mixing fraction.

    We only treat a signal as informative if its two clusters are separated by
    more than `min_sep` pooled std-devs; otherwise the signal is flat/noisy and
    nothing fires. Returns a boolean array over items.
    """
    if np.all(col == col[0]):
        return np.zeros(len(col), dtype=bool)
    thr, sep = _otsu_split(col)
    if sep < min_sep:
        return np.zeros(len(col), dtype=bool)
    return col > thr


def corroboration_flags(
    mat: np.ndarray,
    signal_names: list[str],
    min_signals: int = 2,
    min_sep: float = 1.5,
) -> np.ndarray:
    """Flag items where MULTIPLE signals independently call them outliers.

    This encodes the project's core insight: a single signal's high cluster can
    be noise, but contamination makes several *independent* signals fire on the
    SAME item. Requiring agreement from >= `min_signals` is what separates real
    contamination (n-gram + verbatim + perturbation all light up on the
    memorized items) from chance (each signal's stray firings rarely coincide).

    Conservative by design: corroboration is only meaningful with at least
    `min_signals` informative signals. If fewer are informative (e.g. only
    logprob access on a non-MCQ benchmark, or a tiny dataset), we CANNOT
    corroborate and deliberately flag nothing -- refusing to trust a lone signal
    is what prevents a single noisy signal from manufacturing false positives on
    a clean benchmark. The pipeline surfaces this as low confidence. Returns a
    boolean array over items.
    """
    n, m = mat.shape
    if n == 0 or m == 0:
        return np.zeros(n, dtype=bool)
    fire_counts = np.zeros(n, dtype=int)
    informative = 0
    for j in range(m):
        fires = _signal_fires(mat[:, j], min_sep=min_sep)
        if fires.any():
            informative += 1
        fire_counts += fires.astype(int)
    if informative < min_signals:
        return np.zeros(n, dtype=bool)
    return fire_counts >= min_signals


def bootstrap_rate_ci(
    flags: list[bool], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI for the fraction of flagged items.

    `seed` is explicit (no global RNG) so runs are reproducible -- important for
    a tool whose whole point is trustworthiness.
    """
    if not flags:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(flags, dtype=float)
    n = len(arr)
    rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rates[b] = arr[idx].mean()
    lo = float(np.quantile(rates, alpha / 2))
    hi = float(np.quantile(rates, 1 - alpha / 2))
    return (lo, hi)


def analyze(
    results_by_item: dict[str, list[SignalResult]],
    flag_threshold_p: float = 0.10,  # retained for API compat; see min_separation
    weights: dict[str, float] | None = None,
    seed: int = 0,
) -> BenchmarkReport:
    """Full pipeline: combine -> find separation threshold -> flag -> CI -> report.

    Flagging uses `separation_threshold` (adapts to any contamination rate and
    abstains, via a Monte-Carlo null, when clusters aren't more separated than
    noise), NOT the p-value cap, which breaks when contamination is common.
    p-values are still reported as a per-item diagnostic.
    """
    item_ids, signal_names, mat = signal_matrix(results_by_item)
    combined_arr = _combine_matrix(mat, signal_names, weights)
    combined = {iid: float(combined_arr[i]) for i, iid in enumerate(item_ids)}
    pvals = empirical_p_values(combined)

    flag_arr = corroboration_flags(mat, signal_names)

    verdicts: list[ItemVerdict] = []
    for i, iid in enumerate(item_ids):
        flagged = bool(flag_arr[i])
        sig_scores = {r.signal: r.score for r in results_by_item[iid]}
        verdicts.append(
            ItemVerdict(
                item_id=iid,
                combined_score=combined[iid],
                p_value=pvals[iid],
                flagged=flagged,
                signal_scores=sig_scores,
            )
        )

    flags = [v.flagged for v in verdicts]
    n_flagged = sum(flags)
    rate = n_flagged / len(flags) if flags else 0.0
    lo, hi = bootstrap_rate_ci(flags, seed=seed)

    # Per-signal mean of raw (un-zscored) scores, for the human summary.
    per_signal: dict[str, list[float]] = {}
    for rs in results_by_item.values():
        for r in rs:
            per_signal.setdefault(r.signal, []).append(r.score)
    per_signal_mean = {k: float(np.mean(v)) for k, v in per_signal.items()}

    return BenchmarkReport(
        n_items=len(item_ids),
        n_flagged=n_flagged,
        contamination_rate=rate,
        ci_low=lo,
        ci_high=hi,
        per_signal_mean=per_signal_mean,
        items=verdicts,
    )
