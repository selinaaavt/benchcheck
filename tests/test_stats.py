"""Tests for the statistics layer: combination, flagging, bootstrap CI."""
import numpy as np

from benchcheck import stats
from benchcheck.types import SignalResult


def _results(rows):
    """rows: list of (item_id, {signal: score}) -> results_by_item dict."""
    out = {}
    for iid, scores in rows:
        out[iid] = [SignalResult(iid, name, val) for name, val in scores.items()]
    return out


def test_otsu_splits_bimodal():
    data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    thr, sep = stats._otsu_split(data)
    assert 0.0 < thr < 1.0
    assert sep > 2.0


def test_signal_fires_picks_high_cluster():
    col = np.array([0.0] * 8 + [0.9, 0.95])
    fires = stats._signal_fires(col)
    assert fires[-1] and fires[-2]
    assert not fires[0]


def test_signal_fires_flat_column():
    col = np.zeros(10)
    assert not stats._signal_fires(col).any()


def test_corroboration_requires_two_signals():
    # Item A fires on two signals; item B on only one. Only A should flag.
    mat = np.array(
        [
            [0.9, 0.9, 0.0],  # A: two signals high
            [0.9, 0.0, 0.0],  # B: one signal high
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    flags = stats.corroboration_flags(mat, ["s1", "s2", "s3"], min_signals=2)
    assert flags[0]
    assert not flags[1]


def test_corroboration_silent_when_one_informative_signal():
    # Only one signal has any structure; cannot corroborate -> flag nothing.
    mat = np.zeros((10, 3))
    mat[:3, 0] = 1.0
    flags = stats.corroboration_flags(mat, ["s1", "s2", "s3"], min_signals=2)
    assert not flags.any()


def test_bootstrap_ci_brackets_rate():
    flags = [True] * 30 + [False] * 70
    lo, hi = stats.bootstrap_rate_ci(flags, seed=0)
    assert lo <= 0.30 <= hi


def test_analyze_flags_corroborated_items():
    rows = []
    # 4 clearly contaminated items (two signals high), 16 clean.
    for i in range(4):
        rows.append((f"c{i}", {"ngram_overlap": 0.95, "verbatim_completion": 0.9}))
    for i in range(16):
        rows.append((f"k{i}", {"ngram_overlap": 0.0, "verbatim_completion": 0.0}))
    report = stats.analyze(_results(rows), seed=0)
    flagged = {v.item_id for v in report.items if v.flagged}
    assert flagged == {f"c{i}" for i in range(4)}
