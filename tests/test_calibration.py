"""Tests for the calibration experiment -- the tool's validation harness.

These lock in the headline guarantees: clean benchmarks stay silent, and
realistic contamination levels are detected with high precision.
"""
from benchcheck.calibration import run_calibration


def test_clean_benchmark_not_flagged():
    r = run_calibration(n_items=200, contaminate_frac=0.0, seed=0)
    assert r.false_positive_rate <= 0.02
    assert r.passed


def test_moderate_contamination_detected():
    r = run_calibration(n_items=200, contaminate_frac=0.3, seed=0)
    assert r.precision >= 0.80
    assert r.recall >= 0.70
    assert r.passed


def test_detected_rate_is_lower_bound_ish():
    # Recall < 1 means the detected rate should not wildly exceed the truth.
    r = run_calibration(n_items=200, contaminate_frac=0.3, seed=0)
    assert r.estimated_rate <= r.true_rate + 0.10


def test_reproducible():
    a = run_calibration(n_items=120, contaminate_frac=0.25, seed=7)
    b = run_calibration(n_items=120, contaminate_frac=0.25, seed=7)
    assert a.precision == b.precision
    assert a.recall == b.recall
