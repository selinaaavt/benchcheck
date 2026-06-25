"""Contamination signals. Each returns a SignalResult where higher = more suspicious.

Importing this package registers all built-in signals in the registry
(benchcheck.signals.base), so the CLI and pipeline can look them up by name.
"""
from benchcheck.signals import (  # noqa: F401  (imported for registration side effects)
    ngram_overlap,
    perturbation_gap,
    shuffle_sensitivity,
    verbatim_completion,
)
from benchcheck.signals.base import (  # noqa: F401
    Signal,
    SignalContext,
    all_signals,
    available_for,
    get,
    register,
)
