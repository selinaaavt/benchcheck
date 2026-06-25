"""Model backends. Construct via `load_model(spec)`."""
from __future__ import annotations

from benchcheck.models.base import Capability, capabilities_of  # noqa: F401


def load_model(spec: str):
    """Build a model backend from a CLI spec string.

    Specs:
      - "mock"            -> deterministic MockModel with nothing memorized
      - "mock:clean"      -> same as "mock"
      - "hf:<name>"       -> HuggingFace causal LM (needs torch+transformers)

    The mock with a memorized set is constructed directly in code (e.g. the
    calibration experiment), not via spec string.
    """
    if spec == "mock" or spec == "mock:clean":
        from benchcheck.models.mock import MockModel

        return MockModel()
    if spec.startswith("hf:"):
        from benchcheck.models.hf import HFModel

        return HFModel(spec[len("hf:") :])
    raise ValueError(f"unknown model spec: {spec!r} (try 'mock' or 'hf:gpt2')")
