"""Tests for the corpus index, including C++/Python backend agreement."""
import pytest

from benchcheck import corpus_index as ci

CORPUS = [
    "the quick brown fox jumps over the lazy sleeping dog in the green park today",
    "photosynthesis converts sunlight into chemical energy stored inside plant cells",
]
HIT = "the quick brown fox jumps over the lazy sleeping dog in the green park today"
MISS = "an entirely unrelated sentence about quarterly financial earnings and revenue"


def test_python_index_overlap():
    idx = ci.build_index(CORPUS, n=8, prefer_native=False)
    assert idx.backend == "python"
    assert idx.overlap(HIT) == 1.0
    assert idx.overlap(MISS) == 0.0


def test_empty_query():
    idx = ci.build_index(CORPUS, n=8, prefer_native=False)
    assert idx.overlap("") == 0.0


@pytest.mark.skipif(not ci.native_available(), reason="C++ extension not built")
def test_cpp_matches_python():
    py = ci.build_index(CORPUS, n=8, prefer_native=False)
    cpp = ci.build_index(CORPUS, n=8, prefer_native=True)
    assert cpp.backend == "cpp"
    for q in [HIT, MISS, "the quick brown fox", "photosynthesis converts sunlight"]:
        assert abs(py.overlap(q) - cpp.overlap(q)) < 1e-9


def test_build_index_prefers_native_when_available():
    idx = ci.build_index(CORPUS, n=8, prefer_native=True)
    expected = "cpp" if ci.native_available() else "python"
    assert idx.backend == expected
