"""Tests for perturbations and dataset round-tripping."""
import tempfile
from pathlib import Path

from benchcheck import perturb
from benchcheck.dataset import load_jsonl, save_jsonl
from benchcheck.types import Item


def test_paraphrases_preserve_nothing_identical():
    text = "If a train leaves Chicago, when do they meet?"
    paras = perturb.paraphrases(text)
    assert paras  # produced something
    assert all(p.lower() != text.lower() for p in paras)


def test_whitespace_normalize_is_near_identity():
    assert perturb.whitespace_normalize("a   b\tc\n") == "a b c"


def test_shuffled_orders_are_permutations():
    choices = ["a", "b", "c", "d"]
    orders = perturb.shuffled_choice_orders(choices)
    assert orders
    for o in orders:
        assert sorted(o) == [0, 1, 2, 3]
        assert o != [0, 1, 2, 3]  # actually shuffled


def test_shuffled_orders_empty_for_single_choice():
    assert perturb.shuffled_choice_orders(["only"]) == []


def test_dataset_roundtrip():
    items = [
        Item(id="a", prompt="Q1?", answer="B", choices=["x", "y"]),
        Item(id="b", prompt="Q2?"),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "data.jsonl"
        save_jsonl(items, path)
        loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].id == "a"
    assert loaded[0].choices == ["x", "y"]
    assert loaded[1].answer is None
