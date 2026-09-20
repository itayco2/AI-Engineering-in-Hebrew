"""Chapter 03 asserts its own claims. The ground truth is checked here, not assumed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aihe.cassettes import CassetteLibrary
from aihe.judge import build_cases, contains_fact
from aihe.metrics import ndcg_at_k, recall_at_k

HERE = Path(__file__).resolve().parents[1]
CASES = json.loads((HERE / "data" / "judge-cases.json").read_text(encoding="utf-8"))["cases"]


def test_there_are_twelve_questions_and_twenty_four_answers():
    assert len(CASES) == 12
    assert len(build_cases(CASES)) == 24


def test_every_kept_answer_really_contains_its_fact():
    """The chapter measures judges against this. If it is wrong, everything downstream is."""
    missing = [c["question"] for c in CASES if not contains_fact(c["kept"], c["fact"])]
    assert not missing, f"kept answers missing their fact: {missing}"


def test_no_lost_answer_contains_its_fact():
    leaked = [c["question"] for c in CASES if contains_fact(c["lost"], c["fact"])]
    assert not leaked, f"lost answers that still state the fact: {leaked}"


def test_the_lost_answers_are_not_obviously_bad():
    """They have to be plausible, or the chapter is measuring whether a judge can spot
    gibberish rather than whether it can spot a missing fact."""
    for case in CASES:
        assert len(case["lost"]) > 80
        assert case["lost"] != case["kept"]


def test_every_case_has_a_checkable_criterion():
    for case in CASES:
        assert len(case["criterion"]) > 10
        assert not case["criterion"].endswith("?")


def test_the_cassettes_are_present_and_pinned_to_their_model():
    library = CassetteLibrary(HERE / "cassettes")
    keys = library.keys()
    assert len(keys) == 48, f"expected 48 recorded replies, found {len(keys)}"
    assert library.load(keys[0])["model"] == "Llama-3.2-3B-Instruct-Q4_K_M"


def test_the_metrics_the_chapter_teaches_behave_as_it_says():
    """recall cannot tell rank 1 from rank 3; nDCG can. That is the whole distinction."""
    early, late = [7, 1, 2], [1, 2, 7]
    assert recall_at_k(early, {7}, 3) == recall_at_k(late, {7}, 3) == 1.0
    assert ndcg_at_k(early, {7}, 3) > ndcg_at_k(late, {7}, 3)
