"""Reading a judge's verdict, and measuring whether the judge is worth anything."""

from __future__ import annotations

import pytest

from aihe.judge import (
    Judgement,
    agreement,
    binary_judge,
    contains_fact,
    holistic_judge,
    read_score,
    read_yes_no,
)


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Yes, it does.", True), ("כן", True),
    ("no", False), ("No, it does not.", False), ("לא", False),
])
def test_a_clear_verdict_is_read(text, expected):
    assert read_yes_no(text) is expected


def test_the_first_word_wins_when_a_reply_hedges():
    assert read_yes_no("Yes, although no in some cases") is True
    assert read_yes_no("No, but yes if you squint") is False


def test_waffling_is_reported_rather_than_coerced():
    """Treating 'it depends' as a no is how a judge acquires a bias nobody measured."""
    assert read_yes_no("it depends on what you mean") is None


def test_a_score_is_read_from_prose():
    assert read_score("I would say 4 out of 5") == 4
    assert read_score("no number here") is None


def test_a_judgement_knows_whether_it_decided():
    assert Judgement(True, "yes").decided
    assert not Judgement(None, "hmm").decided


def test_the_binary_judge_asks_about_one_criterion():
    seen = {}

    def ask(prompt):
        seen["prompt"] = prompt
        return "yes"

    result = binary_judge("מה תוקף הטוקן", "שישים דקות", "the token lasts sixty minutes", ask)
    assert result.value is True
    assert "sixty minutes" in seen["prompt"] and "yes or no" in seen["prompt"]


def test_the_holistic_judge_asks_for_a_number():
    result = holistic_judge("q", "a", lambda prompt: "5")
    assert result.value == 5


def test_a_judge_that_says_yes_to_everything_is_caught():
    """Accuracy alone would flatter it whenever most answers happen to be good. The
    false-positive rate is what exposes it."""
    judgements = [Judgement(True, "yes") for _ in range(10)]
    truth = [True] * 8 + [False] * 2
    stats = agreement(judgements, truth)
    assert stats["accuracy"] == pytest.approx(0.8)
    assert stats["said_yes"] == 1.0
    assert stats["false_positive"] == 1.0


def test_a_good_judge_scores_well_everywhere():
    truth = [True, True, False, False]
    judgements = [Judgement(t, str(t)) for t in truth]
    stats = agreement(judgements, truth)
    assert stats["accuracy"] == 1.0
    assert stats["false_positive"] == 0.0 and stats["false_negative"] == 0.0


def test_undecided_verdicts_are_counted_not_guessed():
    stats = agreement([Judgement(True, "y"), Judgement(None, "hmm")], [True, True])
    assert stats["n"] == 1 and stats["undecided"] == 1


def test_a_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        agreement([Judgement(True, "y")], [True, False])


def test_all_undecided_does_not_divide_by_zero():
    assert agreement([Judgement(None, "?")], [True])["n"] == 0


def test_the_judge_with_no_model_in_it():
    """Crude, free, instant, and impossible to flatter - which is why it is the ground truth
    the model judges get measured against."""
    assert contains_fact("The token lasts sixty  minutes.", "sixty minutes")
    assert not contains_fact("The token expires eventually.", "sixty minutes")
