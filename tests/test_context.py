"""The cache question, answered by comparing bytes rather than calling a model."""

from __future__ import annotations

import pytest

from aihe.context import (
    Turn,
    cache_hit_ratio,
    cap_tool_output,
    capped_history,
    common_prefix_length,
    estimate_cost,
    keep_all,
    planted_fact_probe,
    prefix_is_stable,
    render,
    summarise_old,
)

CONVERSATION = [
    Turn("user", "מה תוקף הטוקן"),
    Turn("assistant", "שישים דקות"),
    Turn("user", "ומה קורה אחר כך"),
    Turn("assistant", "צריך לחדש"),
]


def test_appending_keeps_the_prefix_stable():
    """The whole reason keeping everything can be cheaper than summarising."""
    before = render(CONVERSATION[:2])
    after = render(CONVERSATION)
    assert prefix_is_stable(before, after)
    assert cache_hit_ratio(before, after) == 1.0


def test_summarising_destroys_the_prefix():
    before = render(CONVERSATION)
    after = render(summarise_old(CONVERSATION, keep_recent=1, summary="דיברנו על טוקנים"))
    assert not prefix_is_stable(before, after)
    assert cache_hit_ratio(before, after) < 0.1


def test_keep_all_changes_nothing():
    assert keep_all(CONVERSATION) == list(CONVERSATION)


def test_summarising_shortens_the_conversation():
    summarised = summarise_old(CONVERSATION, keep_recent=1, summary="קצר")
    assert len(summarised) == 2
    assert len(render(summarised)) < len(render(CONVERSATION))


def test_summarising_is_a_no_op_when_there_is_nothing_old():
    assert summarise_old(CONVERSATION, keep_recent=10, summary="x") == list(CONVERSATION)


def test_keep_recent_must_not_be_negative():
    with pytest.raises(ValueError):
        summarise_old(CONVERSATION, keep_recent=-1, summary="x")


def test_common_prefix_length_is_exact():
    assert common_prefix_length("abcdef", "abcXYZ") == 3
    assert common_prefix_length("abc", "abc") == 3
    assert common_prefix_length("", "abc") == 0


def test_an_empty_previous_rendering_is_fully_cached():
    assert cache_hit_ratio("", "anything") == 1.0


# --- capping ------------------------------------------------------------------------------


def test_a_short_output_is_left_alone():
    assert cap_tool_output("short", 100) == "short"


def test_a_long_output_is_cut_and_marked():
    capped = cap_tool_output("x" * 500, 20)
    assert capped.startswith("x" * 20) and "truncated" in capped


def test_the_limit_must_be_positive():
    with pytest.raises(ValueError):
        cap_tool_output("x", 0)


def test_capping_on_write_keeps_the_prefix_stable():
    """The cheap win: a cap applied as the output is appended shortens the history without
    ever rewriting text that was already sent."""
    history = [Turn("user", "שאל"), Turn("tool", "y" * 400)]
    capped = capped_history(history, limit=50)
    grown = capped + [Turn("user", "עוד"), Turn("tool", "z" * 400)]
    assert prefix_is_stable(render(capped), render(capped_history(grown, limit=50)))


def test_capping_only_touches_tool_output():
    history = [Turn("user", "u" * 400), Turn("tool", "t" * 400)]
    capped = capped_history(history, limit=50)
    assert len(capped[0].content) == 400
    assert len(capped[1].content) < 400


# --- cost ---------------------------------------------------------------------------------


def test_a_cached_prefix_is_cheaper_than_a_fresh_one():
    text = render(CONVERSATION)
    assert estimate_cost(text, cached_prefix=text) < estimate_cost(text, cached_prefix="")


def test_a_broken_prefix_costs_full_price():
    text = render(CONVERSATION)
    assert estimate_cost(text, cached_prefix="something else") == pytest.approx(
        estimate_cost(text, cached_prefix=""))


# --- the harness --------------------------------------------------------------------------


def test_the_probe_puts_the_fact_first_and_the_question_last():
    turns = planted_fact_probe("הקוד הוא 4417", "מה הקוד", ["א", "ב"])
    assert turns[0].content == "הקוד הוא 4417"
    assert turns[-1].content == "מה הקוד"
    assert len(turns) == 2 + 2 * 2 + 1


def test_the_filler_is_tool_output_so_a_cap_can_touch_it():
    """A cap may only shorten tool results. Filler that is not tool output would make the
    capping strategy a no-op and the whole comparison meaningless."""
    turns = planted_fact_probe("fact", "probe", ["x" * 400])
    assert any(t.role == "tool" for t in turns)
    assert len(render(capped_history(turns, limit=50))) < len(render(turns))


def test_capping_never_touches_the_planted_fact():
    """If the cap could truncate the fact, the experiment would measure the cap rather than
    the context strategy."""
    turns = planted_fact_probe("the code is 4417", "what is the code", ["y" * 400])
    assert "the code is 4417" in render(capped_history(turns, limit=20))
