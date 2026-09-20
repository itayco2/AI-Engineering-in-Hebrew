"""Tool schemas, validation, and the bounded repair loop. No model involved."""

from __future__ import annotations

import pytest

from aihe.tools import Parameter, Tool, repair, summarise, validate

SEARCH = Tool(
    name="search_docs",
    description="Find documentation.",
    parameters=(
        Parameter("query", "string", "what to look for"),
        Parameter("limit", "integer", "how many results", required=False),
        Parameter("scope", "string", "where to search", required=False, enum=("api", "guides")),
    ),
)


def test_a_parameter_must_be_a_primitive_type():
    """Nested objects are exactly what makes a small model start inventing structure."""
    with pytest.raises(ValueError, match="primitive"):
        Parameter("payload", "object")


def test_the_schema_is_the_shape_providers_expect():
    schema = SEARCH.schema()["function"]
    assert schema["name"] == "search_docs"
    assert set(schema["parameters"]["properties"]) == {"query", "limit", "scope"}
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["scope"]["enum"] == ["api", "guides"]


def test_the_prompt_hint_names_every_parameter():
    """Small models frequently ignore the tools parameter, so the prompt has to ask too."""
    hint = SEARCH.prompt_hint()
    assert all(p.name in hint for p in SEARCH.parameters)
    assert "JSON only" in hint


def test_a_correct_call_validates():
    result = validate(SEARCH, {"query": "tokens", "limit": 3, "scope": "api"})
    assert result.ok and result.arguments["limit"] == 3


def test_optional_parameters_may_be_omitted():
    assert validate(SEARCH, {"query": "tokens"}).ok


def test_a_missing_required_key_is_named():
    errors = validate(SEARCH, {"limit": 3}).errors
    assert any("missing required key `query`" in e for e in errors)


def test_a_wrong_type_is_named_with_what_arrived():
    """The message goes straight back to the model, so it has to say what to fix."""
    errors = validate(SEARCH, {"query": "x", "limit": "three"}).errors
    assert any("`limit` must be a integer" in e and "str" in e for e in errors)


def test_a_boolean_is_not_an_integer():
    """bool subclasses int in Python, so a naive isinstance check accepts True for a count."""
    assert not validate(SEARCH, {"query": "x", "limit": True}).ok


def test_a_value_outside_the_enum_is_rejected():
    errors = validate(SEARCH, {"query": "x", "scope": "everywhere"}).errors
    assert any("must be one of" in e for e in errors)


def test_an_invented_key_is_rejected():
    """Small models add keys nobody asked for; silently dropping them hides the failure."""
    assert any("unknown key `page`" in e for e in validate(SEARCH, {"query": "x", "page": 2}).errors)


def test_a_non_object_is_rejected_clearly():
    assert "expected a JSON object" in validate(SEARCH, [1, 2, 3]).errors[0]


def test_feedback_joins_every_error():
    result = validate(SEARCH, {"limit": "x", "page": 1})
    assert result.feedback().count(";") >= 1


# --- the repair loop ----------------------------------------------------------------------


def test_a_good_first_answer_needs_no_repair():
    result = repair(SEARCH, lambda _: '{"query": "tokens"}')
    assert result.ok and result.tries == 1


def test_a_broken_answer_is_repaired_using_the_feedback():
    replies = ['I think {"quer": "tokens"}', '{"query": "tokens"}']
    result = repair(SEARCH, lambda feedback: replies.pop(0))
    assert result.ok and result.tries == 2
    assert not result.attempts[0].valid and result.attempts[1].valid


def test_the_feedback_from_the_previous_attempt_is_passed_back():
    seen = []

    def ask(feedback):
        seen.append(feedback)
        return "not json at all" if len(seen) == 1 else '{"query": "x"}'

    repair(SEARCH, ask)
    assert seen[0] is None
    assert "no JSON object found" in seen[1]


def test_the_loop_is_bounded():
    """An unbounded loop against a model that cannot satisfy the schema is an infinite bill."""
    result = repair(SEARCH, lambda _: "never json", max_attempts=3)
    assert not result.ok and result.tries == 3


def test_every_attempt_is_kept_so_the_chapter_can_show_them():
    result = repair(SEARCH, lambda _: "nope", max_attempts=2)
    assert [a.number for a in result.attempts] == [1, 2]
    assert all(a.raw == "nope" and a.parsed is None for a in result.attempts)


def test_json_buried_in_prose_still_counts():
    result = repair(SEARCH, lambda _: 'Sure! ```json\n{"query": "x"}\n``` hope that helps')
    assert result.ok


def test_summarise_reports_the_numbers_a_chapter_prints():
    good = repair(SEARCH, lambda _: '{"query": "x"}')
    late = repair(SEARCH, lambda f: 'bad' if f is None else '{"query": "x"}')
    never = repair(SEARCH, lambda _: "bad", max_attempts=2)
    stats = summarise([good, late, never])
    assert stats["n"] == 3
    assert stats["first_try"] == pytest.approx(1 / 3)
    assert stats["eventually"] == pytest.approx(2 / 3)
    assert stats["never"] == pytest.approx(1 / 3)


def test_summarising_nothing_does_not_divide_by_zero():
    assert summarise([])["n"] == 0


# --- null for an optional key: 89% of every first-attempt failure in chapter 05 -----------


def test_null_for_an_optional_key_is_treated_as_absent():
    """A model answering `{"limit": null}` has said 'not this one'. The schema already says
    the key is optional, so refusing the answer is pedantry with a measurable price."""
    assert validate(SEARCH, {"query": "tokens", "limit": None}).ok


def test_the_lenient_reading_can_be_turned_off():
    """Chapter 05 needs the strict behaviour to show what it costs."""
    result = validate(SEARCH, {"query": "tokens", "limit": None}, null_is_absent=False)
    assert not result.ok
    assert "NoneType" in result.feedback()


def test_a_null_optional_key_does_not_survive_into_the_arguments():
    result = validate(SEARCH, {"query": "tokens", "limit": None, "scope": None})
    assert result.ok and result.arguments == {"query": "tokens"}


def test_a_required_key_answered_null_is_still_an_error():
    """That is a refusal to do the job, not a declined option."""
    assert not validate(SEARCH, {"query": None}).ok


def test_an_unknown_key_answered_null_is_still_unknown():
    assert not validate(SEARCH, {"query": "x", "page": None}).ok


def test_leniency_does_not_rescue_a_real_type_error():
    assert not validate(SEARCH, {"query": "x", "limit": "three"}).ok
