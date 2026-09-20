"""Chapter 05 asserts its own claims, without calling a model."""

from __future__ import annotations

import json
from pathlib import Path

from aihe.cassettes import CassetteLibrary
from aihe.tools import Parameter, Tool, validate

HERE = Path(__file__).resolve().parents[1]
PLAN = json.loads((HERE / "data" / "tool-tasks.json").read_text(encoding="utf-8"))


def _tools():
    return [Tool(s["name"], s["description"], tuple(Parameter(**p) for p in s["parameters"]))
            for s in PLAN["tools"]]


def test_there_are_two_descriptions_of_the_same_capability():
    tools = _tools()
    assert len(tools) == 2
    names = {p.name for p in tools[0].parameters}
    assert names <= {p.name for p in tools[1].parameters}


def test_one_description_is_short_and_the_other_is_long():
    """Length is the variable under test, so it has to be a real difference."""
    short, long = (t.description for t in _tools())
    assert len(short) < 100
    assert len(long) > 500


def test_every_parameter_is_primitive():
    """Nested objects are what make a small model start inventing structure."""
    for tool in _tools():
        assert all(p.type in {"string", "integer", "number", "boolean"} for p in tool.parameters)


def test_the_tasks_never_mention_json():
    """A task that tells the model how to answer measures the prompt, not the model."""
    for task in PLAN["tasks"]:
        assert "json" not in task.lower()


def test_there_are_eight_tasks_and_a_bounded_attempt_count():
    assert len(PLAN["tasks"]) == 8
    assert 1 < PLAN["max_attempts"] <= 5


def test_null_for_an_optional_key_is_the_failure_the_chapter_is_about():
    """89% of first-attempt failures. Pinned in both directions so the claim stays true."""
    tool = _tools()[0]
    optional = next(p.name for p in tool.parameters if not p.required)
    required = next(p.name for p in tool.parameters if p.required)
    assert validate(tool, {required: "x", optional: None}).ok
    assert not validate(tool, {required: "x", optional: None}, null_is_absent=False).ok


def test_the_cassettes_are_present_and_pinned():
    library = CassetteLibrary(HERE / "cassettes")
    keys = library.keys()
    assert len(keys) >= 25
    assert library.load(keys[0])["model"] == "Llama-3.2-3B-Instruct-Q4_K_M"
