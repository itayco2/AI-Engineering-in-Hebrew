"""The client interface, the cassette library, and the JSON salvage that chapter 05 needs."""

import json
import os

import pytest

from aihe.cassettes import CassetteLibrary, CassetteMiss, key_for
from aihe.models import Response, backend_name, chat, parse_json_object

MESSAGES = [{"role": "user", "content": "שלום"}]


# --- backend selection ------------------------------------------------------------------


def test_backend_defaults_to_ollama(monkeypatch):
    monkeypatch.delenv("AIHE_BACKEND", raising=False)
    assert backend_name() == "ollama"


def test_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("AIHE_BACKEND", "  Replay ")
    assert backend_name() == "replay"


def test_unknown_backend_is_rejected_by_name(monkeypatch):
    monkeypatch.setenv("AIHE_BACKEND", "gpt5")
    with pytest.raises(ValueError, match="AIHE_BACKEND"):
        backend_name()


def test_fake_backend_needs_nothing(monkeypatch):
    monkeypatch.setenv("AIHE_BACKEND", "fake")
    reply = chat(MESSAGES)
    assert reply.text
    assert reply.model.startswith("fake/")


def test_replay_without_a_cassette_directory_says_so(monkeypatch):
    monkeypatch.setenv("AIHE_BACKEND", "replay")
    with pytest.raises(ValueError, match="cassettes"):
        chat(MESSAGES)


def test_response_lists_the_tools_it_called():
    reply = Response(text="", model="m", tool_calls=[{"name": "search"}, {"name": "answer"}])
    assert reply.called_tools == ["search", "answer"]


# --- cassettes --------------------------------------------------------------------------


def test_key_is_stable_across_dict_ordering():
    a = key_for("m", [{"role": "user", "content": "hi"}], None, temperature=0, top_p=1)
    b = key_for("m", [{"role": "user", "content": "hi"}], None, top_p=1, temperature=0)
    assert a == b


def test_key_changes_when_the_request_changes():
    base = key_for("m", MESSAGES, None)
    assert key_for("m", [{"role": "user", "content": "אחר"}], None) != base
    assert key_for("other", MESSAGES, None) != base
    assert key_for("m", MESSAGES, [{"name": "t"}]) != base


def test_a_miss_names_the_command_that_fixes_it(tmp_path):
    library = CassetteLibrary(tmp_path)
    with pytest.raises(CassetteMiss, match="make record"):
        library.load("nosuchkey")


def test_saving_then_loading_round_trips(tmp_path):
    library = CassetteLibrary(tmp_path)
    library.save("abc", request={"messages": MESSAGES}, response={"text": "שלום"}, model="llama3.2:3b")
    assert library.has("abc")
    assert library.load("abc")["response"]["text"] == "שלום"


def test_the_model_is_pinned_inside_the_cassette(tmp_path):
    """So bumping a model is a visible diff in review, not a mystery six weeks later."""
    library = CassetteLibrary(tmp_path)
    path = library.save("abc", request={}, response={"text": ""}, model="llama3.2:3b")
    assert json.loads(path.read_text())["model"] == "llama3.2:3b"


def test_cassettes_are_written_as_readable_utf8(tmp_path):
    """Hebrew must be legible in a diff, not escaped into \\u sequences."""
    library = CassetteLibrary(tmp_path)
    path = library.save("k", request={}, response={"text": "שלום"}, model="m")
    assert "שלום" in path.read_text(encoding="utf-8")


def test_replay_reads_what_was_recorded(tmp_path, monkeypatch):
    library = CassetteLibrary(tmp_path)
    library.save(
        key_for("llama3.2:3b", MESSAGES, None),
        request={"messages": MESSAGES},
        response={"text": "מהקלטת", "tool_calls": [{"name": "search"}]},
        model="llama3.2:3b",
    )
    monkeypatch.setenv("AIHE_BACKEND", "replay")
    reply = chat(MESSAGES, cassettes=tmp_path)
    assert reply.text == "מהקלטת"
    assert reply.called_tools == ["search"]


def test_keys_lists_what_is_on_disk(tmp_path):
    library = CassetteLibrary(tmp_path)
    library.save("bbb", {}, {}, "m")
    library.save("aaa", {}, {}, "m")
    assert library.keys() == ["aaa", "bbb"]


# --- salvaging JSON out of a small model's prose ----------------------------------------


def test_plain_json_is_returned():
    assert parse_json_object('{"city": "חיפה"}') == {"city": "חיפה"}


def test_json_wrapped_in_prose_is_found():
    """What a 3B model actually does when you ask it for JSON."""
    reply = 'Sure! Here is the JSON you asked for:\n{"city": "חיפה"}\nHope that helps!'
    assert parse_json_object(reply) == {"city": "חיפה"}


def test_json_in_a_code_fence_is_found():
    assert parse_json_object('```json\n{"n": 1}\n```') == {"n": 1}


def test_nested_objects_survive():
    assert parse_json_object('x {"a": {"b": [1, 2]}} y') == {"a": {"b": [1, 2]}}


def test_braces_inside_strings_do_not_end_the_object():
    assert parse_json_object('{"text": "a } brace"}') == {"text": "a } brace"}


def test_escaped_quotes_inside_strings_survive():
    assert parse_json_object(r'{"text": "he said \"hi\""}') == {"text": 'he said "hi"'}


def test_the_first_valid_object_wins_when_an_earlier_one_is_broken():
    assert parse_json_object('{not json} then {"ok": true}') == {"ok": True}


def test_no_json_at_all_returns_none():
    assert parse_json_object("I am afraid I cannot do that.") is None


def test_a_bare_array_is_not_an_object():
    assert parse_json_object("[1, 2, 3]") is None


def test_an_unterminated_object_returns_none():
    assert parse_json_object('{"a": 1') is None
