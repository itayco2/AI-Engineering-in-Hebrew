"""Chapter 04 asserts its own claims. The cache half needs no model at all."""

from __future__ import annotations

import json
from pathlib import Path

from aihe.cassettes import CassetteLibrary
from aihe.context import (
    cache_hit_ratio, capped_history, planted_fact_probe, prefix_is_stable, render, summarise_old,
)

HERE = Path(__file__).resolve().parents[1]
PLAN = json.loads((HERE / "data" / "probes.json").read_text(encoding="utf-8"))


def _turns(probe):
    return planted_fact_probe(probe["fact"], probe["question"], probe["filler"])


def test_there_are_six_probes():
    assert len(PLAN["probes"]) == 6


def test_every_fact_is_unguessable():
    """A fact the model could infer would measure its general knowledge, not the context."""
    for probe in PLAN["probes"]:
        assert any(ch.isdigit() or ch in "#-" for ch in probe["fact"])


def test_every_filler_turn_is_longer_than_the_cap():
    """Otherwise the cap would be decorative and the saving would come from nowhere."""
    for probe in PLAN["probes"]:
        assert all(len(f) > PLAN["cap"] for f in probe["filler"])


def test_summarising_removes_the_fact_from_the_context():
    """Not an opinion about the model: the text is simply no longer there."""
    for probe in PLAN["probes"]:
        summarised = render(summarise_old(_turns(probe), keep_recent=4, summary=PLAN["summary"]))
        assert probe["fact"] not in summarised


def test_capping_keeps_the_fact_in_the_context():
    for probe in PLAN["probes"]:
        assert probe["fact"] in render(capped_history(_turns(probe), limit=PLAN["cap"]))


def test_capping_saves_characters():
    for probe in PLAN["probes"]:
        turns = _turns(probe)
        assert len(render(capped_history(turns, limit=PLAN["cap"]))) < len(render(turns))


def test_appending_keeps_the_cache_and_summarising_destroys_it():
    """The chapter's mechanism, decided by comparing bytes."""
    turns = _turns(PLAN["probes"][0])
    before = render(turns[:-1])
    assert prefix_is_stable(before, render(turns))
    assert cache_hit_ratio(before, render(turns)) == 1.0

    summarised = render(summarise_old(turns, keep_recent=4, summary=PLAN["summary"]))
    assert not prefix_is_stable(before, summarised)
    assert cache_hit_ratio(before, summarised) < 0.2


def test_the_cassettes_are_present_and_pinned():
    library = CassetteLibrary(HERE / "cassettes")
    keys = library.keys()
    assert len(keys) == 18
    assert library.load(keys[0])["model"] == "Llama-3.2-3B-Instruct-Q4_K_M"
