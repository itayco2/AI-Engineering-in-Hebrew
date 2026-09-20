"""Chapter 02 asserts its own claims, without downloading a model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aihe.hebrew import compare_to_reference, load_reference, strip_prefixes, tokenize_hebrew
from aihe.pipeline import load_corpus

HERE = Path(__file__).resolve().parents[1]
CORPUS = HERE / "data" / "heq-subset.json"
REFERENCE = HERE / "data" / "dictabert-seg.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS)


def test_the_corpus_is_the_size_the_chapter_claims(corpus):
    assert len(corpus.documents) == 150
    assert len(corpus.queries) == 1036


def test_every_question_points_at_a_passage_that_exists(corpus):
    ids = {d["id"] for d in corpus.documents}
    assert not {r for q in corpus.queries for r in q["relevant"]} - ids


def test_the_passages_are_actually_hebrew(corpus):
    hebrew = re.compile(r"[א-ת]")
    assert all(hebrew.search(d["text"]) for d in corpus.documents)


def test_both_sources_are_represented(corpus):
    """Wikipedia and Geektime write differently; a chapter on Hebrew retrieval should not be
    measured on one register alone."""
    assert {d["source"] for d in corpus.documents} == {"Wikipedia", "Geektime"}


def test_the_recorded_segmentation_names_its_model():
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert data["model"] == "dicta-il/dictabert-seg"
    assert len(data["words"]) > 400


def test_the_rule_agreement_rate_is_what_the_chapter_says():
    """About three quarters. Pinned so that 'rules are cheap and imperfect' stays measured
    rather than remembered."""
    result = compare_to_reference(load_reference(REFERENCE))
    assert 0.70 <= result["rate"] <= 0.80


def test_the_disagreements_are_kept_because_they_are_the_content():
    result = compare_to_reference(load_reference(REFERENCE))
    assert len(result["disagreements"]) > 50
    word, expected, got = result["disagreements"][0]
    assert expected != got


def test_tokenizing_hebrew_adds_stems_without_losing_words():
    """The design that lets a 75%-accurate rule still help: a wrong stem adds noise, it never
    removes the correct match."""
    tokens = tokenize_hebrew("ובמסמכים שבמאגר")
    assert {"ובמסמכים", "מסמכים", "שבמאגר", "מאגר"} <= set(tokens)


def test_splitting_never_shrinks_the_token_list():
    for text in ("ובמסמכים שבמאגר יש מידע", "כשהטוקן פג תוקף"):
        assert len(tokenize_hebrew(text, split=True)) >= len(tokenize_hebrew(text, split=False))


def test_a_known_ambiguous_word_is_still_wrong():
    """If this ever passes, the chapter's measured disagreement rate is stale."""
    assert strip_prefixes("מידע") == ["מ", "ידע"]
