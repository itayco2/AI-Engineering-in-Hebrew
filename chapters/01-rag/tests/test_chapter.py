"""Chapter 01 asserts its own claims, on small inputs, without a model.

These are not the package's unit tests. They check that the things the chapter *says* are
still true of the data and the pipeline it ships with, so a chapter cannot quietly start
teaching something false.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from aihe.pipeline import build_chunks, contextualise, load_corpus

CORPUS = Path(__file__).resolve().parents[1] / "data" / "meridian.json"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(CORPUS)


def test_the_corpus_is_the_size_the_chapter_claims(corpus):
    assert len(corpus.documents) == 46
    assert len(corpus.queries) == 28


def test_every_question_points_at_a_document_that_exists(corpus):
    ids = {d["id"] for d in corpus.documents}
    dangling = {r for q in corpus.queries for r in q["relevant"]} - ids
    assert not dangling, f"questions reference missing documents: {dangling}"


def test_document_ids_are_unique(corpus):
    duplicates = [i for i, n in Counter(d["id"] for d in corpus.documents).items() if n > 1]
    assert not duplicates


def test_all_three_question_kinds_are_present(corpus):
    """The chapter's central argument is that different mechanisms fix different failures,
    which it can only show if all three kinds are in the evaluation set."""
    kinds = Counter(q["kind"] for q in corpus.queries)
    assert set(kinds) == {"identifier", "paraphrase", "mixed"}
    assert min(kinds.values()) >= 5


def test_identifier_questions_really_are_exact_strings(corpus):
    """An identifier question must contain a literal token from its answer, or it is not
    testing what the chapter says it tests."""
    text = {d["id"]: d["text"] for d in corpus.documents}
    for query in corpus.queries:
        if query["kind"] != "identifier":
            continue
        assert any(query["question"] in text[doc] for doc in query["relevant"]), query["id"]


def test_chunking_the_corpus_gives_the_chapter_s_chunk_count(corpus):
    assert len(build_chunks(corpus, size=200, overlap=40)) == 105


def test_every_chunk_maps_back_to_its_document(corpus):
    ids = {d["id"] for d in corpus.documents}
    assert all(c.doc_id in ids for c in build_chunks(corpus))


def test_contextualising_puts_the_title_in_front(corpus):
    """The one-line fix the chapter is built around."""
    chunks = build_chunks(corpus)
    titles = corpus.titles
    for chunk, enriched in zip(chunks, contextualise(chunks, corpus)):
        assert enriched.startswith(titles[chunk.doc_id])
        assert chunk.text in enriched


def test_contextualising_actually_adds_words(corpus):
    chunks = build_chunks(corpus)
    rich = contextualise(chunks, corpus)
    assert all(len(r) > len(c.text) for r, c in zip(rich, chunks))
