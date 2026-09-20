"""Chunking: offsets must be exact, sizes must be respected, nothing may be lost."""

import pytest

from aihe.chunking import STRATEGIES, Chunk, by_structure, fixed, recursive

PROSE = (
    "RAG has two phases. The first one indexes documents.\n\n"
    "The second one answers questions. It retrieves, then it generates.\n\n"
    "Retrieval is the part that usually breaks. Measure it separately."
)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_offsets_point_back_at_the_source(name):
    """doc[chunk.start:chunk.end] is exactly chunk.text — a chunk you cannot locate
    in the original document is a chunk you cannot debug."""
    for chunk in STRATEGIES[name](PROSE, size=120):
        assert PROSE[chunk.start : chunk.end] == chunk.text


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_indices_are_sequential_from_zero(name):
    chunks = STRATEGIES[name](PROSE, size=120)
    assert [c.index for c in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_empty_text_gives_no_chunks(name):
    assert STRATEGIES[name]("", size=60) == []


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_no_chunk_is_empty(name):
    assert all(len(c.text) > 0 for c in STRATEGIES[name](PROSE, size=120))


def test_fixed_covers_the_whole_text():
    chunks = fixed(PROSE, size=50, overlap=10)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(PROSE)
    # consecutive chunks must not leave a gap
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.start <= earlier.end


def test_fixed_overlap_is_exactly_the_requested_step():
    chunks = fixed("x" * 100, size=30, overlap=10)
    starts = [c.start for c in chunks]
    assert starts == [0, 20, 40, 60, 80]


def test_fixed_respects_size():
    assert all(len(c) <= 30 for c in fixed("x" * 100, size=30, overlap=10))


def test_fixed_terminates_with_large_overlap():
    """A step of 1 must still terminate rather than loop forever."""
    chunks = fixed("x" * 20, size=10, overlap=9)
    assert len(chunks) == 11
    assert chunks[-1].end == 20


@pytest.mark.parametrize(
    "size,overlap",
    [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_size_and_overlap_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        fixed(PROSE, size=size, overlap=overlap)


def test_recursive_prefers_paragraph_boundaries():
    """With a size that fits a paragraph, chunks should not straddle the blank lines."""
    chunks = recursive(PROSE, size=70, overlap=0)
    assert all("\n\n" not in c.text for c in chunks)


def test_recursive_falls_back_to_characters_when_nothing_else_fits():
    """A single long word has no separator to split on, so it splits hard."""
    chunks = recursive("y" * 100, size=25, overlap=0)
    assert len(chunks) == 4
    assert all(len(c) <= 25 for c in chunks)


def test_by_structure_strips_paragraph_whitespace():
    """With no room to merge, each paragraph is its own chunk and its padding is gone."""
    chunks = by_structure("alpha\n\n   beta   \n\ngamma", size=6)
    assert [c.text for c in chunks] == ["alpha", "beta", "gamma"]


def test_by_structure_merges_short_neighbours():
    """Given room, short paragraphs join rather than becoming three tiny chunks. The
    merged chunk spans the separators, which keeps its offsets exact."""
    source = "alpha\n\n   beta   \n\ngamma"
    chunks = by_structure(source, size=200)
    assert len(chunks) == 1
    assert source[chunks[0].start : chunks[0].end] == chunks[0].text


def test_by_structure_splits_a_paragraph_that_is_too_long():
    long_para = "word " * 60
    chunks = by_structure(long_para, size=50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunk_len_is_its_text_length():
    chunk = Chunk(text="four", start=0, end=4, index=0)
    assert len(chunk) == 4


def test_recursive_overlap_is_additive_not_carved_out():
    """Pinned because it differs from `fixed` and the difference matters when sizing chunks
    against a model's context window: here a chunk can reach about size + overlap."""
    text = ("Alpha sentence one. Alpha sentence two. Alpha sentence three. "
            "Beta sentence one. Beta sentence two. Beta sentence three.")
    chunks = recursive(text, size=60, overlap=30)
    assert max(len(c) for c in chunks) > 60
    assert max(len(c) for c in chunks) <= 60 + 30
    # and the overlap is real: consecutive chunks share text
    assert chunks[1].start < chunks[0].end


def test_fixed_overlap_is_carved_out_of_size():
    """The contrast that makes the note on `recursive` worth reading."""
    assert all(len(c) <= 60 for c in fixed("z" * 300, size=60, overlap=30))
