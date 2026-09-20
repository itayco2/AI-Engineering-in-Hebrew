"""Cutting documents into pieces small enough to retrieve.

Three strategies, deliberately: the naive one, the one that respects separators, and the
one that respects the document's own structure. Chapter 01 runs all three over the same
corpus so the reader sees what the choice costs in recall rather than being told.

Every chunk carries its character offsets back into the source, because a chunk you cannot
locate in the original document is a chunk you cannot debug.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document.

    `start` and `end` are character offsets into the document the chunk came from, so
    `doc_text[chunk.start:chunk.end]` is always exactly `chunk.text`.
    """

    text: str
    start: int
    end: int
    index: int
    doc_id: str = ""

    def __len__(self) -> int:
        return len(self.text)


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")


def _finish(spans: list[tuple[int, int]], text: str, doc_id: str) -> list[Chunk]:
    return [
        Chunk(text=text[s:e], start=s, end=e, index=i, doc_id=doc_id)
        for i, (s, e) in enumerate(spans)
        if e > s
    ]


def fixed(text: str, size: int = 400, overlap: int = 50, doc_id: str = "") -> list[Chunk]:
    """Cut every `size` characters, stepping back `overlap` each time.

    The strategy that ignores the text completely. It will cut mid-word and mid-sentence,
    which is exactly why it is here — chapter 01 measures what that costs.
    """
    _validate(size, overlap)
    if not text:
        return []

    step = size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start += step
    return _finish(spans, text, doc_id)


def _split_spans(
    text: str, start: int, end: int, size: int, separators: list[str]
) -> list[tuple[int, int]]:
    """Split [start, end) down to pieces of at most `size`, trying separators in order."""
    if end - start <= size:
        return [(start, end)]
    if not separators:
        return [(i, min(i + size, end)) for i in range(start, end, size)]

    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [(i, min(i + size, end)) for i in range(start, end, size)]
    if sep not in text[start:end]:
        return _split_spans(text, start, end, size, rest)

    spans: list[tuple[int, int]] = []
    pos = start
    for part in text[start:end].split(sep):
        piece_end = pos + len(part)
        if piece_end > pos:
            if piece_end - pos > size:
                spans.extend(_split_spans(text, pos, piece_end, size, rest))
            else:
                spans.append((pos, piece_end))
        pos = piece_end + len(sep)
    return spans


def _merge_spans(
    spans: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int]]:
    """Greedily join adjacent pieces up to `size`, then back each one up by `overlap`."""
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[1] - merged[-1][0] <= size:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)

    if overlap == 0:
        return merged
    return [
        (start if i == 0 else max(merged[i - 1][0], start - overlap), end)
        for i, (start, end) in enumerate(merged)
    ]


def recursive(
    text: str,
    size: int = 400,
    overlap: int = 50,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    doc_id: str = "",
) -> list[Chunk]:
    """Split on the largest separator that fits, falling back to smaller ones.

    Paragraph breaks first, then line breaks, then sentences, then words, then characters.
    The result respects sentence boundaries wherever the size budget allows it to.
    """
    _validate(size, overlap)
    if not text:
        return []
    spans = _split_spans(text, 0, len(text), size, list(separators))
    return _finish(_merge_spans(spans, size, overlap), text, doc_id)


def by_structure(text: str, size: int = 400, doc_id: str = "") -> list[Chunk]:
    """One chunk per paragraph, merging short neighbours, splitting only oversized ones.

    No overlap: the document's own structure is the boundary, and overlapping across a
    paragraph break re-introduces the problem the strategy exists to avoid.
    """
    _validate(size, 0)
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    pos = 0
    for para in text.split("\n\n"):
        end = pos + len(para)
        stripped = para.strip()
        if stripped:
            lead = len(para) - len(para.lstrip())
            trail = len(para) - len(para.rstrip())
            p_start, p_end = pos + lead, end - trail
            if p_end - p_start > size:
                spans.extend(_split_spans(text, p_start, p_end, size, list(DEFAULT_SEPARATORS)))
            else:
                spans.append((p_start, p_end))
        pos = end + 2

    return _finish(_merge_spans(spans, size, 0), text, doc_id)


STRATEGIES = {"fixed": fixed, "recursive": recursive, "by_structure": by_structure}
