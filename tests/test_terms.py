"""Terminology may not drift.

Every English technical term used inside Hebrew prose must be declared in TERMS.md. Without
this check the same concept quietly acquires three names across three chapters, which is the
fastest way for Hebrew technical writing to look amateur.

Only bare words are checked. Anything that looks like code - brackets, dots, underscores,
digits, slashes - is an identifier rather than a term and is left alone.
"""

from __future__ import annotations

import re

from tests.conftest import ROOT, markdown_cells, notebook_of

TERMS_FILE = ROOT / "TERMS.md"
INLINE_CODE = re.compile(r"`([^`\n]+)`")
BARE_WORD = re.compile(r"^[A-Za-z][A-Za-z \-]*$")


def _declared() -> set[str]:
    text = TERMS_FILE.read_text(encoding="utf-8").lower()
    return {term.strip() for term in INLINE_CODE.findall(text)} | set(
        re.findall(r"^\| ([a-z][a-z0-9 /()\-]*?) +\|", text, flags=re.MULTILINE)
    )


def _terms_used(text: str) -> set[str]:
    used = set()
    for span in INLINE_CODE.findall(text):
        span = span.strip()
        if BARE_WORD.match(span):
            used.add(span.lower())
    return used


def _is_declared(term: str, declared: set[str]) -> bool:
    if term in declared:
        return True
    if term.endswith("s") and term[:-1] in declared:  # plurals
        return True
    return all(part in declared for part in term.split() if part)


def test_terms_file_is_not_empty():
    assert len(_declared()) > 40, "TERMS.md should define the whole vocabulary"


def test_notebook_terms_are_declared(chapter):
    declared = _declared()
    used = set()
    for cell in markdown_cells(notebook_of(chapter)):
        used |= _terms_used(cell)
    undeclared = sorted(t for t in used if not _is_declared(t, declared))
    assert not undeclared, (
        f"{chapter.name} uses terms that TERMS.md does not define: {undeclared}. "
        "Add them there first - the rejected column is the valuable one."
    )


def test_chapter_prose_terms_are_declared(chapter):
    declared = _declared()
    undeclared = set()
    for name in ("README.md", "interview.md"):
        undeclared |= {
            t for t in _terms_used((chapter / name).read_text(encoding="utf-8"))
            if not _is_declared(t, declared)
        }
    assert not undeclared, f"{chapter.name}: undeclared terms {sorted(undeclared)}"


# --- punctuation that reads as machine-written ------------------------------------------

# Referenced by code point rather than written out, so this file does not contain the very
# characters it forbids.
MACHINE_MARKS = {
    chr(0x2014): "em dash: use a comma, a colon, or two sentences",
    chr(0x2013): "en dash: use a hyphen in ranges and a comma in prose",
    chr(0x2026): "ellipsis character: write three dots",
    chr(0x2018): "curly quote: use a straight one",
    chr(0x2019): "curly quote: use a straight one",
    chr(0x201C): "curly quote: use a straight one",
    chr(0x201D): "curly quote: use a straight one",
}

AUTHORED = (".md", ".py", ".ipynb", ".yml", ".yaml", ".toml")


def _authored_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in AUTHORED:
            continue
        if set(path.parts) & {".venv", ".git", "site", ".cache"} or "cassettes" in path.parts:
            continue
        # data/ holds a third-party corpus. It is quoted, not composed, so it is quoted exactly.
        if path.suffix == ".json" and "data" in path.parts:
            continue
        yield path


def test_no_machine_punctuation_in_authored_files():
    """These marks are what made the docs read as machine-written. The corpus and the
    cassettes are exempt: they are evidence, and evidence is reproduced verbatim."""
    offences = []
    for path in _authored_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for mark, why in MACHINE_MARKS.items():
            if mark in text:
                offences.append(f"{path.relative_to(ROOT)}: U+{ord(mark):04X} {why}")
    assert not offences, "\n  " + "\n  ".join(sorted(set(offences))[:20])


def test_no_decorative_emoji_in_authored_files():
    import re

    emoji = re.compile(
        f"[{chr(0x1F300)}-{chr(0x1FAFF)}{chr(0x2728)}{chr(0x2705)}{chr(0x274C)}{chr(0x2B50)}]"
    )
    offences = [
        str(path.relative_to(ROOT))
        for path in _authored_files()
        if emoji.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not offences, f"decorative emoji in: {offences}"
