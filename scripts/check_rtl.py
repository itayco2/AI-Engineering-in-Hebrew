"""Check the Hebrew writing rules that CSS cannot rescue.

GitHub's Markdown renderer decides a paragraph's direction from its first strong character.
A Hebrew paragraph that opens with an English word or a code span is rendered left to right,
and no stylesheet loaded later can undo that — the decision was made before any CSS ran. The
only fix is the writing rule, so it is checked here.

Run: python scripts/check_rtl.py    (or: make rtl)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL_OPEN = re.compile(r'<div[^>]*dir="rtl"', re.IGNORECASE)
RTL_WITH_MARKDOWN = re.compile(r'<div[^>]*dir="rtl"[^>]*markdown="1"', re.IGNORECASE)
RTL_CLOSE = re.compile(r"</div>", re.IGNORECASE)
HEBREW = re.compile(r"[֐-׿]")
OPENS_LTR = re.compile(r"^[A-Za-z`]")
# a bare figure sitting loose in prose: 12%, 0.911, 5 -> 3, but not `12%` or a table row
BARE_FIGURE = re.compile(r"(?<![`|\w])(\d+(?:\.\d+)?%|\d+\.\d{3})(?![`\w])")
SKIP_PREFIX = ("#", ">", "-", "*", "|", "<", "!", "[", "```")


def _paragraph_starts(lines: list[str]) -> list[tuple[int, str]]:
    """First line of each paragraph inside an RTL block."""
    out, inside, previous_blank = [], False, True
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if RTL_OPEN.search(line):
            inside, previous_blank = True, True
            continue
        if inside and RTL_CLOSE.search(line):
            inside = False
            continue
        if not inside:
            continue
        if not stripped:
            previous_blank = True
            continue
        if previous_blank and not stripped.startswith(SKIP_PREFIX):
            out.append((number, stripped))
        previous_blank = False
    return out


def _check_text(label: str, text: str) -> list[str]:
    problems = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        if RTL_OPEN.search(line) and not RTL_WITH_MARKDOWN.search(line):
            problems.append(
                f'{label}:{number}: <div dir="rtl"> without markdown="1" -> everything inside '
                "renders as literal text: hashes, pipes and link brackets, unformatted."
            )
    for number, paragraph in _paragraph_starts(lines):
        if HEBREW.search(paragraph) and OPENS_LTR.match(paragraph):
            problems.append(
                f"{label}:{number}: Hebrew paragraph opens with "
                f"{'a code span' if paragraph[0] == '`' else 'an English word'} -> renders "
                f"left-to-right.\n    {paragraph[:78]}"
            )
    inside = False
    for number, line in enumerate(lines, start=1):
        if RTL_OPEN.search(line):
            inside = True
            continue
        if inside and RTL_CLOSE.search(line):
            inside = False
            continue
        if not inside or line.strip().startswith(("|", "#", "<")):
            continue
        if HEBREW.search(line):
            for figure in BARE_FIGURE.findall(line):
                problems.append(
                    f"{label}:{number}: figure {figure} is bare in Hebrew prose -> put it in "
                    "backticks or a table, or the signs land in the wrong place."
                )
    return problems


def _markdown_sources() -> list[tuple[str, str]]:
    sources = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".venv", "site", ".git"} for part in path.parts):
            continue
        sources.append((str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")))
    for path in sorted(ROOT.rglob("chapter.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] == "markdown":
                label = f"{path.relative_to(ROOT)} cell {index}"
                sources.append((label, "".join(cell["source"])))
    return sources


def main() -> int:
    problems = []
    for label, text in _markdown_sources():
        problems.extend(_check_text(label, text))
    if problems:
        print(f"{len(problems)} RTL problem(s):\n")
        for problem in problems:
            print("  " + problem)
        print("\nSee CONTRIBUTING.md, 'Writing Hebrew that renders'.")
        return 1
    print("RTL rules: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
