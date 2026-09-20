"""Fail if the built site contains a traceback, or a chapter that produced no output.

This exists because `mkdocs build --strict` does not fail when an executed notebook raises.
mkdocs-jupyter catches the error, embeds the traceback in the page, and the build reports
success. Three chapters shipped `ConnectionRefusedError` to the site that way, and every other
check in the repository was green at the time: the tests passed, nbmake passed, and the build
said it was fine.

Run: python scripts/check_site.py    (or: make docs, which runs it)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"

TRACEBACK = re.compile(
    r"Traceback \(most recent call last\)|"
    r"\b(NameError|ConnectionRefusedError|ConnectionError|CassetteMiss|ModuleNotFoundError|"
    r"FileNotFoundError|ValueError|KeyError|RuntimeError|URLError)\b"
)


def main() -> int:
    if not SITE.is_dir():
        print("no site/ directory - run mkdocs build first")
        return 1

    problems: list[str] = []
    chapters = sorted(SITE.glob("chapters/*/chapter/index.html"))
    if not chapters:
        print("no chapter pages found in site/ - did the nav change?")
        return 1

    for page in chapters:
        name = page.parent.parent.name
        html = page.read_text(encoding="utf-8", errors="replace")

        hits = sorted(set(TRACEBACK.findall(html)))
        if hits or "Traceback (most recent call last)" in html:
            problems.append(f"{name}: published page contains an error - {hits or ['Traceback']}")

        figures = html.count("data:image/png;base64")
        outputs = html.count("jp-OutputArea")
        if outputs == 0:
            problems.append(f"{name}: published page has no cell output at all")
        print(f"  {name:16s} {outputs:3d} outputs, {figures} figures")

    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for problem in problems:
            print("  " + problem)
        print("\nThe chapters are executed at build time. Set AIHE_BACKEND=replay so they read")
        print("their cassettes instead of reaching for a model that is not running.")
        return 1

    print("\nsite: every chapter rendered, no tracebacks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
