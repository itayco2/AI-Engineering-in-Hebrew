"""Shared fixtures: the chapters on disk, and their notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = sorted(p for p in (ROOT / "chapters").iterdir() if p.is_dir())


def chapter_ids() -> list[str]:
    return [p.name for p in CHAPTERS]


@pytest.fixture(params=CHAPTERS, ids=chapter_ids())
def chapter(request) -> Path:
    return request.param


def notebook_of(chapter: Path) -> dict:
    return json.loads((chapter / "chapter.ipynb").read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def markdown_cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"]
