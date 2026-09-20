"""No logic in notebooks. The rule that makes everything else in this repo possible.

A `def` inside a notebook cell cannot be unit-tested, cannot be reviewed as a diff, and
breaks silently when a dependency changes. Logic belongs in `aihe/`; a cell imports, calls,
and shows.
"""

from __future__ import annotations

import ast

from tests.conftest import code_cells, notebook_of

BANNED = {
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "async def",
    ast.ClassDef: "class",
}


def test_no_function_or_class_definitions(chapter):
    offences = []
    for number, source in enumerate(code_cells(notebook_of(chapter)), start=1):
        try:
            tree = ast.parse(source)
        except SyntaxError:  # a cell using notebook-only syntax; nbmake will catch it
            continue
        for node in ast.walk(tree):
            for kind, label in BANNED.items():
                if isinstance(node, kind):
                    offences.append(f"cell {number}: {label} {node.name}")
    assert not offences, (
        f"{chapter.name} defines logic in its notebook: {offences}. "
        "Move it into aihe/ and import it - see CONTRIBUTING.md."
    )


def test_every_cell_parses_as_python(chapter):
    for number, source in enumerate(code_cells(notebook_of(chapter)), start=1):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"{chapter.name} cell {number} is not valid Python: {exc}") from exc


def test_cells_stay_short_enough_to_read(chapter):
    """Long cells are where RTL rendering falls apart, and where logic starts creeping in."""
    for number, source in enumerate(code_cells(notebook_of(chapter)), start=1):
        lines = [line for line in source.splitlines() if line.strip()]
        assert len(lines) <= 25, f"{chapter.name} cell {number} has {len(lines)} lines"
