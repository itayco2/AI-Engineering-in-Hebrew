"""Every chapter has the same shape. Five chapters inventing five shapes is the failure
mode this file exists to prevent."""

from __future__ import annotations

import yaml

from tests.conftest import code_cells, markdown_cells, notebook_of

REQUIRED_FILES = ("README.md", "chapter.ipynb", "meta.yml", "interview.md")
REQUIRED_DIRS = ("data", "tests")
REQUIRED_META = (
    "id", "title_he", "title_en", "needs_llm", "ram_gb",
    "est_minutes", "datasets", "models", "terms",
)


def test_required_files_exist(chapter):
    missing = [f for f in REQUIRED_FILES if not (chapter / f).is_file()]
    assert not missing, f"{chapter.name} is missing {missing}"


def test_required_directories_exist(chapter):
    missing = [d for d in REQUIRED_DIRS if not (chapter / d).is_dir()]
    assert not missing, f"{chapter.name} is missing {missing}"


def test_data_declares_its_licence(chapter):
    assert (chapter / "data" / "SOURCE.md").is_file(), (
        f"{chapter.name}/data needs SOURCE.md naming where the data came from and its licence"
    )


def test_meta_parses_and_is_complete(chapter):
    meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_META if k not in meta]
    assert not missing, f"{chapter.name}/meta.yml is missing {missing}"


def test_meta_id_matches_the_directory(chapter):
    meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
    assert meta["id"] == chapter.name


def test_meta_declares_a_ram_floor_when_it_needs_a_model(chapter):
    """A chapter that runs a model must say how much memory that needs, or the one-command
    promise is a lie for anyone on a small laptop."""
    meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
    if meta["needs_llm"]:
        assert isinstance(meta["ram_gb"], (int, float)) and meta["ram_gb"] >= 4


def test_every_declared_dataset_exists(chapter):
    meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
    for dataset in meta.get("datasets") or []:
        assert (chapter / dataset).exists(), f"{chapter.name}: {dataset} declared but missing"


def test_a_chapter_needing_a_model_ships_cassettes(chapter):
    """Otherwise CI cannot run it without a network call, and the build is not free."""
    meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
    if meta["needs_llm"]:
        cassettes = chapter / "cassettes"
        assert cassettes.is_dir() and any(cassettes.glob("*.json")), (
            f"{chapter.name} needs a model but ships no cassettes"
        )


def test_the_notebook_opens_with_imports(chapter):
    cells = code_cells(notebook_of(chapter))
    assert cells, f"{chapter.name} has no code cells"
    first = cells[0]
    assert "import" in first, f"{chapter.name}: first code cell should be the imports"


def test_the_notebook_ends_by_printing_its_headline_number(chapter):
    cells = code_cells(notebook_of(chapter))
    assert "print" in cells[-1], (
        f"{chapter.name}: the last cell should print the chapter's headline number"
    )


def test_every_cell_has_an_id(chapter):
    """Without ids, nbformat warns and diffs churn."""
    notebook = notebook_of(chapter)
    assert all(c.get("id") for c in notebook["cells"]), f"{chapter.name}: a cell has no id"


def test_the_notebook_has_prose_not_only_code(chapter):
    notebook = notebook_of(chapter)
    assert len(markdown_cells(notebook)) >= len(code_cells(notebook)) / 2, (
        f"{chapter.name}: this is a course, not a script"
    )


def test_readme_links_to_the_interview_questions(chapter):
    assert "interview.md" in (chapter / "README.md").read_text(encoding="utf-8")
