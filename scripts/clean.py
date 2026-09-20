"""Remove build and cache directories. Deliberately explicit about what it deletes."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# .cache holds mkdocs-jupyter's executed-notebook cache. A stale entry there silently ships
# an older version of a chapter to the site - figures missing, numbers from a previous run -
# while the build reports success in under a second.
TARGETS = ("site", ".pytest_cache", ".ipynb_checkpoints", ".cache")


def main() -> int:
    removed = []
    for name in TARGETS:
        path = ROOT / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(ROOT)))
    for cache in ROOT.rglob("__pycache__"):
        if ".venv" not in cache.parts and cache.is_dir():
            shutil.rmtree(cache)
            removed.append(str(cache.relative_to(ROOT)))
    print("removed:", ", ".join(removed) if removed else "nothing to clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
