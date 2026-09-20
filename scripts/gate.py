"""THE GATE. Under a minute, and it runs before anything expensive.

Borrowed from a habit that saved a great deal of time elsewhere: a cheap check that refuses
to let you start the slow thing on a broken setup. It is also the best troubleshooting tool
in the repo, because it is the first thing a stranger runs when nothing works, and it names
the problem instead of failing somewhere strange twenty minutes later.

Run: python scripts/gate.py    (or: make gate)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "  ok  ", " FAIL ", " skip "


def _report(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    started = time.time()
    failures = 0
    print("gate: checking the things that make a chapter runnable\n")

    # 1. the package
    try:
        import aihe
        from aihe.retrieval import rrf

        assert round(dict(rrf([[0], [1, 2, 0]]))[0], 4) == round(1 / 61 + 1 / 63, 4)
        _report(PASS, "package imports", f"aihe {aihe.__version__}")
    except Exception as exc:
        _report(FAIL, "package imports", f"{type(exc).__name__}: {exc}")
        print("\n  Run: make setup")
        return 1

    # 2. chapters declare themselves correctly
    try:
        import yaml
    except ImportError:
        _report(FAIL, "PyYAML missing", "pip install -r requirements-test.txt")
        return 1

    chapters = sorted(p for p in (ROOT / "chapters").iterdir() if p.is_dir())
    for chapter in chapters:
        meta_file = chapter / "meta.yml"
        try:
            meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        except Exception as exc:
            _report(FAIL, f"{chapter.name} meta.yml", str(exc))
            failures += 1
            continue

        missing = [d for d in (meta.get("datasets") or []) if not (chapter / d).exists()]
        if missing:
            _report(FAIL, f"{chapter.name} datasets", f"missing {missing}")
            failures += 1
        elif meta.get("needs_llm") and not any((chapter / "cassettes").glob("*.json")):
            _report(FAIL, f"{chapter.name} cassettes", "needs a model but has no recordings")
            failures += 1
        else:
            needs = "model" if meta.get("needs_llm") else "no model"
            _report(PASS, f"{chapter.name}", f"{meta['title_en']} ({needs})")

    # 3. the embedding model - the one download every chapter depends on
    try:
        from aihe.embeddings import DEFAULT_MODEL, encode

        vector = encode(["שלום עולם"])
        assert vector.shape[0] == 1 and vector.shape[1] > 0
        _report(PASS, "embedding model", f"{DEFAULT_MODEL.split('/')[-1]} -> {vector.shape[1]}d")
    except ImportError as exc:
        _report(SKIP, "embedding model", f"not installed ({exc.__class__.__name__})")
    except Exception as exc:
        _report(FAIL, "embedding model", f"{type(exc).__name__}: {exc}")
        failures += 1

    # 4. the backend, only if some chapter actually needs one
    wants_model = any(
        yaml.safe_load((c / "meta.yml").read_text(encoding="utf-8")).get("needs_llm")
        for c in chapters
    )
    requested = os.environ.get("AIHE_BACKEND", "").strip().lower()
    if not wants_model:
        _report(SKIP, "chat backend", "no chapter needs generation yet")
    elif requested in ("", "replay"):
        # A live model is optional here, and that is the point: every chapter that needs
        # generation ships its replies. Failing because a model nobody asked for is not running
        # would make this gate lie about a repository that works perfectly without one.
        _report(SKIP, "chat backend", "not requested - chapters replay from cassettes")
    else:
        try:
            from aihe.models import backend_name, chat

            name = backend_name()
            chat([{"role": "user", "content": "say ok"}], max_tokens=8)
            _report(PASS, "chat backend", name)
        except Exception as exc:
            # Asked for explicitly and unreachable. That is a real failure.
            _report(FAIL, f"chat backend ({requested})", str(exc).splitlines()[0])
            failures += 1

    elapsed = time.time() - started
    print(f"\n{'gate passed' if not failures else f'{failures} check(s) failed'} in {elapsed:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
