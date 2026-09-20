"""Re-record every chapter's cassettes from a real model, then review the diff.

Recording is always a deliberate, human-run action. CI never records: `AIHE_BACKEND=replay` is
the default there, and a miss fails loudly rather than quietly reaching for the network.

Each chapter that needs generation has a recorder below. A recorder runs **the same code path
the notebook runs** - the same helpers in `aihe/` building the same prompts - which is what
makes a recording match the request the notebook will later replay.

    AIHE_BACKEND=llamacpp AIHE_GGUF=/path/to/model.gguf python scripts/record_cassettes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = "Llama-3.2-3B-Instruct-Q4_K_M"
PARAMS = {"temperature": 0.0, "max_tokens": 64, "seed": 7}


def record_03(chapter: Path) -> int:
    """Chapter 03: both judges over every case."""
    from aihe.judge import build_cases, judge_cases
    from aihe.models import asker

    raw = json.loads((chapter / "data/judge-cases.json").read_text(encoding="utf-8"))["cases"]
    cases = build_cases(raw)
    ask = asker(record_to=chapter / "cassettes", model=MODEL, **PARAMS)
    judge_cases(cases, ask)
    return len(cases) * 2


def record_04(chapter: Path) -> int:
    """Chapter 04: the planted-fact probe under each context strategy."""
    from aihe.context import capped_history, planted_fact_probe, render, summarise_old
    from aihe.models import asker

    plan = json.loads((chapter / "data/probes.json").read_text(encoding="utf-8"))
    ask = asker(record_to=chapter / "cassettes", model=MODEL, **PARAMS)
    calls = 0
    for probe in plan["probes"]:
        turns = planted_fact_probe(probe["fact"], probe["question"], probe["filler"])
        for _name, history in (
            ("keep_all", turns),
            ("summarised", summarise_old(turns, keep_recent=4, summary=plan["summary"])),
            ("capped", capped_history(turns, limit=plan["cap"])),
        ):
            ask(render(history))
            calls += 1
    return calls


def record_05(chapter: Path) -> int:
    """Chapter 05: the repair loop against a narrow schema and a wide one."""
    from aihe.models import asker
    from aihe.tools import Parameter, Tool, run_tasks

    plan = json.loads((chapter / "data/tool-tasks.json").read_text(encoding="utf-8"))
    ask_model = asker(record_to=chapter / "cassettes", model=MODEL, **PARAMS)
    calls = 0
    for spec in plan["tools"]:
        tool = Tool(spec["name"], spec["description"],
                    tuple(Parameter(**p) for p in spec["parameters"]))
        # Both validators, because the two produce different retry feedback and therefore
        # genuinely different requests. Recording only one leaves the other with a miss.
        for lenient in (False, True):
            results = run_tasks(tool, plan["tasks"], ask_model,
                                max_attempts=plan["max_attempts"], null_is_absent=lenient)
            calls += sum(r.tries for r in results)
    return calls


RECORDERS = {"03-evals": record_03, "04-context": record_04, "05-agents": record_05}


def main() -> int:
    import yaml

    from aihe.models import backend_name

    name = backend_name()
    if name in ("replay", "fake"):
        print(f"AIHE_BACKEND is {name!r}, which makes no real call. Set llamacpp or ollama.")
        return 1

    chapters = sorted(p for p in (ROOT / "chapters").iterdir() if p.is_dir())
    total = 0
    for chapter in chapters:
        meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
        if not meta.get("needs_llm"):
            continue
        recorder = RECORDERS.get(chapter.name)
        if recorder is None:
            print(f"{chapter.name}: needs a model but has no recorder in this script")
            continue
        print(f"{chapter.name}: recording with {name} ...")
        calls = recorder(chapter)
        written = len(list((chapter / "cassettes").glob("*.json")))
        print(f"  {calls} calls -> {written} cassettes")
        total += calls

    if not total:
        print("No chapter needs generation yet, so there is nothing to record.")
    else:
        print("\nNow review the diff. A changed answer is a real change to what a chapter teaches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
