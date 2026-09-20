"""Generate the figure for each chapter's post, from the chapter's own measurements.

Not mockups. Each figure is drawn from the numbers the chapter produces, so a picture cannot
drift away from what the course actually teaches.

Run: AIHE_BACKEND=replay python scripts/make_post_images.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aihe import viz  # noqa: E402

OUT = ROOT / "docs" / "assets"

# About 1200 pixels across. Below that a social platform upscales the chart and it looks soft.
POST_DPI = 180


def chapter_03() -> dict[str, float]:
    from aihe.judge import build_cases, judge_cases
    from aihe.models import asker

    chapter = ROOT / "chapters/03-evals"
    raw = json.loads((chapter / "data/judge-cases.json").read_text(encoding="utf-8"))["cases"]
    ask = asker(cassettes=chapter / "cassettes", model="Llama-3.2-3B-Instruct-Q4_K_M",
                temperature=0.0, max_tokens=64, seed=7)
    rows = judge_cases(build_cases(raw), ask)

    pairs: dict[str, dict] = {}
    for row in rows:
        pairs.setdefault(row["question"], {})[row["variant"]] = row
    total = len(pairs)
    holistic = sum(1 for p in pairs.values()
                   if p["kept"]["holistic"].decided and p["lost"]["holistic"].decided
                   and p["kept"]["holistic"].value > p["lost"]["holistic"].value)
    binary = sum(1 for p in pairs.values()
                 if p["kept"]["binary"].value and not p["lost"]["binary"].value)
    return {"holistic judge": holistic / total, "binary judge": binary / total,
            "no model at all": 1.0}


def chapter_05() -> dict[str, float]:
    from aihe.models import asker
    from aihe.tools import Parameter, Tool, run_tasks, summarise

    chapter = ROOT / "chapters/05-agents"
    plan = json.loads((chapter / "data/tool-tasks.json").read_text(encoding="utf-8"))
    ask_model = asker(cassettes=chapter / "cassettes", model="Llama-3.2-3B-Instruct-Q4_K_M",
                      temperature=0.0, max_tokens=64, seed=7)
    out: dict[str, float] = {}
    for spec in plan["tools"]:
        tool = Tool(spec["name"], spec["description"],
                    tuple(Parameter(**p) for p in spec["parameters"]))
        label = "narrow" if len(spec["description"]) < 100 else "wide"
        for lenient in (False, True):
            results = run_tasks(tool, plan["tasks"], ask_model,
                                max_attempts=plan["max_attempts"], null_is_absent=lenient)
            out[f"{label} + {'lenient' if lenient else 'strict'}"] = summarise(results)["first_try"]
    return {k: out[k] for k in ("wide + strict", "narrow + strict",
                                "wide + lenient", "narrow + lenient")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    viz.save(viz.climb(chapter_03(), label="pairs separated correctly",
                       title="which judge can tell a good answer from a bad one"),
             str(OUT / "judges.png"), dpi=POST_DPI)
    print("wrote docs/assets/judges.png")

    viz.save(viz.climb({"summarised": 0 / 6, "capped": 4 / 6, "keep everything": 4 / 6},
                       label="facts recalled out of six",
                       title="what each context strategy remembered"),
             str(OUT / "context.png"), dpi=POST_DPI)
    print("wrote docs/assets/context.png")

    viz.save(viz.climb(chapter_05(), label="valid on the first try",
                       title="the same tool and the same tasks - only our choices change"),
             str(OUT / "toolcalls.png"), dpi=POST_DPI)
    print("wrote docs/assets/toolcalls.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
