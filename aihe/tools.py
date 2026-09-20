"""Tool schemas, validation, and the repair loop that makes small models usable.

An agent is a loop in which a model picks the next action. The action arrives as JSON, and
small models get that JSON wrong often enough that the repair loop is not a nicety - it is the
difference between a demo and something that runs.

Everything here is a pure function of its inputs apart from `repair`, which takes the model
call as an argument. That is deliberate: generated text is not reproducible, so the parts worth
testing are the parts that decide what to do with it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


@dataclass(frozen=True)
class Parameter:
    """One argument of a tool. Primitive types only, on purpose.

    Chapter 05 measures what happens when a schema grows: a 3B model that handles three
    primitive parameters reliably starts inventing structure as soon as it is given nested
    objects or a long description.
    """

    name: str
    type: str
    description: str = ""
    required: bool = True
    enum: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in JSON_TYPES:
            raise ValueError(f"{self.type!r} is not a primitive type: {sorted(JSON_TYPES)}")


@dataclass(frozen=True)
class Tool:
    """A tool the model may call."""

    name: str
    description: str
    parameters: tuple[Parameter, ...] = ()

    def schema(self) -> dict[str, Any]:
        """The OpenAI-style function schema, which Ollama and llama.cpp both accept."""
        properties: dict[str, Any] = {}
        for parameter in self.parameters:
            entry: dict[str, Any] = {"type": parameter.type}
            if parameter.description:
                entry["description"] = parameter.description
            if parameter.enum:
                entry["enum"] = list(parameter.enum)
            properties[parameter.name] = entry
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }

    def prompt_hint(self) -> str:
        """The schema as a sentence, for models that ignore the tools parameter entirely.

        Small models frequently do. Asking in the prompt as well is not redundant.
        """
        lines = [f"Call {self.name}. {self.description}", "Reply with JSON only, with keys:"]
        for parameter in self.parameters:
            allowed = f" one of {list(parameter.enum)}" if parameter.enum else ""
            optional = "" if parameter.required else " (optional)"
            lines.append(f"  {parameter.name}: {parameter.type}{allowed}{optional}")
        return "\n".join(lines)


@dataclass
class Validation:
    """What was wrong with a tool call, in words a model can act on."""

    ok: bool
    arguments: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        return "; ".join(self.errors)


def validate(tool: Tool, arguments: Any, null_is_absent: bool = True) -> Validation:
    """Check a parsed tool call against its schema.

    The error messages are written to be handed straight back to the model. "missing required
    key `city`" produces a correction; "ValidationError" produces another wrong answer.

    `null_is_absent` treats `{"limit": null}` as if `limit` had been omitted, for optional
    parameters only. It defaults to True because it is the right reading: the schema says the
    key is optional, and a model answering `null` has said "not this one". Refusing that is
    pedantry with a measurable price - on chapter 05's runs it accounts for **89% of every
    first-attempt failure**, under a one-line tool description and a six-line one alike.

    Required parameters are never treated this way. A required key answered `null` is a real
    refusal to do the job and should be reported as one.
    """
    if not isinstance(arguments, dict):
        return Validation(False, errors=[f"expected a JSON object, got {type(arguments).__name__}"])

    errors: list[str] = []
    by_name = {p.name: p for p in tool.parameters}

    if null_is_absent:
        optional = {p.name for p in tool.parameters if not p.required}
        arguments = {
            key: value
            for key, value in arguments.items()
            if not (value is None and key in optional)
        }

    for parameter in tool.parameters:
        if parameter.name not in arguments:
            if parameter.required:
                errors.append(f"missing required key `{parameter.name}`")
            continue
        value = arguments[parameter.name]
        expected = JSON_TYPES[parameter.type]
        # bool is a subclass of int in Python; an integer field must not accept True
        if parameter.type in ("integer", "number") and isinstance(value, bool):
            errors.append(f"`{parameter.name}` must be a {parameter.type}, got boolean")
            continue
        if not isinstance(value, expected):
            errors.append(
                f"`{parameter.name}` must be a {parameter.type}, got "
                f"{type(value).__name__} ({json.dumps(value, ensure_ascii=False)[:40]})"
            )
            continue
        if parameter.enum and value not in parameter.enum:
            errors.append(f"`{parameter.name}` must be one of {list(parameter.enum)}, got {value!r}")

    for key in arguments:
        if key not in by_name:
            errors.append(f"unknown key `{key}`")

    return Validation(not errors, arguments=dict(arguments) if not errors else {}, errors=errors)


@dataclass
class Attempt:
    """One pass through the repair loop, kept so the chapter can print the table."""

    number: int
    raw: str
    parsed: dict | None
    valid: bool
    errors: list[str]


@dataclass
class Repair:
    """The outcome of a bounded repair loop."""

    ok: bool
    arguments: dict[str, Any]
    attempts: list[Attempt]

    @property
    def tries(self) -> int:
        return len(self.attempts)


def repair(
    tool: Tool,
    ask: Callable[[str | None], str],
    max_attempts: int = 3,
    null_is_absent: bool = True,
) -> Repair:
    """Ask, validate, and ask again with the error - at most `max_attempts` times.

    `ask` takes the feedback from the previous attempt (or `None` on the first) and returns the
    model's raw text. Injecting it keeps this function testable without a model, and lets the
    chapter drive it from a cassette.

    Bounded on purpose. An unbounded repair loop against a model that cannot satisfy the schema
    is an infinite bill, and chapter 05 shows a schema where exactly that happens.
    """
    from aihe.models import parse_json_object

    attempts: list[Attempt] = []
    feedback: str | None = None

    for number in range(1, max_attempts + 1):
        raw = ask(feedback)
        parsed = parse_json_object(raw)
        if parsed is None:
            result = Validation(False, errors=["no JSON object found in the reply"])
        else:
            result = validate(tool, parsed, null_is_absent=null_is_absent)
        attempts.append(Attempt(number, raw, parsed, result.ok, list(result.errors)))
        if result.ok:
            return Repair(True, result.arguments, attempts)
        feedback = result.feedback()

    return Repair(False, {}, attempts)


def summarise(repairs: Sequence[Repair]) -> dict[str, float]:
    """Aggregate a batch of repair loops into the numbers a chapter reports."""
    if not repairs:
        return {"n": 0, "first_try": 0.0, "eventually": 0.0, "never": 0.0, "mean_tries": 0.0}
    n = len(repairs)
    first = sum(1 for r in repairs if r.ok and r.tries == 1)
    ok = sum(1 for r in repairs if r.ok)
    return {
        "n": n,
        "first_try": first / n,
        "eventually": ok / n,
        "never": (n - ok) / n,
        "mean_tries": sum(r.tries for r in repairs) / n,
    }


TASK_PROMPT = "{hint}\n\nTask: {task}"
RETRY_PROMPT = "\n\nYour previous reply was rejected: {feedback}\nTry again."


def run_tasks(
    tool: Tool,
    tasks: Sequence[str],
    ask_model: Callable[[str], str],
    max_attempts: int = 3,
    null_is_absent: bool = True,
) -> list[Repair]:
    """Run one tool over a list of tasks, repairing each until it validates or runs out.

    This lives here rather than in the notebook for the usual reason - a notebook cell may not
    define logic - and for one specific to recordings: the recorder and the chapter both call
    this, so the prompts they build are byte-identical and a cassette recorded by one is found
    by the other.
    """
    results: list[Repair] = []
    for task in tasks:
        def ask(feedback: str | None, task: str = task) -> str:
            prompt = TASK_PROMPT.format(hint=tool.prompt_hint(), task=task)
            if feedback:
                prompt += RETRY_PROMPT.format(feedback=feedback)
            return ask_model(prompt)

        results.append(repair(tool, ask, max_attempts=max_attempts,
                              null_is_absent=null_is_absent))
    return results
