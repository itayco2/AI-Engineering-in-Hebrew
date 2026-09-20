"""One interface to a model, four ways to get one.

Every chapter that needs generation goes through `chat()`. The interface is deliberately
tiny — messages in, text and tool calls out — because every chapter depends on it and a
churning interface would break all of them at once.

Backends are chosen with the `AIHE_BACKEND` environment variable:

    ollama    the default for a human at a laptop
    llamacpp  the portable path: pip-only, works natively on Windows
    replay    the CI default. Reads committed cassettes, makes no network call
    fake      canned strings, for unit tests

One thing to know before writing a test: **generated text is not reproducible**, even at
temperature 0 with a fixed seed. It drifts between runs and differs across platforms. So no
test in this repo asserts on generated text. Tests assert on the deterministic parts — the
retrieval, the metrics, the validators, the byte-exact prefix check.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aihe.cassettes import CassetteLibrary, key_for

DEFAULT_MODEL = "llama3.2:3b"
BACKENDS = ("ollama", "llamacpp", "replay", "fake")


@dataclass
class Response:
    """What a model said, normalised across backends."""

    text: str
    model: str
    tool_calls: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @property
    def called_tools(self) -> list[str]:
        return [str(call.get("name", "")) for call in self.tool_calls]


def backend_name() -> str:
    name = os.environ.get("AIHE_BACKEND", "ollama").strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"AIHE_BACKEND must be one of {BACKENDS}, got {name!r}")
    return name


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    cassettes: str | Path | None = None,
    **params: Any,
) -> Response:
    """Send messages to a model and get one `Response` back.

    `cassettes` points at a chapter's `cassettes/` directory and is required in `replay`
    mode. It is per-chapter on purpose: re-recording one chapter must not disturb another.
    """
    name = backend_name()
    if name == "fake":
        from aihe.backends.fake import complete
        return complete(messages, tools, model, **params)
    if name == "replay":
        from aihe.backends.replay import complete
        if cassettes is None:
            raise ValueError(
                "replay mode needs a cassettes directory: chat(..., cassettes=CASSETTES)"
            )
        return complete(messages, tools, model, Path(cassettes), **params)
    if name == "llamacpp":
        from aihe.backends.llamacpp import complete
        return complete(messages, tools, model, **params)
    from aihe.backends.ollama import complete
    return complete(messages, tools, model, **params)


def record(
    messages: list[dict],
    cassettes: str | Path,
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    **params: Any,
) -> Path:
    """Call a real model and write the answer to a cassette. Used only by `make record`.

    Uses whichever live backend `AIHE_BACKEND` names. Recording from `replay` would be circular
    and recording from `fake` would write fiction, so both are refused.
    """
    name = backend_name()
    if name in ("replay", "fake"):
        raise ValueError(
            f"cannot record from the {name!r} backend - it makes no real call.\n"
            "Set AIHE_BACKEND=llamacpp (with AIHE_GGUF) or AIHE_BACKEND=ollama."
        )
    if name == "llamacpp":
        from aihe.backends.llamacpp import complete
    else:
        from aihe.backends.ollama import complete

    response = complete(messages, tools, model, **params)
    library = CassetteLibrary(Path(cassettes))
    key = key_for(model, messages, tools, **params)
    return library.save(
        key,
        request={"messages": messages, "tools": tools, "params": params},
        response={"text": response.text, "tool_calls": response.tool_calls},
        model=model,
    )


def asker(
    cassettes: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    record_to: str | Path | None = None,
    **params: Any,
) -> Any:
    """A one-string-in, one-string-out function for the judges and the repair loop.

    The same function serves both directions. With `cassettes` it replays; with `record_to` it
    calls a live model and writes the answer down. That is what keeps a recording byte-identical
    to the request the notebook will later make - both go through this, so the cassette key is
    computed from exactly the same messages.
    """

    def ask(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if record_to is not None:
            from aihe.cassettes import CassetteLibrary, key_for

            library = CassetteLibrary(Path(record_to))
            key = key_for(model, messages, None, **params)
            # Recording is idempotent. Asking the same question twice in one run must not
            # overwrite the first answer, because generated text is not reproducible even at
            # temperature 0 - the second reply differs, the cassette changes underneath the
            # first caller, and replay then walks a path that was never recorded.
            if not library.has(key):
                record(messages, record_to, model=model, **params)
            return library.load(key)["response"]["text"]
        return chat(messages, model=model, cassettes=cassettes, **params).text

    return ask


def parse_json_object(text: str) -> dict | None:
    """Pull the first JSON object out of a model's reply, or return None.

    Small models wrap JSON in prose, in code fences, or in an apology. This is the honest
    first line of defence, and chapter 05 measures exactly how often it is needed.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None
