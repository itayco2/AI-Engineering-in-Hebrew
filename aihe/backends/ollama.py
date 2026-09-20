"""Ollama over HTTP, using only the standard library.

`urllib` rather than `requests` so that running a chapter needs one fewer dependency. The
error message on a refused connection is worth as much as the happy path: someone meeting
this repo for the first time usually has not started Ollama.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from aihe.models import Response

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
TIMEOUT = float(os.environ.get("AIHE_TIMEOUT", "120"))

NOT_RUNNING = f"""Could not reach Ollama at {HOST}.

  1. Install it:  https://ollama.com/download
  2. Pull a small model:  ollama pull llama3.2:3b
  3. Leave it running, then try again.

No Ollama and no wish to install it? Two other routes, both free:
  AIHE_BACKEND=llamacpp   pip-only, works natively on Windows
  AIHE_BACKEND=replay     the recorded answers already committed in this repo
"""


def complete(
    messages: list[dict], tools: list[dict] | None, model: str, **params: Any
) -> Response:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    options = {k: v for k, v in params.items() if not k.startswith("fake_")}
    if options:
        payload["options"] = options

    request = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as handle:
            body = json.loads(handle.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(NOT_RUNNING) from exc

    message = body.get("message", {}) or {}
    return Response(
        text=message.get("content", ""),
        model=body.get("model", model),
        tool_calls=[
            {"name": c.get("function", {}).get("name"), "arguments": c.get("function", {}).get("arguments")}
            for c in message.get("tool_calls", []) or []
        ],
        raw=body,
    )
