"""Read a committed cassette instead of calling a model. The CI default.

A miss fails loudly and names the command that fixes it. That error message is the whole
maintenance story: when a notebook changes, the build says "run `make record`" rather than
failing somewhere strange half an hour later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aihe.cassettes import CassetteLibrary, key_for
from aihe.models import Response


def complete(
    messages: list[dict],
    tools: list[dict] | None,
    model: str,
    cassettes: Path,
    **params: Any,
) -> Response:
    library = CassetteLibrary(Path(cassettes))
    data = library.load(key_for(model, messages, tools, **params))
    response = data.get("response", {})
    return Response(
        text=response.get("text", ""),
        model=data.get("model", model),
        tool_calls=list(response.get("tool_calls", [])),
        raw=data,
    )
