"""A backend that invents answers, for unit tests only.

Never used by a chapter. A chapter that needs generation uses `replay`, because a fake
returns what its author imagined and a cassette returns what a model actually said.
"""

from __future__ import annotations

from typing import Any

from aihe.models import Response

REPLY = "זו תשובה קבועה לבדיקות."


def complete(
    messages: list[dict], tools: list[dict] | None, model: str, **params: Any
) -> Response:
    last = messages[-1]["content"] if messages else ""
    return Response(
        text=params.get("fake_text", REPLY),
        model=f"fake/{model}",
        tool_calls=list(params.get("fake_tool_calls", [])),
        raw={"echo": last},
    )
