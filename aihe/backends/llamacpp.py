"""llama.cpp in-process, for anyone who would rather not install a second service.

Pip-installable and native on Windows, which matters: the zero-cost promise breaks on
hardware and platform far more often than on money.
"""

from __future__ import annotations

import functools
import os
from typing import Any

from aihe.models import Response

MODEL_PATH_VAR = "AIHE_GGUF"

MISSING = f"""llama-cpp-python is not installed, or no model file was given.

  pip install llama-cpp-python
  export {MODEL_PATH_VAR}=/path/to/a/model.gguf

Or use a backend that needs neither:
  AIHE_BACKEND=replay     the recorded answers committed in this repo
"""


@functools.lru_cache(maxsize=1)
def _load(path: str, context: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover
        raise ImportError(MISSING) from exc
    return Llama(model_path=path, n_ctx=context, verbose=False)


def complete(
    messages: list[dict], tools: list[dict] | None, model: str, **params: Any
) -> Response:
    path = os.environ.get(MODEL_PATH_VAR)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(MISSING)

    llama = _load(path, int(params.pop("n_ctx", 4096)))
    kwargs = {k: v for k, v in params.items() if not k.startswith("fake_")}
    result = llama.create_chat_completion(messages=messages, tools=tools or None, **kwargs)
    choice = (result.get("choices") or [{}])[0].get("message", {})
    return Response(
        text=choice.get("content") or "",
        model=os.path.basename(path),
        tool_calls=[
            {"name": c.get("function", {}).get("name"), "arguments": c.get("function", {}).get("arguments")}
            for c in choice.get("tool_calls", []) or []
        ],
        raw=result,
    )
