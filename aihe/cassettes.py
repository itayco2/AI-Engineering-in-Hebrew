"""Recorded model responses, so CI costs nothing and never flakes.

A cassette is one real response from one real model, keyed by a hash of exactly what was
asked. Record once against a live model, commit the file, and every later run in `replay`
mode reads from disk: no network, no key, no bill, no drift.

The distinction that matters: a cassette is not a mock. A mock returns what its author
imagined, which is why a mock of a small model returns clean tool-call JSON and the chapter
about small models emitting *broken* JSON quietly stops being tested. A cassette returns
what the model actually said. It is a mock that was told the truth.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CassetteMiss(KeyError):
    """Raised when replay mode is asked for something that was never recorded."""


def key_for(model: str, messages: list[dict], tools: Any = None, **params: Any) -> str:
    """A stable hash of the request.

    Canonical JSON with sorted keys, so the same request always produces the same key
    regardless of dict ordering. Volatile parameters that do not change the answer's meaning
    are excluded by the caller, not here — this function hashes exactly what it is given.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, "tools": tools, "params": params},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class CassetteLibrary:
    """The cassettes belonging to one chapter.

    Per-chapter and never global, so re-recording chapter 05 cannot break chapter 03.
    """

    directory: Path
    record_command: str = "make record"
    _cache: dict[str, dict] = field(default_factory=dict, repr=False)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def has(self, key: str) -> bool:
        return key in self._cache or self.path_for(key).exists()

    def load(self, key: str) -> dict:
        if key in self._cache:
            return self._cache[key]
        path = self.path_for(key)
        if not path.exists():
            raise CassetteMiss(
                f"No cassette {key} in {self.directory}.\n"
                f"The request changed, or this is new. Re-record with:  {self.record_command}"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self._cache[key] = data
        return data

    def save(self, key: str, request: dict, response: dict, model: str) -> Path:
        """Write a cassette, pinning the model inside it.

        The model name and version live in the file so that bumping a model shows up as a
        visible diff in review rather than as a mystery six weeks later.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        path.write_text(
            json.dumps(
                {"key": key, "model": model, "request": request, "response": response},
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._cache[key] = {"key": key, "model": model, "request": request, "response": response}
        return path

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))
