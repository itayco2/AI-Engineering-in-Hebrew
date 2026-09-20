"""Re-record every chapter's cassettes from a real model, then review the diff.

Recording is always a deliberate, human-run action. CI never records: `AIHE_BACKEND=replay`
is the default there and a miss fails loudly rather than quietly reaching for the network
and a bill.

Run: python scripts/record_cassettes.py    (or: make record)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml

    from aihe.models import DEFAULT_MODEL, record

    chapters = sorted(p for p in (ROOT / "chapters").iterdir() if p.is_dir())
    needing = []
    for chapter in chapters:
        meta = yaml.safe_load((chapter / "meta.yml").read_text(encoding="utf-8"))
        if meta.get("needs_llm"):
            needing.append((chapter, meta))

    if not needing:
        print("No chapter needs generation yet, so there is nothing to record.")
        print("Chapters 01-03 use embeddings only - that is what keeps them free to run.")
        return 0

    for chapter, meta in needing:
        requests_file = chapter / "cassettes" / "requests.json"
        if not requests_file.exists():
            print(f"{chapter.name}: no cassettes/requests.json, skipping")
            continue
        requests = json.loads(requests_file.read_text(encoding="utf-8"))
        print(f"\n{chapter.name}: recording {len(requests)} request(s) with a real model")
        for request in requests:
            path = record(
                messages=request["messages"],
                cassettes=chapter / "cassettes",
                tools=request.get("tools"),
                model=request.get("model", DEFAULT_MODEL),
                **request.get("params", {}),
            )
            print(f"  wrote {path.relative_to(ROOT)}")

    print("\nNow review the diff. A changed answer is a real change to what the chapter teaches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
