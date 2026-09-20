"""Generate the README's hero image from the real pipeline.

Not a mockup and not a screenshot of someone else's chart. This runs the chapter's own
retrieval and draws the numbers it produces, so the picture on the front page cannot drift
away from what the course actually teaches.

Run: python scripts/make_hero.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aihe import viz
from aihe.embeddings import cross_encoder_scorer, encode
from aihe.pipeline import (
    build_chunks, contextualise, dense_retriever, evaluate, hybrid_retriever,
    load_corpus, reranked_retriever,
)
from aihe.retrieval import BM25, tokenize

OUT = ROOT / "docs" / "assets" / "staircase.png"


def main() -> int:
    corpus = load_corpus(ROOT / "chapters/01-rag/data/meridian.json")
    chunks = build_chunks(corpus)
    rich = contextualise(chunks, corpus)
    full = {d["id"]: f"{d['title']}. {d['text']}" for d in corpus.documents}

    questions = encode(corpus.questions())
    bare_vectors = encode([c.text for c in chunks])
    rich_vectors = encode(rich)
    bm25 = BM25([tokenize(t) for t in rich])

    plain = dense_retriever(questions, bare_vectors, chunks)
    withctx = dense_retriever(questions, rich_vectors, chunks)
    hybrid = hybrid_retriever(questions, rich_vectors, bm25, chunks)
    reranked = reranked_retriever(hybrid, lambda d: full[d], cross_encoder_scorer(), shortlist=5)

    steps = {
        "plain": plain,
        "+ context": withctx,
        "+ BM25": hybrid,
        "+ reranker": reranked,
    }
    # recall@1 rather than the failure rate: it is the chapter's headline number, it climbs
    # monotonically across all four steps, and the last step moves it while leaving failed@3
    # flat. A hero image with a flat final bar teaches the wrong thing about reranking.
    results = {name: evaluate(r, corpus, k=1)["recall@1"] for name, r in steps.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    viz.save(
        viz.climb(
            results,
            label="recall@1",
            title="how often the right document comes back first",
        ),
        str(OUT),
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    for name, value in results.items():
        print(f"  {name:12s} {value:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
