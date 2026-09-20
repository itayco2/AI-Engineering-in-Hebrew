"""Measure chapter 01's staircase. Run it before trusting a number in the chapter.

This is the `research/` pattern from driving-rl: one file, one question, run it directly.
The chapter quotes whatever this prints. If the two disagree, this one is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aihe.embeddings import cross_encoder_scorer, encode
from aihe.pipeline import (
    build_chunks,
    contextualise,
    dense_retriever,
    evaluate,
    evaluate_by_kind,
    hybrid_retriever,
    load_corpus,
    reranked_retriever,
    sparse_retriever,
)
from aihe.retrieval import BM25, tokenize

K = 3
CORPUS = Path(__file__).resolve().parent.parent / "chapters/01-rag/data/meridian.json"


def main() -> int:
    corpus = load_corpus(CORPUS)
    chunks = build_chunks(corpus)
    bare = [c.text for c in chunks]
    rich = contextualise(chunks, corpus)
    full_text = {d["id"]: f"{d['title']}. {d['text']}" for d in corpus.documents}

    print(f"{len(corpus.documents)} documents -> {len(chunks)} chunks, {len(corpus.queries)} questions\n")

    bare_vectors = encode(bare)
    rich_vectors = encode(rich)
    question_vectors = encode(corpus.questions())
    bm25_rich = BM25([tokenize(t) for t in rich])

    plain = dense_retriever(question_vectors, bare_vectors, chunks)
    withctx = dense_retriever(question_vectors, rich_vectors, chunks)
    hybrid = hybrid_retriever(question_vectors, rich_vectors, bm25_rich, chunks)
    # shortlist=5, measured. A reranker can only reorder what it is handed, so a wide
    # shortlist gives it more chances to push a good answer down: at top-20 recall@3 falls
    # from 0.911 to 0.875 on this corpus. See PREFLIGHT.md.
    reranked = reranked_retriever(hybrid, lambda d: full_text[d], cross_encoder_scorer(), shortlist=5)

    steps = {
        "1  plain embeddings": plain,
        "2  + context prepended": withctx,
        "3  + BM25 fused in": hybrid,
        "4  + reranker": reranked,
        "   (BM25 alone)": sparse_retriever(bm25_rich, chunks),
    }

    print(f"{'setup':26s} {'r@1':>6s} {'r@3':>6s} {'nDCG@3':>7s} {'failed@3':>9s} {'cut':>7s}")
    baseline = None
    for name, retriever in steps.items():
        at1 = evaluate(retriever, corpus, k=1)["recall@1"]
        stats = evaluate(retriever, corpus, k=K)
        failed = stats[f"failed@{K}"]
        if name.startswith("1"):
            baseline = failed
        cut = "" if baseline in (None, 0) or name.startswith("   ") else f"{(baseline - failed) / baseline * 100:5.0f}%"
        print(
            f"{name:26s} {at1:6.3f} {stats[f'recall@{K}']:6.3f} "
            f"{stats[f'nDCG@{K}']:7.3f} {failed:9.3f} {cut:>7s}"
        )

    print(f"\nrecall@{K} by question kind:")
    header = f"  {'setup':26s}" + "".join(f"{k:>12s}" for k in ("identifier", "mixed", "paraphrase"))
    print(header)
    for name, retriever in steps.items():
        by_kind = evaluate_by_kind(retriever, corpus, k=K)
        row = f"  {name:26s}" + "".join(f"{by_kind.get(k, 0.0):12.2f}" for k in ("identifier", "mixed", "paraphrase"))
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
