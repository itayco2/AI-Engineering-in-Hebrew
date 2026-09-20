"""The whole retrieval pipeline, assembled, so a notebook cell can be one line.

Chapter 01 walks up a staircase: plain embeddings, then context prepended to every chunk,
then BM25 fused in, then a reranker over the shortlist. Each step is one function here, and
each returns ranked document ids so the metrics never have to care which step produced them.

Relevance is judged at document level, not chunk level. A chunk is an implementation detail
of retrieval; what the reader asked is whether the right *document* came back.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aihe.chunking import Chunk, recursive
from aihe.metrics import Query, summary
from aihe.retrieval import BM25, Hits, cosine_search, rrf, tokenize

Ranked = list[str]
Retriever = Callable[[int, str], Ranked]


@dataclass
class Corpus:
    """Documents, questions, and the answers each question should find."""

    documents: list[dict]
    queries: list[dict]

    @property
    def titles(self) -> dict[str, str]:
        return {d["id"]: d["title"] for d in self.documents}

    def questions(self) -> list[str]:
        return [q["question"] for q in self.queries]

    def relevant(self) -> list[set[str]]:
        return [set(q["relevant"]) for q in self.queries]

    def of_kind(self, kind: str) -> list[int]:
        return [i for i, q in enumerate(self.queries) if q.get("kind") == kind]


def load_corpus(path: str | Path) -> Corpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Corpus(documents=data["documents"], queries=data["queries"])


def build_chunks(
    corpus: Corpus, size: int = 200, overlap: int = 40, strategy=recursive
) -> list[Chunk]:
    return [
        chunk
        for doc in corpus.documents
        for chunk in strategy(doc["text"], size=size, overlap=overlap, doc_id=doc["id"])
    ]


def contextualise(chunks: Sequence[Chunk], corpus: Corpus) -> list[str]:
    """Prepend the document's title to each of its chunks.

    This is the cheapest form of Anthropic's contextual retrieval (2024). A chunk cut out of
    the middle of a document loses the one thing that said what the document was about, and
    both the embedding and the keyword index are then searching text that no longer mentions
    its own subject. Putting the title back is one line and, on this corpus, it is the single
    largest improvement in the chapter.
    """
    titles = corpus.titles
    return [f"{titles[c.doc_id]}. {c.text}" for c in chunks]


def to_documents(hits: Hits, chunks: Sequence[Chunk]) -> Ranked:
    """Collapse ranked chunks into ranked documents, keeping first appearance."""
    seen: set[str] = set()
    ordered: Ranked = []
    for index, _ in hits:
        doc_id = chunks[index].doc_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def evaluate(retriever: Retriever, corpus: Corpus, k: int = 3) -> dict[str, float]:
    """Run a retriever over every question and summarise."""
    pairs: list[Query] = [
        (retriever(i, q["question"]), set(q["relevant"]))
        for i, q in enumerate(corpus.queries)
    ]
    return summary(pairs, k=k)


def evaluate_by_kind(retriever: Retriever, corpus: Corpus, k: int = 3) -> dict[str, float]:
    """recall@k split by the kind of question, which is where the mechanisms show."""
    out: dict[str, float] = {}
    for kind in sorted({q.get("kind", "other") for q in corpus.queries}):
        indexes = corpus.of_kind(kind)
        pairs: list[Query] = [
            (retriever(i, corpus.queries[i]["question"]), set(corpus.queries[i]["relevant"]))
            for i in indexes
        ]
        out[kind] = summary(pairs, k=k)[f"recall@{k}"]
    return out


# --- the four steps ---------------------------------------------------------------------


def dense_retriever(query_vectors, chunk_vectors, chunks, width: int = 40) -> Retriever:
    """Step 1 and 2: vector search. Which vectors were built decides which step it is."""

    def retrieve(index: int, _question: str) -> Ranked:
        return to_documents(cosine_search(query_vectors[index], chunk_vectors, k=width), chunks)

    return retrieve


def sparse_retriever(bm25: BM25, chunks, width: int = 40) -> Retriever:
    def retrieve(_index: int, question: str) -> Ranked:
        return to_documents(bm25.search(tokenize(question), k=width), chunks)

    return retrieve


def hybrid_retriever(query_vectors, chunk_vectors, bm25, chunks, width: int = 40) -> Retriever:
    """Step 3: fuse the two ranked lists with RRF, using ranks and never scores."""

    def retrieve(index: int, question: str) -> Ranked:
        dense_ranks = [i for i, _ in cosine_search(query_vectors[index], chunk_vectors, k=width)]
        sparse_ranks = [i for i, _ in bm25.search(tokenize(question), k=width)]
        return to_documents(rrf([dense_ranks, sparse_ranks]), chunks)

    return retrieve


def reranked_retriever(
    base: Retriever,
    texts_for: Callable[[str], str],
    scorer: Callable[[str, Sequence[str]], Sequence[float]],
    shortlist: int = 20,
) -> Retriever:
    """Step 4: re-score the shortlist with a cross-encoder and reorder it.

    Retrieve wide and cheap, then re-score narrow and expensive. The reranker never sees the
    documents the base retriever failed to surface, so it can fix an ordering and can never
    fix a miss, which is why recall deep in the list does not move and recall at rank 1 does.
    """

    def retrieve(index: int, question: str) -> Ranked:
        candidates = base(index, question)[:shortlist]
        if not candidates:
            return []
        scores = list(scorer(question, [texts_for(doc_id) for doc_id in candidates]))
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [candidates[i] for i in order]

    return retrieve


# --- passage-level retrieval ---------------------------------------------------------------
# Chapter 02 evaluates whole passages rather than chunks, because HeQ's passages are already
# the retrieval unit. Keeping these separate from the chunk-based retrievers above avoids
# threading an "are these chunks or documents?" flag through everything.


def encode_corpus(model, corpus: Corpus, asymmetric: bool = False):
    """Embed the passages and the questions, adding e5's prefixes when asked.

    Some embedding models are **asymmetric**: they are trained with `query:` on one side and
    `passage:` on the other, and omitting the prefixes costs accuracy without raising anything.
    It is a trap worth making explicit rather than hiding inside a wrapper.
    """
    docs = [d["text"] for d in corpus.documents]
    questions = corpus.questions()
    if asymmetric:
        docs = [f"passage: {d}" for d in docs]
        questions = [f"query: {q}" for q in questions]
    kwargs = {"normalize_embeddings": True, "show_progress_bar": False, "batch_size": 16}
    return model.encode(docs, **kwargs), model.encode(questions, **kwargs)


def passage_retriever(question_vectors, passage_vectors, corpus: Corpus, width: int = 20):
    """Rank whole passages by cosine similarity."""
    ids = [d["id"] for d in corpus.documents]

    def retrieve(index: int, _question: str) -> Ranked:
        hits = cosine_search(question_vectors[index], passage_vectors, k=width)
        return [ids[i] for i, _ in hits]

    return retrieve


def keyword_retriever(corpus: Corpus, tokenizer, width: int = 20):
    """Rank whole passages with BM25, using whichever tokenizer is passed in.

    The tokenizer is injected because swapping it is the entire experiment in chapter 02.
    """
    ids = [d["id"] for d in corpus.documents]
    bm25 = BM25([tokenizer(d["text"]) for d in corpus.documents])

    def retrieve(_index: int, question: str) -> Ranked:
        return [ids[i] for i, _ in bm25.search(tokenizer(question), k=width)]

    return retrieve
