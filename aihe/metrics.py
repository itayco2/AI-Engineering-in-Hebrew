"""Measuring retrieval, with no model in the loop.

Every function here is deterministic arithmetic over a ranked list and a set of known-good
answers. That is the point: the half of a RAG system that can be measured exactly should be
measured exactly, before anyone asks an LLM to grade anything.

Convention throughout: `retrieved` is a ranked sequence of document ids, best first, and
`relevant` is the set of ids that should have been found. Ids can be ints or strings as long
as one query uses one kind.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Sequence

Doc = int | str


def _top_k(retrieved: Sequence[Doc], k: int) -> list[Doc]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return list(retrieved[:k])


def recall_at_k(retrieved: Sequence[Doc], relevant: Collection[Doc], k: int) -> float:
    """Share of the relevant documents that appear in the top `k`.

    The metric that matters most for RAG: a document the retriever never surfaced cannot be
    used by the generator, no matter how good the prompt is.
    """
    if not relevant:
        return 0.0
    found = len(set(_top_k(retrieved, k)) & set(relevant))
    return found / len(set(relevant))


def precision_at_k(retrieved: Sequence[Doc], relevant: Collection[Doc], k: int) -> float:
    """Share of the top `k` that is relevant. Divided by `k`, not by how many were returned.

    Dividing by the number actually returned flatters a retriever that returns three
    documents, which is why it is not done here.
    """
    if not relevant:
        return 0.0
    found = len(set(_top_k(retrieved, k)) & set(relevant))
    return found / k


def reciprocal_rank(retrieved: Sequence[Doc], relevant: Collection[Doc]) -> float:
    """1 / rank of the first relevant document, or 0.0 if none was retrieved."""
    relevant_set = set(relevant)
    for position, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / position
    return 0.0


def dcg_at_k(retrieved: Sequence[Doc], relevant: Collection[Doc], k: int) -> float:
    """Discounted cumulative gain with binary relevance, log2 discount."""
    relevant_set = set(relevant)
    return sum(
        1.0 / math.log2(position + 1)
        for position, doc in enumerate(_top_k(retrieved, k), start=1)
        if doc in relevant_set
    )


def ndcg_at_k(retrieved: Sequence[Doc], relevant: Collection[Doc], k: int) -> float:
    """DCG divided by the best DCG achievable — so 1.0 means perfectly ordered.

    Unlike recall, this rewards putting the right document at rank 1 rather than rank 9.
    """
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(set(relevant)), k) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def hit_rate(retrieved: Sequence[Doc], relevant: Collection[Doc], k: int) -> float:
    """1.0 if anything relevant made the top `k`. The most forgiving metric there is."""
    return 1.0 if set(_top_k(retrieved, k)) & set(relevant) else 0.0


# --- aggregates over a whole query set -------------------------------------------------

Query = tuple[Sequence[Doc], Collection[Doc]]


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def mean_recall_at_k(queries: Sequence[Query], k: int) -> float:
    return _mean(recall_at_k(r, rel, k) for r, rel in queries)


def mean_precision_at_k(queries: Sequence[Query], k: int) -> float:
    return _mean(precision_at_k(r, rel, k) for r, rel in queries)


def mrr(queries: Sequence[Query]) -> float:
    """Mean reciprocal rank across queries."""
    return _mean(reciprocal_rank(r, rel) for r, rel in queries)


def mean_ndcg_at_k(queries: Sequence[Query], k: int) -> float:
    return _mean(ndcg_at_k(r, rel, k) for r, rel in queries)


def failed_retrieval_rate(queries: Sequence[Query], k: int = 20) -> float:
    """`1 - mean recall@k` — the share of relevant chunks missing from the top `k`.

    This is the metric Anthropic reported their contextual-retrieval staircase in (2024):
    5.7% failed retrievals with plain embeddings, 3.7% with a context sentence prepended to
    every chunk, 2.9% adding BM25, 1.9% adding a reranker. Stated as a failure rate rather
    than as recall because halving a failure rate reads as the improvement it is, where
    "recall went from 94.3% to 97.1%" does not.

    Those four numbers are Anthropic's, measured on their corpora. Chapter 01 reproduces the
    *shape* on a small corpus and reports its own numbers.
    """
    return 1.0 - mean_recall_at_k(queries, k)


def summary(queries: Sequence[Query], k: int = 20) -> dict[str, float]:
    """Every metric at once, for the table a chapter prints."""
    return {
        f"recall@{k}": mean_recall_at_k(queries, k),
        f"precision@{k}": mean_precision_at_k(queries, k),
        f"nDCG@{k}": mean_ndcg_at_k(queries, k),
        "MRR": mrr(queries),
        f"failed@{k}": failed_retrieval_rate(queries, k),
    }
