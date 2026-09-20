"""Finding the right pieces: keyword search, vector search, and merging the two.

BM25 is implemented here rather than imported, for two reasons. It is the thing chapter 01
is teaching, so hiding it behind a dependency defeats the purpose; and a pure-Python
implementation keeps `requirements-test.txt` lean enough that CI stays under a minute.

Every function returns `list[tuple[int, float]]`, (index into the corpus, score) sorted by
score descending. One return shape everywhere means the merge step needs no adapters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

Hits = list[tuple[int, float]]

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word characters.

    Unicode-aware, so Hebrew survives. It is deliberately naive: chapter 02 shows what this
    does to Hebrew, where `ובמסמך` is one token here and four morphemes in reality.
    """
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25, the keyword ranking function from the 1990s that still holds up.

    A document scores higher the more query words it contains; rare words count for more;
    repeating a word has diminishing returns (`k1`); and long documents get no unfair edge
    (`b`). It matches literal strings, so it wins exactly where embeddings blur, error
    codes, product IDs, function names, people's names.

    The IDF here is the Lucene variant, `ln(1 + (N - df + 0.5) / (df + 0.5))`, which stays
    positive even for a term appearing in every document. The textbook form goes negative
    there and lets a ubiquitous term subtract from a score, which is never what you want.
    """

    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b must be in [0, 1], got {b}")
        if k1 < 0.0:
            raise ValueError(f"k1 must not be negative, got {k1}")

        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = [list(doc) for doc in corpus]
        self.n_docs = len(self.corpus)
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avg_doc_len = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        self.term_freqs: list[Counter[str]] = [Counter(doc) for doc in self.corpus]

        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        self.doc_freq = doc_freq
        self.idf = {
            term: math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document against one query."""
        tf = self.term_freqs[doc_index]
        length_norm = self.b * (self.doc_lens[doc_index] / self.avg_doc_len) if self.avg_doc_len else 0.0
        denom_base = self.k1 * (1.0 - self.b + length_norm)

        total = 0.0
        for term in query:
            freq = tf.get(term)
            if not freq:
                continue
            total += self.idf[term] * (freq * (self.k1 + 1.0)) / (freq + denom_base)
        return total

    def search(self, query: Sequence[str], k: int = 10) -> Hits:
        """Rank the whole corpus against `query` and return the top `k`.

        Documents scoring zero are dropped: BM25 cannot rank a document that shares no
        query term, and padding the list with them makes recall look better than it is.
        """
        scored = [(i, self.score(query, i)) for i in range(self.n_docs)]
        hits = [(i, s) for i, s in scored if s > 0.0]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits[:k]


def cosine_search(query_vector, doc_vectors, k: int = 10) -> Hits:
    """Rank documents by cosine similarity to the query vector.

    `doc_vectors` is (n_docs, dim). Vectors are normalised here rather than assumed
    normalised, because a silently un-normalised matrix turns cosine into a dot product and
    quietly favours long documents, a bug that produces plausible-looking results.
    """
    import numpy as np

    docs = np.asarray(doc_vectors, dtype=np.float64)
    query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    if docs.ndim != 2:
        raise ValueError(f"doc_vectors must be 2-D, got shape {docs.shape}")
    if docs.shape[1] != query.shape[0]:
        raise ValueError(f"dimension mismatch: docs {docs.shape[1]}, query {query.shape[0]}")

    doc_norms = np.linalg.norm(docs, axis=1)
    query_norm = np.linalg.norm(query)
    safe = np.where(doc_norms == 0.0, 1.0, doc_norms)
    sims = (docs @ query) / (safe * (query_norm or 1.0))
    sims = np.where(doc_norms == 0.0, -1.0, sims)

    order = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in order]


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> Hits:
    """Reciprocal Rank Fusion (Cormack et al., 2009): merge ranked lists using ranks only.

        score(d) = sum over lists of 1 / (k + rank of d in that list)

    Ranks are 1-based. Using ranks rather than scores is the whole point, you never have to
    decide whether a BM25 score of 8.2 beats a cosine similarity of 0.71.

    `k = 60` is the paper's value and the reason steady agreement beats one lucky first
    place: a document at rank 1 and rank 10 scores 1/61 + 1/70 = 0.0307, while a document at
    rank 3 in both lists scores 2/63 = 0.0317, and wins.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, doc_index in enumerate(ranking, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (k + position)

    hits = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    return hits


def rerank(
    query: str,
    candidates: Sequence[str],
    scorer: Callable[[str, Sequence[str]], Sequence[float]],
    top_k: int = 20,
) -> Hits:
    """Re-score a shortlist with an expensive model and keep the best `top_k`.

    Retrieve wide and cheap, then re-score narrow and expensive. A cross-encoder reads the
    question and one candidate *together*, which is why it is more accurate than comparing
    two independently-computed vectors, and why it is far too slow to run over a million
    documents but fine over 150.

    `scorer` is injected rather than constructed here so this function stays pure and
    testable: the tests pass a deterministic stub, the notebook passes a real model.
    """
    if not candidates:
        return []
    scores = list(scorer(query, list(candidates)))
    if len(scores) != len(candidates):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} candidates")

    hits = sorted(enumerate(float(s) for s in scores), key=lambda pair: (-pair[1], pair[0]))
    return hits[:top_k]


def indices(hits: Hits) -> list[int]:
    """Drop the scores. Ranked lists are what `rrf` and the metrics consume."""
    return [i for i, _ in hits]
