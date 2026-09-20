"""Turning text into vectors, on a CPU, for free.

Chapters 01 to 03 use embeddings and no generation at all. That is deliberate: embeddings
run on any laptop in a few hundred megabytes, they are deterministic, and they let the
course make its zero-cost promise without asking anyone to install Ollama or own 8 GB of
spare RAM. It also means CI can execute those chapters for real rather than from a recording.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

# Small, multilingual, CPU-friendly. 384 dimensions, about 470 MB on disk.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_MISSING = (
    "sentence-transformers is not installed.\n"
    "It is not in requirements-test.txt on purpose — the fast suite must not pay for it.\n"
    "Run:  make setup     (or: pip install -r requirements.txt)"
)


@functools.lru_cache(maxsize=4)
def _load(model_name: str):
    """Load once per process. Reloading a model per cell is the usual notebook mistake."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MISSING) from exc
    return SentenceTransformer(model_name, device="cpu")


def encode(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    normalize: bool = True,
    batch_size: int = 32,
):
    """Embed a list of strings into an (n, dim) float array.

    Normalised by default, because `aihe.retrieval.cosine_search` normalises again and two
    normalisations are harmless, while zero normalisations turn cosine into a dot product
    that quietly favours long documents.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = _load(model_name).encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


@functools.lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_MISSING) from exc
    return CrossEncoder(model_name, device="cpu")


# Multilingual (XLM-R based), so the same reranker carries into the Hebrew chapters.
# NOT cross-encoder/ms-marco-MiniLM-L-6-v2: on torch 2.14 its forward pass returns NaN for
# every pair, with no error. Its parameters are clean and the bi-encoder is unaffected, so
# nothing looks wrong -- the reranker silently becomes a no-op. See PREFLIGHT.md.
DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def cross_encoder_scorer(model_name: str = DEFAULT_RERANKER):
    """Build a scorer for `aihe.retrieval.rerank`.

    A cross-encoder reads the question and one candidate together and scores the pair, which
    is why it beats comparing two vectors computed in ignorance of each other — and why it
    is far too slow for a million documents and fine for 150.
    """

    def score(query: str, candidates: Sequence[str]) -> list[float]:
        model = _load_cross_encoder(model_name)
        scores = [float(s) for s in model.predict([(query, c) for c in candidates])]
        if any(s != s for s in scores):  # NaN != NaN
            raise ValueError(
                f"{model_name} returned NaN scores.\n"
                "A NaN score is not a low score -- sorting leaves the order untouched, so the "
                "reranker becomes a no-op that reports no error and changes no metric. "
                "Fail here instead."
            )
        return scores

    return score
