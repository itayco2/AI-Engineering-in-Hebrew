"""Retrieval: BM25 behaviour, cosine correctness, and the published RRF worked example."""


import pytest

from aihe.retrieval import BM25, cosine_search, indices, rerank, rrf, tokenize

CORPUS = [
    "the cat sat on the mat",
    "error code TS-999 means the token expired",
    "the dog sat on the log",
    "retrieval augmented generation needs a retriever and a generator",
]
TOKENS = [tokenize(doc) for doc in CORPUS]


def test_tokenize_is_lowercasing_and_unicode_aware():
    assert tokenize("Error TS-999") == ["error", "ts", "999"]
    assert tokenize("שלום עולם") == ["שלום", "עולם"]


def test_bm25_finds_the_literal_string_embeddings_would_blur():
    """The case Anthropic uses to justify hybrid search: an exact identifier."""
    bm25 = BM25(TOKENS)
    hits = bm25.search(tokenize("TS-999"), k=3)
    assert hits[0][0] == 1


def test_bm25_drops_documents_sharing_no_query_term():
    """Padding results with zero-scoring documents makes recall look better than it is."""
    bm25 = BM25(TOKENS)
    hits = bm25.search(tokenize("TS-999"), k=10)
    assert len(hits) == 1


def test_bm25_rare_terms_outweigh_common_ones():
    """'the' appears in three of four documents; 'retriever' in one."""
    bm25 = BM25(TOKENS)
    assert bm25.idf["retriever"] > bm25.idf["the"]


def test_bm25_repetition_has_diminishing_returns():
    """Doubling a term's frequency must not double the score, that is what k1 is for."""
    once = BM25([["alpha"], ["beta"]])
    twice = BM25([["alpha", "alpha"], ["beta"]])
    assert twice.score(["alpha"], 0) < 2 * once.score(["alpha"], 0)


def test_bm25_does_not_reward_padding_a_document():
    """Same single match, longer document: b must penalise the longer one."""
    short = BM25([["alpha"], ["beta", "beta", "beta", "beta"]])
    assert short.score(["alpha"], 0) > 0
    padded = BM25([["alpha"] + ["filler"] * 20, ["beta"]])
    assert padded.score(["alpha"], 0) < short.score(["alpha"], 0)


def test_bm25_idf_stays_positive_for_a_ubiquitous_term():
    """The textbook IDF goes negative here and lets a common term subtract from a score."""
    bm25 = BM25([["alpha"], ["alpha"], ["alpha"]])
    assert bm25.idf["alpha"] > 0


@pytest.mark.parametrize("b", [-0.1, 1.1])
def test_bm25_rejects_b_outside_the_unit_interval(b):
    with pytest.raises(ValueError):
        BM25(TOKENS, b=b)


def test_bm25_rejects_negative_k1():
    with pytest.raises(ValueError):
        BM25(TOKENS, k1=-1.0)


def test_bm25_handles_an_empty_corpus():
    assert BM25([]).search(["alpha"], k=5) == []


# --- the published RRF worked example ---------------------------------------------------


def test_rrf_reproduces_the_published_numbers():
    """Cormack et al. 2009 with k=60, as cited in rag/learning-pack.md section 6:
    rank 1 + rank 10 gives 1/61 + 1/70 = 0.0307; rank 3 in both gives 2/63 = 0.0317."""
    lucky = 1 / 61 + 1 / 70
    steady = 2 / 63
    assert round(lucky, 4) == 0.0307
    assert round(steady, 4) == 0.0317

    # doc 0 is rank 1 then rank 10; doc 1 is rank 3 in both lists
    list_a = [0, 90, 1, 91, 92, 93, 94, 95, 96, 97]
    list_b = [80, 81, 1, 82, 83, 84, 85, 86, 87, 0]
    fused = dict(rrf([list_a, list_b], k=60))

    assert fused[0] == pytest.approx(lucky)
    assert fused[1] == pytest.approx(steady)


def test_rrf_steady_agreement_beats_one_lucky_first_place():
    """The whole reason RRF is the default merge."""
    list_a = [0, 90, 1, 91, 92, 93, 94, 95, 96, 97]
    list_b = [80, 81, 1, 82, 83, 84, 85, 86, 87, 0]
    assert indices(rrf([list_a, list_b], k=60))[0] == 1


def test_rrf_needs_no_score_comparison():
    """Ranks only: a BM25 score of 8.2 and a cosine of 0.71 never have to be compared."""
    fused = rrf([[5, 6], [6, 5]], k=60)
    assert dict(fused)[5] == dict(fused)[6]


def test_rrf_ties_break_on_document_id_for_determinism():
    assert indices(rrf([[7, 3], [3, 7]], k=60)) == [3, 7]


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        rrf([[1, 2]], k=0)


def test_rrf_of_one_list_preserves_its_order():
    assert indices(rrf([[4, 9, 2]])) == [4, 9, 2]


# --- cosine -----------------------------------------------------------------------------


def test_cosine_of_an_identical_vector_is_one():
    hits = cosine_search([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], k=2)
    assert hits[0][0] == 0
    assert hits[0][1] == pytest.approx(1.0)


def test_cosine_ignores_magnitude():
    """Normalisation is done here, not assumed, otherwise cosine silently becomes a dot
    product and favours long documents."""
    hits = dict(cosine_search([1.0, 0.0], [[5.0, 0.0], [1.0, 0.0]], k=2))
    assert hits[0] == pytest.approx(hits[1])


def test_cosine_orders_by_angle():
    hits = cosine_search([1.0, 1.0], [[1.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], k=3)
    assert indices(hits) == [1, 0, 2]


def test_cosine_survives_a_zero_vector():
    hits = dict(cosine_search([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0]], k=2))
    assert hits[1] == pytest.approx(1.0)
    assert hits[0] == -1.0


def test_cosine_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_search([1.0, 0.0, 0.0], [[1.0, 0.0]], k=1)


def test_cosine_rejects_a_one_dimensional_matrix():
    with pytest.raises(ValueError):
        cosine_search([1.0], [1.0, 2.0], k=1)


# --- rerank -----------------------------------------------------------------------------


def _stub_scorer(query, candidates):
    """Deterministic stand-in for a cross-encoder: score by shared word count."""
    wanted = set(tokenize(query))
    return [float(len(wanted & set(tokenize(c)))) for c in candidates]


def test_rerank_reorders_by_the_injected_scorer():
    candidates = ["nothing relevant here", "retrieval augmented generation"]
    assert indices(rerank("retrieval generation", candidates, _stub_scorer)) == [1, 0]


def test_rerank_keeps_only_top_k():
    assert len(rerank("a b", ["a", "b", "a b"], _stub_scorer, top_k=2)) == 2


def test_rerank_of_nothing_is_nothing():
    assert rerank("q", [], _stub_scorer) == []


def test_rerank_rejects_a_scorer_that_returns_the_wrong_count():
    with pytest.raises(ValueError):
        rerank("q", ["a", "b"], lambda q, c: [1.0])
