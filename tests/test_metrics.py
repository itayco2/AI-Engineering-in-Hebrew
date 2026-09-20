"""Metrics: exact arithmetic, checked against hand-computed values."""

import math

import pytest

from aihe.metrics import (
    failed_retrieval_rate,
    hit_rate,
    mean_recall_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summary,
)

RANKED = [10, 11, 12, 13, 14]
RELEVANT = {11, 14, 99}  # 99 was never retrieved


def test_recall_counts_only_what_is_in_the_window():
    assert recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 3)


def test_recall_cannot_exceed_one_when_duplicates_are_retrieved():
    assert recall_at_k([11, 11, 11], {11}, 3) == 1.0


def test_precision_divides_by_k_not_by_hits():
    """Dividing by the number returned would flatter a retriever that returns three."""
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(1 / 2)


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank([10, 12], RELEVANT) == 0.0


def test_ndcg_is_one_when_ordering_is_ideal():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)


def test_ndcg_punishes_a_late_hit():
    early = ndcg_at_k([1, 9, 9], {1}, 3)
    late = ndcg_at_k([9, 9, 1], {1}, 3)
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(1 / math.log2(4))
    assert late < early


def test_ndcg_matches_a_hand_computed_value():
    """Relevant at ranks 1 and 3: DCG = 1/log2(2) + 1/log2(4) = 1.5;
    ideal for two relevant documents = 1/log2(2) + 1/log2(3)."""
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([1, 9, 2], {1, 2}, 3) == pytest.approx(1.5 / ideal)


def test_hit_rate_is_binary():
    assert hit_rate(RANKED, RELEVANT, 2) == 1.0
    assert hit_rate([10, 12], RELEVANT, 2) == 0.0


def test_no_relevant_documents_scores_zero_everywhere():
    assert recall_at_k(RANKED, set(), 5) == 0.0
    assert precision_at_k(RANKED, set(), 5) == 0.0
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_is_rejected(k):
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, k)


def test_failed_retrieval_rate_is_one_minus_recall():
    """The form Anthropic reported the contextual-retrieval staircase in."""
    queries = [([1, 2], {1, 2}), ([3, 9], {3, 4})]
    assert mean_recall_at_k(queries, 2) == pytest.approx(0.75)
    assert failed_retrieval_rate(queries, 2) == pytest.approx(0.25)


def test_a_perfect_retriever_fails_nothing():
    assert failed_retrieval_rate([([1], {1})], 1) == pytest.approx(0.0)


def test_mrr_averages_across_queries():
    assert mrr([([1, 2], {1}), ([9, 2], {2})]) == pytest.approx((1.0 + 0.5) / 2)


def test_summary_reports_every_metric_with_k_in_the_name():
    keys = set(summary([([1, 2], {1})], k=2))
    assert keys == {"recall@2", "precision@2", "nDCG@2", "MRR", "failed@2"}


def test_aggregates_over_an_empty_query_set_do_not_divide_by_zero():
    assert mean_recall_at_k([], 5) == 0.0
    assert mrr([]) == 0.0
