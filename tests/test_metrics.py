import math

import pytest

from rag_reranker_lab.metrics import (
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecall:
    def test_finds_everything(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_finds_half(self):
        assert recall_at_k(["a", "z"], {"a", "b"}, 2) == 0.5

    def test_ignores_results_past_the_cutoff(self):
        # The relevant document is at rank 3, so recall@2 must not see it.
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_no_relevant_documents_is_zero_not_a_crash(self):
        assert recall_at_k(["a"], set(), 5) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_divides_by_k_not_by_result_count(self):
        # Only one result returned, but precision@4 still divides by 4 —
        # returning fewer results does not earn a better score.
        assert precision_at_k(["a"], {"a"}, 4) == 0.25

    def test_zero_k_is_zero_not_a_division_error(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestMRR:
    def test_first_position_scores_one(self):
        assert mrr_at_k(["a", "b"], {"a"}, 2) == 1.0

    def test_third_position_scores_a_third(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_only_the_first_hit_counts(self):
        # Two relevant documents, but MRR is about the first one only.
        assert mrr_at_k(["x", "a", "b"], {"a", "b"}, 3) == pytest.approx(0.5)

    def test_nothing_in_the_window_is_zero(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)

    def test_order_matters(self):
        """The property that makes nDCG worth computing at all: two rankings
        containing the same documents score differently."""
        good = ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3)
        bad = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
        assert good > bad

    def test_matches_hand_computed_value(self):
        # One relevant document at rank 2: DCG = 1/log2(3), ideal = 1/log2(2).
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(expected)

    def test_empty_relevant_set_is_zero(self):
        assert ndcg_at_k(["a"], set(), 3) == 0.0


class TestEvaluateRun:
    def test_averages_over_queries(self):
        run = {"q1": ["a"], "q2": ["z"]}
        qrels = {"q1": {"a"}, "q2": {"b"}}
        out = evaluate_run(run, qrels, k=1)
        assert out["recall"] == pytest.approx(0.5)
        assert out["mrr"] == pytest.approx(0.5)

    def test_a_query_the_retriever_missed_entirely_scores_zero(self):
        """A retriever that returns nothing for a query must be penalised, not
        skipped — otherwise failing to answer improves the average."""
        out = evaluate_run({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, k=1)
        assert out["recall"] == pytest.approx(0.5)

    def test_macro_average_gives_each_query_one_vote(self):
        # q1 has four judgements, q2 has one. A micro average would let q1
        # dominate; macro must not.
        run = {"q1": ["a", "b", "c", "d"], "q2": ["z"]}
        qrels = {"q1": {"a", "b", "c", "d"}, "q2": {"y"}}
        assert evaluate_run(run, qrels, k=4)["recall"] == pytest.approx(0.5)

    def test_no_queries_is_zero_not_a_crash(self):
        assert evaluate_run({}, {}, k=5)["ndcg"] == 0.0
