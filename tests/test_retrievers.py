"""Covers the two retrievers that need no model download, plus fusion.

DenseRetriever and CrossEncoderReranker are exercised by scripts/run_eval.py
rather than here: a unit test that pulls 180MB of weights is an integration
test wearing the wrong label, and it makes CI depend on a model host being up.
"""

import pytest

from rag_reranker_lab.fusion import reciprocal_rank_fusion
from rag_reranker_lab.retrievers import (
    BM25Retriever,
    Document,
    Scored,
    TfidfRetriever,
    tokenize,
)

CORPUS = [
    Document("d1", "the cat sat on the mat in the warm kitchen"),
    Document("d2", "dogs bark loudly at the postman every morning"),
    Document("d3", "a kitten slept on a rug beside the stove"),
    Document("d4", "cat cat cat cat cat cat cat cat cat cat"),
    Document(
        "d5",
        "an unrelated passage about database indexes and query plans " * 8,
    ),
]


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, World! 42") == ["hello", "world", "42"]

    def test_drops_symbols_entirely(self):
        assert tokenize("--- *** ---") == []


class TestBM25:
    @pytest.fixture
    def retriever(self):
        r = BM25Retriever()
        r.index(CORPUS)
        return r

    def test_finds_the_obvious_match(self, retriever):
        assert retriever.search("cat mat kitchen", 3)[0].doc_id == "d1"

    def test_returns_at_most_k(self, retriever):
        assert len(retriever.search("cat", 2)) <= 2

    def test_documents_with_no_query_term_are_omitted(self, retriever):
        ids = [s.doc_id for s in retriever.search("postman", 5)]
        assert ids == ["d2"]

    def test_saturation_stops_repetition_from_dominating(self, retriever):
        """d4 is the word 'cat' ten times and nothing else. Term-frequency
        scoring would rank it first for any query containing 'cat'; BM25's k1
        saturation is what stops that."""
        ranked = retriever.search("cat sat mat kitchen", 5)
        assert ranked[0].doc_id == "d1"

    def test_length_normalisation_penalises_the_padded_document(self):
        """With b=0 the long passage is not penalised for its length. Turning
        normalisation on must lower its score relative to a short match."""
        long_query = "database indexes query plans"
        with_norm = BM25Retriever(b=0.75)
        without_norm = BM25Retriever(b=0.0)
        for r in (with_norm, without_norm):
            r.index(CORPUS)
        assert (
            with_norm.search(long_query, 1)[0].score
            < without_norm.search(long_query, 1)[0].score
        )

    def test_empty_index_returns_nothing(self):
        assert BM25Retriever().search("anything", 5) == []

    def test_query_with_no_known_terms_returns_nothing(self, retriever):
        assert retriever.search("zzzz qqqq", 5) == []


class TestTfidf:
    @pytest.fixture
    def retriever(self):
        r = TfidfRetriever()
        r.index(CORPUS)
        return r

    def test_finds_the_obvious_match(self, retriever):
        assert retriever.search("cat mat kitchen", 3)[0].doc_id == "d1"

    def test_scores_are_cosines_so_they_stay_within_one(self, retriever):
        assert all(0 < s.score <= 1.0000001 for s in retriever.search("cat", 5))

    def test_empty_index_returns_nothing(self):
        assert TfidfRetriever().search("anything", 5) == []


class TestFusion:
    def test_a_document_ranked_well_by_both_beats_one_loved_by_either(self):
        """The property RRF exists for. 'b' is second in both lists; 'a' and
        'c' are first in one and absent from the other."""
        run_a = [Scored("a", 9.0), Scored("b", 1.0)]
        run_b = [Scored("c", 9.0), Scored("b", 1.0)]
        assert reciprocal_rank_fusion([run_a, run_b], k=3)[0].doc_id == "b"

    def test_ignores_the_magnitude_of_the_scores(self):
        """Multiplying one run's scores by a thousand must change nothing,
        because only position is used."""
        run_a = [Scored("a", 1.0), Scored("b", 0.9)]
        run_b = [Scored("a", 1000.0), Scored("b", 900.0)]
        scaled = [s.doc_id for s in reciprocal_rank_fusion([run_a, run_b], 2)]
        plain = [s.doc_id for s in reciprocal_rank_fusion([run_a, run_a], 2)]
        assert scaled == plain

    def test_lower_smoothing_lets_a_single_confident_run_dominate(self):
        run_a = [Scored("a", 1.0)]
        run_b = [Scored("x", 1.0), Scored("y", 1.0), Scored("a", 1.0)]
        flat = reciprocal_rank_fusion([run_a, run_b], 3, smoothing=60)
        peaked = reciprocal_rank_fusion([run_a, run_b], 3, smoothing=0)
        assert flat[0].doc_id == "a"
        assert peaked[0].doc_id == "a"
        # With no smoothing the gap between rank 1 and rank 3 is far wider.
        assert (peaked[0].score - peaked[-1].score) > (flat[0].score - flat[-1].score)

    def test_respects_k(self):
        run = [Scored(f"d{i}", 1.0) for i in range(10)]
        assert len(reciprocal_rank_fusion([run], k=4)) == 4

    def test_no_runs_returns_nothing(self):
        assert reciprocal_rank_fusion([], k=5) == []
