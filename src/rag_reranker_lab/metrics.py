"""Information-retrieval metrics, written out rather than imported.

Every number this repo reports comes from these four functions, so they are
pure, dependency-light and the most heavily tested code here. A retrieval
experiment whose metrics you cannot inspect is an experiment you cannot trust.

Conventions used throughout:

  ranked    a list of document ids, best first, as a retriever returned them
  relevant  the set of document ids a human judged relevant for that query
  k         cut-off; metrics look only at the first k results
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    Answers "did we find the material at all?". This is the metric that matters
    for RAG, because a document the retriever never returned cannot be read by
    the model no matter how good the prompt is.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    found = sum(1 for doc in ranked[:k] if doc in relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k that are relevant.

    Answers "how much noise am I paying for?" — directly a token-cost question
    when the top k gets pasted into a prompt.
    """
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    return sum(1 for doc in ranked[:k] if doc in relevant) / k


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Reciprocal of the rank of the first relevant document, 0 if none in k.

    Rewards putting one right answer at the top. Useful when a single good
    chunk is enough, which is most single-fact questions.
    """
    relevant = set(relevant)
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    The only one of the four that is sensitive to the *order* of several
    relevant results, so it is the metric that actually distinguishes a
    reranker from the retriever underneath it.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(ranked[:k], start=1)
        if doc in relevant
    )
    # Ideal ranking: every relevant document packed into the top positions.
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate_run(
    run: dict[str, Sequence[str]],
    qrels: dict[str, Iterable[str]],
    k: int,
) -> dict[str, float]:
    """Average every metric over the queries, macro style.

    Macro rather than micro deliberately: each query counts once regardless of
    how many relevant documents it has, so one query with eight judgements
    cannot outvote seven queries with one each.
    """
    if not qrels:
        return {name: 0.0 for name in METRICS}

    totals = {name: 0.0 for name in METRICS}
    for query_id, relevant in qrels.items():
        ranked = run.get(query_id, [])
        for name, fn in METRICS.items():
            totals[name] += fn(ranked, relevant, k)

    n = len(qrels)
    return {name: total / n for name, total in totals.items()}
