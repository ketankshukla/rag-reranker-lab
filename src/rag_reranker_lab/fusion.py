"""Reciprocal Rank Fusion — combine rankings without comparing their scores.

The reason this exists rather than a weighted score blend: BM25 scores are
unbounded sums, TF-IDF cosines live in [0, 1], and dense cosines cluster in a
narrow band near the top. Adding those together requires normalising three
distributions that change shape per query, and any constant you pick to do it
is the same class of mistake as a fixed similarity threshold.

RRF sidesteps the problem by throwing the scores away and using only position.
"""

from __future__ import annotations

from collections.abc import Sequence

from .retrievers import Scored


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[Scored]],
    k: int,
    smoothing: int = 60,
) -> list[Scored]:
    """Fuse several ranked lists into one.

    Each list contributes 1 / (smoothing + rank) to every document it ranked.

    `smoothing` (the constant usually written as k in the literature, renamed
    here because this function already has a k) flattens the curve near the
    top. At 60 — the value from the original TREC work — the difference
    between rank 1 and rank 2 is small, so a document has to be ranked well by
    *several* retrievers to beat one that a single retriever loves. Lower it
    and one confident retriever dominates the fusion.
    """
    totals: dict[str, float] = {}

    for run in runs:
        for rank, scored in enumerate(run, start=1):
            totals[scored.doc_id] = totals.get(scored.doc_id, 0.0) + 1.0 / (
                smoothing + rank
            )

    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Scored(doc_id, score) for doc_id, score in ranked[:k]]
