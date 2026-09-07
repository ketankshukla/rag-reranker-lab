"""The harness: load the data, run each configuration, print one table.

Nothing here is clever. It exists so that every claim in the README is a
command anyone can rerun, and so that adding a retriever means adding a row
rather than editing a script.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .fusion import reciprocal_rank_fusion
from .metrics import evaluate_run
from .retrievers import Document, Retriever, Scored

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str
    relevant: tuple[str, ...]
    kind: str


def load_corpus(path: Path | None = None) -> list[Document]:
    path = path or DATA / "corpus.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [
            Document(row["doc_id"], row["text"])
            for row in (json.loads(line) for line in fh if line.strip())
        ]


def load_queries(path: Path | None = None) -> list[Query]:
    path = path or DATA / "qrels.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [
            Query(r["query_id"], r["query"], tuple(r["relevant"]), r.get("kind", ""))
            for r in (json.loads(line) for line in fh if line.strip())
        ]


def run_retriever(
    retriever: Retriever, queries: Iterable[Query], k: int
) -> dict[str, list[str]]:
    return {q.query_id: [s.doc_id for s in retriever.search(q.query, k)] for q in queries}


def run_fusion(
    runs: list[dict[str, list[Scored]]], queries: Iterable[Query], k: int
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for q in queries:
        fused = reciprocal_rank_fusion([r[q.query_id] for r in runs], k)
        out[q.query_id] = [s.doc_id for s in fused]
    return out


def score(
    run: dict[str, list[str]], queries: Iterable[Query], k: int
) -> dict[str, float]:
    qrels = {q.query_id: q.relevant for q in queries}
    return evaluate_run(run, qrels, k)


def format_table(rows: list[tuple[str, dict[str, float]]], k: int) -> str:
    header = f"| {'configuration':<26} | recall@{k} | mrr@{k} | ndcg@{k} | precision@{k} |"
    rule = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, rule]
    for name, m in rows:
        lines.append(
            f"| {name:<26} |    {m['recall']:.3f} |   {m['mrr']:.3f} |   "
            f"{m['ndcg']:.3f} |        {m['precision']:.3f} |"
        )
    return "\n".join(lines)


def split_by_kind(
    run: dict[str, list[str]], queries: list[Query], k: int
) -> dict[str, dict[str, float]]:
    """Score lexical and paraphrase queries separately.

    This split is the whole reason the evaluation set is hand-labelled. An
    aggregate number hides the interesting result, which is that the two
    families of query are won by different retrievers.
    """
    out = {}
    for kind in sorted({q.kind for q in queries if q.kind}):
        subset = [q for q in queries if q.kind == kind]
        out[kind] = score(run, subset, k)
    return out
