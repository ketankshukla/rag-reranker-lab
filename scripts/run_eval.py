"""Run every configuration and print the tables the README quotes.

    python scripts/run_eval.py            # everything, downloads two models
    python scripts/run_eval.py --lexical  # BM25 + TF-IDF only, no downloads

Nothing is cached between runs and nothing is written to the repo: the numbers
in the README came from this script and can be reproduced by running it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_reranker_lab.evaluate import (  # noqa: E402
    format_table,
    load_corpus,
    load_queries,
    run_fusion,
    score,
    split_by_kind,
)
from rag_reranker_lab.retrievers import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    TfidfRetriever,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5, help="cut-off for every metric")
    ap.add_argument("--pool", type=int, default=25, help="candidates handed to the reranker")
    ap.add_argument("--lexical", action="store_true", help="skip anything needing a download")
    args = ap.parse_args()

    corpus = load_corpus()
    queries = load_queries()
    by_id = {d.doc_id: d for d in corpus}
    print(f"corpus {len(corpus)} documents · {len(queries)} labelled queries · k={args.k}\n")

    rows = []
    raw: dict[str, dict[str, list]] = {}

    retrievers = [BM25Retriever(), TfidfRetriever()]
    if not args.lexical:
        retrievers.append(DenseRetriever())

    for r in retrievers:
        started = time.perf_counter()
        r.index(corpus)
        indexed = time.perf_counter() - started

        started = time.perf_counter()
        raw[r.name] = {q.query_id: r.search(q.query, args.pool) for q in queries}
        searched = time.perf_counter() - started

        run = {qid: [s.doc_id for s in hits] for qid, hits in raw[r.name].items()}
        rows.append((r.name, score(run, queries, args.k)))
        print(
            f"  {r.name:<10} indexed in {indexed:6.2f}s · "
            f"{searched / len(queries) * 1000:6.1f} ms/query"
        )

    # Fusion over whatever ran.
    fused = run_fusion([raw[r.name] for r in retrievers], queries, args.pool)
    rows.append(("fusion (rrf)", score(fused, queries, args.k)))

    # Rerank the fused candidates.
    if not args.lexical:
        from rag_reranker_lab.rerank import CrossEncoderReranker
        from rag_reranker_lab.retrievers import Scored

        reranker = CrossEncoderReranker(candidates=args.pool)

        # Rerank two different first stages. Reranking the fused list alone
        # cannot tell you whether the reranker helped or merely undid damage
        # the fusion did, so the strongest single retriever gets the same
        # treatment and the two are compared directly.
        stages = {
            "fusion + rerank": fused,
            "dense + rerank": {
                qid: [s.doc_id for s in hits] for qid, hits in raw["dense"].items()
            },
        }
        for label, first_stage in stages.items():
            started = time.perf_counter()
            reranked = {}
            for q in queries:
                pool = [Scored(doc_id, 0.0) for doc_id in first_stage[q.query_id]]
                reranked[q.query_id] = [
                    s.doc_id for s in reranker.rerank(q.query, pool, by_id, args.pool)
                ]
            elapsed = time.perf_counter() - started
            rows.append((label, score(reranked, queries, args.k)))
            print(
                f"  {label:<16} {elapsed / len(queries) * 1000:6.1f} ms/query "
                f"over {args.pool} candidates"
            )

    print("\n" + format_table(rows, args.k))

    # The split is the interesting part: the two families of query are won by
    # different retrievers, and the aggregate hides that.
    print("\nBy query kind (recall@%d):\n" % args.k)
    header = f"| {'configuration':<26} | lexical | paraphrase |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    named = {"bm25": raw.get("bm25"), "tfidf": raw.get("tfidf"), "dense": raw.get("dense")}
    for name, hits in named.items():
        if hits is None:
            continue
        run = {qid: [s.doc_id for s in v] for qid, v in hits.items()}
        parts = split_by_kind(run, queries, args.k)
        print(
            f"| {name:<26} |   {parts.get('lexical', {}).get('recall', 0):.3f} "
            f"|      {parts.get('paraphrase', {}).get('recall', 0):.3f} |"
        )
    parts = split_by_kind(fused, queries, args.k)
    print(
        f"| {'fusion (rrf)':<26} |   {parts.get('lexical', {}).get('recall', 0):.3f} "
        f"|      {parts.get('paraphrase', {}).get('recall', 0):.3f} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
