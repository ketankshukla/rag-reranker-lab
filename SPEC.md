# SPEC — rag-reranker-lab

> A specification, not a tutorial. It says what this repo is for, what it
> deliberately does not do, and what must stay true for a change to be correct.

**Status:** complete for its purpose · **Version:** 0.1.0 · **Last verified:** 7 Sep 2026

---

## 1. Purpose

Answer one question with evidence rather than opinion:

> **For a given corpus, which retrieval configuration should you actually ship?**

The industry default is hybrid retrieval (lexical + dense, fused) followed by a
cross-encoder reranker. This repo exists to make that recommendation testable on
your own documents, because on the corpus here it finished **third of six**.

The deliverable is not a library to depend on. It is a **harness you fork,
repoint at your own corpus, and rerun.**

## 2. Scope

### In scope
- Three retrievers behind one interface: BM25, TF-IDF, dense bi-encoder
- Rank fusion (Reciprocal Rank Fusion)
- Cross-encoder reranking as a second stage
- Four IR metrics: recall, precision, MRR, nDCG — all `@k`
- A hand-labelled evaluation set with a query-kind split
- A single script that prints every configuration as one table

### Explicitly out of scope
- **Chunking.** The corpus arrives pre-chunked. Chunk-size experiments are a
  different repo with a different set of confounds.
- **Generation.** Retrieval is measured on retrieval metrics. Whether an LLM
  then writes a good answer is a separate question and would contaminate this one.
- **A vector database.** Everything is in memory. Indexing 60 documents is not
  the problem being studied, and adding a store would obscure the measurements.
- **Being a benchmark.** 60 documents is a harness, not a benchmark. Any
  conclusion here is about *this* corpus.
- **Training.** No model is fine-tuned. Both models are used as published.

## 3. Architecture

```
data/corpus.jsonl ──┐
                    ├─► Retriever.index()  ──► Retriever.search(query, k) ──┐
data/qrels.jsonl ───┘         (3 impls)                                     │
                                                                            ▼
                                              fusion.reciprocal_rank_fusion()
                                                                            │
                                              rerank.CrossEncoderReranker ◄──┤
                                                                            ▼
                                                       metrics.evaluate_run()
                                                                            ▼
                                                        one table on stdout
```

| module | responsibility | dependencies |
|---|---|---|
| `metrics.py` | the four IR metrics | none beyond stdlib |
| `retrievers.py` | BM25, TF-IDF, dense | numpy; sklearn and torch lazily |
| `fusion.py` | rank fusion | the `Scored` type only |
| `rerank.py` | cross-encoder second stage | sentence-transformers, lazily |
| `evaluate.py` | loading, running, scoring, formatting | the above |

**Dependency rule:** `metrics.py` and `fusion.py` import nothing heavy. Anything
requiring torch or sklearn is imported *inside* the function that needs it, so
the modules under test load instantly and CI needs neither.

## 4. Interfaces

### Retriever protocol

Any retriever satisfying this can be dropped into the harness:

```python
class Retriever(Protocol):
    name: str
    def index(self, documents: list[Document]) -> None: ...
    def search(self, query: str, k: int) -> list[Scored]: ...
```

`Document(doc_id: str, text: str)` · `Scored(doc_id: str, score: float)`

### Data formats

`data/corpus.jsonl` — one document per line:
```json
{"doc_id": "d01", "text": "..."}
```

`data/qrels.jsonl` — one query per line:
```json
{"query_id": "q01", "query": "...", "relevant": ["d01", "d02"], "kind": "paraphrase"}
```

`kind` is `lexical` (phrased in the corpus's vocabulary) or `paraphrase`
(phrased as a user would). This split is required, not optional — it is the
column that revealed dense retrieval also winning the lexical queries.

### CLI

```
python scripts/run_eval.py [-k N] [--pool N] [--lexical]
```

## 5. Invariants

These must hold. A change that breaks one is a bug, not a tradeoff.

1. **No similarity threshold anywhere.** Retrievers return the top *n* by rank.
   A fixed cutoff is not scale-invariant across queries and fails silently when
   too strict. Callers wanting a floor apply it to scores they have seen.
2. **Metrics are macro-averaged.** Each query counts once regardless of how many
   judgements it carries.
3. **A query the retriever missed entirely scores zero**, never skipped.
   Otherwise failing to answer improves the average.
4. **Fusion uses rank, never score.** Multiplying one run's scores by 1000 must
   not change the fused order.
5. **BM25 saturates term frequency and normalises length.** A document that is
   one word repeated must not outrank a document that answers the query.
6. **The test suite performs no network I/O and downloads no model.**
7. **Every number in the README is produced by `scripts/run_eval.py`.** No figure
   is written by hand.

## 6. Verification

| | |
|---|---|
| tests | 36, `pytest -q`, ~18 s, no network |
| what they cover | metrics, BM25 scoring properties, TF-IDF, fusion |
| what they do not | dense retriever, cross-encoder — both need a 90 MB download |
| CI | GitHub Actions, `numpy scikit-learn pytest` only |
| reproducing the README | `python scripts/run_eval.py` |

Tests assert *properties* rather than values where possible: that nDCG is
order-sensitive where recall is not, that BM25's `k1` saturation works, that RRF
ignores score magnitude, that `b=0.75` penalises a padded document relative to
`b=0`.

## 7. Results this repo produced

60 documents · 28 queries · k=5 · CPU

| configuration | recall@5 | mrr@5 | ndcg@5 | latency |
|---|---|---|---|---|
| BM25 | 0.679 | 0.632 | 0.589 | 0.2 ms |
| TF-IDF | 0.643 | 0.634 | 0.587 | 8.8 ms |
| **Dense (MiniLM)** | **0.839** | **0.832** | **0.778** | 39.9 ms |
| Fusion (RRF) | 0.768 | 0.759 | 0.712 | 49 ms |
| Fusion + rerank | 0.821 | 0.817 | 0.751 | 516 ms |
| Dense + rerank | 0.857 | 0.827 | 0.774 | 397 ms |

**Findings:** fusion scored below its own best input; dense won the lexical
queries too (0.950 vs BM25's 0.850); the reranker moved recall 1.8 points while
nDCG fell.

## 8. Known limitations

- **60 documents.** Small enough that a single query's judgement moves a metric
  by ~3.6%. Directional, not precise.
- **One domain.** Prose about agent infrastructure. Legal text, product SKUs or
  code identifiers could invert the lexical-vs-dense result.
- **Binary relevance.** A document is relevant or not; there are no graded
  judgements, so nDCG is less discriminating than it would otherwise be.
- **Single annotator.** No inter-annotator agreement, so the labels carry my own
  reading of what each query means.
- **Reranker domain mismatch.** `ms-marco-MiniLM-L-6-v2` is trained on web
  search passages. Its weak showing may be domain, not a general result.

## 9. Extending it

To add a retriever: implement the `Retriever` protocol, append it to the list in
`scripts/run_eval.py`. Nothing else changes.

To use your own corpus: replace both JSONL files. Aim for 25+ queries and tag
each `lexical` or `paraphrase`.

## 10. Related

- Post: [Hybrid retrieval made my results worse](https://ketanshukla.dev/blog/hybrid-retrieval-made-it-worse)
- Predecessor: [Deleting the similarity threshold is what made retrieval work](https://ketanshukla.dev/blog/deleting-the-similarity-threshold)
- Decision history: [CHATLOG.md](CHATLOG.md)
