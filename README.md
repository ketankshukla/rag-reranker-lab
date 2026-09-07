# 🔬 RAG Reranker Lab

**A retrieval experiment that came out the wrong way round.**

Everyone tells you to use hybrid retrieval. On this corpus, fusing lexical and dense retrieval made results **worse than dense alone** — and a cross-encoder reranker costing 10× the latency bought almost nothing back.

This repo is the harness that measured it: three retrievers, rank fusion, a cross-encoder second stage, four IR metrics written out by hand, and 28 hand-labelled queries over 60 documents. Every number below is printed by `scripts/run_eval.py` and can be reproduced by running it.

📊 **Metrics from scratch:** [`src/rag_reranker_lab/metrics.py`](src/rag_reranker_lab/metrics.py)
🔍 **BM25 from scratch:** [`src/rag_reranker_lab/retrievers.py`](src/rag_reranker_lab/retrievers.py)
📝 **Related write-up:** [Deleting the similarity threshold is what made retrieval work](https://ketanshukla.dev/blog/deleting-the-similarity-threshold)

---

## The headline

60 documents · 28 labelled queries · k=5 · CPU only

| configuration | recall@5 | mrr@5 | ndcg@5 | precision@5 | latency |
|---|---|---|---|---|---|
| BM25 | 0.679 | 0.632 | 0.589 | 0.214 | **0.2 ms** |
| TF-IDF (scikit-learn) | 0.643 | 0.634 | 0.587 | 0.200 | 8.8 ms |
| **Dense (MiniLM)** | **0.839** | **0.832** | **0.778** | 0.279 | 39.9 ms |
| Fusion (RRF) | 0.768 | 0.759 | 0.712 | 0.250 | 49 ms |
| Fusion + rerank | 0.821 | 0.817 | 0.751 | 0.271 | 516 ms |
| Dense + rerank | 0.857 | 0.827 | 0.774 | **0.286** | 397 ms |

Three things in that table are worth more than the winning row.

### 1. Hybrid fusion lost to its own best input

Dense retrieval scores **0.839**. Fusing it with BM25 and TF-IDF drops it to **0.768** — an 8.5% relative loss.

Reciprocal Rank Fusion has no way to know that one of its inputs is much better than the other two. It rewards documents that several retrievers rank highly, which is exactly right when retrievers have *complementary* failures, and exactly wrong when two of them are simply worse. Here the lexical retrievers outvoted the good one.

**Fusion is not free.** It helps when your retrievers fail in different places. Measure that before assuming it.

### 2. Dense won the lexical queries too

The evaluation set labels each query `lexical` (phrased in the corpus's own vocabulary — `security definer bypasses row level security`) or `paraphrase` (phrased as a user would — `can another customer see my uploaded files`).

| | lexical | paraphrase |
|---|---|---|
| BM25 | 0.850 | 0.583 |
| TF-IDF | 0.750 | 0.583 |
| **Dense** | **0.950** | **0.778** |
| Fusion (RRF) | 0.900 | 0.694 |

The premise going in was that BM25 would win exact-term queries and dense would win paraphrases, and fusion would collect both. **Half of that was wrong.** Dense beat BM25 even where the query used the corpus's exact words, because a modern sentence embedding has no trouble with exact matches — it just also handles the other case.

The gap that *did* survive is dense's own: 0.950 lexical against 0.778 paraphrase. Paraphrased questions are still the hard ones for everybody.

### 3. The reranker was a rounding error at 10× the cost

Dense alone: **0.839** recall, **0.778** nDCG, 40 ms.
Dense + cross-encoder: **0.857** recall, **0.774** nDCG, 397 ms.

Recall up 1.8 points. nDCG — the only metric here sensitive to the *order* of several relevant results, and therefore the one a reranker is supposed to move — went **down**.

A cross-encoder reads the query and document together, so it genuinely knows more than a bi-encoder does. It is also a 22M-parameter model trained on MS MARCO web passages, being asked about MCP internals. On a domain it has never seen, "knows more" and "knows better" are different claims.

Where reranking clearly *did* help: recovering from the fusion damage (0.768 → 0.821). Which is an argument against the fusion, not for the reranker.

---

## What this repo is not

It is not a benchmark, and 60 documents is not a corpus. It is a harness small enough that every result is reproducible on a laptop in under two minutes, on a domain narrow enough that the labels are trustworthy.

The conclusion that transfers is the method, not the numbers: **run this against your own documents before choosing an architecture.** Hybrid-plus-reranker is the default recommendation everywhere, and here it was third best.

---

## Running it

```bash
pip install -r requirements.txt

python scripts/run_eval.py            # everything (downloads ~180 MB of weights)
python scripts/run_eval.py --lexical  # BM25 + TF-IDF only, no downloads
python scripts/run_eval.py -k 3 --pool 40
```

```bash
pytest -q     # 36 tests, no network, no model downloads
```

---

## How it is put together

```
src/rag_reranker_lab/
├─ metrics.py      recall / precision / MRR / nDCG, written out and unit tested
├─ retrievers.py   BM25 (from scratch), TF-IDF (sklearn), dense (PyTorch)
├─ fusion.py       reciprocal rank fusion
├─ rerank.py       cross-encoder second stage
└─ evaluate.py     the harness
data/
├─ corpus.jsonl    60 passages
└─ qrels.jsonl     28 queries with judgements, tagged lexical or paraphrase
```

### BM25 is written out rather than imported

Two parameters, both worth understanding rather than accepting:

- **`k1 = 1.5`** — how quickly extra occurrences of a term stop helping. There is a test in the suite that proves it: a document consisting of the word "cat" ten times does not outrank a document that actually answers the query. Raw term-frequency scoring gets that wrong.
- **`b = 0.75`** — how hard to penalise long documents. There is a test for that too, comparing `b=0.75` against `b=0`. With normalisation off, long chunks win by accident, which is the classic chunking bug.

The IDF term is floored at zero. Without the floor, a term appearing in more than half the corpus gets a *negative* weight and a document is penalised for containing it — defensible in theory, surprising in production.

### Why RRF instead of adding the scores

BM25 scores are unbounded sums. TF-IDF cosines live in [0, 1]. Dense cosines cluster in a narrow band near the top. Blending them requires normalising three distributions whose shape changes per query, and any constant you pick to do that is the same class of mistake as a fixed similarity threshold.

RRF throws the scores away and uses only position. There is a test asserting exactly that: multiplying one run's scores by 1000 changes nothing.

### There is no similarity threshold anywhere

Retrievers return the top *n*, never "everything above 0.7". A fixed cutoff is not scale-invariant across queries, and when it is too strict it fails **silently** — the answer never arrives and nothing logs a reason. That is the subject of [the write-up](https://ketanshukla.dev/blog/deleting-the-similarity-threshold) this repo grew out of.

### Metrics are macro-averaged

Each query counts once regardless of how many judgements it carries, so one heavily annotated question cannot outvote seven lightly annotated ones. There is a test for that too.

---

## Tests

36 tests, no network, no model downloads — they cover the metrics, BM25's scoring properties, TF-IDF, and fusion. The dense retriever and the reranker are exercised by `run_eval.py` instead: a unit test that pulls 180 MB of weights is an integration test wearing the wrong label, and it makes CI depend on a model host being reachable.

The tests worth reading are the ones asserting a *property* rather than a value — that nDCG is order-sensitive where recall is not, that BM25 saturation works, that RRF ignores score magnitude, that a query the retriever missed entirely scores zero rather than being skipped.

---

## Stack

Python · PyTorch (CPU) · sentence-transformers · scikit-learn · NumPy · pytest

MIT licensed. Built by [Ketan Shukla](https://ketanshukla.dev).
