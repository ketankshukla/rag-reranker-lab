# CHATLOG — rag-reranker-lab

> A record of the decisions that produced this repo: what was chosen, what was
> rejected, and what a measurement changed. Written from the session it was
> built in, not reconstructed afterwards.

---

## 7 Sep 2026 — the whole build

Built in one session with Claude Opus 5, directed through a plan → build →
measure → revise loop.

### The brief

Three repos were wanted that would legitimately justify listing PyTorch,
scikit-learn and HuggingFace on a resume — libraries that appeared nowhere in
60 existing repos, and had therefore been deliberately kept off the skills list.

The constraint set the shape of everything: **each project had to genuinely
need the library, not bolt it on.** An LLM app with a decorative `import torch`
would have been worse than not listing it, because a reader who catches one
unbacked claim starts doubting the rest.

For retrieval that constraint is easy to satisfy honestly — a bi-encoder and a
cross-encoder are the right tools, not decoration.

### Decision: extend an existing finding rather than start cold

An earlier post argued that deleting a fixed similarity threshold fixed a
silent retrieval failure. The obvious next question was whether the *rest* of
the received wisdom held up. That gave the repo a real question instead of a
demo.

### Decision: write BM25 and the metrics by hand

**Considered:** `rank_bm25`, `ir_measures`, `pytrec_eval`.

**Chose:** implement both.

Reasoning: the entire value of the repo is that the scoring is inspectable. A
result produced by a black box you imported is a result you cannot defend when
someone asks why fusion lost. Writing BM25 out also forced the `k1` and `b`
parameters into the open, and both got tests asserting what they do.

Cost: about 80 lines and the risk of getting it wrong. Mitigated by property
tests — a document that is one word repeated ten times must not outrank a
document that answers the query.

### Decision: RRF rather than a weighted score blend

BM25 scores are unbounded sums, TF-IDF cosines live in [0,1], dense cosines
cluster near the top. Blending requires normalising three distributions whose
shape changes per query, and **any constant chosen to do that is the same class
of mistake as the fixed similarity threshold this work came out of.**

RRF discards scores and uses position. A test asserts that multiplying one run's
scores by 1000 changes nothing.

### The corpus was built to make the experiment discriminating

60 passages written about agent infrastructure, with 28 queries deliberately
split:

- `lexical` — phrased in the corpus's own vocabulary
- `paraphrase` — phrased as a user would ask

**Hypothesis going in:** BM25 wins lexical, dense wins paraphrase, fusion
collects both.

### First run — the hypothesis was half wrong

```
bm25          0.679    tfidf 0.643    dense 0.839    fusion 0.768
```

Fusion scored **below dense alone**. And the split showed dense winning the
lexical queries too, 0.950 against BM25's 0.850.

There was no complementary failure for fusion to collect, because the dense
retriever was not weak anywhere the lexical ones were strong. RRF cannot know
one input is better; it averaged a good voter with two worse ones.

**Kept the finding rather than tuning until it agreed with the literature.**
The README leads with it.

### Revision: rerank two first stages, not one

The first version reranked only the fused list, giving `fusion + rerank = 0.821`.
That number is uninterpretable on its own — it cannot distinguish "the reranker
helped" from "the reranker partially undid the fusion damage".

Added a second arm reranking the strongest single retriever:

```
fusion + rerank  0.821
dense  + rerank  0.857   ← still ahead
```

So the reranker cannot fully recover what fusion cost. This was the single most
valuable revision in the build, and the general lesson went into the post: **a
pipeline stage measured against one predecessor is measuring the pair.**

### Decision: what the tests cover, and what they do not

Tests cover metrics, BM25, TF-IDF and fusion — everything that needs no model.
The dense retriever and reranker are exercised by `run_eval.py`.

A unit test that pulls 180 MB of weights is an integration test wearing the
wrong label, and it makes CI depend on a model host being reachable. Same
separation used elsewhere to make an agent loop's iteration cap testable
against a mocked model client.

Result: 36 tests, ~18 s, no network.

### Things that went wrong

- **Python 3.14 was a real risk.** Checked `pip install --dry-run torch` before
  designing anything around it. Torch 2.14.0 has a 3.14 wheel; it would have
  changed the plan if not.
- **`sentence-transformers` silently failed to install** on the first attempt
  and only surfaced as a `ModuleNotFoundError` mid-run. Reinstalled explicitly.

### Deliberately not done

- No chunking experiments — different confounds, different repo.
- No generation step — retrieval measured on retrieval metrics.
- No vector store — indexing 60 documents is not the problem being studied.
- No fine-tuning — both models used as published.

---

## Conventions for future entries

Each entry should record **what changed, what it was measured against, and what
was rejected.** An entry that only says what was added is a commit message and
belongs in git instead.
