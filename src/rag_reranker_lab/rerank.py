"""Cross-encoder reranking: the second pass that reads the query and the
document together.

Why a second model at all, when the bi-encoder already scored everything:

  A bi-encoder embeds the query and the document *separately* and compares the
  two vectors. That is what makes it fast enough to search a corpus — you embed
  the documents once, ahead of time. It is also its ceiling: the document's
  vector was computed without knowing what would be asked of it.

  A cross-encoder feeds query and document through the model together, so
  attention runs across both. Far more accurate, and far too slow to run over a
  corpus — a full pass over 10,000 documents is 10,000 forward passes.

So the two compose: retrieve wide and cheap, rerank narrow and expensive. The
only real design decision is how wide, and that is `candidates` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .retrievers import Document, Scored


@dataclass
class CrossEncoderReranker:
    """Rerank a candidate list with a cross-encoder.

    `candidates` is the knob that matters. Reranking is linear in it, so
    doubling the candidate pool doubles the latency; but a document the first
    stage never returned can never be recovered by the reranker. The harness
    reports both, so the tradeoff is a number rather than an opinion.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidates: int = 25
    name: str = "rerank"

    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        ranked: list[Scored],
        corpus: dict[str, Document],
        k: int,
    ) -> list[Scored]:
        pool = ranked[: self.candidates]
        if not pool:
            return []

        model = self._load()
        pairs = [(query, corpus[s.doc_id].text) for s in pool if s.doc_id in corpus]
        ids = [s.doc_id for s in pool if s.doc_id in corpus]
        if not pairs:
            return ranked[:k]

        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(ids)), key=lambda i: -float(scores[i]))
        return [Scored(ids[i], float(scores[i])) for i in order[:k]]
