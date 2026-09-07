"""Three retrievers with one interface, so they can be swapped in the harness.

  TfidfRetriever  scikit-learn, sparse lexical. Fast, exact-word matching.
  BM25Retriever   written out here rather than imported, because the whole
                  point of this repo is that the scoring is inspectable.
  DenseRetriever  sentence-transformers over PyTorch, semantic matching.

Only DenseRetriever needs a model download; the other two run anywhere, which
is why the test suite covers them and skips the third.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Scored:
    doc_id: str
    score: float


class Retriever(Protocol):
    name: str

    def index(self, documents: list[Document]) -> None: ...

    def search(self, query: str, k: int) -> list[Scored]: ...


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Deliberately boring and shared by both
    lexical retrievers, so a difference between them is a scoring difference
    rather than a tokenisation difference."""
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------- #
#  BM25
# --------------------------------------------------------------------------- #


@dataclass
class BM25Retriever:
    """Okapi BM25, implemented directly.

    Two parameters, and both are worth understanding rather than accepting:

      k1  how quickly extra occurrences of a term stop helping. At k1=1.5 the
          tenth occurrence of a word adds far less than the second — which is
          the behaviour raw term-frequency scoring gets wrong.
      b   how hard to punish long documents. b=0.75 partially normalises by
          length; b=0 ignores length entirely and lets long chunks win by
          accident, which is the classic chunking bug.
    """

    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"

    _docs: list[Document] = field(default_factory=list, repr=False)
    _tf: list[Counter[str]] = field(default_factory=list, repr=False)
    _df: Counter[str] = field(default_factory=Counter, repr=False)
    _lengths: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _avg_len: float = 0.0

    def index(self, documents: list[Document]) -> None:
        self._docs = list(documents)
        self._tf = []
        self._df = Counter()

        lengths = []
        for doc in self._docs:
            tokens = tokenize(doc.text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._df.update(counts.keys())
            lengths.append(len(tokens))

        self._lengths = np.array(lengths, dtype=float)
        self._avg_len = float(self._lengths.mean()) if lengths else 0.0

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at zero.

        Without the floor, a term appearing in more than half the corpus gets a
        negative weight and a document is penalised for containing it — which
        is defensible in theory and surprising in practice.
        """
        n = len(self._docs)
        df = self._df.get(term, 0)
        return max(0.0, np.log(1 + (n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, k: int) -> list[Scored]:
        if not self._docs:
            return []

        terms = tokenize(query)
        scores = np.zeros(len(self._docs))

        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, counts in enumerate(self._tf):
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[i] / (self._avg_len or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)

        return self._top(scores, k)

    def _top(self, scores: np.ndarray, k: int) -> list[Scored]:
        order = np.argsort(-scores)[:k]
        return [
            Scored(self._docs[i].doc_id, float(scores[i]))
            for i in order
            if scores[i] > 0
        ]


# --------------------------------------------------------------------------- #
#  TF-IDF (scikit-learn)
# --------------------------------------------------------------------------- #


@dataclass
class TfidfRetriever:
    """Cosine similarity over scikit-learn's TF-IDF, as the lexical baseline.

    Included alongside BM25 because they disagree more often than people
    expect, and the disagreement is the interesting part: TF-IDF has no
    saturation term, so repetition keeps paying.
    """

    name: str = "tfidf"
    ngram_range: tuple[int, int] = (1, 2)

    _docs: list[Document] = field(default_factory=list, repr=False)
    _matrix: object = field(default=None, repr=False)
    _vectorizer: object = field(default=None, repr=False)

    def index(self, documents: list[Document]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._docs = list(documents)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(d.text for d in self._docs)

    def search(self, query: str, k: int) -> list[Scored]:
        if not self._docs:
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        vec = self._vectorizer.transform([query])
        scores = cosine_similarity(vec, self._matrix)[0]
        order = np.argsort(-scores)[:k]
        return [
            Scored(self._docs[i].doc_id, float(scores[i]))
            for i in order
            if scores[i] > 0
        ]


# --------------------------------------------------------------------------- #
#  Dense (sentence-transformers / PyTorch)
# --------------------------------------------------------------------------- #


@dataclass
class DenseRetriever:
    """Bi-encoder retrieval: embed once at index time, embed the query, dot.

    No similarity threshold anywhere. Scores are compared to each other and the
    top n is returned — a fixed cutoff is not scale-invariant across queries,
    and cutting below it fails silently.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "dense"

    _docs: list[Document] = field(default_factory=list, repr=False)
    _embeddings: np.ndarray | None = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, documents: list[Document]) -> None:
        model = self._load()
        self._docs = list(documents)
        self._embeddings = model.encode(
            [d.text for d in self._docs],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query: str, k: int) -> list[Scored]:
        if self._embeddings is None or not self._docs:
            return []
        model = self._load()
        vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        scores = self._embeddings @ vec
        order = np.argsort(-scores)[:k]
        return [Scored(self._docs[i].doc_id, float(scores[i])) for i in order]
