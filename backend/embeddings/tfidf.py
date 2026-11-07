"""TF-IDF sparse lexical index (baseline + fallback). Independent of torch / any model download."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.vectorstore.base import Hit


class TfidfIndex:
    name = "tfidf"
    family = "tfidf"

    def __init__(self, ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True, max_features=None):
        self.params = dict(ngram_range=tuple(ngram_range), min_df=min_df, max_df=max_df,
                           sublinear_tf=sublinear_tf, max_features=max_features)
        self.vectorizer = TfidfVectorizer(stop_words="english", dtype=np.float32, **self.params)
        self.matrix = None
        self._ids: list[str] = []

    @property
    def version(self) -> str:
        return f"tfidf:sklearn:{hashlib.sha1(json.dumps(self.params, sort_keys=True, default=str).encode()).hexdigest()[:8]}"

    @property
    def key(self) -> str:
        return "tfidf_" + "_".join(f"{k}{v}" for k, v in self.params.items()).replace(" ", "").replace(",", "-")\
            .replace("(", "").replace(")", "").replace(".", "p")

    def fit(self, ids: list[str], texts: list[str]) -> "TfidfIndex":
        if len(ids) != len(texts) or not ids:
            raise ValueError("ids/texts must be non-empty and equal length")
        self.matrix = self.vectorizer.fit_transform(texts)    # L2-normalised rows
        self._ids = list(ids)
        return self

    @property
    def size(self) -> int:
        return len(self._ids)

    def ids(self) -> list[str]:
        return list(self._ids)

    def _columns(self):
        """Column-major copy of the matrix (built lazily; not pickled) so a query only touches the columns of its own terms."""
        if getattr(self, "_csc_m", None) is None:
            self._csc_m = self.matrix.tocsc()
        return self._csc_m

    def score_all(self, query: str) -> np.ndarray:
        """Cosine score of every chunk (0 where no query term occurs). Cost scales with the postings of the query's terms,
        not with the corpus size."""
        q = self.vectorizer.transform([query])
        if q.nnz == 0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self._columns()[:, q.indices] @ q.data).ravel()

    def search(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[Hit]:
        if self.matrix is None:
            raise RuntimeError("index not fitted")
        s = self.score_all(query)
        pos = np.flatnonzero(s > 0)                      # zero overlap is not a match
        if len(pos) == 0 or k <= 0:
            return []
        if allow is None and len(pos) > k:               # partial selection; ties are kept so the order stays deterministic
            kth = np.partition(s[pos], len(pos) - k)[len(pos) - k]
            pos = pos[s[pos] >= kth]
        order = pos[np.lexsort((pos, -s[pos]))]          # score desc, then index asc (== stable argsort semantics)
        hits: list[Hit] = []
        for i in order:
            if allow is None or allow(self._ids[i]):
                hits.append(Hit(self._ids[i], float(s[i]), len(hits) + 1))
                if len(hits) >= k:
                    break
        return hits

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "tfidf.pkl", "wb") as f:
            pickle.dump({"params": self.params, "vectorizer": self.vectorizer, "matrix": self.matrix, "ids": self._ids}, f)

    @classmethod
    def load(cls, directory: Path) -> "TfidfIndex":
        with open(Path(directory) / "tfidf.pkl", "rb") as f:   # our own artifact, not untrusted input
            d = pickle.load(f)
        o = cls(**d["params"])
        o.vectorizer, o.matrix, o._ids = d["vectorizer"], d["matrix"], d["ids"]
        return o
