"""FAISS exact inner-product store (vectors are L2-normalised, so IP == cosine).

Exact search is appropriate at this corpus size (10^3-10^5 chunks). Metadata filtering is done by
post-filtering with adaptive oversampling since flat FAISS has no native filters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from backend.vectorstore.base import Hit, VectorStore


class FaissStore(VectorStore):
    name = "faiss"

    def __init__(self, dim: int, model_version: str):
        super().__init__(dim, model_version)
        self.index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"expected {(len(ids), self.dim)}, got {vectors.shape}")
        if set(ids) & set(self._ids):
            raise ValueError("duplicate chunk ids")
        self.index.add(vectors)
        self._ids.extend(ids)

    def search(self, query_vec: np.ndarray, k: int, allow: Callable[[str], bool] | None = None) -> list[Hit]:
        n = self.index.ntotal
        if n == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(query_vec, dtype="float32").reshape(1, -1))
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        fetch = min(n, k if allow is None else max(k * 4, 50))
        while True:
            scores, idx = self.index.search(q, fetch)
            hits = [(self._ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]
            if allow is not None:
                hits = [h for h in hits if allow(h[0])]
            if len(hits) >= k or fetch >= n:
                break
            fetch = min(n, fetch * 4)
        return [Hit(c, s, r + 1) for r, (c, s) in enumerate(hits[:k])]

    def vector(self, chunk_id: str):
        if getattr(self, "_pos", None) is None or len(self._pos) != len(self._ids):
            self._pos = {c: i for i, c in enumerate(self._ids)}
        i = self._pos.get(chunk_id)
        return None if i is None else self.index.reconstruct(i)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def ids(self) -> list[str]:
        return list(self._ids)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "meta.json").write_text(json.dumps(
            {"dim": self.dim, "model_version": self.model_version, "ids": self._ids, "store": self.name}))

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        s = cls(meta["dim"], meta["model_version"])
        s.index = faiss.read_index(str(directory / "index.faiss"))
        s._ids = meta["ids"]
        if s.index.ntotal != len(s._ids) or s.index.d != s.dim:
            raise ValueError("corrupt index: id/vector count or dimension mismatch")
        return s
