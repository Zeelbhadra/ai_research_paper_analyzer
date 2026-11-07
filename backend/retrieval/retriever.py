"""Common retrieval interface over dense (vector store) and sparse (TF-IDF) indexes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from backend.embeddings.base import DenseEmbedder
from backend.embeddings.tfidf import TfidfIndex
from backend.vectorstore.base import Hit, VectorStore


def make_filter(chunk_meta: dict[str, dict] | None, document_ids: set[str] | None = None,
                sections: set[str] | None = None) -> Callable[[str], bool] | None:
    """Metadata filter over chunk_id -> {document_id, section,...}; None if no constraints."""
    if not document_ids and not sections:
        return None

    def allow(cid: str) -> bool:
        m = (chunk_meta or {}).get(cid)
        if m is None:
            return False
        return (not document_ids or m["document_id"] in document_ids) and (not sections or m["section"] in sections)
    return allow


class Retriever(ABC):
    kind = "base"          # 'dense' | 'sparse' | 'hybrid' (reported in explanations)
    name = ""

    @abstractmethod
    def retrieve(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[Hit]: ...

    def health(self) -> bool:
        return True

    def evidence_signal(self, query: str, chunk_ids: list[str]) -> float | None:
        """Max query-chunk cosine over `chunk_ids` under the dense model, or None if this retriever has no dense model."""
        return None


class DenseRetriever(Retriever):
    kind = "dense"

    def __init__(self, embedder: DenseEmbedder, store: VectorStore):
        if store.model_version != embedder.version:
            raise ValueError(f"index built with {store.model_version}, embedder is {embedder.version}: "
                             "mixing embedding models is not allowed")
        self.embedder, self.store = embedder, store
        self.name = embedder.key

    def retrieve(self, query, k, allow=None):
        return self.store.search(self.embedder.embed_queries([query])[0], k, allow)

    def health(self) -> bool:
        return self.store.size > 0 and self.embedder.health_check()

    def evidence_signal(self, query, chunk_ids):
        vecs = [v for v in (self.store.vector(c) for c in chunk_ids) if v is not None]
        if not vecs:
            return None
        q = self.embedder.embed_queries([query])[0]
        return float(max(v @ q for v in vecs))


class SparseRetriever(Retriever):
    kind = "sparse"

    def __init__(self, index: TfidfIndex):
        self.index = index
        self.name = index.key

    def retrieve(self, query, k, allow=None):
        return self.index.search(query, k, allow)

    def health(self) -> bool:
        return self.index.size > 0
