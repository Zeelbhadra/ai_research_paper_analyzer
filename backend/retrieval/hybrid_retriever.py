"""Hybrid sparse+dense retrieval via weighted Reciprocal Rank Fusion (RRF).

RRF needs no score calibration between TF-IDF cosine and dense cosine: score(d) = sum_i w_i / (k + rank_i(d)).
It is evaluated as an *experimental candidate*, never assumed superior.
"""
from __future__ import annotations

from typing import Callable

from backend.retrieval.retriever import Retriever
from backend.vectorstore.base import Hit


class HybridRetriever(Retriever):
    kind = "hybrid"

    def __init__(self, sparse: Retriever, dense: Retriever, w_sparse: float = 1.0, w_dense: float = 1.0,
                 rrf_k: int = 60, depth: int = 50, name: str | None = None):
        if w_sparse < 0 or w_dense < 0 or (w_sparse + w_dense) == 0:
            raise ValueError("fusion weights must be non-negative and not both zero")
        self.sparse, self.dense = sparse, dense
        self.w_sparse, self.w_dense, self.rrf_k, self.depth = w_sparse, w_dense, rrf_k, depth
        self.name = name or f"hybrid[{sparse.name}+{dense.name}|ws={w_sparse},wd={w_dense},k={rrf_k}]"
        self.last_components: dict[str, dict[str, int]] = {}   # chunk_id -> {'sparse_rank','dense_rank'} (for explainability)

    def retrieve(self, query: str, k: int, allow: Callable[[str], bool] | None = None) -> list[Hit]:
        depth = max(self.depth, k)
        s_hits = self.sparse.retrieve(query, depth, allow)
        d_hits = self.dense.retrieve(query, depth, allow)
        fused: dict[str, float] = {}
        comp: dict[str, dict[str, int]] = {}
        for hits, w, tag in ((s_hits, self.w_sparse, "sparse_rank"), (d_hits, self.w_dense, "dense_rank")):
            for h in hits:
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + w / (self.rrf_k + h.rank)
                comp.setdefault(h.chunk_id, {})[tag] = h.rank
        order = sorted(fused, key=lambda c: (-fused[c], c))[:k]
        self.last_components = {c: comp[c] for c in order}
        return [Hit(c, fused[c], i + 1) for i, c in enumerate(order)]

    def evidence_signal(self, query, chunk_ids):
        return self.dense.evidence_signal(query, chunk_ids)

    def health(self) -> bool:
        return self.sparse.health() or self.dense.health()   # degrades gracefully to the healthy half
