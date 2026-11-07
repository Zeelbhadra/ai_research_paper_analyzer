"""Cross-encoder reranker (optional; adopted only if the benchmark shows a significant gain)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)
DEFAULT_RERANKER = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    def __init__(self, checkpoint: str = DEFAULT_RERANKER, device: str | None = None, max_length: int = 512, batch_size: int = 16):
        self.checkpoint, self.max_length, self.batch_size = checkpoint, max_length, batch_size
        self.device = device
        self._model = None

    def load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from backend.embeddings.base import detect_device
            self._model = CrossEncoder(self.checkpoint, max_length=self.max_length, device=self.device or detect_device())
        return self

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self.load()
        return [float(x) for x in self._model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                                       show_progress_bar=False)]

    def rerank(self, query: str, candidates: list[tuple], top_k: int | None = None) -> list[tuple]:
        """candidates: [(chunk, score, rank)] -> same tuples re-ordered by cross-encoder score (score replaced)."""
        s = self.score(query, [c[0].text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-s[i], i))
        out = [(candidates[i][0], s[i], r + 1) for r, i in enumerate(order)]
        return out[:top_k] if top_k else out
