"""Dense embedding backend interface + sentence-transformers implementations (SBERT, E5, BGE)."""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


def detect_device(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class EmbeddingError(Exception):
    pass


class DenseEmbedder(ABC):
    """Embeds chunks (documents) and queries into a shared, L2-normalised space."""
    family: str = "dense"
    checkpoint: str = ""
    dim: int = 0

    @abstractmethod
    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...

    def doc_text(self, t: str) -> str:
        return t

    def query_text(self, t: str) -> str:
        return t

    @property
    def version(self) -> str:
        """Changes whenever anything that affects vectors changes (checkpoint, prefixes, dim)."""
        sig = f"{self.checkpoint}|{self.doc_text('')}|{self.query_text('')}|{self.dim}"
        return f"{self.family}:{self.checkpoint}:{hashlib.sha1(sig.encode()).hexdigest()[:8]}"

    @property
    def key(self) -> str:
        """Filesystem-safe id used for index directories."""
        return self.version.replace("/", "_").replace(":", "_")

    def _validate(self, X: np.ndarray, n: int) -> np.ndarray:
        X = np.asarray(X, dtype="float32")
        if X.ndim != 2 or X.shape[0] != n:
            raise EmbeddingError(f"bad embedding shape {X.shape} for {n} texts")
        if self.dim and X.shape[1] != self.dim:
            raise EmbeddingError(f"dimension mismatch: expected {self.dim}, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise EmbeddingError("non-finite values in embeddings")
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-12)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        try:
            return self._validate(self._encode([self.doc_text(t) for t in texts], batch_size), len(texts))
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"{self.checkpoint}: {e}") from e

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        try:
            return self._validate(self._encode([self.query_text(t) for t in texts], batch_size), len(texts))
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"{self.checkpoint}: {e}") from e

    def health_check(self) -> bool:
        try:
            self.embed_queries(["health check"])
            return True
        except Exception as e:
            log.error("health check failed for %s: %s", self.checkpoint, e)
            return False


class SentenceTransformerEmbedder(DenseEmbedder):
    """Lazy-loading wrapper. Subclasses only set prefixes/instructions."""
    family = "sbert"
    query_prefix = ""
    doc_prefix = ""
    max_seq_length: int | None = 512

    def __init__(self, checkpoint: str, device: str | None = "auto", max_seq_length: int | None = None):
        self.checkpoint = checkpoint
        self.device = detect_device(device)
        self._max_len = max_seq_length or self.max_seq_length
        self._model = None
        self.dim = 0           # discovered at load time

    def load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                m = SentenceTransformer(self.checkpoint, device=self.device)
                if self._max_len:
                    m.max_seq_length = min(self._max_len, m.max_seq_length or self._max_len)
            except Exception as e:
                raise EmbeddingError(f"cannot load {self.checkpoint}: {e}") from e
            self._model = m
            self.dim = int(m.get_sentence_embedding_dimension())
        return self._model

    def doc_text(self, t: str) -> str:
        return self.doc_prefix + t

    def query_text(self, t: str) -> str:
        return self.query_prefix + t

    @property
    def version(self) -> str:
        if not self.dim:
            self.load()
        return super().version

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        m = self.load()
        return m.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)


class SBERTEmbedder(SentenceTransformerEmbedder):
    family = "sbert"


class E5Embedder(SentenceTransformerEmbedder):
    """E5 requires 'query: ' / 'passage: ' prefixes (model card); omitting them degrades quality."""
    family = "e5"
    query_prefix = "query: "
    doc_prefix = "passage: "


class BGEEmbedder(SentenceTransformerEmbedder):
    """BGE v1.5: short query->passage retrieval uses the instruction on the QUERY side only;
    passages get no prefix. Vectors are normalised, similarity = inner product (cosine)."""
    family = "bge"
    query_prefix = "Represent this sentence for searching relevant passages: "
    doc_prefix = ""


DEFAULT_CHECKPOINTS = {
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
}
_CLASSES = {"sbert": SBERTEmbedder, "e5": E5Embedder, "bge": BGEEmbedder}


def make_embedder(family: str, checkpoint: str | None = None, device: str | None = "auto",
                  max_seq_length: int | None = None) -> DenseEmbedder:
    if family not in _CLASSES:
        raise ValueError(f"unknown dense family {family!r}; choose from {sorted(_CLASSES)}")
    return _CLASSES[family](checkpoint or DEFAULT_CHECKPOINTS[family], device, max_seq_length)
