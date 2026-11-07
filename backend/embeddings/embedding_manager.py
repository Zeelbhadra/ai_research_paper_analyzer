"""Embedding generation with a persistent content-addressed cache, plus per-model index building.

Layout (under storage/):
  embedding_cache/<model_key>/<sha1 of chunk text>.npy-shards   (vectors, never shared across models)
  indexes/<model_key>/<chunk_cfg_key>/                          (one index per model x chunking config)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config.settings import get_settings
from backend.embeddings.base import DenseEmbedder
from backend.embeddings.tfidf import TfidfIndex
from backend.preprocessing.chunking import Chunk
from backend.vectorstore.base import VectorStore
from backend.vectorstore.faiss_store import FaissStore

log = logging.getLogger(__name__)


def text_hash(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """One npz shard per call; an index file maps text-hash -> (shard,row). Keyed by model version."""

    def __init__(self, root: Path, model_key: str, model_version: str):
        self.dir = Path(root) / model_key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.version = model_version
        self._idx_path = self.dir / "index.json"
        self._map: dict[str, list] = {}
        if self._idx_path.exists():
            d = json.loads(self._idx_path.read_text())
            if d.get("version") == model_version:     # stale cache from a different version is ignored
                self._map = d["map"]
        self._shards: dict[str, np.ndarray] = {}

    def _shard(self, name: str) -> np.ndarray:
        if name not in self._shards:
            self._shards[name] = np.load(self.dir / name)
        return self._shards[name]

    def get(self, hashes: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for h in hashes:
            if h in self._map:
                shard, row = self._map[h]
                try:
                    out[h] = self._shard(shard)[row]
                except Exception:       # damaged shard => treat as miss
                    self._map.pop(h, None)
        return out

    def put(self, hashes: list[str], vecs: np.ndarray) -> None:
        if not hashes:
            return
        name = f"shard_{int(time.time() * 1000)}_{len(self._map)}.npy"
        np.save(self.dir / name, vecs.astype("float32"))
        for i, h in enumerate(hashes):
            self._map[h] = [name, i]
        self._idx_path.write_text(json.dumps({"version": self.version, "map": self._map}))


@dataclass
class BuildStats:
    n_chunks: int
    n_embedded: int          # newly computed (cache misses)
    embed_seconds: float
    index_seconds: float
    index_bytes: int
    index_dir: str
    reused_index: bool = False


class EmbeddingManager:
    def __init__(self, storage_dir: Path | None = None, batch_size: int = 32):
        self.root = Path(storage_dir or get_settings().storage_dir)
        self.batch_size = batch_size

    def index_dir(self, model_key: str, chunk_cfg_key: str) -> Path:
        return self.root / "indexes" / model_key / chunk_cfg_key

    def embed_chunks(self, model: DenseEmbedder, chunks: list[Chunk]) -> tuple[np.ndarray, int, float]:
        """Return (vectors aligned with chunks, n_newly_embedded, seconds). Dedupes identical texts."""
        cache = EmbeddingCache(self.root / "embedding_cache", model.key, model.version)
        hashes = [text_hash(c.text) for c in chunks]
        have = cache.get(list(set(hashes)))
        missing = [h for h in dict.fromkeys(hashes) if h not in have]
        texts = {h: c.text for h, c in zip(hashes, chunks)}
        t0 = time.perf_counter()
        if missing:
            vecs = model.embed_documents([texts[h] for h in missing], self.batch_size)
            cache.put(missing, vecs)
            have.update(dict(zip(missing, vecs)))
        dt = time.perf_counter() - t0
        return np.stack([have[h] for h in hashes]) if chunks else np.zeros((0, model.dim), "float32"), len(missing), dt

    def build_dense_index(self, model: DenseEmbedder, chunks: list[Chunk], chunk_cfg_key: str,
                          store_cls: type[VectorStore] = FaissStore, force: bool = False) -> tuple[VectorStore, BuildStats]:
        d = self.index_dir(model.key, chunk_cfg_key)
        ids = [c.chunk_id for c in chunks]
        if not force and (d / "meta.json").exists():
            try:
                s = store_cls.load(d)
                if s.model_version == model.version and set(s.ids()) == set(ids):
                    return s, BuildStats(len(ids), 0, 0.0, 0.0, s.disk_bytes(d), str(d), True)
            except Exception as e:
                log.warning("index at %s unusable (%s); rebuilding", d, e)
        X, n_new, dt = self.embed_chunks(model, chunks)
        for c in chunks:
            c.embedding_model_version = model.version
        t0 = time.perf_counter()
        s = store_cls(model.dim, model.version)
        s.add(ids, X)
        s.save(d)
        return s, BuildStats(len(ids), n_new, dt, time.perf_counter() - t0, s.disk_bytes(d), str(d))

    def build_tfidf_index(self, chunks: list[Chunk], chunk_cfg_key: str, **params) -> tuple[TfidfIndex, BuildStats]:
        idx = TfidfIndex(**params)
        t0 = time.perf_counter()
        idx.fit([c.chunk_id for c in chunks], [c.text for c in chunks])
        dt = time.perf_counter() - t0
        d = self.index_dir(idx.key, chunk_cfg_key)
        idx.save(d)
        size = sum(f.stat().st_size for f in d.glob("*"))
        return idx, BuildStats(len(chunks), len(chunks), 0.0, dt, size, str(d))

    def load_tfidf_index(self, key: str, chunk_cfg_key: str) -> TfidfIndex:
        return TfidfIndex.load(self.index_dir(key, chunk_cfg_key))
