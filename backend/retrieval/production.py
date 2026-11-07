"""Glue between the model registry and the retrieval service."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.benchmarking.model_registry import ModelRegistry
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.base import make_embedder
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.monitoring.logger import EventLog
from backend.retrieval.factory import build_retriever, tfidf_index_dir
from backend.retrieval.retriever import make_filter
from backend.retrieval.service import RetrievalService, ServiceConfig

log = logging.getLogger(__name__)
DEFAULT_CFG = "recursive-256-15"


def components_of(name: str) -> list[str]:
    """hybrid:e5:1.0:0.5 -> [tfidf, e5];  bge -> [bge];  tfidf -> [tfidf]."""
    return ["tfidf", name.split(":")[1]] if name.startswith("hybrid:") else [name]


def ensure_indexes(name: str, repo: Repository, mgr: EmbeddingManager, cfg_key: str) -> list[str]:
    """Create any missing index a candidate needs (model change => index (re)generation). Existing,
    version-matching indexes are reused; dense embeddings come from the persistent cache when possible.
    Returns the list of components that had to be built."""
    built, chunks = [], None
    for comp in components_of(name):
        if comp == "tfidf":
            try:
                tfidf_index_dir(mgr, cfg_key)
                continue
            except FileNotFoundError:
                chunks = chunks or repo.get_chunks(cfg_key)
                mgr.build_tfidf_index(chunks, cfg_key)
                built.append("tfidf")
        else:
            e = make_embedder(comp)
            e.load()
            chunks = chunks or repo.get_chunks(cfg_key)
            _, st = mgr.build_dense_index(e, chunks, cfg_key)
            if not st.reused_index:
                built.append(comp)
    return built


class ProductionRetrieval:
    """Owns the RetrievalService, the chunk lookup used to hydrate hits, and registry synchronisation."""

    def __init__(self, cfg_key: str = DEFAULT_CFG, storage_dir: Path | None = None,
                 service_config: ServiceConfig | None = None):
        s = get_settings()
        self.cfg_key = cfg_key
        self.storage = Path(storage_dir or s.storage_dir)
        self.repo = Repository(self.storage / "papers.sqlite")
        self.mgr = EmbeddingManager(self.storage)
        self.registry = ModelRegistry(self.storage / "registry")
        self.events = EventLog(self.storage / "logs" / "retrieval_events.jsonl")
        from backend.monitoring.metrics import MetricsStore
        self.metrics = MetricsStore(self.storage / "metrics.sqlite")
        self.corpus_version = 0
        prod = self.registry.production
        if prod is None:
            raise RuntimeError("no production model in the registry; run backend.benchmarking.run_benchmark first")
        self.chunks = {c.chunk_id: c for c in self.repo.get_chunks(cfg_key)}
        self.meta = {cid: {"document_id": c.document_id, "section": c.section} for cid, c in self.chunks.items()}
        name = prod["config"].get("_name", prod["name"])
        fb = None if prod["family"] == "tfidf" else (lambda: build_retriever("tfidf", self.mgr, cfg_key))
        self.service = RetrievalService(lambda: build_retriever(name, self.mgr, cfg_key), fb, name, prod["version"],
                                        "tfidf" if fb else None, service_config, self.events)

    def search(self, query: str, k: int = 10, document_ids: set[str] | None = None,
               sections: set[str] | None = None) -> dict:
        try:
            res = self.service.retrieve(query, k, make_filter(self.meta, document_ids, sections))
        except Exception:
            self.metrics.record("retrieval_failure", 1)
            raise
        self.metrics.record("retrieval_latency_ms", res.latency_ms, mode=res.mode)
        self.metrics.record("fallback_used", 1.0 if res.fallback_active else 0.0)
        out = []
        for h in res.hits:
            c = self.chunks.get(h.chunk_id)
            if c is None:
                continue
            out.append({"rank": h.rank, "score": round(h.score, 5), "chunk_id": h.chunk_id, "document_id": c.document_id,
                        "title": c.paper_title, "section": c.section, "pages": [c.page_start, c.page_end],
                        "source_url": c.source_url, "text": c.text})
        return {"query": query, "mode": res.mode, "fallback_active": res.fallback_active, "retriever": res.retriever,
                "retriever_kind": res.kind, "latency_ms": round(res.latency_ms, 2), "bypass_reason": res.error,
                "results": out}

    def sync_with_registry(self) -> tuple[bool, str]:
        """If the registry's production model differs from the serving one: ensure its indexes exist, build and
        validate it, and only then swap. A failure leaves the working model serving."""
        self.registry = ModelRegistry(self.storage / "registry")          # re-read from disk
        prod = self.registry.production
        if prod is None or prod["version"] == self.service.status.primary_version:
            return False, "already serving the production model"
        name = prod["config"].get("_name", prod["name"])
        try:
            built = ensure_indexes(name, self.repo, self.mgr, self.cfg_key)
        except Exception as e:
            self.events.emit("model_switch_failed", to=name, error=f"index build failed: {e}")
            return False, f"index build failed for {name}: {e}"
        ok, msg = self.service.switch_primary(lambda: build_retriever(name, self.mgr, self.cfg_key), name, prod["version"])
        return ok, msg + (f" (built indexes: {built})" if built else "")

    def refresh_corpus(self) -> dict:
        """Reload chunks/metadata from the repository and rebuild the retrievers over the (already updated) indexes."""
        chunks = {c.chunk_id: c for c in self.repo.get_chunks(self.cfg_key)}
        meta = {cid: {"document_id": c.document_id, "section": c.section} for cid, c in chunks.items()}
        ok, msg = self.service.reload()
        if ok:                                   # swap data only together with a validated retriever
            self.chunks, self.meta = chunks, meta
            self.corpus_version += 1
        self.events.emit("corpus_refreshed", ok=ok, message=msg, n_chunks=len(chunks))
        return {"ok": ok, "message": msg, "n_chunks": len(chunks), "n_documents": len({c.document_id for c in chunks.values()})}

    def status(self) -> dict:
        st = self.service.snapshot()
        st["n_chunks"], st["n_documents"] = len(self.chunks), len({c.document_id for c in self.chunks.values()})
        st["production"] = self.registry.production
        return st
