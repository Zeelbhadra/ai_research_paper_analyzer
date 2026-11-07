"""Build retrievers from candidate names: 'tfidf', 'sbert', 'e5', 'bge', 'hybrid:<dense>[:ws:wd]'."""
from __future__ import annotations

from pathlib import Path

from backend.embeddings.base import make_embedder
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.embeddings.tfidf import TfidfIndex
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.retriever import DenseRetriever, Retriever, SparseRetriever
from backend.vectorstore.faiss_store import FaissStore


def tfidf_index_dir(mgr: EmbeddingManager, cfg_key: str) -> Path:
    d = next((mgr.root / "indexes").glob("tfidf_*/" + cfg_key), None)
    if d is None:
        raise FileNotFoundError("TF-IDF index missing; run embed_corpus first")
    return d


def build_retriever(name: str, mgr: EmbeddingManager, cfg_key: str, device: str = "auto",
                    _cache: dict | None = None) -> Retriever:
    cache = _cache if _cache is not None else {}
    if name.startswith("hybrid:"):
        parts = name.split(":")
        partner = parts[1]
        ws, wd = (float(parts[2]), float(parts[3])) if len(parts) == 4 else (1.0, 1.0)
        return HybridRetriever(build_retriever("tfidf", mgr, cfg_key, device, cache),
                               build_retriever(partner, mgr, cfg_key, device, cache), ws, wd, name=name)
    if name in cache:
        return cache[name]
    if name == "tfidf":
        r: Retriever = SparseRetriever(TfidfIndex.load(tfidf_index_dir(mgr, cfg_key)))
    else:
        e = make_embedder(name, device=device)
        e.load()
        r = DenseRetriever(e, FaissStore.load(mgr.index_dir(e.key, cfg_key)))
    cache[name] = r
    return r
