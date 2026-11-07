"""Load/performance tests. Thresholds are deliberately generous (they catch regressions of an order of magnitude, not
noise); precise measured numbers come from `python -m backend.benchmarking.perf_suite` (see DEPLOYMENT.md)."""
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from backend.embeddings.embedding_manager import EmbeddingManager
from backend.embeddings.tfidf import TfidfIndex
from backend.ingestion.document_processor import ingest_pdf_file
from backend.preprocessing.chunking import Chunk, ChunkConfig
from backend.tests.test_embeddings import FakeEmbedder

pytestmark = pytest.mark.perf


def synthetic_texts(n, words=120, vocab=3000, seed=1):
    rng = np.random.default_rng(seed)
    v = [f"w{i}x" for i in range(vocab)]
    return [" ".join(rng.choice(v, words)) for _ in range(n)], v


def test_tfidf_retrieval_latency_at_6000_chunks():
    texts, vocab = synthetic_texts(6000)
    idx = TfidfIndex(min_df=2).fit([f"c{i}" for i in range(len(texts))], texts)
    rng = np.random.default_rng(2)
    lat = []
    for _ in range(100):
        q = " ".join(rng.choice(vocab, 8))
        t = time.perf_counter()
        hits = idx.search(q, 10)
        lat.append((time.perf_counter() - t) * 1000)
        assert len(hits) <= 10 and all(a.score >= b.score for a, b in zip(hits, hits[1:]))
    assert np.percentile(lat, 95) < 25, f"p95 {np.percentile(lat, 95):.1f} ms"


def test_repeated_queries_do_not_leak_memory():
    texts, vocab = synthetic_texts(2000, vocab=1500)
    idx = TfidfIndex(min_df=2).fit([f"c{i}" for i in range(len(texts))], texts)
    rng = np.random.default_rng(5)
    for _ in range(50):
        idx.search(" ".join(rng.choice(vocab, 6)), 10)                       # warm-up (lazy CSC build, caches)
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    for _ in range(600):
        idx.search(" ".join(rng.choice(vocab, 6)), 10)
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert (cur - base) / 2**20 < 10, f"heap grew {(cur - base) / 2**20:.1f} MB over 600 queries"


def test_large_document_ingestion_time_and_memory(repo, tmp_path):
    import fitz
    rng = np.random.default_rng(7)
    vocab = [f"term{i}" for i in range(500)]
    doc = fitz.open()
    heads = ["Abstract", "1 Introduction", "2 Methodology", "3 Results", "4 Conclusion", "References"]
    for i in range(150):
        pg = doc.new_page()
        pg.insert_text((72, 40), "Journal Of Load Tests", fontsize=9)
        y = 80
        if i % 25 == 0:
            pg.insert_text((72, y), heads[min(i // 25, 5)], fontsize=14)
            y += 28
        pg.insert_textbox(fitz.Rect(72, y, 520, 780), " ".join(rng.choice(vocab, 320)) + ".", fontsize=10)
        pg.insert_text((290, 810), str(i + 1), fontsize=9)
    p = tmp_path / "big.pdf"
    p.write_bytes(doc.tobytes())
    doc.close()
    tracemalloc.start()
    t = time.perf_counter()
    chunks = ingest_pdf_file(repo, p, cfg=ChunkConfig("recursive", 256, 0.15))
    dt = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(chunks) > 100 and dt < 30 and peak / 2**20 < 200, (len(chunks), round(dt, 1), round(peak / 2**20))
    assert 120 <= max(c.page_end for c in chunks) <= 127 and repo.count("chunks") == len(chunks)         # reaches the end of the body; the references section (from p.126) is excluded by design
    assert "references" not in {c.section for c in chunks}


def test_embedding_cache_makes_second_pass_free(tmp_path):
    class Costly(FakeEmbedder):
        def _encode(self, texts, batch_size):
            time.sleep(0.002 * len(texts))                                                                # simulated model cost
            return super()._encode(texts, batch_size)
    chunks = [Chunk("d", "T", f"c{i}", 1, 1, "x", f"chunk text number {i} about topic {i % 17}", "u") for i in range(200)]
    mgr, e = EmbeddingManager(tmp_path), Costly()
    t = time.perf_counter()
    X1, n1, _ = mgr.embed_chunks(e, chunks)
    first = time.perf_counter() - t
    calls = e.calls
    t = time.perf_counter()
    X2, n2, _ = mgr.embed_chunks(e, chunks)
    second = time.perf_counter() - t
    assert n1 == 200 and n2 == 0 and e.calls == calls and np.allclose(X1, X2)
    assert second < first / 5, f"cache pass {second:.3f}s vs cold {first:.3f}s"


def test_many_concurrent_http_search_requests(monkeypatch):
    import app as flask_app
    from backend.api.routes import retrieval
    from backend.monitoring.logger import EventLog
    from backend.retrieval.retriever import Retriever
    from backend.retrieval.service import RetrievalService, ServiceConfig
    from backend.vectorstore.base import Hit

    class R(Retriever):
        kind, name = "sparse", "r"

        def retrieve(self, q, k, allow=None):
            time.sleep(0.01)
            return [Hit("c0", 1.0, 1)]
    svc = RetrievalService(lambda: R(), None, "r", "v", None, ServiceConfig(), EventLog())
    ch = Chunk("d", "Paper", "c0", 1, 1, "x", "text " * 40, "http://u")

    class Prod:
        service = svc

        def search(self, q, k, d=None, s=None):
            r = svc.retrieve(q, k)
            return {"mode": r.mode, "results": [{"chunk_id": "c0", "text": ch.text}]}
    monkeypatch.setattr(retrieval, "get_production", lambda: Prod())
    client = flask_app.app.test_client()

    def hit(i):
        t = time.perf_counter()
        r = client.get(f"/api/retrieval/search?q=question+number+{i}")
        return r.status_code, (time.perf_counter() - t) * 1000
    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(hit, range(160)))
    assert all(s == 200 for s, _ in out)
    assert np.percentile([ms for _, ms in out], 95) < 1000
