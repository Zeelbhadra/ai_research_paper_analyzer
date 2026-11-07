"""End-to-end integration tests on a small generated corpus (no network, no model downloads; the dense side uses the
deterministic hashed bag-of-words embedder from test_embeddings, the real-model runs live in the demo scripts)."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from backend.benchmarking.model_registry import ModelRegistry
from backend.benchmarking.model_selector import maybe_promote
from backend.benchmarking.retrieval_evaluator import evaluate_retriever
from backend.benchmarking.selection import Candidate, derive_budgets, load_policy, select
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.embeddings.tfidf import TfidfIndex
from backend.explainability.evidence_explainer import explain_answer
from backend.ingestion.document_processor import ingest_directory
from backend.monitoring.logger import EventLog
from backend.preprocessing.chunking import ChunkConfig
from backend.rag.llm import LLMProvider, LLMResponse, ProviderError
from backend.rag.pipeline import RAGConfig, RAGPipeline
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.retriever import DenseRetriever, Retriever, SparseRetriever
from backend.retrieval.service import RetrievalService, ServiceConfig
from backend.tests.conftest import make_topic_pdf
from backend.tests.test_embeddings import FakeEmbedder
from backend.vectorstore.base import Hit

PAPERS = {
    "Swin Windows": "The Swin Transformer computes self attention inside shifted windows which reduces the complexity of vision models.",
    "Graph Matching": "Dual decomposition solvers provide tight lower bounds for quadratic assignment in graph matching benchmarks.",
    "Ride Sharing": "Autonomous vehicles change the profit maximisation of ride sharing platforms with human drivers and dynamic pricing.",
}


@pytest.fixture
def corpus(tmp_path, repo):
    d = tmp_path / "pdfs"
    d.mkdir()
    for title, sent in PAPERS.items():
        (d / f"{title}.pdf").write_bytes(make_topic_pdf(title, sent))
    stats = ingest_directory(repo, d, ChunkConfig("recursive", 128, 0.15, min_chars=40))
    assert stats["processed"] == 3 and stats["failed"] == 0
    chunks = repo.get_chunks("recursive-128-15")
    return repo, chunks, tmp_path


class FakeProd:
    def __init__(self, chunks, service):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.meta = {c.chunk_id: {"document_id": c.document_id, "section": c.section} for c in chunks}
        self.service = service


def hybrid_over(chunks, tmp_path, dense_fails=None):
    mgr = EmbeddingManager(tmp_path / "store")
    sp = SparseRetriever(TfidfIndex(min_df=1, max_df=1.0).fit([c.chunk_id for c in chunks], [c.text for c in chunks]))
    emb = FakeEmbedder()
    store, _ = mgr.build_dense_index(emb, chunks, "cfg")
    dense = DenseRetriever(emb, store)
    if dense_fails is not None:
        dense.retrieve = lambda q, k, allow=None: (_ for _ in ()).throw(RuntimeError(dense_fails))
    return HybridRetriever(sp, dense, name="hybrid:fake"), sp


class Stub(LLMProvider):
    name, model = "stub", "stub-1"

    def __init__(self, text):
        self.text = text

    def generate(self, messages, max_tokens=500, temperature=0.0):
        return LLMResponse(self.text, "stub", "stub-1", 100, 20, 0.01, 0.0)


# ---------------- ingestion -> indexing -> retrieval -> RAG -> explanation ----------------
def test_pdf_to_cited_answer_keeps_page_and_section_provenance(corpus):
    repo, chunks, tmp = corpus
    assert all(c.page_start >= 1 and c.section for c in chunks) and {c.document_id for c in chunks}.__len__() == 3
    hybrid, sp = hybrid_over(chunks, tmp)
    svc = RetrievalService(lambda: hybrid, lambda: sp, "hybrid:fake", "v1", "tfidf", ServiceConfig(), EventLog())
    ans = "The Swin Transformer computes self attention inside shifted windows which reduces the complexity of vision models [S1]."
    pipe = RAGPipeline(FakeProd(chunks, svc), Stub(ans), None, RAGConfig(min_coverage=None))
    r = pipe.ask("How does the Swin Transformer reduce complexity with shifted windows?")
    assert r["answer_mode"] == "llm" and r["retrieval"]["kind"] == "hybrid"
    top = r["references"][0]
    assert "Swin" in top["title"] and top["pages"][0] >= 1 and top["section"] in {"abstract", "introduction", "methodology", "results", "conclusion", "front_matter", "other"}
    assert r["citation_quality"]["valid_citation_fraction"] == 1.0
    e = explain_answer(r)
    first = e["sentences"][0]
    assert first["kind"] == "stated_in_source" and first["evidence"][0]["highlights"] and first["evidence"][0]["pages"] == top["pages"]


def test_document_filter_restricts_rag_to_selected_paper(corpus):
    repo, chunks, tmp = corpus
    hybrid, sp = hybrid_over(chunks, tmp)
    svc = RetrievalService(lambda: hybrid, lambda: sp, "hybrid:fake", "v1", "tfidf", ServiceConfig(), EventLog())
    gm = next(c.document_id for c in chunks if "Graph" in c.paper_title)
    r = RAGPipeline(FakeProd(chunks, svc), None, None, RAGConfig()).ask("What lower bounds are provided?", document_ids=[gm])
    assert {x["paper_id"] for x in r["references"]} == {gm} and r["answer_mode"] in ("extractive", "insufficient")


def test_dense_failure_falls_back_to_tfidf_and_answer_is_still_grounded(corpus):
    repo, chunks, tmp = corpus
    hybrid, sp = hybrid_over(chunks, tmp, dense_fails="vector store unavailable")
    ev = EventLog()
    svc = RetrievalService(lambda: hybrid, lambda: sp, "hybrid:fake", "v1", "tfidf", ServiceConfig(failure_threshold=1), ev)
    pipe = RAGPipeline(FakeProd(chunks, svc), Stub("Autonomous vehicles change the profit of ride sharing platforms with human drivers [S1]."), None, RAGConfig())
    r = pipe.ask("How do autonomous vehicles change ride sharing profit?")
    assert r["retrieval"]["fallback_active"] and r["retrieval"]["kind"] == "sparse" and r["references"] and "Ride Sharing" in r["references"][0]["title"]
    assert svc.snapshot()["fallback_active"] and "fallback_activated" in [e["kind"] for e in ev.tail()]


# ---------------- benchmark -> selection -> registry -> guarded promotion ----------------
def test_benchmark_to_selection_to_promotion(corpus):
    repo, chunks, tmp = corpus
    hybrid, sp = hybrid_over(chunks, tmp)
    dense = hybrid.dense
    queries = []
    for c in chunks[::3]:                                                     # queries derived from the passages themselves
        queries.append({"query_id": f"q{len(queries)}", "query": " ".join(c.text.split()[10:22]), "rel": {c.chunk_id: 2}, "category": "technical"})
    assert len(queries) >= 8
    cands, per_model = [], {}
    for name, r, lat in (("tfidf", sp, 1.0), ("dense", dense, 9.0), ("hybrid", hybrid, 10.0)):
        res = evaluate_retriever(name, lambda q, k, r=r: [h.chunk_id for h in r.retrieve(q, k)], queries, lambda q: q["rel"])
        m = res.means()
        per_model[name] = m["nDCG@10"]
        cands.append(Candidate(name, {"nDCG@10": m["nDCG@10"], "Recall@5": m["R@5"], "MRR@10": m["MRR@10"], "Precision@5": m["P@5"], "MAP": m["MAP"], "latency_p50_ms": lat,
                                      "latency_p95_ms": lat * 1.5, "throughput_qps": 100, "memory_mb": 20.0 + lat, "indexing_seconds": lat},
                              [res.per_query[q["query_id"]]["nDCG@10"] for q in queries]))
    pol = load_policy()
    sel1 = select(cands, pol, derive_budgets(pol, available_mb=8000))
    sel2 = select(cands, pol, derive_budgets(pol, available_mb=8000))
    assert sel1.winner == sel2.winner and sel1.winner in per_model                              # deterministic
    reg = ModelRegistry(tmp / "reg")
    rows = [{"name": c.name, "family": c.name, "checkpoint": "c", "version": "v-" + c.name, "dim": 1, "index_location": "x", "overall_score": 50.0 + i, "eligible": not sel1.feasible[c.name],
             "fail_reasons": sel1.feasible[c.name], "metrics": c.metrics, "config": {"_name": c.name}} for i, c in enumerate(cands)]
    rid = reg.record_run({"dataset_version": "d", "selection_split": "validation", "n_queries": len(queries)}, rows, {"name": sel1.winner})
    ok, msg = maybe_promote(reg, rid, sel1.winner, {}, lambda cand: (True, "ok"))
    assert ok and reg.production["name"] == sel1.winner
    ok2, _ = maybe_promote(reg, rid, sel1.winner, {}, lambda cand: (True, "ok"))
    assert not ok2                                                                                # already in production


# ---------------- concurrency / thread safety ----------------
class Slow(Retriever):
    kind = "dense"

    def __init__(self, name, delay=0.05, fail_every=0):
        self.name, self.delay, self.fail_every, self.n, self._l = name, delay, fail_every, 0, threading.Lock()

    def retrieve(self, query, k, allow=None):
        time.sleep(self.delay)
        with self._l:
            self.n += 1
            bad = self.fail_every and self.n % self.fail_every == 0
        if bad:
            raise RuntimeError("injected")
        return [Hit(f"c{i}", 1.0 / (i + 1), i + 1) for i in range(k)]


def test_concurrent_requests_run_in_parallel_not_serialised():
    svc = RetrievalService(lambda: Slow("p"), None, "p", "v", None, ServiceConfig(), EventLog())
    t = time.perf_counter()
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: svc.retrieve(f"q{i}", 5), range(32)))
    wall = time.perf_counter() - t
    assert all(r.mode == "primary" and len(r.hits) == 5 for r in res)
    assert wall < 32 * 0.05 * 0.6, f"requests look serialised: {wall:.2f}s for 32x50ms"          # serial would be 1.6 s
    assert svc.snapshot()["total_requests"] == 32


def test_counters_stay_consistent_under_concurrent_failures():
    primary, fb = Slow("p", 0.005, fail_every=3), Slow("f", 0.001)
    svc = RetrievalService(lambda: primary, lambda: fb, "p", "v", "f", ServiceConfig(failure_threshold=10 ** 6), EventLog())
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda i: svc.retrieve("q", 3), range(120)))
    st = svc.snapshot()
    assert len(res) == 120 and all(r.hits for r in res)                                           # every request is served, by primary or fallback
    assert st["total_requests"] == 120 and st["fallback_requests"] == sum(r.mode == "fallback" for r in res) == st["total_failures"] == 40


# ---------------- health probes ----------------
def test_health_and_readiness(tmp_path, monkeypatch):
    import app as flask_app
    from backend.api.routes import health, library
    c = flask_app.app.test_client()
    assert c.get("/healthz").get_json()["status"] == "ok"
    monkeypatch.setattr(library, "_repo", Repository_for(tmp_path))
    from backend.config import settings
    monkeypatch.setattr(settings, "get_settings", lambda: _S(tmp_path))
    r = c.get("/readyz")
    assert r.status_code == 503 and r.get_json()["status"] == "not_ready"
    assert "no production model" in r.get_json()["checks"]["production_model"]["detail"]
    reg = ModelRegistry(tmp_path / "registry")
    (tmp_path / "indexes" / "tfidf_x" / "cfg").mkdir(parents=True)
    cand = {"name": "tfidf", "family": "tfidf", "checkpoint": "c", "version": "v", "dim": None, "index_location": "D:\\other\\machine\\storage\\indexes\\tfidf_x\\cfg",
            "overall_score": 1.0, "eligible": True, "fail_reasons": [], "metrics": {}, "config": {"_name": "tfidf"}}
    rid = reg.record_run({"dataset_version": "d", "selection_split": "validation", "n_queries": 1}, [cand], {"name": "tfidf"})
    reg.promote(rid, "tfidf", "t")
    ok = c.get("/readyz")
    assert ok.status_code == 200 and ok.get_json()["checks"]["production_model"]["ok"]                  # foreign absolute path resolves into this storage
    import shutil
    shutil.rmtree(tmp_path / "indexes")
    assert c.get("/readyz").status_code == 503


def Repository_for(tmp):
    from backend.database.repositories import Repository
    return Repository(tmp / "p.sqlite")


class _S:
    def __init__(self, tmp):
        self.storage_dir, self.db_path = tmp, tmp / "p.sqlite"


# ---------------- TF-IDF rewrite equals brute force ----------------
def test_tfidf_matches_bruteforce_including_ties_and_filters():
    rng = np.random.default_rng(3)
    vocab = [f"tok{i}" for i in range(60)]
    texts = [" ".join(rng.choice(vocab, 25)) for _ in range(300)] + ["tok1 tok2 tok3"] * 6                  # duplicates => exact score ties
    ids = [f"c{i}" for i in range(len(texts))]
    idx = TfidfIndex(min_df=1, max_df=1.0).fit(ids, texts)
    for q in ["tok1 tok2", "tok5 tok9 tok33", "tok1 tok2 tok3", "nothingmatches", "tok7"]:
        s = idx.vectorizer.transform([q])
        full = np.asarray((idx.matrix @ s.T).todense()).ravel()
        for k, allow in ((1, None), (5, None), (10, None), (400, None), (5, lambda c: int(c[1:]) % 3 == 0)):
            order = [i for i in np.argsort(-full, kind="stable") if full[i] > 0 and (allow is None or allow(ids[i]))][:k]
            got = idx.search(q, k, allow)
            assert [h.chunk_id for h in got] == [ids[i] for i in order]
            assert np.allclose([h.score for h in got], [full[i] for i in order], atol=1e-6)
