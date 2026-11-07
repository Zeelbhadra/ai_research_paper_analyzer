import json

import pytest

from backend.database.repositories import Repository
from backend.ingestion.models import PaperRecord
from backend.preprocessing.chunking import Chunk
from backend.tests.conftest import make_pdf_bytes


@pytest.fixture
def client():
    import app as flask_app
    return flask_app.app.test_client()


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """Library/dashboard routes pointed at a temporary repository and storage directory."""
    from backend.api.routes import dashboard, library
    repo = Repository(tmp_path / "p.sqlite")
    monkeypatch.setattr(library, "_repo", repo)
    monkeypatch.setattr(dashboard, "_storage", lambda: tmp_path)
    return repo, tmp_path


def add_paper(repo, pid, title, authors, published, status="processed", enrichment=None, n_chunks=2):
    p = PaperRecord(title=title, source="arxiv", authors=authors, published=published, abstract="abstract about " + title, status=status)
    p.paper_id = pid
    p.enrichment = enrichment or {}
    repo.upsert_paper(p)
    chunks = [Chunk(pid, title, f"{pid}:recursive-256-15:{i}", i + 1, i + 1, "methodology" if i else "abstract", f"text {i} " * 30, "u", None, i) for i in range(n_chunks)]
    repo.replace_chunks(pid, "recursive-256-15", chunks)


def test_json_keeps_server_order(client):
    from flask import jsonify
    import app as flask_app
    with flask_app.app.test_request_context():
        assert list(json.loads(jsonify({"zeta": 1, "alpha": 2}).get_data())) == ["zeta", "alpha"]


# ---------------- legacy endpoints ----------------
def test_legacy_search_validation(client):
    assert client.get("/api/search").status_code == 400
    assert client.get("/api/search?q=" + "x" * 400).status_code == 400
    assert client.get("/api/search?q=attention&limit=abc").status_code == 400
    assert client.get("/api/search?q=attention&source=pubmed").status_code == 400


def test_legacy_search_contract_and_source_filter(client, monkeypatch):
    import src.services.paper_fetcher as pf
    import src.services.paper_service as ps
    calls = []
    monkeypatch.setattr(pf, "fetch_arxiv", lambda q, n: calls.append("arxiv") or [{"title": "A", "authors": ["x"], "abstract": "First sentence is long enough to count. Second one too. " * 3, "url": "http://a", "source": "arxiv"}])
    monkeypatch.setattr(pf, "fetch_semantic_scholar", lambda q, n: calls.append("s2") or [])
    monkeypatch.setattr(ps, "get_summary", lambda pid: None)
    monkeypatch.setattr(ps, "save_summary", lambda *a, **k: None)
    from backend.api.routes import legacy
    monkeypatch.setitem(legacy._BREAKER, "until", 1e18)                        # no network: use the extractive summariser
    d = client.get("/api/search?q=attention&source=arxiv").get_json()
    assert calls == ["arxiv"] and d["count"] == 1 and d["results"][0]["abstract"] and d["results"][0]["summary"]
    assert set(d["results"][0]) >= {"title", "authors", "abstract", "url", "source", "summary"}


def test_legacy_analyze_validation_and_success(client, monkeypatch):
    assert client.post("/api/analyze", json={"text": "short"}).status_code == 400
    from io import BytesIO
    assert client.post("/api/analyze", data={"file": (BytesIO(b"hello"), "x.txt")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/analyze", data={"file": (BytesIO(b"not a pdf" * 300), "x.pdf")}, content_type="multipart/form-data").status_code == 422
    import src.keyword_extractor as ke
    import src.topic_classifier as tc
    monkeypatch.setattr(ke, "extract_keywords", lambda t, top_n=10: [("graph matching", 0.8)])
    monkeypatch.setattr(tc, "classify_topic", lambda t, labels=None, top_n=5: [{"label": "Computer Vision", "score": 0.9}])
    text = "We study graph matching for computer vision. The method aligns keypoints across images using a learned affinity. " * 4
    d = client.post("/api/analyze", json={"text": text}).get_json()
    assert d["summary"] and d["keywords"] == [{"word": "graph matching", "score": 0.8}] and d["topics"][0]["label"] == "Computer Vision"
    pdf = client.post("/api/analyze", data={"file": (BytesIO(make_pdf_bytes()), "p.pdf")}, content_type="multipart/form-data")
    assert pdf.status_code == 200 and pdf.get_json()["pages"] == 6


def test_legacy_similar_returns_one_row_per_paper(client, monkeypatch):
    from backend.api.routes import retrieval

    class Svc:
        def evidence_signal(self, q, ids):
            return 0.9, "v"

    class Prod:
        service = Svc()

        def search(self, q, k):
            r = lambda cid, doc: {"chunk_id": cid, "document_id": doc, "title": "T" + doc, "source_url": "http://" + doc, "section": "x", "pages": [1, 1]}
            return {"mode": "primary", "results": [r("a1", "A"), r("a2", "A"), r("b1", "B")]}
    monkeypatch.setattr(retrieval, "get_production", lambda: Prod())
    d = client.post("/api/similar", json={"text": "graph matching of keypoints for deformable shapes"}).get_json()
    assert [s["paper"]["paper_id"] for s in d["similar"]] == ["A", "B"] and d["similar"][0]["similarity"] == 0.9
    assert client.post("/api/similar", json={"text": "x"}).status_code == 400


# ---------------- library ----------------
def test_library_filters_detail_and_facets(client, lib):
    repo, _ = lib
    add_paper(repo, "arxiv:2401.00001", "Graph Matching Solvers", ["Ada Lovelace", "Bob"], "2024-03-01", enrichment={"summary": "S", "keywords": [["graph matching", 0.9]], "topics": [{"label": "Computer Vision", "score": 0.9}]})
    add_paper(repo, "title:abc", "Ride Sharing Pricing", ["Carol"], "2021-05-01", enrichment={"topics": [{"label": "Transportation", "score": 0.8}]})
    add_paper(repo, "file:x", "Uploaded Paper Without Metadata", [], None)
    add_paper(repo, "arxiv:cs/0301001", "Failed One", [], None, status="failed")
    g = lambda qs: client.get("/api/library/papers?" + qs).get_json()
    assert g("")["total"] == 4 and g("status=processed")["total"] == 3
    assert [p["title"] for p in g("q=ride&status=processed")["papers"]] == ["Ride Sharing Pricing"]
    assert g("author=lovelace")["total"] == 1 and g("topic=transport")["total"] == 1
    assert g("year_from=2022")["total"] == 1 and g("year_to=2021")["total"] == 1             # papers without a year never match a year filter
    assert g("q=graph matching")["papers"][0]["keywords"] == ["graph matching"]               # keywords are searchable
    assert client.get("/api/library/papers?limit=zz").status_code == 400
    d = client.get("/api/library/paper?id=arxiv:2401.00001").get_json()
    assert d["n_chunks"] == 2 and [s["name"] for s in d["sections"]] == ["abstract", "methodology"] and d["sections"][1]["chunks"][0]["pages"] == [2, 2]
    assert client.get("/api/library/paper?id=arxiv:cs/0301001").status_code == 200           # ids may contain ':' and '/'
    assert client.get("/api/library/paper?id=nope").status_code == 404
    f = client.get("/api/library/facets").get_json()
    assert ["Ada Lovelace", 1] in f["authors"] and [2024, 1] in [list(y) for y in f["years"]]


# ---------------- dashboard / benchmarks ----------------
def write_registry(tmp, adjudicated_last=True):
    cand = lambda n, nd, q: {"name": n, "family": n, "checkpoint": "c", "version": "v" + n, "dim": 1, "index_location": "x", "overall_score": q, "eligible": True, "fail_reasons": [],
                             "footprint": 1.0, "pareto_optimal": True, "non_inferiority_vs_reference": None, "config": {"_name": n},
                             "metrics": {"nDCG@10": nd, "Recall@5": 0.5, "MRR@10": 0.5, "Precision@5": 0.3, "MAP": 0.4, "latency_p50_ms": 1, "latency_p95_ms": 2, "throughput_qps": 9, "memory_mb": 5, "indexing_seconds": 1}}
    meta = lambda adj: {"selection_split": "validation", "n_queries": 21, "dataset_version": "d", "n_chunks": 100, "adjudication": adj, "trace": ["t"]}
    runs = [{"run_id": "run0001", "timestamp": "2026-01-01T00:00:00", "meta": meta(None), "candidates": [cand("tfidf", 0.7, 60), cand("sbert", 0.5, 40)], "winner": {"name": "tfidf"}},
            {"run_id": "run0002", "timestamp": "2026-01-02T00:00:00", "meta": meta("adj.json"), "candidates": [cand("tfidf", 0.99, 99)], "winner": {"name": "tfidf"}}]
    prod = {"name": "tfidf", "family": "tfidf", "checkpoint": "c", "version": "vtfidf", "promoted_at": "2026-01-01T00:00:00", "run_id": "run0001", "overall_score": 60, "config": {"_name": "tfidf"}}
    (tmp / "registry").mkdir(exist_ok=True)
    (tmp / "registry" / "registry.json").write_text(json.dumps({"runs": runs, "production": prod, "promotions": [{"at": "t", "from": None, "to": "tfidf", "run_id": "run0001", "reason": "r"}]}))
    (tmp / "benchmark").mkdir(exist_ok=True)
    pq = {"tfidf": {f"q{i}": {**{f"{m}@{k}": 0.5 for m in ("P", "R", "nDCG") for k in (1, 3, 5, 10)}, "MRR@10": 0.6, "AP@10": 0.4} for i in range(3)}}
    (tmp / "benchmark" / "retrieval_validation_recursive-256-15_per_query.json").write_text(json.dumps(pq))
    recs = [{"query_id": f"q{i}", "split": "dev" if i < 2 else "test", "category": "definition", "human_verified": False} for i in range(3)]
    recs.append({"query_id": "q9", "split": "dev", "category": "definition", "human_verified": False, "excluded_reason": "context_dependent"})
    (tmp / "benchmark" / "dataset.json").write_text(json.dumps({"version": "v1", "records": recs}))


def test_dashboard_ignores_adjudicated_runs_and_counts_distinct_failures(client, lib):
    repo, tmp = lib
    write_registry(tmp)
    add_paper(repo, "arxiv:1", "One", [], None)
    for _ in range(3):
        repo.log_failure("p", "C:\\secret\\dir\\bad.pdf", "pymupdf: no pages; failed on C:\\secret\\dir\\bad.pdf")      # same document, logged repeatedly
    d = client.get("/api/dashboard").get_json()
    assert d["latest_benchmark"]["run_id"] == "run0001" and d["latest_benchmark"]["winner"] == "tfidf"        # run0002 is a sensitivity run
    assert d["corpus"]["extraction_failures"] == 1 and d["corpus"]["papers"] == 1 and d["active_model"]["name"] == "tfidf"
    assert d["retrieval_serving"]["latency_ms"]["n"] == 0                                                        # no traffic recorded: reported as such, not invented


def test_benchmarks_overview_aggregates_stored_results(client, lib):
    _, tmp = lib
    write_registry(tmp)
    o = client.get("/api/benchmarks/overview").get_json()
    assert [r["run_id"] for r in o["runs"]] == ["run0001", "run0002"] and o["runs"][1]["adjudicated"] and not o["runs"][0]["adjudicated"]
    m = o["ranking_metrics"]["validation"]["tfidf"]
    assert m["n"] == 3 and m["P@5"] == pytest.approx(0.5) and m["MRR@10"] == pytest.approx(0.6) and m["MAP"] == pytest.approx(0.4) and o["ranking_metrics"]["test"] is None
    assert o["dataset"]["retained"] == 3 and o["dataset"]["generated"] == 4 and o["dataset"]["splits"] == {"dev": 2, "test": 1}      # excluded queries not counted
    assert o["production"]["name"] == "tfidf" and len(o["promotions"]) == 1


def test_overview_without_any_benchmark_does_not_crash(client, lib):
    o = client.get("/api/benchmarks/overview").get_json()
    assert o["runs"] == [] and o["production"] is None and o["dataset"] is None


# ---------------- explain / pipeline ----------------
def test_explain_response_endpoint(client):
    assert client.post("/api/explain/response", json={"response": {"answer": "x"}}).status_code == 400
    ok = {"answer": "a", "answer_mode": "llm", "references": [], "sentences": [{"text": "Hello world again.", "citations": [], "invalid_citations": [], "status": "uncited_claim", "support": None, "supporting_label": None}]}
    d = client.post("/api/explain/response", json={"response": ok}).get_json()
    assert d["sentences"][0]["kind"] == "unverified" and "No valid citation" in d["sentences"][0]["warning"]


def test_pipeline_failures_hide_local_paths(client, monkeypatch, tmp_path):
    from backend.api.routes import pipeline
    repo = Repository(tmp_path / "f.sqlite")
    repo.log_failure("p", "C:\\Users\\someone\\secret\\papers\\Corrupt Paper.pdf", "pymupdf: no pages; pdfplumber failed on C:\\Users\\someone\\secret\\papers\\Corrupt Paper.pdf: Unexpected EOF")
    repo.log_failure("p", "C:\\Users\\someone\\secret\\papers\\Corrupt Paper.pdf", "second attempt")

    class SV:
        pass
    sv = SV()
    sv.repo = repo
    monkeypatch.setattr(pipeline, "get_system", lambda: (sv, None, None))
    out = client.get("/api/pipeline/failures").get_json()
    assert len(out) == 1                                                       # one row per document
    blob = json.dumps(out)
    assert "someone" not in blob and "secret" not in blob and out[0]["path"] == "Corrupt Paper.pdf"
