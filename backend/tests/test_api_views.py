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


# ---------------- discover endpoints ----------------
def paper_rec(title, source="arxiv", pdf="http://x/p.pdf", year="2023-01-01", **kw):
    from backend.ingestion.models import PaperRecord
    return PaperRecord(title=title, source=source, abstract="An abstract about " + title + ". It is long enough to analyze properly.", authors=["A. Author"],
                       published=year, abs_url=kw.pop("abs_url", "http://arxiv.org/abs/2301.00001"), pdf_url=pdf, **kw)


def test_search_validation(client):
    assert client.get("/api/search").status_code == 400
    assert client.get("/api/search?q=" + "x" * 400).status_code == 400
    assert client.get("/api/search?q=attention&limit=abc").status_code == 400


def test_search_contract_dedup_interleave_and_in_library(client, monkeypatch, lib):
    repo, _ = lib
    add_paper(repo, "arxiv:2301.00001", "Attention Is All You Need Reloaded", ["X"], "2023-01-01")
    import backend.ingestion.arxiv_client as ac
    import backend.ingestion.semantic_scholar_client as sc
    monkeypatch.setattr(ac, "search_arxiv", lambda q, n, **k: iter([paper_rec("Attention Is All You Need Reloaded"), paper_rec("Graph Matching Solvers For Vision", abs_url="http://arxiv.org/abs/2301.00002")]))
    monkeypatch.setattr(sc, "search_semantic_scholar", lambda q, n, **k: iter([paper_rec("attention is all you need reloaded.", source="semantic_scholar"), paper_rec("Ride Sharing With Autonomous Vehicles", source="semantic_scholar", abs_url="http://s2/3")]))
    d = client.get("/api/search?q=attention&limit=10").get_json()
    titles = [r["title"] for r in d["results"]]
    assert d["count"] == 3 and len(set(t.lower().strip(".") for t in titles)) == 3                  # the same paper from two sources appears once
    first = d["results"][0]
    assert first["in_library"] == "arxiv:2301.00001" and d["results"][1]["in_library"] is None
    assert set(first) == {"title", "authors", "abstract", "url", "pdf_url", "source", "year", "venue", "citation_count", "doi", "in_library"}


def test_search_survives_one_source_failing(client, monkeypatch, lib):
    import backend.ingestion.arxiv_client as ac
    import backend.ingestion.semantic_scholar_client as sc
    monkeypatch.setattr(ac, "search_arxiv", lambda q, n, **k: iter([paper_rec("Only Arxiv Answers This Time")]))
    monkeypatch.setattr(sc, "search_semantic_scholar", lambda q, n, **k: iter([]))                 # rate-limited source yields nothing
    assert client.get("/api/search?q=x+y").get_json()["count"] == 1


def test_analyze_validation_and_success(client, monkeypatch):
    assert client.post("/api/analyze", json={"text": "short"}).status_code == 400
    from io import BytesIO
    assert client.post("/api/analyze", data={"file": (BytesIO(b"hello"), "x.txt")}, content_type="multipart/form-data").status_code == 400
    assert client.post("/api/analyze", data={"file": (BytesIO(b"not a pdf" * 300), "x.pdf")}, content_type="multipart/form-data").status_code == 422
    import src.keyword_extractor as ke
    import src.topic_classifier as tc
    from backend.api.routes import legacy
    monkeypatch.setitem(legacy._BREAKER, "until", 1e18)                                              # extractive summary, no network
    monkeypatch.setattr(ke, "extract_keywords", lambda t, top_n=10: [("graph matching", 0.8), ("keypoints", 0.5)])
    seen = {}
    monkeypatch.setattr(tc, "classify_topic", lambda t, labels=None, top_n=5: seen.update(labels=labels) or [{"label": "Computer Vision", "score": 0.9}])
    text = "We study graph matching for computer vision. The method aligns keypoints across images using a learned affinity. " * 4
    d = client.post("/api/analyze", json={"text": text}).get_json()
    assert d["summary"] and d["keywords"][0]["word"] == "graph matching" and d["topics"][0]["label"] == "Computer Vision" and d["search_hint"] == "graph matching keypoints"
    assert seen["labels"] and "Machine Learning" in seen["labels"]                                  # the configured, informative taxonomy, not "Computer Science"
    pdf = client.post("/api/analyze", data={"file": (BytesIO(make_pdf_bytes()), "p.pdf")}, content_type="multipart/form-data")
    assert pdf.status_code == 200 and pdf.get_json()["pages"] == 6


def test_similar_returns_one_row_per_paper_without_backend_fields(client, monkeypatch):
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
    assert [s["paper_id"] for s in d["similar"]] == ["A", "B"] and set(d["similar"][0]) == {"paper_id", "title", "url", "closeness"}
    assert client.post("/api/similar", json={"text": "x"}).status_code == 400


# ---------------- library ----------------
def test_library_filters_detail_and_facets(client, lib):
    repo, _ = lib
    add_paper(repo, "arxiv:2401.00001", "Graph Matching Solvers", ["Ada Lovelace", "Bob"], "2024-03-01", enrichment={"summary": "S", "keywords": [["graph matching", 0.9]], "topics": [{"label": "Computer Vision", "score": 0.9}]})
    add_paper(repo, "title:abc", "Ride Sharing Pricing", ["Carol"], "2021-05-01", enrichment={"topics": [{"label": "Transportation", "score": 0.8}]})
    add_paper(repo, "file:x", "Uploaded Paper Without Metadata", [], None)
    add_paper(repo, "arxiv:cs/0301001", "Failed One", [], None, status="failed")
    g = lambda qs: client.get("/api/library/papers?" + qs).get_json()
    assert g("")["total"] == 3                                                                      # only readable papers are listed
    assert [p["title"] for p in g("q=ride")["papers"]] == ["Ride Sharing Pricing"]
    assert g("author=lovelace")["total"] == 1 and g("topic=transport")["total"] == 1
    assert g("year_from=2022")["total"] == 1 and g("year_to=2021")["total"] == 1                    # papers without a year never match a year filter
    assert g("q=graph matching")["papers"][0]["keywords"] == ["graph matching"]
    assert client.get("/api/library/papers?limit=zz").status_code == 400
    card = g("q=graph")["papers"][0]
    assert not ({"n_chunks", "status", "chunk_id", "source", "category"} & set(card))                # no backend fields
    d = client.get("/api/library/paper?id=arxiv:2401.00001").get_json()
    assert [s["name"] for s in d["sections"]] == ["abstract", "methodology"] and d["sections"][1]["chunks"][0]["pages"] == [2, 2]
    assert d["topics"] == [{"label": "Computer Vision", "score": 0.9}] and "chunk_id" not in d["sections"][0]["chunks"][0]
    assert client.get("/api/library/paper?id=arxiv:cs/0301001").status_code == 404                  # unreadable papers are not exposed
    assert client.get("/api/library/paper?id=nope").status_code == 404
    f = client.get("/api/library/facets").get_json()
    assert ["Ada Lovelace", 1] in f["authors"] and set(f) == {"years", "authors", "topics"}


def test_add_to_library_validation_status_and_duplicates(client, monkeypatch, lib):
    repo, _ = lib
    assert client.post("/api/library/add", json={"title": "x"}).status_code == 400
    r = client.post("/api/library/add", json={"title": "A Paper Without Any Open Access PDF", "abstract": "abstract"})
    assert r.status_code == 422 and "open-access" in r.get_json()["error"]
    add_paper(repo, "arxiv:2301.00001", "Already Here In The Library", ["X"], "2023-01-01")
    ok = client.post("/api/library/add", json={"title": "Already Here In The Library", "pdf_url": "http://x/p.pdf", "url": "http://arxiv.org/abs/2301.00001"})
    assert ok.status_code == 200 and ok.get_json() == {"status": "ready", "paper_id": "arxiv:2301.00001"}
    submitted = {}

    class Orch:
        def submit(self, name, params):
            submitted.update(name=name, params=params)
    from backend.api.routes import pipeline
    monkeypatch.setattr(pipeline, "get_system", lambda: (None, Orch(), None))
    new = client.post("/api/library/add", json={"title": "A Brand New Paper To Read", "pdf_url": "http://x/new.pdf", "url": "http://arxiv.org/abs/2399.00009", "authors": ["Z"]})
    assert new.status_code == 202 and new.get_json()["status"] == "adding" and submitted["name"] == "ingest_paper"
    assert submitted["params"]["paper"]["pdf_url"] == "http://x/new.pdf" and submitted["params"]["paper"]["title"] == "A Brand New Paper To Read"
    st = client.get("/api/library/add/status?id=arxiv:2301.00001").get_json()
    assert st["status"] == "ready" and "ask the assistant" in st["message"].lower()
    assert client.get("/api/library/add/status?id=nope").status_code == 404


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
