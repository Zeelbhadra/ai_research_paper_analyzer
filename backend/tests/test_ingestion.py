from types import SimpleNamespace
from datetime import datetime

import pytest
import requests

from backend.ingestion.deduplication import Deduplicator, paper_key
from backend.ingestion.document_processor import ingest_pdf_file, process_pdf, validate_pdf_bytes
from backend.ingestion.models import (HttpError, PaperRecord, extract_arxiv_id, normalize_doi,
                                      normalize_title, request_with_retry)
from backend.ingestion.pdf_extractor import ExtractionError, extract_pdf
from backend.ingestion.semantic_scholar_client import search_semantic_scholar
from backend.ingestion import arxiv_client


# ---------- identifiers / dedup ----------
@pytest.mark.parametrize("v,exp", [
    ("http://arxiv.org/abs/2101.00001v3", "2101.00001"),
    ("https://arxiv.org/pdf/2101.00001v1.pdf", "2101.00001"),
    ("arXiv:1706.03762", "1706.03762"),
    ("https://example.com/x", None), (None, None)])
def test_arxiv_id(v, exp):
    assert extract_arxiv_id(v) == exp


def test_normalizers():
    assert normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert normalize_title("Attention: Is All  You Need!") == "attention is all you need"


def test_dedup_across_sources_and_merge():
    a = PaperRecord("Attention Is All You Need", "arxiv", arxiv_id="1706.03762", abs_url="http://arxiv.org/abs/1706.03762v5")
    b = PaperRecord("Attention is all you need.", "semantic_scholar", doi="10.5555/3295222", citation_count=100000,
                    pdf_url="https://x/y.pdf")
    c = PaperRecord("Other long enough title for matching", "semantic_scholar", arxiv_id="1706.03762v1")
    d = Deduplicator()
    fresh = d.dedupe([a, b, c])
    assert len(fresh) == 2 or len(fresh) == 1  # a,b merge by title; c merges by arxiv id
    assert len(fresh) == 1
    rec = fresh[0]
    assert rec.doi == "10.5555/3295222" and rec.citation_count == 100000
    assert set(rec.sources_seen) == {"arxiv", "semantic_scholar"}
    # known DOI now also dedupes
    assert d.find(PaperRecord("zzz", "x", doi="10.5555/3295222")) is rec


def test_dedup_seeded_with_existing_skips():
    p = PaperRecord("Some Reasonably Long Paper Title", "arxiv", arxiv_id="2001.00001")
    p.paper_id = paper_key(p)
    assert Deduplicator([p]).dedupe([PaperRecord("different", "s2", arxiv_id="2001.00001")]) == []


def test_short_titles_not_title_matched():
    d = Deduplicator()
    assert len(d.dedupe([PaperRecord("Survey", "a"), PaperRecord("Survey", "b")])) == 2


# ---------- HTTP retry ----------
class FakeResp:
    def __init__(self, code, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.content = code, payload or {}, headers or {}, b""

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def request(self, method, url, **kw):
        self.calls.append((url, kw.get("params")))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_retry_on_429_then_success_honours_retry_after():
    sleeps = []
    s = FakeSession([FakeResp(429, headers={"Retry-After": "3"}), requests.ConnectionError("x"), FakeResp(200, {"ok": 1})])
    r = request_with_retry("GET", "http://x", session=s, sleep=sleeps.append)
    assert r.json() == {"ok": 1} and sleeps[0] == 3.0 and len(s.calls) == 3


def test_retry_gives_up_and_4xx_fails_fast():
    with pytest.raises(HttpError):
        request_with_retry("GET", "http://x", session=FakeSession([FakeResp(503)] * 3), max_retries=2, sleep=lambda _: None)
    s = FakeSession([FakeResp(404), FakeResp(200)])
    with pytest.raises(HttpError):
        request_with_retry("GET", "http://x", session=s, sleep=lambda _: None)
    assert len(s.calls) == 1


# ---------- clients (mocked) ----------
def test_semantic_scholar_pagination_and_mapping():
    item = lambda i: {"title": f"Paper {i}", "abstract": "abs", "authors": [{"name": "A"}], "year": 2020,
                      "citationCount": i, "url": f"https://s2/{i}", "externalIds": {"DOI": f"10.1/{i}", "ArXiv": f"2001.0000{i}"},
                      "openAccessPdf": None, "venue": "NeurIPS", "fieldsOfStudy": ["Computer Science"]}
    s = FakeSession([FakeResp(200, {"data": [item(1), item(2)], "next": 2}),
                     FakeResp(200, {"data": [item(3)]})])
    out = list(search_semantic_scholar("q", max_results=10, page_size=2, session=s, sleep=lambda _: None))
    assert [p.title for p in out] == ["Paper 1", "Paper 2", "Paper 3"]
    assert s.calls[1][1]["offset"] == 2
    assert out[0].pdf_url == "https://arxiv.org/pdf/2001.00001" and out[0].doi == "10.1/1" and out[0].venue == "NeurIPS"


def test_semantic_scholar_failure_is_swallowed():
    s = FakeSession([FakeResp(500)] * 10)
    assert list(search_semantic_scholar("q", session=s, sleep=lambda _: None)) == []


def test_arxiv_client_mapping_and_error(monkeypatch):
    res = SimpleNamespace(entry_id="http://arxiv.org/abs/2101.00001v2", title="T  itle", summary="S\nummary",
                          authors=[SimpleNamespace(name="X")], published=datetime(2021, 1, 1), primary_category="cs.LG",
                          doi=None, pdf_url="http://arxiv.org/pdf/2101.00001v2", journal_ref=None)

    class C:
        def __init__(self, **kw): pass
        def results(self, search):
            yield res
            raise RuntimeError("boom")
    out = list(arxiv_client.search_arxiv("q", 5, client_factory=C))
    assert len(out) == 1 and out[0].arxiv_id == "2101.00001" and out[0].title == "T itle" and out[0].category == "cs.LG"


# ---------- PDF extraction ----------
def test_extract_pages_and_backends(pdf_file):
    ex = extract_pdf(pdf_file)
    assert ex.n_pages == 6 and ex.extractor == "pymupdf" and not ex.likely_scanned
    ex2 = extract_pdf(pdf_file, backend="pdfplumber")
    assert ex2.extractor == "pdfplumber" and ex2.n_pages == 6


def test_corrupt_pdf_raises(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 garbage" * 100)
    with pytest.raises(ExtractionError):
        extract_pdf(p)
    with pytest.raises(ExtractionError):
        validate_pdf_bytes(b"<html>nope</html>" * 100)


def test_scanned_pdf_flagged(tmp_path):
    import fitz
    d = fitz.open()
    for _ in range(3):
        d.new_page()
    p = tmp_path / "scan.pdf"
    d.save(p)
    assert extract_pdf(p).likely_scanned
    with pytest.raises(ExtractionError, match="scanned"):
        process_pdf(p, PaperRecord("", "upload"))


def test_process_pdf_sections_and_header_removal(pdf_file):
    doc = process_pdf(pdf_file, PaperRecord("", "upload"))
    assert "Journal of Tests" not in doc.structured.text
    for sec in ("abstract", "introduction", "related_work", "methodology", "results", "conclusion", "references"):
        assert sec in doc.sections, sec
    assert doc.paper.title.startswith("A Study of Chunking")


def test_ingest_end_to_end_and_failure_logging(repo, pdf_file, tmp_path):
    chunks = ingest_pdf_file(repo, pdf_file)
    assert chunks and all(c.page_start >= 1 and c.page_end >= c.page_start and c.section for c in chunks)
    assert not any(c.section == "references" for c in chunks)
    assert repo.count("papers", "WHERE status='processed'") == 1
    assert len(repo.get_chunks("recursive-256-15")) == len(chunks)
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all" * 200)
    assert ingest_pdf_file(repo, bad) == []
    assert repo.count("extraction_failures") == 1
    assert repo.list_papers("failed")


def test_duplicate_upload_skipped(repo, pdf_file, tmp_path):
    d = Deduplicator()
    assert ingest_pdf_file(repo, pdf_file, dedup=d)
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(pdf_file.read_bytes())
    assert ingest_pdf_file(repo, copy, dedup=d) == []


def test_papers_sharing_a_template_title_are_not_merged_as_duplicates(repo, tmp_path):
    """Regression: two journal-template PDFs whose first line is the same boilerplate used to be deduplicated into one paper,
    silently dropping the second."""
    from backend.ingestion.deduplication import Deduplicator
    from backend.ingestion.document_processor import ingest_directory
    from backend.preprocessing.chunking import ChunkConfig
    from backend.tests.conftest import make_topic_pdf
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "Emotion in Reinforcement Learning Agents.pdf").write_bytes(make_topic_pdf("\u201dMachine Learning\u201d manuscript No.", "Emotion signals shape exploration in reinforcement learning agents and robots."))
    (d / "Explanatory machine learning for sequential teaching.pdf").write_bytes(make_topic_pdf("\u201dMachine Learning\u201d manuscript No.", "Explanatory learning helps humans teach sequential concepts to machine learners."))
    stats = ingest_directory(repo, d, ChunkConfig("recursive", 128, 0.15, min_chars=40))
    assert stats["processed"] == 2 and stats["failed"] == 0
    titles = sorted(p.title for p in repo.list_papers("processed"))
    assert titles == ["Emotion in Reinforcement Learning Agents", "Explanatory machine learning for sequential teaching"]      # file names, not the template line
    dd = Deduplicator()
    a, b = PaperRecord("\u201dMachine Learning\u201d manuscript No.", "upload"), PaperRecord("\u201dMachine Learning\u201d manuscript No.", "upload")
    assert len(dd.dedupe([a, b])) == 2                                         # junk titles never identify a paper
    assert len(Deduplicator().dedupe([PaperRecord("A Genuinely Descriptive Paper Title Here", "arxiv"), PaperRecord("a genuinely descriptive paper title here.", "s2")])) == 1
