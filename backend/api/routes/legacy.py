"""Discover endpoints: topic search, paper analysis (summary, keywords, related topics, similar papers).

Kept at the original URLs (/api/search, /api/analyze, /api/similar). Search now uses the retrying, de-duplicating discovery clients and
reports which results are already in the library; the summary uses an LLM when one is available and an extractive summary otherwise
(a circuit breaker stops repeated failing calls)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)
bp = Blueprint("discover", __name__)

_BREAKER = {"until": 0.0}                    # LLM circuit breaker (no credits / auth errors / outage)
BREAKER_SECONDS = 600
_AUTOMATION = Path(__file__).resolve().parents[2] / "config" / "automation.json"


def _topic_labels() -> list[str] | None:
    try:
        return json.loads(_AUTOMATION.read_text())["enrichment"].get("topic_labels") or None
    except Exception:
        return None


def _extractive_summary(text: str) -> str:
    from src.summarizer import _generate_tfidf_summary
    from backend.agents.document_agent import clean_for_enrichment
    return _generate_tfidf_summary(clean_for_enrichment(text)[:6000])


def summarize_abstract(text: str) -> tuple[str, str]:
    """(summary, method). Tries the LLM unless the breaker is open; falls back to the extractive summariser."""
    if time.time() >= _BREAKER["until"]:
        try:
            from backend.rag.llm import OpenAIProvider, ProviderError
            p = OpenAIProvider()
            if p.available():
                r = p.generate([{"role": "system", "content": "Summarize this research paper text for a curious non-expert in 3-4 short sentences. Plain words."},
                                {"role": "user", "content": text[:6000]}], 220, 0.3)
                return r.text.strip(), "written"
        except ProviderError as e:
            _BREAKER["until"] = time.time() + BREAKER_SECONDS
            log.warning("LLM summaries disabled for %ds: %s", BREAKER_SECONDS, str(e)[:120])
        except Exception:
            log.exception("summary failed")
    return _extractive_summary(text), "extractive"


def _in_library(records):
    from backend.api.routes.library import repo
    from backend.ingestion.deduplication import Deduplicator
    d = Deduplicator([p for p in repo().list_papers("processed")])
    return {id(r): (d.find(r).paper_id if d.find(r) else None) for r in records}


@bp.get("/api/search")
def api_search():
    query = (request.args.get("q") or "").strip()
    if not query or len(query) > 300:
        return jsonify({"error": "Please enter a topic (1-300 characters)."}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 10)), 20))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    try:
        from backend.ingestion.arxiv_client import search_arxiv
        from backend.ingestion.deduplication import Deduplicator
        from backend.ingestion.semantic_scholar_client import search_semantic_scholar
        lists = [list(search_arxiv(query, limit)), list(search_semantic_scholar(query, limit))]
        mixed = [x for pair in zip(*lists) for x in pair] + lists[0][len(lists[1]):] + lists[1][len(lists[0]):]   # interleave sources
        papers = Deduplicator().dedupe(mixed)[:limit]
        known = _in_library(papers)
        out = [{"title": p.title, "authors": p.authors, "abstract": p.abstract, "url": p.abs_url, "pdf_url": p.pdf_url, "source": p.source, "year": (p.published or "")[:4] or None,
                "venue": p.venue, "citation_count": p.citation_count, "doi": p.doi, "in_library": known[id(p)]} for p in papers]
        return jsonify({"results": out, "count": len(out), "query": query})
    except Exception:
        log.exception("search failed")
        return jsonify({"error": "Search is unavailable right now. Please try again."}), 500


@bp.post("/api/analyze")
def api_analyze():
    """JSON {"text": ...} (an abstract or any text) or a multipart PDF upload (field 'file').
    Returns summary, keywords, related topics, and a hint for searching related papers online."""
    text, pages, title = "", None, None
    if request.files.get("file") is not None:
        f = request.files["file"]
        data = f.read()
        if not (f.filename or "").lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400
        try:
            from backend.ingestion.document_processor import validate_pdf_bytes
            from backend.ingestion.pdf_extractor import ExtractionError, extract_pdf
            from backend.preprocessing.text_cleaner import clean_pages
            validate_pdf_bytes(data)
            ex = extract_pdf(data=data)
            if ex.likely_scanned:
                return jsonify({"error": "This PDF has no selectable text (it looks scanned), so it cannot be analyzed."}), 422
            text, pages = "\n".join(clean_pages(ex.pages)), ex.n_pages
        except ExtractionError:
            return jsonify({"error": "This PDF could not be read."}), 422
    else:
        body = request.get_json(silent=True) or {}
        text = (body.get("text") or "").strip()
        title = (body.get("title") or "").strip() or None
    if len(text) < 40:
        return jsonify({"error": "Please provide at least a sentence or two of text, or a PDF."}), 400
    out = {"pages": pages}
    try:
        out["summary"], _ = summarize_abstract(text)
    except Exception:
        log.exception("summary failed")
        out["summary"] = text[:400]
    try:
        from src.keyword_extractor import extract_keywords
        out["keywords"] = [{"word": w, "score": float(s)} for w, s in extract_keywords(text[:5000], top_n=10)]
    except Exception:
        log.exception("keywords failed")
        out["keywords"] = []
    try:
        from src.topic_classifier import classify_topic
        out["topics"] = [t for t in classify_topic(text[:2000], _topic_labels(), top_n=5)]
    except Exception:
        log.exception("topics failed")
        out["topics"] = []
    out["search_hint"] = " ".join(k["word"] for k in out["keywords"][:2]) or (title or "")
    return jsonify(out)


@bp.post("/api/similar")
def api_similar():
    """The most similar papers in the library to a text (one row per paper)."""
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if len(text) < 20:
        return jsonify({"error": "text is required"}), 400
    try:
        from backend.api.routes.retrieval import get_production
        pr = get_production()
        res = pr.search(text[:1000], 30)
        seen, similar = set(), []
        for r in res["results"]:
            if r["document_id"] in seen:
                continue
            seen.add(r["document_id"])
            sig, _ = pr.service.evidence_signal(text[:1000], [r["chunk_id"]])
            similar.append({"paper_id": r["document_id"], "title": r["title"], "url": r["source_url"], "closeness": None if sig is None else round(sig, 3)})
            if len(similar) >= 6:
                break
        return jsonify({"similar": similar})
    except Exception:
        log.exception("similar failed")
        return jsonify({"similar": [], "error": "Similar papers are unavailable right now."}), 200
