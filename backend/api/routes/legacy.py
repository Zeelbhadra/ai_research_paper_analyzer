"""The original PaperLens endpoints, kept and fixed.

Fixes versus the previous app.py: the response now carries `count` and `abstract` (the UI expected them), the `source`
filter is honoured, `/api/analyze` and `/api/similar` (called by the UI but never implemented) exist, input is validated,
and summarisation no longer fires one failing OpenAI call per result (a circuit breaker + extractive fallback)."""
from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)
bp = Blueprint("legacy", __name__)

_BREAKER = {"until": 0.0}                    # OpenAI circuit breaker (no credits / auth errors / outages)
BREAKER_SECONDS = 600


def _extractive_summary(text: str) -> str:
    from src.summarizer import _generate_tfidf_summary
    return _generate_tfidf_summary(text[:6000])


def summarize_abstract(text: str) -> tuple[str, str]:
    """(summary, method). Tries OpenAI unless the breaker is open; falls back to the extractive summariser."""
    if time.time() >= _BREAKER["until"]:
        try:
            from backend.rag.llm import OpenAIProvider, ProviderError
            p = OpenAIProvider()
            if p.available():
                r = p.generate([{"role": "system", "content": "Summarize research papers concisely."},
                                {"role": "user", "content": text[:6000]}], 200, 0.3)
                return r.text.strip(), "openai"
        except ProviderError as e:
            _BREAKER["until"] = time.time() + BREAKER_SECONDS
            log.warning("OpenAI summaries disabled for %ds: %s", BREAKER_SECONDS, str(e)[:120])
        except Exception:
            log.exception("summary failed")
    return _extractive_summary(text), "extractive"


@bp.get("/api/search")
def api_search():
    query = (request.args.get("q") or "").strip()
    if not query or len(query) > 300:
        return jsonify({"error": "Missing or too long query (1-300 characters)"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 10)), 20))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    source = request.args.get("source", "all")
    if source not in ("all", "arxiv", "semantic_scholar"):
        return jsonify({"error": "source must be all, arxiv or semantic_scholar"}), 400
    try:
        from src.services.paper_fetcher import fetch_arxiv, fetch_semantic_scholar
        from src.services.paper_service import get_summary, save_summary
        papers = []
        if source in ("all", "arxiv"):
            papers += fetch_arxiv(query, limit)
        if source in ("all", "semantic_scholar"):
            papers += fetch_semantic_scholar(query, limit)
        results = []
        for p in papers:
            pid = p.get("url") or p.get("title")
            summary = get_summary(pid)
            if not summary and p.get("abstract"):
                summary, method = summarize_abstract(p["abstract"])
                if method == "openai":                       # only cache LLM summaries (extractive is cheap to recompute)
                    save_summary(pid, p, summary)
            results.append({"title": p.get("title"), "authors": p.get("authors") or [], "abstract": p.get("abstract") or "",
                            "url": p.get("url"), "source": p.get("source"), "summary": summary or "Summary not available"})
        return jsonify({"results": results, "count": len(results)})
    except Exception:
        log.exception("search failed")
        return jsonify({"error": "Search failed; check the server log"}), 500


@bp.post("/api/analyze")
def api_analyze():
    """Accepts JSON {"text": ...} or a multipart PDF upload (field 'file'); returns summary, keywords and topics."""
    text = ""
    pages = None
    if request.files.get("file") is not None:
        f = request.files["file"]
        data = f.read()
        if not (f.filename or "").lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF uploads are supported"}), 400
        try:
            from backend.ingestion.document_processor import validate_pdf_bytes
            from backend.ingestion.pdf_extractor import ExtractionError, extract_pdf
            from backend.preprocessing.text_cleaner import clean_pages
            validate_pdf_bytes(data)
            ex = extract_pdf(data=data)
            if ex.likely_scanned:
                return jsonify({"error": "This PDF has no extractable text (scanned images; OCR is not supported)"}), 422
            text, pages = "\n".join(clean_pages(ex.pages)), ex.n_pages
        except ExtractionError as e:
            return jsonify({"error": f"Could not read PDF: {e}"}), 422
    else:
        body = request.get_json(silent=True) or {}
        text = (body.get("text") or "").strip()
    if len(text) < 40:
        return jsonify({"error": "Provide at least a sentence or two of text (or a PDF)"}), 400
    out = {"chars": len(text), "pages": pages}
    try:
        out["summary"] = _extractive_summary(text)
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
        out["topics"] = classify_topic(text[:2000], top_n=5)
    except Exception:
        log.exception("topics failed")
        out["topics"] = []
    return jsonify(out)


@bp.post("/api/similar")
def api_similar():
    """Most similar indexed papers to a text, via the production retriever (one row per paper)."""
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
            similar.append({"paper": {"title": r["title"], "url": r["source_url"], "paper_id": r["document_id"]},
                            "similarity": None if sig is None else round(sig, 3), "section": r["section"], "pages": r["pages"]})
            if len(similar) >= 8:
                break
        return jsonify({"similar": similar, "retrieval_mode": res["mode"]})
    except Exception:
        log.exception("similar failed")
        return jsonify({"error": "Similarity search is unavailable"}), 503
