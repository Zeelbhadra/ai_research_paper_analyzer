"""Library API: browse the papers the assistant can discuss; add a discovered paper so it can be read and asked about.

Reader-facing only: no chunk counts, internal statuses, chunk ids or model details."""
from __future__ import annotations

import logging
import re
from collections import Counter

from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)
bp = Blueprint("library", __name__, url_prefix="/api/library")
CFG = "recursive-256-15"
_repo = None


def repo():
    global _repo
    if _repo is None:
        from backend.config.settings import get_settings
        from backend.database.repositories import Repository
        _repo = Repository(get_settings().db_path)
    return _repo


def _year(p) -> int | None:
    m = re.match(r"(\d{4})", p.published or "")
    return int(m.group(1)) if m else None


def _topic(p) -> str | None:
    t = (p.enrichment or {}).get("topics")
    if t:
        return t[0]["label"]
    return p.category


def _card(p) -> dict:
    e = p.enrichment or {}
    return {"paper_id": p.paper_id, "title": p.title, "authors": p.authors, "year": _year(p), "venue": p.venue, "topic": _topic(p),
            "citation_count": p.citation_count, "doi": p.doi, "abs_url": p.abs_url, "pdf_url": p.pdf_url,
            "summary": e.get("summary"), "keywords": [k for k, _ in e.get("keywords", [])], "abstract": p.abstract}


@bp.get("/papers")
def papers():
    q = (request.args.get("q") or "").strip().lower()
    author = (request.args.get("author") or "").strip().lower()
    topic = (request.args.get("topic") or "").strip().lower()
    try:
        y0, y1 = int(request.args.get("year_from", 0) or 0), int(request.args.get("year_to", 9999) or 9999)
        limit, offset = max(1, min(int(request.args.get("limit", 24)), 100)), max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "year_from, year_to, limit and offset must be integers"}), 400
    out = []
    for p in repo().list_papers("processed"):
        y = _year(p)
        if (y0 or y1 != 9999) and (y is None or not (y0 <= y <= y1)):
            continue
        if author and not any(author in a.lower() for a in p.authors):
            continue
        if topic and topic not in (_topic(p) or "").lower():
            continue
        if q and q not in f"{p.title} {p.abstract} {' '.join(p.authors)} {' '.join(k for k, _ in (p.enrichment or {}).get('keywords', []))}".lower():
            continue
        out.append(_card(p))
    out.sort(key=lambda c: (c["year"] or 0, c["title"] or ""), reverse=True)
    return jsonify({"total": len(out), "offset": offset, "papers": out[offset:offset + limit]})


@bp.get("/paper")
def paper():
    """Detail by ?id= (ids contain ':' and '/'): summary, keywords, related topics and the paper's text by section with pages."""
    pid = request.args.get("id", "")
    p = repo().get_paper(pid)
    if p is None or p.status != "processed":
        return jsonify({"error": "unknown paper"}), 404
    sections: dict[str, list[dict]] = {}
    for c in repo().get_chunks(CFG, [pid]):
        sections.setdefault(c.section, []).append({"pages": [c.page_start, c.page_end], "text": c.text})
    card = _card(p)
    card["topics"] = [{"label": t["label"], "score": t["score"]} for t in (p.enrichment or {}).get("topics", [])]
    card["sections"] = [{"name": k, "chunks": v} for k, v in sections.items()]
    return jsonify(card)


@bp.get("/facets")
def facets():
    ps = repo().list_papers("processed")
    years = Counter(y for y in (_year(p) for p in ps) if y)
    authors = Counter(a for p in ps for a in p.authors)
    topics = Counter(t for t in (_topic(p) for p in ps) if t)
    return jsonify({"years": sorted(years.items()), "authors": authors.most_common(40), "topics": topics.most_common(30)})


# ---------------- add a discovered paper to the library ----------------
def _find_existing(rec):
    from backend.ingestion.deduplication import Deduplicator
    return Deduplicator(repo().list_papers()).find(rec)


@bp.post("/add")
def add():
    """Body: the paper as returned by /api/search ({title, abstract, authors, url, pdf_url, source, year}). Starts downloading, reading and
    indexing it in the background; poll /api/library/add/status?id=. Returns 200 with the id if it is already in the library."""
    from backend.ingestion.models import PaperRecord
    b = request.get_json(silent=True) or {}
    title = str(b.get("title") or "").strip()
    if len(title) < 5:
        return jsonify({"error": "A title is required."}), 400
    rec = PaperRecord(title=title, source=str(b.get("source") or "web")[:30], abstract=str(b.get("abstract") or "")[:5000],
                      authors=[str(a)[:120] for a in (b.get("authors") or [])][:50], published=str(b.get("year") or "") or None,
                      doi=b.get("doi"), arxiv_id=None, abs_url=str(b.get("url") or "")[:500] or None, pdf_url=str(b.get("pdf_url") or "")[:500] or None,
                      venue=b.get("venue"), citation_count=b.get("citation_count") if isinstance(b.get("citation_count"), int) else None)
    if not rec.pdf_url:
        return jsonify({"error": "This paper has no open-access PDF, so it cannot be added. You can still analyze its abstract."}), 422
    hit = _find_existing(rec)
    if hit is not None and hit.status == "processed":
        return jsonify({"status": "ready", "paper_id": hit.paper_id})
    from backend.api.routes.pipeline import get_system
    from backend.agents.orchestrator import AlreadyRunning
    _, orch, _ = get_system()
    try:
        orch.submit("ingest_paper", {"paper": {**rec.to_dict(), "paper_id": hit.paper_id if hit else ""}})
    except AlreadyRunning:                       # one paper is read at a time; if it is this one, just keep waiting
        if hit is not None and hit.status in ("discovered", "downloaded"):
            return jsonify({"status": "adding", "paper_id": hit.paper_id}), 202
        return jsonify({"error": "Another paper is being added right now. Please try again in a minute."}), 409
    from backend.ingestion.deduplication import paper_key
    return jsonify({"status": "adding", "paper_id": (hit.paper_id if hit else paper_key(rec))}), 202


@bp.get("/add/status")
def add_status():
    pid = request.args.get("id", "")
    p = repo().get_paper(pid)
    if p is None:
        return jsonify({"status": "unknown"}), 404
    state = {"processed": "ready", "failed": "failed"}.get(p.status, "adding")
    msg = {"ready": "Ready: you can now ask the assistant about this paper.", "failed": "This paper could not be read (the PDF may be scanned or unavailable).",
           "adding": "Reading the paper..."}[state]
    return jsonify({"status": state, "paper_id": pid, "message": msg})
