"""Research library API: browse/filter the stored papers, read extracted content, summaries, keywords, topics."""
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


def _card(p, n_chunks: int) -> dict:
    e = p.enrichment or {}
    return {"paper_id": p.paper_id, "title": p.title, "authors": p.authors, "year": _year(p), "published": p.published, "source": p.source,
            "category": p.category, "topic": _topic(p), "status": p.status, "citation_count": p.citation_count, "doi": p.doi, "arxiv_id": p.arxiv_id,
            "abs_url": p.abs_url, "pdf_url": p.pdf_url, "venue": p.venue, "n_chunks": n_chunks,
            "summary": e.get("summary"), "keywords": [k for k, _ in e.get("keywords", [])], "abstract": p.abstract}


def _chunk_counts() -> dict[str, int]:
    with repo()._lock:
        rows = repo()._conn.execute("SELECT document_id, COUNT(*) FROM chunks WHERE config_key=? GROUP BY document_id", (CFG,)).fetchall()
    return {r[0]: r[1] for r in rows}


@bp.get("/papers")
def papers():
    q = (request.args.get("q") or "").strip().lower()
    author = (request.args.get("author") or "").strip().lower()
    topic = (request.args.get("topic") or "").strip().lower()
    status = request.args.get("status") or ""
    try:
        y0, y1 = int(request.args.get("year_from", 0) or 0), int(request.args.get("year_to", 9999) or 9999)
        limit, offset = max(1, min(int(request.args.get("limit", 24)), 100)), max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "year_from, year_to, limit and offset must be integers"}), 400
    counts = _chunk_counts()
    out = []
    for p in repo().list_papers(status or None):
        y = _year(p)
        if (y0 or y1 != 9999) and (y is None or not (y0 <= y <= y1)):
            continue
        if author and not any(author in a.lower() for a in p.authors):
            continue
        if topic and topic not in (_topic(p) or "").lower():
            continue
        if q and q not in f"{p.title} {p.abstract} {' '.join(p.authors)} {' '.join(k for k, _ in (p.enrichment or {}).get('keywords', []))}".lower():
            continue
        out.append(_card(p, counts.get(p.paper_id, 0)))
    out.sort(key=lambda c: (c["year"] or 0, c["title"] or ""), reverse=True)
    return jsonify({"total": len(out), "offset": offset, "papers": out[offset:offset + limit]})


@bp.get("/paper")
def paper():
    """Detail by ?id= (ids contain ':' and '/'). Includes extracted content grouped by section, with page ranges."""
    pid = request.args.get("id", "")
    p = repo().get_paper(pid)
    if p is None:
        return jsonify({"error": "unknown paper"}), 404
    chunks = repo().get_chunks(CFG, [pid])
    sections: dict[str, list[dict]] = {}
    for c in chunks:
        sections.setdefault(c.section, []).append({"chunk_id": c.chunk_id, "pages": [c.page_start, c.page_end], "text": c.text})
    card = _card(p, len(chunks))
    card["enrichment"] = p.enrichment
    card["sections"] = [{"name": k, "chunks": v} for k, v in sections.items()]
    return jsonify(card)


@bp.get("/facets")
def facets():
    ps = repo().list_papers()
    years = Counter(y for y in (_year(p) for p in ps) if y)
    authors = Counter(a for p in ps for a in p.authors)
    topics = Counter(t for t in (_topic(p) for p in ps) if t)
    return jsonify({"years": sorted(years.items()), "authors": authors.most_common(40), "topics": topics.most_common(30),
                    "sources": Counter(p.source for p in ps).most_common(), "statuses": Counter(p.status for p in ps).most_common()})
