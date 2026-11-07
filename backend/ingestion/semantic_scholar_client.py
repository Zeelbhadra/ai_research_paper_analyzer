"""Semantic Scholar Graph API client with offset pagination, retry and rate-limit handling."""
from __future__ import annotations

import logging
import time
from typing import Iterator

from backend.config.settings import get_settings
from backend.ingestion.models import HttpError, PaperRecord, request_with_retry

log = logging.getLogger(__name__)
BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = ("title,abstract,authors,year,publicationDate,citationCount,url,externalIds,"
          "openAccessPdf,venue,fieldsOfStudy,publicationVenue")


def _to_record(item: dict) -> PaperRecord | None:
    if not item.get("title"):
        return None
    ext = item.get("externalIds") or {}
    pdf = (item.get("openAccessPdf") or {}).get("url")
    arx = ext.get("ArXiv")
    if not pdf and arx:
        pdf = f"https://arxiv.org/pdf/{arx}"
    fos = item.get("fieldsOfStudy") or []
    return PaperRecord(
        title=" ".join(item["title"].split()),
        source="semantic_scholar",
        abstract=" ".join((item.get("abstract") or "").split()),
        authors=[a.get("name", "") for a in item.get("authors") or []],
        published=item.get("publicationDate") or (str(item["year"]) if item.get("year") else None),
        category=fos[0] if fos else None,
        doi=ext.get("DOI"),
        arxiv_id=arx,
        abs_url=item.get("url"),
        pdf_url=pdf,
        citation_count=item.get("citationCount"),
        venue=item.get("venue") or None,
    )


def search_semantic_scholar(query: str, max_results: int = 20, *, page_size: int = 100,
                            min_interval: float = 1.1, session=None, sleep=time.sleep, raise_on_error: bool = False) -> Iterator[PaperRecord]:
    """Yield PaperRecords. Public API is limited (~1 req/s with a key, shared pool without);
    we pace requests by `min_interval` and rely on Retry-After/backoff for 429s."""
    s = get_settings()
    headers = {"User-Agent": s.user_agent}
    if s.semantic_scholar_api_key:
        headers["x-api-key"] = s.semantic_scholar_api_key
    offset, yielded = 0, 0
    while yielded < max_results and offset < 1000:      # API hard limit: offset+limit < 1000
        limit = min(page_size, max_results - yielded)
        try:
            r = request_with_retry("GET", f"{BASE}/paper/search", session=session, headers=headers,
                                   params={"query": query, "offset": offset, "limit": limit, "fields": FIELDS},
                                   max_retries=s.http_max_retries, timeout=s.http_timeout, sleep=sleep)
            payload = r.json()
        except (HttpError, ValueError) as e:
            log.error("Semantic Scholar search %r failed at offset %d: %s", query, offset, e)
            if raise_on_error:
                raise
            return
        data = payload.get("data") or []
        for item in data:
            rec = _to_record(item)
            if rec:
                yield rec
                yielded += 1
                if yielded >= max_results:
                    return
        if not data or "next" not in payload:
            return
        offset = payload["next"]
        sleep(min_interval)
