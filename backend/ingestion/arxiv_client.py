"""arXiv discovery client (wraps the `arxiv` package; paginated, retrying, rate-limited)."""
from __future__ import annotations

import logging
from typing import Callable, Iterator

from backend.ingestion.models import PaperRecord, extract_arxiv_id

log = logging.getLogger(__name__)


def _to_record(r) -> PaperRecord:
    aid = extract_arxiv_id(r.entry_id)
    return PaperRecord(
        title=" ".join((r.title or "").split()),
        source="arxiv",
        abstract=" ".join((r.summary or "").split()),
        authors=[a.name for a in r.authors],
        published=r.published.date().isoformat() if r.published else None,
        category=r.primary_category,
        doi=getattr(r, "doi", None),
        arxiv_id=aid,
        abs_url=r.entry_id,
        pdf_url=r.pdf_url or (f"https://arxiv.org/pdf/{aid}" if aid else None),
        venue=getattr(r, "journal_ref", None),
    )


def search_arxiv(query: str, max_results: int = 20, *, page_size: int = 50, delay_seconds: float = 3.0,
                 num_retries: int = 4, client_factory: Callable | None = None, raise_on_error: bool = False) -> Iterator[PaperRecord]:
    """Yield PaperRecords. arXiv asks for >=3s between requests; the client enforces `delay_seconds`
    and retries transient failures. Errors after retries are logged and end the iteration
    (results already yielded are kept)."""
    import arxiv
    client = (client_factory or arxiv.Client)(page_size=min(page_size, max_results),
                                               delay_seconds=delay_seconds, num_retries=num_retries)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    try:
        for r in client.results(search):
            yield _to_record(r)
    except Exception as e:
        log.error("arXiv search %r failed: %s", query, e)
        if raise_on_error:
            raise
