"""Cross-source deduplication by DOI, arXiv id and normalised title."""
from __future__ import annotations

import hashlib

from backend.ingestion.models import PaperRecord, normalize_title
from backend.preprocessing.section_extractor import is_junk_title


def paper_key(p: PaperRecord) -> str:
    """Stable internal id: prefers arXiv id, then DOI, then normalised title."""
    if p.arxiv_id:
        return "arxiv:" + p.arxiv_id
    if p.doi:
        return "doi:" + hashlib.sha1(p.doi.encode()).hexdigest()[:16]
    return "title:" + hashlib.sha1(normalize_title(p.title).encode()).hexdigest()[:16]


def merge(base: PaperRecord, other: PaperRecord) -> PaperRecord:
    """Fill gaps in `base` from `other`; keeps the larger citation count; records sources."""
    for f in ("abstract", "published", "category", "doi", "arxiv_id", "abs_url", "pdf_url", "venue"):
        if not getattr(base, f) and getattr(other, f):
            setattr(base, f, getattr(other, f))
    if not base.authors and other.authors:
        base.authors = other.authors
    if other.citation_count is not None:
        base.citation_count = max(base.citation_count or 0, other.citation_count)
    for s in other.sources_seen:
        if s not in base.sources_seen:
            base.sources_seen.append(s)
    return base


class Deduplicator:
    """In-memory index; seed it with already-stored papers to avoid re-ingesting them."""

    def __init__(self, existing: list[PaperRecord] | None = None):
        self._by_doi: dict[str, PaperRecord] = {}
        self._by_arxiv: dict[str, PaperRecord] = {}
        self._by_title: dict[str, PaperRecord] = {}
        for p in existing or []:
            self._index(p)

    def _index(self, p: PaperRecord) -> None:
        if not p.paper_id:
            p.paper_id = paper_key(p)
        if p.doi:
            self._by_doi[p.doi] = p
        if p.arxiv_id:
            self._by_arxiv[p.arxiv_id] = p
        nt = normalize_title(p.title)
        if nt and not is_junk_title(p.title):          # template text ("manuscript No.", arXiv stamps) is not an identity
            self._by_title[nt] = p

    def find(self, p: PaperRecord) -> PaperRecord | None:
        if p.doi and p.doi in self._by_doi:
            return self._by_doi[p.doi]
        if p.arxiv_id and p.arxiv_id in self._by_arxiv:
            return self._by_arxiv[p.arxiv_id]
        nt = normalize_title(p.title)
        if is_junk_title(p.title):
            return None
        return self._by_title.get(nt) if len(nt) >= 15 else None   # short titles are too ambiguous

    def add(self, p: PaperRecord) -> tuple[PaperRecord, bool]:
        """Returns (canonical_record, is_new). Duplicates are merged into the canonical record."""
        hit = self.find(p)
        if hit is not None:
            merge(hit, p)
            self._index(hit)     # new identifiers (e.g. DOI learned from the 2nd source)
            return hit, False
        self._index(p)
        return p, True

    def dedupe(self, papers: list[PaperRecord]) -> list[PaperRecord]:
        """Unique, merged records from `papers` that were not already known."""
        fresh: list[PaperRecord] = []
        for p in papers:
            rec, is_new = self.add(p)
            if is_new:
                fresh.append(rec)
        return fresh
