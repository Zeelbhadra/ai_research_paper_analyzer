"""PDF -> cleaned, sectioned, chunked document; plus PDF download and upload ingestion."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.ingestion.deduplication import Deduplicator, paper_key
from backend.ingestion.models import HttpError, PaperRecord, extract_arxiv_id, request_with_retry
from backend.ingestion.pdf_extractor import ExtractionError, extract_pdf
from backend.preprocessing.chunking import Chunk, ChunkConfig, chunk_document
from backend.preprocessing.section_extractor import (StructuredText, build_structured_text,
                                                     extract_named_sections, guess_title)
from backend.preprocessing.text_cleaner import clean_pages

log = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    paper: PaperRecord
    structured: StructuredText
    sections: dict[str, str]
    n_pages: int
    extractor: str
    warnings: list[str] = field(default_factory=list)


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) < 1024:
        raise ExtractionError("file too small to be a PDF")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise ExtractionError("missing %PDF header")


def process_pdf(path: str | Path | None, paper: PaperRecord, data: bytes | None = None,
                backend: str = "pymupdf") -> ProcessedDocument:
    """Extract -> clean -> structure. Raises ExtractionError for unusable files (incl. scanned)."""
    s = get_settings()
    ex = extract_pdf(path, data, backend=backend, min_chars_per_page=s.min_chars_per_page_for_text_pdf)
    if ex.likely_scanned:
        raise ExtractionError("scanned/image-only PDF: no extractable text (OCR not supported)")
    pages = clean_pages(ex.pages)
    st = build_structured_text(pages)
    if len(st.text) < 500:
        raise ExtractionError(f"too little text after cleaning ({len(st.text)} chars)")
    if not paper.title:
        paper.title = guess_title(pages, ex.metadata) or Path(str(path or "upload")).stem
    sections = extract_named_sections(st)
    if not paper.abstract and "abstract" in sections:
        paper.abstract = sections["abstract"][:3000]
    return ProcessedDocument(paper, st, sections, ex.n_pages, ex.extractor, ex.warnings)


def download_pdf(paper: PaperRecord, dest_dir: Path | None = None, session=None) -> Path | None:
    """Download paper.pdf_url to storage. Safe filename from paper_id; validates the payload."""
    s = get_settings()
    if not paper.pdf_url:
        return None
    dest = (dest_dir or s.pdf_dir) / (re.sub(r"[^A-Za-z0-9_.-]", "_", paper.paper_id) + ".pdf")
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    try:
        r = request_with_retry("GET", paper.pdf_url, session=session, max_retries=s.http_max_retries,
                               timeout=max(s.http_timeout, 60), headers={"User-Agent": s.user_agent})
        validate_pdf_bytes(r.content)
    except (HttpError, ExtractionError) as e:
        log.warning("download failed for %s: %s", paper.pdf_url, e)
        return None
    dest.write_bytes(r.content)
    return dest


def ingest_pdf_file(repo: Repository, path: str | Path, paper: PaperRecord | None = None,
                    cfg: ChunkConfig | None = None, embed=None, dedup: Deduplicator | None = None) -> list[Chunk]:
    """Full path for one PDF: extract, structure, chunk, persist. Failures are logged to the DB
    and re-raised as ExtractionError-free return of [] so batch jobs continue."""
    cfg = cfg or ChunkConfig()
    path = Path(path)
    paper = paper or PaperRecord(title="", source="upload")
    paper.pdf_path = str(path)
    if not paper.paper_id:
        # id before title is known: derive from arXiv id/doi if any, else from file content-independent name
        paper.paper_id = paper_key(paper) if (paper.arxiv_id or paper.doi or paper.title) else "file:" + re.sub(r"\W+", "_", path.stem.lower())[:60]
    try:
        doc = process_pdf(path, paper)
    except ExtractionError as e:
        paper.status = "failed"
        repo.upsert_paper(paper)
        repo.log_failure(paper.paper_id, str(path), str(e))
        log.error("extraction failed %s: %s", path, e)
        return []
    except Exception as e:  # unexpected library errors must not kill a batch
        paper.status = "failed"
        repo.upsert_paper(paper)
        repo.log_failure(paper.paper_id, str(path), f"unexpected: {e!r}")
        log.exception("unexpected failure on %s", path)
        return []
    if dedup is not None:
        rec, is_new = dedup.add(doc.paper)
        if not is_new and rec.status == "processed" and repo.count(
                "chunks", "WHERE document_id=? AND config_key=?", (rec.paper_id, cfg.key())):
            log.info("duplicate skipped: %s", doc.paper.title)
            return []
        doc.paper = rec
    p = doc.paper
    if p.paper_id.startswith("file:") and p.title:
        p.paper_id = paper_key(p)
    chunks = chunk_document(doc.structured, p.paper_id, p.title, p.abs_url or p.pdf_url or str(path), cfg, embed=embed)
    p.status = "processed"
    repo.upsert_paper(p)
    repo.replace_chunks(p.paper_id, cfg.key(), chunks)
    return chunks


def ingest_directory(repo: Repository, directory: str | Path, cfg: ChunkConfig | None = None) -> dict:
    stats = {"processed": 0, "failed": 0, "chunks": 0}
    dedup = Deduplicator(repo.list_papers())
    for pdf in sorted(Path(directory).glob("*.pdf")):
        chunks = ingest_pdf_file(repo, pdf, cfg=cfg, dedup=dedup)
        if chunks:
            stats["processed"] += 1
            stats["chunks"] += len(chunks)
        else:
            stats["failed"] += 1
    return stats
