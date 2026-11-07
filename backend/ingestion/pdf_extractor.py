"""Page-level PDF extraction with PyMuPDF (primary) and pdfplumber (fallback)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class ExtractionError(Exception):
    pass


@dataclass
class ExtractedPDF:
    path: str
    pages: list[str]                    # raw text per page, index 0 = page 1
    metadata: dict = field(default_factory=dict)
    extractor: str = ""
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    likely_scanned: bool = False

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p) for p in self.pages)


def _pymupdf(path: str | None, data: bytes | None) -> tuple[list[str], dict]:
    import fitz
    doc = fitz.open(path) if data is None else fitz.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass:
            raise ExtractionError("PDF is encrypted")
        pages = [p.get_text("text") for p in doc]
        return pages, {k: v for k, v in (doc.metadata or {}).items() if v}
    finally:
        doc.close()


def _pdfplumber(path: str | None, data: bytes | None) -> tuple[list[str], dict]:
    import io
    import pdfplumber
    src = path if data is None else io.BytesIO(data)
    with pdfplumber.open(src) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages], dict(pdf.metadata or {})


BACKENDS = {"pymupdf": _pymupdf, "pdfplumber": _pdfplumber}


def extract_pdf(path: str | Path | None = None, data: bytes | None = None,
                backend: str = "pymupdf", min_chars_per_page: int = 100) -> ExtractedPDF:
    """Extract per-page text. Falls back to the other backend on failure.
    Flags (does not OCR) scanned PDFs: avg chars/page below `min_chars_per_page`."""
    if path is None and data is None:
        raise ValueError("path or data required")
    order = [backend] + [b for b in BACKENDS if b != backend]
    errors: list[str] = []
    for name in order:
        t0 = time.perf_counter()
        try:
            pages, meta = BACKENDS[name](str(path) if path else None, data)
        except ExtractionError:
            raise
        except Exception as e:  # corrupted file, missing library, ...
            msg = f"{name} failed on {path or '<bytes>'}: {e}"
            log.warning(msg)
            errors.append(msg)
            continue
        out = ExtractedPDF(str(path or "<bytes>"), pages, meta, name, time.perf_counter() - t0, errors[:])
        if not pages:
            errors.append(f"{name}: no pages recovered")
            continue
        if out.n_chars / len(pages) < min_chars_per_page:
            out.likely_scanned = True
            out.warnings.append("likely scanned / image-only PDF (OCR not performed)")
        return out
    raise ExtractionError("; ".join(errors) or "no extractor produced pages")
