"""Compare PDF extractors on a directory of PDFs: speed, yield and section-detection success.

    python -m backend.ingestion.extractor_compare data  [--out storage/extractor_comparison.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from backend.ingestion.pdf_extractor import BACKENDS, ExtractionError, extract_pdf
from backend.preprocessing.section_extractor import build_structured_text
from backend.preprocessing.text_cleaner import clean_pages


def compare(directory: str | Path) -> dict:
    pdfs = sorted(Path(directory).glob("*.pdf"))
    res = {}
    for name in BACKENDS:
        secs = chars = fails = pages = 0
        t0 = time.perf_counter()
        for p in pdfs:
            try:
                ex = extract_pdf(p, backend=name)
                if ex.extractor != name:          # silently fell back; count as failure of `name`
                    fails += 1
                    continue
            except ExtractionError:
                fails += 1
                continue
            st = build_structured_text(clean_pages(ex.pages))
            secs += len({n for n in st.section_names} & {"abstract", "introduction", "conclusion", "references"})
            chars += ex.n_chars
            pages += ex.n_pages
        dt = time.perf_counter() - t0
        ok = max(len(pdfs) - fails, 1)
        res[name] = {"files": len(pdfs), "failures": fails, "total_seconds": round(dt, 2),
                     "sec_per_page": round(dt / max(pages, 1), 4), "total_chars": chars,
                     "avg_core_sections_found_of_4": round(secs / ok, 2)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--out", default="storage/extractor_comparison.json")
    a = ap.parse_args()
    r = compare(a.directory)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
