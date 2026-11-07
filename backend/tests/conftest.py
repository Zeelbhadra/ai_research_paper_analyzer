import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SECTION_TEXT = {
    "Abstract": "We study retrieval of scientific text. Our method improves recall on a new benchmark. " * 4,
    "1 Introduction": "Scientific literature grows quickly and researchers need tools to search it. " * 12,
    "2 Related Work": "Prior work on dense retrieval includes SBERT, E5 and BGE models. " * 12,
    "3 Methodology": "We split documents into chunks and embed each chunk with a transformer encoder. " * 14,
    "4 Results": "Table 1 shows that our approach reaches 0.82 nDCG at ten on the test set. " * 12,
    "5 Conclusion": "We presented a retrieval pipeline and discussed its limitations and future work. " * 8,
    "References": "[1] A. Author. A paper about things. Journal, 2020. " * 6,
}


def make_pdf_bytes(title="A Study of Chunking for Scientific Retrieval", pages=6, header="Journal of Tests") -> bytes:
    """Build a small text PDF with headings, running header/footer and page numbers."""
    import fitz
    doc = fitz.open()
    flow = [title] + [x for h, body in SECTION_TEXT.items() for x in (h, body)]
    # distribute flow items across pages
    per_page = max(1, len(flow) // pages + 1)
    for pi in range(pages):
        page = doc.new_page()
        page.insert_text((72, 40), header, fontsize=9)
        y = 90
        for item in flow[pi * per_page:(pi + 1) * per_page]:
            rect = fitz.Rect(72, y, 520, y + 260)
            used = page.insert_textbox(rect, item, fontsize=11)
            y += 24 if len(item) < 60 else 200
        page.insert_text((290, 800), str(pi + 1), fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes():
    return make_pdf_bytes()


@pytest.fixture
def pdf_file(tmp_path, pdf_bytes):
    p = tmp_path / "paper.pdf"
    p.write_bytes(pdf_bytes)
    return p


@pytest.fixture
def repo(tmp_path):
    from backend.database.repositories import Repository
    return Repository(tmp_path / "t.sqlite")


def make_topic_pdf(title: str, sentence: str, pages: int = 4) -> bytes:
    """A small text PDF whose body is about one topic (distinct vocabulary per paper) with real section headings."""
    import fitz
    doc = fitz.open()
    heads = ["Abstract", "1 Introduction", "2 Methodology", "3 Results", "4 Conclusion", "References"]
    for pi in range(pages):
        page = doc.new_page()
        page.insert_text((72, 40), "Running Header Of The Journal", fontsize=9)
        y = 80
        if pi == 0:
            page.insert_textbox(fitz.Rect(72, y, 520, y + 40), title, fontsize=15)
            y += 50
        page.insert_text((72, y), heads[min(pi, len(heads) - 1)], fontsize=13)
        y += 26
        page.insert_textbox(fitz.Rect(72, y, 520, 760), (sentence + " ") * 14, fontsize=10.5)
        page.insert_text((290, 810), str(pi + 1), fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data
