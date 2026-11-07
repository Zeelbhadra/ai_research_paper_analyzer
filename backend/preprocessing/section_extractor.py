"""Heading-based section detection that preserves page provenance.

`build_structured_text` concatenates cleaned pages into one string and records, for every
character range, its page number and canonical section. Chunkers work on the flat string and
map any chunk back to (page_start, page_end, section) via `locate`.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

CANONICAL = [
    ("abstract", r"abstract"),
    ("introduction", r"introduction"),
    ("related_work", r"related\s+work|background|literature\s+review|prior\s+work"),
    ("methodology", r"method(?:s|ology)?|approach|proposed\s+method|model|framework|materials\s+and\s+methods"),
    ("results", r"experiments?|experimental\s+(?:results|setup|evaluation)|results?(?:\s+and\s+discussion)?|evaluation"),
    ("discussion", r"discussion|limitations?|analysis"),
    ("conclusion", r"conclusions?(?:\s+and\s+future\s+work)?|summary|future\s+work|concluding\s+remarks"),
    ("references", r"references|bibliography"),
    ("acknowledgements", r"acknowledge?ments?"),
]
_NUM = r"(?:(?:\d{1,2}|[IVX]{1,5})\.?(?:\d{1,2}\.?)*\s+)?"
_HEADING = [(name, re.compile(rf"^\s*{_NUM}(?:{pat})\s*:?\s*$", re.I)) for name, pat in CANONICAL]
_INLINE_ABSTRACT = re.compile(r"^\s*abstract\s*[-:.—–]\s*\S", re.I)


# A numbered, capitalised, short, unpunctuated line ("3 Swin Transformer") is a section we cannot name,
# but it still ends the previous canonical section instead of being absorbed into it.
_GENERIC = re.compile(r"^\s*(?:\d{1,2}|[IVX]{1,5})\.?\s+[A-Z][A-Za-z0-9 ,\-:&/()']{2,50}$")


def classify_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 60:
        return None
    for name, rx in _HEADING:
        if rx.match(line):
            return name
    if _INLINE_ABSTRACT.match(line):
        return "abstract"
    if _GENERIC.match(line) and len(line.split()) <= 7 and not line.rstrip().endswith((".", ",")):
        return "other"
    return None


@dataclass
class StructuredText:
    text: str
    page_starts: list[int]                       # char offset where each page begins
    page_numbers: list[int]                      # 1-based
    section_starts: list[int] = field(default_factory=list)
    section_names: list[str] = field(default_factory=list)

    def page_at(self, pos: int) -> int:
        i = bisect.bisect_right(self.page_starts, pos) - 1
        return self.page_numbers[max(i, 0)]

    def section_at(self, pos: int) -> str:
        i = bisect.bisect_right(self.section_starts, pos) - 1
        return self.section_names[i] if i >= 0 else "front_matter"

    def locate(self, start: int, end: int) -> tuple[int, int, str]:
        """(page_start, page_end, majority section) for char range [start, end)."""
        last = max(end - 1, start)
        sec_len: dict[str, int] = {}
        bounds = [start] + [s for s in self.section_starts if start < s < last] + [last + 1]
        for a, b in zip(bounds, bounds[1:]):
            sec = self.section_at(a)
            sec_len[sec] = sec_len.get(sec, 0) + (b - a)
        return self.page_at(start), self.page_at(last), max(sec_len, key=sec_len.get)

    def section_ranges(self) -> list[tuple[str, int, int]]:
        if not self.section_starts:
            return [("front_matter", 0, len(self.text))]
        lead = self.section_starts[0] > 0
        starts = ([0] if lead else []) + self.section_starts
        names = (["front_matter"] if lead else []) + self.section_names
        ends = starts[1:] + [len(self.text)]
        return [(n, s, e) for n, s, e in zip(names, starts, ends) if e > s]


def build_structured_text(pages: list[str]) -> StructuredText:
    page_starts, pos = [], 0
    for p in pages:
        page_starts.append(pos)
        pos += len(p) + 1  # pages are joined with "\n"
    text = "\n".join(pages)
    st = StructuredText(text, page_starts, list(range(1, len(pages) + 1)))
    offset = 0
    for line in text.split("\n"):
        name = classify_heading(line)
        if name and (not st.section_names or st.section_names[-1] != name):
            st.section_starts.append(offset)
            st.section_names.append(name)
        offset += len(line) + 1
    return st


JUNK_TITLE = re.compile(
    r"^\W*(title of (the )?paper|paper title|untitled)|^microsoft|^arxiv|^\W*\d|\.(docx?|tex|dvi|pdf)$|proceedings of|^journal of|manuscript no|preprint|submitted to|inserted by the editor|\(\d{4}\)$",
    re.I)


def is_junk_title(t: str) -> bool:
    """PDF-template text, arXiv stamps, venue banners and other strings that are not a paper title."""
    t = (t or "").strip()
    return len(t) < 12 or bool(JUNK_TITLE.search(t))


def looks_like_title(line: str) -> bool:
    """A title is a short phrase, not a sentence: few words, no sentence punctuation inside or at the end, mostly letters."""
    words = line.split()
    letters = sum(ch.isalpha() for ch in line)
    return 2 <= len(words) <= 25 and ". " not in line and not line.endswith((".", ";", ",")) and letters / len(line) >= 0.75


def guess_title(pages: list[str], pdf_meta: dict | None = None) -> str:
    t = ((pdf_meta or {}).get("title") or "").strip()
    if 8 <= len(t) <= 300 and not is_junk_title(t):
        return t
    for line in (pages[0] if pages else "").split("\n")[:12]:
        line = line.strip()
        if 12 <= len(line) <= 200 and classify_heading(line) is None and not is_junk_title(line) and looks_like_title(line):
            return line
    return ""


def extract_named_sections(st: StructuredText) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, s, e in st.section_ranges():
        out.setdefault(name, st.text[s:e].strip())
    return out
