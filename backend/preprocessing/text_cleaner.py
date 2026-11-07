"""Text cleaning: dehyphenation, ligatures, header/footer removal, normalisation."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_LIGS = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": ""}
_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
_DIGITS = re.compile(r"\d+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGS.items():
        text = text.replace(k, v)
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _key(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())


def remove_headers_footers(pages: list[str], edge_lines: int = 3, min_frac: float = 0.4) -> list[str]:
    """Drop lines in the first/last `edge_lines` of a page that recur (digits masked) on
    >= min_frac of pages, plus bare page numbers. Needs >=4 pages to infer repetition."""
    split = [p.split("\n") for p in pages]
    repeated: set[str] = set()
    if len(pages) >= 4:
        counts: Counter = Counter()
        for lines in split:
            edge = [l for l in lines if l.strip()]
            cand = edge[:edge_lines] + edge[-edge_lines:]
            counts.update({_key(l) for l in cand if len(l.strip()) > 2})
        repeated = {k for k, c in counts.items() if c / len(pages) >= min_frac}
    out = []
    for lines in split:
        nz = [i for i, l in enumerate(lines) if l.strip()]
        edge_idx = set(nz[:edge_lines] + nz[-edge_lines:])
        kept = []
        for i, l in enumerate(lines):
            if _PAGE_NUM.match(l):
                continue
            if i in edge_idx and _key(l) in repeated:
                continue
            kept.append(l)
        out.append("\n".join(kept))
    return out


def clean_pages(pages: list[str]) -> list[str]:
    pages = [normalize(p) for p in pages]
    pages = remove_headers_footers(pages)
    return [normalize(p) for p in pages]
