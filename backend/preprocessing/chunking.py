"""Configurable chunking strategies. All strategies emit `Chunk`s that keep provenance.

Size is measured in whitespace-delimited "tokens" (a model-agnostic proxy; real subword
tokens are roughly 1.3x more numerous). The benchmark sweeps sizes empirically, so the proxy
only needs to be consistent across configurations.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from backend.preprocessing.section_extractor import StructuredText

_WORD = re.compile(r"\S+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])")
Span = tuple[int, int]  # char range into StructuredText.text


@dataclass
class Chunk:
    document_id: str
    paper_title: str
    chunk_id: str
    page_start: int
    page_end: int
    section: str
    text: str
    source_url: str = ""
    embedding_model_version: str | None = None   # filled in when embedded
    chunk_index: int = 0
    strategy: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkConfig:
    strategy: str = "recursive"
    size: int = 256
    overlap_pct: float = 0.15
    min_chars: int = 80
    semantic_threshold_percentile: float = 25.0   # semantic: cut at the lowest-similarity gaps

    @property
    def overlap(self) -> int:
        return int(self.size * self.overlap_pct)

    def key(self) -> str:
        return f"{self.strategy}-{self.size}-{int(round(self.overlap_pct * 100))}"


def _n(text: str) -> int:
    return len(_WORD.findall(text))


def fixed_spans(text: str, size: int, overlap: int) -> list[Span]:
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not words:
        return []
    step = max(1, size - overlap)
    spans = []
    for i in range(0, len(words), step):
        j = min(i + size, len(words))
        spans.append((words[i][0], words[j - 1][1]))
        if j == len(words):
            break
    return spans


def sentence_spans_raw(text: str) -> list[Span]:
    spans, start = [], 0
    for m in _SENT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _pack(units: list[Span], text: str, size: int, overlap: int) -> list[Span]:
    """Greedy-pack consecutive units up to `size` tokens, carrying back <= overlap tokens."""
    spans: list[Span] = []
    cur: list[Span] = []
    cur_n = 0
    for u in units:
        un = _n(text[u[0]:u[1]])
        if cur and cur_n + un > size:
            spans.append((cur[0][0], cur[-1][1]))
            keep: list[Span] = []
            kn = 0
            for prev in reversed(cur):
                pn = _n(text[prev[0]:prev[1]])
                if kn + pn > overlap:
                    break
                keep.insert(0, prev)
                kn += pn
            cur, cur_n = keep, kn
        cur.append(u)
        cur_n += un
    if cur:
        spans.append((cur[0][0], cur[-1][1]))
    return spans


def sentence_spans(text: str, size: int, overlap: int) -> list[Span]:
    units: list[Span] = []
    for a, b in sentence_spans_raw(text):
        if _n(text[a:b]) > size:  # over-long "sentences" (tables, bad extraction) are word-split
            units += [(a + x, a + y) for x, y in fixed_spans(text[a:b], size, 0)]
        else:
            units.append((a, b))
    return _pack(units, text, size, overlap)


def recursive_spans(text: str, size: int, overlap: int) -> list[Span]:
    """Split on paragraph -> line -> sentence -> word boundaries until pieces fit, then pack."""
    def split(a: int, b: int, level: int) -> list[Span]:
        seg = text[a:b]
        if _n(seg) <= size:
            return [(a, b)] if seg.strip() else []
        if level == 0:
            seps = list(re.finditer(r"\n\s*\n", seg))
        elif level == 1:
            seps = list(re.finditer(r"\n", seg))
        elif level == 2:
            seps = list(_SENT.finditer(seg))
        else:
            return [(a + x, a + y) for x, y in fixed_spans(seg, size, 0)]
        if not seps:
            return split(a, b, level + 1)
        pieces, last = [], 0
        for m in seps:
            pieces.append((a + last, a + m.start()))
            last = m.end()
        pieces.append((a + last, b))
        res: list[Span] = []
        for x, y in pieces:
            res += split(x, y, level + 1)
        return res

    return _pack(split(0, len(text), 0), text, size, overlap)


def semantic_spans(text: str, size: int, overlap: int, embed: Callable[[list[str]], np.ndarray],
                   percentile: float = 25.0) -> list[Span]:
    """Embed sentences; cut where adjacent-sentence cosine similarity is in the lowest
    `percentile`; then enforce the `size` cap."""
    sents = sentence_spans_raw(text)
    if len(sents) < 3:
        return sentence_spans(text, size, overlap)
    E = np.asarray(embed([text[a:b] for a, b in sents]), dtype="float32")
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sims = (E[:-1] * E[1:]).sum(1)
    cut = np.percentile(sims, percentile)
    groups: list[list[Span]] = []
    cur = [sents[0]]
    for s, sim in zip(sents[1:], sims):
        if sim <= cut:
            groups.append(cur)
            cur = []
        cur.append(s)
    groups.append(cur)
    spans: list[Span] = []
    for g in groups:
        a, b = g[0][0], g[-1][1]
        if _n(text[a:b]) > size:
            spans += [(a + x, a + y) for x, y in sentence_spans(text[a:b], size, overlap)]
        else:
            spans.append((a, b))
    return spans


def section_aware_spans(st: StructuredText, size: int, overlap: int) -> list[Span]:
    """Never crosses a section boundary; inside a section uses recursive splitting."""
    spans: list[Span] = []
    for name, s, e in st.section_ranges():
        spans += [(s + a, s + b) for a, b in recursive_spans(st.text[s:e], size, overlap)]
    return spans


def make_doc_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def chunk_document(st: StructuredText, document_id: str, title: str, source_url: str,
                   cfg: ChunkConfig, embed: Callable[[list[str]], np.ndarray] | None = None,
                   keep_references: bool = False) -> list[Chunk]:
    text = st.text
    if cfg.strategy == "fixed":
        spans = fixed_spans(text, cfg.size, cfg.overlap)
    elif cfg.strategy == "recursive":
        spans = recursive_spans(text, cfg.size, cfg.overlap)
    elif cfg.strategy == "sentence":
        spans = sentence_spans(text, cfg.size, cfg.overlap)
    elif cfg.strategy == "semantic":
        if embed is None:
            raise ValueError("semantic chunking requires an embed function")
        spans = semantic_spans(text, cfg.size, cfg.overlap, embed, cfg.semantic_threshold_percentile)
    elif cfg.strategy == "section":
        spans = section_aware_spans(st, cfg.size, cfg.overlap)
    else:
        raise ValueError(f"unknown chunking strategy {cfg.strategy!r}")

    chunks: list[Chunk] = []
    for a, b in spans:
        raw = text[a:b].strip()
        if len(raw) < cfg.min_chars:
            continue
        ps, pe, sec = st.locate(a, b)
        if sec == "references" and not keep_references:
            continue
        idx = len(chunks)
        chunks.append(Chunk(document_id, title, f"{document_id}:{cfg.key()}:{idx:04d}", ps, pe, sec,
                            raw, source_url, None, idx, cfg.strategy))
    return chunks
