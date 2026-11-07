"""Turn ranked chunks into a bounded, de-duplicated, ordered, labelled context ([S1], [S2], ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.preprocessing.chunking import Chunk

_W = re.compile(r"[a-z0-9]+")
STOP = frozenset("""a an the of to in on for and or is are was were be been by with as at from that this it its how what which does do
did using used use can could should would about into than then there their they them these those such not also more most we our
paper papers study work""".split())


def content_words(t: str) -> set[str]:
    return {w for w in _W.findall(t.lower()) if len(w) > 2 and w not in STOP}


def shingles(t: str, n: int = 5) -> set[tuple]:
    w = _W.findall(t.lower())
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


@dataclass
class ContextItem:
    label: str                 # 'S1'
    chunk: Chunk
    score: float
    rank: int                  # rank in the retrieval list


@dataclass
class ContextResult:
    items: list[ContextItem]
    dropped_duplicates: int
    truncated: int
    coverage: float            # share of query content words found in the selected context (evidence-strength signal)
    top_score: float


def query_coverage(query: str, texts: list[str]) -> float:
    q = content_words(query)
    if not q:
        return 0.0
    have = set().union(*[content_words(t) for t in texts]) if texts else set()
    return len(q & have) / len(q)


def build_context(query: str, ranked: list[tuple[Chunk, float, int]], max_chunks: int = 6, max_words: int = 1600,
                  per_doc_cap: int | None = 3, dup_threshold: float = 0.6, min_docs: int = 1, max_docs: int | None = None) -> ContextResult:
    """ranked: [(chunk, score, rank)] best first.
    Steps: drop near-duplicates (5-gram Jaccard) -> select by rank with a per-document cap (so one paper cannot crowd
    out a comparison) -> enforce a word budget -> order by document (best-ranked first) then page/position."""
    kept: list[tuple[Chunk, float, int]] = []
    sh: list[set] = []
    dups = 0
    for ch, sc, rk in ranked:
        s = shingles(ch.text)
        if any(s and o and len(s & o) / len(s | o) >= dup_threshold for o in sh):
            dups += 1
            continue
        kept.append((ch, sc, rk))
        sh.append(s)
    if max_docs:        # paper-level diversification: only the best `max_docs` papers, taken round-robin (best chunk of each first)
        docs: list[str] = []
        for ch, _, _ in kept:
            if ch.document_id not in docs and len(docs) < max_docs:
                docs.append(ch.document_id)
        by = {d: [t for t in kept if t[0].document_id == d] for d in docs}
        kept = []
        while any(by.values()):
            for d in docs:
                if by[d]:
                    kept.append(by[d].pop(0))
    chosen, per_doc, words, trunc = [], {}, 0, 0
    for ch, sc, rk in kept:
        if len(chosen) >= max_chunks:
            trunc += 1
            continue
        if per_doc_cap and per_doc.get(ch.document_id, 0) >= per_doc_cap:
            trunc += 1
            continue
        n = len(ch.text.split())
        if chosen and words + n > max_words:
            trunc += 1
            continue
        chosen.append((ch, sc, rk))
        per_doc[ch.document_id] = per_doc.get(ch.document_id, 0) + 1
        words += n
    doc_order: dict[str, int] = {}
    for ch, _, rk in chosen:
        doc_order.setdefault(ch.document_id, rk)
    chosen.sort(key=lambda t: (doc_order[t[0].document_id], t[0].document_id, t[0].page_start, t[0].chunk_index))
    items = [ContextItem(f"S{i + 1}", ch, sc, rk) for i, (ch, sc, rk) in enumerate(chosen)]
    return ContextResult(items, dups, trunc, query_coverage(query, [i.chunk.text for i in items]),
                         max((i.score for i in items), default=0.0))
