"""Citation parsing and verification.

Support levels are LEXICAL heuristics (share of a sentence's content words, and every number, found in the cited
passage). They flag likely-unsupported claims; they are not calibrated probabilities and cannot prove entailment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.rag.context_builder import ContextItem, content_words

CITE = re.compile(r"\[\s*S(\d+(?:\s*[,;]\s*S?\d+)*)\s*\]", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
_CITE_RUN = r"(?:\s*\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\])*"
_BOUNDARY = re.compile("([.!?])(" + _CITE_RUN + r")(\s+)(?!\[\s*S\d)(?=[A-Z0-9\[\"'(•\-])|\n+")
_ABBREV = re.compile(r"(?:\be\.g|\bi\.e|\bet al|\bFig|\bEq|\bvs|\bcf|\bNo|\bSec|\bTab|\bResp)\.$", re.I)
SUPPORTED, WEAK = 0.5, 0.25


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation (keeping trailing [S#] citations attached) and newlines; does not split
    after common abbreviations such as e.g., i.e., et al., Fig."""
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        if m.group(1) is None:                       # newline boundary
            end = m.start()
        else:
            if not m.group(2) and _ABBREV.search(text[max(0, m.start() - 8):m.start() + 1]):
                continue
            end = m.start(3)
        seg = text[start:end].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def cited_labels(sentence: str) -> list[str]:
    out = []
    for m in CITE.finditer(sentence):
        out += [f"S{n}" for n in re.findall(r"\d+", m.group(1))]
    return list(dict.fromkeys(out))


def strip_citations(s: str) -> str:
    return re.sub(r"\s+", " ", CITE.sub("", s)).strip()


@dataclass
class SentenceAnalysis:
    text: str
    citations: list[str]
    invalid_citations: list[str]
    support: float | None            # best lexical support among cited passages
    status: str                      # supported | weak | unsupported | interpretation | uncited_claim | meta
    supporting_label: str | None = None


@dataclass
class AnswerAnalysis:
    sentences: list[SentenceAnalysis]
    n_claims: int                    # sentences that make factual assertions (not meta/refusal)
    n_cited: int
    n_valid_cited: int
    n_supported: int
    citation_validity: float         # cited sentences whose labels all exist in the context
    cited_fraction: float
    supported_fraction: float        # of cited sentences, share with support >= SUPPORTED
    unsupported: list[str] = field(default_factory=list)
    all_citations_valid: bool = True


def _support(sentence: str, passages: list[str]) -> float:
    words = content_words(strip_citations(sentence))
    if not words:
        return 1.0
    nums = set(NUM.findall(strip_citations(sentence)))
    best = 0.0
    for p in passages:
        pw = content_words(p)
        frac = len(words & pw) / len(words)
        if nums and not nums <= set(NUM.findall(p)):      # an unsupported number caps support
            frac = min(frac, SUPPORTED - 0.01)
        best = max(best, frac)
    return best


def analyze_answer(answer: str, items: list[ContextItem]) -> AnswerAnalysis:
    by_label = {i.label: i for i in items}
    out: list[SentenceAnalysis] = []
    for s in split_sentences(answer):
        plain = strip_citations(s)
        if len(plain.split()) < 4 or plain.endswith(":") or re.match(r"^(extractive answer|note|sources?)\b", plain, re.I):
            out.append(SentenceAnalysis(s, [], [], None, "meta"))
            continue
        labels = cited_labels(s)
        bad = [l for l in labels if l not in by_label]
        good = [l for l in labels if l in by_label]
        interp = bool(re.match(r"^[\W_]*(interpretation|inference|my interpretation)\b", plain, re.I))
        if interp:                       # model-labelled synthesis: reported separately, not scored as a stated fact
            out.append(SentenceAnalysis(s, good, bad, None, "interpretation"))
            continue
        if not labels:
            out.append(SentenceAnalysis(s, [], [], None, "uncited_claim"))
            continue
        sup, best = 0.0, None
        for l in good:
            v = _support(s, [by_label[l].chunk.text])
            if v >= sup:
                sup, best = v, l
        status = "unsupported" if (bad and not good) else "supported" if sup >= SUPPORTED else "weak" if sup >= WEAK else "unsupported"
        out.append(SentenceAnalysis(s, good, bad, round(sup, 3), status, best))
    claims = [x for x in out if x.status not in ("meta", "interpretation")]
    cited = [x for x in claims if x.citations or x.invalid_citations]
    valid = [x for x in cited if not x.invalid_citations]
    sup = [x for x in cited if x.status == "supported"]
    return AnswerAnalysis(out, len(claims), len(cited), len(valid), len(sup),
                          len(valid) / len(cited) if cited else 0.0,
                          len(cited) / len(claims) if claims else 0.0,
                          len(sup) / len(cited) if cited else 0.0,
                          [x.text for x in claims if x.status in ("unsupported", "uncited_claim") or x.invalid_citations],
                          all(not x.invalid_citations for x in out))


def best_quote(passage: str, query_or_claim: str, max_words: int = 60) -> str:
    """Passage sentence(s) with the highest content-word overlap with the claim; used as the highlighted evidence."""
    sents = split_sentences(passage) or [passage]
    cw = content_words(query_or_claim)
    best = max(sents, key=lambda s: len(cw & content_words(s)))
    w = best.split()
    return " ".join(w[:max_words]) + (" ..." if len(w) > max_words else "")


def build_references(items: list[ContextItem], analysis: AnswerAnalysis | None, question: str) -> list[dict]:
    used = {l for s in (analysis.sentences if analysis else []) for l in s.citations}
    refs = []
    for it in items:
        c = it.chunk
        claims = " ".join(s.text for s in (analysis.sentences if analysis else []) if it.label in s.citations)
        refs.append({"label": it.label, "cited": it.label in used, "paper_id": c.document_id, "title": c.paper_title,
                     "pages": [c.page_start, c.page_end], "section": c.section, "url": c.source_url,
                     "chunk_id": c.chunk_id, "retrieval_score": round(it.score, 5), "retrieval_rank": it.rank,
                     "quote": best_quote(c.text, claims or question), "text": c.text})
    return refs
