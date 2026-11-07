"""Source-grounded answer explanations: which passage supports which sentence, where, and how strongly.

'Evidence strength' is an ORDINAL, UNCALIBRATED label derived from lexical support. It is not a probability that the
claim is true.
"""
from __future__ import annotations

import re

from backend.rag.context_builder import STOP, content_words

_TOK = re.compile(r"[A-Za-z0-9]+(?:[.\-'][A-Za-z0-9]+)*")


def highlight_spans(claim: str, passage: str, max_gap: int = 1) -> list[dict]:
    """Character spans of `passage` that match `claim`: runs of >=2 matching content words (a stopword gap of up to
    `max_gap` tokens is allowed inside a run), plus standalone numbers that appear in the claim."""
    cw = content_words(claim)
    nums = set(re.findall(r"\d+(?:\.\d+)?", claim))
    toks = [(m.start(), m.end(), m.group().lower()) for m in _TOK.finditer(passage)]
    hit = [(t[2] in cw) or (t[2] in nums) for t in toks]
    spans, i = [], 0
    while i < len(toks):
        if not hit[i]:
            i += 1
            continue
        j, last, gap = i, i, 0
        while j + 1 < len(toks):
            if hit[j + 1]:
                last, j, gap = j + 1, j + 1, 0
            elif toks[j + 1][2] in STOP and gap < max_gap:
                gap, j = gap + 1, j + 1
            else:
                break
        n_hits = sum(hit[i:last + 1])
        if n_hits >= 2 or (n_hits == 1 and toks[i][2] in nums):
            spans.append({"start": toks[i][0], "end": toks[last][1], "text": passage[toks[i][0]:toks[last][1]]})
        i = last + 1
    return spans


def evidence_strength(status: str, support: float | None) -> str:
    if status == "supported":
        return "strong" if (support or 0) >= 0.75 else "moderate"
    if status == "weak":
        return "weak"
    return "none"


def explain_answer(resp: dict) -> dict:
    """resp: the dict returned by RAGPipeline.ask / the final stream event."""
    refs = {r["label"]: r for r in resp["references"]}
    sents, per_paper = [], {}
    for s in resp["sentences"]:
        status = s["status"]
        kind = ("interpretation" if status == "interpretation" else "meta" if status == "meta" else
                "stated_in_source" if status in ("supported", "weak") else "unverified")
        ev = []
        for lab in s["citations"]:
            r = refs.get(lab)
            if not r:
                continue
            ev.append({"label": lab, "paper_id": r["paper_id"], "title": r["title"], "pages": r["pages"], "section": r["section"],
                       "url": r["url"], "supporting": lab == s.get("supporting_label"), "quote": r["quote"],
                       "highlights": highlight_spans(s["text"], r["text"]), "text": r["text"]})
            if kind in ("stated_in_source", "unverified"):
                d = per_paper.setdefault(r["paper_id"], {"title": r["title"], "claims_cited": 0, "claims_supported": 0})
                d["claims_cited"] += 1
                d["claims_supported"] += int(status == "supported" and lab == s.get("supporting_label"))
        sents.append({"text": s["text"], "kind": kind, "status": status, "support": s["support"],
                      "evidence_strength": evidence_strength(status, s["support"]), "evidence": ev,
                      "warning": ("No valid citation: this statement is not backed by a retrieved passage." if status == "uncited_claim" else
                                  "Cites a source that was not retrieved." if s["invalid_citations"] else
                                  "Cited passage shares few of this statement's terms or numbers." if status == "unsupported" else None)})
    total = sum(d["claims_cited"] for d in per_paper.values()) or 1
    for d in per_paper.values():
        d["share_of_cited_claims"] = round(d["claims_cited"] / total, 3)
    return {"answer_mode": resp["answer_mode"], "verification": resp.get("verification"), "sentences": sents,
            "papers": per_paper, "unsupported_claims": (resp.get("citation_quality") or {}).get("unsupported_claims", []),
            "interpretation_sentences": [x["text"] for x in sents if x["kind"] == "interpretation"],
            "note": "Evidence strength is an ordinal lexical-overlap indicator, not a calibrated probability or an entailment check."
                    + (" This is an extractive answer: sentences are quoted verbatim, so 'strong' shows provenance only, "
                       "not that the passage answers the question." if resp["answer_mode"] == "extractive" else "")}
