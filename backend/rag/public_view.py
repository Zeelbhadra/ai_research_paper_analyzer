"""Reader-facing view of a RAG response.

The internal response carries retriever names, scores, ranks, gate values, token counts and fallback state. None of that is
meant for readers, so the public API returns only: the answer, numbered sources (paper, pages, section, quote, highlight
offsets) and short human notes. Citation labels S1, S2 are renumbered 1, 2."""
from __future__ import annotations

import re

from backend.explainability.evidence_explainer import explain_answer
from backend.rag.citation_manager import CITE

SUGGESTIONS_PAPER = ["What methods did the authors use?", "What are the main results?", "What are the limitations?", "What do the key terms mean?"]
SUGGESTIONS_ANY = ["Explain that in simpler words.", "Can you give an example?", "What should I read next?"]


def renumber(text: str) -> str:
    """[S1][S3] / [S1, S3] -> [1][3]"""
    return CITE.sub(lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))), text)


def to_public(resp: dict) -> dict:
    expl = explain_answer(resp)
    refs = {r["label"]: r for r in resp["references"]}
    spans: dict[str, list[dict]] = {}
    for s in expl["sentences"]:
        for e in s["evidence"]:
            spans.setdefault(e["label"], []).extend({"start": h["start"], "end": h["end"]} for h in e["highlights"])
    mode = resp["answer_mode"]
    ver = resp.get("verification") or {}
    sentences = []
    for s in expl["sentences"]:
        if s["kind"] == "meta":                                   # internal headings such as the passages preamble
            continue
        kind = {"stated_in_source": "stated", "interpretation": "interpretation", "unverified": "unverified"}.get(s["kind"], "stated")
        sentences.append({"text": renumber(s["text"]), "kind": kind, "sources": sorted({int(e["label"][1:]) for e in s["evidence"]})})
    cited = {n for s in sentences for n in s["sources"]}
    sources = []
    for label, r in refs.items():
        n = int(label[1:])
        sources.append({"n": n, "cited": n in cited, "paper_id": r["paper_id"], "title": r["title"], "pages": r["pages"], "section": r["section"],
                        "url": r["url"], "quote": r["quote"], "text": r["text"], "highlights": spans.get(label, [])})
    sources.sort(key=lambda s: (not s["cited"], s["n"]))
    notes: list[str] = []
    if ver.get("replaced"):
        notes.append("I could not confirm a written answer against the papers, so here are the most relevant passages instead.")
    elif mode == "extractive":
        notes.append("Here are the most relevant passages from the paper.")
    if expl["unsupported_claims"] and mode == "llm":
        notes.append("Part of this answer may go beyond what the papers say; check the sources.")
    for w in resp["query"].get("warnings", []):
        notes.append("Tip: choose a paper first to ask about it specifically." if "this paper" in w else w)
    answer = "I could not find enough information in the papers to answer that. Try rephrasing, or pick a different paper." if mode == "insufficient" else renumber(resp["answer"])
    if mode == "extractive":                                       # drop the internal preamble line, keep the passages
        answer = "\n".join(x for x in answer.splitlines() if not x.lower().startswith("extractive answer")).strip()
    return {"answer": answer, "kind": {"llm": "answer", "extractive": "passages", "insufficient": "none"}.get(mode, "answer"), "sentences": sentences,
            "sources": sources, "notes": notes,
            "suggestions": (SUGGESTIONS_PAPER if resp["query"].get("document_ids") else SUGGESTIONS_ANY) if mode != "insufficient" else [],
            "context": resp["query"]["retrieval_query"]}
