"""Add keywords / extractive summary / topics to papers that have none (e.g. the PDFs that were ingested before enrichment existed).

    python -m backend.agents.backfill_enrichment [--force]
Uses the project's original modules (KeyBERT, TF-IDF extractive summary, BART-MNLI zero-shot) through DocumentAgent._enrich.
"""
from __future__ import annotations

import argparse
import time

from backend.agents.document_agent import DocumentAgent
from backend.agents.services import Services


def pick_text(chunks) -> str:
    """Most informative text for topic/keyword/summary: the abstract section if present, else the introduction, else the
    start of the document. (Front matter such as author lists and affiliations misleads zero-shot topic classification.)"""
    for section in ("abstract", "introduction"):
        sel = [c.text for c in chunks if c.section == section][:2]
        if sel:
            return " ".join(sel)
    return " ".join(c.text for c in chunks[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if enrichment exists")
    ap.add_argument("--parts", default="keywords,summary,topics", help="comma list of parts to recompute; others are kept")
    a = ap.parse_args()
    sv = Services.create()
    agent = DocumentAgent(sv)
    todo = [p for p in sv.repo.list_papers("processed") if a.force or not p.enrichment.get("summary")]
    print(f"{len(todo)} papers to enrich", flush=True)
    for i, p in enumerate(todo, 1):
        chunks = sv.repo.get_chunks(sv.cfg_key, [p.paper_id])
        text = p.abstract or pick_text(chunks)
        t = time.perf_counter()
        which = {**sv.automation["enrichment"], **{k: (k in a.parts.split(",")) for k in ("keywords", "summary", "topics")}}
        p.enrichment = {**p.enrichment, **agent._enrich(text, which)}
        sv.repo.upsert_paper(p)
        print(f"[{i}/{len(todo)}] {time.perf_counter() - t:4.1f}s {p.title[:60]} -> {(p.enrichment.get('topics') or [{}])[0].get('label')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
