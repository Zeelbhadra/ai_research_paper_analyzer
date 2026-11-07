"""Research Discovery Agent: find papers, avoid duplicates, download open-access PDFs, track status."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from backend.agents.orchestrator import Agent, AgentError, TransientError
from backend.agents.services import Services
from backend.ingestion.arxiv_client import search_arxiv
from backend.ingestion.deduplication import Deduplicator, paper_key
from backend.ingestion.document_processor import download_pdf
from backend.ingestion.semantic_scholar_client import search_semantic_scholar

log = logging.getLogger(__name__)


class DiscoveryAgent(Agent):
    name = "discovery"

    def __init__(self, sv: Services, arxiv_search=search_arxiv, s2_search=search_semantic_scholar, downloader=download_pdf,
                 sleep=time.sleep):
        self.sv, self._sources, self._download, self._sleep = sv, {"arxiv": arxiv_search, "semantic_scholar": s2_search}, downloader, sleep

    def actions(self):
        return {"search": self.search, "download": self.download, "register": self.register}

    def search(self, topic: str, max_results: int = 10, sources: list[str] | None = None) -> dict:
        if not topic or not topic.strip():
            raise AgentError("topic is required")
        sources = sources or list(self._sources)
        found, status = [], {}
        for src in sources:
            fn = self._sources.get(src)
            if fn is None:
                status[src] = {"ok": False, "error": "unknown source"}
                continue
            try:
                recs = list(fn(topic, max_results, raise_on_error=True))
                found += recs
                status[src] = {"ok": True, "n": len(recs)}
            except Exception as e:                     # one failing source must not sink the others
                status[src] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
                log.warning("discovery source %s failed: %s", src, e)
        if not any(v["ok"] for v in status.values()):
            raise TransientError(f"all discovery sources failed: {status}")
        known = Deduplicator(self.sv.repo.list_papers())
        fresh = known.dedupe(found)
        for p in fresh:
            p.paper_id = p.paper_id or paper_key(p)
            p.status = "discovered"
            self.sv.repo.upsert_paper(p)
        self.sv.events.emit("discovery", topic=topic, found=len(found), new=len(fresh), sources=status)
        return {"topic": topic, "found": len(found), "new": len(fresh), "duplicates": len(found) - len(fresh),
                "new_paper_ids": [p.paper_id for p in fresh], "source_status": status, "partial": not all(v["ok"] for v in status.values())}

    def register(self, paper: dict) -> dict:
        """Store one paper chosen by a user (from search results) so it can be downloaded and read. Re-uses the id of a known duplicate."""
        from backend.ingestion.models import PaperRecord
        rec = PaperRecord(**{k: v for k, v in paper.items() if k in PaperRecord.__dataclass_fields__})
        hit = Deduplicator(self.sv.repo.list_papers()).find(rec)
        if hit is not None:
            if hit.status != "processed":                       # retry a failed/incomplete earlier attempt
                hit.pdf_url = hit.pdf_url or rec.pdf_url
                hit.status = "discovered"
                self.sv.repo.upsert_paper(hit)
            return {"paper_ids": [hit.paper_id], "already_known": True}
        rec.paper_id, rec.status = paper_key(rec), "discovered"
        self.sv.repo.upsert_paper(rec)
        return {"paper_ids": [rec.paper_id], "already_known": False}

    def download(self, paper_ids: list[str], delay_s: float | None = None) -> dict:
        """Downloads only the open-access PDF URL a source provided; never scrapes or bypasses access controls."""
        delay = self.sv.automation["discovery"].get("download_delay_seconds", 1.0) if delay_s is None else delay_s
        got, present, failed = [], [], []
        for i, pid in enumerate(paper_ids):
            rec = self.sv.repo.get_paper(pid)
            if rec is None:
                failed.append({"paper_id": pid, "reason": "unknown paper"})
                continue
            if rec.pdf_path and Path(rec.pdf_path).exists():
                present.append(pid)
                continue
            if not rec.pdf_url:
                failed.append({"paper_id": pid, "reason": "no open-access PDF URL"})
                continue
            if i and delay:
                self._sleep(delay)
            path = self._download(rec)
            if path is None:
                rec.status = "failed"
                self.sv.repo.upsert_paper(rec)
                failed.append({"paper_id": pid, "reason": "download or PDF validation failed"})
                continue
            rec.pdf_path, rec.status = str(path), "downloaded"
            self.sv.repo.upsert_paper(rec)
            got.append(pid)
        if paper_ids and not got and not present:
            raise AgentError(f"none of the {len(paper_ids)} PDFs could be downloaded: {failed[:3]}")
        return {"downloaded": got, "already_present": present, "failed": failed, "ready_ids": got + present}
