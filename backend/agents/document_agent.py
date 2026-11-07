"""Document Intelligence Agent: extract, section, chunk, enrich (keywords, summary, topics) and refresh indexes."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.agents.orchestrator import Agent, AgentError
from backend.agents.services import Services
from backend.ingestion.document_processor import ingest_pdf_file
from backend.preprocessing.chunking import ChunkConfig

log = logging.getLogger(__name__)


_AFFIL = re.compile(r"@|(University|Universit\w+|Institute|Department|Dept\.|Laboratory|School of|Faculty|Technische|Corresponding)", re.I)


def clean_for_enrichment(text: str) -> str:
    """Drop author/affiliation/e-mail sentences that otherwise dominate extractive summaries and keywords of PDFs."""
    from backend.rag.citation_manager import split_sentences
    kept = [s for s in split_sentences(text) if not _AFFIL.search(s)]
    out = " ".join(kept)
    return out if len(out) > 200 else text


def parse_cfg_key(key: str) -> ChunkConfig:
    """'recursive-256-15' -> ChunkConfig('recursive', 256, 0.15)."""
    try:
        strategy, size, ov = key.split("-")
        return ChunkConfig(strategy, int(size), int(ov) / 100)
    except Exception as e:
        raise AgentError(f"bad chunk config key {key!r}") from e


class DocumentAgent(Agent):
    name = "document"

    def __init__(self, sv: Services):
        self.sv = sv

    def actions(self):
        return {"process": self.process, "process_files": self.process_files, "update_indexes": self.update_indexes}

    # ---- enrichment: reuses the original project's modules (kept, not rewritten) ----
    def _enrich(self, text: str, which: dict) -> dict:
        out: dict = {}
        text = clean_for_enrichment(text)
        if not text.strip():
            return out
        if which.get("keywords"):
            try:
                from src.keyword_extractor import extract_keywords
                out["keywords"] = [(k, round(float(s), 4)) for k, s in extract_keywords(text[:4000], top_n=8)]
            except Exception as e:
                out["keywords_error"] = str(e)[:120]
        if which.get("summary"):
            try:
                from src.summarizer import _generate_tfidf_summary          # extractive; Pegasus is opt-in (heavy)
                out["summary"] = _generate_tfidf_summary(text[:6000])
                out["summary_method"] = "tfidf-extractive"
            except Exception as e:
                out["summary_error"] = str(e)[:120]
        if which.get("topics"):
            try:
                from src.topic_classifier import classify_topic
                out["topics"] = classify_topic(text[:2000], which.get("topic_labels") or None, top_n=3)
            except Exception as e:
                out["topics_error"] = str(e)[:120]
        return out

    def process(self, paper_ids: list[str], chunk_cfg: str | None = None, enrich: dict | None = None) -> dict:
        cfg_key = chunk_cfg or self.sv.cfg_key
        cfg = parse_cfg_key(cfg_key)
        which = self.sv.automation.get("enrichment", {}) if enrich is None else enrich
        done, failed, n_chunks = [], [], 0
        for pid in paper_ids:
            rec = self.sv.repo.get_paper(pid)
            if rec is None or not rec.pdf_path or not Path(rec.pdf_path).exists():
                failed.append({"paper_id": pid, "reason": "no local PDF"})
                continue
            chunks = ingest_pdf_file(self.sv.repo, rec.pdf_path, rec, cfg)
            if not chunks:
                reason = next((f["reason"] for f in self.sv.repo.failures() if f["paper_id"] == pid), "extraction failed")
                failed.append({"paper_id": pid, "reason": reason})
                continue
            rec = self.sv.repo.get_paper(pid)
            text = rec.abstract or " ".join(c.text for c in chunks[:6])
            rec.enrichment = self._enrich(text, which)
            self.sv.repo.upsert_paper(rec)
            done.append(pid)
            n_chunks += len(chunks)
        if paper_ids and not done:
            raise AgentError(f"no document could be processed: {failed[:3]}")
        return {"processed": done, "failed": failed, "n_chunks": n_chunks, "chunk_cfg": cfg_key}

    def process_files(self, paths: list[str], chunk_cfg: str | None = None) -> dict:
        """Manual uploads: same pipeline, papers identified by title/content."""
        from backend.ingestion.deduplication import Deduplicator
        cfg = parse_cfg_key(chunk_cfg or self.sv.cfg_key)
        dedup = Deduplicator(self.sv.repo.list_papers())
        ids, failed = [], []
        for p in paths:
            chunks = ingest_pdf_file(self.sv.repo, p, None, cfg, dedup=dedup)
            if chunks:
                ids.append(chunks[0].document_id)
            else:
                failed.append({"path": str(p), "reason": "failed or duplicate"})
        return {"processed": ids, "failed": failed}

    def update_indexes(self, chunk_cfg: str | None = None) -> dict:
        """Rebuild the indexes the production model needs over the current chunk set. Dense embeddings are cached per chunk
        text, so only new chunks are embedded."""
        from backend.retrieval.production import components_of
        from backend.embeddings.base import make_embedder
        cfg_key = chunk_cfg or self.sv.cfg_key
        chunks = self.sv.repo.get_chunks(cfg_key)
        if not chunks:
            raise AgentError(f"no chunks stored for {cfg_key}")
        prod = self.sv.registry.production
        name = prod["config"].get("_name", prod["name"]) if prod else "tfidf"
        built, embedded = [], 0
        for comp in components_of(name):
            if comp == "tfidf":
                self.sv.mgr.build_tfidf_index(chunks, cfg_key)
                built.append("tfidf")
            else:
                e = make_embedder(comp)
                e.load()
                _, st = self.sv.mgr.build_dense_index(e, chunks, cfg_key)
                built.append(comp)
                embedded += st.n_embedded
        return {"production": name, "indexes_built": built, "n_chunks": len(chunks), "newly_embedded_chunks": embedded}
