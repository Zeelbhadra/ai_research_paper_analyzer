"""End-to-end RAG: query processing -> retrieval (with fallback) -> optional rerank -> context -> generation ->
citation verification -> structured, source-grounded response."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from backend.rag.citation_manager import AnswerAnalysis, analyze_answer, build_references
from backend.rag.context_builder import ContextResult, build_context
from backend.rag.generator import INSUFFICIENT, GenerationResult, extractive_answer, generate, stream_generate
from backend.rag.llm import LLMProvider
from backend.rag.query_processor import InvalidQuery, ProcessedQuery, process_query
from backend.rag.multidoc import interleave
from backend.retrieval.retriever import make_filter

log = logging.getLogger(__name__)
CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "config" / "rag_calibration.json"


@dataclass
class RAGConfig:
    candidate_depth: int = 20
    max_chunks: int = 6
    max_words: int = 1600
    per_doc_cap: int = 3
    cross_doc_depth: int = 20                  # candidate depth when the question spans papers (deeper pool => more papers to choose from)
    cross_doc_max_docs: int | None = None      # diversify the context over the best N papers (None = off)
    min_cited_fraction: float = 0.6            # LLM answers citing fewer of their claims are not shown as grounded answers
    min_supported_fraction: float = 0.5        # ... nor are answers whose citations mostly fail the lexical support check
    use_reranker: bool = False                 # flipped on only if rerank_benchmark.json says adopt
    min_dense_cosine: float | None = None      # primary evidence gate (calibrated per model version)
    dense_model_version: str | None = None
    min_coverage: float | None = None          # weak lexical gate, used only when the dense signal is unavailable

    @classmethod
    def load(cls) -> "RAGConfig":
        c = cls()
        if CALIBRATION_PATH.exists():
            d = json.loads(CALIBRATION_PATH.read_text())
            c.min_dense_cosine, c.dense_model_version = d.get("min_dense_cosine"), d.get("dense_model_version")
            c.min_coverage = d.get("min_coverage_lexical")
            c.use_reranker = bool(d.get("use_reranker", False))
            c.cross_doc_depth = int(d.get("cross_doc_depth", c.cross_doc_depth))
            c.cross_doc_max_docs = d.get("cross_doc_max_docs", c.cross_doc_max_docs)
        return c


class RAGPipeline:
    def __init__(self, retrieval, provider: LLMProvider | None = None, reranker=None, config: RAGConfig | None = None):
        """retrieval: object with .service (RetrievalService), .chunks {id: Chunk}, .meta {id: {document_id, section}}."""
        self.r, self.provider, self.reranker = retrieval, provider, reranker
        self.cfg = config or RAGConfig.load()

    # ---- steps ----
    def _retrieve(self, pq: ProcessedQuery):
        t = time.perf_counter()
        depth = self.cfg.cross_doc_depth if pq.scope == "cross_document" else self.cfg.candidate_depth
        secs = set(pq.sections) or None
        if len(pq.document_ids) > 1:                 # explicit multi-paper selection: retrieve per paper so every paper contributes
            per = max(5, depth // len(pq.document_ids))
            results = [self.r.service.retrieve(pq.retrieval_query, per, make_filter(self.r.meta, {d}, secs)) for d in pq.document_ids]
            res = next((x for x in results if x.fallback_active), results[-1])
            hits = interleave([x.hits for x in results])
        else:
            res = self.r.service.retrieve(pq.retrieval_query, depth, make_filter(self.r.meta, set(pq.document_ids) or None, secs))
            hits = res.hits
        cands = [(self.r.chunks[h.chunk_id], h.score, h.rank) for h in hits if h.chunk_id in self.r.chunks]
        t_ret = time.perf_counter() - t
        t_rr = 0.0
        if self.cfg.use_reranker and self.reranker is not None and cands:
            t = time.perf_counter()
            cands = self.reranker.rerank(pq.retrieval_query, cands)
            t_rr = time.perf_counter() - t
        return res, cands, t_ret, t_rr

    def _context(self, pq: ProcessedQuery, cands) -> ContextResult:
        return build_context(pq.retrieval_query, cands, self.cfg.max_chunks, self.cfg.max_words,
                             self.cfg.per_doc_cap if pq.scope == "cross_document" else None,
                             max_docs=self.cfg.cross_doc_max_docs if pq.scope == "cross_document" else None)

    def _gate(self, pq: ProcessedQuery, ctx: ContextResult, cands) -> tuple[bool, dict]:
        """Is the retrieved evidence strong enough to attempt an answer? Dense cosine (AUC 0.999 on the calibration data)
        when the primary dense model matches the calibrated version; otherwise a lenient lexical-coverage check."""
        info = {"gate": "none", "signal": None, "threshold": None}
        if not ctx.items:
            return False, {**info, "gate": "empty"}
        if self.cfg.min_dense_cosine is not None:
            sig, ver = self.r.service.evidence_signal(pq.retrieval_query, [c[0].chunk_id for c in cands[:5]])
            if sig is not None and ver == self.cfg.dense_model_version:
                return sig >= self.cfg.min_dense_cosine, {"gate": "dense_cosine", "signal": round(sig, 4), "threshold": self.cfg.min_dense_cosine}
        if self.cfg.min_coverage is not None:
            return ctx.coverage >= self.cfg.min_coverage, {"gate": "lexical_coverage(weak)", "signal": round(ctx.coverage, 3), "threshold": self.cfg.min_coverage}
        return True, info

    def _assemble(self, pq, res, cands, ctx, gen: GenerationResult, analysis: AnswerAnalysis | None, timings: dict, sufficient: bool, gate: dict | None = None) -> dict:
        refs = build_references(ctx.items, analysis, pq.normalized)
        sents = [{"text": s.text, "citations": s.citations, "invalid_citations": s.invalid_citations, "status": s.status,
                  "support": s.support, "supporting_label": s.supporting_label} for s in (analysis.sentences if analysis else [])]
        return {
            "question": pq.original, "answer": gen.text, "answer_mode": gen.mode, "sufficient_evidence": sufficient,
            "sentences": sents, "references": refs,
            "citation_quality": None if analysis is None else {
                "claims": analysis.n_claims, "cited_fraction": round(analysis.cited_fraction, 3),
                "valid_citation_fraction": round(analysis.citation_validity, 3), "supported_fraction": round(analysis.supported_fraction, 3),
                "unsupported_claims": analysis.unsupported, "note": "lexical-overlap heuristic, not entailment"},
            "query": {"intent": pq.intent, "scope": pq.scope, "is_followup": pq.is_followup, "retrieval_query": pq.retrieval_query,
                      "document_ids": pq.document_ids, "section_hints": pq.section_hints, "warnings": pq.warnings},
            "retrieval": {"mode": res.mode if res else None, "retriever": res.retriever if res else None, "kind": res.kind if res else None,
                          "fallback_active": res.fallback_active if res else None, "candidates": len(cands),
                          "reranked": bool(self.cfg.use_reranker and self.reranker is not None),
                          "dropped_duplicates": ctx.dropped_duplicates},
            "evidence": {"coverage": round(ctx.coverage, 3), "top_score": round(ctx.top_score, 5), **(gate or {})},
            "generation": {"provider": gen.provider, "model": gen.model, "prompt_tokens": gen.prompt_tokens,
                           "completion_tokens": gen.completion_tokens, "cost_usd": gen.cost_usd, "provider_errors": gen.provider_errors},
            "timings": {k: round(v, 3) for k, v in timings.items()},
        }

    def _verify(self, pq: ProcessedQuery, ctx: ContextResult, gen: GenerationResult):
        """Guard against ungrounded generation. An LLM answer whose claims are mostly uncited or unsupported is
        replaced by the extractive answer (verbatim cited passages); the original is kept for transparency."""
        analysis = analyze_answer(gen.text, ctx.items) if gen.mode != "insufficient" else None
        info = {"verified": True, "replaced": False, "llm_answer": None, "reason": None}
        if gen.mode == "llm" and analysis is not None and analysis.n_claims > 0:
            bad = None
            if analysis.cited_fraction < self.cfg.min_cited_fraction:
                bad = f"only {analysis.cited_fraction:.0%} of claims carry citations (need {self.cfg.min_cited_fraction:.0%})"
            elif analysis.supported_fraction < self.cfg.min_supported_fraction:
                bad = f"only {analysis.supported_fraction:.0%} of cited claims are supported by their sources"
            elif not analysis.all_citations_valid:
                bad = "answer cites sources that do not exist"
            if bad:
                alt = extractive_answer(pq.normalized, ctx.items)
                info = {"verified": False, "replaced": True, "llm_answer": gen.text, "reason": bad}
                gen = GenerationResult(alt, "extractive" if alt != INSUFFICIENT else "insufficient", "extractive", None,
                                       provider_errors=gen.provider_errors + [f"LLM answer rejected: {bad}"],
                                       prompt_tokens=gen.prompt_tokens, completion_tokens=gen.completion_tokens,
                                       cost_usd=gen.cost_usd, latency_s=gen.latency_s)
                analysis = analyze_answer(gen.text, ctx.items) if gen.mode != "insufficient" else None
        return gen, analysis, info

    def _record(self, out: dict) -> None:
        m = getattr(self.r, "metrics", None)
        if m is None:
            return
        m.record("rag_total_s", out["timings"]["total_s"], mode=out["answer_mode"])
        m.record("rag_answer_replaced", 1.0 if (out.get("verification") or {}).get("replaced") else 0.0)
        m.record("rag_insufficient", 1.0 if out["answer_mode"] == "insufficient" else 0.0)
        cq = out.get("citation_quality")
        if cq and out["answer_mode"] == "llm":
            m.record("rag_valid_citation_fraction", cq["valid_citation_fraction"])

    # ---- public ----
    def ask(self, question: str, history: list[dict] | None = None, document_ids: list[str] | None = None,
            sections: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        pq = process_query(question, document_ids, sections, history)
        res, cands, t_ret, t_rr = self._retrieve(pq)
        ctx = self._context(pq, cands)
        ok, gate = self._gate(pq, ctx, cands)
        t = time.perf_counter()
        gen = generate(pq, ctx.items if ok else [], self.provider, history)
        t_gen = time.perf_counter() - t
        gen, analysis, ver = self._verify(pq, ctx, gen)
        timings = {"retrieval_s": t_ret, "rerank_s": t_rr, "generation_s": t_gen, "total_s": time.perf_counter() - t0}
        out = self._assemble(pq, res, cands, ctx, gen, analysis, timings, ok, gate)
        out["verification"] = ver
        self._record(out)
        return out

    def stream(self, question: str, history=None, document_ids=None, sections=None) -> Iterator[dict]:
        """Events: {'type':'meta'}, {'type':'token','text'}, {'type':'final', ...full response...}."""
        t0 = time.perf_counter()
        pq = process_query(question, document_ids, sections, history)
        res, cands, t_ret, t_rr = self._retrieve(pq)
        ctx = self._context(pq, cands)
        ok, gate = self._gate(pq, ctx, cands)
        yield {"type": "meta", "retrieval_mode": res.mode, "fallback_active": res.fallback_active, "retriever": res.retriever,
               "sources": [{"label": i.label, "title": i.chunk.paper_title, "pages": [i.chunk.page_start, i.chunk.page_end],
                            "section": i.chunk.section} for i in ctx.items], "sufficient_evidence": ok}
        parts, t = [], time.perf_counter()
        for tok in stream_generate(pq, ctx.items if ok else [], self.provider, history):
            parts.append(tok)
            yield {"type": "token", "text": tok}
        text = "".join(parts).strip()
        mode = "insufficient" if (not ok or text == INSUFFICIENT) else ("extractive" if text.startswith("Extractive answer") else "llm")
        gen = GenerationResult(text, mode, getattr(self.provider, "name", None) if mode == "llm" else mode)
        gen, analysis, ver = self._verify(pq, ctx, gen)
        timings = {"retrieval_s": t_ret, "rerank_s": t_rr, "generation_s": time.perf_counter() - t, "total_s": time.perf_counter() - t0}
        yield {"type": "final", "verification": ver, **self._assemble(pq, res, cands, ctx, gen, analysis, timings, ok, gate)}
