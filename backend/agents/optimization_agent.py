"""Retrieval Optimization Agent: runs the deterministic benchmark, deploys validated winners, sweeps chunking configs.

All numbers come from the Python evaluation code; nothing here asks an LLM to score, choose or promote.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np

from backend.agents.orchestrator import Agent, AgentError
from backend.agents.services import Services

log = logging.getLogger(__name__)


class OptimizationAgent(Agent):
    name = "optimization"

    def __init__(self, sv: Services, benchmark_fn=None):
        self.sv, self._bench = sv, benchmark_fn

    def actions(self):
        return {"benchmark": self.benchmark, "deploy": self.deploy, "chunk_sweep": self.chunk_sweep}

    def benchmark(self, chunk_cfg: str | None = None, report_test: bool = False, refresh_indexes: bool = True) -> dict:
        """Re-run retrieval evaluation + selection on the current corpus. Latency/memory figures are reused from the last
        isolated performance run (storage/benchmark/performance.json); rerun performance_benchmark to refresh them."""
        cfg = chunk_cfg or self.sv.cfg_key
        refreshed = self.refresh_all_indexes(cfg) if refresh_indexes and self._bench is None else None
        bench = self._bench
        if bench is None:
            from backend.benchmarking.run_benchmark import main as bench
        argv = ["--chunk-cfg", cfg] + (["--report-test"] if report_test else [])
        run_id = bench(argv)
        reg = self.sv.registry
        run = reg.get_run(run_id)
        perf = self.sv.storage / "benchmark" / "performance.json"
        return {"run_id": run_id, "indexes_refreshed": refreshed, "winner": run.get("winner"), "reference": run["meta"].get("reference"),
                "production": (reg.production or {}).get("name"), "n_queries": run["meta"]["n_queries"],
                "trace": run["meta"].get("trace", []), "performance_data": str(perf) if perf.exists() else None}

    def refresh_all_indexes(self, cfg: str) -> dict:
        """Candidates must be compared over the SAME, current corpus: rebuild every candidate's index (dense embeddings are
        cached per chunk text, so only new chunks cost model time)."""
        from backend.embeddings.base import make_embedder
        chunks = self.sv.repo.get_chunks(cfg)
        self.sv.mgr.build_tfidf_index(chunks, cfg)
        out = {"tfidf": len(chunks)}
        for fam in ("sbert", "e5", "bge"):
            e = make_embedder(fam)
            e.load()
            _, st = self.sv.mgr.build_dense_index(e, chunks, cfg)
            out[fam] = st.n_embedded
        return {"n_chunks": len(chunks), "newly_embedded": out}

    def deploy(self) -> dict:
        """Make the serving path follow the registry (safe switch: build + probe before swapping)."""
        ok, msg = self.sv.get_production().sync_with_registry()
        return {"switched": ok, "message": msg}

    def chunk_sweep(self, configs: list[str], families: list[str] | None = None, split: str = "validation", budget_words: int = 1500) -> dict:
        """Evaluate chunking configurations by retrieval quality. Re-chunks the PDFs per config, rebuilds TF-IDF and the
        requested (cheap) dense models, and scores each config on the SAME queries by mapping evidence text onto the new
        chunks. Reported for `split` only; nothing is promoted automatically."""
        from backend.agents.document_agent import parse_cfg_key
        from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks, records_for
        from backend.benchmarking.retrieval_evaluator import bootstrap_ci, budgeted_evidence_coverage, evaluate_retriever
        from backend.embeddings.base import make_embedder
        from backend.ingestion.document_processor import ingest_directory
        from backend.retrieval.retriever import DenseRetriever, SparseRetriever
        families = families or ["tfidf", "sbert"]
        ds = load_dataset(self.sv.storage / "benchmark" / "dataset.json")
        queries = records_for(ds, split)
        results = {}
        for key in configs:
            t0 = time.perf_counter()
            cfg = parse_cfg_key(key)
            if self.sv.repo.count("chunks", "WHERE config_key=?", (key,)) == 0:
                ingest_directory(self.sv.repo, self.sv.settings.data_dir, cfg)
            chunks = self.sv.repo.get_chunks(key)
            rels = {q["query_id"]: map_evidence_to_chunks(q["evidence"], chunks) for q in queries}
            usable = [q for q in queries if any(g > 0 for g in rels[q["query_id"]].values())]
            text = {c.chunk_id: c.text for c in chunks}
            doc = {c.chunk_id: c.document_id for c in chunks}
            row = {"n_chunks": len(chunks), "avg_words": round(float(np.mean([len(c.text.split()) for c in chunks])), 1),
                   "queries_with_labels": f"{len(usable)}/{len(queries)}"}
            for fam in families:
                if fam == "tfidf":
                    idx, _ = self.sv.mgr.build_tfidf_index(chunks, key)
                    r = SparseRetriever(idx)
                else:
                    e = make_embedder(fam)
                    e.load()
                    st, _ = self.sv.mgr.build_dense_index(e, chunks, key)
                    r = DenseRetriever(e, st)
                res = evaluate_retriever(f"{fam}@{key}", lambda q, k, r=r: [h.chunk_id for h in r.retrieve(q, k)], usable,
                                         lambda q: rels[q["query_id"]])
                m = res.means()
                lo, hi = res.ci("nDCG@10")
                cov = [budgeted_evidence_coverage([h.chunk_id for h in r.retrieve(q["query"], 30)], text, doc, q["evidence"], budget_words)
                       for q in usable]
                clo, chi = bootstrap_ci(cov)
                row[fam] = {"nDCG@10": round(m["nDCG@10"], 4), "ci95": [round(lo, 4), round(hi, 4)], "R@5": round(m["R@5"], 4),
                            "MRR@10": round(m["MRR@10"], 4),
                            "ctx_coverage": round(float(np.mean(cov)), 4), "ctx_coverage_ci95": [round(clo, 4), round(chi, 4)]}
            row["seconds"] = round(time.perf_counter() - t0, 1)
            results[key] = row
        out = {"split": split, "families": families, "budget_words": budget_words, "results": results,
               "note": "Only the listed (cheap) model families were swept; E5/BGE/hybrid were not re-evaluated per config."}
        (self.sv.storage / "benchmark" / f"chunk_sweep_{split}.json").write_text(json.dumps(out, indent=1))
        return out
