"""Does cross-encoder reranking of the production retriever's top-N improve retrieval? (measure before adopting)

    python -m backend.benchmarking.rerank_benchmark [--depth 20] [--splits dev validation test]

Decision rule (fixed in advance): adopt only if, on VALIDATION, the lower bound of the 95% paired-bootstrap CI of
mean(nDCG@10 reranked - nDCG@10 baseline) is > 0. Dev/test are reported for context. Scores are cached on disk so
reruns do not recompute pairs.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks, records_for
from backend.benchmarking.retrieval_evaluator import query_metrics
from backend.benchmarking.selection import paired_diff_ci
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.retrieval.factory import build_retriever
from backend.retrieval.reranker import DEFAULT_RERANKER, CrossEncoderReranker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--splits", nargs="+", default=["dev", "validation", "test"])
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    a = ap.parse_args()
    s = get_settings()
    b = s.storage_dir / "benchmark"
    reg = json.loads((s.storage_dir / "registry" / "registry.json").read_text())
    name = reg["production"]["config"]["_name"]
    ds = load_dataset(b / "dataset.json")
    chunks = {c.chunk_id: c for c in Repository(s.db_path).get_chunks(a.chunk_cfg)}
    base = build_retriever(name, EmbeddingManager(s.storage_dir), a.chunk_cfg)
    rr = CrossEncoderReranker().load()
    cache_path = b / f"rerank_cache_{DEFAULT_RERANKER.replace('/', '_')}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    result = {"baseline": name, "reranker": DEFAULT_RERANKER, "depth": a.depth, "splits": {}}
    lat = []
    for split in a.splits:
        qs = records_for(ds, split)
        rels = {q["query_id"]: map_evidence_to_chunks(q["evidence"], list(chunks.values())) for q in qs}
        base_m, rr_m = [], []
        for q in qs:
            hits = base.retrieve(q["query"], a.depth)
            ids = [h.chunk_id for h in hits]
            base_m.append(query_metrics(ids, rels[q["query_id"]])["nDCG@10"])
            key = q["query_id"] + "|" + q["query"]
            if key not in cache or set(cache[key]) != set(ids):
                t = time.perf_counter()
                sc = rr.score(q["query"], [chunks[i].text for i in ids])
                lat.append(time.perf_counter() - t)
                cache[key] = dict(zip(ids, sc))
                cache_path.write_text(json.dumps(cache))
            order = sorted(ids, key=lambda i: -cache[key][i])
            rr_m.append(query_metrics(order, rels[q["query_id"]])["nDCG@10"])
        m, lo, hi = paired_diff_ci(np.array(rr_m), np.array(base_m))
        result["splits"][split] = {"n": len(qs), "baseline_nDCG@10": round(float(np.mean(base_m)), 4),
                                   "reranked_nDCG@10": round(float(np.mean(rr_m)), 4), "diff": round(m, 4),
                                   "diff_ci95": [round(lo, 4), round(hi, 4)]}
        print(split, result["splits"][split], flush=True)
    result["rerank_seconds_per_query_measured"] = round(float(np.mean(lat)), 2) if lat else None
    v = result["splits"].get("validation")
    result["adopt"] = bool(v and v["diff_ci95"][0] > 0)
    result["decision_rule"] = "adopt iff validation 95% CI lower bound of (reranked - baseline) nDCG@10 > 0"
    (b / "rerank_benchmark.json").write_text(json.dumps(result, indent=1))
    print("ADOPT:", result["adopt"], "| sec/query (uncached):", result["rerank_seconds_per_query_measured"])


if __name__ == "__main__":
    main()
