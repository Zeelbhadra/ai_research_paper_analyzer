"""Choose hybrid partner + RRF weights using the DEV split only.   python -m backend.benchmarking.tune_hybrid"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks, records_for
from backend.benchmarking.retrieval_evaluator import evaluate_retriever
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.retrieval.factory import build_retriever

PARTNERS = ["sbert", "e5", "bge"]
WEIGHTS = [(1.0, 0.5), (1.0, 1.0), (1.0, 2.0)]      # (sparse, dense)


def main(cfg_key="recursive-256-15"):
    s = get_settings()
    out_dir = s.storage_dir / "benchmark"
    ds = load_dataset(out_dir / "dataset.json")
    dev = records_for(ds, "dev")
    chunks = Repository(s.db_path).get_chunks(cfg_key)
    cache_rel = {q["query_id"]: map_evidence_to_chunks(q["evidence"], chunks) for q in dev}
    mgr, cache, rows = EmbeddingManager(s.storage_dir), {}, []
    for p in PARTNERS:
        for ws, wd in WEIGHTS:
            r = build_retriever(f"hybrid:{p}:{ws}:{wd}", mgr, cfg_key, _cache=cache)
            res = evaluate_retriever(r.name, lambda q, k, r=r: [h.chunk_id for h in r.retrieve(q, k)], dev,
                                     lambda q: cache_rel[q["query_id"]])
            m = res.means()
            rows.append({"partner": p, "w_sparse": ws, "w_dense": wd, "dev_nDCG@10": round(m["nDCG@10"], 4),
                         "dev_R@5": round(m["R@5"], 4), "dev_MRR@10": round(m["MRR@10"], 4)})
            print(rows[-1], flush=True)
    best = max(rows, key=lambda r: r["dev_nDCG@10"])
    (out_dir / "hybrid_tuning_dev.json").write_text(json.dumps({"grid": rows, "chosen": best, "split": "dev"}, indent=1))
    print("CHOSEN (dev):", best)


if __name__ == "__main__":
    main(*sys.argv[1:])
