"""Retrieval quality evaluation of every model on identical queries/chunks.

    python -m backend.benchmarking.run_retrieval_eval --split validation [--chunk-cfg recursive-256-15]

Reads the prebuilt indexes (see embed_corpus), maps evidence -> graded relevance for the chosen chunk
config, runs all models on the same queries, writes JSON + CSV with bootstrap CIs and paired
significance tests against the best model. Latency here is indicative only; the isolated performance
benchmark (Step 5) is the source for latency/memory numbers used in scoring.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from backend.benchmarking.dataset_builder import assert_no_leakage, load_dataset, map_evidence_to_chunks, records_for
from backend.benchmarking.retrieval_evaluator import (EvalResult, evaluate_retriever, paired_bootstrap_pvalue)
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.retrieval.factory import build_retriever

HEADLINE = ["P@1", "P@5", "R@5", "R@10", "MRR@10", "MAP", "nDCG@10"]


def build_retrievers(mgr: EmbeddingManager, cfg_key: str, families: list[str], device="auto"):
    cache: dict = {}
    return {f: build_retriever(f, mgr, cfg_key, device, cache) for f in families}


def run(split: str, cfg_key: str, families: list[str], dataset_path: Path, out_dir: Path, adjudication: Path | None = None) -> dict:
    s = get_settings()
    ds = load_dataset(dataset_path)
    assert_no_leakage(ds)
    queries = records_for(ds, split)
    chunks = Repository(s.db_path).get_chunks(cfg_key)
    cache: dict[str, dict[str, int]] = {}
    extra = json.loads(Path(adjudication).read_text())["added_relevant"] if adjudication else {}

    def rels_for(q):
        if q["query_id"] not in cache:
            cache[q["query_id"]] = map_evidence_to_chunks(q["evidence"], chunks)
            for cid, g in extra.get(q["query_id"], {}).items():
                cache[q["query_id"]][cid] = max(cache[q["query_id"]].get(cid, 0), g)
        return cache[q["query_id"]]

    mgr = EmbeddingManager(s.storage_dir)
    results: dict[str, EvalResult] = {}
    for name, r in build_retrievers(mgr, cfg_key, families).items():
        results[name] = evaluate_retriever(name, lambda q, k, r=r: [h.chunk_id for h in r.retrieve(q, k)],
                                           queries, rels_for)
    best = max(results, key=lambda n: results[n].mean("nDCG@10"))
    summary = {"split": split, "chunk_cfg": cfg_key, "dataset_version": ds["version"], "n_queries": len(queries),
               "human_verified_fraction": sum(q["human_verified"] for q in queries) / max(len(queries), 1),
               "best_by_ndcg@10": best, "models": {}}
    for n, res in results.items():
        m = res.means()
        ci = res.ci("nDCG@10")
        summary["models"][n] = {
            **{k: round(m[k], 4) for k in HEADLINE if k in m}, "nDCG@10_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "by_category": {c: {k: round(v, 4) if k != "n" else v for k, v in d.items()} for c, d in res.by_category.items()},
            "p_vs_best_nDCG@10": None if n == best else round(paired_bootstrap_pvalue(
                [results[best].per_query[q]["nDCG@10"] for q in results[best].per_query],
                [res.per_query[q]["nDCG@10"] for q in results[best].per_query]), 4)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval_{split}_{cfg_key}" + ("_adjudicated" if adjudication else "")
    summary["adjudication"] = str(adjudication) if adjudication else None
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"{stem}_per_query.json").write_text(json.dumps({n: r.per_query for n, r in results.items()}))
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + HEADLINE + ["nDCG@10_ci_lo", "nDCG@10_ci_hi", "p_vs_best"])
        for n, d in summary["models"].items():
            w.writerow([n] + [d.get(k) for k in HEADLINE] + d["nDCG@10_ci95"] + [d["p_vs_best_nDCG@10"]])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["dev", "validation", "test"])
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    ap.add_argument("--families", nargs="+", default=["tfidf", "sbert", "e5", "bge"])
    a = ap.parse_args()
    s = get_settings()
    bdir = s.storage_dir / "benchmark"
    r = run(a.split, a.chunk_cfg, a.families, bdir / "dataset.json", bdir)
    print(json.dumps({k: v for k, v in r.items() if k != "models"}, indent=1))
    for n, d in r["models"].items():
        print(f"{n:6s}", {k: d[k] for k in HEADLINE}, "CI", d["nDCG@10_ci95"], "p", d["p_vs_best_nDCG@10"])


if __name__ == "__main__":
    main()
