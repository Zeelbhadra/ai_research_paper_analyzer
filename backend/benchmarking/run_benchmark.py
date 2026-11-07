"""End-to-end deterministic benchmark: retrieval metrics (selection split) + isolated performance ->
weighted scores -> threshold check -> winner -> registry -> guarded promotion.

    python -m backend.benchmarking.run_benchmark [--candidates ...] [--report-test]

The held-out TEST split is evaluated only with --report-test, AFTER the selection is recorded, and is
attached to the run as a report; it never influences selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.benchmarking.model_registry import ModelRegistry
from backend.benchmarking.model_selector import maybe_promote
from backend.benchmarking.run_retrieval_eval import run as run_retrieval
from backend.benchmarking.selection import Candidate, load_policy, select
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.retrieval.factory import build_retriever, tfidf_index_dir


def candidate_info(name: str, mgr: EmbeddingManager, cfg_key: str, cache: dict) -> dict:
    """Identity/index metadata for the registry."""
    r = build_retriever(name, mgr, cfg_key, _cache=cache)

    def loc(p) -> str:                           # storage-relative, so the registry stays valid when storage is mounted elsewhere
        try:
            return Path(p).relative_to(mgr.root).as_posix()
        except ValueError:
            return str(p)
    if name == "tfidf":
        return {"family": "tfidf", "checkpoint": "sklearn.TfidfVectorizer", "version": r.index.version, "dim": None,
                "index_location": loc(tfidf_index_dir(mgr, cfg_key)), "config": r.index.params}
    if name.startswith("hybrid:"):
        sp, de = r.sparse, r.dense
        return {"family": "hybrid", "checkpoint": f"tfidf+{de.embedder.checkpoint}",
                "version": f"hybrid:{sp.index.version}+{de.embedder.version}", "dim": de.embedder.dim,
                "index_location": ";".join([loc(tfidf_index_dir(mgr, cfg_key)), loc(mgr.index_dir(de.embedder.key, cfg_key))]),
                "config": {"fusion": "rrf", "rrf_k": r.rrf_k, "w_sparse": r.w_sparse, "w_dense": r.w_dense, "depth": r.depth,
                           "tuned_on": "dev"}}
    e = r.embedder
    return {"family": e.family, "checkpoint": e.checkpoint, "version": e.version, "dim": e.dim,
            "index_location": loc(mgr.index_dir(e.key, cfg_key)),
            "config": {"query_prefix": e.query_text(""), "doc_prefix": e.doc_text(""), "device": e.device, "normalize": True}}


def make_validator(mgr: EmbeddingManager, cfg_key: str):
    """Promotion gate: index loads, retriever healthy, and a smoke query returns hits."""
    def validate(cand: dict) -> tuple[bool, str]:
        try:
            name = cand["config"].get("_name") or cand["name"]
            r = build_retriever(name, mgr, cfg_key)
            if not r.health():
                return False, "health check failed"
            if not r.retrieve("neural network", 3):
                return False, "smoke query returned nothing"
            return True, "ok"
        except Exception as e:
            return False, repr(e)
    return validate


def main(argv=None) -> str:
    """Run the deterministic benchmark; returns the registry run_id. `argv` allows agents to call it programmatically."""
    s = get_settings()
    bdir = s.storage_dir / "benchmark"
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-cfg", default="recursive-256-15")
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--report-test", action="store_true")
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--adjudication", type=Path, default=None, help="sensitivity run with extra judged labels (never promotes)")
    a = ap.parse_args(argv)
    policy = load_policy()
    split = policy["selection_split"]
    assert split != "test", "selection must never use the test split"
    tuned = json.loads((bdir / "hybrid_tuning_dev.json").read_text())["chosen"]
    hyb = f"hybrid:{tuned['partner']}:{tuned['w_sparse']}:{tuned['w_dense']}"
    names = a.candidates or ["tfidf", "sbert", "e5", "bge", hyb]
    perf = json.loads((bdir / "performance.json").read_text())

    summary = run_retrieval(split, a.chunk_cfg, names, bdir / "dataset.json", bdir, a.adjudication)
    stem = f"retrieval_{split}_{a.chunk_cfg}" + ("_adjudicated" if a.adjudication else "")
    per_query = json.loads((bdir / f"{stem}_per_query.json").read_text())
    mgr, cache = EmbeddingManager(s.storage_dir), {}
    cands, infos = [], {}
    for n in names:
        q, p = summary["models"][n], perf[n]
        qids = sorted(per_query[n])
        m = {"nDCG@10": q["nDCG@10"], "Recall@5": q["R@5"], "MRR@10": q["MRR@10"], "Precision@5": q["P@5"], "MAP": q["MAP"],
             "latency_p50_ms": p["latency_p50_ms"], "latency_p95_ms": p["latency_p95_ms"], "throughput_qps": p["throughput_qps"],
             "memory_mb": p["memory_mb"], "indexing_seconds": p["indexing_seconds"], "index_bytes": p["index_bytes"]}
        cands.append(Candidate(n, m, [per_query[n][i]["nDCG@10"] for i in qids], p["health_ok"]))
        infos[n] = candidate_info(n, mgr, a.chunk_cfg, cache)
    sel = select(cands, policy)

    rows = []
    for c in cands:
        i = infos[c.name]
        rows.append({"name": c.name, **i, "config": {**i["config"], "_name": c.name}, "metrics": c.metrics,
                     "overall_score": round(sel.quality_index[c.name], 3),          # = quality index (mean of quality metrics), NOT a weighted score
                     "footprint": round(sel.footprint.get(c.name, float("nan")), 3), "pareto_optimal": c.name in sel.pareto,
                     "non_inferiority_vs_reference": sel.non_inferior.get(c.name),
                     "eligible": not sel.feasible[c.name], "fail_reasons": sel.feasible[c.name]})
    rows.sort(key=lambda r: -r["overall_score"])
    reg = ModelRegistry(s.storage_dir / "registry")
    meta = {"dataset_version": summary["dataset_version"], "chunk_cfg": a.chunk_cfg, "selection_split": split,
            "n_queries": summary["n_queries"], "human_verified_fraction": summary["human_verified_fraction"],
            "n_chunks": Repository(s.db_path).count("chunks", "WHERE config_key=?", (a.chunk_cfg,)),
            "selection_method": "feasibility -> quality reference -> paired-bootstrap non-inferiority -> min footprint (no metric weights)",
            "policy": policy, "budgets": sel.budgets, "reference": sel.reference, "trace": sel.trace, "seed": s.seed,
            "adjudication": str(a.adjudication) if a.adjudication else None,
            "indexing_seconds_note": "dense indexing time extrapolated from a 192-chunk sample"}
    winner = next((r for r in rows if r["name"] == sel.winner), None)
    run_id = reg.record_run(meta, rows, winner and {"name": winner["name"], "overall_score": winner["overall_score"]})
    print(f"run {run_id} on '{split}' ({summary['n_queries']} queries){' [ADJUDICATED labels, sensitivity only]' if a.adjudication else ''}")
    for line in sel.trace:
        print("  *", line)
    if not a.no_promote and not a.adjudication:
        ok, msg = maybe_promote(reg, run_id, sel.winner, policy, make_validator(mgr, a.chunk_cfg))
        print("PROMOTION:", ok, msg)
    for r in rows:
        mm = r["metrics"]
        print(f"  {r['name']:20s} quality_index={r['overall_score']:5.1f} nDCG@10={mm['nDCG@10']:.3f} footprint={r['footprint']:7.1f} "
              f"pareto={r['pareto_optimal']} feasible={r['eligible']}")

    if a.report_test and not a.adjudication:
        t = run_retrieval("test", a.chunk_cfg, names, bdir / "dataset.json", bdir)
        reg.attach(run_id, "heldout_test", {n: {k: v for k, v in d.items() if k != "by_category"} for n, d in t["models"].items()})
        print("HELD-OUT TEST (reported, not used for selection):")
        for n, d in t["models"].items():
            print(f"  {n:20s} nDCG@10={d['nDCG@10']:.3f} CI={d['nDCG@10_ci95']} R@5={d['R@5']:.3f} MRR={d['MRR@10']:.3f} p_vs_best={d['p_vs_best_nDCG@10']}")
    return run_id


if __name__ == "__main__":
    main()
