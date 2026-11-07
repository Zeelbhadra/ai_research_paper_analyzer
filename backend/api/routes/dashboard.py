"""Dashboard and benchmark-dashboard data APIs (read-only aggregation of what the pipeline already stores)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)
bp = Blueprint("dashboard", __name__, url_prefix="/api")
CFG = "recursive-256-15"


def _storage() -> Path:
    from backend.config.settings import get_settings
    return get_settings().storage_dir


def _load(rel: str):
    p = _storage() / rel
    return json.loads(p.read_text()) if p.exists() else None


def _registry() -> dict:
    return _load("registry/registry.json") or {"runs": [], "production": None, "promotions": []}


@bp.get("/dashboard")
def dashboard():
    from backend.api.routes.library import repo
    r = repo()
    reg = _registry()
    prod = reg.get("production")
    selection_runs = [r for r in reg["runs"] if not r["meta"].get("adjudication")]     # sensitivity runs are not the production record
    latest = selection_runs[-1] if selection_runs else None
    out = {
        "corpus": {"papers": r.count("papers"), "processed": r.count("papers", "WHERE status='processed'"),
                   "failed": r.count("papers", "WHERE status='failed'"), "chunks": r.count("chunks", "WHERE config_key=?", (CFG,)),
                   "extraction_failures": len({f["path"] for f in r.failures()})},
        "active_model": None if not prod else {"name": prod["name"], "family": prod["family"], "checkpoint": prod["checkpoint"],
                                               "version": prod["version"], "promoted_at": prod["promoted_at"], "quality_index": prod["overall_score"]},
        "vector_db": {"name": "FAISS", "kind": "exact inner product (flat)", "persistence": "on disk, one index per model version and chunk config"},
        "latest_benchmark": None,
        "retrieval_serving": None, "pipeline": None, "fallback": None,
    }
    if latest:
        out["latest_benchmark"] = {"run_id": latest["run_id"], "timestamp": latest["timestamp"], "split": latest["meta"]["selection_split"],
                                   "n_queries": latest["meta"]["n_queries"], "winner": (latest.get("winner") or {}).get("name"),
                                   "candidates": [{"name": c["name"], "nDCG@10": c["metrics"].get("nDCG@10"), "eligible": c["eligible"]} for c in latest["candidates"]]}
    try:                                                       # latency from the metrics store (works without loading any model)
        from backend.monitoring.metrics import MetricsStore
        m = MetricsStore(_storage() / "metrics.sqlite")
        since = time.time() - 24 * 3600
        out["retrieval_serving"] = {"window": "24h", "latency_ms": m.stats("retrieval_latency_ms", since), "fallback_rate": m.rate("fallback_used", since),
                                    "failures": m.stats("retrieval_failure", since)["n"]}
    except Exception:
        log.exception("metrics unavailable")
    try:
        jobs = r.jobs(100)
        by = {}
        for j in jobs:
            by[j["status"]] = by.get(j["status"], 0) + 1
        out["pipeline"] = {"jobs_by_status": by, "recent_failures": [j for j in jobs if j["status"] == "failed"][:3]}
        from backend.api.routes import pipeline as pl
        if pl._sys is not None:
            out["pipeline"]["running_workflows"] = pl._sys[1].status()["running"]
    except Exception:
        log.exception("jobs unavailable")
    try:                                                       # fallback status only if the service is already loaded (never forces a model load)
        from backend.api.routes import retrieval as rt
        if rt._prod is not None:
            s = rt._prod.service.snapshot()
            out["fallback"] = {k: s[k] for k in ("mode", "fallback_active", "reason", "primary", "fallback", "total_failures", "fallback_requests", "total_requests")}
    except Exception:
        log.exception("fallback status unavailable")
    return jsonify(out)


def _means_from_per_query(path: str, ks=(1, 3, 5, 10)) -> dict | None:
    pq = _load(path)
    if not pq:
        return None
    out = {}
    for name, qs in pq.items():
        rows = list(qs.values())
        out[name] = {"n": len(rows)}
        for k in ks:
            out[name][f"P@{k}"] = float(np.mean([x[f"P@{k}"] for x in rows]))
            out[name][f"R@{k}"] = float(np.mean([x[f"R@{k}"] for x in rows]))
            out[name][f"nDCG@{k}"] = float(np.mean([x[f"nDCG@{k}"] for x in rows]))
        out[name]["MRR@10"] = float(np.mean([x["MRR@10"] for x in rows]))
        out[name]["MAP"] = float(np.mean([x["AP@10"] for x in rows]))
    return out


@bp.get("/benchmarks/overview")
def benchmarks_overview():
    reg = _registry()
    runs = [{"run_id": r["run_id"], "timestamp": r["timestamp"], "split": r["meta"]["selection_split"], "n_queries": r["meta"]["n_queries"],
             "dataset_version": r["meta"]["dataset_version"], "winner": (r.get("winner") or {}).get("name"), "reference": r["meta"].get("reference"),
             "adjudicated": bool(r["meta"].get("adjudication")), "n_chunks": r["meta"].get("n_chunks"),
             "candidates": [{"name": c["name"], "quality_index": c["overall_score"], "footprint": c.get("footprint"), "pareto_optimal": c.get("pareto_optimal"),
                             "eligible": c["eligible"], "fail_reasons": c["fail_reasons"], "non_inferiority": c.get("non_inferiority_vs_reference"),
                             "metrics": c["metrics"]} for c in r["candidates"]],
             "trace": r["meta"].get("trace", []), "budgets": r["meta"].get("budgets"), "heldout_test": (r.get("attachments") or {}).get("heldout_test")}
            for r in reg["runs"]]
    prod = reg.get("production")
    return jsonify({
        "runs": runs, "production": prod and {k: prod[k] for k in ("name", "family", "version", "promoted_at", "run_id")},
        "promotions": reg.get("promotions", []),
        "ranking_metrics": {"validation": _means_from_per_query("benchmark/retrieval_validation_recursive-256-15_per_query.json"),
                            "test": _means_from_per_query("benchmark/retrieval_test_recursive-256-15_per_query.json"),
                            "dev": _means_from_per_query("benchmark/retrieval_dev_recursive-256-15_per_query.json")},
        "by_category": (_load("benchmark/retrieval_validation_recursive-256-15.json") or {}).get("models"),
        "performance": _load("benchmark/performance.json"), "chunk_sweep": {"dev": _load("benchmark/chunk_sweep_dev.json"), "validation": _load("benchmark/chunk_sweep_validation.json")},
        "reranker": _load("benchmark/rerank_benchmark.json"), "paraphrase_probe": _load("benchmark/paraphrase_probe.json"),
        "hybrid_tuning": _load("benchmark/hybrid_tuning_dev.json"),
        "dataset": _dataset_card(),
    })


def _dataset_card():
    ds = _load("benchmark/dataset.json")
    if not ds:
        return None
    recs = ds["records"]
    kept = [r for r in recs if not r.get("excluded_reason")]
    split = {}
    for r in kept:
        split[r["split"]] = split.get(r["split"], 0) + 1
    cat = {}
    for r in kept:
        cat[r["category"]] = cat.get(r["category"], 0) + 1
    return {"version": ds["version"], "generated": len(recs), "retained": len(kept), "splits": split, "categories": cat,
            "human_verified": sum(1 for r in kept if r.get("human_verified")), "label_source": "synthetic_llm_auto_validated"}
