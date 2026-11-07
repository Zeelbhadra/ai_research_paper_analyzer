"""Real-model fault-injection demo: python -m backend.benchmarking.demo_fallback

Uses the actual production hybrid (TF-IDF + E5) and real indexes. Faults are injected into the live E5 encoder;
the registry-switch part runs on a COPY of storage/ so the real registry is not modified."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from backend.benchmarking.model_registry import ModelRegistry
from backend.config.settings import get_settings
from backend.embeddings.base import EmbeddingError
from backend.retrieval.production import ProductionRetrieval
from backend.retrieval.service import ServiceConfig

QUERIES = ["How does self-attention work in vision transformers?",
           "What algorithms were compared for graph matching?",
           "How do autonomous vehicles change ride-sharing profit?"]


def top(res):
    return [r["chunk_id"][-12:] for r in res["results"][:3]]


def main():
    out = {}
    s = get_settings()
    pr = ProductionRetrieval(service_config=ServiceConfig(failure_threshold=2, recovery_checks=2, cooldown_base_s=0.5))
    svc = pr.service
    out["production"] = pr.status()["primary"]
    base = [pr.search(q, 5) for q in QUERIES]
    out["normal"] = [{"mode": r["mode"], "retriever": r["retriever"], "latency_ms": r["latency_ms"]} for r in base]

    dense = svc._primary.dense.embedder
    orig = dense._encode

    def broken(texts, batch_size):
        raise EmbeddingError("injected: embedding model unavailable")
    dense._encode = broken
    fb = [pr.search(q, 5) for q in QUERIES for _ in range(2)]
    st = pr.status()
    out["during_fault"] = {"modes": [r["mode"] for r in fb], "fallback_active": st["fallback_active"], "reason": st["reason"],
                           "all_served": all(len(r["results"]) == 5 for r in fb)}
    tf = pr.service._get_fallback()
    same = all([h.chunk_id for h in tf.retrieve(q, 5)] == [r["chunk_id"] for r in fb[2 * i]["results"]] for i, q in enumerate(QUERIES))
    out["fallback_results_equal_plain_tfidf"] = same

    dense._encode = orig                                   # repair
    time.sleep(0.6)
    r1 = svc.check_health()
    time.sleep(0.6)
    r2 = svc.check_health()
    after = pr.search(QUERIES[0], 5)
    out["recovery"] = {"probe1": r1, "probe2": r2, "mode_after": after["mode"], "retriever_after": after["retriever"],
                       "same_top3_as_before": top(after) == top(base[0])}
    out["events"] = [e["kind"] for e in pr.events.tail(30)]

    # ---- registry-driven model switch on a copy of storage ----
    tmp = Path(tempfile.mkdtemp(prefix="storage_copy_"))
    try:
        for sub in ("registry", "indexes", "embedding_cache", "papers.sqlite"):
            src = s.storage_dir / sub
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, tmp / sub)
        pr2 = ProductionRetrieval(storage_dir=tmp)
        reg = ModelRegistry(tmp / "registry")
        run_id = reg.latest_run()["run_id"] if False else [r for r in reg.data["runs"] if not r["meta"].get("adjudication")][-1]["run_id"]
        shutil.rmtree(next((tmp / "indexes").glob("sbert_*")))             # force index regeneration
        reg.promote(run_id, "sbert", "demo: switch to SBERT")
        t = time.perf_counter()
        ok, msg = pr2.sync_with_registry()
        out["model_switch"] = {"switched": ok, "message": msg, "seconds": round(time.perf_counter() - t, 1),
                               "now_serving": pr2.search(QUERIES[0], 3)["retriever"], "history": reg.history()[-1]}
        # a promoted model whose index cannot be built must NOT replace the working one
        reg = ModelRegistry(tmp / "registry")
        cand = next(c for c in reg.get_run(run_id)["candidates"] if c["name"] == "bge")
        cand["checkpoint"] = "BAAI/does-not-exist-xyz"
        cand["config"]["_name"] = "bge"
        reg.data["production"] = {**cand, "name": "bge", "run_id": run_id, "version": "bge:BAAI/does-not-exist-xyz:0"}
        reg._save()
        import backend.retrieval.production as prod_mod
        orig_make = prod_mod.make_embedder
        prod_mod.make_embedder = lambda fam, *a, **k: orig_make(fam, "BAAI/does-not-exist-xyz")
        ok2, msg2 = pr2.sync_with_registry()
        prod_mod.make_embedder = orig_make
        out["bad_model_switch"] = {"switched": ok2, "message": msg2[:160], "still_serving": pr2.search(QUERIES[0], 3)["retriever"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (s.storage_dir / "benchmark" / "fallback_demo.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
