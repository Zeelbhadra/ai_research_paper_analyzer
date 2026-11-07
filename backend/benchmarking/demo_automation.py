"""Real automation run on a COPY of storage/:  python -m backend.benchmarking.demo_automation "<topic>"

Live arXiv / Semantic Scholar, real PDF download + extraction, real E5 indexing, the real deterministic benchmark.
STORAGE_DIR is pointed at the copy so the real corpus, indexes and registry are never modified."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REAL = Path(__file__).resolve().parents[2] / "storage"
topic = sys.argv[1] if len(sys.argv) > 1 else "retrieval-augmented generation"
tmp = Path(tempfile.mkdtemp(prefix="storage_auto_"))
for sub in ("registry", "indexes", "embedding_cache", "benchmark", "models"):
    shutil.copytree(REAL / sub, tmp / sub)
shutil.copy2(REAL / "papers.sqlite", tmp / "papers.sqlite")
os.environ["STORAGE_DIR"] = str(tmp)

from backend.agents.services import Services          # noqa: E402  (after STORAGE_DIR is set)
from backend.agents.workflows import build_system      # noqa: E402


def brief(r):
    return {k: {"status": v.status, "s": round(v.seconds, 1), "attempts": v.attempts, "error": v.error,
                "out": {a: (b if not isinstance(b, (list, dict)) or len(str(b)) < 300 else f"<{type(b).__name__} {len(b)}>") for a, b in v.output.items()}}
            for k, v in r.steps.items()}


def main():
    sv = Services.create(cfg_key="recursive-256-15")
    orch = build_system(sv)
    out = {"storage_copy": str(tmp), "topic": topic}
    n0 = sv.repo.count("chunks", "WHERE config_key=?", (sv.cfg_key,))
    p0 = sv.repo.count("papers")
    t = time.perf_counter()
    r = orch.run_workflow("ingest_topic", {"topic": topic})
    out["ingest"] = {"status": r.status, "seconds": round(time.perf_counter() - t, 1), "steps": brief(r),
                     "papers_before_after": [p0, sv.repo.count("papers")], "chunks_before_after": [n0, sv.repo.count("chunks", "WHERE config_key=?", (sv.cfg_key,))]}
    print("INGEST", r.status, json.dumps(out["ingest"]["papers_before_after"]), json.dumps(out["ingest"]["chunks_before_after"]), flush=True)

    # does serving now find a newly ingested paper?
    pr = sv.get_production()
    new = [p for p in sv.repo.list_papers() if p.status == "processed" and p.enrichment][-3:]
    served = []
    for p in new:
        res = pr.search(p.title, 3)
        served.append({"title": p.title[:70], "top_hit_is_this_paper": res["results"][0]["document_id"] == p.paper_id if res["results"] else False,
                       "keywords": [k for k, _ in p.enrichment.get("keywords", [])][:5], "summary": (p.enrichment.get("summary") or "")[:160]})
    out["serving_check"] = served

    r = orch.run_workflow("monitor")
    out["monitor"] = {"status": r.status, "check": r.context["check"], "steps": {k: v.status for k, v in r.steps.items()}}
    print("MONITOR", r.status, r.context["check"]["benchmark_trigger_reasons"], flush=True)

    t = time.perf_counter()
    r = orch.run_workflow("benchmark_and_deploy")
    out["benchmark_and_deploy"] = {"status": r.status, "seconds": round(time.perf_counter() - t, 1), "steps": brief(r)}
    print("BENCH", r.status, round(time.perf_counter() - t), "s", flush=True)

    r = orch.run_workflow("answer_question", {"question": "How does retrieval-augmented generation reduce hallucinations?"})
    a = r.context["answer"]["response"]
    out["answer_question"] = {"status": r.status, "mode": a["answer_mode"], "gate": a["evidence"], "generation_errors": a["generation"]["provider_errors"],
                              "answer": a["answer"][:300], "explain_status": r.steps["explain"].status}
    out["jobs"] = [(j["kind"], j["status"]) for j in sv.repo.jobs(40)][:25]
    (REAL / "benchmark" / "automation_demo.json").write_text(json.dumps(out, indent=1, default=str))
    print("done", flush=True)
    orch.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
