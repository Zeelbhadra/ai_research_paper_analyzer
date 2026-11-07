"""Pipeline API: job status, user-triggered workflows, monitoring report."""
from __future__ import annotations

import json
import logging
import re
import threading

from flask import Blueprint, jsonify, request

from backend.agents.orchestrator import AlreadyRunning

bp = Blueprint("pipeline", __name__, url_prefix="/api/pipeline")
log = logging.getLogger(__name__)
_sys = None
_lock = threading.Lock()
ALLOWED_USER_WORKFLOWS = {"ingest_topic", "benchmark_and_deploy", "monitor", "report", "answer_question"}


def get_system():
    """Lazy singleton: (Services, Orchestrator, AutomationScheduler). Shares the retrieval route's production object."""
    global _sys
    with _lock:
        if _sys is None:
            from backend.agents.scheduler import AutomationScheduler
            from backend.agents.services import Services
            from backend.agents.workflows import build_system
            from backend.api.routes.retrieval import get_production
            sv = Services.create(pr_factory=get_production)
            orch = build_system(sv)
            sched = AutomationScheduler(orch, sv.automation)
            if sched.enabled():
                sched.start()
            _sys = (sv, orch, sched)
        return _sys


@bp.get("/status")
def status():
    sv, orch, sched = get_system()
    s = orch.status()
    s["scheduler"] = {"enabled": sched.enabled(), "running": sched.sched.running,
                      "jobs": [{"id": j.id, "next_run": str(j.next_run_time)} for j in sched.sched.get_jobs()] if sched.sched.running else []}
    return jsonify(s)


@bp.get("/jobs")
def jobs():
    sv, _, _ = get_system()
    return jsonify(sv.repo.jobs(min(int(request.args.get("limit", 50)), 500)))


@bp.get("/failures")
def failures():
    sv, _, _ = get_system()
    latest: dict = {}
    for f in sv.repo.failures():                       # newest first; keep one row per document
        latest.setdefault(f["path"], f)
    out = []
    for f in latest.values():                          # never expose local directory structure to the browser
        out.append({**f, "path": re.split(r"[\\/]", f["path"])[-1], "reason": re.sub(r"[A-Za-z]:[\\/][^;]*[\\/]", "", f["reason"])})
    return jsonify(out)


@bp.post("/run")
def run():
    body = request.get_json(silent=True) or {}
    wf, params = body.get("workflow"), body.get("params") or {}
    if wf not in ALLOWED_USER_WORKFLOWS:
        return jsonify({"error": f"workflow must be one of {sorted(ALLOWED_USER_WORKFLOWS)}"}), 400
    if not isinstance(params, dict):
        return jsonify({"error": "params must be an object"}), 400
    if wf == "ingest_topic" and not str(params.get("topic", "")).strip():
        return jsonify({"error": "params.topic is required"}), 400
    _, orch, _ = get_system()
    try:
        orch.submit(wf, params)
    except AlreadyRunning as e:
        return jsonify({"error": str(e)}), 409
    return jsonify({"started": wf}), 202


@bp.get("/report")
def report():
    sv, _, _ = get_system()
    p = sv.storage / "reports" / "latest.json"
    if not p.exists():
        return jsonify({"error": "no report generated yet; POST /api/pipeline/run {\"workflow\": \"report\"}"}), 404
    return jsonify(json.loads(p.read_text()))
