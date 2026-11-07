"""APScheduler wiring: scheduled discovery/ingestion, monitoring, benchmarking and reports.

Off unless automation.json 'enabled' is true or AUTOMATION=1. APScheduler (in-process, no broker) is sufficient because
the workload is a single-node batch pipeline; Celery/Redis would add infrastructure without a benefit here.
A workflow that is still running is never started twice (the orchestrator raises AlreadyRunning and the tick is skipped).
"""
from __future__ import annotations

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.agents.orchestrator import AlreadyRunning, Orchestrator

log = logging.getLogger(__name__)


def job_specs(cfg: dict) -> list[dict]:
    """Pure function (easy to test): the jobs the scheduler will register."""
    specs = []
    d = cfg["discovery"]
    for i, topic in enumerate(d["topics"]):
        specs.append({"id": f"discover_{i}", "workflow": "ingest_topic", "params": {"topic": topic}, "trigger": ("cron", d["cron"])})
    specs.append({"id": "monitor", "workflow": "monitor", "params": {}, "trigger": ("interval_minutes", cfg["monitoring"]["interval_minutes"])})
    specs.append({"id": "benchmark", "workflow": "benchmark_and_deploy", "params": {}, "trigger": ("cron", cfg["benchmark"]["cron"])})
    specs.append({"id": "report", "workflow": "report", "params": {}, "trigger": ("cron", cfg["report"]["cron"])})
    return specs


def _trigger(t):
    kind, val = t
    return CronTrigger.from_crontab(val) if kind == "cron" else IntervalTrigger(minutes=val)


class AutomationScheduler:
    def __init__(self, orch: Orchestrator, cfg: dict):
        self.orch, self.cfg = orch, cfg
        self.sched = BackgroundScheduler(job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 600})

    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled")) or os.getenv("AUTOMATION") == "1"

    def _tick(self, workflow: str, params: dict) -> None:
        try:
            self.orch.submit(workflow, params)
        except AlreadyRunning:
            log.info("skipping scheduled %s: previous run still in progress", workflow)

    def register(self) -> list[str]:
        ids = []
        for s in job_specs(self.cfg):
            self.sched.add_job(self._tick, _trigger(s["trigger"]), args=[s["workflow"], s["params"]], id=s["id"], replace_existing=True)
            ids.append(s["id"])
        return ids

    def start(self) -> list[str]:
        ids = self.register()
        self.sched.start()
        log.info("automation scheduler started with jobs: %s", ids)
        return ids

    def stop(self) -> None:
        if self.sched.running:
            self.sched.shutdown(wait=False)

    def trigger_now(self, workflow: str, params: dict | None = None):
        """User-triggered run (same code path as scheduled)."""
        return self.orch.submit(workflow, params or {})
