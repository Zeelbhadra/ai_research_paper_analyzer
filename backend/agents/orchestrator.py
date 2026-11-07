"""Deterministic workflow orchestrator.

* Agents are plain-Python components exposing named actions; no LLM decides control flow, benchmark winners or
  production changes.
* Retries apply only to transient failures (network/HTTP/explicit TransientError); logic errors fail immediately.
* A workflow is an ordered list of steps; each step's output joins a shared context that later steps read via
  dotted paths ("discover.new_paper_ids"). Steps can be optional (continue on failure) or conditional (`when`).
* One run per workflow name at a time (heavy jobs must not overlap).
* Every task is recorded in the repository `jobs` table; inter-agent messages go through a small pub/sub bus that is
  also written to an event log.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from backend.database.repositories import Repository
from backend.ingestion.models import HttpError
from backend.monitoring.logger import EventLog

log = logging.getLogger(__name__)


class TransientError(Exception):
    """Retryable failure (network blip, rate limit, temporary unavailability)."""


class AgentError(Exception):
    """Permanent failure: do not retry."""


class AlreadyRunning(Exception):
    pass


TRANSIENT = (TransientError, HttpError, requests.ConnectionError, requests.Timeout, TimeoutError)


class Agent:
    name = "agent"

    def actions(self) -> dict[str, Callable[..., dict]]:
        raise NotImplementedError


@dataclass
class Step:
    name: str
    agent: str
    action: str
    params: dict = field(default_factory=dict)          # static parameters
    inputs: dict = field(default_factory=dict)          # parameter -> dotted path into context ("params.x" or "step.key")
    required: bool = True
    retries: int = 2
    when: str | None = None                             # dotted path that must be truthy for the step to run


@dataclass
class TaskResult:
    agent: str
    action: str
    status: str                                         # success | failed | skipped
    output: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    seconds: float = 0.0
    job_id: int | None = None


@dataclass
class WorkflowResult:
    name: str
    status: str                                         # success | partial | failed
    steps: dict[str, TaskResult]
    context: dict
    seconds: float
    job_id: int | None = None


def resolve(ctx: dict, path: str) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise AgentError(f"workflow input '{path}' is not available (missing '{part}')")
    return cur


class Orchestrator:
    def __init__(self, repo: Repository, events: EventLog | None = None, metrics=None, sleep: Callable[[float], None] = time.sleep,
                 max_workers: int = 2, backoff: float = 2.0):
        self.repo, self.events, self.metrics, self.sleep, self.backoff = repo, events or EventLog(), metrics, sleep, backoff
        self.agents: dict[str, Agent] = {}
        self.workflows: dict[str, list[Step]] = {}
        self._subs: dict[str, list[Callable[[dict], None]]] = {}
        self._running: set[str] = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="wf")

    # ---- registration / messaging ----
    def register(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    def define(self, name: str, steps: list[Step]) -> None:
        self.workflows[name] = steps

    def subscribe(self, topic: str, fn: Callable[[dict], None]) -> None:
        self._subs.setdefault(topic, []).append(fn)

    def publish(self, topic: str, payload: dict) -> None:
        self.events.emit("message", topic=topic, **{k: v for k, v in payload.items() if k != "kind"})
        for fn in self._subs.get(topic, []):
            try:
                fn(payload)
            except Exception:
                log.exception("subscriber for %s failed", topic)

    # ---- single task ----
    def run_task(self, agent: str, action: str, params: dict | None = None, retries: int = 2, kind: str | None = None) -> TaskResult:
        if agent not in self.agents:
            return TaskResult(agent, action, "failed", error=f"unknown agent '{agent}'")
        fn = self.agents[agent].actions().get(action)
        if fn is None:
            return TaskResult(agent, action, "failed", error=f"agent '{agent}' has no action '{action}'")
        job = self.repo.start_job(kind or f"{agent}.{action}")
        t0 = time.perf_counter()
        attempts, err = 0, None
        while True:
            attempts += 1
            try:
                out = fn(**(params or {})) or {}
                dt = time.perf_counter() - t0
                self.repo.finish_job(job, "success", json.dumps({"attempts": attempts, "output_keys": list(out)}, default=str))
                self._metric("task_seconds", dt, agent=agent, action=action)
                return TaskResult(agent, action, "success", out, None, attempts, dt, job)
            except TRANSIENT as e:
                err = f"{type(e).__name__}: {e}"
                if attempts > retries:
                    break
                delay = self.backoff * (2 ** (attempts - 1))
                log.warning("%s.%s transient failure (%s); retry %d/%d in %.1fs", agent, action, err, attempts, retries, delay)
                self.events.emit("task_retry", agent=agent, action=action, attempt=attempts, error=err)
                self.sleep(delay)
            except Exception as e:                       # permanent
                err = f"{type(e).__name__}: {e}"
                log.error("%s.%s failed permanently: %s\n%s", agent, action, err, traceback.format_exc(limit=3))
                break
        dt = time.perf_counter() - t0
        self.repo.finish_job(job, "failed", json.dumps({"attempts": attempts, "error": err}))
        self._metric("task_failure", 1, agent=agent, action=action)
        self.publish("task_failed", {"agent": agent, "action": action, "error": err, "attempts": attempts})
        return TaskResult(agent, action, "failed", {}, err, attempts, dt, job)

    def _metric(self, name, value, **tags):
        if self.metrics is not None:
            try:
                self.metrics.record(name, value, **tags)
            except Exception:
                log.exception("metric write failed")

    # ---- workflows ----
    def run_workflow(self, name: str, params: dict | None = None) -> WorkflowResult:
        if name not in self.workflows:
            raise KeyError(f"unknown workflow '{name}'")
        with self._lock:
            if name in self._running:
                raise AlreadyRunning(f"workflow '{name}' is already running")
            self._running.add(name)
        t0 = time.perf_counter()
        job = self.repo.start_job(f"workflow:{name}")
        ctx: dict = {"params": dict(params or {})}
        results: dict[str, TaskResult] = {}
        failed_required = False
        try:
            for st in self.workflows[name]:
                if failed_required:
                    results[st.name] = TaskResult(st.agent, st.action, "skipped", error="an earlier required step failed")
                    continue
                try:
                    if st.when is not None and not _truthy(_lookup(ctx, st.when)):
                        results[st.name] = TaskResult(st.agent, st.action, "skipped", error=f"condition '{st.when}' not met (or its step did not run)")
                        continue
                    p = dict(st.params)
                    p.update({k: resolve(ctx, path) for k, path in st.inputs.items()})
                except AgentError as e:
                    results[st.name] = TaskResult(st.agent, st.action, "failed", error=str(e))
                    failed_required = failed_required or st.required
                    continue
                r = self.run_task(st.agent, st.action, p, st.retries, kind=f"{name}:{st.name}")
                results[st.name] = r
                ctx[st.name] = r.output
                if r.status == "failed" and st.required:
                    failed_required = True
            statuses = [r.status for r in results.values()]
            status = "failed" if failed_required else ("partial" if "failed" in statuses else "success")
            dt = time.perf_counter() - t0
            self.repo.finish_job(job, status, json.dumps({n: r.status for n, r in results.items()}))
            self.publish("workflow_finished", {"workflow": name, "status": status, "seconds": round(dt, 2)})
            return WorkflowResult(name, status, results, ctx, dt, job)
        finally:
            with self._lock:
                self._running.discard(name)

    def submit(self, name: str, params: dict | None = None) -> Future:
        """Run a workflow in the background; raises AlreadyRunning immediately if one is in flight."""
        with self._lock:
            if name in self._running:
                raise AlreadyRunning(f"workflow '{name}' is already running")
        return self._pool.submit(self.run_workflow, name, params)

    def is_running(self, name: str) -> bool:
        with self._lock:
            return name in self._running

    def status(self) -> dict:
        with self._lock:
            running = sorted(self._running)
        return {"running": running, "workflows": sorted(self.workflows), "agents": {a: sorted(g.actions()) for a, g in self.agents.items()},
                "recent_jobs": self.repo.jobs(25)}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _lookup(ctx: dict, path: str) -> Any:
    """Like resolve(), but a missing path means 'condition not met' (the producing step was skipped)."""
    try:
        return resolve(ctx, path)
    except AgentError:
        return None


def _truthy(v: Any) -> bool:
    return bool(v)
