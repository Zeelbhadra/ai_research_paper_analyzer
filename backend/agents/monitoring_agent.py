"""Evaluation & Monitoring Agent: tracks serving metrics and job health, flags degradation with fixed, documented rules,
decides when a benchmark re-run is warranted, and writes reports. Deterministic: thresholds come from automation.json."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from backend.agents.orchestrator import Agent
from backend.agents.services import Services


class MonitoringAgent(Agent):
    name = "monitoring"

    def __init__(self, sv: Services, clock: Callable[[], float] = time.time):
        self.sv, self.clock = sv, clock

    def actions(self):
        return {"check": self.check, "report": self.report}

    def _cfg(self) -> dict:
        return self.sv.automation["monitoring"]

    @staticmethod
    def _parse_ts(ts: str) -> float:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").timestamp()

    def check(self) -> dict:
        c, now, m = self._cfg(), self.clock(), self.sv.metrics
        since = now - c["window_minutes"] * 60
        lat = m.stats("retrieval_latency_ms", since)
        fb = m.rate("fallback_used", since)
        fails = m.stats("retrieval_failure", since)["n"]
        replaced = m.rate("rag_answer_replaced", since)
        n_req = lat["n"] + fails
        alerts: list[dict] = []
        enough = n_req >= c["min_requests_for_alerts"]
        if enough and lat["n"] and lat["p95"] > c["latency_p95_ms"]:
            alerts.append({"type": "latency_degradation", "detail": f"retrieval p95 {lat['p95']:.0f} ms > {c['latency_p95_ms']} ms over {c['window_minutes']} min"})
        if enough and fb is not None and fb > c["max_fallback_rate"]:
            alerts.append({"type": "high_fallback_rate", "detail": f"{fb:.0%} of requests served by the TF-IDF fallback (limit {c['max_fallback_rate']:.0%})"})
        if enough and fails / n_req > c["max_retrieval_failure_rate"]:
            alerts.append({"type": "retrieval_failures", "detail": f"{fails}/{n_req} requests failed"})
        n_rag = m.stats("rag_total_s", since)["n"]
        if n_rag >= c["min_requests_for_alerts"] and replaced is not None and replaced > c["max_answer_replaced_rate"]:
            alerts.append({"type": "ungrounded_answers", "detail": f"{replaced:.0%} of LLM answers rejected by the citation check (limit {c['max_answer_replaced_rate']:.0%})"})
        failed_jobs = [j for j in self.sv.repo.jobs(200) if j["status"] == "failed" and (j["finished"] or 0) >= since]
        if failed_jobs:
            alerts.append({"type": "job_failures", "detail": f"{len(failed_jobs)} failed job(s) in the window: {sorted({j['kind'] for j in failed_jobs})[:5]}"})
        prod = self.sv.production
        if prod is not None and prod.service.snapshot()["fallback_active"]:
            alerts.append({"type": "fallback_active", "detail": prod.service.snapshot().get("reason")})

        reasons = self._benchmark_triggers(now)
        return {"window_minutes": c["window_minutes"], "requests": n_req, "retrieval_latency_ms": lat, "fallback_rate": fb,
                "retrieval_failures": fails, "answer_replaced_rate": replaced, "alerts": alerts,
                "trigger_benchmark": bool(reasons), "benchmark_trigger_reasons": reasons}

    def _benchmark_triggers(self, now: float) -> list[str]:
        """Re-benchmark when (a) the corpus grew materially since the last run, or (b) the last run is old. Cooldown applies."""
        c = self._cfg()
        run = self.sv.registry.latest_run()
        if run is None:
            return ["no benchmark has been run yet"]
        age_h = (now - self._parse_ts(run["timestamp"])) / 3600
        if age_h < c["benchmark_cooldown_hours"]:
            return []
        reasons = []
        n_then = run["meta"].get("n_chunks")
        n_now = self.sv.repo.count("chunks", "WHERE config_key=?", (self.sv.cfg_key,))
        if n_then and n_now >= n_then * (1 + c["corpus_growth_trigger"]):
            reasons.append(f"corpus grew {n_now / n_then - 1:.0%} since the last benchmark ({n_then} -> {n_now} chunks)")
        if age_h / 24 > c["max_benchmark_age_days"]:
            reasons.append(f"last benchmark is {age_h / 24:.0f} days old (limit {c['max_benchmark_age_days']})")
        return reasons

    def report(self) -> dict:
        now = self.clock()
        chk = self.check()
        reg = self.sv.registry
        run, prod = reg.latest_run(), reg.production
        jobs = self.sv.repo.jobs(500)
        by_status: dict[str, int] = {}
        for j in jobs:
            by_status[j["status"]] = by_status.get(j["status"], 0) + 1
        corpus = {"papers": self.sv.repo.count("papers"), "processed": self.sv.repo.count("papers", "WHERE status='processed'"),
                  "failed": self.sv.repo.count("papers", "WHERE status='failed'"), "chunks": self.sv.repo.count("chunks", "WHERE config_key=?", (self.sv.cfg_key,))}
        lines = [f"# Pipeline report - {datetime.fromtimestamp(now):%Y-%m-%d %H:%M}", "",
                 f"## Corpus\n- papers: {corpus['papers']} (processed {corpus['processed']}, failed {corpus['failed']})\n- chunks ({self.sv.cfg_key}): {corpus['chunks']}", "",
                 f"## Production model\n- {prod['name'] if prod else 'none'} (promoted {prod['promoted_at'] if prod else '-'})"]
        if run:
            lines += [f"- latest benchmark: {run['run_id']} at {run['timestamp']} on '{run['meta']['selection_split']}' ({run['meta']['n_queries']} queries); "
                      f"winner: {(run.get('winner') or {}).get('name')}"]
        lines += ["", f"## Serving ({chk['window_minutes']} min window)", f"- requests: {chk['requests']}",
                  f"- retrieval latency p50/p95 (ms): {chk['retrieval_latency_ms'].get('p50', '-')}/{chk['retrieval_latency_ms'].get('p95', '-')}",
                  f"- fallback rate: {chk['fallback_rate']}", f"- failures: {chk['retrieval_failures']}", "",
                  "## Jobs (last 500)", ", ".join(f"{k}: {v}" for k, v in sorted(by_status.items())) or "none", "",
                  "## Alerts"] + ([f"- **{a['type']}**: {a['detail']}" for a in chk["alerts"]] or ["- none"]) + [
                  "", "## Benchmark triggers"] + ([f"- {r}" for r in chk["benchmark_trigger_reasons"]] or ["- none"])
        out_dir = Path(self.sv.storage / "reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"report_{datetime.fromtimestamp(now):%Y%m%d_%H%M%S}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        (out_dir / "latest.json").write_text(json.dumps({"check": chk, "corpus": corpus, "jobs": by_status, "markdown": str(path)}, indent=1, default=str))
        return {"path": str(path), "alerts": chk["alerts"], "trigger_benchmark": chk["trigger_benchmark"], "corpus": corpus}
