import copy
import time
from datetime import datetime
from pathlib import Path

import pytest

from backend.agents.discovery_agent import DiscoveryAgent
from backend.agents.document_agent import DocumentAgent, parse_cfg_key
from backend.agents.monitoring_agent import MonitoringAgent
from backend.agents.orchestrator import (Agent, AgentError, AlreadyRunning, Orchestrator, Step, TransientError, resolve)
from backend.agents.scheduler import AutomationScheduler, job_specs
from backend.agents.services import Services, load_automation
from backend.agents.workflows import WORKFLOWS, build_system
from backend.benchmarking.model_registry import ModelRegistry
from backend.database.repositories import Repository
from backend.ingestion.models import HttpError, PaperRecord
from backend.tests.conftest import make_pdf_bytes


# ---------------- orchestrator ----------------
class Toy(Agent):
    name = "toy"

    def __init__(self):
        self.calls, self.fail_times = {}, {}

    def actions(self):
        return {"ok": lambda x=1: {"value": x * 2}, "flaky": self.flaky, "boom": self.boom, "echo": lambda **kw: dict(kw),
                "slow": lambda: (time.sleep(0.3), {"done": True})[1]}

    def flaky(self):
        self.calls["flaky"] = self.calls.get("flaky", 0) + 1
        if self.calls["flaky"] <= self.fail_times.get("flaky", 2):
            raise HttpError("503 from upstream")
        return {"attempts_needed": self.calls["flaky"]}

    def boom(self):
        self.calls["boom"] = self.calls.get("boom", 0) + 1
        raise ValueError("logic bug")


@pytest.fixture
def orch(repo):
    sleeps = []
    o = Orchestrator(repo, sleep=sleeps.append, backoff=1.0)
    o.sleeps = sleeps
    o.toy = Toy()
    o.register(o.toy)
    return o


def test_transient_errors_retry_with_exponential_backoff_then_succeed(orch):
    r = orch.run_task("toy", "flaky", retries=3)
    assert r.status == "success" and r.attempts == 3 and r.output == {"attempts_needed": 3}
    assert orch.sleeps == [1.0, 2.0]


def test_transient_errors_give_up_after_retries_and_permanent_errors_do_not_retry(orch):
    orch.toy.fail_times["flaky"] = 99
    r = orch.run_task("toy", "flaky", retries=2)
    assert r.status == "failed" and r.attempts == 3 and "HttpError" in r.error
    b = orch.run_task("toy", "boom", retries=5)
    assert b.status == "failed" and b.attempts == 1 and orch.toy.calls["boom"] == 1


def test_unknown_agent_or_action_and_job_records(orch, repo):
    assert orch.run_task("nope", "x").status == "failed" and "unknown agent" in orch.run_task("nope", "x").error
    assert "no action" in orch.run_task("toy", "zzz").error
    orch.run_task("toy", "ok", {"x": 4})
    orch.run_task("toy", "boom")
    jobs = {j["kind"]: j["status"] for j in repo.jobs()}
    assert jobs["toy.ok"] == "success" and jobs["toy.boom"] == "failed"


def test_failure_publishes_message_to_subscribers(orch):
    seen = []
    orch.subscribe("task_failed", seen.append)
    orch.run_task("toy", "boom")
    assert seen and seen[0]["agent"] == "toy" and seen[0]["action"] == "boom"


def test_workflow_passes_outputs_between_steps_and_resolves_paths(orch):
    orch.define("wf", [Step("a", "toy", "ok", inputs={"x": "params.n"}), Step("b", "toy", "echo", inputs={"got": "a.value"}, params={"fixed": 1})])
    r = orch.run_workflow("wf", {"n": 21})
    assert r.status == "success" and r.context["b"] == {"got": 42, "fixed": 1}
    with pytest.raises(AgentError):
        resolve({"a": {}}, "a.missing")


def test_required_failure_skips_rest_optional_failure_is_partial_and_conditions(orch):
    orch.define("hard", [Step("a", "toy", "boom", retries=0), Step("b", "toy", "ok")])
    r = orch.run_workflow("hard")
    assert r.status == "failed" and r.steps["b"].status == "skipped"
    orch.define("soft", [Step("a", "toy", "boom", required=False, retries=0), Step("b", "toy", "ok")])
    assert orch.run_workflow("soft").status == "partial"
    orch.define("cond", [Step("a", "toy", "ok"), Step("b", "toy", "ok", when="params.go"), Step("c", "toy", "echo", inputs={"v": "nothing.here"})])
    r = orch.run_workflow("cond", {"go": False})
    assert r.steps["b"].status == "skipped" and r.steps["c"].status == "failed" and "not available" in r.steps["c"].error


def test_only_one_run_per_workflow_at_a_time(orch):
    orch.define("slow", [Step("s", "toy", "slow")])
    fut = orch.submit("slow")
    time.sleep(0.05)
    assert orch.is_running("slow")
    with pytest.raises(AlreadyRunning):
        orch.submit("slow")
    assert fut.result(timeout=5).status == "success" and not orch.is_running("slow")
    orch.shutdown()


# ---------------- fixtures for real agents ----------------
@pytest.fixture
def sv(tmp_path):
    auto = copy.deepcopy(load_automation())
    auto["enrichment"] = {"keywords": False, "summary": True, "topics": False}
    auto["discovery"]["download_delay_seconds"] = 0
    s = Services.create(tmp_path / "storage", automation=auto)
    s.settings.data_dir = tmp_path / "data"
    return s


def tfidf_production(sv):
    reg = ModelRegistry(sv.storage / "registry")
    cand = {"name": "tfidf", "family": "tfidf", "checkpoint": "c", "version": "tfidf:v", "dim": None, "index_location": "x",
            "overall_score": 50.0, "eligible": True, "fail_reasons": [], "metrics": {}, "config": {"_name": "tfidf"}}
    rid = reg.record_run({"selection_split": "validation", "n_queries": 5, "n_chunks": 10, "dataset_version": "d"}, [cand], {"name": "tfidf"})
    reg.promote(rid, "tfidf", "test")
    return reg


def rec(i, pdf=True, **kw):
    return PaperRecord(title=f"Graph neural networks for scientific retrieval study number {i}", source="arxiv", arxiv_id=f"2401.0000{i}",
                       abstract="We study retrieval of scientific text with graph neural networks and report results.", authors=["A"],
                       abs_url=f"http://arxiv.org/abs/2401.0000{i}", pdf_url=f"http://x/{i}.pdf" if pdf else None, **kw)


# ---------------- discovery agent ----------------
def test_discovery_dedupes_stores_and_survives_a_failing_source(sv):
    def arxiv(topic, n, raise_on_error=False):
        return iter([rec(1), rec(2)])

    def s2(topic, n, raise_on_error=False):
        raise HttpError("429 from api.semanticscholar.org")
    d = DiscoveryAgent(sv, arxiv, s2, downloader=lambda r: None, sleep=lambda s: None)
    out = d.search("graph retrieval")
    assert out["new"] == 2 and out["partial"] and out["source_status"]["semantic_scholar"]["ok"] is False
    assert {p.status for p in sv.repo.list_papers()} == {"discovered"}
    again = d.search("graph retrieval")
    assert again["new"] == 0 and again["duplicates"] == 2                      # nothing re-ingested


def test_discovery_all_sources_failing_is_transient_and_retried_by_orchestrator(sv, repo):
    n = {"c": 0}

    def bad(topic, n_, raise_on_error=False):
        n["c"] += 1
        raise HttpError("503")
    o = Orchestrator(sv.repo, sleep=lambda s: None)
    o.register(DiscoveryAgent(sv, bad, bad))
    r = o.run_task("discovery", "search", {"topic": "x"}, retries=2)
    assert r.status == "failed" and r.attempts == 3 and n["c"] == 6
    with pytest.raises(AgentError):
        DiscoveryAgent(sv, bad, bad).search("  ")


def test_download_marks_status_skips_existing_and_reports_failures(sv, tmp_path):
    papers = [rec(1), rec(2), rec(3, pdf=False)]
    d0 = DiscoveryAgent(sv, lambda *a, **k: iter(papers), lambda *a, **k: iter([]), downloader=lambda r: None, sleep=lambda s: None)
    ids = d0.search("x")["new_paper_ids"]
    pdfdir = tmp_path / "pdfs"
    pdfdir.mkdir()

    def fake_dl(r):
        if r.arxiv_id.endswith("2"):
            return None                                                      # simulated failed download
        p = pdfdir / f"{r.arxiv_id}.pdf"
        p.write_bytes(make_pdf_bytes(title=r.title))
        return p
    d = DiscoveryAgent(sv, lambda *a, **k: iter([]), lambda *a, **k: iter([]), downloader=fake_dl, sleep=lambda s: None)
    out = d.download(ids)
    assert len(out["downloaded"]) == 1 and len(out["failed"]) == 2
    assert {f["reason"] for f in out["failed"]} == {"download or PDF validation failed", "no open-access PDF URL"}
    again = d.download(ids)
    assert again["already_present"] == out["downloaded"]
    with pytest.raises(AgentError):
        d.download([ids[1]])


# ---------------- document agent + full ingest workflow ----------------
def test_parse_cfg_key():
    c = parse_cfg_key("section-512-20")
    assert (c.strategy, c.size, c.overlap_pct) == ("section", 512, 0.2)
    with pytest.raises(AgentError):
        parse_cfg_key("nonsense")


class FakeProd:
    """Stands in for ProductionRetrieval where the real one would load models."""
    def __init__(self):
        self.refreshed = 0

    def refresh_corpus(self):
        self.refreshed += 1
        return {"ok": True, "n_chunks": 123}


def test_ingest_topic_workflow_end_to_end_with_stubbed_network(sv, tmp_path):
    tfidf_production(sv)
    sv.production = FakeProd()
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()

    def dl(r):
        p = pdfs / (r.arxiv_id + ".pdf")
        p.write_bytes(make_pdf_bytes(title=r.title))
        return p
    disc = DiscoveryAgent(sv, lambda *a, **k: iter([rec(1), rec(2)]), lambda *a, **k: iter([rec(2)]), dl, lambda s: None)
    orch = build_system(sv, discovery=disc)
    r = orch.run_workflow("ingest_topic", {"topic": "graph retrieval"})
    assert r.status == "success", {k: (v.status, v.error) for k, v in r.steps.items()}
    assert r.steps["discover"].output["new"] == 2 and r.steps["discover"].output["duplicates"] == 1      # cross-source dedup
    assert len(r.steps["process"].output["processed"]) == 2
    assert r.steps["index"].output["indexes_built"] == ["tfidf"] and sv.production.refreshed == 1
    papers = sv.repo.list_papers()
    assert all(p.status == "processed" and p.enrichment.get("summary") for p in papers)
    assert sv.repo.count("chunks", "WHERE config_key=?", (sv.cfg_key,)) == r.steps["index"].output["n_chunks"] > 0
    # second run: nothing new => download/process/etc are skipped, and no duplicate papers appear
    r2 = orch.run_workflow("ingest_topic", {"topic": "graph retrieval"})
    assert r2.status == "success"                                              # nothing new is a clean no-op, not a failure
    assert r2.steps["discover"].output["new"] == 0 and len(sv.repo.list_papers()) == 2
    assert [r2.steps[s].status for s in ("download", "process", "index", "refresh")] == ["skipped"] * 4 and sv.production.refreshed == 1


def test_document_agent_logs_corrupt_pdf_and_continues(sv, tmp_path):
    good, bad = tmp_path / "g.pdf", tmp_path / "b.pdf"
    good.write_bytes(make_pdf_bytes())
    bad.write_bytes(b"not a pdf" * 300)
    for i, p in enumerate((good, bad)):
        r = rec(i + 5)
        r.paper_id, r.pdf_path, r.status = f"arxiv:2401.0000{i + 5}", str(p), "downloaded"
        sv.repo.upsert_paper(r)
    out = DocumentAgent(sv).process(["arxiv:2401.00005", "arxiv:2401.00006"], enrich={})
    assert out["processed"] == ["arxiv:2401.00005"] and out["failed"][0]["paper_id"] == "arxiv:2401.00006"
    assert sv.repo.count("extraction_failures") == 1
    with pytest.raises(AgentError):
        DocumentAgent(sv).process(["arxiv:2401.00006"], enrich={})


# ---------------- monitoring ----------------
class Clock:
    def __init__(self, t=None):
        self.t = t or time.time()

    def __call__(self):
        return self.t


def test_monitoring_alert_rules_and_minimum_traffic(sv):
    clock = Clock()
    m = MonitoringAgent(sv, clock)
    for i in range(5):                                                         # too little traffic: no traffic-based alerts
        sv.metrics.record("retrieval_latency_ms", 9000, ts=clock.t - 60)
    assert not [a for a in m.check()["alerts"] if a["type"] == "latency_degradation"]
    for i in range(20):
        sv.metrics.record("retrieval_latency_ms", 900, ts=clock.t - 60)
        sv.metrics.record("fallback_used", 1.0 if i % 2 else 0.0, ts=clock.t - 60)
    kinds = {a["type"] for a in m.check()["alerts"]}
    assert {"latency_degradation", "high_fallback_rate"} <= kinds
    sv.metrics.record("retrieval_latency_ms", 1, ts=clock.t - 99999)          # outside the window: ignored
    assert m.check()["retrieval_latency_ms"]["n"] == 25


def test_monitoring_benchmark_triggers_and_cooldown(sv):
    clock = Clock()
    m = MonitoringAgent(sv, clock)
    assert m.check()["benchmark_trigger_reasons"] == ["no benchmark has been run yet"]
    reg = tfidf_production(sv)
    run_ts = datetime.strptime(reg.latest_run()["timestamp"], "%Y-%m-%dT%H:%M:%S").timestamp()
    clock.t = run_ts + 3600                                                    # inside the 24h cooldown
    assert m.check()["benchmark_trigger_reasons"] == []
    clock.t = run_ts + 3 * 86400                                               # cooldown over, corpus 0 -> not grown, not old
    assert m.check()["benchmark_trigger_reasons"] == []
    sv.repo.replace_chunks("d", sv.cfg_key, [__import__("backend.preprocessing.chunking", fromlist=["Chunk"]).Chunk("d", "T", f"c{i}", 1, 1, "x", "t" * 100, "u") for i in range(20)])
    reasons = m.check()["benchmark_trigger_reasons"]                           # run meta says 10 chunks -> now 20 (+100%)
    assert len(reasons) == 1 and "corpus grew" in reasons[0]
    clock.t = run_ts + 40 * 86400
    assert any("days old" in r for r in m.check()["benchmark_trigger_reasons"])


def test_monitor_workflow_runs_benchmark_only_when_triggered(sv):
    tfidf_production(sv)

    class StubOpt(Agent):
        name = "optimization"
        calls = []

        def actions(self):
            return {"benchmark": lambda **k: (self.calls.append("benchmark") or {"run_id": "x"}), "deploy": lambda: (self.calls.append("deploy") or {})}
    stub = StubOpt()
    clock = Clock()
    orch = build_system(sv, optimization=stub, monitoring=MonitoringAgent(sv, clock))
    run_ts = datetime.strptime(sv.registry.latest_run()["timestamp"], "%Y-%m-%dT%H:%M:%S").timestamp()
    clock.t = run_ts + 3600
    r = orch.run_workflow("monitor")
    assert r.steps["benchmark"].status == "skipped" and stub.calls == []
    assert Path(r.steps["report"].output["path"]).exists()
    clock.t = run_ts + 90 * 86400
    r = orch.run_workflow("monitor")
    assert r.steps["benchmark"].status == "success" and stub.calls == ["benchmark", "deploy"]


# ---------------- scheduler ----------------
def test_job_specs_and_scheduler_registration_and_skip_when_running(sv):
    cfg = copy.deepcopy(sv.automation)
    cfg["discovery"]["topics"] = ["a b", "c d"]
    ids = [s["id"] for s in job_specs(cfg)]
    assert ids == ["discover_0", "discover_1", "monitor", "benchmark", "report"]
    orch = build_system(sv)
    sch = AutomationScheduler(orch, cfg)
    assert not sch.enabled() and sch.register() == ids and {j.id for j in sch.sched.get_jobs()} == set(ids)
    cfg2 = dict(cfg, enabled=True)
    assert AutomationScheduler(orch, cfg2).enabled()
    orch.define("slow", [Step("s", "monitoring", "report")])
    orch._running.add("slow")                                                  # simulate an in-flight run
    sch._tick("slow", {})                                                      # must not raise or start a second run
    orch._running.discard("slow")
    assert set(WORKFLOWS) >= {"ingest_topic", "ingest_uploads", "benchmark_and_deploy", "monitor", "report", "answer_question"}


def test_pipeline_api_validation_and_conflict(monkeypatch, sv):
    import app as flask_app
    from backend.api.routes import pipeline as route
    orch = build_system(sv)
    sched = AutomationScheduler(orch, sv.automation)
    monkeypatch.setattr(route, "get_system", lambda: (sv, orch, sched))
    c = flask_app.app.test_client()
    assert c.post("/api/pipeline/run", json={"workflow": "rm -rf"}).status_code == 400
    assert c.post("/api/pipeline/run", json={"workflow": "ingest_topic", "params": {}}).status_code == 400
    orch._running.add("report")
    assert c.post("/api/pipeline/run", json={"workflow": "report"}).status_code == 409
    orch._running.discard("report")
    assert c.get("/api/pipeline/status").get_json()["scheduler"]["enabled"] is False
    assert c.get("/api/pipeline/report").status_code == 404


def test_budgeted_evidence_coverage_is_granularity_fair():
    from backend.benchmarking.retrieval_evaluator import budgeted_evidence_coverage
    words = [f"w{i}" for i in range(300)]
    ev = [{"paper_id": "A", "text": " ".join(words[:60])}]
    text = {"big": " ".join(words), "small": " ".join(words[:60]), "other": " ".join(f"z{i}" for i in range(60)), "wrongdoc": " ".join(words[:60])}
    doc = {"big": "A", "small": "A", "other": "A", "wrongdoc": "B"}
    # a 300-word chunk exceeds a 100-word budget: it is only taken if it is the first (always include >=1)
    assert budgeted_evidence_coverage(["big"], text, doc, ev, 100) == 1.0
    assert budgeted_evidence_coverage(["other", "big"], text, doc, ev, 100) == 0.0          # budget spent on an irrelevant chunk
    assert budgeted_evidence_coverage(["other", "small"], text, doc, ev, 150) == 1.0        # small chunk fits and covers the evidence
    assert budgeted_evidence_coverage(["wrongdoc"], text, doc, ev, 100) == 0.0              # same text in another paper does not count
