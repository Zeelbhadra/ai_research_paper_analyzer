"""Workflow definitions and system assembly."""
from __future__ import annotations

from backend.agents.discovery_agent import DiscoveryAgent
from backend.agents.document_agent import DocumentAgent
from backend.agents.monitoring_agent import MonitoringAgent
from backend.agents.optimization_agent import OptimizationAgent
from backend.agents.orchestrator import Orchestrator, Step
from backend.agents.research_agent import ExplainabilityAgent, ResearchAssistantAgent
from backend.agents.services import Services

WORKFLOWS: dict[str, list[Step]] = {
    # discover -> download -> extract/chunk/enrich -> update indexes -> make the serving path see them
    "ingest_topic": [
        Step("discover", "discovery", "search", inputs={"topic": "params.topic"}, params={}, retries=2),
        Step("download", "discovery", "download", inputs={"paper_ids": "discover.new_paper_ids"}, when="discover.new_paper_ids"),
        Step("process", "document", "process", inputs={"paper_ids": "download.ready_ids"}, when="download.ready_ids"),
        Step("index", "document", "update_indexes", when="process.processed"),
        Step("refresh", "research", "refresh", when="process.processed"),
    ],
    # manual uploads follow the same path from extraction on
    "ingest_uploads": [
        Step("process", "document", "process_files", inputs={"paths": "params.paths"}),
        Step("index", "document", "update_indexes"),
        Step("refresh", "research", "refresh"),
    ],
    # deterministic re-evaluation; promotion is guarded inside the benchmark (thresholds + validation probe)
    "benchmark_and_deploy": [
        Step("benchmark", "optimization", "benchmark", retries=0),
        Step("deploy", "optimization", "deploy", retries=0),
        Step("report", "monitoring", "report", required=False),
    ],
    # periodic health review; triggers a benchmark only when a documented rule fires
    "monitor": [
        Step("check", "monitoring", "check"),
        Step("benchmark", "optimization", "benchmark", retries=0, when="check.trigger_benchmark", required=False),
        Step("deploy", "optimization", "deploy", retries=0, when="check.trigger_benchmark", required=False),
        Step("report", "monitoring", "report", required=False),
    ],
    "report": [Step("report", "monitoring", "report")],
    # user-triggered question with an evidence explanation
    "answer_question": [
        Step("answer", "research", "ask", inputs={"question": "params.question"}, retries=0),
        Step("explain", "explain", "explain_answer", inputs={"response": "answer.response"}, required=False, retries=0),
    ],
}


def build_system(sv: Services, pipeline_factory=None, **agent_overrides) -> Orchestrator:
    orch = Orchestrator(sv.repo, sv.events, sv.metrics)
    for agent in (agent_overrides.get("discovery") or DiscoveryAgent(sv), agent_overrides.get("document") or DocumentAgent(sv),
                  agent_overrides.get("optimization") or OptimizationAgent(sv), ResearchAssistantAgent(sv, pipeline_factory),
                  ExplainabilityAgent(sv), agent_overrides.get("monitoring") or MonitoringAgent(sv)):
        orch.register(agent)
    for name, steps in WORKFLOWS.items():
        orch.define(name, steps)
    # inter-agent messaging: failures are recorded as metrics the monitoring agent reads
    orch.subscribe("task_failed", lambda p: sv.metrics.record("agent_task_failed", 1, agent=p["agent"], action=p["action"]))
    return orch
