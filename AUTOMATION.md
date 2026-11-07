# Orchestration, agents and automation (Step 9)

## Design stance
The pipeline's control flow is **deterministic Python**. There is no LLM in the orchestration path: benchmark scores, winner selection, promotion, retries and alerts are computed by code. The only LLM use in the system is answer generation inside the Research Assistant, behind the verification guard (Step 7).
"Agents" are plain-Python components with named actions; they exist where a responsibility boundary is useful, not as autonomous decision makers. Infrastructure: **APScheduler** in-process (no Celery/Redis: a single-node batch workload gains nothing from a broker).

## Components (`backend/agents/`)
| Agent | Actions | Responsibility |
|---|---|---|
| Discovery | `search`, `download` | arXiv + Semantic Scholar search, cross-source dedup against the stored corpus, open-access PDF download only (never bypasses access controls), status tracking. One failing source = partial success; all sources failing = transient error (retried). |
| Document intelligence | `process`, `process_files`, `update_indexes` | extraction, cleaning, sectioning, chunking, enrichment with the **original project's modules** (KeyBERT keywords, TF-IDF extractive summary; zero-shot topic classification available, off by default), index refresh for the production model's components (embeddings cached per chunk text, so only new chunks cost model time). |
| Retrieval optimization | `benchmark`, `deploy`, `chunk_sweep` | runs `run_benchmark` (deterministic), refreshes **every** candidate's index first so models are compared over the same current corpus, deploys via the guarded registry sync, sweeps chunking configs. |
| Research assistant | `ask`, `compare`, `research_gaps`, `refresh` | RAG pipeline wrappers; `refresh` makes serving see newly indexed papers (the new retriever must pass a probe before it replaces the old one). |
| Explainability | `explain_answer`, `explain_retrieval` | evidence explanation and retrieval explanation (Step 8). |
| Monitoring | `check`, `report` | metrics-driven degradation rules, benchmark triggers, markdown/JSON reports. |

## Orchestrator (`orchestrator.py`)
- `run_task`: retries **only transient** failures (`TransientError`, HTTP errors, connection/timeouts) with exponential backoff; logic errors fail immediately. Every task is a row in the `jobs` table; failures are published on a small message bus (`task_failed`) which feeds the metrics store.
- `run_workflow`: ordered steps; outputs flow through a shared context via dotted paths (`discover.new_paper_ids`); steps can be optional (`required=False` -> workflow `partial`) or conditional (`when=`; a missing producer means "not met"). A failed required step skips the rest.
- One run per workflow at a time (`AlreadyRunning`); scheduled ticks that collide are skipped, user requests get HTTP 409.

## Workflows
`ingest_topic` (discover -> download -> process -> update indexes -> refresh serving; a run that finds nothing new is a clean no-op), `ingest_uploads`, `benchmark_and_deploy`, `monitor` (check -> benchmark+deploy only when a rule fires -> report), `report`, `answer_question` (ask -> explain).

## Scheduling (`scheduler.py`, `config/automation.json`)
Scheduled discovery per topic (cron), monitoring (interval), weekly benchmark, daily report. **Off by default**: set `"enabled": true` or `AUTOMATION=1`. User-triggered runs use the same code path: `POST /api/pipeline/run {"workflow": ..., "params": ...}` (allow-listed names; 202 started / 409 already running). Also `GET /api/pipeline/status|jobs|failures|report`.

## Monitoring rules (deterministic, thresholds in `automation.json`)
Computed over a sliding window from the `metrics` table that the serving path writes (retrieval latency, fallback use, failures, RAG latency, answer-rejection and citation validity):
- p95 retrieval latency above the SLO; fallback rate above the limit; retrieval failure rate above the limit; share of LLM answers rejected by the citation check above the limit; failed jobs in the window; fallback currently active. Traffic-based alerts need a minimum request count to avoid noise.
- **Benchmark triggers:** corpus grew >= 20% in chunks since the last benchmark, or the last benchmark is older than 30 days; a 24 h cooldown prevents repeated runs. Latency or fallback alerts are incidents, not benchmark triggers.
- Promotion after a triggered benchmark still goes through the same thresholds, non-inferiority test and health/probe validation as a manual run.

## Verified on live data (`python -m backend.benchmarking.demo_automation`, run on a copy of storage; output `storage/benchmark/automation_demo.json`)
Topic "retrieval-augmented generation":
| Step | Result |
|---|---|
| discover | 10 papers from arXiv (31 s); Semantic Scholar returned HTTP 429 after 5 retries -> recorded as partial, run continued |
| download | 10/10 open-access PDFs (122 s, 1 s politeness delay) |
| process | 10/10 extracted -> 344 chunks, keywords + extractive summary stored (21 s) |
| update indexes | TF-IDF rebuilt; E5: only the 344 new chunks embedded (110 s) |
| refresh | serving swapped to the rebuilt retriever after a probe; 2,222 chunks served |
| serving check | 3/3 spot-checked new papers were the top hit for their own title |
| monitor | no alerts; no benchmark trigger (corpus +18% < 20%; correct) |
| benchmark_and_deploy | all candidate indexes refreshed incrementally (SBERT/BGE 344 new embeddings each, E5 0); hybrid remained the winner; no switch needed (166 s) |
| answer_question | OpenAI 429 recorded, extractive answer + explanation produced |

## Limits / not done
- Semantic Scholar needs an API key for reliable use (`SEMANTIC_SCHOLAR_API_KEY`); without it the shared public pool rate-limits.
- Scanned PDFs are detected and logged, not OCR'd.
- The benchmark reuses the last isolated latency/memory measurements; re-run `performance_benchmark --all` after hardware or model changes.
- New papers have no labeled queries; benchmark quality numbers still come from the original 89-query set (now with more distractor chunks).
- The scheduler and thread pool were tested for registration/skip logic, not for long-running multi-day operation or concurrent heavy workflows.
- Multi-worker deployments (gunicorn with several workers) would each create their own orchestrator/scheduler; run one scheduler process.
