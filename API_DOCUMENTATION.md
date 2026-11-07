# API documentation

Base URL `http://localhost:5000`. JSON in and out; errors are `{"error": "..."}` with 400 (bad input), 404, 409 (workflow already running), 422 (unreadable PDF), 503 (retrieval unavailable) or 500. There is **no authentication**: do not expose the pipeline and retrieval-admin endpoints publicly. Responses keep the key order shown (Flask key sorting is off).

## Public vs admin
By default only the **reader API** is served: `/api/search`, `/api/analyze`, `/api/similar`, `/api/library/*`, `/api/chat`, `/api/chat/stream`, `/healthz`, `/readyz`. The dashboard, benchmark, pipeline, retrieval-admin and RAG-admin endpoints described below are registered **only when `ADMIN_API=1`** and return 404 otherwise. Sections below that describe them are admin-only. `/readyz` returns just `{"status"}` in public mode.

**Reader endpoints added in this version**
| Method | Path | Notes |
|---|---|---|
| POST | `/api/chat` | `{message, history?, paper_ids?}` -> `{answer, kind, sentences[{text,kind,sources}], sources[{n,cited,paper_id,title,pages,section,url,quote,text,highlights}], notes, suggestions, context}`. No model, score or provider fields. |
| POST | `/api/chat/stream` | same input, server-sent events: `start`, `token`, `final` (the object above), `error` (friendly message). |
| GET | `/api/library/papers`, `/paper`, `/facets` | processed papers only; cards and detail without chunk or status internals. |
| POST | `/api/library/add` | `{paper:{title,pdf_url,...}}`: queues download + processing; 422 without a PDF link, 409 if busy, 200 `ready` if already in the library. |
| GET | `/api/library/add/status` | progress of the add job. |

## Health
| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | liveness `{"status":"ok","uptime_s"}`; never touches a model |
| GET | `/readyz` | readiness: database, writable storage, production model + indexes present. 200 `{"status":"ready","checks":{...}}` or 503 `not_ready` listing what is missing. Does not force a model load. |

## Original PaperLens endpoints (kept, fixed)
| Method | Path | Notes |
|---|---|---|
| GET | `/api/search?q=&limit=10&source=all|arxiv|semantic_scholar` | `q` 1-300 chars, `limit` 1-20. Returns `{"results":[{title,authors,abstract,url,source,summary}],"count"}`. Summaries are cached (Mongo, if configured) LLM summaries, else an extractive summary (an OpenAI circuit breaker stops repeated failing calls). |
| POST | `/api/analyze` | JSON `{"text": "..."}` (>= 40 chars) **or** multipart `file` (PDF, <= 16 MB). Returns `{chars,pages,summary,keywords:[{word,score}],topics:[{label,score}]}`. 422 for scanned/corrupt PDFs. |
| POST | `/api/similar` | `{"text": "..."}` -> `{"similar":[{paper:{title,url,paper_id},similarity,section,pages}],"retrieval_mode"}`: one row per indexed paper, similarity = dense cosine. |

## Retrieval
| Method | Path | Description |
|---|---|---|
| GET | `/api/retrieval/search?q=&k=10&documents=id1,id2&sections=methodology,results` | `k` 1-50. Returns `{query,mode:"primary"|"fallback",fallback_active,retriever,retriever_kind,latency_ms,bypass_reason,results:[{rank,score,chunk_id,document_id,title,section,pages:[a,b],source_url,text}]}`. 503 only when both primary and fallback fail. |
| GET | `/api/retrieval/status` | service state: mode, fallback_active, reason, primary/fallback names, failure and request counters, recovery state, corpus size, production registry entry. Loads the models on first call. |
| POST | `/api/retrieval/health-check[?force=1]` | probe the primary; in fallback mode this is a recovery attempt (cooldown applies unless `force=1`). |
| POST | `/api/retrieval/sync` | make serving follow the registry's production model (build + probe before swapping). `{"switched": bool, "message"}` |
| GET | `/api/retrieval/events` | last 100 service events (`primary_failure`, `fallback_activated`, `primary_restored`, `model_switched`, ...). |

## RAG
`POST /api/rag/ask` body `{"question": str (3-1000 chars), "history": [{role, content, retrieval_query?}] (<= 40), "document_ids": [str], "sections": [str]}`. With one `document_id` the question is document-scoped; with several, retrieval runs **per paper** and results are interleaved so every selected paper contributes. Response:
```
answer, answer_mode ("llm" | "extractive" | "insufficient"), sufficient_evidence,
sentences:[{text, citations:[S1..], invalid_citations, status, support, supporting_label}],
references:[{label, cited, paper_id, title, pages, section, url, chunk_id, retrieval_score, retrieval_rank, quote, text}],
citation_quality:{claims, cited_fraction, valid_citation_fraction, supported_fraction, unsupported_claims, note},
query:{intent, scope, is_followup, retrieval_query, document_ids, section_hints, warnings},
retrieval:{mode, retriever, kind, fallback_active, candidates, reranked, dropped_duplicates},
evidence:{coverage, top_score, gate, signal, threshold},
generation:{provider, model, prompt_tokens, completion_tokens, cost_usd, provider_errors},
timings:{retrieval_s, rerank_s, generation_s, total_s},
verification:{verified, replaced, llm_answer, reason}
```
`status` per sentence: `supported`, `weak`, `unsupported`, `interpretation`, `uncited_claim`, `meta` (lexical heuristic, not entailment). If an LLM answer is mostly uncited/unsupported it is replaced by extractive passages and the draft is returned in `verification.llm_answer`. If the evidence gate fails (`answer_mode: "insufficient"`) no LLM is called.

`POST /api/rag/stream` takes the same body and returns server-sent events: `{"type":"meta",retrieval_mode,fallback_active,retriever,sources,sufficient_evidence}`, then `{"type":"token","text"}` per piece, then `{"type":"final",...}` (the full response above), or `{"type":"error"}`.

## Explainability
| Method | Path | Description |
|---|---|---|
| GET | `/api/explain/retrieval?q=&k=5&lime=0|1&documents=` | `k` <= 10. Per result: component ranks/cosines, RRF parts, exact TF-IDF term contributions, matched terms, best-matching passage, relevance-model probability, SHAP (always) and LIME (`lime=1`, slower). `model_warning` explains when the relevance model is unavailable. |
| POST | `/api/explain/answer` | `{"question", ...}`: runs RAG and returns `{answer, explanation, retrieval, evidence_gate}`. |
| POST | `/api/explain/response` | `{"response": <a /api/rag response>}` -> sentence-level grounding (kind, evidence strength, highlight offsets, per-paper shares, unsupported claims); does not re-run retrieval or the LLM. |
| GET | `/api/explain/model` | relevance-model availability, features, training/evaluation report. |

## Library, dashboards
| Method | Path | Description |
|---|---|---|
| GET | `/api/library/papers?q=&author=&topic=&year_from=&year_to=&status=processed&limit=24&offset=0` | `limit` <= 100. Papers without a year never match a year filter. |
| GET | `/api/library/paper?id=` | detail (ids contain `:` and `/`, hence the query parameter): metadata, summary, keywords, topics, `sections:[{name, chunks:[{chunk_id,pages,text}]}]`. 404 if unknown. |
| GET | `/api/library/facets` | years, top authors, topics, sources, statuses. |
| GET | `/api/dashboard` | corpus counts, active model, vector DB, 24 h serving metrics, latest selection benchmark (adjudicated sensitivity runs excluded), pipeline job counts, fallback status when the service is loaded. |
| GET | `/api/benchmarks/overview` | registry runs with candidate metrics and selection traces, ranking metrics per K for dev/validation/test, per-category results, performance, chunk sweep, reranker, lexical-bias probe, relevance model report, dataset card, promotions. |

## Pipeline
| Method | Path | Description |
|---|---|---|
| GET | `/api/pipeline/status` | running workflows, available workflows/agents, scheduler state, recent jobs. |
| GET | `/api/pipeline/jobs?limit=50` | job rows (`kind`, `status`, `started`, `finished`, `detail`); limit <= 500. |
| GET | `/api/pipeline/failures` | extraction failures, one row per document, file names only (no local paths). |
| GET | `/api/pipeline/report` | latest monitoring report (alerts, benchmark triggers, corpus counts); 404 until one is generated. |
| POST | `/api/pipeline/run` | `{"workflow": "ingest_topic"|"benchmark_and_deploy"|"monitor"|"report"|"answer_question", "params": {...}}`. `ingest_topic` requires `params.topic`. 202 started; 409 if that workflow is already running. |

## Examples
```
curl "localhost:5000/api/retrieval/search?q=shifted+window+attention&k=3"
curl -X POST localhost:5000/api/rag/ask -H 'Content-Type: application/json' \
     -d '{"question":"Compare the methodologies of these papers.","document_ids":["arxiv:2506.06962","title:3f753894119784d4"]}'
curl -N -X POST localhost:5000/api/rag/stream -H 'Content-Type: application/json' -d '{"question":"What datasets were used?"}'
curl -X POST localhost:5000/api/pipeline/run -H 'Content-Type: application/json' -d '{"workflow":"ingest_topic","params":{"topic":"graph neural networks"}}'
```
