# Architecture

```
            sources                    ingestion & data engineering                 indexing                    selection                    serving
 arXiv ─┐                    ┌──────────────────────────────────────┐
 Semantic Scholar ─┼─ discovery ─► dedupe (DOI/arXiv/title) ─► download ─► extract (PyMuPDF │ pdfplumber) ─► clean ─► sections ─► chunk ─┐
 PDF uploads ─┘                                                                                                                    ▼
                                                              SQLite: papers, chunks (page+section), jobs, failures      TF-IDF index  FAISS per model (SBERT/E5/BGE)
                                                                                                                                  │ embedding cache (per chunk text)
                                         benchmark dataset (synthetic, dev/val/test by paper) ─► evaluate (P/R/nDCG/MRR/MAP, bootstrap CIs)
                                         isolated performance runs (latency, memory, indexing) ─► selection (feasibility → non-inferiority → footprint)
                                                                                                                                  ▼
                                                                                             model registry (JSON/CSV, history) ─► guarded promotion
                                                                                                                                  ▼
 user ─► Flask API/UI ─► RetrievalService (production = hybrid TF-IDF+E5, fallback = TF-IDF, health + recovery) ─► query processing ─► context builder
                                                                          ▲                                                         ─► LLM chain (OpenAI → local → extractive)
                      orchestrator + agents + APScheduler (discovery, documents, benchmark, monitoring) ─ metrics ─ reports             ─► citation verification ─► explanations (SHAP/LIME/evidence)
```

## Layers (code under `backend/`, original code kept in `src/`)
| Layer | Modules | Responsibility |
|---|---|---|
| Ingestion | `ingestion/` (arxiv/semantic-scholar clients, `models.py` retry, `deduplication.py`, `pdf_extractor.py`, `document_processor.py`) | paginated, rate-limit-aware discovery; cross-source dedup (junk template titles never identify a paper); extraction with fallback; corrupt and scanned PDFs logged, never fatal |
| Preprocessing | `preprocessing/` (`text_cleaner`, `section_extractor`, `chunking`) | header/footer removal, dehyphenation, heading-based sections with page spans, five chunking strategies; every chunk keeps document, title, id, pages, section, text, URL, embedding-model version |
| Embeddings / vectors | `embeddings/` (SBERT/E5/BGE wrappers, TF-IDF, `embedding_manager`), `vectorstore/` (interface + FAISS) | normalised vectors, dimension and NaN validation, version string per model/prefix/dim, persistent cache keyed by chunk text, one index per model version and chunk config |
| Retrieval | `retrieval/` (dense, sparse, hybrid RRF, reranker, factory, `service.py`, `production.py`) | common `Retriever` interface; production service with fallback state machine, health probes, safe model switching |
| RAG | `rag/` (query processor, context builder, multidoc, LLM providers, generator, citation manager, pipeline) | intent/scope, de-duplicated and budgeted context, strict grounded prompt, provider failover, citation verification and a guard that replaces ungrounded answers |
| Benchmarking | `benchmarking/` | dataset builder, metrics, performance harness, selection, registry, tuning/probe/sweep scripts, report generator |
| Explainability | `explainability/` | interpretable features, relevance model, SHAP, LIME, evidence explainer |
| Agents | `agents/` | deterministic orchestrator, six agents, workflows, scheduler, monitoring rules, maintenance |
| API / UI | `api/routes/`, `templates/`, `static/` | 27 routes, hash-routed single-page UI |

## Key decisions and why
- **SQLite + FAISS flat, no extra services.** The corpus is small (about 2k chunks; 20k tested), exact search is fast, and persistence is a file. Trade-off: no native metadata filtering (post-filtering with oversampling) and a single writer. The `VectorStore` interface allows Chroma/pgvector adapters (not implemented).
- **Selection without metric weights.** Fixed weights made the winner an artefact of the weights. The rule (quality first, a cheaper model wins only when shown not materially worse, cost breaks ties) is dynamic: budgets come from the machine's available RAM, and the result changes with the amount of evidence.
- **Hybrid via reciprocal rank fusion.** No score calibration between lexical and dense scores; weights tuned on the dev split only.
- **TF-IDF as a real fallback.** Independent index and code path, no model download; the service serves failing requests from it immediately, switches mode after repeated failures, and recovers after consecutive healthy probes with exponential backoff.
- **Deterministic orchestration.** No LLM decides control flow, scores, winners or promotion. Retries are limited to transient errors; one run per workflow at a time.
- **Grounding over fluency.** Answers must cite numbered passages; sentences are checked lexically; an answer that is mostly uncited or unsupported is replaced by quoted passages; weak evidence (dense-cosine gate) means no LLM call at all.
- **Explainability with scope.** SHAP/LIME explain a surrogate relevance model over named features; exact facts (ranks, RRF parts, term contributions) are shown separately; nothing claims to explain embedding dimensions.
- **Concurrency.** One gunicorn worker with threads. The retrieval service lock guards only state; searches run in parallel (this fixed a serialisation that capped throughput at about 20 queries/s).

## Storage layout (`storage/`)
`papers.sqlite` (papers, chunks, extraction failures, jobs) - `indexes/<model>/<chunk config>/` - `embedding_cache/<model>/` - `registry/registry.json|csv` - `benchmark/` (dataset, per-query results, studies, performance) - `models/relevance_model.joblib` - `metrics.sqlite` - `logs/*.jsonl` - `reports/`. Only the dataset, registry history and enrichment are irreplaceable; the rest can be rebuilt.

## Failure handling summary
Corrupt/scanned PDF -> logged, batch continues. API rate limit -> backoff, partial discovery result. Embedding/vector failure -> TF-IDF fallback + event + UI banner, auto-recovery. New model fails probe -> not promoted/swapped. LLM unavailable or ungrounded -> extractive passages. Weak evidence -> explicit refusal. Scheduled workflow overlap -> skipped. Missing production indexes -> `/readyz` 503.

## Extending
New embedding model: add a class in `embeddings/base.py` (prefixes!) and a name in `retrieval/factory.py`; the benchmark, registry and selection pick it up. New vector store: implement `vectorstore/base.py`. New LLM: implement `LLMProvider` in `rag/llm.py`. New source: a client yielding `PaperRecord`s plus an entry in `DiscoveryAgent`. New agent: subclass `Agent`, register it and define a workflow in `agents/workflows.py`.
