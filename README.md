# PaperLens: AI Research Paper Intelligence System

An upgrade of the original *AI Research Paper Analyzer* into a retrieval-augmented system for academic papers: automated ingestion, benchmarked retrieval (TF-IDF, SBERT, E5, BGE and a hybrid), grounded question answering with page-level citations, explainability, deterministic orchestration and a web UI. The original search / summarize / classify features are preserved.

> **Read this first.** Retrieval was benchmarked on **synthetic, not human-verified** queries (89 retained; 21 validation, 24 test), so the numbers are indicative. **Answer generation has not been evaluated**: no LLM was available (the OpenAI account has no credits; the 1.5B local model ignored citation instructions), so the assistant currently returns extractive, cited passages, and faithfulness / answer-quality metrics do not exist yet. The Docker image has **not been built** (no running daemon). Details: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md).

## What it does
- **Ingest**: arXiv and Semantic Scholar discovery (pagination, retries, rate limits), cross-source deduplication, open-access PDF download, manual uploads; PyMuPDF/pdfplumber extraction, header/footer removal, section detection, five chunking strategies; every chunk keeps paper, pages, section, URL and embedding version.
- **Retrieve**: separate TF-IDF and per-model FAISS indexes with cached embeddings; production model chosen by measurement; automatic TF-IDF fallback with health checks and recovery.
- **Answer**: intent/scope detection, follow-ups, de-duplicated budgeted context, strict grounded prompt, `[S#]` citations, lexical citation verification, refusal when evidence is weak, provider chain (OpenAI, local model, extractive).
- **Explain the paper**: the assistant explains the *content* of a paper in plain language, each statement tied to the page it comes from (highlighted source passages). SHAP/LIME explained the retrieval model rather than the papers, so they were removed (kept in `archive/`).
- **Automate**: deterministic orchestrator, six agents, APScheduler, monitoring rules, reports.
- **UI (readers see three pages)**: **Discover** (enter a topic, get papers; Analyze gives a summary, keywords, related topics and similar papers; add a paper to the library), **Library**, **Assistant** (chatbot that answers in simple terms with cited sources). Model names, scores and pipeline details are never shown to readers.
- **Admin API**: dashboard, benchmark, pipeline and retrieval-admin endpoints exist but are off unless `ADMIN_API=1` is set (they return 404 otherwise).

## Headline results (measured; full tables and caveats in EXPERIMENT_RESULTS.md)
Production: **hybrid TF-IDF + E5** with a TF-IDF fallback. Held-out test nDCG@10: hybrid 0.698, TF-IDF 0.683, E5 0.630, BGE 0.629, SBERT 0.493. The hybrid's edge over TF-IDF is **not statistically significant** (p = 0.44 on test); dense-only models are each worse than the hybrid on test at p < 0.05 (uncorrected for multiple comparisons). TF-IDF is about 33x faster than E5/BGE (1.2 ms vs 39 ms p50). The existing chunking (256 words, 15% overlap, recursive) was the best of nine configurations under a fixed context budget. Cross-paper questions remain weak without explicitly selecting the papers.

## Quick start
```
pip install -r requirements-dev.txt          # or requirements.lock for the tested versions
cp .env.example .env                         # optional keys (OPENAI_KEY, SEMANTIC_SCHOLAR_API_KEY)
python app.py                                # http://localhost:5000  (first request downloads the models)
```
Docker: `docker compose up --build` (see [DEPLOYMENT.md](DEPLOYMENT.md); untested build). To build the data and indexes from scratch, see "Build the data and indexes" in DEPLOYMENT.md.

## Tests
`python -m pytest backend/tests` (178 tests: units, integration, API contracts, performance) and, with the app running in public mode, `node tests/ui_smoke.mjs http://127.0.0.1:5000 outdir` (drives real Chrome: 22 checks, including that no backend terms appear and admin endpoints return 404). Real-model runs live in `backend/benchmarking/` (`demo_fallback`, `demo_rag`, `demo_automation`, `perf_suite`).

## Documentation
| File | Content |
|---|---|
| [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) | final experimental report and model selection (generated from stored artifacts) |
| [MODEL_BENCHMARKING.md](MODEL_BENCHMARKING.md) | benchmarking log: dataset, bias audits, selection method, chunk sweep |
| [RAG_EVALUATION.md](RAG_EVALUATION.md) | RAG pipeline, gate calibration, real runs, what is not evaluated |
| [EXPLAINABILITY.md](EXPLAINABILITY.md), [AUTOMATION.md](AUTOMATION.md), [FRONTEND.md](FRONTEND.md) | per-area details |
| [ARCHITECTURE.md](ARCHITECTURE.md), [API_DOCUMENTATION.md](API_DOCUMENTATION.md), [DEPLOYMENT.md](DEPLOYMENT.md) | design, endpoints, running it |
| [EXISTING_ARCHITECTURE.md](EXISTING_ARCHITECTURE.md), [DEPENDENCY_AUDIT.md](DEPENDENCY_AUDIT.md), [UPGRADE_PLAN.md](UPGRADE_PLAN.md) | the audit of the original project and the plan |

## Status against the original specification
| Phase | Status |
|---|---|
| 0 Audit | done; found the original app could not start (fixed) and the UI called two endpoints that did not exist (implemented) |
| 1 Ingestion, extraction, chunking | done; no OCR for scanned PDFs; 3 of 49 supplied PDFs are corrupt |
| 2 Embedding benchmark, registry, fallback | done for TF-IDF, SBERT, E5, BGE, hybrid; **only FAISS was implemented and used** (Chroma/pgvector were not built or compared, only the interface exists); ROC/PR-AUC apply only to the relevance classifier |
| 3 RAG | done and tested; exercised end to end only with extractive answers and stubbed model output |
| 4 Explainability | redirected: content explanation (plain-language answers with highlighted evidence). SHAP/LIME on the relevance model were built, then archived because they explained the model, not the papers |
| 5 Agents and automation | done as deterministic workflows; no LLM agents by design |
| 6 RAG evaluation | retrieval evaluation done; **generation metrics (faithfulness, relevance, context precision/recall, groundedness, citation correctness vs humans, cost) not done** |
| 7 Frontend | done (3 reader pages: Discover, Library, Assistant) |
| 8 Backend structure and API | done; adapted to the existing repository (`backend/` beside the original `src/`) |
| 9 Testing | 178 backend tests + 22-check browser smoke test |
| 10 Docker / deployment | files written and validated with `docker compose config`; **image not built or run** |
| 11 Experimental report | done, including error analysis and statistical comparison; limitations listed |

## Bugs this work found and fixed (selection)
App failed to import (missing `papers_collection`); UI called non-existent `/api/analyze` and `/api/similar`; sentence splitter detached citations; title de-duplication merged two journal-template papers and silently dropped one; compare mode could ignore a selected paper (8 of 15 covered, now 15 of 15); retrieval service serialised all requests (throughput flat at about 20 queries/s, now 38 at 8 clients); TF-IDF cost grew with corpus size (100 ms at 20k chunks, now 0.7 ms, identical results); registry stored absolute host paths; several UI defects found by viewing the rendered pages.

## Original project (preserved)
The first version searched arXiv and Semantic Scholar, summarised abstracts with OpenAI and cached summaries in MongoDB Atlas. These endpoints and modules (`src/`, `/api/search`) still work; the summary cache is optional (`MONGO_URI`), and without an OpenAI key an extractive summary is used.

## Known limitations
No human-verified labels; small query sets; synthetic query bias towards lexical overlap; weak cross-paper retrieval without explicit selection; no authentication; CPU-only measurements (dense indexing times extrapolated); authors/years unavailable for papers added as plain PDFs; zero-shot topic labels unvalidated. See EXPERIMENT_RESULTS.md section 12 for the evaluation still required.
