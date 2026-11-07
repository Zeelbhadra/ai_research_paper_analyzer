# Upgrade Plan

## Principles
- Keep `src/` (legacy features) working; add new code under `backend/`, following the suggested layout, imported by Flask via blueprints.
- Every step leaves the app runnable and has pytest tests. No fabricated numbers: all benchmark results come from `python -m backend.benchmarking.run`.

## Architectural decisions (with trade-offs)
| Decision | Choice | Why |
|---|---|---|
| PDF extraction | **PyMuPDF** primary, pdfplumber fallback | PyMuPDF is typically much faster and gives per-page text + font-size info for heading detection. pdfplumber retained for fallback only. The extractor comparison is a script (`backend/ingestion/extractor_compare.py`) run on the 50 PDFs in `data/`; the choice is revisited with measured numbers. |
| Vector store | **FAISS** (flat inner-product, persisted per model) behind `VectorStore` interface | Already a dependency; at this corpus scale (10³–10⁵ chunks) exact search is fast; no extra service. Chroma/pgvector adapters are lower priority; filtering is done by post-filtering on a metadata table. Trade-off: no native metadata filtering. |
| Sparse | scikit-learn TF-IDF (own index) + optional BM25 | Required fallback; independent of torch. |
| Metadata/job store | MongoDB when configured, **SQLite** local default | Mongo Atlas was unreachable in this environment (DNS failure); the app must run offline. |
| Scheduler | APScheduler | No Redis/Celery needed. |
| Agents | Deterministic orchestrator with plain-Python "agents"; LLM only in answer generation/judging | Benchmarks/selection must not be LLM-decided. |
| Explainability | SHAP/LIME on a **learned relevance reranker** (GBM/logistic on interpretable features) | Gives SHAP/LIME a real prediction task; never claims to explain embedding dimensions. |

## Steps
1. ✅ Audit (this doc, EXISTING_ARCHITECTURE.md, DEPENDENCY_AUDIT.md) + fix startup crash.
2. Ingestion, extraction, cleaning, section detection, chunking (5 strategies), dedup, SQLite store.
3. Embedding backends (TF-IDF/SBERT/E5/BGE) + FAISS store + caching + model versioning.
4. Benchmark dataset (dev/val/test, document-level splits) + ranking metrics.
5. Weighted scoring + registry.
6. Auto-selection with thresholds + TF-IDF fallback.
7. RAG (query processing, reranker benchmark, LLM provider abstraction, citations).
8. SHAP/LIME relevance model + evidence explanations.
9. Orchestrator + scheduler.
10. Frontend dashboards.
11. Tests, Docker.
12. Experiment report, docs.

## Known constraints
- CPU only, so E5/BGE-base embedding of the whole corpus will take minutes; benchmark config exposes `max_chunks` and model lists.
- Models must be downloaded from HuggingFace (needs network on first run).
- Ground-truth relevance labels: synthetic/abstract-derived queries are **not** human-verified unless a human reviews them; the dataset builder marks `label_source` and the report must state this.
- OpenAI key needed for generation; without it the RAG pipeline returns extractive, citation-only answers.

---
# Final status
All 12 steps were executed. Deviations from the plan, each documented where it happens: only FAISS was implemented as a vector store (Chroma/pgvector not built or compared); model selection uses a weight-free evidence-driven rule instead of the originally suggested weighted score (by decision); generation evaluation (Phase 6.2) was not done because no LLM was available; the Docker image was written and validated but not built; labels are synthetic and unreviewed. See README.md for the status table and EXPERIMENT_RESULTS.md section 12 for the evaluation still required.
