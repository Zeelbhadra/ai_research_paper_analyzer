# Dependency Audit

Environment inspected: Python 3.13.3, Windows 11, **no GPU detected** (CPU-only benchmarks).

| Package | In requirements.txt | Installed | Used by | Verdict |
|---|---|---|---|---|
| flask | ✅ | 3.1.0 | app | keep |
| arxiv | ✅ | 2.4.1 | paper_fetcher | keep (wrap with retry) |
| requests | ✅ (listed twice) | ✔ | fetcher, downloader | keep, dedupe line |
| PyMuPDF | ✅ | 1.27.2 | pdf_extractor | **selected** primary extractor (see UPGRADE_PLAN) |
| pdfplumber | ❌ | 0.11.10 | – | add as fallback / comparison candidate |
| transformers / torch | ✅ | 5.3.0 / 2.10.0 | summarizer, classifier | keep; heavy |
| keybert | ✅ | 0.9.0 | keywords | keep |
| sentence-transformers | ✅ | 5.2.3 | embeddings | keep (hosts SBERT, E5, BGE) |
| faiss-cpu | ✅ | 1.13.2 | embeddings | keep → primary vector store |
| scikit-learn | ✅ | 1.6.1 | TF-IDF | keep |
| pymongo, dnspython | ✅ (unpinned) | 4.16.0 | cache | keep, pin |
| openai | ✅ (unpinned) | 2.29.0 | summarizer | keep behind provider abstraction |
| python-dotenv | ✅ | 1.2.2 | db | keep |
| gunicorn | ✅ | – | deploy | keep (Linux/Docker only) |
| rouge_score, bert_score, tqdm | ❌ **missing** | tqdm only | evaluation_metrics, tests | add |
| shap | ❌ | 0.52.0 | – | needed for Phase 4 |
| lime | ❌ | not installed | – | needed for Phase 4 |
| rank-bm25 | ❌ | not installed | – | hybrid sparse candidate |
| pytest | ❌ | 9.0.3 | – | add as dev dependency |
| apscheduler | ❌ | not installed | – | chosen scheduler (Phase 5), no Redis needed |

Redundant / unnecessary: `gunicorn` is unusable on Windows (fine for Docker); `requests` duplicate. Not adding Celery/Redis: the workload is a single-node batch pipeline; APScheduler + a SQLite job table is sufficient.

Heavy-import risk: `transformers`/`torch` import time dominates app startup; new code must lazy-import them.
