# Deployment

> **Status of the container setup:** `Dockerfile` and `docker-compose.yml` were written and `docker compose config` validates both the default and the `mongo` profile, but **the image has not been built or run**: the Docker daemon was not running on the development machine. Treat the first `docker compose up --build` as the real test, and expect to adjust (for example Python 3.12 wheels versus the 3.13 used in development). Everything else in this document was exercised locally without Docker.

## What runs where
| Piece | Where | Notes |
|---|---|---|
| Web app + API | one gunicorn worker, 8 threads | one worker on purpose: models load once, and the scheduler must run exactly once. Concurrency comes from threads. |
| Documents, chunks, jobs | SQLite `storage/papers.sqlite` | no extra service needed |
| Vector indexes | FAISS (exact inner product) + TF-IDF pickles in `storage/indexes/<model>/<chunk config>/` | persisted; rebuilt incrementally (embeddings cached per chunk text in `storage/embedding_cache/`) |
| Model registry / history | `storage/registry/registry.json` + `registry.csv` | production pointer + all benchmark runs |
| Metrics, event logs, reports | `storage/metrics.sqlite`, `storage/logs/*.jsonl`, `storage/reports/` | |
| Legacy summary cache | MongoDB (optional) | the app runs without it; unset `MONGO_URI` skips the connection |
| LLM | OpenAI via `OPENAI_KEY` (optional) | without a working key the assistant returns extractive, cited passages |

## Local install (no Docker)
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                      # or requirements.lock to reproduce the tested versions
cp .env.example .env                                     # then fill in keys
python app.py                                            # http://localhost:5000
```
First request downloads the Hugging Face models (SBERT / E5 / BGE) into `~/.cache/huggingface` (or `HF_HOME`).

## Build the data and indexes from scratch
```
python -m backend.ingestion.ingest_cli --dir data                         # extract, clean, section, chunk -> SQLite
python -m backend.benchmarking.embed_corpus                               # TF-IDF + SBERT + E5 + BGE indexes (about 1 min SBERT, 9-14 min each for E5 and BGE on a 16-thread CPU when nothing else runs; E5 took 33 min once while other jobs competed)
python -m backend.benchmarking.build_dataset                              # synthetic benchmark queries with a local LLM (about 50 min on CPU; needs the Qwen model)
python -m backend.benchmarking.tune_hybrid                                # hybrid weights on the dev split
python -m backend.benchmarking.performance_benchmark --all                # isolated latency/memory/indexing measurements
python -m backend.benchmarking.run_benchmark --report-test                # score, select, record, promote
python -m backend.explainability.relevance_model train                    # surrogate model for SHAP/LIME
python -m backend.benchmarking.calibrate_evidence_gate                    # "insufficient evidence" threshold for the RAG answers
python -m backend.agents.backfill_enrichment                              # keywords, summaries, topics for the library view
```
`/readyz` stays at 503 until a production model and its indexes exist. The existing `storage/` of this project can simply be mounted into the container.

## Docker
```
cp .env.example .env                       # keys; the file is excluded from the image and from git
docker compose up --build                  # app on http://localhost:5000  (HOST_PORT to change)
docker compose --profile mongo up          # + local MongoDB; set MONGO_URI=mongodb://mongo:27017 in .env
AUTOMATION=1 docker compose up             # also start the scheduler
```
- **Volumes:** `storage` (all mutable state), `hf_cache` (model weights), `./data` mounted read-only (source PDFs). To start from this project's existing state: `docker compose run --rm -v "$PWD/storage:/seed:ro" app cp -a /seed/. /app/storage/` (or bind-mount `./storage:/app/storage` instead of the named volume).
- **Secrets:** never in the image (`.dockerignore` excludes `.env`); provided at run time through `env_file`/environment.
- **Health:** the container `HEALTHCHECK` calls `/healthz` (liveness, no dependencies). `/readyz` (readiness) checks the database, a writable storage directory and that the production model's indexes exist; it returns 503 with the failing checks, and never forces a model load.
- **Logs:** `LOG_FORMAT=json` writes JSON lines to stdout; compose rotates them (10 MB x 5). Domain events (fallback activations, recoveries, model switches, discovery runs) are also appended to `storage/logs/*.jsonl`.
- **Memory:** the three embedding models plus indexes need roughly 1-2 GB; `mem_limit: 6g` leaves room for benchmark runs.
- **CPU only:** the image installs the CPU wheel of torch. For GPU, change the base image and the torch install line; the code selects `cuda` automatically.
- Registry entries store index locations relative to `storage/`; older entries with absolute paths from another machine are resolved against the mounted storage.

## Configuration
All settings are environment variables (see `.env.example`): `OPENAI_KEY`, `RAG_OPENAI_MODEL`, `RAG_USE_LOCAL_LLM`, `SEMANTIC_SCHOLAR_API_KEY`, `MONGO_URI`, `STORAGE_DIR`, `DATA_DIR`, `RANDOM_SEED`, `LOG_LEVEL`, `LOG_FORMAT`, `WARMUP`, `AUTOMATION`, `PORT`. Policy files: `backend/config/selection.json` (quality floor, resource budgets, non-inferiority margin), `backend/config/automation.json` (topics, schedules, monitoring thresholds, enrichment taxonomy), `backend/config/rag_calibration.json` (generated by the calibration script).

## Measured performance (CPU, 16 logical cores; `python -m backend.benchmarking.perf_suite`, `storage/benchmark/perf_suite.json`)
| Check | Result |
|---|---|
| Concurrency, production retriever (TF-IDF + E5), 48 queries | **Before** the lock fix the service was serialised: 17.9 / 19.4 / 18.8 / 20.2 queries/s with 1 / 2 / 4 / 8 clients, p95 63 / 116 / 235 / 440 ms. **After:** 19.5 / 24.2 / 31.9 / 38.2 queries/s, p95 61 / 107 / 169 / 259 ms, 0 errors. Scaling is sub-linear because query encoding is CPU-bound. |
| TF-IDF at 20,000 synthetic chunks | **Before** the sparse-column rewrite: 100 ms p50 per query. **After:** 0.7 ms p50 / 1.0 ms p95 (about 140x), identical rankings and scores on 124 real query comparisons. Index build 27 s; the synthetic random-word corpus makes the bigram vocabulary unrealistically large (770 MB RAM during fit, 133 MB on disk); use `min_df=2` or `max_features` beyond roughly 50k chunks of real text. |
| Large document ingestion (generated 300-page PDF, 0.5 MB) | 0.94 s total (extract 0.26 + clean/sections 0.35 + chunk 0.33), 320 pages/s, 7 MB peak Python heap. Real, image-heavy PDFs will be slower; scanned PDFs are detected but not OCR'd. |
| Batch embedding, SBERT MiniLM, 192 real chunks | 48 / 46 / 48 chunks/s at batch size 8 / 32 / 64: batch size does not matter on this CPU. E5/BGE-base are about 10-20x slower per chunk. |
| Retrieval on the real corpus (2,222 chunks) | hybrid p50 about 40 ms, TF-IDF p50 about 1 ms (isolated benchmark). |

## Operational notes and limits
- **One scheduler:** do not run several app replicas with `AUTOMATION=1`; each would start its own scheduler. Scale reads by running extra replicas with `AUTOMATION=0` against shared, read-mostly storage (SQLite is single-writer, so heavy concurrent ingestion should stay on one instance).
- **No authentication or rate limiting:** do not expose `/api/pipeline/*`, `/api/retrieval/sync` or `/api/retrieval/health-check` publicly; put the app behind an authenticating reverse proxy.
- **Semantic Scholar:** without `SEMANTIC_SCHOLAR_API_KEY` the shared public pool returns HTTP 429; discovery then continues with arXiv only (reported as a partial result).
- **Backups:** back up the `storage` volume (SQLite, registry, indexes, embedding cache); indexes and caches are rebuildable, the benchmark dataset and registry history are not.
- **The `.env` of the development machine** contains a `MONGO_URI` whose Atlas cluster no longer resolves; it only causes a logged connection error and a disabled summary cache. Remove or replace it.
