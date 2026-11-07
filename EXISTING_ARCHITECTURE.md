# Existing Architecture (audit of commit 88f5093)

Verified against the actual repository, not the README.

## What is actually wired up

```
Browser (templates/index.html, static/script.js)
   │  GET /api/search?q=&limit=
   ▼
app.py (Flask)  ──► services/paper_fetcher.py  (arXiv lib + Semantic Scholar REST)
   │            ──► services/paper_service.py  (MongoDB get/save summary, collection `summaries`)
   └──────────► services/summarizer.py         (OpenAI gpt-4o-mini, abstract → summary)
```

Only `/api/search` exists. Everything else in `src/` is **not reachable from the web app**:

| Module | Purpose | Wired into app? |
|---|---|---|
| `src/pdf_extractor.py` | PyMuPDF whole-text extraction (no page tracking) | No |
| `src/text_processor.py` | regex cleaning + heading-based section split | No |
| `src/summarizer.py` | Pegasus-arXiv abstractive + TF-IDF extractive fallback | No |
| `src/keyword_extractor.py` | KeyBERT (MMR) + TF-IDF fallback | No |
| `src/topic_classifier.py` | BART-MNLI zero-shot, 10 coarse labels | No |
| `src/embeddings_manager.py` | all-MiniLM-L6-v2 + in-memory FAISS `IndexFlatL2`, one vector per *paper* | No |
| `src/evaluation_metrics.py` | ROUGE / BERTScore for summarization | No (tests only) |
| `src/pdf_downloader.py` | background PDF caching | No (and broken) |
| `tests/evaluate_summarizer.py` | summarization eval script | Broken import |

## Stack
- Backend: Flask 3, gunicorn listed. Frontend: single-page vanilla JS/CSS (776 lines CSS, 288 lines JS).
- DB: MongoDB Atlas via pymongo, collection `summaries` (key = paper URL), unique index on `paper_id`.
- LLM: OpenAI `gpt-4o-mini` (abstract summarization only).
- Embeddings: only SBERT MiniLM; no E5/BGE/TF-IDF retrieval; no chunking.
- Docker: **none**. CI: none. Tests: one script, not pytest-style.
- Data: 50 PDFs in `data/` (topics: ML, computer vision, ride-sharing) — useful as the seed corpus for the benchmark.

## Defects found
1. **App does not start.** `services/summarizer.py` and `pdf_downloader.py` import `papers_collection`; `db.py` defines only `summaries_collection` → `ImportError` on `import app`.
2. **Frontend/backend mismatch.** `script.js` calls `/api/analyze` and `/api/similar` and sends `source=`; none exist / is read.
3. `pdf_downloader._download_task` uses the paper URL as a filename (invalid on Windows), computes an unused `safe_filename`, and the `abs→pdf` rewrite appends `.pdf` incorrectly for versioned ids.
4. `tests/evaluate_summarizer.py` imports non-existent `src.arxiv_fetcher`; `rouge_score`, `bert_score`, `tqdm` are not in `requirements.txt`.
5. `db.py` runs a network ping at import time; failure leaves `summaries_collection=None` (handled in paper_service, not elsewhere).
6. `paper_fetcher`: no retry, no rate-limit handling, no pagination, no PDF URL/DOI/year/citations captured (Semantic Scholar fields requested but dropped), no dedup across the two sources.
7. `text_processor.clean_text` truncates at the first "References" heading, discarding references we want as a section; page numbers lost by `pdf_extractor` (joins pages with `\n`), so page-level citations are impossible today.
8. `embeddings_manager` stores one vector per paper using `IndexFlatL2` on normalised vectors (works, but cosine/IP is the clearer choice); model name hard-coded; no model/version tracking.
9. `.gitignore` lists `__pycache__/` but **15 `.pyc` files are tracked** in git.
10. `Flask` `limit=int(...)` unvalidated → 500 on bad input; `traceback` + `str(e)` returned to clients.
11. Two summarizers with the same name in different modules (`src/summarizer.py` vs `src/services/summarizer.py`).

## Reusable as-is
`paper_fetcher` (as a base for the new clients), `keyword_extractor`, `topic_classifier`, `summarizer` (Pegasus + TF-IDF fallback), `text_processor.segment_sections` (regex idea), frontend shell, `paper_service` Mongo caching.
