# Frontend (Step 10)

Single-page app served by Flask (`templates/index.html`), vanilla JS with hash routing, no build step, no CDN scripts. The original PaperLens design and its search / upload-and-analyze flow are kept as the **Discover** view; the new views are added beside it.

## Views
| Route | What it shows | Data source |
|---|---|---|
| `#/discover` | original search (arXiv / Semantic Scholar), PDF or text analysis (summary, keywords, topics, similar indexed papers) | `/api/search`, `/api/analyze`, `/api/similar` |
| `#/dashboard` | papers indexed, processed documents, chunks, active embedding model, vector DB, retrieval latency (live if traffic exists, else the isolated benchmark, labelled), latest benchmark, fallback status with health-check button, pipeline summary | `/api/dashboard`, `/api/benchmarks/overview` |
| `#/library` | search + author/year/topic/status filters, paper cards, detail drawer with metadata, summary, keywords, zero-shot topics, extracted content by section with page ranges, "ask the assistant about this paper" | `/api/library/*` |
| `#/assistant` | streamed RAG chat: scope to one/several papers, compare, follow-ups, clickable `[S#]` citations, evidence cards (paper, pages, section, retrieval rank/score, highlighted terms, full passage), per-paper contribution chart, warnings for unverified/rejected answers and insufficient evidence | `/api/rag/stream`, `/api/explain/response` |
| `#/benchmarks` | production vs highest-scoring model, run selector and history, Precision/Recall/nDCG@K, MRR, MAP, latency, throughput, memory, indexing time, index size, selection reasoning with confidence intervals, held-out test, per-category table, chunk sweep, reranker, lexical-bias probe, dataset card, model-switch history. Every chart has a table view. | `/api/benchmarks/overview` |
| `#/explain` | per result: exact retrieval facts (component ranks, rank-fusion parts, term contributions, best passage), relevance probability, SHAP and optional LIME bars; relevance-model metrics and global importance; sentence-level answer grounding | `/api/explain/*` |
| `#/pipeline` | running workflows, scheduler state, job table with status/duration, failed documents, run-a-workflow form, model-switching history, retrieval-service events, latest monitoring report and alerts | `/api/pipeline/*`, `/api/retrieval/events` |

A banner under the header appears on every page while retrieval is in **fallback mode** (it polls `/api/dashboard`, which never forces a model load).

## Charts
Own small SVG/HTML chart code (`static/js/charts.js`): thin rounded bars, legend for 2+ series, value labels only on the leading bar of each group, hover and keyboard tooltips, table view for every chart, one axis (no dual axes). Model colours are fixed per model (TF-IDF blue, SBERT orange, E5 aqua, BGE yellow, hybrid magenta); the palette was validated on the app's dark surface (worst adjacent CVD distance 8.4, normal-vision 19.3, all >= 3:1 contrast). SHAP/LIME use a two-hue diverging pair around a neutral zero line with arrows and signs, not colour alone.

## Run
`PORT=5055 python app.py` (set `AUTOMATION=1` to start the scheduler). The retrieval service and models load on first use (first assistant/explain request takes several seconds).

## Tests
- `python -m pytest backend/tests` (API contracts, view data, regressions).
- `node tests/ui_smoke.mjs http://127.0.0.1:5055 <outDir>`: drives real Chrome through DevTools (no dependencies; Node >= 22): every route renders without JS errors or stray `null`/`undefined`/`NaN` text; an assistant answer with citations, highlighted evidence and active-card linking; off-topic refusal; follow-up detection; SHAP/LIME bars; library drawer; triggering a workflow; legacy search and analyze. Writes screenshots. Last run: 13/13 passed, 0 JS errors.

## Bugs found by looking at the rendered pages (not by the checks that passed)
- `replaceChildren(null)` printed the literal text "null" in the answer and scope bar. Fixed with a guard in `core.js` and a regression check.
- The dashboard's "latest benchmark" showed the adjudicated *sensitivity* run instead of the selection run.
- Flask's default JSON key sorting scrambled ranked dictionaries (global feature importance); `sort_keys` is now off.
- The API leaked absolute local paths in failure reasons; they are reduced to file names.
- Several library papers had junk titles (PDF template text, arXiv stamps, journal banners) and many more kept only the first line of a multi-line title; titles are now repaired from the file names (`python -m backend.agents.maintenance fix-titles`), future ingests reject such titles. The original UI also called `/api/analyze` and `/api/similar`, which never existed; both now do.

## Known limitations
- Papers added as plain PDFs have no authors or publication year, so those filters only cover papers that came from arXiv / Semantic Scholar.
- Topic labels are zero-shot over a fixed taxonomy (`config/automation.json`), unvalidated; some are wrong. Summaries are extractive.
- The assistant currently produces extractive answers because no LLM credits are available; the interface is built for model answers (streaming, citations, verification notices) and was exercised end to end only with the extractive path and stubbed model output in tests.
- Mobile layout is responsive (single column, 16 px gutters) but was only checked at desktop width in the screenshots.
- No authentication: do not expose the pipeline endpoints publicly.
