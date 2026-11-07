# Frontend

Vanilla JavaScript, no build step. `templates/index.html` plus `static/js/` (`core.js`, `view_discover.js`, `view_library.js`, `view_assistant.js`) and `static/css/`. A hash router serves three pages; any other hash falls back to Discover.

| Page | What the reader does |
|---|---|
| **Discover** | Types a topic; `/api/search` returns papers (arXiv and Semantic Scholar). **Analyze** opens a panel with a summary, keywords, related topics, similar papers and more papers on the topic. **Add to library** downloads and processes the paper so it can be read and discussed. |
| **Library** | Browse and filter processed papers (year, author, topic). A paper opens with its summary, keywords, sections and one-click prompts: Explain it simply, Why does it matter?, Key ideas, How does it work?, What did they find?, Explain the jargon. |
| **Assistant** | A chatbot. Optionally pick one or more papers; answers stream in, each sentence carries numbered citations that open the source passage (page, section, highlighted text). Follow-up suggestions are offered. Off-topic questions get a polite "could not find enough information". |

## What readers never see
Model names, scores, retrieval mode, latency, costs, pipeline state, dashboards and benchmarks. This is enforced at three levels: the UI has no such pages, the admin endpoints return 404 unless `ADMIN_API=1`, and chat responses are reduced to reader-facing fields by `backend/rag/public_view.py`. The old admin views are kept in `archive/admin_ui/`.

## Engineering notes
- DOM is built with a small safe helper (`PL.h`); no `innerHTML` with data, so paper text cannot inject markup.
- Chat uses server-sent events (`/api/chat/stream`: start, token, final, error).
- Accessibility: landmarks, labelled controls, `aria-live` on answers, keyboard operable, dark mode via `prefers-color-scheme`.
- Test: `node tests/ui_smoke.mjs http://127.0.0.1:5000 outdir` drives headless Chrome (22 checks).

## Limitation
Without an LLM (no credits were available) answers are cited passages chosen by section and intent; they are not rewritten into simpler words. With a working `OPENAI_KEY` the prompt asks for plain language and explained terms, but that path has not been evaluated.
