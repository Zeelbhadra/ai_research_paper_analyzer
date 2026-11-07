# RAG pipeline and evaluation (Step 7: pipeline built; Phase-6 generation metrics NOT yet implemented)

## Pipeline (`backend/rag/`)
`process_query` -> retrieval through `RetrievalService` (production hybrid, TF-IDF fallback) -> optional reranker -> `build_context` -> evidence gate -> `generate` (provider chain) -> `analyze_answer` (citation verification) -> verification guard -> structured response.

- **Query processing:** validation (length, control characters), rule-based intent (comparison, limitations, research_gap, evolution, paper_discovery, datasets, results, methodology, definition, summary) and scope (document vs cross-document). All ten example queries from the specification classify as expected (unit-tested). Intent is recorded, not used to re-rank (an unmeasured boost would be unjustified). Hard filters are applied only for explicitly selected documents/sections. Follow-ups are condensed deterministically with the previous standalone query; "this"/"that" are not treated as anaphora (a bug found by a test).
- **Context:** 5-gram near-duplicate removal, per-document cap of 3 for cross-document questions, word budget, document-grouped page ordering, labels `[S1]..[Sn]`.
- **Providers (`llm.py`):** OpenAI (`OPENAI_KEY`, else `OPENAI_API_KEY`; read from `.env`, never logged) -> local Qwen2.5-1.5B (opt-in) -> extractive. Streaming supported (`POST /api/rag/stream`, SSE).
- **Strict prompt:** answer only from numbered passages, cite every factual sentence, mark own synthesis as `Interpretation:`, state insufficiency explicitly.
- **Citation verification:** per sentence, lexical support (content-word overlap and every number must appear in the cited passage) -> supported / weak / unsupported / interpretation / uncited_claim. **This is a heuristic, not entailment, and not a calibrated probability.**
- **Verification guard:** an LLM answer is replaced by the extractive answer (original kept in `verification.llm_answer`) if < 60% of its claims are cited, < 50% of cited claims are supported, or it cites non-existent sources. These two thresholds are conservative policy values, not tuned.

## Decisions backed by measurements
1. **Reranker not adopted** (`storage/benchmark/rerank_benchmark.json`). BAAI/bge-reranker-base on the top-20 of the production retriever: nDCG@10 change dev +0.008 (CI -0.050..+0.072), validation +0.050 (CI -0.015..+0.123), test +0.052 (CI -0.022..+0.130). The pre-registered rule (validation CI lower bound > 0) failed, and the cost is about 6.5 s/query on this CPU. The consistently positive point estimates justify re-testing with more queries.
2. **Evidence gate** (`config/rag_calibration.json`). Signals for "is there enough evidence to answer": lexical query coverage (AUC 0.785) vs maximum dense cosine of the top-5 hits (AUC 0.999). Using the dense cosine with threshold 0.8225 (E5, specific to the calibrated model version; a different production model disables it and falls back to a lenient lexical check):

| | kept in-domain | off-topic refused |
|---|---|---|
| fit: dev (44) / 15 off-topic | 97.7% | 100% |
| held-out: validation (21) / test (24) / 15 other off-topic | 100% / 95.8% | 100% |

Caveats: the 30 off-topic questions were written by the assistant, are clearly out of domain, and the sets are small. **Near-domain unanswerable questions (a plausible question the papers simply do not answer) are not covered by this calibration.** The lexical fallback gate is weak (keeps 75% of test queries, refuses 67% of off-topic).

## Real end-to-end run (`python -m backend.benchmarking.demo_rag`, output `storage/benchmark/rag_demo.json`)
Production hybrid (TF-IDF + E5) on the 45-paper corpus; provider chain OpenAI -> local Qwen2.5-1.5B.
- **OpenAI is not usable yet:** `RateLimitError 429: no credits remaining` for the key in `.env`; the chain records this and moves on. Generation with GPT models is therefore **untested**; add credits and no code change is needed.
- Retrieval 0.05-0.35 s; local-model generation 60 s (2.9k prompt tokens) and 110 s (2.6k) on this CPU.
- **The 1.5B local model did not follow the citation instruction** (0% of claims cited in both answers) and its text contained fabricated details (e.g. an invented description of a "second paper"). The guard rejected both, and the extractive answers were shown instead.
- **Extractive answers are weak:** fragment filtering helped, but selected sentences are often only loosely relevant (e.g. a generic "summary of attention categories" for a question about Swin's complexity). It is a last-resort fallback, not a substitute for an LLM.
- Off-topic question ("best way to bake sourdough bread?"): refused in 0.1 s without any LLM call (cosine 0.744 < 0.8225).
- The follow-up "What accuracy does it reach on ImageNet?" was condensed with the previous question and retrieved the Swin Transformer results passages.

## Not done yet (Phase 6 and others)
- No generation-quality evaluation: faithfulness, answer relevance, context precision/recall, groundedness, citation correctness against a human-validated subset, RAGAS-style judging, cost per query. These need a working LLM (and ideally a judge model); only the lexical citation heuristics exist today.
- Multi-document synthesis quality and "research gap" answers are untested with a capable model.
- Answer explainability highlights (Step 8) and the chat UI (Step 10).

---
# Addendum (Step 12)
- **Compare mode fixed.** Selecting several papers now retrieves per paper and interleaves (`backend/rag/multidoc.py`); before, a dominating paper could crowd the others out of the context (both selected papers present for 8 of 15 cross-paper queries, now 15 of 15; regression test in `test_rag.py`).
- **Unscoped cross-paper questions** put both relevant papers in the context for only 2 of 15 queries; a pre-registered tuning study (candidate depth, diversification over the best N papers) found no gain worth adopting (test 1/5 before and after), so the baseline is kept. Select the papers to compare.
- The evidence-gate calibration, relevance classifier and context tuning were measured on the 48-paper corpus snapshot; the corpus now has 49 papers (see MODEL_BENCHMARKING.md).
- Still not evaluated: faithfulness, answer relevance, context precision/recall, groundedness, citation correctness against human judgement, cost per query (all need a working LLM).
