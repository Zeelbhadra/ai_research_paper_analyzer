# Explainability (Step 8)

Two levels, with an explicit statement of what each can and cannot claim.

## 1. Retrieval explainability (`backend/explainability/`)
**Exact, model-free facts about the production ranking** (always shown): per-component corpus rank and score (TF-IDF cosine, dense cosine), the RRF parts (`1.0/(60+tfidf_rank)` and `0.5/(60+dense_rank)`), the exact per-term decomposition of the TF-IDF score, matched query terms, the best-matching sentence of the chunk under the dense model, and the retrieval path (hybrid / sparse / TF-IDF fallback).

**A relevance model + SHAP + LIME** (answers "which interpretable signals make a chunk look relevant?").
- *Task:* given (query, candidate chunk from the production top-20), predict relevant (benchmark grade >= 1). 13 interpretable features: dense cosine and rank, TF-IDF cosine and rank, query-term and bigram overlap, title (metadata) overlap, section indicators and intent-section match, chunk position, chunk length.
- *Training/selection:* trained on dev; candidate = depth-3 GBM, depth-2 regularized GBM, logistic regression; chosen by **validation PR-AUC** (rule fixed in advance) -> **logistic regression**. The depth-3 GBM fit dev almost perfectly (ROC-AUC 0.997) but generalized worse. Threshold = best F1 on validation.
- *Held-out classification metrics (the one place ROC/PR-AUC, F1 and accuracy are meaningful):*

| split | positive rate | ROC-AUC | PR-AUC | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|---|---|
| validation | 10.2% | 0.888 | 0.623 | 0.542 | 0.605 | 0.571 | 0.907 |
| test | 11.9% | 0.856 | 0.546 | 0.396 | 0.737 | 0.515 | 0.835 |

With a ~12% positive rate, accuracy alone is misleading; PR-AUC is the informative number. Labels are the synthetic benchmark labels (incomplete), so precision is pessimistic.
- *Not a reranker:* used to rerank the hybrid's top-20 it changes nDCG@10 by -0.007 (validation, CI -0.072..+0.049) and +0.008 (test, CI -0.015..+0.032): no significant gain on either split, so it is **not adopted** and exists to be explained. (The model was retrained after 13 paper titles were repaired; an earlier version showed a small test-set gain that did not survive retraining, a reminder that this difference is within noise.)
- *SHAP* (exact linear SHAP, log-odds units): additivity verified (max error 0.0). Global importance on validation: dense rank (0.77) >> TF-IDF rank (0.39) > TF-IDF cosine (0.26) > title overlap (0.25) > dense cosine (0.24). *LIME* (tabular, local linear, on request via `lime=1`).
- *SHAP vs LIME agreement* on 40 held-out instances: top-3 feature overlap 0.42 (chance 0.23), Spearman of |attribution| 0.56, mean LIME local R^2 only 0.28. **Treat LIME as the weaker, more approximate explanation; SHAP is exact for this model.**
- *Sign artefacts:* some coefficients are negative although the feature intuitively helps (title overlap -0.30, section-matches-intent -0.26, results-section -0.33 per standard deviation). 54% of the unlabeled hard negatives in the training pool come from the same paper as a relevant chunk, so paper-level and metadata features cannot separate them and the model uses them as corrections. Read signs as properties of this training pool, not as causal statements (the Explainability view says so).
- *Limits:* SHAP/LIME explain the surrogate model over named features. They do **not** explain embedding dimensions or the semantic content of the E5 vectors, and they do not reproduce the RRF arithmetic (the exact RRF parts are shown separately). Rank features are correlated, so credit is shared among them. The relevance model is tied to a production model version; a different production model disables it until retrained (`python -m backend.explainability.relevance_model train`).

## 2. Answer explainability
`explain_answer` turns a RAG response into per-sentence evidence: supporting paper, page range, section, URL, the quoted passage, **character-offset highlights** of the matching terms/numbers, an ordinal evidence strength (strong / moderate / weak / none), and a kind: `stated_in_source`, `interpretation` (model-labelled synthesis), or `unverified` (uncited or unsupported, with a warning). It also reports each paper's share of cited claims for multi-document answers and lists unsupported claims.
- Evidence strength is **lexical and uncalibrated**; it is not a probability of truth or an entailment check.
- For extractive answers sentences are quoted verbatim, so "strong" shows provenance only, not that the passage answers the question (the note in the output says so).

## API
`GET /api/explain/retrieval?q=&k=&lime=0|1&documents=` - `POST /api/explain/answer` - `GET /api/explain/model`.

## Verified on the real system (`python -m backend.benchmarking.explain_demo`, `storage/benchmark/explain_demo.json`)
Top result for "How does the Swin Transformer reduce the computational complexity of self-attention?": rank 1 in both TF-IDF and dense corpus rankings, P(relevant) 0.995; SHAP: TF-IDF cosine +1.80, dense rank +1.78, bigram overlap +1.66, TF-IDF rank +0.97 (log-odds); best passage "...computational complexity of its self-attention is quadratic..." (cosine 0.866).

## Bugs found while building this (all fixed, with regression tests)
- The sentence splitter detached a trailing `[S#]` citation from its sentence and split after "e.g." / "Fig." / "et al."; this had made extractive answers look uncited and affected the Step 7 citation checker.
- A redundant feature (the RRF score, an exact function of two rank features) produced confusing opposite-sign attributions and was removed.
