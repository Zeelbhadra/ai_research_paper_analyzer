# Model Benchmarking (in progress: Steps 3-4 complete)

All numbers below were measured on this repository's corpus by the scripts named here. Nothing is estimated.

## Corpus
45 papers processed from `data/` (3 PDFs are truncated downloads; one more was not processed), 1,878 chunks with `recursive-256-15` (256 whitespace-tokens, 15% overlap). Chunking-strategy/size sweeps are NOT done yet; every number here is for this single chunk config.

## Models (CPU only, no GPU)
| Family | Checkpoint | Dim | Notes |
|---|---|---|---|
| TF-IDF | sklearn, 1-2 grams, sublinear tf | sparse | lexical baseline/fallback |
| SBERT | sentence-transformers/all-MiniLM-L6-v2 | 384 | no prefixes |
| E5 | intfloat/e5-base-v2 | 768 | `query: ` / `passage: ` |
| BGE | BAAI/bge-base-en-v1.5 | 768 | instruction on query side only |

Corpus embedding wall-times (`storage/embed_timings_recursive-256-15.json`): SBERT 55.6 s, BGE 836.8 s, E5 1984.8 s.
**Caveat:** E5 ran while other jobs were using the CPU, so its time is inflated (same architecture/size as BGE). Latency/throughput/memory used for scoring will come from an isolated performance benchmark (Step 5), not from these numbers.

## Benchmark dataset (`storage/benchmark/dataset.json`, version `abb1ed6844`)
- 101 queries generated with local `Qwen/Qwen2.5-1.5B-Instruct` (greedy decoding, seed 42), 6 categories (definition, methodology, experimental_results, comparative, technical, cross_paper). OpenAI generation was not possible (account out of credits).
- Automatic validation: question form and length, banned context-dependent phrases, 4-gram copy ratio <= 0.25, de-duplication, and an LLM answerability check (P(Yes) >= 0.6).
- A post-hoc filter (applied **before** any model results were examined) excluded 12 context-dependent queries ("the given passage", "Algorithm 4", ...). **89 queries retained**: dev 44, validation 21, test 24. This is below the 100-query target.
- Splits are by document (27/9/9 papers), leakage asserted in code. **The test split has not been used yet.**
- Relevance is stored as evidence text and mapped to any chunking config by shingle coverage (>=50% of the evidence in a chunk = grade 2, >=10% = grade 1), so the same dataset can score other chunk sizes.
- **All labels are synthetic and `human_verified = false`.** `storage/benchmark/review.csv` is exported for manual review; `apply_review_csv` ingests the verdicts.

### Known biases
1. **Lexical bias.** On average 57% of a query's content words appear in its source passage (9% of queries >= 80%). This favours TF-IDF.
2. **Incomplete labels.** Each query is labeled from one passage (two for cross_paper); other passages may also answer it. Absolute precision/recall are therefore pessimistic.
3. **Small n.** Validation has 21 queries, so confidence intervals are wide.
4. The generator is a 1.5B model; some questions are vague or poorly posed. Human review has not happened.

## Retrieval results (nDCG@10 etc., `storage/benchmark/retrieval_*.json|csv`)
| Split | Model | P@1 | R@5 | MRR@10 | MAP | nDCG@10 (95% bootstrap CI) | paired p vs best |
|---|---|---|---|---|---|---|---|
| dev (44) | TF-IDF | 0.750 | 0.617 | 0.851 | 0.529 | 0.722 (0.654-0.790) | - |
| dev | SBERT | 0.614 | 0.436 | 0.726 | 0.397 | 0.558 (0.468-0.646) | <0.001 |
| dev | E5 | 0.682 | 0.594 | 0.800 | 0.504 | 0.662 (0.583-0.737) | 0.086 |
| dev | BGE | 0.659 | 0.571 | 0.784 | 0.481 | 0.641 (0.555-0.721) | 0.005 |
| validation (21) | TF-IDF | 0.762 | 0.601 | 0.849 | 0.519 | 0.718 (0.637-0.792) | - |
| validation | SBERT | 0.524 | 0.462 | 0.675 | 0.376 | 0.553 (0.429-0.672) | 0.005 |
| validation | E5 | 0.524 | 0.529 | 0.691 | 0.431 | 0.625 (0.517-0.732) | 0.111 |
| validation | BGE | 0.667 | 0.529 | 0.786 | 0.453 | 0.664 (0.546-0.774) | 0.315 |

Per-category nDCG@10 is in the JSON files. All models are weak on `cross_paper` (about 0.4-0.55).

### Interpretation (what the evidence supports, and what it doesn't)
- On this dataset **TF-IDF has the highest mean nDCG@10 on both splits**. It is significantly better than SBERT. Against E5 and BGE the difference is **not statistically significant on validation** (p = 0.11 and 0.32), and E5/BGE are close to TF-IDF on `definition` and `technical` queries.
- This does not show that TF-IDF is better in general. The lexical bias above plausibly inflates it. It does mean that **no dense model can currently be justified over TF-IDF**, and per the project rules the winner is decided by measured scores, not by assumption.
- Open questions for the next steps: whether a hybrid (BM25/TF-IDF + dense) beats every single model; how chunk size/strategy change the ranking; and whether a human-reviewed subset changes the picture.

## Reproduce
```
python -m backend.ingestion.extractor_compare data
python -m backend.benchmarking.embed_corpus --chunk-cfg recursive-256-15
python -m backend.benchmarking.build_dataset [--resume]
python -m backend.benchmarking.run_retrieval_eval --split dev        # or validation
```
Dataset generation took about 50 minutes of CPU time; it was interrupted once and resumed from a checkpoint, which changes the random sampling of the last three test cells relative to an uninterrupted run (output is otherwise deterministic).

---
# Step 5: bias audit, weight-free selection, performance, registry

Protocol: hybrid parameters tuned on **dev** only; selection on **validation** (21 queries); **test** (24 queries) evaluated once after the selection was recorded and only reported.

## 5a. Is TF-IDF's lead real? (audit before deciding)
1. **Label-completeness audit** (`storage/benchmark/adjudication_validation.json`). I reviewed the 24 chunks that some model ranked #1 on a validation query but that were unlabeled; 8 were genuinely relevant (33%). **The reviewer was an AI (Claude), not a human**, and only rank-1 items were reviewed (ranks 2-3 were not). Adding them raised every model by ~0.02-0.03 nDCG@10 and **did not change the ranking** (hybrid 0.769 > TF-IDF 0.747 > BGE 0.685 > E5 0.648 > SBERT 0.586). So label incompleteness does not explain the result.
2. **Paraphrase-robustness probe** (`paraphrase_probe.py`, all 89 retained queries, diagnostic only). The local LLM reworded each query; mean query-passage vocabulary overlap fell 0.61 -> 0.44 (a modest reduction; many technical terms were kept). Meaning drift is possible but affects all models alike.

| nDCG@10 | original | reworded | drop |
|---|---|---|---|
| TF-IDF | 0.710 | 0.602 | **-0.109** |
| SBERT | 0.539 | 0.500 | -0.040 |
| E5 | 0.645 | 0.574 | -0.071 |
| BGE | 0.643 | 0.560 | -0.083 |
| Hybrid TF-IDF+E5 | 0.733 | **0.651** | -0.082 |

TF-IDF degrades the most, and the hybrid is significantly better than TF-IDF on reworded queries (paired p = 0.001). Dense models alone still did not overtake TF-IDF here. **Conclusion:** the original queries do favour lexical matching, and correcting part of that bias strengthens the case for the hybrid; it does not make E5/BGE/SBERT better than TF-IDF on this corpus.
3. **Not done:** human review of `review.csv` (the 101 generated queries are still `human_verified = false`).

## 5b. Selection without hand-set weights (`selection.py`, `config/selection.json`)
The earlier fixed-weight composite was removed because its result was driven by arbitrary weights and ratio scaling (it let a 40x cheaper model win on cost alone). The policy now contains tolerances and budgets, not metric weights:
1. **Feasibility:** quality floor (nDCG@10 >= 0.50, Recall@5 >= 0.40), health check, and budgets: memory <= 25% of the RAM actually available at run time (1,740 MB in run0002), p95 latency <= 500 ms, throughput >= 2 qps, re-index time <= 2 h.
2. **Reference:** feasible candidate with the highest nDCG@10.
3. **Non-inferiority:** another candidate may replace it only if the lower bound of the 95% paired-bootstrap CI of the nDCG@10 difference is >= -0.02. Little data -> wide CI -> the quality leader wins; plenty of data and a truly equivalent cheaper model -> the cheaper model wins.
4. **Footprint tie-break** among the tied set: geometric mean of (latency, memory, indexing) ratios to the cheapest feasible candidate. No tunable weights.
`quality_index` (plain mean of nDCG@10, Recall@5, MRR@10, P@5, MAP) is a display number only. The ROC/PR-AUC term is not used because no relevance classifier exists yet.

## 5c. Isolated performance (fresh process each, CPU, 16 logical cores, 89 queries x 3 rounds, sequential)
| Candidate | p50 ms | p95 ms | QPS | Memory MB | Indexing s | Index size |
|---|---|---|---|---|---|---|
| TF-IDF | 1.2 | 1.5 | 831 | 28 | 0.7 (measured) | 7.9 MB |
| SBERT | 9.1 | 13.1 | 103 | 343 | 41 (extrapolated) | 3.0 MB |
| E5 | 38.9 | 52.5 | 25.4 | 629 | 536 (extrapolated) | 5.9 MB |
| BGE | 39.2 | 54.8 | 24.7 | 628 | 546 (extrapolated) | 5.9 MB |
| Hybrid (TF-IDF+E5) | 40.2 | 53.8 | 24.2 | 664 | 547 (extrapolated) | 13.7 MB |

Dense indexing time = measured s/chunk on a fixed 192-chunk sample x 1,878 chunks + index build (an extrapolation). The 1,985 s E5 corpus embedding time seen earlier was CPU contention.

## 5d. Result (run0002; validation selection, test reported afterwards)
| Candidate | nDCG@10 val (n=21) | nDCG@10 **test** (n=24, 95% CI) | Difference to reference on val (95% CI) |
|---|---|---|---|
| **Hybrid TF-IDF+E5 (reference, selected)** | 0.743 | 0.697 (0.595-0.796) | - |
| TF-IDF | 0.718 | 0.682 (0.584-0.784) | -0.025 (-0.081, +0.017) not shown non-inferior |
| BGE | 0.664 | 0.629 (0.506-0.749) | -0.079 (-0.202, +0.040) |
| E5 | 0.625 | 0.630 (0.519-0.743) | -0.118 (-0.235, -0.010) |
| SBERT | 0.553 | 0.493 (0.382-0.608) | -0.190 (-0.320, -0.067) |

**Production: hybrid TF-IDF + E5** (RRF weights 1.0/0.5, tuned on dev). **Fallback: TF-IDF** (the cheapest feasible model, now a genuinely different system). The same decision holds with the adjudicated labels (run0003, sensitivity only, no promotion).

### Honest limits
- The hybrid's advantage over TF-IDF alone is small and **not statistically significant on validation or test** (test p = 0.41); it was selected because TF-IDF could not be shown to be non-inferior with 21 queries, and the paraphrase probe supports it. More (and human-verified) queries could flip the decision towards TF-IDF; the selection will then do so automatically.
- Hybrid weights were chosen from a 9-point grid on 44 dev queries; some overfitting is possible.
- All labels are synthetic; E5/BGE/SBERT were evaluated with a single chunking configuration (256 tokens, 15% overlap); chunk-size/strategy sweeps remain to be run.

Reproduce:
```
python -m backend.benchmarking.tune_hybrid
python -m backend.benchmarking.performance_benchmark --all
python -m backend.benchmarking.run_benchmark --report-test
python -m backend.benchmarking.paraphrase_probe generate && python -m backend.benchmarking.paraphrase_probe evaluate
python -m backend.benchmarking.run_benchmark --adjudication storage/benchmark/adjudication_validation.json   # sensitivity
```
Artifacts: `storage/registry/registry.json|csv` (3 runs, 2 promotions), `storage/benchmark/*.json`.

---
# Step 6: automatic production selection and TF-IDF fallback

**Flow:** `run_benchmark` -> registry (history) -> guarded promotion -> `ProductionRetrieval.sync_with_registry()` ensures the new model's indexes exist (regenerated from the persistent embedding cache when only the index is missing), builds and probes the new retriever, and only then swaps it in. A model that cannot be built or fails its probe never replaces the working one (`backend/retrieval/production.py`, `service.py`).

**Fallback state machine** (`backend/retrieval/service.py`):
- A failing request is served by TF-IDF immediately. After 2 consecutive failures (or a failed health probe, or a primary that fails to load) the service enters FALLBACK and stops calling the primary.
- Recovery: probes with exponential backoff (5 s doubling, max 300 s); the primary returns after 2 consecutive healthy probes (a smoke query must return hits). Flapping resets the streak.
- If the production model is itself TF-IDF there is no independent fallback and failures surface as HTTP 503.
- Structured events (`storage/logs/retrieval_events.jsonl`): `primary_failure`, `fallback_activated`, `recovery_probe_*`, `primary_restored`, `model_switched`, `model_switch_failed`.
- API: `GET /api/retrieval/search`, `GET /api/retrieval/status` (`fallback_active` flag for the UI), `POST /api/retrieval/health-check`, `POST /api/retrieval/sync`, `GET /api/retrieval/events`.

**Measured with the real production hybrid + real E5** (`python -m backend.benchmarking.demo_fallback`, output in `storage/benchmark/fallback_demo.json`; faults injected into the live E5 encoder):
- Normal: 3 queries served by `hybrid:e5` in 27-82 ms.
- Fault: all 6 requests still returned 5 results; mode switched to fallback after the 2nd failure; fallback results were identical to plain TF-IDF.
- Repair: two health probes restored the primary; top-3 results matched the pre-fault results.
- Registry switch (on a copy of storage): promoting SBERT with its index deleted rebuilt the index (about 10 s, embeddings reused from cache) and switched. Promoting a model whose checkpoint cannot be loaded was rejected and SBERT kept serving.
- Not tested: real GPU/OOM failures, a corrupted FAISS file on disk (the loader rejects id/dimension mismatches but this was only unit-tested), concurrent load during a mode switch, and the UI banner (Step 10).

---
# Chunking sweep (closes Phase 1.3; run by `OptimizationAgent.chunk_sweep`, output `storage/benchmark/chunk_sweep_*.json`)

Nine configurations over the 49 PDFs, scored on the same queries by mapping each query's evidence text onto the new chunks. Only **TF-IDF and SBERT** were swept (E5/BGE/hybrid would need ~30 min of CPU embedding per config); results for the other models may differ.

**The obvious metric misleads.** nDCG@10 rises with chunk size (TF-IDF dev: 128 words 0.56, 256 -> 0.72, 512 -> 0.81, 768 -> 0.82) because larger chunks make retrieval easier by construction (fewer candidates; a large chunk contains the evidence more easily) while saying nothing about answer quality. A granularity-fair RAG measure fixes the context: **evidence coverage within a 1,500-word budget** (share of the evidence passage's shingles found in the top-ranked chunks that fit the budget, ~ the LLM context limit):

| config (words/overlap) | chunks | TF-IDF dev | TF-IDF val | SBERT dev | SBERT val |
|---|---|---|---|---|---|
| recursive 128/15% | 3772 | 0.63 | 0.63 | 0.65 | 0.67 |
| **recursive 256/15% (current default)** | 1878 | **0.89** | **0.93** | **0.75** | **0.71** |
| recursive 512/15% | 939 | 0.84 | 0.83 | 0.52 | 0.33 |
| recursive 768/15% | 631 | 0.79 | 0.63 | 0.37 | 0.23 |
| recursive 256/0% | 1650 | 0.74 | 0.83 | 0.63 | 0.68 |
| recursive 256/30% | 2255 | 0.82 | 0.80 | 0.66 | 0.60 |
| sentence 256/15% | 1890 | 0.74 | 0.82 | 0.66 | 0.65 |
| fixed 256/15% | 1871 | 0.81 | 0.88 | 0.66 | 0.67 |
| section 256/15% | 2005 | 0.73 | 0.83 | 0.67 | 0.61 |

Reading (dev n=44, validation n=21; 95% CIs are wide and overlap for many rows, see the JSON):
- Under a fixed context budget **256-word chunks are best on both splits for both models**; 128 fragments the evidence; 512/768 waste the budget (SBERT collapses, partly because MiniLM truncates long inputs).
- Recursive splitting is at or above the alternatives on both splits (fixed is next); sentence-only and section-aware chunking are not better.
- Overlap is inconclusive: 0%, 15% and 30% are non-monotonic (e.g. 0.74 / 0.89 / 0.82 on dev), i.e. within noise. 15% was kept.
- Conclusion: the existing default (recursive, 256, 15%) is supported by the sweep. Semantic chunking was implemented and unit-tested but **not benchmarked** (needs embedding every sentence).
- Not measured: downstream answer quality (needs a working LLM), E5/BGE behaviour, larger context budgets.

---
# Final corrections and error analysis (Step 12)

**Corpus repair and re-run.** One PDF ("Explanatory machine learning for sequential human teaching") had been silently dropped: it shared a journal-template first line with another paper and the title de-duplicator treated them as the same paper. After fixing de-duplication (junk/template titles are never an identity) the paper was added (83 chunks; 49 papers, 1,961 chunks) and the **core evaluation was re-run** as registry run `run0004` (selection on validation, test reported once more, dev re-evaluated for all five models). The winner is unchanged (hybrid TF-IDF + E5); TF-IDF's validation nDCG@10 moved from 0.718 to 0.701 because the new paper adds distractors, and its interval against the hybrid is now -0.042 [-0.096, +0.002]. **The numbers earlier in this file are from the earlier 48-paper snapshot (run0002); `EXPERIMENT_RESULTS.md` reports run0004.** The supporting studies (hybrid tuning, chunk sweep, reranker, bias probe, label audit, relevance classifier, gate calibration, context tuning, efficiency) were not repeated on the 49-paper corpus.

**Error analysis** (`python -m backend.benchmarking.error_analysis`, `storage/benchmark/error_analysis.json`): TF-IDF and the hybrid always retrieve a chunk of the right paper in the top 10 (the hybrid ranks the labelled passage first for 68 of 89 queries); dense-only models fully miss 4-7 queries. Cross-paper queries are a weakness: TF-IDF and the hybrid retrieve exactly one of the two relevant papers in all 15, never both. Context tuning (`tune_context.py`) found no worthwhile gain from deeper candidate pools or paper-level diversification and kept the baseline; the limit is retrieval recall (and incomplete labels). The analysis also exposed a real bug: with several papers *selected*, a single filtered top-20 could be dominated by one paper (both papers in context for 8 of 15 queries); retrieval is now per paper (15 of 15).
