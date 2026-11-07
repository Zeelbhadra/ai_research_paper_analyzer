"""Ranking metrics with graded relevance, plus classification metrics for a *defined* binary task.

Conventions
- `ranked`: list of chunk ids, best first. `rels`: {chunk_id: grade>0}; absent => grade 0.
- Precision/Recall/MRR/MAP binarise at grade >= 1. nDCG uses the grades (gain = 2^g - 1).
- Recall@K is relative to ALL relevant items for the query (not just those retrievable in top-K).
- Queries with no relevant items are rejected at load time, never silently scored as 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

KS = (1, 3, 5, 10)


def precision_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    return sum(1 for c in ranked[:k] if rels.get(c, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    total = sum(1 for g in rels.values() if g > 0)
    if total == 0:
        raise ValueError("query has no relevant items")
    return sum(1 for c in ranked[:k] if rels.get(c, 0) > 0) / total


def reciprocal_rank(ranked: Sequence[str], rels: dict[str, int], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k], 1):
        if rels.get(c, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], rels: dict[str, int], k: int | None = None) -> float:
    """AP over the (optionally truncated) ranking, normalised by the total number of relevant items."""
    total = sum(1 for g in rels.values() if g > 0)
    if total == 0:
        raise ValueError("query has no relevant items")
    hits, s = 0, 0.0
    for i, c in enumerate(ranked[:k] if k else ranked, 1):
        if rels.get(c, 0) > 0:
            hits += 1
            s += hits / i
    return s / total


def dcg(gains: Iterable[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    ideal = dcg(sorted((g for g in rels.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        raise ValueError("query has no relevant items")
    return dcg(rels.get(c, 0) for c in ranked[:k]) / ideal


def query_metrics(ranked: Sequence[str], rels: dict[str, int], ks: Sequence[int] = KS) -> dict[str, float]:
    m: dict[str, float] = {}
    for k in ks:
        m[f"P@{k}"] = precision_at_k(ranked, rels, k)
        m[f"R@{k}"] = recall_at_k(ranked, rels, k)
        m[f"nDCG@{k}"] = ndcg_at_k(ranked, rels, k)
    m["MRR@10"] = reciprocal_rank(ranked, rels, 10)
    m["AP@10"] = average_precision(ranked, rels, 10)
    m["Hit@10"] = float(any(rels.get(c, 0) > 0 for c in ranked[:10]))
    return m


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_pvalue(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 42) -> float:
    """Two-sided paired bootstrap test on mean(a-b) (per-query values aligned by query)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    centred = d - d.mean()
    boots = centred[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float((np.abs(boots) >= abs(d.mean())).mean())


@dataclass
class EvalResult:
    name: str
    per_query: dict[str, dict[str, float]]                    # query_id -> metrics
    latencies_ms: list[float] = field(default_factory=list)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        v = [m[metric] for m in self.per_query.values()]
        return float(np.mean(v)) if v else float("nan")

    def means(self) -> dict[str, float]:
        if not self.per_query:
            return {}
        keys = next(iter(self.per_query.values())).keys()
        return {k: self.mean(k) for k in keys} | {"MAP": self.mean("AP@10")}

    def ci(self, metric: str) -> tuple[float, float]:
        return bootstrap_ci([m[metric] for m in self.per_query.values()])


def evaluate_retriever(name: str, retrieve: Callable[[str, int], list[str]], queries: list[dict],
                       rels_for: Callable[[dict], dict[str, int]], k: int = 10) -> EvalResult:
    """Run every query through `retrieve(query_text, k) -> ranked chunk ids`, time each call."""
    import time
    per_q, lat = {}, []
    cats: dict[str, list[dict]] = {}
    for q in queries:
        rels = rels_for(q)
        if not any(g > 0 for g in rels.values()):
            raise ValueError(f"{q['query_id']} has no relevant chunks under this chunking config")
        t = time.perf_counter()
        ranked = retrieve(q["query"], k)
        lat.append((time.perf_counter() - t) * 1000)
        per_q[q["query_id"]] = query_metrics(ranked, rels)
        cats.setdefault(q.get("category", "?"), []).append(per_q[q["query_id"]])
    res = EvalResult(name, per_q, lat)
    for c, ms in cats.items():
        res.by_category[c] = {k_: float(np.mean([m[k_] for m in ms])) for k_ in ("nDCG@10", "R@5", "MRR@10")} | {"n": len(ms)}
    return res


def budgeted_evidence_coverage(ranked_ids: Sequence[str], chunk_text: dict[str, str], chunk_doc: dict[str, str],
                               evidence: list[dict], budget_words: int = 1500) -> float:
    """Granularity-fair RAG metric: take ranked chunks until the word budget is used, then measure what share of each
    evidence passage's 6-word shingles appears in the retrieved text of the same paper (mean over evidence passages).
    Unlike nDCG over chunks it does not reward simply making chunks bigger."""
    from backend.benchmarking.dataset_builder import shingles
    used, kept = 0, []
    for cid in ranked_ids:
        n = len(chunk_text[cid].split())
        if kept and used + n > budget_words:
            break
        kept.append(cid)
        used += n
    cover = []
    for ev in evidence:
        es = shingles(ev["text"])
        if not es:
            continue
        have: set = set()
        for cid in kept:
            if chunk_doc[cid] == ev["paper_id"]:
                have |= shingles(chunk_text[cid])
        cover.append(len(es & have) / len(es))
    return float(np.mean(cover)) if cover else 0.0


# ---- Classification metrics: ONLY for a real binary task (e.g. a trained relevance classifier) ----
def classification_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(set(y_true.tolist())) < 2:
        raise ValueError("need both classes present for ROC-AUC / PR-AUC")
    pred = (y_score >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y_true, y_score)), "pr_auc": float(average_precision_score(y_true, y_score)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)), "accuracy": float(accuracy_score(y_true, pred))}
