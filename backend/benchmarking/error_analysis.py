"""Where do the retrievers fail?   python -m backend.benchmarking.error_analysis

For every retained benchmark query and every model, classify the outcome from the top-10 ranking:
  hit@1            a relevant chunk is ranked first
  hit@2-5 / @6-10  first relevant chunk further down (right passage found, ranked lower)
  right paper, wrong passage   no relevant chunk in the top 10 but a chunk of a relevant PAPER is
  wrong paper      no chunk of any relevant paper in the top 10
Cross-paper queries are additionally scored by how many of their 2 papers are covered.
Output: storage/benchmark/error_analysis.json (+ printed summary). Deterministic; uses the same labels as the benchmark."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.retrieval.factory import build_retriever

MODELS = ["tfidf", "sbert", "e5", "bge", "hybrid:e5:1.0:0.5"]
CFG = "recursive-256-15"


def classify(ranked: list[str], rels: dict[str, int], doc_of: dict[str, str], rel_docs: set[str]) -> str:
    for i, c in enumerate(ranked[:10], 1):
        if rels.get(c, 0) > 0:
            return "hit@1" if i == 1 else "hit@2-5" if i <= 5 else "hit@6-10"
    return "right_paper_wrong_passage" if any(doc_of[c] in rel_docs for c in ranked[:10]) else "wrong_paper"


def main():
    s = get_settings()
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    chunks = Repository(s.db_path).get_chunks(CFG)
    doc_of = {c.chunk_id: c.document_id for c in chunks}
    title_of = {c.document_id: c.paper_title for c in chunks}
    mgr, cache = EmbeddingManager(s.storage_dir), {}
    R = {m: build_retriever(m, mgr, CFG, _cache=cache) for m in MODELS}
    recs = [r for r in ds["records"] if not r.get("excluded_reason")]
    out = {"n_queries": len(recs), "by_model": {}, "by_model_category": {}, "cross_paper_coverage": {}, "examples": []}
    cat = defaultdict(lambda: defaultdict(Counter))
    cov = defaultdict(Counter)
    for m in MODELS:
        cnt = Counter()
        for r in recs:
            rels = map_evidence_to_chunks(r["evidence"], chunks)
            rel_docs = set(r["relevant_paper_ids"])
            ranked = [h.chunk_id for h in R[m].retrieve(r["query"], 10)]
            k = classify(ranked, rels, doc_of, rel_docs)
            cnt[k] += 1
            cat[m][r["category"]][k] += 1
            if r["category"] == "cross_paper":
                cov[m][len({doc_of[c] for c in ranked} & rel_docs)] += 1
            if m.startswith("hybrid") and k in ("wrong_paper", "right_paper_wrong_passage") and len(out["examples"]) < 40:
                out["examples"].append({"query_id": r["query_id"], "split": r["split"], "category": r["category"], "outcome": k, "query": r["query"],
                                        "expected_papers": [title_of[d][:60] for d in sorted(rel_docs)], "top1_paper": title_of[doc_of[ranked[0]]][:60] if ranked else None})
        out["by_model"][m] = dict(cnt)
    out["by_model_category"] = {m: {c: dict(v) for c, v in d.items()} for m, d in cat.items()}
    out["cross_paper_coverage"] = {m: {f"{k}_of_2_papers": v for k, v in sorted(c.items())} for m, c in cov.items()}
    (s.storage_dir / "benchmark" / "error_analysis.json").write_text(json.dumps(out, indent=1))
    order = ["hit@1", "hit@2-5", "hit@6-10", "right_paper_wrong_passage", "wrong_paper"]
    print(f"{'model':22s}" + "".join(f"{o:>27s}" for o in order))
    for m in MODELS:
        print(f"{m:22s}" + "".join(f"{out['by_model'][m].get(o, 0):>27d}" for o in order))
    print("cross-paper coverage:", out["cross_paper_coverage"])


if __name__ == "__main__":
    main()
