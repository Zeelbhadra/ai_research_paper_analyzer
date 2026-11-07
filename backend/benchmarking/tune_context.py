"""Choose the cross-document context settings (candidate depth, number of papers to diversify over).
    python -m backend.benchmarking.tune_context

Rule fixed in advance, using DEV + VALIDATION queries only:
  maximise the share of cross-paper queries whose context contains BOTH relevant papers, subject to the mean evidence coverage of
  all other queries (share of the evidence passage's shingles present in the context) dropping by at most 0.02 vs the baseline
  (depth 20, no diversification), and ADOPT only if it beats the baseline by >= MIN_GAIN, otherwise keep the baseline.
  The TEST split is evaluated once afterwards and only reported. A second check measures the explicit route (the user selects the
  papers), where the retrieval filter itself guarantees both papers can appear."""
from __future__ import annotations

import json

import numpy as np

from backend.benchmarking.dataset_builder import load_dataset, shingles
from backend.config.settings import get_settings
from backend.database.repositories import Repository
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.rag.context_builder import build_context
from backend.rag.multidoc import retrieve_per_document
from backend.rag.pipeline import CALIBRATION_PATH
from backend.retrieval.factory import build_retriever

GRID = [(d, m) for d in (20, 30, 50) for m in (None, 2, 3, 4)]
MAX_DROP = 0.02
MIN_GAIN = 0.2          # adopt only a clear gain (2 more queries out of 10); smaller differences are noise at this sample size


def evidence_coverage(items, evidence) -> float:
    cov = []
    for ev in evidence:
        es = shingles(ev["text"])
        have: set = set()
        for it in items:
            if it.chunk.document_id == ev["paper_id"]:
                have |= shingles(it.chunk.text)
        cov.append(len(es & have) / len(es) if es else 0.0)
    return float(np.mean(cov))


def main():
    s = get_settings()
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    chunks = {c.chunk_id: c for c in Repository(s.db_path).get_chunks("recursive-256-15")}
    r = build_retriever("hybrid:e5:1.0:0.5", EmbeddingManager(s.storage_dir), "recursive-256-15")
    recs = [x for x in ds["records"] if not x.get("excluded_reason")]
    cache: dict = {}

    def evaluate(split_names, depth, max_docs):
        both, other = [], []
        for q in recs:
            if q["split"] not in split_names:
                continue
            key = (q["query_id"], depth)
            if key not in cache:
                cache[key] = [(chunks[h.chunk_id], h.score, h.rank) for h in r.retrieve(q["query"], depth)]
            ctx = build_context(q["query"], cache[key], 6, 1600, 3, max_docs=max_docs)
            if q["category"] == "cross_paper":
                both.append(len({i.chunk.document_id for i in ctx.items} & set(q["relevant_paper_ids"])) == 2)
            else:
                other.append(evidence_coverage(ctx.items, q["evidence"]))
        return {"cross_both_papers": float(np.mean(both)), "n_cross": len(both), "other_evidence_coverage": float(np.mean(other)), "n_other": len(other)}

    tune = ("dev", "validation")
    rows = {f"depth{d}_docs{m}": {"depth": d, "max_docs": m, **evaluate(tune, d, m)} for d, m in GRID}
    base = rows["depth20_docsNone"]
    ok = {k: v for k, v in rows.items() if base["other_evidence_coverage"] - v["other_evidence_coverage"] <= MAX_DROP}
    best_key = max(ok, key=lambda k: (ok[k]["cross_both_papers"], -ok[k]["depth"], ok[k]["max_docs"] is None, -(ok[k]["max_docs"] or 99)))
    best = ok[best_key]
    adopted = best["cross_both_papers"] - base["cross_both_papers"] >= MIN_GAIN
    if not adopted:
        best_key, best = "depth20_docsNone", base
    test_base, test_best = evaluate(("test",), 20, None), evaluate(("test",), best["depth"], best["max_docs"])
    explicit = []                                                                       # user selects the two papers => filtered retrieval
    for q in recs:
        if q["category"] != "cross_paper":
            continue
        allowed = set(q["relevant_paper_ids"])
        hits = retrieve_per_document(lambda qq, k, al: r.retrieve(qq, k, al), q["query"], sorted(allowed), 20,
                                     lambda d: (lambda cid: chunks[cid].document_id == d))
        ctx = build_context(q["query"], [(chunks[h.chunk_id], h.score, h.rank) for h in hits], 6, 1600, 3)
        explicit.append(len({i.chunk.document_id for i in ctx.items} & allowed) == 2)
    out = {"rule": f"max cross-paper both-papers coverage on dev+validation s.t. other-query evidence coverage drop <= {MAX_DROP}", "grid_dev_validation": rows,
           "adopted_change": adopted, "chosen": {"cross_doc_depth": best["depth"], "cross_doc_max_docs": best["max_docs"]}, "test_baseline": test_base, "test_chosen": test_best,
           "explicit_paper_selection_both_in_context": f"{sum(explicit)}/{len(explicit)}"}
    (s.storage_dir / "benchmark" / "context_tuning.json").write_text(json.dumps(out, indent=1))
    cal = json.loads(CALIBRATION_PATH.read_text())
    cal["cross_doc_depth"], cal["cross_doc_max_docs"] = best["depth"], best["max_docs"]
    CALIBRATION_PATH.write_text(json.dumps(cal, indent=1))
    print(f"{'config':20s} cross both-papers (n={base['n_cross']})   other-query evidence coverage (n={base['n_other']})")
    for k, v in rows.items():
        print(f"{k:20s} {v['cross_both_papers']:.2f}{'':>26s} {v['other_evidence_coverage']:.3f}{'   <- chosen' if k == best_key else ''}{'   (rejected: coverage drop)' if k not in ok else ''}")
    print("ADOPTED CHANGE:", adopted, "| explicit selection, both papers in context:", f"{sum(explicit)}/{len(explicit)}")
    print("TEST baseline:", {k: round(v, 3) for k, v in test_base.items()}, "\nTEST chosen:  ", {k: round(v, 3) for k, v in test_best.items()})


if __name__ == "__main__":
    main()
