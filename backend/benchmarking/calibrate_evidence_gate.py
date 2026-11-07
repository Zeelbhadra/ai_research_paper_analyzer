"""Calibrate the RAG 'insufficient evidence' gate (min query-term coverage of the retrieved context).

    python -m backend.benchmarking.calibrate_evidence_gate

Threshold is fitted on DEV in-domain queries + a calibration half of hand-written off-topic questions, then
evaluated on VALIDATION/TEST in-domain queries and the held-out half of off-topic questions. Off-topic
questions are written by the assistant (not human-curated); the set is small, so treat the numbers as indicative.
"""
from __future__ import annotations

import json

import numpy as np

from backend.benchmarking.dataset_builder import load_dataset, records_for
from backend.config.settings import get_settings
from backend.rag.context_builder import build_context
from backend.rag.pipeline import CALIBRATION_PATH, RAGConfig
from backend.retrieval.production import ProductionRetrieval
from backend.retrieval.retriever import make_filter

OFF_TOPIC = [  # even index -> calibration, odd index -> held-out
    "What is the recipe for a classic sourdough bread starter?", "Who won the FIFA World Cup in 2018?",
    "How does CRISPR gene editing cut DNA?", "What is the capital city of Australia?",
    "How do blockchain consensus mechanisms prevent double spending?", "Which medications treat type 2 diabetes?",
    "What causes the northern lights?", "How do I change a flat tire on a bicycle?",
    "What were the main causes of the French Revolution?", "How does a four-stroke combustion engine work?",
    "What is the plot of Shakespeare's Hamlet?", "How do vaccines train the human immune system?",
    "What is the tallest mountain in South America?", "How is cheese made from milk?",
    "Explain how photosynthesis converts light into chemical energy.", "What are good strategies for long-term stock investing?",
    "How do airplanes stay in the air?", "What is the boiling point of ethanol at sea level?",
    "How do tectonic plates cause earthquakes?", "Who painted the Mona Lisa and when?",
    "What are the symptoms of vitamin D deficiency?", "How does the Roman law of inheritance work?",
    "What is the best way to learn to play guitar?", "How do hurricanes form over the ocean?",
    "What is the chemical formula of table salt?", "How does a nuclear reactor generate electricity?",
    "What are the rules of cricket?", "How do bees make honey?", "What is the GDP of Brazil?",
    "How does the human heart pump blood?"]


def main():
    s = get_settings()
    pr = ProductionRetrieval()
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    ver = pr.service.status.primary_version

    def signals(q: str) -> tuple[float, float]:
        res = pr.service.retrieve(q, 20, None)
        cands = [(pr.chunks[h.chunk_id], h.score, h.rank) for h in res.hits if h.chunk_id in pr.chunks]
        cos, _ = pr.service.evidence_signal(q, [c[0].chunk_id for c in cands[:5]])
        return build_context(q, cands).coverage, cos

    ind = {sp: np.array([signals(r["query"]) for r in records_for(ds, sp)]) for sp in ("dev", "validation", "test")}
    off = np.array([signals(q) for q in OFF_TOPIC])
    off_cal, off_held = off[0::2], off[1::2]
    out = {"dense_model_version": ver, "note": "off-topic questions were written by the assistant (30, not human-curated); thresholds fitted on dev + half of them",
           "n": {"dev": len(ind["dev"]), "validation": len(ind["validation"]), "test": len(ind["test"]), "off_cal": len(off_cal), "off_heldout": len(off_held)}}
    for name, j, lo, hi in (("dense", 1, 0.5, 1.0), ("lexical", 0, 0.05, 1.0)):
        grid = np.round(np.arange(lo, hi, 0.0025 if name == "dense" else 0.025), 4)
        score = lambda t: (ind["dev"][:, j] >= t).mean() + (off_cal[:, j] < t).mean() - 1       # Youden J
        top = max(score(t) for t in grid)
        best_ts = [t for t in grid if score(t) >= top - 1e-9]
        t = float(np.median(best_ts))                                                               # middle of the optimal plateau
        if name == "lexical":                                                                       # weak signal: be lenient (keep >=95% of dev)
            t = float(min(t, np.percentile(ind["dev"][:, 0], 5)))
        out[name] = {"threshold": t, "dev_in_domain_kept": float((ind["dev"][:, j] >= t).mean()),
                     "off_topic_refused_cal": float((off_cal[:, j] < t).mean()),
                     "heldout_validation_kept": float((ind["validation"][:, j] >= t).mean()),
                     "heldout_test_kept": float((ind["test"][:, j] >= t).mean()),
                     "heldout_off_topic_refused": float((off_held[:, j] < t).mean())}
    out["min_dense_cosine"], out["min_coverage_lexical"] = out["dense"]["threshold"], out["lexical"]["threshold"]
    out["use_reranker"] = False
    rr = s.storage_dir / "benchmark" / "rerank_benchmark.json"
    if rr.exists():
        out["use_reranker"] = bool(json.loads(rr.read_text())["adopt"])
    CALIBRATION_PATH.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
