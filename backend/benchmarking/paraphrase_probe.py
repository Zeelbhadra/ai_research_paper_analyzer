"""Lexical-bias probe: reword each query to reduce vocabulary overlap, keep the same relevance labels, and
compare models. DIAGNOSTIC ONLY (paraphrases by a small LLM can drift in meaning; drift hurts all models alike).

    python -m backend.benchmarking.paraphrase_probe generate     # slow (local LLM)
    python -m backend.benchmarking.paraphrase_probe evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from backend.benchmarking.dataset_builder import load_dataset, map_evidence_to_chunks, words
from backend.benchmarking.retrieval_evaluator import evaluate_retriever, paired_bootstrap_pvalue
from backend.config.settings import get_settings

STOP = set("the a an of to in on for and or is are was were be by with as at from that this it its how what which does do did using used use".split())


def overlap(q: str, ev: str) -> float:
    qs = {w for w in words(q) if w not in STOP and len(w) > 2}
    return len(qs & set(words(ev))) / max(len(qs), 1)


def generate():
    from backend.benchmarking.llm_local import LocalLLM
    s = get_settings()
    ds = load_dataset(s.storage_dir / "benchmark" / "dataset.json")
    llm = LocalLLM().load()
    out = {}
    for r in ds["records"]:
        if r.get("excluded_reason"):
            continue
        msg = [{"role": "system", "content": "Rewrite the search question so it means exactly the same thing but uses different "
                "wording: replace words with synonyms or descriptions and restructure the sentence. Keep proper names, numbers and "
                "symbols. Output only the rewritten question, ending with a question mark."},
               {"role": "user", "content": r["query"]}]
        p = llm.generate(msg, 70).split("\n")[0].strip().strip('"')
        ev = " ".join(e["text"] for e in r["evidence"])
        ok = p.endswith("?") and 4 <= len(p.split()) <= 60 and p.lower() != r["query"].lower()
        out[r["query_id"]] = {"original": r["query"], "paraphrase": p if ok else r["query"], "valid": ok,
                              "overlap_before": round(overlap(r["query"], ev), 3), "overlap_after": round(overlap(p if ok else r["query"], ev), 3)}
        print(r["query_id"], ok, out[r["query_id"]]["overlap_before"], "->", out[r["query_id"]]["overlap_after"], flush=True)
        (s.storage_dir / "benchmark" / "paraphrases.json").write_text(json.dumps(out, indent=1))


def evaluate(splits=("dev", "validation", "test"), cfg_key="recursive-256-15"):
    from backend.database.repositories import Repository
    from backend.embeddings.embedding_manager import EmbeddingManager
    from backend.retrieval.factory import build_retriever
    s = get_settings()
    b = s.storage_dir / "benchmark"
    ds = load_dataset(b / "dataset.json")
    para = json.loads((b / "paraphrases.json").read_text())
    chunks = Repository(s.db_path).get_chunks(cfg_key)
    mgr, cache = EmbeddingManager(s.storage_dir), {}
    names = ["tfidf", "sbert", "e5", "bge", "hybrid:e5:1.0:0.5"]
    R = {n: build_retriever(n, mgr, cfg_key, _cache=cache) for n in names}
    recs = [r for r in ds["records"] if not r.get("excluded_reason") and r["query_id"] in para and r["split"] in splits]
    rels = {r["query_id"]: map_evidence_to_chunks(r["evidence"], chunks) for r in recs}
    valid = [r for r in recs if para[r["query_id"]]["valid"]]
    res = {}
    for variant in ("original", "paraphrase"):
        qs = [{**r, "query": para[r["query_id"]][variant]} for r in valid]
        res[variant] = {n: evaluate_retriever(n, lambda q, k, n=n: [h.chunk_id for h in R[n].retrieve(q, k)], qs, lambda q: rels[q["query_id"]])
                        for n in names}
    summ = {"n_queries": len(valid), "mean_overlap_before": float(np.mean([para[r["query_id"]]["overlap_before"] for r in valid])),
            "mean_overlap_after": float(np.mean([para[r["query_id"]]["overlap_after"] for r in valid])), "models": {}}
    for n in names:
        o, p = res["original"][n], res["paraphrase"][n]
        qids = list(o.per_query)
        summ["models"][n] = {"nDCG@10_original": round(o.mean("nDCG@10"), 4), "nDCG@10_paraphrase": round(p.mean("nDCG@10"), 4),
                             "drop": round(o.mean("nDCG@10") - p.mean("nDCG@10"), 4),
                             "p_drop": round(paired_bootstrap_pvalue([o.per_query[q]["nDCG@10"] for q in qids], [p.per_query[q]["nDCG@10"] for q in qids]), 4)}
    best = max(summ["models"], key=lambda n: summ["models"][n]["nDCG@10_paraphrase"])
    summ["best_on_paraphrase"] = best
    pb = res["paraphrase"]
    summ["p_tfidf_vs_best_paraphrase"] = None if best == "tfidf" else round(paired_bootstrap_pvalue(
        [pb[best].per_query[q]["nDCG@10"] for q in pb[best].per_query], [pb["tfidf"].per_query[q]["nDCG@10"] for q in pb[best].per_query]), 4)
    (b / "paraphrase_probe.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"generate": generate, "evaluate": evaluate}[sys.argv[1]]()
