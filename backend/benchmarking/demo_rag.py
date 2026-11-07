"""Real end-to-end RAG run: python -m backend.benchmarking.demo_rag   (slow: local LLM on CPU)

Chain = OpenAI (fails with the account's 429 until credits are added) -> local Qwen2.5-1.5B -> extractive."""
from __future__ import annotations

import json
import time

from backend.config.settings import get_settings
from backend.rag.llm import LocalProvider, OpenAIProvider, ProviderChain
from backend.rag.pipeline import RAGConfig, RAGPipeline
from backend.retrieval.production import ProductionRetrieval

QUESTIONS = [
    ("single", "How does the Swin Transformer reduce the computational complexity of self-attention?", None),
    ("compare", "Compare how the ride-sharing papers model the role of autonomous vehicles versus human drivers.", None),
    ("off_topic", "What is the best way to bake sourdough bread?", None),
]


def summarize(r: dict) -> dict:
    return {"mode": r["answer_mode"], "sufficient": r["sufficient_evidence"], "evidence": r["evidence"],
            "intent": r["query"]["intent"], "scope": r["query"]["scope"], "retrieval": r["retrieval"],
            "generation": r["generation"], "timings": r["timings"], "citation_quality": r["citation_quality"],
            "answer": r["answer"],
            "sources": [(x["label"], x["title"][:50], x["pages"], x["section"], x["cited"]) for x in r["references"]],
            "sentence_status": [(s["status"], s["support"]) for s in r["sentences"]]}


def main():
    s = get_settings()
    pr = ProductionRetrieval()
    pipe = RAGPipeline(pr, ProviderChain([OpenAIProvider(), LocalProvider()]), None, RAGConfig.load())
    out = {"gate_config": {"min_dense_cosine": pipe.cfg.min_dense_cosine, "reranker": pipe.cfg.use_reranker}, "runs": {}}
    history = []
    for name, q, docs in QUESTIONS:
        t = time.perf_counter()
        r = pipe.ask(q, history, docs)
        out["runs"][name] = summarize(r) | {"wall_s": round(time.perf_counter() - t, 1)}
        print(name, "->", r["answer_mode"], out["runs"][name]["wall_s"], "s", flush=True)
        if name == "single" and r["answer_mode"] != "insufficient":
            history = [{"role": "user", "content": q, "retrieval_query": q}, {"role": "assistant", "content": r["answer"]}]
    # follow-up using history (retrieval only + extractive to keep it fast)
    fast = RAGPipeline(pr, None, None, RAGConfig.load())
    f = fast.ask("What accuracy does it reach on ImageNet?", history)
    out["runs"]["followup_extractive"] = summarize(f)
    # same question through the pure extractive path for comparison
    out["runs"]["single_extractive"] = summarize(fast.ask(QUESTIONS[0][1]))
    (s.storage_dir / "benchmark" / "rag_demo.json").write_text(json.dumps(out, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    main()
