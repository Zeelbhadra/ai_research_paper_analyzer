"""Reproducible benchmark-dataset construction.

Honest scope: queries are SYNTHETIC (LLM-generated from corpus passages) and only automatically
validated; `human_verified` stays False until a person reviews them (see export_review_csv /
apply_review_csv). Known biases are documented in MODEL_BENCHMARKING.md:
  * a query is written from one passage, so other relevant passages may exist but be unlabeled
    (labels are incomplete => absolute recall/precision are pessimistic);
  * generated questions may reuse passage vocabulary (favours lexical retrieval) - mitigated with an
    n-gram-copy filter, not eliminated.

Relevance is stored as *evidence text*, not only chunk ids, so the same dataset can score any chunking
configuration (`map_evidence_to_chunks`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from backend.preprocessing.chunking import Chunk

CATEGORIES = ["definition", "methodology", "experimental_results", "comparative", "technical", "cross_paper"]
SECTION_POOLS = {
    "definition": ["introduction", "related_work", "other", "abstract"],
    "methodology": ["methodology", "other"],
    "experimental_results": ["results", "discussion", "other"],
    "comparative": ["related_work", "results", "discussion", "introduction"],
    "technical": ["methodology", "results", "other", "introduction"],
    "cross_paper": ["introduction", "methodology", "related_work", "other"],
}
PROMPTS = {
    "definition": "Write ONE question asking what a specific concept, term or problem described in the passage is, or means.",
    "methodology": "Write ONE question asking how a specific method, algorithm, model or procedure described in the passage works or is built.",
    "experimental_results": "Write ONE question asking about a specific experimental finding, measurement or performance result reported in the passage.",
    "comparative": "Write ONE question asking how approaches, models or settings described in the passage compare, differ or trade off against each other.",
    "technical": "Write ONE detailed technical question about an implementation detail, parameter, dataset, architecture component or formula in the passage.",
}
BANNED = re.compile(r"\b(the|this|that) (passage|text|excerpt|paper|article|study|document|authors?)\b|according to|in the (given|above|following)", re.I)
_W = re.compile(r"[a-z0-9]+")


def words(t: str) -> list[str]:
    return _W.findall(t.lower())


def shingles(t: str, n: int = 6) -> set[tuple]:
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def ngram_copy_ratio(query: str, passage: str, n: int = 4) -> float:
    q = words(query)
    qs = {tuple(q[i:i + n]) for i in range(max(len(q) - n + 1, 0))}
    return len(qs & shingles(passage, n)) / len(qs) if qs else 0.0


def usable_chunk(c: Chunk) -> bool:
    w = c.text.split()
    if not (100 <= len(w) <= 400) or c.section in ("references", "acknowledgements"):
        return False
    alpha = sum(ch.isalpha() for ch in c.text) / max(len(c.text), 1)
    return alpha > 0.7          # drops tables / equations / garbled extraction


def split_documents(doc_ids: list[str], seed: int = 42, fractions=(0.6, 0.2, 0.2)) -> dict[str, str]:
    """Document-level split => no passage or paper appears in two splits."""
    ids = sorted(doc_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_dev, n_val = round(n * fractions[0]), round(n * fractions[1])
    out = {}
    for i, d in enumerate(ids):
        out[d] = "dev" if i < n_dev else "validation" if i < n_dev + n_val else "test"
    return out


def validate_query(q: str, passage: str, max_copy: float = 0.25) -> str | None:
    """Return None if OK else the rejection reason."""
    q = q.strip()
    n = len(q.split())
    if not q.endswith("?"):
        return "no_question_mark"
    if not 6 <= n <= 40:
        return "bad_length"
    if BANNED.search(q):
        return "context_dependent_phrase"
    if "\n" in q:
        return "multiline"
    if ngram_copy_ratio(q, passage) > max_copy:
        return "copies_passage"
    return None


def evidence_of(c: Chunk) -> dict:
    return {"paper_id": c.document_id, "paper_title": c.paper_title, "page_start": c.page_start,
            "page_end": c.page_end, "section": c.section, "text": c.text}


def map_evidence_to_chunks(evidence: list[dict], chunks: list[Chunk],
                           full_thr: float = 0.5, partial_thr: float = 0.1) -> dict[str, int]:
    """Graded relevance of `chunks` (any chunking config) for the given evidence passages.
    coverage = share of the evidence's 6-word shingles present in the chunk, same document only.
      >= full_thr -> grade 2 ;  >= partial_thr -> grade 1 ;  else 0 (omitted)."""
    rels: dict[str, int] = {}
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    for ev in evidence:
        es = shingles(ev["text"])
        if not es:
            continue
        for c in by_doc.get(ev["paper_id"], []):
            cov = len(es & shingles(c.text)) / len(es)
            g = 2 if cov >= full_thr else 1 if cov >= partial_thr else 0
            if g:
                rels[c.chunk_id] = max(rels.get(c.chunk_id, 0), g)
    return rels


@dataclass
class BuildConfig:
    seed: int = 42
    canonical_chunk_cfg: str = "recursive-256-15"
    per_category: dict[str, int] = field(default_factory=lambda: {"dev": 15, "validation": 5, "test": 5})
    min_answerable: float = 0.6
    oversample: int = 3            # candidate attempts per wanted query


def _gen_messages(cat: str, passage: str) -> list[dict]:
    return [{"role": "system", "content":
             "You write search questions for a research-paper search engine. The question must be fully "
             "self-contained and specific: include the key technical terms, but do NOT copy phrases from the "
             "passage, and never say 'the passage', 'the paper', 'the authors' or 'according to'. "
             "Output only the question."},
            {"role": "user", "content": f"{PROMPTS[cat]}\n\nPassage:\n{passage}\n\nQuestion:"}]


def _cross_messages(a: str, b: str) -> list[dict]:
    return [{"role": "system", "content":
             "You write search questions for a research-paper search engine. The question must be self-contained, "
             "specific, and require information from BOTH passages (e.g. relating, contrasting or combining them). "
             "Do not copy phrases, and never say 'passage', 'paper' or 'according to'. Output only the question."},
            {"role": "user", "content": f"Passage A:\n{a}\n\nPassage B:\n{b}\n\nQuestion:"}]


def _check_messages(q: str, passages: list[str]) -> list[dict]:
    ctx = "\n\n".join(passages)
    return [{"role": "user", "content": f"Passage(s):\n{ctx}\n\nQuestion: {q}\n\nCan the question be answered using the "
                                        f"passage(s)? Answer Yes or No."}]


def paper_similarity_pairs(chunks: list[Chunk], docs: list[str]) -> list[tuple[str, str]]:
    """Doc pairs ranked by TF-IDF cosine of concatenated text (cross-paper questions need shared topic)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = {d: " ".join(c.text for c in chunks if c.document_id == d) for d in docs}
    ds = sorted(texts)
    if len(ds) < 2:
        return []
    X = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit_transform([texts[d] for d in ds])
    S = (X @ X.T).toarray()
    pairs = [(S[i, j], ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))]
    return [(a, b) for _, a, b in sorted(pairs, reverse=True)]


def build_dataset(chunks: list[Chunk], generate: Callable[[list[dict]], str],
                  yes_prob: Callable[[list[dict]], float], cfg: BuildConfig | None = None,
                  log: Callable[[str], None] = print,
                  on_progress: Callable[[list[dict]], None] | None = None,
                  resume: list[dict] | None = None) -> dict:
    cfg = cfg or BuildConfig()
    rng = random.Random(cfg.seed)
    docs = sorted({c.document_id for c in chunks})
    split_of = split_documents(docs, cfg.seed)
    pool = [c for c in chunks if usable_chunk(c)]
    records: list[dict] = []
    rejected: dict[str, int] = {}
    seen_q: set[str] = set()
    for r in resume or []:      # continue an interrupted build; finished (split, category) cells are skipped
        records.append(r)
        seen_q.add(re.sub(r"\W+", " ", r["query"].lower()).strip())

    def accept(cat, split, q, evs, ans):
        qid = f"q{len(records) + 1:03d}"
        records.append({
            "query_id": qid, "query": q, "category": cat, "split": split,
            "relevant_paper_ids": sorted({e["paper_id"] for e in evs}),
            "relevant_chunk_ids": [], "relevance": {}, "evidence": evs,
            "answerable_prob": round(ans, 3), "label_source": "synthetic_llm_auto_validated",
            "human_verified": False})
        seen_q.add(re.sub(r"\W+", " ", q.lower()).strip())

    for split in ("dev", "validation", "test"):
        split_docs = [d for d in docs if split_of[d] == split]
        for cat in CATEGORIES:
            want = cfg.per_category[split]
            got, tries = sum(1 for r in records if r["split"] == split and r["category"] == cat), 0
            if got >= want:
                continue
            if cat == "cross_paper":
                pairs = paper_similarity_pairs(chunks, split_docs)
                cands = []
                for a, b in pairs:
                    pa = [c for c in pool if c.document_id == a and c.section in SECTION_POOLS[cat]]
                    pb = [c for c in pool if c.document_id == b and c.section in SECTION_POOLS[cat]]
                    if pa and pb:
                        cands.append((rng.choice(pa), rng.choice(pb)))
                # top-similarity pairs first; cycle if more are needed
                cands = cands[: want * cfg.oversample]
                items = [(c,) if False else c for c in cands]
            else:
                cp = [c for c in pool if split_of[c.document_id] == split and c.section in SECTION_POOLS[cat]]
                rng.shuffle(cp)
                # spread across documents: round-robin by document
                by_doc: dict[str, list[Chunk]] = {}
                for c in cp:
                    by_doc.setdefault(c.document_id, []).append(c)
                items = []
                while any(by_doc.values()) and len(items) < want * cfg.oversample:
                    for d in sorted(by_doc):
                        if by_doc[d]:
                            items.append(by_doc[d].pop())
            for it in items:
                if got >= want:
                    break
                tries += 1
                if cat == "cross_paper":
                    a, b = it
                    evs, texts = [evidence_of(a), evidence_of(b)], [a.text, b.text]
                    q = generate(_cross_messages(a.text, b.text))
                else:
                    evs, texts = [evidence_of(it)], [it.text]
                    q = generate(_gen_messages(cat, it.text))
                q = q.strip().split("\n")[0].strip().strip('"')
                reason = validate_query(q, " ".join(texts))
                if reason is None and re.sub(r"\W+", " ", q.lower()).strip() in seen_q:
                    reason = "duplicate"
                ans = 0.0
                if reason is None:
                    ans = yes_prob(_check_messages(q, texts))
                    if ans < cfg.min_answerable:
                        reason = "not_answerable"
                if reason:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                accept(cat, split, q, evs, ans)
                got += 1
            log(f"[{split}/{cat}] accepted {got}/{want} after {tries} candidates")
            if on_progress:
                on_progress(records)
    # fill canonical-config relevance
    canon = [c for c in chunks]
    for r in records:
        r["relevance"] = map_evidence_to_chunks(r["evidence"], canon)
        r["relevant_chunk_ids"] = sorted(r["relevance"])
    body = {"config": {"seed": cfg.seed, "canonical_chunk_cfg": cfg.canonical_chunk_cfg,
                       "per_category": cfg.per_category, "min_answerable": cfg.min_answerable},
            "doc_split": split_of, "records": records, "rejected": rejected}
    flag_context_dependent(body)
    body["version"] = hashlib.sha1(json.dumps([(r["query"], r["category"], r["split"]) for r in records]).encode()).hexdigest()[:10]
    return body


def save_dataset(ds: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ds, indent=1))


def load_dataset(path: Path) -> dict:
    ds = json.loads(Path(path).read_text())
    for r in ds["records"]:
        if not any(g > 0 for g in r["relevance"].values()):
            raise ValueError(f"{r['query_id']} has no relevance labels")
    return ds


CONTEXT_DEPENDENT = re.compile(
    r"\bpassages?\b|\bgiven\b|in the context of the|\b(algorithm|figure|table|equation|section|theorem|lemma)\s+\d", re.I)


def flag_context_dependent(ds: dict) -> int:
    """Mark queries that cannot stand alone in a corpus-wide search (they refer to 'the given passage',
    'Algorithm 4', ...). Flagged records stay in the file with `excluded_reason`; evaluation skips them."""
    n = 0
    for r in ds["records"]:
        if not r.get("excluded_reason") and CONTEXT_DEPENDENT.search(r["query"]):
            r["excluded_reason"] = "context_dependent"
            n += 1
    return n


def records_for(ds: dict, split: str, include_excluded: bool = False) -> list[dict]:
    return [r for r in ds["records"] if r["split"] == split and (include_excluded or not r.get("excluded_reason"))]


def assert_no_leakage(ds: dict) -> None:
    split_of = ds["doc_split"]
    for r in ds["records"]:
        for p in r["relevant_paper_ids"]:
            if split_of[p] != r["split"]:
                raise AssertionError(f"{r['query_id']} ({r['split']}) references paper in {split_of[p]}")


def export_review_csv(ds: dict, path: Path) -> None:
    """For human verification: reviewer sets `verdict` to keep/drop and optionally edits the query."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "split", "category", "query", "evidence_titles", "evidence_pages", "evidence_text", "verdict", "edited_query"])
        for r in ds["records"]:
            w.writerow([r["query_id"], r["split"], r["category"], r["query"],
                        " | ".join(e["paper_title"] for e in r["evidence"]),
                        " | ".join(f"{e['page_start']}-{e['page_end']}" for e in r["evidence"]),
                        " || ".join(e["text"] for e in r["evidence"]), "", ""])


def apply_review_csv(ds: dict, path: Path) -> dict:
    """Apply a reviewed CSV: verdict 'keep' => human_verified; 'drop' => removed; edited_query replaces query."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = {r["query_id"]: r for r in csv.DictReader(f)}
    out = []
    for r in ds["records"]:
        v = rows.get(r["query_id"], {})
        verdict = (v.get("verdict") or "").strip().lower()
        if verdict == "drop":
            continue
        if verdict == "keep":
            r["human_verified"] = True
            r["label_source"] = "synthetic_llm_human_verified"
            if (v.get("edited_query") or "").strip():
                r["query"] = v["edited_query"].strip()
        out.append(r)
    ds = dict(ds)
    ds["records"] = out
    return ds
