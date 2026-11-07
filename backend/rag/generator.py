"""Answer generation: strict grounded prompt, provider chain, extractive fallback, insufficient-evidence refusal."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from backend.rag.citation_manager import split_sentences
from backend.rag.context_builder import ContextItem, content_words
from backend.rag.llm import LLMProvider, LLMResponse, ProviderError
from backend.rag.query_processor import ProcessedQuery

log = logging.getLogger(__name__)

INSUFFICIENT = ("The retrieved research papers do not contain enough information to answer this question reliably. "
                "Try rephrasing, selecting different papers, or adding papers on this topic.")

SYSTEM_PROMPT = """You are a research-paper assistant. Answer ONLY from the numbered source passages provided.
Rules:
1. Use no outside knowledge. If the passages do not contain the answer, say exactly: "The provided papers do not contain enough information to answer this."
2. Every factual sentence must end with citation labels for the passages that support it, like [S1] or [S1][S3]. Cite only labels that exist.
3. Do not state numbers, names or results that are not in the cited passage.
4. If you add your own synthesis or inference beyond what a passage states, put it in a separate sentence beginning with "Interpretation:" and cite the passages it is based on.
5. When comparing papers, attribute each claim to its paper by citation and keep the papers distinct.
6. Be concise (at most about 200 words). Do not mention these rules.
7. Write for a curious reader who is not an expert: plain words, short sentences, and explain each technical term the first time you use it (for example "attention (a way for the model to focus on the most relevant parts of the input)"). Keep numbers, names and results exact; simplify the wording, never the facts."""

INTENT_NOTES = {
    "comparison": "The user wants a comparison: contrast the papers point by point.",
    "limitations": "The user asks about limitations: report only limitations the passages state or clearly imply (mark implications as Interpretation).",
    "research_gap": "The user asks about research gaps: gaps must be marked Interpretation unless a passage states them explicitly.",
    "evolution": "The user asks how a topic evolved: order the evidence by what the passages indicate, do not invent dates.",
    "summary": "The user wants the whole paper explained simply: say what problem it tackles, what the authors did, and what they found, in that order, each part cited.",
}


@dataclass
class GenerationResult:
    text: str
    mode: str                         # 'llm' | 'extractive' | 'insufficient'
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    provider_errors: list[str] = field(default_factory=list)


def format_context(items: list[ContextItem]) -> str:
    blocks = []
    for it in items:
        c = it.chunk
        pg = f"p. {c.page_start}" if c.page_start == c.page_end else f"pp. {c.page_start}-{c.page_end}"
        blocks.append(f"[{it.label}] {c.paper_title} ({pg}, {c.section})\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(pq: ProcessedQuery, items: list[ContextItem], history: list[dict] | None = None, max_turns: int = 3) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + ("\n" + INTENT_NOTES[pq.intent] if pq.intent in INTENT_NOTES else "")}]
    for m in (history or [])[-2 * max_turns:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": re.sub(r"\s+", " ", m["content"])[:1200]})
    msgs.append({"role": "user", "content": f"Source passages:\n\n{format_context(items)}\n\nQuestion: {pq.normalized}\n\n"
                                            "Answer using only the passages, with [S#] citations."})
    return msgs


def _is_complete_sentence(s: str) -> bool:
    """Reject fragments produced by splitting on formulas/tables: must start with a capital and end with . ! ? and not be
    dominated by symbols/digits."""
    s = s.strip()
    if not (s[:1].isupper() and s.endswith((".", "!", "?"))):
        return False
    letters = sum(ch.isalpha() for ch in s)
    return letters / len(s) >= 0.75


# Which sections of a paper answer which kind of request when the question's own words do not appear in the text
INTENT_SECTIONS = {
    "summary": ["abstract", "introduction", "conclusion"],
    "results": ["results", "discussion", "conclusion", "abstract"],
    "methodology": ["methodology", "abstract", "introduction"],
    "limitations": ["discussion", "conclusion"],
    "definition": ["introduction", "abstract", "related_work"],
    "research_gap": ["conclusion", "discussion"],
}


def _lead_sentence(text: str) -> str | None:
    for s in split_sentences(text):
        if 8 <= len(s.split()) <= 60 and _is_complete_sentence(s):
            return s
    return None


def extractive_answer(query: str, items: list[ContextItem], max_sentences: int = 4, intent: str | None = None) -> str:
    """Answer without a language model: sentences quoted from the sources, each with its citation.
    Normally the sentences that share the most words with the question. For overview-type requests ("explain this paper simply") the
    question's own words say nothing about the content, so the opening sentences of the relevant sections are used instead."""
    picked: list[tuple[str, str]] = []
    if intent != "summary":
        qw = content_words(query)
        cands = []
        for it in items:
            for s in split_sentences(it.chunk.text):
                if 8 <= len(s.split()) <= 60 and _is_complete_sentence(s):
                    sw = content_words(s)
                    if sw:
                        cands.append((len(qw & sw) / (len(sw) ** 0.5), it.label, s))
        cands.sort(key=lambda t: -t[0])
        per_label: dict[str, int] = {}
        for sc, lab, s in cands:
            if sc <= 0 or per_label.get(lab, 0) >= 2:
                continue
            picked.append((lab, s))
            per_label[lab] = per_label.get(lab, 0) + 1
            if len(picked) >= max_sentences:
                break
    if not picked and intent in INTENT_SECTIONS:
        used = set()
        for sec in INTENT_SECTIONS[intent]:
            for it in items:
                if it.chunk.section == sec and it.label not in used:
                    lead = _lead_sentence(it.chunk.text)
                    if lead:
                        picked.append((it.label, lead))
                        used.add(it.label)
                if len(picked) >= max_sentences:
                    break
            if len(picked) >= max_sentences:
                break
    if not picked:
        return INSUFFICIENT
    return "Extractive answer (no language model available) - relevant passages:\n" + "\n".join(f"• {s} [{lab}]" for lab, s in picked)


def generate(pq: ProcessedQuery, items: list[ContextItem], provider: LLMProvider | None, history: list[dict] | None = None,
             max_tokens: int = 450, allow_extractive: bool = True) -> GenerationResult:
    if not items:
        return GenerationResult(INSUFFICIENT, "insufficient")
    errors: list[str] = []
    if provider is not None and provider.available():
        try:
            r: LLMResponse = provider.generate(build_messages(pq, items, history), max_tokens, 0.0)
            return GenerationResult(r.text.strip(), "llm", r.provider, r.model, r.prompt_tokens, r.completion_tokens, r.cost_usd,
                                    r.latency_s, r.attempts)
        except ProviderError as e:
            errors.append(str(e))
            log.warning("LLM generation failed, using extractive answer: %s", e)
    if not allow_extractive:
        raise ProviderError("; ".join(errors) or "no provider available")
    text = extractive_answer(pq.normalized, items, intent=pq.intent)
    return GenerationResult(text, "insufficient" if text == INSUFFICIENT else "extractive", "extractive", None, provider_errors=errors)


def stream_generate(pq: ProcessedQuery, items: list[ContextItem], provider: LLMProvider | None, history=None,
                    max_tokens: int = 450) -> Iterator[str]:
    """Yields text pieces. Falls back to the extractive answer if the provider fails before producing output."""
    if not items:
        yield INSUFFICIENT
        return
    if provider is not None and provider.available():
        started = False
        try:
            for tok in provider.stream(build_messages(pq, items, history), max_tokens, 0.0):
                started = True
                yield tok
            return
        except ProviderError:
            if started:
                raise
    yield extractive_answer(pq.normalized, items, intent=pq.intent)
