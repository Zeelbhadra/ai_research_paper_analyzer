"""Query validation, intent/scope detection, follow-up condensation.

Everything here is deterministic. Intent and section hints are recorded for explainability and prompting only;
they do NOT change retrieval ranking (an unmeasured boost would violate the measure-before-adopt rule). Hard
filters are applied only when the caller explicitly selects documents/sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_LEN, MIN_LEN = 1000, 3

INTENTS: list[tuple[str, re.Pattern]] = [
    # overview requests come first: "explain this paper in simple terms: what problem ..., what did they find" also contains
    # words of other intents, but the user is asking for the whole picture
    ("summary", re.compile(r"\bexplain (this|the) (paper|study|article|work|research)\b|\bin (simple|plain) (terms|words|language)\b|\beli5\b|\bfor a beginner\b|\bwhat is (this|the) (paper|study|article) about\b|\bwhy (does|is|do|are) .{0,40}(matter|important|significant)\b", re.I)),
    ("comparison", re.compile(r"\b(compar\w*|contrast|differ\w*|versus|vs\.?|similarit\w*|better than|trade-?offs?)\b", re.I)),
    ("limitations", re.compile(r"\b(limitations?|weakness\w*|drawbacks?|shortcomings?|threats? to validity|fail\w*)\b", re.I)),
    ("research_gap", re.compile(r"\b(research gaps?|gaps?|open (problems?|questions?|challenges?)|future (work|research|directions?)|unexplored|missing)\b", re.I)),
    ("evolution", re.compile(r"\b(evolution|evolved|history of|over time|trend\w*|progress(ion)?|timeline|state of the art)\b", re.I)),
    ("paper_discovery", re.compile(r"\bwhich (papers?|studies|works?|articles?)\b|\bwhat papers\b|\bpapers? (that|which|on|about)\b", re.I)),
    ("datasets", re.compile(r"\b(datasets?|corpus|corpora|benchmarks?|data sets?)\b", re.I)),
    ("results", re.compile(r"\b(results?|find(ings?|s)?|discover\w*|performance|accuracy|f1|auc|outperform\w*|achiev\w*|improv\w*|score)\b", re.I)),
    ("methodology", re.compile(r"\b(method(s|ology)?|approach|algorithm|architecture|pipeline|how (does|do|is|are|did)|technique|procedure|implement\w*)\b", re.I)),
    ("definition", re.compile(r"\b(what is|what are|what (does|do) .* (mean|refer)|define|definition|meaning of|refers? to|jargon|terminology|technical terms?)\b", re.I)),
    ("summary", re.compile(r"\b(summar\w*|overview|main (idea|contribution)s?|key (points?|contributions?)|tl;?dr)\b", re.I)),
]
SECTION_HINTS = {"methodology": ["methodology"], "results": ["results", "discussion"], "limitations": ["discussion", "conclusion"],
                 "research_gap": ["conclusion", "discussion"], "definition": ["introduction", "abstract", "related_work"],
                 "datasets": ["methodology", "results"], "summary": ["abstract", "conclusion"]}
CROSS_CUES = re.compile(r"\b(across|papers|studies|works|these (two )?(papers|studies)|both|each|all (the )?papers|compar\w*|literature)\b", re.I)
SINGLE_CUES = re.compile(r"\b(this|the) (paper|study|article|work)\b", re.I)
FOLLOWUP_START = re.compile(r"^(and|also|but|then|so)|^(what|how) about", re.I)
# Only true anaphora counts ("this"/"that" are usually determiners: "in this benchmark").
FOLLOWUP_PRONOUN = re.compile(r"(it|its|they|their|them|these|those|the (same|former|latter))", re.I)


class InvalidQuery(ValueError):
    pass


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    retrieval_query: str                      # standalone form actually sent to retrieval
    intent: str
    scope: str                                # 'document' | 'cross_document'
    document_ids: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)       # explicit user filter
    section_hints: list[str] = field(default_factory=list)  # informational only
    is_followup: bool = False
    warnings: list[str] = field(default_factory=list)


def validate_and_normalize(q: str) -> str:
    if not isinstance(q, str):
        raise InvalidQuery("query must be a string")
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < MIN_LEN:
        raise InvalidQuery("query is too short")
    if len(q) > MAX_LEN:
        raise InvalidQuery(f"query is too long (max {MAX_LEN} characters)")
    if not re.search(r"[A-Za-z]{2}", q):
        raise InvalidQuery("query must contain words")
    return q


def detect_intent(q: str) -> str:
    for name, rx in INTENTS:
        if rx.search(q):
            return name
    return "general"


def process_query(q: str, document_ids: list[str] | None = None, sections: list[str] | None = None,
                  history: list[dict] | None = None) -> ProcessedQuery:
    norm = validate_and_normalize(q)
    docs = list(dict.fromkeys(document_ids or []))
    intent = detect_intent(norm)
    warnings: list[str] = []
    if len(docs) == 1 and intent != "comparison":
        scope = "document"
    elif len(docs) > 1 or intent in ("comparison", "paper_discovery", "research_gap", "evolution") or CROSS_CUES.search(norm):
        scope = "cross_document"
    else:
        scope = "cross_document"
    if SINGLE_CUES.search(norm) and not docs:
        warnings.append("question refers to 'this paper' but no document was selected; searching all papers")
    follow = False
    rq = norm
    prev = [m for m in (history or []) if m.get("role") == "user"]
    if prev and (len(norm.split()) <= 8 or FOLLOWUP_START.search(norm) or (len(norm.split()) <= 16 and FOLLOWUP_PRONOUN.search(norm))):
        # Deterministic condensation: carry the previous standalone retrieval query for context.
        follow = True
        rq = f"{prev[-1].get('retrieval_query') or prev[-1]['content']} {norm}"[:MAX_LEN]
    return ProcessedQuery(q, norm, rq, intent, scope, docs, list(sections or []), SECTION_HINTS.get(intent, []), follow, warnings)
