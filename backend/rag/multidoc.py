"""Per-document retrieval for questions scoped to several explicitly selected papers.

A single filtered top-N list can be dominated by one paper, so the others never reach the context ("compare these two papers"
silently answering from one). Retrieving per paper and interleaving by rank guarantees every selected paper contributes
candidates; the context builder's per-paper cap then keeps the mix balanced."""
from __future__ import annotations

from typing import Callable

from backend.vectorstore.base import Hit


def interleave(lists: list[list[Hit]]) -> list[Hit]:
    """Round-robin merge (rank 1 of each list first, then rank 2, ...), ranks reassigned; each paper keeps its own order."""
    out: list[Hit] = []
    for i in range(max((len(x) for x in lists), default=0)):
        for lst in lists:
            if i < len(lst):
                out.append(Hit(lst[i].chunk_id, lst[i].score, len(out) + 1))
    return out


def retrieve_per_document(retrieve: Callable[[str, int, Callable[[str], bool] | None], list[Hit]], query: str, doc_ids: list[str], depth: int,
                          make_allow: Callable[[str], Callable[[str], bool] | None]) -> list[Hit]:
    per = max(5, depth // max(len(doc_ids), 1))
    return interleave([retrieve(query, per, make_allow(d)) for d in doc_ids])
