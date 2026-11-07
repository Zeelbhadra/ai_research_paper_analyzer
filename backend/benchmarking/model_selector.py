"""Guarded promotion of a selected candidate (selection itself lives in selection.py). No LLM involved."""
from __future__ import annotations

from typing import Callable

from backend.benchmarking.model_registry import ModelRegistry


def maybe_promote(registry: ModelRegistry, run_id: str, winner_name: str | None, thr: dict,
                  validate: Callable[[dict], tuple[bool, str]]) -> tuple[bool, str]:
    """Promote `winner_name` if (a) it exists/eligible, (b) `validate(candidate)` passes (index loads, health ok,
    smoke query works) and (c) it beats the current production candidate's score in the SAME run by at least
    `min_gain_over_current_production`. Otherwise the working production model is left untouched."""
    if winner_name is None:
        return False, "no eligible candidate; production unchanged"
    run = registry.get_run(run_id)
    cand = next(c for c in run["candidates"] if c["name"] == winner_name)
    prod = registry.production
    if prod and prod["version"] == cand["version"]:
        return False, f"{winner_name} is already in production"
    ok, why = validate(cand)
    if not ok:
        return False, f"validation failed for {winner_name}: {why}; production unchanged"
    if prod:
        cur = next((c for c in run["candidates"] if c["version"] == prod["version"]), None)
        if cur is not None and cand["overall_score"] - cur["overall_score"] < thr.get("min_gain_over_current_production", 0.0):
            return False, "gain over current production below required margin; production unchanged"
    registry.promote(run_id, winner_name, f"highest eligible overall score ({cand['overall_score']:.2f})")
    return True, f"promoted {winner_name}"
