"""Evidence-driven, weight-free model selection.

1. Feasibility: quality floor, health check, resource budgets (memory derived from the machine's available RAM).
2. Reference: feasible candidate with the highest mean primary metric (nDCG@10).
3. Non-inferiority: a candidate may replace the reference only if the lower bound of the paired-bootstrap
   confidence interval of mean(candidate - reference) on per-query primary metric is >= -margin, i.e. it is
   *demonstrably* not materially worse. With small query sets the interval is wide, so the quality leader wins;
   with more evidence an equivalent but cheaper model can win.
4. Among {reference} U non-inferior candidates, choose the smallest resource footprint = geometric mean of
   (latency_p50, memory, indexing_seconds) each divided by the smallest value among feasible candidates. This is
   scale-free and has no tunable weights.
Also exposes a 0-100 `quality_index` (plain mean of the quality metrics) for display, and the Pareto front.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "selection.json"


def load_policy(path: Path | None = None) -> dict:
    return json.loads(Path(path or CONFIG_PATH).read_text())


def available_memory_mb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2**20


def derive_budgets(policy: dict, available_mb: float | None = None) -> dict:
    b = dict(policy["budgets"])
    avail = available_memory_mb() if available_mb is None else available_mb
    b["memory_mb"] = avail * b.pop("memory_fraction_of_available_ram")
    b["available_ram_mb"] = avail
    return b


def paired_diff_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95, n_boot: int = 5000, seed: int = 42):
    """Bootstrap CI for mean(a - b) over aligned per-query values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    alpha = (1 - confidence) / 2
    return float(d.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


@dataclass
class Candidate:
    name: str
    metrics: dict[str, float | None]          # quality metrics + latency_p50_ms/latency_p95_ms/throughput_qps/memory_mb/indexing_seconds
    per_query: list[float]                    # primary metric per query, aligned across candidates
    healthy: bool = True


@dataclass
class Selection:
    winner: str | None
    reference: str | None
    feasible: dict[str, list[str]]            # name -> infeasibility reasons ([] => feasible)
    non_inferior: dict[str, dict]
    footprint: dict[str, float]
    quality_index: dict[str, float]
    pareto: list[str]
    budgets: dict
    trace: list[str] = field(default_factory=list)


def quality_index(metrics: dict, names: list[str]) -> float:
    vals = [metrics.get(n) for n in names]
    return 100.0 * float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else float("nan")


def infeasible_reasons(c: Candidate, policy: dict, budgets: dict) -> list[str]:
    m, r = c.metrics, []
    for k, floor in policy["quality_floor"].items():
        v = m.get(k)
        if v is None:
            r.append(f"{k} not measured")
        elif v < floor:
            r.append(f"{k} {v:.3f} < floor {floor}")
    if policy.get("require_health_check", True) and not c.healthy:
        r.append("health check failed")
    checks = [("latency_p95_ms", "<=", budgets["latency_p95_ms"]), ("memory_mb", "<=", budgets["memory_mb"]),
              ("indexing_seconds", "<=", budgets["reindex_window_seconds"]), ("throughput_qps", ">=", budgets["min_qps"])]
    for k, op, lim in checks:
        v = m.get(k)
        if v is None:
            r.append(f"{k} not measured")
        elif (op == "<=" and v > lim) or (op == ">=" and v < lim):
            r.append(f"{k} {v:.1f} violates budget {op} {lim:.1f}")
    return r


def pareto_front(cands: list[Candidate]) -> list[str]:
    """Non-dominated set over (quality up; latency, memory, indexing down)."""
    def vec(c):
        m = c.metrics
        return (m["nDCG@10"], -m["latency_p50_ms"], -m["memory_mb"], -m["indexing_seconds"])
    out = []
    for c in cands:
        v = vec(c)
        if not any(all(x >= y for x, y in zip(vec(o), v)) and any(x > y for x, y in zip(vec(o), v)) for o in cands if o is not c):
            out.append(c.name)
    return out


def select(cands: list[Candidate], policy: dict | None = None, budgets: dict | None = None) -> Selection:
    policy = policy or load_policy()
    budgets = budgets or derive_budgets(policy)
    ni = policy["non_inferiority"]
    qnames = policy["quality_metrics_for_index"]
    feas = {c.name: infeasible_reasons(c, policy, budgets) for c in cands}
    ok = [c for c in cands if not feas[c.name]]
    sel = Selection(None, None, feas, {}, {}, {c.name: quality_index(c.metrics, qnames) for c in cands},
                    pareto_front(cands) if all(c.metrics.get("latency_p50_ms") and c.metrics.get("memory_mb") and c.metrics.get("indexing_seconds") for c in cands) else [],
                    budgets)
    t = sel.trace
    t.append(f"budgets: memory<={budgets['memory_mb']:.0f}MB (={policy['budgets']['memory_fraction_of_available_ram']:.0%} of {budgets['available_ram_mb']:.0f}MB available), "
             f"p95<={budgets['latency_p95_ms']}ms, qps>={budgets['min_qps']}, reindex<={budgets['reindex_window_seconds']}s")
    for n, r in feas.items():
        if r:
            t.append(f"infeasible {n}: {'; '.join(r)}")
    if not ok:
        t.append("no feasible candidate; production unchanged")
        return sel
    ref = max(ok, key=lambda c: (c.metrics["nDCG@10"], c.name))
    sel.reference = ref.name
    t.append(f"reference (highest {policy['primary_metric']} among feasible): {ref.name} = {ref.metrics['nDCG@10']:.3f}")
    pool = [ref]
    for c in ok:
        if c is ref:
            continue
        mean, lo, hi = paired_diff_ci(c.per_query, ref.per_query, ni["confidence"], ni["n_boot"])
        passed = lo >= -ni["margin"]
        sel.non_inferior[c.name] = {"mean_diff": round(mean, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "non_inferior": passed}
        t.append(f"{c.name} vs {ref.name}: diff {mean:+.3f} CI[{lo:+.3f}, {hi:+.3f}] -> "
                 f"{'non-inferior (lower bound >= -' + str(ni['margin']) + ')' if passed else 'not shown to be non-inferior'}")
        if passed:
            pool.append(c)
    mins = {k: min(c.metrics[k] for c in ok) for k in ("latency_p50_ms", "memory_mb", "indexing_seconds")}
    for c in ok:
        sel.footprint[c.name] = math.exp(sum(math.log(c.metrics[k] / mins[k]) for k in mins) / len(mins))
    win = min(pool, key=lambda c: (sel.footprint[c.name], -c.metrics["nDCG@10"], c.name))
    sel.winner = win.name
    t.append(f"candidates tied on quality: {[c.name for c in pool]}; smallest footprint -> {win.name} "
             f"(footprints: { {c.name: round(sel.footprint[c.name], 2) for c in pool} })")
    return sel
