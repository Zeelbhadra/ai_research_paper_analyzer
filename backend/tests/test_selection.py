import numpy as np
import pytest

from backend.benchmarking.selection import Candidate, derive_budgets, load_policy, paired_diff_ci, select

POL = load_policy()
BUD = derive_budgets(POL, available_mb=8000)          # memory budget = 2000 MB


def cand(name, ndcg_pq, lat=1.0, mem=30.0, idx=1.0, healthy=True, p95=None, qps=500.0, recall=0.6):
    pq = list(ndcg_pq)
    m = {"nDCG@10": float(np.mean(pq)), "Recall@5": recall, "MRR@10": 0.8, "Precision@5": 0.3, "MAP": 0.5,
         "latency_p50_ms": lat, "latency_p95_ms": p95 or lat * 1.3, "throughput_qps": qps, "memory_mb": mem, "indexing_seconds": idx}
    return Candidate(name, m, pq, healthy)


def noisy(mean, n, seed, sd=0.2):
    r = np.random.default_rng(seed)
    return np.clip(r.normal(mean, sd, n), 0, 1)


def test_policy_has_no_metric_weights():
    assert "weights" not in POL and "weights" not in str(POL.keys())


def test_budget_memory_is_derived_from_machine():
    assert derive_budgets(POL, 8000)["memory_mb"] == pytest.approx(2000)
    assert derive_budgets(POL, 1000)["memory_mb"] == pytest.approx(250)
    assert derive_budgets(POL)["memory_mb"] > 0


def test_quality_leader_wins_when_evidence_is_insufficient():
    base = noisy(0.65, 20, 1)
    cheap = cand("cheap", base)                                 # same data scale but n=20 => wide CI
    pricey = cand("pricey", base + 0.03, lat=40, mem=600, idx=500)
    s = select([cheap, pricey], POL, BUD)
    assert s.reference == "pricey" and s.winner == "pricey"     # cheap not *shown* non-inferior
    assert s.non_inferior["cheap"]["ci_lo"] < -POL["non_inferiority"]["margin"]


def test_cheaper_model_wins_when_demonstrably_equivalent():
    base = noisy(0.65, 400, 2)
    cheap = cand("cheap", base - 0.005)
    pricey = cand("pricey", base, lat=40, mem=600, idx=500)
    s = select([cheap, pricey], POL, BUD)
    assert s.reference == "pricey" and s.non_inferior["cheap"]["non_inferior"] and s.winner == "cheap"


def test_materially_worse_cheap_model_loses_even_with_lots_of_data():
    base = noisy(0.7, 400, 3)
    s = select([cand("cheap", base - 0.10), cand("pricey", base, lat=40, mem=600, idx=500)], POL, BUD)
    assert s.winner == "pricey" and not s.non_inferior["cheap"]["non_inferior"]


def test_constraints_make_selection_dynamic():
    base = noisy(0.7, 100, 4)
    big = cand("big", base + 0.2, lat=40, mem=900, idx=500)
    small = cand("small", base, mem=30)
    assert select([big, small], POL, derive_budgets(POL, 8000)).winner == "big"        # 900 <= 2000
    s = select([big, small], POL, derive_budgets(POL, 2000))                           # budget 500MB -> big infeasible
    assert s.winner == "small" and any("memory_mb" in r for r in s.feasible["big"])


def test_floor_health_latency_and_throughput_are_hard_constraints():
    base = noisy(0.7, 50, 5)
    ok = cand("ok", base)
    assert select([ok, cand("weak", noisy(0.3, 50, 6))], POL, BUD).feasible["weak"]
    assert "health check failed" in select([ok, cand("sick", base + .1, healthy=False)], POL, BUD).feasible["sick"]
    assert select([ok, cand("slow", base + .1, p95=5000)], POL, BUD).winner == "ok"
    assert select([ok, cand("lowqps", base + .1, qps=0.5)], POL, BUD).winner == "ok"


def test_no_feasible_candidate_means_no_winner():
    s = select([cand("a", noisy(0.2, 30, 7))], POL, BUD)
    assert s.winner is None and "production unchanged" in s.trace[-1]


def test_footprint_is_geometric_mean_of_ratios_and_pareto():
    base = noisy(0.7, 30, 8)
    a, b = cand("a", base, lat=1, mem=10, idx=1), cand("b", base + .05, lat=4, mem=40, idx=16)
    s = select([a, b], POL, BUD)
    assert s.footprint["a"] == pytest.approx(1.0) and s.footprint["b"] == pytest.approx((4 * 4 * 16) ** (1 / 3))
    assert set(s.pareto) == {"a", "b"}                                    # a cheaper, b better
    dominated = cand("c", base - .1, lat=5, mem=50, idx=20)
    assert "c" not in select([a, b, dominated], POL, BUD).pareto


def test_paired_ci_brackets_true_difference():
    r = np.random.default_rng(0); x = r.normal(0.6, 0.2, 300)
    m, lo, hi = paired_diff_ci(x + 0.05, x)
    assert m == pytest.approx(0.05) and lo == pytest.approx(0.05) and hi == pytest.approx(0.05)
    y = x + r.normal(0.05, 0.1, 300)
    m2, lo2, hi2 = paired_diff_ci(y, x)
    assert lo2 < 0.05 < hi2 and hi2 - lo2 > 0.01
