"""MCS solver — Crude Monte Carlo Simulation.

Self-contained implementation: sample from the joint distribution via ERANataf,
evaluate the LSF in batch, count failures. CoV is `sqrt((1-Pf) / (N*Pf))`.

Two modes:
- Fixed budget (target_cov=None): draw exactly `n` samples.
- Adaptive (target_cov set): start with `n_start`, double until estimated
  CoV drops to target_cov or cumulative N hits n_max.

The convergence trace records running Pf — log-spaced for fixed mode, one
point per batch for adaptive mode.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def _trace_points(n: int, num: int = 12) -> np.ndarray:
    """Log-spaced sample counts for the running-Pf trace, ending at n."""
    if n <= 1:
        return np.array([n], dtype=int)
    pts = np.unique(np.geomspace(min(10, n), n, num).astype(int))
    pts = pts[pts <= n]
    if pts.size == 0 or pts[-1] != n:
        pts = np.append(pts, n)
    return pts


def _cov_from(pf: float, n: int) -> float:
    if 0.0 < pf < 1.0 and n > 0:
        return float(np.sqrt((1.0 - pf) / (n * pf)))
    return float("inf")


def solve_mcs(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 10000,
    seed: int = 0,
    target_cov: float | None = None,
    n_start: int = 10_000,
    n_max: int = 10_000_000,
) -> ReliabilityResult:
    """Run crude Monte Carlo Simulation.

    When ``target_cov`` is None, draw exactly ``n`` samples (legacy behaviour).
    When set, sample in doubling batches starting from ``n_start`` until the
    estimated CoV is ``<= target_cov`` or cumulative samples reach ``n_max``.
    """
    if target_cov is not None:
        return _solve_mcs_adaptive(distr, lsf, target_cov, n_start, n_max, seed)
    return _solve_mcs_fixed(distr, lsf, n, seed)


def _solve_mcs_fixed(
    distr: JointDistribution,
    lsf: LSF,
    n: int,
    seed: int,
) -> ReliabilityResult:
    np.random.seed(seed)
    start = time.perf_counter()

    # Sample N points (shape (N, dim) per ERA convention)
    X = distr.nataf.random(n)
    # Vectorised LSF evaluation
    g = lsf(X)
    fail = (np.asarray(g) <= 0).astype(float)

    # Build running-Pf trace at log-spaced sample counts
    points = _trace_points(n)
    raw_trace = []
    for m in points:
        pf_m = float(np.mean(fail[:m]))
        raw_trace.append({
            "step": int(m),
            "pf": pf_m,
            "extra": {"cov": _cov_from(pf_m, int(m))},
        })

    elapsed = time.perf_counter() - start
    pf_val = float(np.mean(fail))
    n_fail = int(np.sum(fail))

    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    return ReliabilityResult(
        method="MCS",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=trace_from_raw(raw_trace),
        raw={
            "n_total": int(n),
            "n_failures": n_fail,
            "cov": _cov_from(pf_val, n),
        },
    )


def _solve_mcs_adaptive(
    distr: JointDistribution,
    lsf: LSF,
    target_cov: float,
    n_start: int,
    n_max: int,
    seed: int,
) -> ReliabilityResult:
    if target_cov <= 0.0:
        raise ValueError(f"target_cov must be positive, got {target_cov}")
    if n_start < 1 or n_max < n_start:
        raise ValueError(
            f"need 1 <= n_start <= n_max, got n_start={n_start}, n_max={n_max}"
        )

    np.random.seed(seed)
    start = time.perf_counter()

    n_total = 0
    n_fail_total = 0
    batch = n_start
    raw_trace: list[dict] = []
    reached = False

    while n_total < n_max:
        take = min(batch, n_max - n_total)
        X = distr.nataf.random(take)
        g = lsf(X)
        n_fail_total += int(np.sum(np.asarray(g) <= 0))
        n_total += take

        pf_m = n_fail_total / n_total
        cov_m = _cov_from(pf_m, n_total)
        raw_trace.append({
            "step": int(n_total),
            "pf": float(pf_m),
            "extra": {"cov": float(cov_m)},
        })

        if 0.0 < pf_m < 1.0 and cov_m <= target_cov:
            reached = True
            break

        batch *= 2

    if not reached:
        warnings.warn(
            f"MCS target_cov={target_cov} not reached after n={n_total} "
            f"(Pf={n_fail_total / n_total:.3e}, "
            f"CoV={_cov_from(n_fail_total / n_total, n_total):.3f})",
            RuntimeWarning,
            stacklevel=3,
        )

    elapsed = time.perf_counter() - start
    pf_val = n_fail_total / n_total

    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    return ReliabilityResult(
        method="MCS",
        beta=beta,
        pf=float(pf_val),
        elapsed=elapsed,
        convergence=trace_from_raw(raw_trace),
        raw={
            "n_total": int(n_total),
            "n_failures": int(n_fail_total),
            "cov": _cov_from(pf_val, n_total),
            "target_cov": float(target_cov),
            "target_cov_reached": bool(reached),
            "n_max": int(n_max),
            "n_start": int(n_start),
        },
    )
