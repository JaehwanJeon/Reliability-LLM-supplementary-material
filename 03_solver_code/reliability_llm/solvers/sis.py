"""SIS solver — Sequential Importance Sampling with adaptive Conditional Sampling.

Wraps ERA's SIS_aCS. SIS anneals an indicator-approximation parameter
`sigma` toward the true failure indicator while resampling/MCMC-walking
between levels. The reliability-LLM patch surfaces a per-level Pf trace
computed as `prod(Sk[:m+1]) * mean(I*W)` at each level m.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._era import SIS_aCS
from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def solve_sis(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 1000,
    p: float = 0.1,
    burn: int = 0,
    tar_cov: float = 1.5,
    seed: int = 0,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run Sequential Importance Sampling with adaptive Conditional Sampling."""
    np.random.seed(seed)

    # SIS_aCS calls g(u) with BOTH single points (1, n) for MCMC and batched
    # (N, n) for the initial level evaluation. Squeeze ONLY when output is
    # a single-element ndarray; pass arrays through unchanged for batched case.
    def lsf_for_sis(X: np.ndarray):
        g = lsf(X)
        if isinstance(g, np.ndarray) and g.size == 1:
            return float(g.item())
        return g

    start = time.perf_counter()
    buf = io.StringIO() if suppress_stdout else None
    cm = contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext()
    with cm:
        Pr, l_tot, _samplesU, _samplesX, _W_final, _f_iid, trace_raw = SIS_aCS(
            n, p, lsf_for_sis, distr.nataf, burn, tar_cov, 0,
        )
    elapsed = time.perf_counter() - start

    pf_val = float(Pr)
    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    # Replace per-level index with cumulative LSF-eval count (initial MC + each
    # iteration draws N samples; approximate cumulative at level i as (i+1)*n).
    convergence = trace_from_raw(trace_raw)
    for i, s in enumerate(convergence):
        s.step = (i + 1) * n

    return ReliabilityResult(
        method="SIS",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=convergence,
        raw={
            "n_levels": int(l_tot),
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
