"""SuS solver — Subset Simulation, thin wrapper over vendored ERA SuS.

ERA's SuS exposes per-level threshold (`b`) and a naive per-level Pf schedule
(`Pf`). The schedule is exact for intermediate levels (Pf[i] = p0^(i+1)) but
the FINAL level's true probability is `Pf_SuS` rather than `p0^m`, so the
trace replaces the last entry accordingly.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._era import SuS
from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def solve_sus(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 1000,
    p0: float = 0.1,
    seed: int = 0,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run Subset Simulation."""
    np.random.seed(seed)  # ERA samplers use np.random globally

    # ERA's SuS calls g_fun via aCS internally with shape (1, n) per evaluation;
    # squeeze the single-element ndarray our LSF returns into a Python float.
    def lsf_for_sus(X: np.ndarray):
        return float(np.atleast_1d(lsf(X))[0])

    start = time.perf_counter()
    buf = io.StringIO() if suppress_stdout else None
    cm = contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext()
    with cm:
        (
            Pf_SuS,
            delta_SuS,
            b,
            Pf,
            b_line,
            Pf_line,
            _samplesU,
            _samplesX,
            _f_s_iid,
        ) = SuS(n, p0, lsf_for_sus, distr.nataf, 0)
    elapsed = time.perf_counter() - start

    pf_val = float(Pf_SuS)
    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    # Per-level trace. Intermediate levels follow Pf[i] = p0^(i+1) by construction;
    # the last level uses the actual final estimate Pf_SuS instead of p0^m.
    # `step` is cumulative LSF-eval count: level 0 uses N initial MC samples;
    # levels k>=1 reuse p0*N seeds and resample (1-p0)*N via MCMC.
    m = int(len(b))
    raw_trace: list[dict] = []
    for i in range(max(m - 1, 0)):
        raw_trace.append({
            "step": int(n + i * (1.0 - p0) * n),
            "pf": float(Pf[i]),
            "extra": {"threshold": float(b[i])},
        })
    if m > 0:
        raw_trace.append({
            "step": int(n + (m - 1) * (1.0 - p0) * n),
            "pf": pf_val,
            "extra": {"threshold": float(b[-1])},
        })

    return ReliabilityResult(
        method="SuS",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=trace_from_raw(raw_trace),
        raw={
            "delta": float(delta_SuS),
            "n_levels": m,
            "b_line": np.asarray(b_line).tolist(),
            "Pf_line": np.asarray(Pf_line).tolist(),
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
