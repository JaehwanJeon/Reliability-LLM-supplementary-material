"""iCE solver — improved Cross-Entropy with Gaussian Mixture model.

Wraps ERA's iCE_GM. Compared to plain CE, iCE uses a smooth indicator
approximation parameterised by an annealing scale `sigma_t` that adapts toward
the failure region; convergence is governed by `CV_target` rather than a
percentile. The reliability-LLM patch surfaces a per-iteration Pf trace.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._era import iCE_GM
from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def solve_ice(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 1000,
    p: float = 0.1,
    max_it: int = 20,
    cv_target: float = 2.0,
    k_init: int = 3,
    seed: int = 0,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run improved Cross-Entropy with adaptive sigma_t annealing."""
    np.random.seed(seed)

    start = time.perf_counter()
    buf = io.StringIO() if suppress_stdout else None
    cm = contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext()
    with cm:
        Pr, lv, N_tot, _samplesU, _samplesX, k_fin, _W_final, _f_iid, trace_raw = iCE_GM(
            n, p, lsf, distr.nataf, max_it, cv_target, k_init, 0,
        )
    elapsed = time.perf_counter() - start

    pf_val = float(Pr)
    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    # Replace per-iteration index with cumulative LSF-eval count.
    convergence = trace_from_raw(trace_raw)
    for i, s in enumerate(convergence):
        s.step = (i + 1) * n

    return ReliabilityResult(
        method="iCE",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=convergence,
        raw={
            "n_iters": int(lv),
            "n_total": int(N_tot),
            "k_final": int(k_fin),
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
