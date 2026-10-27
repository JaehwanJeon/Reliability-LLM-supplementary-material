"""CE solver — Cross-Entropy importance sampling with Gaussian Mixture model.

Wraps ERA's CEIS_GM. ERA-side reliability-LLM patch surfaces a per-iteration
Pf trace (estimated from each iteration's GMM samples via the same importance
sampling estimator ERA uses for the final Pf).
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._era import CEIS_GM
from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def solve_ce(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 1000,
    p: float = 0.1,
    k_init: int = 3,
    seed: int = 0,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run Cross-Entropy importance sampling with Gaussian Mixture density."""
    np.random.seed(seed)

    # CEIS_GM calls G_LSF(u) -> g_fun(u2x(u)); inputs are already (N, n) batches
    # so no scalar squeeze needed (unlike SuS/CLS which call point-by-point).
    start = time.perf_counter()
    buf = io.StringIO() if suppress_stdout else None
    cm = contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext()
    with cm:
        Pr, lv, N_tot, gamma_hat, _samplesU, _samplesX, k_fin, _W_final, _f_iid, trace_raw = CEIS_GM(
            n, p, lsf, distr.nataf, k_init, 0,
        )
    elapsed = time.perf_counter() - start

    pf_val = float(Pr)
    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    # Replace per-iteration index with cumulative LSF-eval count (each CE
    # iteration evaluates N samples, so cumulative at iter i is (i+1)*n).
    convergence = trace_from_raw(trace_raw)
    for i, s in enumerate(convergence):
        s.step = (i + 1) * n

    return ReliabilityResult(
        method="CE",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=convergence,
        raw={
            "n_iters": int(lv),
            "n_total": int(N_tot),
            "k_final": int(k_fin),
            "gamma_hat": np.asarray(gamma_hat).tolist(),
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
