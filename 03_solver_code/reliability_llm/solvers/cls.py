"""CLS solver — Combination Line Sampling, FORM-chained.

Runs FORM first to obtain the design point u*, then probes the failure surface
along lines parallel to the FORM-anchored direction. ERA's CLS performs the
inner per-line loop and returns a per-direction-update Pf trace (see the
reliability-LLM modification in `_era/CLS.py`).
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._era import CLS
from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.form import solve_form
from reliability_llm.solvers.types import (
    LSF,
    JointDistribution,
    ReliabilityResult,
)


def solve_cls(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 200,
    form_result: Optional[ReliabilityResult] = None,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run CLS using FORM's design point as the search direction."""
    if form_result is None:
        form_result = solve_form(distr, lsf, suppress_stdout=suppress_stdout)

    u_star = np.asarray(form_result.raw["u_star"], dtype=float)

    # CLS calls g_fun internally with shape (1, n) and expects scalar output;
    # squeeze the single-element ndarray our LSF returns into a Python float.
    def lsf_for_cls(X: np.ndarray):
        return float(np.atleast_1d(lsf(X))[0])

    start = time.perf_counter()
    buf = io.StringIO() if suppress_stdout else None
    cm = contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext()
    with cm:
        pf_ls, cv_ls, u_new, x_new, n_eval, trace_raw = CLS(
            u_star, n, lsf_for_cls, distr.nataf,
        )
    elapsed = time.perf_counter() - start + form_result.elapsed

    pf_val = float(pf_ls)
    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    # Replace step index with cumulative LSF-eval count from the line search.
    convergence = trace_from_raw(trace_raw)
    if convergence:
        convergence[-1].step = int(n_eval)

    return ReliabilityResult(
        method="CLS",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=convergence,
        raw={
            "cv": float(cv_ls),
            "n_eval": int(n_eval),
            "u_new": np.asarray(u_new).tolist(),
            "form_beta": form_result.beta,
            "form_pf": form_result.pf,
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
