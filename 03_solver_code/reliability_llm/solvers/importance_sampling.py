"""IS solver — FORM-anchored Importance Sampling (textbook variant).

Self-contained clean-room implementation following Melchers, "Structural
Reliability Analysis" Ch 3:
- Run FORM to obtain the design point u* in standard normal space
- Sample u_i ~ N(u*, I_d) for i = 1..N (proposal centred on u*)
- Transform to physical space via Nataf and evaluate g(x_i)
- Likelihood ratio: w_i = phi(u_i) / phi(u_i - u*) = exp(-u_i·u* + 0.5||u*||^2)
- Estimator: Pf ≈ (1/N) sum(I[g_i <= 0] * w_i)

No FERUM/Pystra source consulted — math from the textbook only.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers._trace import trace_from_raw
from reliability_llm.solvers.form import solve_form
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def _trace_points(n: int, num: int = 12) -> np.ndarray:
    if n <= 1:
        return np.array([n], dtype=int)
    pts = np.unique(np.geomspace(min(10, n), n, num).astype(int))
    pts = pts[pts <= n]
    if pts.size == 0 or pts[-1] != n:
        pts = np.append(pts, n)
    return pts


def solve_is(
    distr: JointDistribution,
    lsf: LSF,
    n: int = 10000,
    form_result: Optional[ReliabilityResult] = None,
    seed: int = 0,
) -> ReliabilityResult:
    """Run FORM-anchored Importance Sampling."""
    if form_result is None:
        form_result = solve_form(distr, lsf)

    u_star = np.asarray(form_result.raw["u_star"], dtype=float)
    u_star_norm_sq = float(u_star @ u_star)
    dim = u_star.size

    np.random.seed(seed)
    start = time.perf_counter()

    # Sample U from N(u*, I_d). Shape (N, dim).
    u = u_star[None, :] + np.random.randn(n, dim)

    # Transform to physical space and evaluate the LSF
    x = distr.nataf.U2X(u)
    g = np.asarray(lsf(x))

    # Likelihood ratio  w = exp(-u·u* + 0.5||u*||^2)
    log_w = -u @ u_star + 0.5 * u_star_norm_sq
    w = np.exp(log_w)

    fail_indicator = (g <= 0).astype(float)
    contribs = fail_indicator * w

    # Running-Pf trace at log-spaced sample counts
    points = _trace_points(n)
    raw_trace = []
    for idx, m in enumerate(points):
        sub = contribs[:m]
        pf_m = float(np.mean(sub))
        # MC variance estimate of weighted indicator mean
        if m > 1:
            var_m = float(np.var(sub, ddof=1) / m)
            cov_m = float(np.sqrt(var_m) / pf_m) if pf_m > 0 else float("inf")
        else:
            cov_m = float("inf")
        raw_trace.append({
            "step": int(m),
            "pf": pf_m,
            "extra": {"cov": cov_m},
        })

    elapsed = time.perf_counter() - start + form_result.elapsed
    pf_val = float(np.mean(contribs))
    n_fail = int(np.sum(fail_indicator))

    if pf_val <= 0.0:
        beta = float("inf")
    elif pf_val >= 1.0:
        beta = float("-inf")
    else:
        beta = float(-sp_stats.norm.ppf(pf_val))

    if pf_val > 0.0 and n > 1:
        var_final = float(np.var(contribs, ddof=1) / n)
        cov_final = float(np.sqrt(var_final) / pf_val)
    else:
        cov_final = float("inf")

    return ReliabilityResult(
        method="IS",
        beta=beta,
        pf=pf_val,
        elapsed=elapsed,
        convergence=trace_from_raw(raw_trace),
        raw={
            "n_total": int(n),
            "n_fail_raw": n_fail,
            "u_star": u_star.tolist(),
            "u_star_norm": float(np.sqrt(u_star_norm_sq)),
            "cov": cov_final,
            "form_beta": form_result.beta,
            "form_pf": form_result.pf,
        },
    )
