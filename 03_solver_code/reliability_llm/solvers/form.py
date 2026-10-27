"""FORM solver — thin wrapper over vendored ERA FORM_HLRF.

Uses HL-RF (Hasofer-Lind / Rackwitz-Fiessler) to find the design point u* in
standard normal space, then computes reliability index β = ||u*|| and
Pf = Φ(-β). Gradient is computed via forward finite differences internally
by ERA when no analytical dg is provided.
"""

from __future__ import annotations

import contextlib
import io
import time

from reliability_llm.solvers._era import FORM_HLRF
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def solve_form(
    distr: JointDistribution,
    lsf: LSF,
    u0: float = 0.1,
    tol: float = 1e-6,
    maxit: int = 500,
    suppress_stdout: bool = True,
) -> ReliabilityResult:
    """Run FORM HL-RF on the given joint distribution and limit-state function."""
    start = time.perf_counter()

    buf = io.StringIO() if suppress_stdout else None
    with contextlib.redirect_stdout(buf) if suppress_stdout else contextlib.nullcontext():
        u_star, x_star, beta, alpha, pf = FORM_HLRF(
            g=lsf,
            dg=[],  # empty → ERA falls back to forward finite differences
            distr=distr.nataf,
            u0=u0,
            tol=tol,
            maxit=maxit,
        )
    elapsed = time.perf_counter() - start

    return ReliabilityResult(
        method="FORM",
        beta=float(beta),
        pf=float(pf),
        elapsed=elapsed,
        convergence=[],  # FORM traces not captured (per user, 2026-04-24)
        raw={
            "u_star": u_star.tolist(),
            "x_star": x_star.tolist(),
            "alpha": alpha.tolist(),
            "stdout": buf.getvalue() if buf is not None else "",
        },
    )
