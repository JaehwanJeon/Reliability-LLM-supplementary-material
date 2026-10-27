"""SORM — Second-Order Reliability Method (Curvature Fitting).

Algorithm — Curvature Fitting at the FORM design point u* in U-space:
1. Run FORM → β, u*, α
2. Compute ∇G(u*) and ∇²G(u*) via central FFD
3. Householder reflection R aligns α with e_n (last axis)
4. Rotated reduced Hessian A_red = R H Rᵀ / ‖∇G‖, take its (n−1)×(n−1) block
5. Principal curvatures κ = eigenvalues(A_red)
6. Breitung:        Pf ≈ Φ(−β) · ∏ 1/√(1 + β·κ_i)
   Breitung-HR:     Pf ≈ Φ(−β) · ∏ 1/√(1 + ψ·κ_i),  ψ = φ(β)/Φ(−β)
   Tvedt three-term: Pf ≈ A1 + A2 + A3 (Tvedt 1990; complex-arithmetic safe)

Reference math: Melchers "Structural Reliability" Ch.3; Breitung 1984;
Tvedt 1990 (paper, not source).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from reliability_llm.solvers.form import solve_form
from reliability_llm.solvers.types import LSF, JointDistribution, ReliabilityResult


def _u_space_lsf(distr: JointDistribution, lsf: LSF):
    """Return G(u) = lsf(Nataf.U2X(u)), with scalar output for any u shape."""
    def G(u: np.ndarray) -> float:
        u_2d = np.atleast_2d(u)
        x = distr.nataf.U2X(u_2d)
        g = lsf(x)
        return float(np.atleast_1d(g)[0])
    return G


def _ffd_gradient(G, u: np.ndarray, h: float) -> np.ndarray:
    """Central-difference gradient of G at u."""
    n = u.size
    grad = np.zeros(n)
    for i in range(n):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += h
        u_minus[i] -= h
        grad[i] = (G(u_plus) - G(u_minus)) / (2.0 * h)
    return grad


def _ffd_hessian(G, u: np.ndarray, h: float) -> np.ndarray:
    """Central-difference Hessian of G at u (n×n symmetric)."""
    n = u.size
    H = np.zeros((n, n))
    g0 = G(u)

    # Diagonal: central differences
    for i in range(n):
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i] += h
        u_minus[i] -= h
        H[i, i] = (G(u_plus) - 2.0 * g0 + G(u_minus)) / (h * h)

    # Off-diagonal: 4-point central differences
    for i in range(n):
        for j in range(i + 1, n):
            u_pp = u.copy()
            u_pm = u.copy()
            u_mp = u.copy()
            u_mm = u.copy()
            u_pp[i] += h; u_pp[j] += h
            u_pm[i] += h; u_pm[j] -= h
            u_mp[i] -= h; u_mp[j] += h
            u_mm[i] -= h; u_mm[j] -= h
            val = (G(u_pp) - G(u_pm) - G(u_mp) + G(u_mm)) / (4.0 * h * h)
            H[i, j] = val
            H[j, i] = val

    return H


def _householder_rotation(alpha: np.ndarray) -> np.ndarray:
    """Build orthogonal R such that R @ alpha = e_n (last std basis vector).

    Uses a single Householder reflection. If alpha is already aligned with e_n
    (or its negative), returns identity (or a sign flip).
    """
    n = alpha.size
    e_n = np.zeros(n)
    e_n[-1] = 1.0

    v = alpha - e_n
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.eye(n)
    v = v / nv
    return np.eye(n) - 2.0 * np.outer(v, v)


def _breitung(beta: float, kappa: np.ndarray) -> float:
    """Breitung's asymptotic formula. Uses complex arithmetic for robustness;
    when 1 + β·κ_i < 0 for some i the formula is technically invalid and the
    result here may be unphysical — that's a real signal for the planner."""
    factors = 1.0 / np.sqrt(1.0 + beta * kappa + 0j)
    pf = float(sp_stats.norm.cdf(-beta)) * float(np.prod(factors).real)
    return pf


def _breitung_hr(beta: float, kappa: np.ndarray) -> float:
    """Hohenbichler-Rackwitz refinement of Breitung: replace β with ψ = φ(β)/Φ(−β).
    Same formula otherwise."""
    psi = float(sp_stats.norm.pdf(beta) / sp_stats.norm.cdf(-beta))
    factors = 1.0 / np.sqrt(1.0 + psi * kappa + 0j)
    pf = float(sp_stats.norm.cdf(-beta)) * float(np.prod(factors).real)
    return pf


def _tvedt_three_term(beta: float, kappa: np.ndarray) -> float:
    """Tvedt's three-term approximation. More accurate than Breitung when
    β is small or curvatures are large; uses complex arithmetic in the third
    term to handle 1 + (β + i)·κ_i robustly."""
    Phi_mb = float(sp_stats.norm.cdf(-beta))
    phi_b = float(sp_stats.norm.pdf(beta))
    prod_b = float(np.prod(1.0 / np.sqrt(1.0 + beta * kappa + 0j)).real)
    prod_b1 = float(np.prod(1.0 / np.sqrt(1.0 + (beta + 1.0) * kappa + 0j)).real)
    prod_bi = float(np.prod(1.0 / np.sqrt(1.0 + (beta + 1j) * kappa)).real)
    A1 = Phi_mb * prod_b
    A2 = (beta * Phi_mb - phi_b) * (prod_b - prod_b1)
    A3 = (beta + 1.0) * (beta * Phi_mb - phi_b) * (prod_b - prod_bi)
    return float(A1 + A2 + A3)


def _pf_to_beta(pf: float) -> float:
    if pf <= 0.0:
        return float("inf")
    if pf >= 1.0:
        return float("-inf")
    return float(-sp_stats.norm.ppf(pf))


def solve_sorm(
    distr: JointDistribution,
    lsf: LSF,
    form_result: Optional[ReliabilityResult] = None,
    h: float = 5e-4,
) -> ReliabilityResult:
    """Run SORM (curvature fitting) on top of a FORM design point.

    Reports `beta`/`pf` from Breitung's formula by default; alternative
    estimates (Breitung-HR, Tvedt three-term, FORM) are stored in `raw`.
    """
    if form_result is None:
        form_result = solve_form(distr, lsf)

    u_star = np.asarray(form_result.raw["u_star"], dtype=float)
    alpha = np.asarray(form_result.raw["alpha"], dtype=float)
    beta = float(form_result.beta)

    G = _u_space_lsf(distr, lsf)

    start = time.perf_counter()
    grad = _ffd_gradient(G, u_star, h)
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-12:
        # Degenerate: gradient vanished at design point; cannot rotate
        elapsed = time.perf_counter() - start + form_result.elapsed
        return ReliabilityResult(
            method="SORM",
            beta=beta,
            pf=form_result.pf,
            elapsed=elapsed,
            convergence=[],
            raw={
                "form_beta": beta, "form_pf": form_result.pf,
                "warning": "zero gradient at design point — SORM falls back to FORM",
            },
        )

    H = _ffd_hessian(G, u_star, h)
    R = _householder_rotation(alpha)
    A = R @ H @ R.T / grad_norm
    A_reduced = A[:-1, :-1]
    curvatures = np.linalg.eigvalsh(0.5 * (A_reduced + A_reduced.T))  # symmetrize

    pf_breitung = _breitung(beta, curvatures)
    pf_breitung_hr = _breitung_hr(beta, curvatures)
    pf_tvedt = _tvedt_three_term(beta, curvatures)
    elapsed = time.perf_counter() - start + form_result.elapsed

    pf_primary = pf_breitung
    beta_primary = _pf_to_beta(pf_primary)

    return ReliabilityResult(
        method="SORM",
        beta=beta_primary,
        pf=pf_primary,
        elapsed=elapsed,
        convergence=[],  # SORM is single-point; no iteration trace
        raw={
            "curvatures": curvatures.tolist(),
            "grad_norm": grad_norm,
            "h_step": h,
            "form_beta": beta,
            "form_pf": form_result.pf,
            "pf_breitung": pf_breitung,
            "pf_breitung_hr": pf_breitung_hr,
            "pf_tvedt": pf_tvedt,
            "beta_breitung": _pf_to_beta(pf_breitung),
            "beta_breitung_hr": _pf_to_beta(pf_breitung_hr),
            "beta_tvedt": _pf_to_beta(pf_tvedt),
        },
    )
