"""Unified types for the solvers module.

All `solve_*` functions take a `JointDistribution` and an `LSF`, and return a
`ReliabilityResult`. Convergence trace is a list of `ConvergenceStep` entries
with a stable schema across methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


LSF = Callable[[np.ndarray], np.ndarray]
"""Limit-state function signature.

Takes an array of shape (n_rvs,) for a single evaluation or (n_rvs, n_samples)
for batched evaluation, and returns either a scalar or shape (n_samples,).

By convention, g(X) < 0 is the failure region.
"""


@dataclass
class ConvergenceStep:
    """One point on the convergence trace.

    `step` is the iteration / sample count / subset level index (method-specific).
    `pf` is the current Pf estimate. `extra` holds method-specific diagnostics
    (e.g. threshold `b` for SuS, GMM parameter count for CE, CoV for MCS).
    """

    step: int
    pf: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ReliabilityResult:
    """Uniform return type for every `solve_*` function."""

    method: str
    beta: Optional[float] = None
    pf: Optional[float] = None
    elapsed: float = 0.0
    convergence: list[ConvergenceStep] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class JointDistribution:
    """Joint distribution over marginals with an optional correlation matrix.

    This is a thin wrapper around ERA's `ERANataf` construction. Solvers use
    `self.nataf` to sample, transform U↔X, and compute Jacobians. Build
    instances via `spec_converter.build_joint_distribution(spec)` rather than
    directly.
    """

    def __init__(self, nataf, variable_names: list[str]):
        self.nataf = nataf
        self.variable_names = list(variable_names)

    @property
    def n(self) -> int:
        return len(self.variable_names)

    def __repr__(self) -> str:
        return f"JointDistribution(n={self.n}, vars={self.variable_names})"
