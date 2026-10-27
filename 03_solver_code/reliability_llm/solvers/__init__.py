"""Unified pure-Python reliability solvers (TF migration, 2026-04-24).

Goal: self-contained, `pip install`-friendly module replacing the subprocess /
matlab.engine path in `reliability_llm/backends/`. All methods expose the same
signature:

    solve_<method>(distr: JointDistribution, lsf: LSF, **opts) -> ReliabilityResult

Status: EXPERIMENTAL. See memory file `pure_python_migration.md`. The
`backends/` stack is preserved untouched as rollback.

License: MIT for our code. Vendored ERA sources under `_era/` retain their
upstream MIT notice (see `_era/LICENSE.txt`).
"""

from reliability_llm.solvers.ce import solve_ce
from reliability_llm.solvers.cls import solve_cls
from reliability_llm.solvers.form import solve_form
from reliability_llm.solvers.ice import solve_ice
from reliability_llm.solvers.importance_sampling import solve_is
from reliability_llm.solvers.mcs import solve_mcs
from reliability_llm.solvers.sis import solve_sis
from reliability_llm.solvers.sorm import solve_sorm
from reliability_llm.solvers.spec_converter import (
    build_joint_distribution,
    build_lsf,
)
from reliability_llm.solvers.sus import solve_sus
from reliability_llm.solvers.types import (
    LSF,
    ConvergenceStep,
    JointDistribution,
    ReliabilityResult,
)

__all__ = [
    "LSF",
    "ConvergenceStep",
    "JointDistribution",
    "ReliabilityResult",
    "build_joint_distribution",
    "build_lsf",
    "solve_form",
    "solve_sorm",
    "solve_cls",
    "solve_sus",
    "solve_ce",
    "solve_ice",
    "solve_sis",
    "solve_mcs",
    "solve_is",
]
