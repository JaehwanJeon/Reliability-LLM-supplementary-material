"""Internal helpers for converting ERA-side trace dicts into ConvergenceStep.

Each method's ERA-side patch (or our self-built MCS/IS/SORM) appends raw dicts
of the form `{"step": int, "pf": float, "extra": dict}` to a list. Wrappers
call `trace_from_raw` once to convert to a list of `ConvergenceStep` objects
for inclusion in `ReliabilityResult.convergence`.
"""

from __future__ import annotations

from reliability_llm.solvers.types import ConvergenceStep


def trace_from_raw(raw_list: list[dict]) -> list[ConvergenceStep]:
    return [
        ConvergenceStep(
            step=int(t["step"]),
            pf=float(t["pf"]),
            extra=dict(t.get("extra", {})),
        )
        for t in raw_list
    ]
