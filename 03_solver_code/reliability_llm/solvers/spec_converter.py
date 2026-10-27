"""Convert a validated ProblemSpec into solver inputs.

Two public functions:
- `build_joint_distribution(spec)` → `JointDistribution` backed by ERA's ERANataf
- `build_lsf(spec)` → Python callable that evaluates g(X) with numpy broadcasting

The distribution name mapping below was extracted from the vendored ERADist.py
(see `_era/ERADist.py` branches under `opt.upper() == 'MOM'`).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from reliability_llm.schemas.types import (
    Distribution,
    ProblemSpec,
    RandomVariable,
)
from reliability_llm.solvers._era import ERADist, ERANataf
from reliability_llm.solvers.types import LSF, JointDistribution


# Standard distributions: (ERA name, how many of [mean, std] to pass).
# BETA and TYPE_III_SMALLEST need extra_params and are handled as early
# returns in _rv_to_eradist.
_STANDARD: dict[Distribution, tuple[str, int]] = {
    Distribution.NORMAL:          ("normal",      2),
    Distribution.LOGNORMAL:       ("lognormal",   2),
    Distribution.GUMBEL:          ("gumbel",      2),
    Distribution.WEIBULL:         ("weibull",     2),
    Distribution.UNIFORM:         ("uniform",     2),
    Distribution.GAMMA:           ("gamma",       2),
    Distribution.TYPE_I_SMALLEST: ("gumbelmin",   2),
    Distribution.FRECHET:         ("frechet",     2),
    Distribution.EXPONENTIAL:     ("exponential", 1),  # rate = 1/mean
    Distribution.RAYLEIGH:        ("rayleigh",    1),  # alpha derived from mean
    Distribution.CHI_SQUARE:      ("chisquare",   1),  # k = mean
}


def _rv_to_eradist(rv: RandomVariable) -> ERADist:
    """Convert one `RandomVariable` to an `ERADist` instance."""
    if rv.distribution is Distribution.TYPE_III_SMALLEST:
        # ERA has no 3-parameter Weibull. Build a 2-param Weibull for the
        # unshifted part (Y = X - epsilon) then swap self.Dist for the shifted
        # scipy distribution. ERADist's public methods delegate to self.Dist.
        eps = rv.extra_params.get("epsilon")
        if eps is None:
            raise KeyError(f"TYPE_III_SMALLEST {rv.name!r} needs extra_params['epsilon'].")
        era = ERADist("weibull", "MOM", [rv.mean - eps, rv.std])
        era.Dist = stats.weibull_min(c=era.Par["k"], scale=era.Par["a_n"], loc=eps)
        era.Par["epsilon"] = eps
        return era

    if rv.distribution is Distribution.BETA:
        if "support_lower" not in rv.extra_params or "support_upper" not in rv.extra_params:
            raise KeyError(
                f"Beta {rv.name!r} needs extra_params 'support_lower' and "
                f"'support_upper' (support bounds)."
            )
        return ERADist("beta", "MOM", [rv.mean, rv.std,
                                       rv.extra_params["support_lower"],
                                       rv.extra_params["support_upper"]])

    if rv.distribution not in _STANDARD:
        raise ValueError(f"Distribution {rv.distribution.value!r} is not supported.")
    name, arity = _STANDARD[rv.distribution]
    return ERADist(name, "MOM", [rv.mean, rv.std][:arity])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_joint_distribution(spec: ProblemSpec) -> JointDistribution:
    """Convert a ProblemSpec into a JointDistribution (ERANataf under the hood)."""
    marginals = [_rv_to_eradist(rv) for rv in spec.random_variables]
    n = len(marginals)

    if spec.correlation.matrix:
        corr = np.array(spec.correlation.matrix, dtype=float)
        if corr.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {corr.shape} does not match number "
                f"of random variables {n}."
            )
    else:
        corr = np.eye(n)

    nataf = ERANataf(marginals, corr)
    return JointDistribution(
        nataf=nataf,
        variable_names=[rv.name for rv in spec.random_variables],
    )


_SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    "min": np.minimum,
    "max": np.maximum,
}


def build_lsf(spec: ProblemSpec) -> LSF:
    """Compile `spec.limit_state_expr` into a vectorized Python callable.

    Follows the ERA convention: X is shape (n_samples, n_rvs) for batched
    evaluation, shape (n_rvs,) for a single sample. Returns shape (n_samples,)
    or a scalar. By convention, g(X) < 0 means failure.
    """
    names = [rv.name for rv in spec.random_variables]
    expr = spec.limit_state_expr
    try:
        code = compile(expr, f"<lsf:{id(spec)}>", "eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid limit-state expression {expr!r}: {exc}"
        ) from exc

    def lsf(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            # single sample: X is shape (n_rvs,)
            local_ns = {name: X[i] for i, name in enumerate(names)}
        elif X.ndim == 2:
            # batch: X is shape (n_samples, n_rvs), each column is one variable
            local_ns = {name: X[:, i] for i, name in enumerate(names)}
        else:
            raise ValueError(
                f"LSF input must be 1-D or 2-D, got shape {X.shape}."
            )
        try:
            return eval(code, _SAFE_GLOBALS, local_ns)
        except NameError as exc:
            raise NameError(
                f"Undefined name in LSF {expr!r}: {exc}. "
                f"Available variables: {names}."
            ) from exc

    return lsf
