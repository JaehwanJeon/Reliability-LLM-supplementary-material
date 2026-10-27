"""Pydantic schemas for reliability LLM.

Every agent I/O is typed via these models. This replaces the scattered
field_status / field_issues / validation_message dicts in the old code.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class Distribution(str, Enum):
    """Supported marginal distributions.

    solvers/spec_converter.py maps these to ERADist distribution names and
    builds each marginal with ERADist's MOM initialization.

    Distributions needing extra parameters beyond (mean, std):
    - BETA: support bounds support_lower, support_upper (stored in RandomVariable.extra_params)
    - TYPE_III_SMALLEST: shift epsilon (stored in RandomVariable.extra_params)
    """

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    GUMBEL = "gumbel"                        # = Type I Largest Value
    WEIBULL = "weibull"                      # = Type III Smallest w/ epsilon=0
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"              # one-parameter; rate = 1/mean
    GAMMA = "gamma"
    BETA = "beta"                            # requires support_lower, support_upper in extra_params
    RAYLEIGH = "rayleigh"                    # one-parameter; derived from mean
    CHI_SQUARE = "chi_square"
    TYPE_I_SMALLEST = "type_i_smallest"      # = Type I Smallest Value (ERADist "gumbelmin")
    FRECHET = "frechet"                      # = Type II Largest Value
    TYPE_III_SMALLEST = "type_iii_smallest"  # requires epsilon in extra_params


class RandomVariable(BaseModel):
    """A single random variable in the reliability problem.

    mean/std are the required stats for most distributions. The solver layer
    converts these to native parameters (lambda/zeta for lognormal, shape/scale
    for weibull, etc.) via ERADist's MOM initialization.

    A few distributions need parameters that cannot be derived from (mean, std)
    alone; those go in `extra_params` as a flat dict:

    - BETA: {"support_lower": float, "support_upper": float} — support bounds
    - TYPE_III_SMALLEST: {"epsilon": float}    — shift parameter

    All other distributions should leave `extra_params` empty (default).
    """

    name: str = Field(description="Variable name as used in the LSF, e.g. 'R'.")
    distribution: Distribution
    mean: float
    std: float
    description: Optional[str] = Field(
        default=None,
        description="Physical meaning, e.g. 'yield strength (MPa)'.",
    )
    extra_params: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Extra distribution parameters that cannot be derived from "
            "(mean, std). Beta needs {'support_lower':..., 'support_upper':...} "
            "(the distribution's support bounds, NOT shape parameters); "
            "TYPE_III_SMALLEST needs {'epsilon':...}. Empty for all other "
            "distributions."
        ),
    )

    @field_validator("std")
    @classmethod
    def std_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("std must be positive")
        return v

    @field_validator("name")
    @classmethod
    def name_valid(cls, v: str) -> str:
        v = v.strip()
        if not v or not v.replace("_", "").isalnum():
            raise ValueError(f"Invalid variable name: {v!r}")
        return v


class CorrelationMatrix(BaseModel):
    """Correlation matrix in the order of random_variables list."""

    matrix: list[list[float]] = Field(default_factory=list)

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            return v
        n = len(v)
        for row in v:
            if len(row) != n:
                raise ValueError("Correlation matrix must be square")
        for i in range(n):
            if abs(v[i][i] - 1.0) > 1e-6:
                raise ValueError("Diagonal entries must be 1.0")
            for j in range(n):
                if abs(v[i][j] - v[j][i]) > 1e-6:
                    raise ValueError("Matrix must be symmetric")
                if not -1.0 <= v[i][j] <= 1.0:
                    raise ValueError("Correlations must be in [-1, 1]")
        return v


# ---------------------------------------------------------------------------
# Problem specification (Analyst output)
# ---------------------------------------------------------------------------


class ProblemSpec(BaseModel):
    """Fully structured reliability problem.

    This is the handoff contract from Analyst to Planner.
    """

    random_variables: list[RandomVariable]
    limit_state_expr: str = Field(
        description=(
            "Algebraic expression g(X). g < 0 indicates failure. "
            "Must use variable names from random_variables. "
            "Python/NumPy syntax: **, *, +, -, /, np.exp, np.log, np.sqrt."
        )
    )
    correlation: CorrelationMatrix = Field(default_factory=CorrelationMatrix)

    @field_validator("random_variables")
    @classmethod
    def at_least_one_variable(cls, v: list[RandomVariable]) -> list[RandomVariable]:
        if not v:
            raise ValueError("At least one random variable required")
        names = [rv.name for rv in v]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate variable names: {names}")
        return v


class AnalystOutput(BaseModel):
    """What the Analyst agent returns.

    Either a complete ProblemSpec OR a clarification question.
    """

    is_complete: bool
    spec: Optional[ProblemSpec] = None
    clarification_question: Optional[str] = None
    missing_items: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Brief explanation for audit log")


# ---------------------------------------------------------------------------
# Analysis plan (Planner output)
# ---------------------------------------------------------------------------


class AnalysisMethod(str, Enum):
    FORM = "FORM"
    SORM = "SORM"
    MCS = "MCS"           # standard Monte Carlo
    IS = "IS"             # Importance Sampling from FORM design point (simple IS)
    CE = "CE"
    ICE = "iCE"
    SIS = "SIS"           # Sequential Importance Sampling (adaptive; distinct from IS)
    SUS = "SuS"
    CLS = "CLS"           # Combination Line Sampling (FORM-chained)


class MethodFamily(str, Enum):
    """High-level categorization used in the paper's experimental analysis."""

    GRADIENT = "gradient"  # FORM, SORM
    ADAPTIVE_SAMPLING = "adaptive_sampling"  # CE, iCE, SIS
    MCMC = "mcmc"  # SuS


METHOD_FAMILY: dict[AnalysisMethod, MethodFamily] = {
    AnalysisMethod.FORM: MethodFamily.GRADIENT,
    AnalysisMethod.SORM: MethodFamily.GRADIENT,
    AnalysisMethod.MCS: MethodFamily.ADAPTIVE_SAMPLING,  # baseline (not truly adaptive)
    AnalysisMethod.IS:  MethodFamily.ADAPTIVE_SAMPLING,  # simple IS (not truly adaptive)
    AnalysisMethod.CE: MethodFamily.ADAPTIVE_SAMPLING,
    AnalysisMethod.ICE: MethodFamily.ADAPTIVE_SAMPLING,
    AnalysisMethod.SIS: MethodFamily.ADAPTIVE_SAMPLING,
    AnalysisMethod.SUS: MethodFamily.MCMC,
    AnalysisMethod.CLS: MethodFamily.ADAPTIVE_SAMPLING,  # FORM-chained line sampling
}


# Planner-level 3-category bucketing. Planner recommends one of these; the
# runtime cascading flow (graph.py) maps the chosen category to a concrete
# AnalysisMethod (FORM-based starts at FORM; Advanced presents the menu).
AnalysisCategory = Literal["FORM-based", "MCS", "Advanced"]

CATEGORY_METHODS: dict[str, list[AnalysisMethod]] = {
    "FORM-based": [AnalysisMethod.FORM, AnalysisMethod.SORM, AnalysisMethod.IS],
    "MCS":        [AnalysisMethod.MCS],
    "Advanced":   [AnalysisMethod.CE, AnalysisMethod.ICE,
                   AnalysisMethod.SIS, AnalysisMethod.SUS, AnalysisMethod.CLS],
}


class AnalysisPlan(BaseModel):
    """What the Planner agent returns.

    Planner picks ONE category. Multiple categories may be acceptable for a
    problem (the label is a set); SFT data generation expands each multi-tick
    label into per-category training pairs. At inference, the Planner emits a
    single pick and the metric checks set-membership against the gold label.
    """

    recommended_category: AnalysisCategory = Field(
        description="One of: FORM-based, MCS, Advanced."
    )


class MethodMenuReply(BaseModel):
    """Classification of a user's reply at the Advanced-method menu (step 3 HITL).

    Used at three HITL menus: the up-front category confirm (category_confirm),
    the Advanced solver menu (method_resolve), and the post-FORM SORM/IS offer
    (form_offer). The user's reply is classified here:
    - ``pick``     — they chose (``category`` for the category confirm, else ``method``)
    - ``question`` — they asked something (answer it, then re-show the menu)
    - ``decline``  — they declined the offer / are done (used by the FORM offer)
    - ``digress``  — off-topic / change the problem (hand back to Orchestrator)
    """

    kind: Literal["pick", "question", "decline", "digress"]
    method: Optional[AnalysisMethod] = Field(
        default=None, description="Chosen solver; set on a solver-menu pick."
    )
    category: Optional[AnalysisCategory] = Field(
        default=None, description="Chosen category; set on a category-confirm pick."
    )
    note: str = Field(default="", description="Brief rationale or restated question.")


# ---------------------------------------------------------------------------
# Code generation (Code Engineer output)
# ---------------------------------------------------------------------------


class GeneratedCode(BaseModel):
    """What the Code Engineer agent returns."""

    code: str = Field(description="Complete executable Python source")
    language: Literal["python"] = "python"
    entry_file: str = Field(default="run_analysis.py")
    reasoning: str = Field(default="")


# ---------------------------------------------------------------------------
# Execution result (Runner tool output)
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    success: bool
    beta: Optional[float] = None
    pf: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0
    error_type: Optional[str] = None
    raw_outputs: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Interpreter output
# ---------------------------------------------------------------------------


class EngineeringInterpretation(BaseModel):
    """Final human-facing explanation."""

    summary: str
    reliability_level: Literal[
        "very_low", "low", "moderate", "high", "very_high"
    ]
    comparison_to_target: Optional[str] = None
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrator decision
# ---------------------------------------------------------------------------


class UserIntent(str, Enum):
    NEW_PROBLEM = "new_problem"
    MODIFY_PROBLEM = "modify_problem"
    EXPLAIN_RESULT = "explain_result"
    GENERAL_QUESTION = "general_question"
    CLARIFICATION_REPLY = "clarification_reply"


class OrchestratorDecision(BaseModel):
    intent: UserIntent
    next_agent: Literal["analyst", "planner", "interpreter", "direct_answer"]
    reasoning: str = ""
