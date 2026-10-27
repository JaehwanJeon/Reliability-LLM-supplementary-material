"""Code execution backend.

After the (C) cleanup, only one execution path remains: a Python subprocess
that runs generated code importing `reliability_llm.solvers`. FERUM and
ERA_MATLAB paths were removed.
"""

from reliability_llm.backends.era_python import run

__all__ = ["run"]
