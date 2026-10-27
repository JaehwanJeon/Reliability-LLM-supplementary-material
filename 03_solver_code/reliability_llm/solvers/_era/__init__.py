"""Vendored ERA Python sources.

Source: https://www.bgu.tum.de/era/software/eradist/ and
        https://www.bgu.tum.de/era/software/software00/
Original license: MIT (see LICENSE.txt in this directory).
Copyright (c) 2022 ERA Group (Engineering Risk Analysis Group, TU Munich).

Vendored as-is; no modifications to the numerical code. Keep the bare
intra-module imports working by prepending this directory to sys.path at
package load. This module does not own any algorithm — it's a thin
packaging shim around the upstream ERA code.
"""

from __future__ import annotations

import os
import sys

_ERA_DIR = os.path.dirname(__file__)
if _ERA_DIR not in sys.path:
    sys.path.insert(0, _ERA_DIR)

from ERADist import ERADist  # noqa: E402
from ERANataf import ERANataf  # noqa: E402
from FORM_HLRF import FORM_HLRF  # noqa: E402
from SuS import SuS  # noqa: E402
from CEIS_GM import CEIS_GM  # noqa: E402
from iCE_GM import iCE_GM  # noqa: E402
from SIS_aCS import SIS_aCS  # noqa: E402
from CLS import CLS  # noqa: E402

__all__ = [
    "ERADist",
    "ERANataf",
    "FORM_HLRF",
    "SuS",
    "CEIS_GM",
    "iCE_GM",
    "SIS_aCS",
    "CLS",
]
