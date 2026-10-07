"""Protection timing (manual 6.1): IEC curves and definite-time elements."""
from __future__ import annotations

import math


def iec_standard_inverse(current_a: float, pickup_a: float, tm: float) -> float:
    """IEC 60255 standard inverse operating time (s); inf below pickup."""
    if current_a <= pickup_a:
        return math.inf
    return tm * 0.14 / ((current_a / pickup_a) ** 0.02 - 1.0)
