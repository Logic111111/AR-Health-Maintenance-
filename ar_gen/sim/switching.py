"""Switching mechanics: contact wear curve, operating times, failure probability."""
from __future__ import annotations

import math
import random

FLAT_CURRENT_A = 800.0      # manual 2.1.4: flat 30,000 operations up to ~0.8 kA
FLAT_OPS = 30000.0


def ops_at_current(current_a: float, rated_break_ka: float, rated_fault_ops: float) -> float:
    """Permitted interruptions N(I) from the manual's breaking-duty curve.

    Flat at 30,000 up to 800 A, then a straight line on log-log axes down to the
    rated number of fault operations at the rated breaking current.
    """
    i_rated = rated_break_ka * 1000.0
    if current_a <= FLAT_CURRENT_A:
        return FLAT_OPS
    if current_a >= i_rated:
        return float(rated_fault_ops)
    frac = (math.log10(current_a) - math.log10(FLAT_CURRENT_A)) / (
        math.log10(i_rated) - math.log10(FLAT_CURRENT_A))
    log_n = math.log10(FLAT_OPS) + frac * (math.log10(rated_fault_ops) - math.log10(FLAT_OPS))
    return 10.0 ** log_n


def contact_wear_increment(current_a: float, rated_break_ka: float, rated_fault_ops: float) -> float:
    """Contact wear (%) consumed by interrupting ``current_a`` once on one phase."""
    if current_a <= 0:
        return 0.0
    return 100.0 / ops_at_current(current_a, rated_break_ka, rated_fault_ops)


def opening_time_ms(base_ms: float, mech_deg: float, cfg: dict, r: random.Random) -> float:
    jitter = 1.0 + r.gauss(0.0, cfg["time_jitter"])
    return base_ms * (1.0 + cfg["open_wear_coeff"] * mech_deg ** 2) * jitter


def closing_time_ms(base_ms: float, mech_deg: float, cfg: dict, r: random.Random) -> float:
    jitter = 1.0 + r.gauss(0.0, cfg["time_jitter"])
    return base_ms * (1.0 + cfg["close_wear_coeff"] * mech_deg ** 2) * jitter


def fail_probability(mech_deg: float, cfg: dict) -> float:
    """Probability that a trip/close request fails, rising once wear passes 0.8."""
    return min(0.9, cfg["fail_base_prob"] + cfg["fail_gain"] * max(0.0, mech_deg - 0.8) ** 1.5)
