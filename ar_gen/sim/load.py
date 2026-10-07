"""Feeder load model: daily curves, weekends, temperature and noise."""
from __future__ import annotations

import cmath
import math
import random

from .state import AssetState

# Hourly shapes (per-unit of peak), local time 00:00 .. 23:00.
SHAPES = {
    "RESIDENTIAL": [0.45, 0.40, 0.38, 0.37, 0.38, 0.45, 0.55, 0.55, 0.52, 0.50, 0.50, 0.52,
                    0.53, 0.52, 0.50, 0.50, 0.55, 0.65, 0.85, 1.00, 0.98, 0.90, 0.75, 0.58],
    "COMMERCIAL": [0.30, 0.28, 0.27, 0.27, 0.28, 0.32, 0.45, 0.65, 0.85, 0.95, 1.00, 1.00,
                   0.98, 1.00, 1.00, 0.97, 0.90, 0.80, 0.70, 0.62, 0.55, 0.48, 0.40, 0.34],
    "INDUSTRIAL": [0.60, 0.58, 0.57, 0.57, 0.58, 0.62, 0.75, 0.90, 0.98, 1.00, 1.00, 0.98,
                   0.95, 0.98, 1.00, 0.98, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.66, 0.62],
}
SHAPES["MIXED"] = [(a + b) / 2 for a, b in zip(SHAPES["RESIDENTIAL"], SHAPES["COMMERCIAL"])]


def shape_at(feeder_type: str, hour: float) -> float:
    s = SHAPES[feeder_type]
    h0 = int(hour) % 24
    frac = hour - int(hour)
    return s[h0] * (1 - frac) + s[(h0 + 1) % 24] * frac


def load_current(a: AssetState, cfg: dict, local_hour: float, weekday: int, ambient_c: float,
                 r: random.Random) -> float:
    """Average phase current (A) this recloser would carry if closed and energised."""
    lc = cfg["load"]
    base = shape_at(a.feeder_type, local_hour)
    if weekday >= 5:
        base *= lc["weekend_factor"][a.feeder_type][weekday - 5]
    base *= 1.0 + lc["temp_coeff_per_c"] * max(0.0, ambient_c - 28.0)
    a.load_noise = lc["noise_ar"] * a.load_noise + r.gauss(0.0, lc["noise_sigma"])
    return max(0.0, a.peak_a * base * (1.0 + a.load_noise))


def phase_currents(a: AssetState, i_avg: float, cfg: dict, r: random.Random) -> list[float]:
    s = cfg["load"]["phase_noise_sigma"]
    return [max(0.0, i_avg * (1.0 + k) * (1.0 + r.gauss(0.0, s))) for k in a.imbalance]


def residual_current(ia: float, ib: float, ic: float) -> float:
    """Magnitude of the phasor sum (balanced 120-degree phases)."""
    rot = cmath.exp(-2j * math.pi / 3)
    return abs(ia + ib * rot + ic * rot * rot)
