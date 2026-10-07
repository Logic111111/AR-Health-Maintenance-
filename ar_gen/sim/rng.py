"""Deterministic random streams.

Every (asset, interval) and (area, interval/hour) gets its own ``random.Random``
seeded from the global seed and integer keys. A stream never depends on how many
numbers earlier intervals consumed, so generating two years in one run or in
70,000 separate 15-minute runs yields identical data.
"""
from __future__ import annotations

import random

_MASK = (1 << 64) - 1

# Stream identifiers (first key after the global seed)
FLEET = 1
AREA_HOUR = 2
AREA_INTERVAL = 3
ASSET_INTERVAL = 4


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def derive_seed(*keys: int) -> int:
    """Mix any number of integers into one 64-bit seed."""
    h = 0x6A09E667F3BCC909
    for k in keys:
        h = _splitmix64(h ^ (int(k) & _MASK))
    return h


def rng(*keys: int) -> random.Random:
    return random.Random(derive_seed(*keys))


def poisson(r: random.Random, lam: float) -> int:
    """Poisson draw (Knuth) for the small rates used per 15-minute interval."""
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(round(r.gauss(lam, lam ** 0.5))))
    import math
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= r.random()
        if p <= limit:
            return k
        k += 1


def weighted_choice(r: random.Random, weights: dict):
    """Pick a key of ``weights`` proportionally to its value (deterministic order)."""
    items = list(weights.items())
    total = sum(w for _, w in items)
    x = r.random() * total
    acc = 0.0
    for key, w in items:
        acc += w
        if x < acc:
            return key
    return items[-1][0]
