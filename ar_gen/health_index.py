"""Health Index (HI) mathematical model.

Step 1  sub-score   s_i = 100 * clamp((bad - x) / (bad - good), 0, 1)   (two-sided band for V1,
                    look-up maps for V41/V42, rated-relative V18/V21, channel-specific V31)
Step 2  factor      F_k = sum(w_i * s_i) / sum(w_i) over the available variables of factor k
Step 3  index       HI  = sum_k(W_k * F_k) / sum_k(W_k),  W_k = sum of the factor's weights
                    (with every variable present this is exactly sum_i(w_i * s_i) / 100)
Step 4  caps        if a critical variable's sub-score is 0 the HI is capped (config caps)
Step 5  class       GOOD >= 80, FAIR >= 60, POOR >= 40, else CRITICAL; healthy = HI >= 60
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .db.schema import SCORE_COLUMNS, V_COLUMNS, V_SHORT

FACTOR_COLUMN = {
    "BATTERY": "f_battery", "FAULTS": "f_faults", "SWITCHING": "f_switching", "COMMS": "f_comms",
    "CONTROLLING": "f_controlling", "AGING": "f_aging", "ENVIRONMENT": "f_environment",
}


# ------------------------------------------------------------------ sub-scores
def _linear(x, good, bad):
    return 100.0 * np.clip((bad - x) / (bad - good), 0.0, 1.0)


def _band(x, lo, hi, bad_lo, bad_hi):
    below = 100.0 * np.clip((x - bad_lo) / (lo - bad_lo), 0.0, 1.0)
    above = 100.0 * np.clip((bad_hi - x) / (bad_hi - hi), 0.0, 1.0)
    return np.where(x < lo, below, np.where(x > hi, above, 100.0))


def score_values(short: str, values, thr: dict, rated_break_ka=None, rated_mech_ops=None,
                 channel=None) -> np.ndarray:
    """Vectorised sub-score for one variable; NaN where the value is missing."""
    if "map" in thr:
        mapping = {str(k): float(v) for k, v in thr["map"].items()}
        ser = pd.Series(values, dtype="object")
        keys = ser.map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v))
                       else str(int(v)) if isinstance(v, (int, float, np.integer, np.floating, bool))
                       else str(v))
        return keys.map(mapping).astype(float).to_numpy()

    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if "band" in thr:
        lo, hi = thr["band"]
        bad_lo, bad_hi = thr["bad"]
        s = _band(x, lo, hi, bad_lo, bad_hi)
    elif "good_frac_rated" in thr:
        rated_a = np.asarray(rated_break_ka, dtype=float) * 1000.0
        s = _linear(x, thr["good_frac_rated"] * rated_a, thr["bad_frac_rated"] * rated_a)
    elif "good_frac_mech" in thr:
        ops = np.asarray(rated_mech_ops, dtype=float)
        s = _linear(x, thr["good_frac_mech"] * ops, thr["bad_frac_mech"] * ops)
    elif "MOBILE" in thr:
        ch = np.asarray(channel, dtype=object)
        good = np.where(ch == "FIBRE", thr["FIBRE"]["good"], thr["MOBILE"]["good"]).astype(float)
        bad = np.where(ch == "FIBRE", thr["FIBRE"]["bad"], thr["MOBILE"]["bad"]).astype(float)
        s = _linear(x, good, bad)
    else:
        s = _linear(x, float(thr["good"]), float(thr["bad"]))
    return np.where(np.isnan(x), np.nan, s)


def sub_score(short: str, value, thr: dict, asset: dict) -> float:
    """Scalar convenience wrapper around :func:`score_values`."""
    out = score_values(short, [value], thr, [asset.get("rated_break_ka")],
                       [asset.get("rated_mech_ops")], [asset.get("comms_channel")])
    return float(out[0])


def classify(hi: float, cfg: dict) -> str:
    for name, lower in sorted(cfg["classes"].items(), key=lambda kv: -kv[1]):
        if hi >= lower:
            return name
    return "CRITICAL"


# ------------------------------------------------------------------ index
def compute_health_index(variables: pd.DataFrame, assets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute sub-scores, factor scores, HI, class and caps for every row of ``variables``.

    ``variables`` holds asset_id, ts and the v01..v42 columns; ``assets`` holds
    asset_id, rated_break_ka, rated_mech_ops and comms_channel.
    """
    ctx = variables[["asset_id"]].merge(
        assets[["asset_id", "rated_break_ka", "rated_mech_ops", "comms_channel"]],
        on="asset_id", how="left")
    n = len(variables)
    scores = np.full((n, len(V_COLUMNS)), np.nan)
    shorts = [c[:3] for c in V_COLUMNS]
    for j, col in enumerate(V_COLUMNS):
        short = shorts[j]
        scores[:, j] = score_values(short, variables[col].to_numpy(), cfg["thresholds"][short],
                                    ctx["rated_break_ka"].to_numpy(), ctx["rated_mech_ops"].to_numpy(),
                                    ctx["comms_channel"].to_numpy())

    weights = np.array([float(cfg["weights"][s]) for s in shorts])
    present = ~np.isnan(scores)
    filled = np.where(present, scores, 0.0)

    out = pd.DataFrame({"asset_id": variables["asset_id"].to_numpy(),
                        "ts": variables["ts"].to_numpy()})
    hi_num = np.zeros(n)
    hi_den = np.zeros(n)
    for factor, members in cfg["factors"].items():
        idx = [shorts.index(m) for m in members]
        w = weights[idx]
        num = (filled[:, idx] * w).sum(axis=1)
        den = (present[:, idx] * w).sum(axis=1)
        f = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
        out[FACTOR_COLUMN[factor]] = np.round(f, 3)
        w_k = w.sum()
        valid = ~np.isnan(f)
        hi_num += np.where(valid, f * w_k, 0.0)
        hi_den += np.where(valid, w_k, 0.0)
    hi = np.divide(hi_num, hi_den, out=np.zeros(n), where=hi_den > 0)

    cap_applied = np.full(n, None, dtype=object)
    for var, cap in sorted(cfg.get("caps", {}).items(), key=lambda kv: -kv[1]):
        j = shorts.index(var)
        hit = present[:, j] & (scores[:, j] <= 0.0) & (hi > cap)
        hi = np.where(hit, float(cap), hi)
        cap_applied = np.where(hit, var, cap_applied)

    deficits = np.where(present, weights * (100.0 - scores), 0.0)
    order = np.argsort(-deficits, axis=1, kind="stable")[:, : cfg.get("worst_n", 3)]
    worst = []
    for i in range(n):
        names = [shorts[j] for j in order[i] if deficits[i, j] > 1e-9]
        worst.append(",".join(names) if names else None)

    out["health_index"] = np.round(hi, 3)
    out["health_class"] = [classify(v, cfg) for v in hi]
    out["is_healthy"] = hi >= cfg["healthy_min"]
    out["cap_applied"] = cap_applied
    out["worst_variables"] = worst
    for j, col in enumerate(SCORE_COLUMNS):
        out[col] = np.round(scores[:, j], 3)
    return out
