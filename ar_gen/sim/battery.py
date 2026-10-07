"""12 V sealed lead-acid backup battery electrical model (manual 2.2.11, 4.8)."""
from __future__ import annotations

import bisect

# Open-circuit voltage of a 12 V lead-acid battery vs state of charge.
_OCV_SOC = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_OCV_V = [10.50, 11.31, 11.58, 11.75, 11.90, 12.06, 12.20, 12.32, 12.42, 12.50, 12.73]

FLOAT_V_25C = 13.70
BULK_BOOST_V = 0.40
TEMP_COMP_V_PER_C = -0.018


def ocv(soc: float) -> float:
    """Open-circuit voltage (V) for a state of charge in [0, 1]."""
    soc = min(1.0, max(0.0, soc))
    i = bisect.bisect_right(_OCV_SOC, soc)
    if i >= len(_OCV_SOC):
        return _OCV_V[-1]
    x0, x1 = _OCV_SOC[i - 1], _OCV_SOC[i]
    y0, y1 = _OCV_V[i - 1], _OCV_V[i]
    return y0 + (y1 - y0) * (soc - x0) / (x1 - x0)


def soc_at_ocv(voltage: float) -> float:
    """Inverse of :func:`ocv`."""
    if voltage <= _OCV_V[0]:
        return 0.0
    if voltage >= _OCV_V[-1]:
        return 1.0
    i = bisect.bisect_right(_OCV_V, voltage)
    x0, x1 = _OCV_V[i - 1], _OCV_V[i]
    y0, y1 = _OCV_SOC[i - 1], _OCV_SOC[i]
    return y0 + (y1 - y0) * (voltage - x0) / (x1 - x0)


def battery_rint(soh: float, r0: float, exponent: float) -> float:
    """Internal resistance (ohm); grows steeply as state of health falls."""
    return r0 / max(soh, 0.05) ** exponent


def terminal_voltage(soc: float, current_a: float, rint: float, mode: str, temp_c: float,
                     bulk_fraction: float = 0.0, float_v_25c: float = FLOAT_V_25C,
                     bulk_boost_v: float = BULK_BOOST_V, temp_comp: float = TEMP_COMP_V_PER_C,
                     ocv_offset: float = 0.0) -> float:
    """Battery terminal voltage.

    mode: ``FLOAT``/``BULK`` (charger on), ``DISCHARGE`` (on battery) or ``REST``.
    ``ocv_offset`` models a shorted cell (about -2.1 V).
    """
    if mode in ("FLOAT", "BULK"):
        v = float_v_25c + temp_comp * (temp_c - 25.0) + ocv_offset
        if mode == "BULK":
            v += bulk_boost_v * max(0.0, min(1.0, bulk_fraction))
        return v
    if mode == "DISCHARGE":
        return ocv(soc) + ocv_offset - abs(current_a) * rint
    return ocv(soc) + ocv_offset


def calendar_soh(equivalent_age_days: float, service_interval_years: float, coeff: float) -> float:
    """Capacity fraction left after temperature-weighted ageing (before cycle/extra losses)."""
    x = (equivalent_age_days / 365.25) / service_interval_years
    return 1.0 - coeff * x * x


def temperature_age_factor(temp_c: float, doubling_c: float) -> float:
    """Ageing speed relative to 25 C (life halves for every +doubling_c)."""
    return max(0.5, 2.0 ** ((temp_c - 25.0) / doubling_c))
