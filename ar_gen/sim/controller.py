"""RC controller malfunctions (manual Appendix F) and their side effects."""
from __future__ import annotations

import math

from .timeutil import HOUR_MS, INTERVALS_PER_YEAR, SEC_MS


def _random_running_time(ctx) -> int | None:
    segs = ctx.ups.running_segments
    total = sum(e - s for s, e in segs)
    if total <= 0:
        return None
    x = ctx.r.random() * total
    for s, e in segs:
        if x < e - s:
            return s + int(x)
        x -= e - s
    return segs[-1][1] - 1


def step(ctx) -> None:
    a, r, cfg, out = ctx.a, ctx.r, ctx.cfg, ctx.out
    cc = cfg["controller"]
    if not ctx.ups.running_segments:
        return

    # A worn-out module eventually fails for good (cleared only by replacement).
    if a.ctrl_deg >= 1.0 and a.persistent_fault is None:
        t = _random_running_time(ctx)
        sig = cc["persistent_signals"][r.randrange(len(cc["persistent_signals"]))]
        if sig in a.alarms:
            a.alarms[sig]["clear_at"] = None
        else:
            out.raise_alarm(a, t, sig, "MALFUNCTION", "Persistent")
        a.persistent_fault = sig

    rates = {s: v for s, v in cc["malfunction_rates_per_year"].items()
             if s != "GPS Malfunction" or a.relay_module.startswith("REL-15")}
    mult = 1.0 + cc["deg_gain"] * a.ctrl_deg ** 2
    if a.coastal:
        mult *= cc["coastal_mult"]
    if a.harsh and ctx.cond.humidity_pct >= 85:
        mult *= cc["harsh_humidity_mult"]
    p_total = sum(rates.values()) * mult / INTERVALS_PER_YEAR
    if r.random() >= p_total:
        return
    x = r.random() * sum(rates.values())
    sig = list(rates)[-1]
    for name, w in rates.items():
        if x < w:
            sig = name
            break
        x -= w
    t = _random_running_time(ctx)
    if t is None or sig in a.alarms:
        return

    if sig == "Battery Charger Fault":
        out.raise_alarm(a, t, sig, "MALFUNCTION", "SIM charger output lost")
        a.charger_fault, a.charger_fault_since = True, t
    elif sig == "External Load Overload":
        until = t + int(r.uniform(*cc["ext_overload_restore_h"]) * HOUR_MS)
        out.raise_alarm(a, t, sig, "MALFUNCTION", "External supply switched off", clear_at=until)
        modem_was_on = a.ctrl == "RUN" and a.ext_on
        a.ext_overload_until, a.ext_on = until, False
        if modem_was_on:
            ctx.ups.power_events.append((t, False))
    else:
        dur_s = cc["duration_median_s"] * math.exp(r.gauss(0.0, cc["duration_sigma"]))
        out.raise_alarm(a, t, sig, "MALFUNCTION", None, clear_at=t + max(SEC_MS, int(dur_s * SEC_MS)))
