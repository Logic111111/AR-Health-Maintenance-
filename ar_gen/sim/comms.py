"""SCADA communications: modem power, disconnections, watchdog restarts, RS485 errors."""
from __future__ import annotations

import math

from . import rng as rngmod
from .timeutil import MIN_MS, SEC_MS

INTERVAL_MS = 15 * MIN_MS


def signal_dbm(a, cfg: dict, weather: str, r) -> float:
    c = cfg["comms"]
    s = a.base_signal_dbm + r.gauss(0.0, c["signal_noise_db"][a.channel]) - c["deg_signal_loss_db"] * a.comms_deg
    if a.channel == "MOBILE":
        s += c["rain_fade_db"][weather]
    return s


def _hazard(a, cfg: dict, weather: str, sig: float) -> float:
    """Expected disconnections per 15-minute interval while connected."""
    c = cfg["comms"]
    lam = c["disconnect_prob_per_interval"][a.channel] * c["weather_mult"][weather]
    if a.channel == "MOBILE":
        lam *= math.exp(max(0.0, c["weak_signal_knee_dbm"] - sig) / c["weak_signal_scale_db"])
    return lam * (1.0 + c["deg_mult"] * a.comms_deg ** 2)


def _reconnect_delay(a, cfg: dict, r) -> int | None:
    """Time until the link comes back by itself; None when the modem/antenna is dead."""
    c = cfg["comms"]
    p_hung = min(1.0, max(0.0, (a.comms_deg - 0.9) / 0.2))
    if r.random() < p_hung:
        return None
    m = c["reconnect_median_min"] * (1.0 + 4.0 * a.comms_deg ** 2)
    return max(5 * SEC_MS, int(min(360.0, m * math.exp(r.gauss(0.0, c["reconnect_sigma"]))) * MIN_MS))


def _watchdog_delay(a, cfg: dict, t: int) -> int:
    """Watchdog back-off: 20 min, then about half the outage so far, at most 4 h."""
    base = int(cfg["comms"]["watchdog_restart_after_min"] * MIN_MS)
    since = t - a.disconnected_since if a.disconnected_since is not None else 0
    return max(base, min(4 * 60 * MIN_MS, since // 2))


def _disconnect(ctx, t: int) -> None:
    a, out, cfg = ctx.a, ctx.out, ctx.cfg
    out.comms_event(a, t, "DISCONNECT")
    a.connected, a.disconnected_since = False, t
    d = _reconnect_delay(a, cfg, ctx.r)
    a.reconnect_at = None if d is None else t + d
    a.restart_at = t + int(cfg["comms"]["watchdog_restart_after_min"] * MIN_MS)


def _modem_power(ctx, t: int, on: bool) -> None:
    a, cfg = ctx.a, ctx.cfg
    if on == a.modem_on:
        return
    a.modem_on = on
    if not on:
        if a.connected:
            ctx.out.comms_event(a, t, "DISCONNECT")
            a.connected, a.disconnected_since = False, t
        a.reconnect_at = a.restart_at = None
    else:
        boot = int(ctx.r.uniform(*cfg["comms"]["modem_boot_min"]) * MIN_MS)
        d = _reconnect_delay(a, cfg, ctx.r)
        a.reconnect_at = None if d is None else t + boot + min(d, 2 * MIN_MS)
        a.restart_at = t + boot + int(cfg["comms"]["watchdog_restart_after_min"] * MIN_MS)


def _run_segment(ctx, t: int, end: int, lam: float) -> None:
    """Modem powered on during [t, end)."""
    a, out, cfg, r = ctx.a, ctx.out, ctx.cfg, ctx.r
    for _ in range(1000):
        if t >= end:
            return
        if a.connected:
            if lam <= 0:
                return
            dt = r.expovariate(lam / INTERVAL_MS)
            if t + dt >= end:
                return
            t = t + max(1, int(dt))
            _disconnect(ctx, t)
            continue
        rc = a.reconnect_at
        rs = a.restart_at
        if rc is not None and rc < end and (rs is None or rc <= rs):
            t = max(t, rc)
            out.comms_event(a, t, "RECONNECT")
            a.connected, a.disconnected_since = True, None
            a.reconnect_at = a.restart_at = None
            continue
        if rs is not None and rs < end:
            t = max(t, rs)
            out.comms_event(a, t, "MODEM_RESTART")
            boot = int(r.uniform(*cfg["comms"]["modem_boot_min"]) * MIN_MS)
            d = _reconnect_delay(a, cfg, r)
            a.reconnect_at = None if d is None else t + boot + d
            a.restart_at = t + boot + _watchdog_delay(a, cfg, t)
            continue
        return


def step(ctx) -> float | None:
    """Advance the comms link through the interval; returns the signal reading at t1."""
    a, r, cfg, ups, out = ctx.a, ctx.r, ctx.cfg, ctx.ups, ctx.out
    t0, t1 = ctx.t0, ctx.t1
    w = ctx.cond.weather
    sig = signal_dbm(a, cfg, w, r)
    lam = _hazard(a, cfg, w, sig)

    if a.modem_on != ups.modem_on_start:
        _modem_power(ctx, t0, ups.modem_on_start)
    t = t0
    for te, on in sorted(ups.power_events):
        if a.modem_on:
            _run_segment(ctx, t, te, lam)
        _modem_power(ctx, te, on)
        t = te
    if a.modem_on:
        _run_segment(ctx, t, t1, lam)

    # RS485 link to an external device (counted by the SCADA front end)
    if a.has_rs485 and ups.running_segments:
        lam485 = cfg["comms"]["rs485_errors_per_month"] / (30.0 * 96.0) * (1.0 + 4.0 * a.comms_deg ** 2)
        for _ in range(rngmod.poisson(r, lam485)):
            tb = t0 + 1 + int(r.random() * (t1 - t0 - 12 * SEC_MS))
            for j in range(r.randint(*cfg["comms"]["rs485_burst"])):
                out.comms_event(a, tb + j * 2 * SEC_MS, "RS485_ERROR", "RS485")

    return round(sig, 1) if a.modem_on and a.ctrl == "RUN" else None
