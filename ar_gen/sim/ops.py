"""Switching operations and protection sequences (manual 6.1-6.2, 7.2, 7.7).

Every interval the engine builds a :class:`Ctx` and dispatches due queue events
(``a.queue``) to the handlers in :data:`HANDLERS`.
"""
from __future__ import annotations

import math

from . import maintenance
from . import rng as rngmod
from .load import residual_current
from .protection import iec_standard_inverse
from .switching import (closing_time_ms, contact_wear_increment, fail_probability,
                        opening_time_ms)
from .timeutil import HOUR_MS, MIN_MS, SEC_MS, ms_to_dt

FAULT_GROUP = {"OC_A": "OC", "OC_B": "OC", "OC_C": "OC"}


class Ctx:
    """Everything an event handler needs for one asset and one interval."""

    def __init__(self, cfg, tz, out, a, r, ups, cond, t0, t1, i_phase):
        self.cfg, self.tz, self.out, self.a, self.r = cfg, tz, out, a, r
        self.ups, self.cond, self.t0, self.t1 = ups, cond, t0, t1
        self.i_phase = i_phase
        self.switch_at_start = a.switch
        self.switch_log: list[tuple[int, str]] = []

    def set_switch(self, t: int, state: str, reason: str | None) -> None:
        self.a.switch, self.a.open_reason = state, reason
        self.switch_log.append((t, state))

    def running(self, t: int) -> bool:
        if self.t0 <= t < self.t1:
            return self.ups.running(t)
        return self.a.ctrl in ("RUN", "STANDBY")

    def line_live(self, t: int) -> bool:
        return self.ups.line_live(t)

    def load_currents(self) -> list:
        ia, ib, ic = self.i_phase
        return [ia, ib, ic, residual_current(ia, ib, ic)]


# ------------------------------------------------------------------ primitives
def do_open(ctx: Ctx, t: int, source: str, ar_state, currents, fault=None, force=False):
    a, s, r, out = ctx.a, ctx.cfg["switching"], ctx.r, ctx.out
    op_time = opening_time_ms(a.to_base_ms, a.mech_deg, s, r)
    fail = (not force) and r.random() < fail_probability(a.mech_deg, s)
    row = out.op(a, t, "OPEN", source, ar_state, currents, None if fail else op_time,
                 "FAIL" if fail else "SUCCESS", fault["id"] if fault else None)
    if fault is not None:
        fault["ops"].append(row)
    else:
        out.emit_op(row)
    if fail:
        out.event(a, t, "Trip Request Fail", "WARNING", "Faulty Actuator or Mechanism Failure")
        return False, row, None
    a.co_total += 1
    for p, key in enumerate(("trip_ia", "trip_ib", "trip_ic")):
        a.wear[p] += contact_wear_increment(row[key] or 0.0, a.rated_break_ka, a.rated_fault_ops)
    a.mech_deg += s["mech_deg_per_op"]
    if op_time > s["excessive_to_ms"]:
        out.event(a, t, "Excessive To", "MALFUNCTION", f"{op_time:.0f} ms")
    return True, row, op_time


def do_close(ctx: Ctx, t: int, source: str, ar_state, fault=None, force=False):
    a, s, r, out = ctx.a, ctx.cfg["switching"], ctx.r, ctx.out
    op_time = closing_time_ms(a.tc_base_ms, a.mech_deg, s, r)
    fail = (not force) and r.random() < fail_probability(a.mech_deg, s)
    row = out.op(a, t, "CLOSE", source, ar_state, None, None if fail else op_time,
                 "FAIL" if fail else "SUCCESS", fault["id"] if fault else None)
    if fault is not None:
        fault["ops"].append(row)
    else:
        out.emit_op(row)
    if fail:
        out.event(a, t, "Close Request Fail", "WARNING", "Faulty Actuator or Mechanism Failure")
        return False, row, None
    a.mech_deg += s["mech_deg_per_op"]
    if op_time > s["excessive_tc_ms"]:
        out.event(a, t, "Excessive Tc", "MALFUNCTION", f"{op_time:.0f} ms")
    return True, row, op_time


def _lognormal_ms(r, median_h: float, sigma: float) -> int:
    return int(median_h * math.exp(r.gauss(0.0, sigma)) * HOUR_MS)


def _clear_alerts(ctx: Ctx, t: int) -> None:
    for sig in [s for s, info in ctx.a.alarms.items() if info["cat"] == "ALERT"]:
        ctx.out.clear_alarm(ctx.a, t, sig)


# ------------------------------------------------------------------ protection
def start_fault(ctx: Ctx, t: int, payload: dict) -> None:
    a, r, cfg, out = ctx.a, ctx.r, ctx.cfg, ctx.out
    if a.switch != "CLOSED" or a.active_fault is not None or a.in_maintenance:
        return
    if not ctx.running(t) or not ctx.line_live(t):
        return
    pc = cfg["protection"]
    ftype = payload["type"]
    load = list(ctx.i_phase)
    rated_a = a.rated_break_ka * 1000.0 * 0.98
    tms = pc["tm_by_trip"]
    min_rt = pc["min_relay_time_s"]
    cur = list(load)
    if ftype == "OC":
        phases = rngmod.weighted_choice(r, pc["oc_phase_sets"])
        i_f = a.prospective_fault_ka * 1000.0 * pc["fault_current_median_frac"] * math.exp(
            r.gauss(0.0, pc["fault_current_sigma"]))
        i_f = min(rated_a, max(1.5 * a.oc_pickup_a, i_f))
        for p in phases:
            cur["ABC".index(p)] = i_f * r.uniform(0.9, 1.0)
        dom = max((("ABC".index(p)) for p in phases), key=lambda i: cur[i])
        i_n = (0.01 if len(phases) == 3 else 0.03) * i_f * r.random()
        ftype_db = f"OC_{'ABC'[dom]}"
        relay = [max(min_rt, iec_standard_inverse(max(cur), a.oc_pickup_a, tm)) for tm in tms]
        max_i = max(cur)
    elif ftype == "EF":
        p = r.randrange(3)
        i_ef = min(rated_a, max(1.5 * a.ef_pickup_a, r.uniform(*pc["ef_current_ka"]) * 1000.0))
        cur[p] += i_ef
        i_n, phases, ftype_db, max_i = i_ef, f"{'ABC'[p]}N", "EF", i_ef
        relay = [max(min_rt, iec_standard_inverse(i_ef, a.ef_pickup_a, tm)) for tm in tms]
    elif ftype == "SEF":
        p = r.randrange(3)
        i_n = r.uniform(*pc["sef_current_a"])
        cur[p] += i_n
        phases, ftype_db, max_i = f"{'ABC'[p]}N", "SEF", i_n
        relay = [pc["sef_time_s"]] * 3
    elif ftype == "NPS":
        p = r.randrange(3)
        cur[p] += r.uniform(*pc["nps_current_a"])
        i_n = residual_current(*cur)
        phases, ftype_db, max_i = "ABC"[p], "NPS", cur[p]
        relay = [pc["definite_time_s"]["NPS"]] * 3
    else:  # UV / OV / UF / OF: voltage or frequency element trips at load current
        i_n = residual_current(*cur)
        phases, ftype_db, max_i = "ABC", ftype, max(cur)
        relay = [pc["definite_time_s"][ftype]] * 3

    perm_p = pc["permanent_prob_storm"] if ctx.cond.weather == "STORM" else pc["permanent_prob"]
    if r.random() < perm_p:
        persist = 99
    else:
        u = r.random()
        p1, p2 = pc["temp_clear_after_first"], pc["temp_clear_after_second"]
        persist = 0 if u < p1 else (1 if u < p1 + (1 - p1) * p2 else 2)

    a.active_fault = dict(id=out.next_id("fault"), start=int(t), type=ftype_db, phases=phases,
                          group=FAULT_GROUP.get(ftype_db, ftype_db), currents=cur + [i_n],
                          max_current=max_i, relay_s=relay, trips=0, recloses=0, fault_on_ms=0.0,
                          persist=persist, ops=[])
    out.schedule(a, t + int(relay[0] * SEC_MS + a.to_base_ms), "trip")


def finalize(ctx: Ctx, f: dict, outcome: str, end: int) -> None:
    a, out = ctx.a, ctx.out
    out.fault(dict(fault_id=f["id"], start_ts=ms_to_dt(f["start"]), end_ts=ms_to_dt(end),
                   asset_id=a.asset_id, fault_type=f["type"], phases=f["phases"],
                   max_fault_current_a=round(f["max_current"], 1),
                   duration_ms=int(round(f["fault_on_ms"])), reclose_attempts=f["recloses"],
                   outcome=outcome))
    for row in f["ops"]:
        out.emit_op(row)
    a.active_fault = None


def _lockout(ctx: Ctx, f: dict, t: int) -> None:
    ctx.set_switch(t, "LOCKOUT", "LOCKOUT")
    finalize(ctx, f, "LOCKOUT", t)
    ctx.out.raise_alarm(ctx.a, t, f"Open({f['group']})", "ALERT", f"Lockout after {f['trips']} trip(s)")
    pc = ctx.cfg["protection"]
    ctx.out.schedule(ctx.a, t + _lognormal_ms(ctx.r, pc["restore_median_h"], pc["restore_sigma"]),
                     "crew_close", {"attempt": 0})


def h_trip(ctx: Ctx, t: int, payload: dict) -> None:
    a, r, cfg, out = ctx.a, ctx.r, ctx.cfg, ctx.out
    f = a.active_fault
    if f is None:
        return
    k = f["trips"]
    if not ctx.running(t):
        f["fault_on_ms"] += f["relay_s"][k] * SEC_MS + 400
        finalize(ctx, f, "UPSTREAM_CLEARED", t + 400)
        return
    cur = [c * r.uniform(0.97, 1.03) for c in f["currents"]]
    ok, row, op_time = do_open(ctx, t, "PROTECTION", None, cur, fault=f)
    if not ok:
        f["fault_on_ms"] += f["relay_s"][k] * SEC_MS + 400
        finalize(ctx, f, "UPSTREAM_CLEARED", t + 400)
        a.dead_from = t + 400
        a.dead_until = t + 400 + int(r.uniform(*cfg["outages"]["upstream_isolation_h"]) * HOUR_MS)
        out.schedule(a, a.dead_until, "isolate")
        maintenance.register_failure(ctx, t, "MECHANISM", "ALARM",
                                     "Trip Request Fail during a fault - backup protection cleared it")
        return
    f["fault_on_ms"] += f["relay_s"][k] * SEC_MS + op_time
    f["trips"] += 1
    if f["trips"] >= cfg["protection"]["trips_to_lockout"]:
        row["ar_state"] = "OPEN1_LOCKOUT"
        _lockout(ctx, f, t)
    else:
        row["ar_state"] = f"OPEN{f['trips'] + 1}"
        ctx.set_switch(t, "OPEN", "AR")
        dead = cfg["protection"]["reclose_times_s"][f["trips"] - 1]
        out.schedule(a, t + int(dead * SEC_MS), "reclose")


def h_reclose(ctx: Ctx, t: int, payload: dict) -> None:
    a, out = ctx.a, ctx.out
    f = a.active_fault
    if f is None:
        return
    if not ctx.running(t) or not ctx.line_live(t):
        if f["ops"]:
            f["ops"][-1]["ar_state"] = "OPEN1_LOCKOUT"
        _lockout(ctx, f, t)
        return
    ok, row, _ = do_close(ctx, t, "AUTO_RECLOSE", f"CLOSE{f['trips'] + 1}", fault=f)
    f["recloses"] += 1
    if not ok:
        _lockout(ctx, f, t)
        return
    ctx.set_switch(t, "CLOSED", None)
    if f["persist"] >= f["recloses"]:
        out.schedule(a, t + int(f["relay_s"][f["trips"]] * SEC_MS + a.to_base_ms), "trip")
    else:
        finalize(ctx, f, "CLEARED_BY_RECLOSE", t)


# ------------------------------------------------------------------ crew / manual
def _blocked_for_manual(ctx: Ctx) -> bool:
    a = ctx.a
    return a.in_maintenance or "MECHANISM" in a.failed or "CONTACTS" in a.failed


def h_crew_close(ctx: Ctx, t: int, payload: dict) -> None:
    a, r, out = ctx.a, ctx.r, ctx.out
    if a.switch == "CLOSED" or _blocked_for_manual(ctx):
        return
    if a.active_fault is not None or not ctx.running(t) or not ctx.line_live(t):
        out.schedule(a, t + 30 * MIN_MS, "crew_close", payload)
        return
    source = "SCADA" if a.connected and r.random() < 0.4 else "HMI"
    ok, _, _ = do_close(ctx, t, source, "CLOSE1")
    if not ok:
        attempt = payload.get("attempt", 0) + 1
        if attempt >= 2:
            maintenance.register_failure(ctx, t, "MECHANISM", "PATROL",
                                         "Close Request Fail - recloser cannot be closed after lockout")
        else:
            out.schedule(a, t + 15 * MIN_MS, "crew_close", {"attempt": attempt})
        return
    ctx.set_switch(t, "CLOSED", None)
    _clear_alerts(ctx, t)


def h_manual_open(ctx: Ctx, t: int, payload: dict) -> None:
    a, r, cfg, out = ctx.a, ctx.r, ctx.cfg, ctx.out
    if a.switch != "CLOSED" or a.active_fault is not None or a.failed or a.in_maintenance:
        return
    if not ctx.running(t) or not ctx.line_live(t):
        return
    source = payload.get("source") or rngmod.weighted_choice(r, cfg["switching"]["manual_sources"])
    if source in ("SCADA", "CMS") and not a.connected:
        source = "HMI"
    ok, _, _ = do_open(ctx, t, source, None, ctx.load_currents())
    if not ok:
        attempt = payload.get("attempt", 0) + 1
        if attempt >= 2:
            maintenance.register_failure(ctx, t, "MECHANISM", "SCADA",
                                         "Trip Request Fail - recloser did not open on command")
        else:
            out.schedule(a, t + 5 * MIN_MS, "manual_open", {"source": source, "attempt": attempt})
        return
    ctx.set_switch(t, "OPEN", "PLANNED")
    sc = cfg["switching"]
    out.schedule(a, t + _lognormal_ms(r, sc["manual_close_median_h"], sc["manual_close_sigma"]),
                 "manual_close", {"source": source})


def h_manual_close(ctx: Ctx, t: int, payload: dict) -> None:
    a, out = ctx.a, ctx.out
    if a.switch == "CLOSED" or _blocked_for_manual(ctx) or a.active_fault is not None:
        return
    if not ctx.running(t) or not ctx.line_live(t):
        out.schedule(a, t + 30 * MIN_MS, "manual_close", payload)
        return
    source = payload.get("source", "HMI")
    if source in ("SCADA", "CMS") and not a.connected:
        source = "HMI"
    ok, _, _ = do_close(ctx, t, source, "CLOSE1")
    if not ok:
        attempt = payload.get("attempt", 0) + 1
        if attempt >= 2:
            maintenance.register_failure(ctx, t, "MECHANISM", "SCADA",
                                         "Close Request Fail - recloser did not close on command")
        else:
            out.schedule(a, t + 5 * MIN_MS, "manual_close", {"source": source, "attempt": attempt})
        return
    ctx.set_switch(t, "CLOSED", None)


def h_mech_trip(ctx: Ctx, t: int, payload: dict) -> None:
    a, r, out = ctx.a, ctx.r, ctx.out
    if a.switch != "CLOSED" or a.active_fault is not None or a.failed or a.in_maintenance:
        return
    if not ctx.running(t) or not ctx.line_live(t):
        return
    do_open(ctx, t, "MECH_TRIP", None, ctx.load_currents(), force=True)
    out.raise_alarm(a, t, "Mechanically Locked", "WARNING", "Mechanical trip ring pulled")
    ctx.set_switch(t, "OPEN", "MECH")
    out.schedule(a, t + _lognormal_ms(r, 1.5, 0.6), "mech_reset")


def h_mech_reset(ctx: Ctx, t: int, payload: dict) -> None:
    a = ctx.a
    if a.open_reason != "MECH":
        return
    ctx.out.clear_alarm(a, t, "Mechanically Locked")
    ctx.out.schedule(a, t + int(ctx.r.uniform(1, 5) * MIN_MS), "manual_close", {"source": "HMI"})


def h_isolate(ctx: Ctx, t: int, payload: dict) -> None:
    """Crew isolates a recloser that failed to trip, using the mechanical trip ring."""
    a, out = ctx.a, ctx.out
    if a.switch == "CLOSED":
        if ctx.running(t):
            do_open(ctx, t, "MECH_TRIP", None, [0.0, 0.0, 0.0, 0.0], force=True)
        ctx.set_switch(t, "OPEN", "FAILED")
    out.raise_alarm(a, t, "Mechanically Locked", "WARNING", "Isolated after failed trip")
    a.dead_from = a.dead_until = None


def h_test_open(ctx: Ctx, t: int, payload: dict) -> None:
    a = ctx.a
    if a.switch != "CLOSED" or a.active_fault is not None or not ctx.running(t):
        return
    ok, _, _ = do_open(ctx, t, "HMI", None, ctx.load_currents())
    if ok:
        ctx.set_switch(t, "OPEN", "MAINT")
    else:
        maintenance.register_failure(ctx, t, "MECHANISM", "PATROL",
                                     "Trip failure during preventive maintenance test")


def h_test_close(ctx: Ctx, t: int, payload: dict) -> None:
    a = ctx.a
    if a.switch == "CLOSED" or a.open_reason != "MAINT" or not ctx.running(t):
        return
    ok, _, _ = do_close(ctx, t, "HMI", "CLOSE1")
    if ok:
        ctx.set_switch(t, "CLOSED", None)
    else:
        maintenance.register_failure(ctx, t, "MECHANISM", "PATROL",
                                     "Close failure during preventive maintenance test")


HANDLERS = {
    "fault": start_fault,
    "trip": h_trip,
    "reclose": h_reclose,
    "crew_close": h_crew_close,
    "manual_open": h_manual_open,
    "manual_close": h_manual_close,
    "mech_trip": h_mech_trip,
    "mech_reset": h_mech_reset,
    "isolate": h_isolate,
    "test_open": h_test_open,
    "test_close": h_test_close,
    "maintenance": lambda ctx, t, p: maintenance.h_maintenance(ctx, t, p),
    "osm_swap_end": lambda ctx, t, p: maintenance.h_osm_swap_end(ctx, t, p),
    "battery_test": lambda ctx, t, p: maintenance.battery_test(ctx, t, p.get("trigger", "MANUAL")),
}
