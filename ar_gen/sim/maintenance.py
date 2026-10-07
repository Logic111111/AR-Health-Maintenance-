"""Failures, corrective and preventive maintenance, battery tests (manual 4.8.2, 10)."""
from __future__ import annotations

import math
from datetime import timedelta

from . import rng as rngmod
from .battery import battery_rint, ocv
from .timeutil import DAY_MS, HOUR_MS, MIN_MS, SEC_MS, date_str, local_date, local_to_ms, parse_date

MODULE_SIGNALS_REPAIRED = ("SIM Module Fault", "Relay Module Fault", "CAN Bus Malfunction",
                           "Battery Charger Fault", "Module Comms Error", "SIM Comms Error")


# ------------------------------------------------------------------ helpers
def work_time(ctx, t: int) -> int:
    """A random time inside local working hours on the day of ``t`` (or the next day)."""
    lo, hi = ctx.cfg["maintenance"]["work_hours_local"]
    d = local_date(t, ctx.tz)
    for _ in range(3):
        cand = local_to_ms(d, ctx.r.uniform(lo, hi - 1), ctx.tz)
        if cand > t:
            return cand
        d = d + timedelta(days=1)
    return t + HOUR_MS


def latest_firmware(cfg: dict, t_date) -> str:
    valid = [x["version"] for x in cfg["firmware_releases"] if parse_date(str(x["release_date"])) <= t_date]
    return valid[-1] if valid else cfg["firmware_releases"][0]["version"]


def _reset_onset(ctx, proc: str, t: int) -> None:
    a, r, cfg = ctx.a, ctx.r, ctx.cfg
    years = r.expovariate(1.0 / cfg["maintenance"]["next_onset_mean_years"][proc])
    a.onsets[proc] = t + int(years * 365.25 * DAY_MS)


def register_failure(ctx, t: int, component: str, detected_by: str, description: str):
    a, out, cfg = ctx.a, ctx.out, ctx.cfg
    if component in a.failed:
        return None
    fid = out.failure(a, t, component, detected_by, description)
    a.failed[component] = fid
    lo, hi = cfg["maintenance"]["corrective_delay_days"][component]
    t_fix = work_time(ctx, t + int(ctx.r.uniform(lo, hi) * DAY_MS))
    out.schedule(a, t_fix, "maintenance", {"kind": "CORRECTIVE", "component": component})
    return fid


# ------------------------------------------------------------------ periodic checks
def check_failures(ctx) -> None:
    """Called at the end of every interval: hidden thresholds -> failure_log, PM scheduling."""
    a, cfg, t1 = ctx.a, ctx.cfg, ctx.t1
    b = cfg["battery"]
    if a.consecutive_failed >= b["failed_tests_for_failure"]:
        register_failure(ctx, t1, "BATTERY", "BATTERY_TEST",
                         f"{a.consecutive_failed} consecutive battery tests failed")
    ab = a.alarms.get("Battery Status Abnormal")
    if ab and t1 - ab["since"] > 24 * HOUR_MS:
        register_failure(ctx, t1, "BATTERY", "ALARM", "Battery Status Abnormal for over 24 h (suspected shorted cell)")
    if a.shutdown_hours is not None:
        expected_h = 0.8 * b["rated_ah"] / (a.ctrl_load_a + a.ext_load_a)
        if a.shutdown_hours < b["shutdown_failure_autonomy_fraction"] * expected_h:
            register_failure(ctx, t1, "BATTERY", "ALARM",
                             f"Controller shut down after only {a.shutdown_hours:.1f} h on battery")
        a.shutdown_hours = None
    if a.mech_deg >= cfg["switching"]["mech_failure_deg"]:
        register_failure(ctx, t1, "MECHANISM", "PATROL", "Excessive operating times - mechanism worn")
    if max(a.wear) >= 100.0:
        register_failure(ctx, t1, "CONTACTS", "SCADA",
                         "Contact wear reached 100 % - vacuum interrupter end of life")
    c = cfg["comms"]
    if a.comms_deg >= 1.0 and not a.connected and a.disconnected_since is not None \
            and t1 - a.disconnected_since > c["comms_failure_disconnected_h"] * HOUR_MS:
        register_failure(ctx, t1, "COMMS", "SCADA", "Communication lost for over 12 h - modem/antenna failure")
    if a.persistent_fault:
        register_failure(ctx, t1, "CONTROLLER", "ALARM", f"Persistent {a.persistent_fault}")
    if a.charger_fault and a.charger_fault_since is not None and \
            t1 - a.charger_fault_since > cfg["controller"]["charger_fault_failure_h"] * HOUR_MS:
        register_failure(ctx, t1, "CONTROLLER", "ALARM", "Battery charger fault (SIM module)")
    if a.next_pm_ms is not None and t1 >= a.next_pm_ms and not a.in_maintenance:
        ctx.out.schedule(a, work_time(ctx, t1), "maintenance", {"kind": "PREVENTIVE"})
        a.next_pm_ms = None


# ------------------------------------------------------------------ battery
def battery_test(ctx, t: int, trigger: str) -> None:
    from .ups import soh as soh_of
    a, out, cfg = ctx.a, ctx.out, ctx.cfg
    b, bt = cfg["battery"], cfg["battery"]["test"]
    if not ctx.running(t):
        return
    reason = None
    if a.last_test_ms is not None and t - a.last_test_ms < 5 * MIN_MS:
        reason = "RESTING"
    elif not a.ac_on:
        reason = "AC_OFF"
    elif a.charger_fault and a.soc < 0.97:
        reason = "DISCHARGING"
    offset = -b["shorted_cell_ocv_drop_v"] if a.shorted else 0.0
    v_oc = ocv(a.soc) + offset
    if reason is None and v_oc + bt["surface_charge_v"] < bt["min_voltage_v"]:
        reason = "LOW_VOLTAGE"
    a.last_test_ms = t
    if reason is not None:
        out.battery_test(a, t, trigger, "NOT_PERFORMED", reason)
        return
    s = soh_of(a, b)
    rint = battery_rint(s, b["r0_ohm"], b["r_soh_exponent"])
    pol0, pol_extra = bt["polarization_v"]
    v_before = v_oc + bt["surface_charge_v"] + ctx.r.gauss(0.0, 0.01)
    v_min = v_oc - bt["load_a"] * rint - (pol0 + pol_extra * max(0.0, 1.0 - min(1.0, s))) \
        + ctx.r.gauss(0.0, bt["noise_v"])
    v_min = min(v_min, v_before - 0.05)
    cap = b["rated_ah"] * s
    a.soc = max(0.0, a.soc - bt["load_a"] * bt["duration_s"] / 3600.0 / cap)
    if v_min < bt["fail_below_v"]:
        out.battery_test(a, t, trigger, "FAILED", None, v_before, v_min, bt["load_a"])
        a.consecutive_failed += 1
        out.raise_alarm(a, t, "Check Battery", "WARNING", f"Test min {v_min:.2f} V")
    else:
        out.battery_test(a, t, trigger, "PASSED", None, v_before, v_min, bt["load_a"])
        a.consecutive_failed = 0
        out.clear_alarm(a, t, "Check Battery")


def replace_battery(ctx, t: int, maint_type: str, failure_id, description: str) -> None:
    a, out, cfg, r = ctx.a, ctx.out, ctx.cfg, ctx.r
    b = cfg["battery"]
    model = rngmod.weighted_choice(r, b["replacement_model_mix"])
    new_date = date_str(local_date(t, ctx.tz))
    changes = {
        "battery_model": [a.battery_model, model],
        "battery_install_date": [a.battery_install, new_date],
        "battery_service_interval_years": [a.battery_interval_y, b["models"][model]["service_interval_years"]],
        "last_maintenance_date": [a.last_maint, new_date],
    }
    out.raise_alarm(a, t, "Battery Off (On AC Supply)", "WARNING", "Battery replacement",
                    clear_at=t + 10 * MIN_MS)
    a.battery_model = model
    a.battery_interval_y = int(b["models"][model]["service_interval_years"])
    a.battery_install = new_date
    a.last_maint = new_date
    a.init_cap = min(1.05, max(0.9, r.gauss(1.0, b["capacity_tolerance_sigma"])))
    a.age_eff_days = a.cycle_loss = a.extra_loss = 0.0
    a.shorted = False
    a.soc = 0.9
    a.consecutive_failed = 0
    a.dis_start_soc = None
    a.dis_min_soc = a.soc
    for sig in ("Check Battery", "Battery Status Abnormal"):
        out.clear_alarm(a, t, sig)
    if a.ctrl == "OFF":
        a.ctrl, a.boot_at = "BOOT", t + int(b["boot_time_s"] * SEC_MS)
    _reset_onset(ctx, "BATTERY_DEGRADING", t)
    a.next_test_ms = t + int(b["test"]["interval_days"] * DAY_MS)
    out.schedule(a, t + 15 * MIN_MS, "battery_test", {"trigger": "MANUAL"})
    out.maintenance(a, t, "BATTERY", maint_type, description, failure_id, changes)
    if failure_id is not None:
        a.failed.pop("BATTERY", None)


# ------------------------------------------------------------------ maintenance events
def h_maintenance(ctx, t: int, payload: dict) -> None:
    if payload.get("kind") == "PREVENTIVE":
        _preventive(ctx, t)
        return
    a, out = ctx.a, ctx.out
    comp = payload["component"]
    fid = a.failed.get(comp)
    if fid is None:
        return
    if comp == "BATTERY":
        replace_battery(ctx, t, "CORRECTIVE", fid, "Battery replaced after failure")
    elif comp in ("MECHANISM", "CONTACTS"):
        _start_osm_swap(ctx, t)
    elif comp == "COMMS":
        a.comms_deg = 0.0
        if not a.connected:
            a.reconnect_at, a.restart_at = t + 3 * MIN_MS, None
        _reset_onset(ctx, "COMMS_PROBLEM", t)
        out.maintenance(a, t, "COMMS", "CORRECTIVE", "Modem and antenna replaced", fid,
                        {"last_maintenance_date": [a.last_maint, date_str(local_date(t, ctx.tz))]})
        a.last_maint = date_str(local_date(t, ctx.tz))
        a.failed.pop("COMMS", None)
    elif comp == "CONTROLLER":
        for sig in [s for s, info in a.alarms.items() if info["cat"] == "MALFUNCTION"]:
            out.clear_alarm(a, t, sig)
        a.ctrl_deg = 0.0
        a.persistent_fault = None
        a.charger_fault, a.charger_fault_since = False, None
        new_fw = latest_firmware(ctx.cfg, local_date(t, ctx.tz))
        today = date_str(local_date(t, ctx.tz))
        changes = {"last_maintenance_date": [a.last_maint, today]}
        if new_fw != a.firmware:
            changes["firmware_version"] = [a.firmware, new_fw]
        a.firmware, a.last_maint = new_fw, today
        out.event(a, t + 2 * MIN_MS, "Power Restart", "WARNING", "Module replacement")
        _reset_onset(ctx, "CONTROLLER_PROBLEM", t)
        out.maintenance(a, t, "CONTROLLER", "CORRECTIVE", "Relay/SIM module replaced", fid, changes)
        a.failed.pop("CONTROLLER", None)


def _start_osm_swap(ctx, t: int) -> None:
    from .ops import do_open
    a, out, r = ctx.a, ctx.out, ctx.r
    if a.in_maintenance:
        return
    if a.active_fault is not None:
        out.schedule(a, t + HOUR_MS, "maintenance",
                     {"kind": "CORRECTIVE", "component": "MECHANISM" if "MECHANISM" in a.failed else "CONTACTS"})
        return
    a.in_maintenance = True
    if a.switch == "CLOSED":
        if ctx.running(t):
            source = "MECH_TRIP" if "MECHANISM" in a.failed else "HMI"
            do_open(ctx, t, source, None, ctx.load_currents(), force=True)
        ctx.set_switch(t, "OPEN", "MAINT")
    else:
        a.open_reason = "MAINT"
    out.raise_alarm(a, t + 2 * MIN_MS, "OSM Disconnected", "WARNING", "OSM tank replacement")
    lo, hi = ctx.cfg["maintenance"]["osm_replacement_hours"]
    out.schedule(a, t + int(r.uniform(lo, hi) * HOUR_MS), "osm_swap_end")


def h_osm_swap_end(ctx, t: int, payload: dict) -> None:
    a, out, r, cfg = ctx.a, ctx.out, ctx.r, ctx.cfg
    today = local_date(t, ctx.tz)
    manufactured = today - timedelta(days=r.randint(30, 180))
    code = cfg["models"][a.model]["serial_code"]
    serial = f"0{code}{manufactured.year % 100:02d}{manufactured.month:02d}{r.randint(0, 9_999_999):07d}"
    changes = {
        "osm_serial_number": [a.serial, serial],
        "manufactured_date": [a.manufactured, date_str(manufactured)],
        "installed_date": [a.installed, date_str(today)],
        "last_maintenance_date": [a.last_maint, date_str(today)],
    }
    old_counters = {"co_total": a.co_total, "contact_wear_pct": [round(w, 3) for w in a.wear]}
    a.serial, a.manufactured, a.installed, a.last_maint = serial, date_str(manufactured), date_str(today), date_str(today)
    a.co_total, a.wear = 0, [0.0, 0.0, 0.0]
    a.mech_deg = 0.01
    s = cfg["switching"]
    a.to_base_ms, a.tc_base_ms = r.uniform(*s["open_time_ms"]), r.uniform(*s["close_time_ms"])
    for sig in ("OSM Disconnected", "Mechanically Locked"):
        out.clear_alarm(a, t, sig)
    for sig in [x for x, info in a.alarms.items() if info["cat"] == "ALERT"]:
        out.clear_alarm(a, t, sig)
    first = True
    for comp in ("MECHANISM", "CONTACTS"):
        fid = a.failed.pop(comp, None)
        if fid is None:
            continue
        desc = ("OSM tank replaced (mechanism failure)" if comp == "MECHANISM"
                else "OSM tank replaced (vacuum interrupter contacts worn out)")
        out.maintenance(a, t, comp, "CORRECTIVE", desc, fid, changes if first else None,
                        {"counters_reset": True, "old_counters": old_counters})
        first = False
    if first:  # no open failure (should not happen) - still record the swap
        out.maintenance(a, t, "MECHANISM", "CORRECTIVE", "OSM tank replaced", None, changes,
                        {"counters_reset": True, "old_counters": old_counters})
    _reset_onset(ctx, "MECHANISM_WEAR", t)
    a.in_maintenance = False
    out.schedule(a, t + 5 * MIN_MS, "manual_close", {"source": "HMI"})


def _preventive(ctx, t: int) -> None:
    a, out, r, cfg = ctx.a, ctx.out, ctx.r, ctx.cfg
    m = cfg["maintenance"]
    today = local_date(t, ctx.tz)
    if a.switch == "CLOSED" and a.active_fault is None and not a.failed and ctx.running(t):
        for j in range(m["pm_test_operations"]):
            out.schedule(a, t + (2 * j + 1) * MIN_MS, "test_open")
            out.schedule(a, t + (2 * j + 2) * MIN_MS, "test_close")
    out.schedule(a, t + 10 * MIN_MS, "battery_test", {"trigger": "MANUAL"})
    a.pollution = min(a.pollution, 0.02)
    out.maintenance(a, t, "MECHANISM", "PREVENTIVE",
                    "Annual inspection: test operations, bushings cleaned, door seal checked", None,
                    {"last_maintenance_date": [a.last_maint, date_str(today)]})
    a.last_maint = date_str(today)
    batt_age_y = (today - parse_date(a.battery_install)).days / 365.25
    if batt_age_y >= a.battery_interval_y and r.random() < m["battery_replace_compliance"] \
            and "BATTERY" not in a.failed:
        replace_battery(ctx, t + 20 * MIN_MS, "PREVENTIVE", None,
                        f"Battery replaced at service interval (age {batt_age_y:.1f} y)")
    if r.random() < m["firmware_upgrade_prob"]:
        new_fw = latest_firmware(cfg, today)
        if new_fw != a.firmware:
            out.maintenance(a, t + 30 * MIN_MS, "CONTROLLER", "PREVENTIVE", "Firmware upgraded", None,
                            {"firmware_version": [a.firmware, new_fw]})
            a.firmware = new_fw
    jitter = r.randint(-m["preventive_jitter_days"], m["preventive_jitter_days"])
    a.next_pm_ms = t + (m["preventive_interval_days"] + jitter) * DAY_MS
