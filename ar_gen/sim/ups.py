"""RC cubicle UPS: battery charge/discharge, external load timer, standby and shutdown (manual 4.8).

The 15-minute interval is split at every supply change, timer expiry and threshold
crossing, so state of charge, averages and alarm times are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .battery import battery_rint, calendar_soh, soc_at_ocv, terminal_voltage
from .environment import AreaCond
from .out import Out
from .state import AssetState
from .timeutil import HOUR_MS, MIN_MS, SEC_MS

AC_OFF = "AC Off (On Battery Supply)"
CRITICAL = "Critical Battery Level"
ABNORMAL = "Battery Status Abnormal"


def soh(a: AssetState, bcfg: dict) -> float:
    cal = calendar_soh(a.age_eff_days, a.battery_interval_y, bcfg["calendar_fade_coeff"])
    return min(1.05, max(0.05, a.init_cap * cal - a.cycle_loss - a.extra_loss))


def _in(t: int, s, e) -> bool:
    return s is not None and e is not None and s <= t < e


@dataclass
class UpsResult:
    outage: tuple
    dead: tuple
    aux: tuple
    running_segments: list = field(default_factory=list)
    power_events: list = field(default_factory=list)     # (t, modem_on)
    modem_on_start: bool = True
    v_avg: float | None = None
    i_avg: float | None = None
    v_last: float | None = None
    i_last: float | None = None

    def line_live(self, t: int) -> bool:
        return not _in(t, *self.outage) and not _in(t, *self.dead)

    def ac_avail(self, t: int) -> bool:
        return self.line_live(t) and not _in(t, *self.aux)

    def running(self, t: int) -> bool:
        return any(s <= t < e for s, e in self.running_segments)

    def live_fraction(self, t0: int, t1: int) -> float:
        edges = sorted({t0, t1, *[x for w in (self.outage, self.dead) for x in w
                                   if x is not None and t0 < x < t1]})
        live = sum(b - a for a, b in zip(edges, edges[1:]) if self.line_live(a))
        return live / (t1 - t0)


class _Ups:
    def __init__(self, a: AssetState, cfg: dict, cond: AreaCond, t0: int, t1: int, out: Out):
        self.a, self.cfg, self.b, self.out = a, cfg, cfg["battery"], out
        self.t0, self.t1 = t0, t1
        self.res = UpsResult(outage=(cond.outage_start, cond.outage_until),
                             dead=(a.dead_from, a.dead_until), aux=(a.aux_from, a.aux_until))
        self.soh = soh(a, self.b)
        self.cap = self.b["rated_ah"] * self.soh
        self.rint = battery_rint(self.soh, self.b["r0_ohm"], self.b["r_soh_exponent"])
        self.offset = -self.b["shorted_cell_ocv_drop_v"] if a.shorted else 0.0
        self.ext_ms = a.ext_load_time_min * MIN_MS
        self.v_acc = self.i_acc = self.time_acc = 0.0
        self.run_start = t0 if self._running() else None
        self.modem = self._modem_on()
        self.res.modem_on_start = self.modem

    # ------------------------------------------------------------ helpers
    def _running(self) -> bool:
        return self.a.ctrl in ("RUN", "STANDBY")

    def _modem_on(self) -> bool:
        return self.a.ctrl == "RUN" and self.a.ext_on

    def _state_changed(self, t: int) -> None:
        running = self._running()
        if running and self.run_start is None:
            self.run_start = t
        elif not running and self.run_start is not None:
            if t > self.run_start:
                self.res.running_segments.append((self.run_start, t))
            self.run_start = None
        m = self._modem_on()
        if m != self.modem:
            self.res.power_events.append((t, m))
            self.modem = m

    def _ext_should_be_on(self, t: int) -> bool:
        a = self.a
        if a.ext_overload_until is not None or a.ctrl != "RUN":
            return False
        if a.ac_on or self.ext_ms == 0:
            return True
        return a.ac_off_since is not None and t < a.ac_off_since + self.ext_ms

    # ------------------------------------------------------------ transitions
    def ac_change(self, t: int, ac: bool) -> None:
        a, out = self.a, self.out
        if not ac:
            a.ac_on = False
            a.ac_off_since = t
            a.dis_start_soc = a.soc
            a.dis_min_soc = a.soc
            a.deep_penalized = False
            if self._running():
                out.raise_alarm(a, t, AC_OFF, "WARNING")
            elif a.ctrl == "BOOT" and a.soc < self.b["shutdown_level_pct"] / 100.0 + 0.05:
                a.ctrl, a.boot_at = "OFF", None
        else:
            a.ac_on = True
            a.ac_off_since = None
            if a.dis_start_soc is not None:
                dod = max(0.0, a.dis_start_soc - a.dis_min_soc)
                a.cycle_loss += self.b["cycle_fade_per_full_dod"] * dod ** self.b["cycle_fade_exponent"]
                a.dis_start_soc = None
            if self._running():
                out.clear_alarm(a, t, AC_OFF)
                out.clear_alarm(a, t, CRITICAL)
                a.ctrl, a.standby_since = "RUN", None
            elif a.ctrl == "OFF":
                a.ctrl = "BOOT"
                a.boot_at = t + int(self.b["boot_time_s"] * SEC_MS)
        a.ext_on = self._ext_should_be_on(t)
        self._state_changed(t)

    def process_due(self, t: int) -> None:
        a, out = self.a, self.out
        changed = True
        while changed:
            changed = False
            if a.ctrl == "BOOT" and a.boot_at is not None and a.boot_at <= t:
                tb = max(a.boot_at, self.t0)
                a.ctrl, a.boot_at = "RUN", None
                out.event(a, tb, "Power Restart", "WARNING", "Battery/AC restored")
                if a.ac_on:
                    out.clear_alarm(a, tb, AC_OFF)
                out.clear_alarm(a, tb, CRITICAL)
                a.ext_on = self._ext_should_be_on(tb)
                self._state_changed(tb)
                changed = True
            if a.ctrl == "STANDBY" and a.standby_since + self.b["standby_to_powerdown_min"] * MIN_MS <= t:
                self.power_down(a.standby_since + self.b["standby_to_powerdown_min"] * MIN_MS)
                changed = True
            if a.ext_overload_until is not None and a.ext_overload_until <= t:
                te = max(a.ext_overload_until, self.t0)
                a.ext_overload_until = None
                a.ext_on = self._ext_should_be_on(te)
                self._state_changed(te)
                changed = True
            if a.ext_on and not a.ac_on and self.ext_ms > 0 and a.ac_off_since is not None \
                    and a.ac_off_since + self.ext_ms <= t:
                a.ext_on = False
                self._state_changed(max(a.ac_off_since + self.ext_ms, self.t0))
                changed = True

    def next_event(self, t: int, limit: int) -> int:
        a = self.a
        cands = [limit]
        if a.ctrl == "BOOT" and a.boot_at is not None:
            cands.append(a.boot_at)
        if a.ctrl == "STANDBY":
            cands.append(a.standby_since + self.b["standby_to_powerdown_min"] * MIN_MS)
        if a.ext_overload_until is not None:
            cands.append(a.ext_overload_until)
        if a.ext_on and not a.ac_on and self.ext_ms > 0 and a.ac_off_since is not None:
            cands.append(a.ac_off_since + self.ext_ms)
        return min(c for c in cands if c > t) if any(c > t for c in cands) else limit

    def standby(self, t: int) -> None:
        a = self.a
        a.ctrl, a.standby_since = "STANDBY", t
        self.out.raise_alarm(a, t, CRITICAL, "WARNING", f"{a.soc * 100:.0f} %")
        a.ext_on = False
        self._state_changed(t)

    def power_down(self, t: int) -> None:
        a = self.a
        a.ctrl, a.standby_since, a.boot_at = "OFF", None, None
        self.out.event(a, t, "Shutdown", "WARNING", "Power Supply")
        if a.ac_off_since is not None:
            a.shutdown_hours = (t - a.ac_off_since) / HOUR_MS
        if not a.deep_penalized:
            a.cycle_loss += self.b["deep_discharge_penalty"]
            a.deep_penalized = True
        a.ext_on = False
        self._state_changed(t)

    # ------------------------------------------------------------ integration
    def integrate(self, t: int, te: int) -> int:
        """Advance the battery from t to te; returns where it stopped (earlier on a threshold)."""
        a, b = self.a, self.b
        dt_h = (te - t) / HOUR_MS
        temp = a.batt_temp
        mode, bulk, current = "REST", 0.0, 0.0
        if a.ctrl == "OFF":
            current = -b["parasitic_off_a"]
            a.soc = max(0.0, a.soc - b["parasitic_off_a"] * dt_h / self.cap)
        elif a.ac_on:
            if a.charger_fault:
                current = -0.01
                a.soc = max(0.0, a.soc - b["self_discharge_per_day"] * dt_h / 24.0)
            else:
                i_float = b["float_current_a"] + b["float_current_aging_a"] * max(0.0, 1.0 - self.soh)
                i_taper = min(b["charge_current_max_a"], b["charge_taper"] * (1.0 - a.soc) * self.cap) \
                    if a.soc < 0.9995 else 0.0
                current = max(i_float, i_taper)
                a.soc = min(1.0, a.soc + i_taper * b["charge_efficiency"] * dt_h / self.cap)
                if i_taper > 1.5 * i_float:
                    mode, bulk = "BULK", i_taper / b["charge_current_max_a"]
                else:
                    mode = "FLOAT"
        else:
            load = a.ctrl_load_a + (a.ext_load_a if a.ext_on else 0.0)
            current = -load
            mode = "DISCHARGE"
            if a.ctrl == "RUN":
                s_th = max(b["shutdown_level_pct"] / 100.0,
                           soc_at_ocv(b["power_down_v"] - self.offset + load * self.rint))
                drop = load * dt_h / self.cap
                if a.soc - drop <= s_th:
                    t_cross = t + max(0, int((a.soc - s_th) * self.cap / load * HOUR_MS))
                    t_cross = min(t_cross, te)
                    a.soc = max(0.0, min(a.soc, s_th))
                    self._accumulate(t, t_cross, current, mode, bulk, temp)
                    a.dis_min_soc = min(a.dis_min_soc, a.soc)
                    self.standby(t_cross)
                    return t_cross
            a.soc = max(0.0, a.soc - load * dt_h / self.cap)
            a.dis_min_soc = min(a.dis_min_soc, a.soc)
        v = self._accumulate(t, te, current, mode, bulk, temp)
        if self._running() and v is not None and v < b["shutdown_v"]:
            self.power_down(te)
        return te

    def _accumulate(self, t, te, current, mode, bulk, temp):
        if not self._running() or te <= t:
            return None
        b = self.b
        v = terminal_voltage(self.a.soc, current, self.rint, mode, temp, bulk, b["float_voltage_25c"],
                             b["bulk_voltage_boost"], b["temp_comp_v_per_c"], self.offset)
        dt = te - t
        self.v_acc += v * dt
        self.i_acc += current * dt
        self.time_acc += dt
        self.res.v_last = v
        self.res.i_last = current
        # Battery Status Abnormal (low / shorted cell)
        a, out = self.a, self.out
        if (v < b["status_abnormal_low_v"] or a.shorted) and ABNORMAL not in a.alarms:
            out.raise_alarm(a, te, ABNORMAL, "WARNING", f"Low {v:.1f} V")
        elif ABNORMAL in a.alarms and v >= b["status_abnormal_clear_v"] and not a.shorted:
            out.clear_alarm(a, te, ABNORMAL)
        return v

    def run(self) -> UpsResult:
        a, t0, t1 = self.a, self.t0, self.t1
        sub = int(self.b["substep_minutes"] * MIN_MS)
        edges = {t0, t1, *range(t0 + sub, t1, sub)}
        for w in (self.res.outage, self.res.dead, self.res.aux):
            for x in w:
                if x is not None and t0 < x < t1:
                    edges.add(x)
        bounds = sorted(edges)
        for ta, tb in zip(bounds, bounds[1:]):
            ac = self.res.ac_avail(ta)
            if ac != a.ac_on:
                self.ac_change(ta, ac)
            t = ta
            while t < tb:
                self.process_due(t)
                te = self.next_event(t, tb)
                t = self.integrate(t, te)
        self.process_due(t1)
        if self.run_start is not None and t1 > self.run_start:
            self.res.running_segments.append((self.run_start, t1))
        if self.time_acc > 0:
            self.res.v_avg = self.v_acc / self.time_acc
            self.res.i_avg = self.i_acc / self.time_acc
        return self.res


def run_ups(a: AssetState, cfg: dict, cond: AreaCond, t0: int, t1: int, out: Out) -> UpsResult:
    return _Ups(a, cfg, cond, t0, t1, out).run()
