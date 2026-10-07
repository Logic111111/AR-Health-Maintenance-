"""Simulation engine: advances the whole fleet one 15-minute interval at a time.

The order inside one interval (t0, t1] for every recloser is:
environment -> hidden degradation -> auxiliary supply -> UPS -> load -> controller
malfunctions -> random events -> queued events (protection, switching, maintenance)
-> comms -> automatic battery test -> alarm clears -> failure checks -> telemetry poll at t1.
"""
from __future__ import annotations

import heapq
import math
from typing import Iterator

from . import comms, controller, maintenance
from . import rng as rngmod
from .battery import battery_rint, ocv, temperature_age_factor, terminal_voltage
from .environment import step_area
from .fleet import asset_update_row
from .load import load_current, phase_currents, residual_current
from .ops import HANDLERS, Ctx
from .out import Chunk, Out
from .state import DEGRADATION_PROCESSES, AssetState, SimState
from .timeutil import DAY_MS, HOUR_MS, INTERVALS_PER_YEAR, MIN_MS, SEC_MS, date_str, local_date, ms_to_dt, to_local
from .ups import run_ups, soh

CUBICLE_TAU_MIN = 40.0
BATTERY_TAU_MIN = 120.0
CUBICLE_SELF_HEATING_C = 2.0


def _rand_t(r, t0: int, t1: int) -> int:
    return t0 + 1 + int(r.random() * (t1 - t0 - 1))


class Simulator:
    def __init__(self, cfg: dict, state: SimState):
        self.cfg = cfg
        self.state = state
        self.tz = cfg["timezone"]
        self.step_ms = int(cfg["interval_minutes"]) * MIN_MS

    # ------------------------------------------------------------ public API
    def iter_chunks(self, end_ms: int, chunk_intervals: int = 96) -> Iterator[Chunk]:
        """Simulate up to ``end_ms`` (exclusive of partial intervals), yielding row chunks."""
        st = self.state
        while st.last_ms + self.step_ms <= end_ms:
            out = Out(st)
            n = 0
            while n < chunk_intervals and st.last_ms + self.step_ms <= end_ms:
                t0 = st.last_ms
                self.step_interval(out, t0, t0 + self.step_ms)
                st.last_ms = t0 + self.step_ms
                n += 1
            yield out.finish(asset_update_row)

    def step_interval(self, out: Out, t0: int, t1: int) -> None:
        cfg, st = self.cfg, self.state
        k = t1 // self.step_ms
        local_mid = to_local((t0 + t1) // 2, self.tz)
        conds = [step_area(area, cfg, t0, t1, local_mid, k) for area in st.areas]
        for a in st.assets:
            r = rngmod.rng(cfg["seed"], rngmod.ASSET_INTERVAL, a.idx, k)
            self._step_asset(out, a, r, conds[a.area_idx], t0, t1)
        if cfg.get("ground_truth_daily", True):
            today = date_str(local_date(t1, self.tz))
            if today != st.last_gt_date:
                for a in st.assets:
                    out.ground_truth.append(self._ground_truth(a, st.last_gt_date, t1))
                st.last_gt_date = today

    # ------------------------------------------------------------ one asset
    def _step_asset(self, out: Out, a: AssetState, r, cond, t0: int, t1: int) -> None:
        cfg = self.cfg
        self._degrade(a, r, cond, t0, t1)
        self._aux_supply(a, r, cond, t0, t1)
        ups = run_ups(a, cfg, cond, t0, t1, out)
        i_avg = load_current(a, cfg, cond.local_hour, cond.weekday, cond.ambient_c, r)
        i_phase = phase_currents(a, i_avg, cfg, r)
        ctx = Ctx(cfg, self.tz, out, a, r, ups, cond, t0, t1, i_phase)

        controller.step(ctx)
        self._random_events(ctx)
        while a.queue and a.queue[0][0] <= t1:
            t, _, kind, payload = heapq.heappop(a.queue)
            HANDLERS[kind](ctx, t, payload)
        sig = comms.step(ctx)

        bt = cfg["battery"]["test"]
        if bt["auto_enabled"] and a.next_test_ms <= t1:
            maintenance.battery_test(ctx, max(a.next_test_ms, t0 + 1), "AUTO")
            while a.next_test_ms <= t1:
                a.next_test_ms += int(bt["interval_days"] * DAY_MS)

        out.flush_alarm_clears(a, t1)
        maintenance.check_failures(ctx)
        if a.ctrl in ("RUN", "STANDBY"):
            out.telemetry(self._telemetry(ctx, i_avg, sig))

    # ------------------------------------------------------------ hidden processes
    def _degrade(self, a: AssetState, r, cond, t0: int, t1: int) -> None:
        cfg = self.cfg
        b, sc, pc = cfg["battery"], cfg["switching"], cfg["protection"]
        dt_min = (t1 - t0) / MIN_MS
        dt_days = dt_min / 1440.0

        target = cond.ambient_c + a.solar_gain * cond.sun + CUBICLE_SELF_HEATING_C
        a.cub_temp += (target - a.cub_temp) * (1.0 - math.exp(-dt_min / CUBICLE_TAU_MIN))
        a.batt_temp += (a.cub_temp - a.batt_temp) * (1.0 - math.exp(-dt_min / BATTERY_TAU_MIN))

        a.age_eff_days += dt_days * temperature_age_factor(a.batt_temp, b["temp_doubling_c"])
        on = {p: a.onsets.get(p, 1 << 62) <= t1 for p in DEGRADATION_PROCESSES}
        if on["BATTERY_DEGRADING"]:
            a.extra_loss += a.rates["BATTERY_DEGRADING"] * dt_days
            if not a.shorted and soh(a, b) < 0.6 and r.random() < b["shorted_cell_prob_per_day"] * dt_days:
                a.shorted = True
        a.mech_deg += sc["mech_deg_initial_per_year"] / 365.25 * dt_days
        if on["MECHANISM_WEAR"]:
            a.mech_deg += a.rates["MECHANISM_WEAR"] * dt_days
        if on["COMMS_PROBLEM"]:
            a.comms_deg += a.rates["COMMS_PROBLEM"] * dt_days
        if on["CONTROLLER_PROBLEM"]:
            a.ctrl_deg += a.rates["CONTROLLER_PROBLEM"] * dt_days

        growth = pc["pollution_growth_per_day"]["coastal" if a.coastal else "inland"]
        a.pollution += growth * (2.0 if a.harsh else 1.0) * dt_days
        a.pollution *= pc["pollution_wash"].get(cond.weather, 1.0)
        a.pollution = min(1.0, max(0.0, a.pollution))

    def _aux_supply(self, a: AssetState, r, cond, t0: int, t1: int) -> None:
        o = self.cfg["outages"]
        if a.aux_until is not None and a.aux_until <= t0:
            a.aux_from = a.aux_until = None
        if a.dead_until is not None and a.dead_until <= t0:
            a.dead_from = a.dead_until = None
        u_aux, u_surge = r.random(), r.random()
        if a.aux_until is not None:
            return
        if u_aux < o["local_aux_rate_per_year"] / INTERVALS_PER_YEAR:
            start = _rand_t(r, t0, t1)
            hours = min(o["local_aux_duration_cap_h"],
                        o["local_aux_duration_median_h"] * math.exp(r.gauss(0.0, o["local_aux_duration_sigma"])))
            a.aux_from, a.aux_until = start, start + max(MIN_MS, int(hours * HOUR_MS))
        elif cond.weather == "STORM" and u_surge < o["surge_prob_per_storm_interval"]:
            start = _rand_t(r, t0, t1 - 30 * SEC_MS)
            a.aux_from, a.aux_until = start, start + int(r.uniform(*o["surge_duration_s"]) * SEC_MS)

    def _random_events(self, ctx: Ctx) -> None:
        a, r, cfg, out, cond = ctx.a, ctx.r, ctx.cfg, ctx.out, ctx.cond
        t0, t1 = ctx.t0, ctx.t1
        pc, sc = cfg["protection"], cfg["switching"]
        rate = pc["fault_rate_per_year"] * a.exposure * pc["weather_mult"][cond.weather]
        if a.coastal:
            rate *= pc["coastal_mult"]
        rate *= 1.0 + pc["pollution_flashover_gain"] * a.pollution ** 2
        for _ in range(rngmod.poisson(r, rate / INTERVALS_PER_YEAR)):
            ftype = rngmod.weighted_choice(r, pc["type_share"])
            out.schedule(a, _rand_t(r, t0, t1), "fault", {"type": ftype})
        day = 8.0 <= cond.local_hour < 17.0
        if r.random() < sc["manual_ops_per_year"] / INTERVALS_PER_YEAR * (2.4 if day else 0.25):
            out.schedule(a, _rand_t(r, t0, t1), "manual_open", {})
        if r.random() < sc["mech_trip_per_year"] / INTERVALS_PER_YEAR * (2.67 if day else 0.0):
            out.schedule(a, _rand_t(r, t0, t1), "mech_trip", {})

    # ------------------------------------------------------------ outputs
    def _telemetry(self, ctx: Ctx, i_avg: float, sig) -> dict:
        a, r, cfg, ups, cond, t1 = ctx.a, ctx.r, ctx.cfg, ctx.ups, ctx.cond, ctx.t1
        lc, b = cfg["load"], cfg["battery"]
        live = ups.line_live(t1)
        if a.switch == "CLOSED" and live:
            ia, ib, ic = ctx.i_phase
        else:
            ia = ib = ic = 0.0
        i_n = residual_current(ia, ib, ic) + (lc["leakage_a_at_full_pollution"] * a.pollution if live else 0.0)
        if live:
            drop = lc["voltage_load_drop"] * min(1.2, i_avg / a.peak_a if a.switch == "CLOSED" else 0.0)
            base = a.system_kv * (1.0 + a.v_offset - drop)
            uab, ubc, uca = (base * (1.0 + r.gauss(0.0, lc["voltage_noise"])) for _ in range(3))
            freq = round(lc["frequency_hz"] + r.gauss(0.0, lc["frequency_sigma"]), 3)
        else:
            uab = ubc = uca = 0.0
            freq = None

        v, i = ups.v_last, ups.i_last
        if v is None:
            s = soh(a, b)
            rint = battery_rint(s, b["r0_ohm"], b["r_soh_exponent"])
            offset = -b["shorted_cell_ocv_drop_v"] if a.shorted else 0.0
            mode = "FLOAT" if a.ac_on and not a.charger_fault else "REST"
            v = terminal_voltage(a.soc, 0.0, rint, mode, a.batt_temp, 0.0, b["float_voltage_25c"],
                                 b["bulk_voltage_boost"], b["temp_comp_v_per_c"], offset)
            i = b["float_current_a"] if mode == "FLOAT" else -(a.ctrl_load_a if not a.ac_on else 0.0)
        charge = int(round(min(100.0, max(0.0, a.soc * 100.0 + r.gauss(0.0, b["charge_pct_noise"])))))

        return dict(
            asset_id=a.asset_id, ts=ms_to_dt(t1),
            ia=round(ia, 1), ib=round(ib, 1), ic=round(ic, 1), in_residual=round(i_n, 1),
            uab=round(uab, 2), ubc=round(ubc, 2), uca=round(uca, 2), frequency_hz=freq,
            switch_state=a.switch, ac_supply_on=bool(a.ac_on), external_load_on=bool(a.ext_on),
            battery_voltage=round(v + r.gauss(0.0, 0.01), 2), battery_current=round(i, 2),
            battery_charge_pct=charge, battery_temp_c=round(a.batt_temp, 1),
            cubicle_temp_c=round(a.cub_temp, 1), ambient_temp_c=round(cond.ambient_c, 1),
            humidity_pct=cond.humidity_pct, weather=cond.weather, lightning_strikes_10km=cond.lightning,
            comms_connected=bool(a.connected), signal_strength_dbm=sig,
            co_total=a.co_total, mechanical_wear_pct=round(a.co_total / a.rated_mech_ops * 100.0, 3),
            contact_wear_pct_a=round(a.wear[0], 3), contact_wear_pct_b=round(a.wear[1], 3),
            contact_wear_pct_c=round(a.wear[2], 3))

    def _ground_truth(self, a: AssetState, day: str, t: int) -> dict:
        b = self.cfg["battery"]
        return dict(
            date=day, asset_id=a.asset_id, profiles="+".join(a.profiles),
            active_processes="+".join(p for p in DEGRADATION_PROCESSES if a.onsets.get(p, 1 << 62) <= t),
            battery_soh=round(soh(a, b), 4), battery_soc=round(a.soc, 4), shorted_cell=a.shorted,
            charger_fault=a.charger_fault, mech_deg=round(a.mech_deg, 4), comms_deg=round(a.comms_deg, 4),
            ctrl_deg=round(a.ctrl_deg, 4), pollution=round(a.pollution, 4),
            contact_wear_max=round(max(a.wear), 3), persistent_fault=a.persistent_fault,
            open_failures="+".join(sorted(a.failed)))
