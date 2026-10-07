"""V1-V42 "final data" calculator (spec section 5).

Pure pandas/numpy: raw DataFrames in, one row per asset per 15-minute interval out.
Every value at time t uses only raw data with timestamps <= t (no look-ahead), so
computing a range at once or interval by interval gives identical results and the
ML labels never leak into the features.

Window convention: W24 = (t - 24 h, t], W30 = (t - 30 d, t]. Events are bucketed to the
interval that contains them (interval = (t - 15 min, t]).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from .db.schema import V_COLUMNS

STEP_MS = 15 * 60 * 1000
NEG_INF = np.iinfo(np.int64).min // 4
POS_INF = np.iinfo(np.int64).max // 4

OC_TYPES = ("OC_A", "OC_B", "OC_C")
PERMANENT = ("LOCKOUT", "UPSTREAM_CLEARED")

# Health-Index reset after maintenance: the windows of the repaired part start at the reset.
RESET_GROUPS = ("BATTERY", "SWITCHING", "COMMS", "CONTROLLER")
MAINT_GROUP = {"BATTERY": "BATTERY", "MECHANISM": "SWITCHING", "CONTACTS": "SWITCHING",
               "COMMS": "COMMS", "CONTROLLER": "CONTROLLER"}
RESET_COLUMNS = ["asset_id", "ts", "part_groups"]
_EMPTY = np.array([], dtype=np.int64)


@dataclass
class RawInputs:
    """Raw data for one computation range (event tables already cover the look-back)."""
    assets: pd.DataFrame
    telemetry: pd.DataFrame
    faults: pd.DataFrame
    ops: pd.DataFrame
    alarms: pd.DataFrame
    tests: pd.DataFrame
    comms: pd.DataFrame
    maintenance: pd.DataFrame
    firmware: pd.DataFrame
    asset_start: dict = field(default_factory=dict)   # asset_id -> ms of first data (exclusive)
    resets: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=RESET_COLUMNS))  # hi_reset_log


# ------------------------------------------------------------------ helpers
def to_ms(values) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    return pd.to_datetime(pd.Series(values)).to_numpy(dtype="datetime64[ms]").astype(np.int64)


def date_ms(d) -> int | None:
    if d is None or (isinstance(d, float) and math.isnan(d)) or d is pd.NaT:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    if isinstance(d, (datetime, pd.Timestamp)):
        d = d.date()
    return int(np.datetime64(d, "ms").astype(np.int64))


def ms_to_ts(ms: np.ndarray) -> np.ndarray:
    return np.asarray(ms, dtype=np.int64).astype("datetime64[ms]").astype("datetime64[ns]")


def _details(v) -> dict:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return {}
    if isinstance(v, str):
        try:
            return json.loads(v) or {}
        except ValueError:
            return {}
    return dict(v)


def attribute_changes(maint: pd.DataFrame) -> dict:
    """{asset_id: {field: [(ts_ms, old, new), ...]}} from maintenance_log.details.changes."""
    out: dict = {}
    if maint is None or maint.empty:
        return out
    m = maint.assign(_ms=to_ms(maint["ts"])).sort_values(["_ms", "maint_id"] if "maint_id" in maint else ["_ms"])
    for aid, ms, det in zip(m["asset_id"], m["_ms"], m["details"]):
        for fld, pair in (_details(det).get("changes") or {}).items():
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                out.setdefault(aid, {}).setdefault(fld, []).append((int(ms), pair[0], pair[1]))
    return out


def _value_at(grid: np.ndarray, changes: list, current) -> np.ndarray:
    """Attribute value at each grid time, reconstructed from the change history."""
    if not changes:
        return np.full(len(grid), current, dtype=object)
    times = np.array([c[0] for c in changes], dtype=np.int64)
    vals = np.array([changes[0][1]] + [c[2] for c in changes], dtype=object)
    return vals[np.searchsorted(times, grid, side="right")]


def parse_groups(text) -> list[str]:
    """'SWITCHING,COMMS' or 'ALL' -> list of part groups (unknown names are ignored)."""
    parts = [p.strip().upper() for p in str(text or "").split(",") if p.strip()]
    if "ALL" in parts:
        return list(RESET_GROUPS)
    return [p for p in RESET_GROUPS if p in parts]


def reset_points(maintenance: pd.DataFrame | None, resets: pd.DataFrame | None) -> dict:
    """{asset_id: {group: sorted reset times (ms)}} from CORRECTIVE maintenance + manual resets."""
    pts: dict = {}
    if maintenance is not None and not maintenance.empty and {"maint_type", "component"} <= set(maintenance.columns):
        m = maintenance[maintenance["maint_type"] == "CORRECTIVE"]
        for aid, ms, comp in zip(m["asset_id"], to_ms(m["ts"]), m["component"]):
            grp = MAINT_GROUP.get(comp)
            if grp:
                pts.setdefault(aid, {}).setdefault(grp, []).append(int(ms))
    if resets is not None and not resets.empty:
        for aid, ms, groups in zip(resets["asset_id"], to_ms(resets["ts"]), resets["part_groups"]):
            for grp in parse_groups(groups):
                pts.setdefault(aid, {}).setdefault(grp, []).append(int(ms))
    return {aid: {grp: np.array(sorted(v), dtype=np.int64) for grp, v in d.items()} for aid, d in pts.items()}


class _Grid:
    """Output grid G (interval ends in (start, end]) and the look-back grid F behind it."""

    def __init__(self, start_ms: int, end_ms: int, n_long: int, n_short: int):
        self.G = np.arange(start_ms + STEP_MS, end_ms + 1, STEP_MS, dtype=np.int64)
        self.n_long, self.n_short = n_long, n_short
        self.F0 = int(self.G[0]) - (n_long + 1) * STEP_MS
        self.F = np.arange(self.F0, int(self.G[-1]) + 1, STEP_MS, dtype=np.int64)
        self.gpos = ((self.G - self.F0) // STEP_MS).astype(np.int64)

    def bucket(self, ts_ms: np.ndarray):
        pos = -((self.F0 - np.asarray(ts_ms, dtype=np.int64)) // STEP_MS)   # ceil division
        keep = (pos >= 0) & (pos < len(self.F))
        return pos, keep

    def window_sum(self, per_row: np.ndarray, n: int) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(per_row, dtype=float)])
        return cs[self.gpos + 1] - cs[self.gpos + 1 - n]

    def count(self, ts_ms: np.ndarray, n: int) -> np.ndarray:
        pos, keep = self.bucket(ts_ms)
        b = np.bincount(pos[keep], minlength=len(self.F)).astype(float)
        return self.window_sum(b, n).astype(np.int64)

    def rolling(self, arr: np.ndarray, n: int, how: str) -> np.ndarray:
        r = pd.Series(arr).rolling(n, min_periods=1)
        return getattr(r, how)().to_numpy()[self.gpos]

    def place(self, ts_ms: np.ndarray, values, fill=np.nan, dtype=float) -> np.ndarray:
        arr = np.full(len(self.F), fill, dtype=dtype)
        pos = (np.asarray(ts_ms, dtype=np.int64) - self.F0) // STEP_MS
        ok = (pos >= 0) & (pos < len(self.F)) & ((np.asarray(ts_ms) - self.F0) % STEP_MS == 0)
        arr[pos[ok]] = np.asarray(values, dtype=dtype)[ok]
        return arr


def _sel(df: pd.DataFrame, aid: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    return df[df["asset_id"] == aid]


def _intervals_integral(starts: np.ndarray, ends: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Total length of the (sorted, non-overlapping) intervals that lies before each t."""
    if len(starts) == 0:
        return np.zeros(len(t))
    lengths = ends - starts
    cum = np.concatenate([[0], np.cumsum(np.minimum(lengths, POS_INF // 4), dtype=np.int64)])
    k = np.searchsorted(starts, t, side="right")
    last = np.maximum(k - 1, 0)
    partial = np.where(k > 0, np.clip(np.minimum(ends[last], t) - starts[last], 0, None), 0)
    return (cum[last] + partial).astype(float)


def _alarm_intervals(al: pd.DataFrame, signal: str, lo_ms: int):
    ev = al[al["signal_name"] == signal]
    starts, ends = [], []
    open_at = None
    first = True
    for ms, state in zip(to_ms(ev["ts"]), ev["state"]):
        if state == "RAISED":
            if open_at is None:
                open_at = int(ms)
        elif state == "CLEARED":
            if open_at is None and first:
                open_at = lo_ms
            if open_at is not None:
                starts.append(open_at)
                ends.append(int(ms))
                open_at = None
        first = False
    if open_at is not None:
        starts.append(open_at)
        ends.append(POS_INF)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


# ------------------------------------------------------------------ main entry
def compute_variables(inp: RawInputs, start_ms: int, end_ms: int, cfg: dict) -> pd.DataFrame:
    """V1-V42 for every asset and every interval end in (start_ms, end_ms]."""
    vc = cfg["variables"]
    n_long = int(vc["long_window_days"] * 96)
    n_short = int(vc["short_window_hours"] * 4)
    start_ms -= start_ms % STEP_MS
    if end_ms <= start_ms:
        return pd.DataFrame(columns=["asset_id", "ts", "telemetry_present", *V_COLUMNS])
    grid = _Grid(start_ms, end_ms, n_long, n_short)
    changes = attribute_changes(inp.maintenance)
    fw = inp.firmware.copy()
    fw["_ms"] = [date_ms(d) for d in fw["release_date"]]
    fw = fw.sort_values("_ms")
    fw_versions = list(fw["version"])
    fw_ms = fw["_ms"].to_numpy(dtype=np.int64)

    resets = reset_points(inp.maintenance, inp.resets)
    frames = []
    for asset in inp.assets.to_dict("records"):
        frames.append(_asset_variables(asset, inp, grid, changes.get(asset["asset_id"], {}),
                                       fw_versions, fw_ms, cfg, resets.get(asset["asset_id"], {})))
    frames = [f for f in frames if len(f)]
    columns = ["asset_id", "ts", "telemetry_present", *V_COLUMNS]          # table order (V1 ... V42)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _asset_variables(asset: dict, inp: RawInputs, g: _Grid, chg: dict, fw_versions, fw_ms, cfg,
                     rp: dict | None = None) -> pd.DataFrame:
    vc, sc = cfg["variables"], cfg["switching"]
    aid = asset["asset_id"]
    nL, nS = g.n_long, g.n_short
    a_start = inp.asset_start.get(aid)
    G = g.G
    if a_start is not None:
        keep_g = G > a_start
    else:
        keep_g = np.ones(len(G), dtype=bool)
    if not keep_g.any():
        return pd.DataFrame()
    out: dict = {}

    # ---------------- maintenance resets: window of group g at t is (max(t - W, R_g(t)), t]
    rp = rp or {}

    def last_reset(grp):
        arr = rp.get(grp, _EMPTY)
        if not len(arr):
            return np.full(len(G), NEG_INF, dtype=np.int64)
        k = np.searchsorted(arr, G, side="right")
        return np.where(k > 0, arr[np.maximum(k - 1, 0)], NEG_INF)

    R = {grp: last_reset(grp) for grp in RESET_GROUPS}
    R_any = np.maximum.reduce([R[grp] for grp in RESET_GROUPS])

    def count_since(ev_ms, n_rows, grp):
        ev = np.sort(np.asarray(ev_ms, dtype=np.int64))
        lo = np.maximum(G - n_rows * STEP_MS, R[grp])
        return (np.searchsorted(ev, G, side="right") - np.searchsorted(ev, lo, side="right")).astype(np.int64)

    def rows_sum(per_row, n_rows, grp):
        cs = np.concatenate([[0.0], np.cumsum(per_row, dtype=float)])
        lo = np.maximum(G - n_rows * STEP_MS, R[grp])
        return cs[np.searchsorted(g.F, G, side="right")] - cs[np.searchsorted(g.F, lo, side="right")]

    def rows_rolling(arr, n_rows, how, grp):
        resets_g = rp.get(grp, _EMPTY)
        if not len(resets_g):
            return g.rolling(arr, n_rows, how)
        seg = np.searchsorted(resets_g, g.F, side="left")          # a row at exactly R is still "before"
        rolled = getattr(pd.Series(arr).groupby(seg).rolling(n_rows, min_periods=1), how)()
        return rolled.reset_index(level=0, drop=True).sort_index().to_numpy()[g.gpos]

    # ---------------- telemetry on the look-back grid
    tl = _sel(inp.telemetry, aid)
    t_ms = to_ms(tl["ts"])

    def col(name, dtype=float, fill=np.nan):
        return g.place(t_ms, tl[name].to_numpy(), fill, dtype)

    present = g.place(t_ms, np.ones(len(t_ms)), 0.0)
    volt = col("battery_voltage")
    cur = col("battery_current")
    charge = col("battery_charge_pct")
    btemp = col("battery_temp_c")
    amb = col("ambient_temp_c")
    light = col("lightning_strikes_10km")
    conn = g.place(t_ms, tl["comms_connected"].astype(float).to_numpy(), 0.0)
    sig = col("signal_strength_dbm")
    co_total = col("co_total")
    wear = np.fmax.reduce([col(c) for c in ("mechanical_wear_pct", "contact_wear_pct_a",
                                            "contact_wear_pct_b", "contact_wear_pct_c")])
    weather = g.place(t_ms, tl["weather"].to_numpy(), None, object)

    gp = g.gpos
    out["telemetry_present"] = present[gp] > 0
    out["v01_battery_voltage_v"] = volt[gp]

    disc = np.nan_to_num(cur, nan=0.0) < vc["discharge_current_a"]
    starts = disc & ~np.concatenate([[False], disc[:-1]])
    out["v02_discharge_episodes_24h"] = rows_sum(starts.astype(float), nS, "BATTERY").astype(np.int64)

    # ---------------- attribute history (asset_master + maintenance changes)
    def attr(name):
        return _value_at(G, chg.get(name, []), asset.get(name))

    b_install = np.array([date_ms(x) for x in attr("battery_install_date")], dtype=object)
    b_interval = attr("battery_service_interval_years").astype(float)
    b_install_ms = np.array([x if x is not None else np.nan for x in b_install], dtype=float)
    batt_age_d = (G - b_install_ms) / 86_400_000.0
    out["v04_battery_life_used_pct"] = batt_age_d / (b_interval * 365.25) * 100.0

    # battery replacements -> "current battery" start
    rep = np.array(sorted(c[0] for c in chg.get("battery_install_date", [])), dtype=np.int64)
    k = np.searchsorted(rep, G, side="right")
    batt_start = np.where(k > 0, rep[np.maximum(k - 1, 0)] if len(rep) else NEG_INF, NEG_INF)

    # ---------------- alarms
    al = _sel(inp.alarms, aid).sort_values(["ts", "event_id"] if "event_id" in inp.alarms else ["ts"])
    al_ms = to_ms(al["ts"])
    raised = (al["state"] == "RAISED").to_numpy()
    sigs = al["signal_name"].to_numpy()
    cats = al["category"].to_numpy()
    states = al["state"].to_numpy()

    ac_sig = vc["ac_off_signal"]
    out["v05_ac_failures_30d"] = g.count(al_ms[raised & (sigs == ac_sig)], nL)
    out["v06_max_dod_30d_pct"] = 100.0 - rows_rolling(charge, nL, "min", "BATTERY")
    batt_sigs = set(vc["battery_alarm_signals"])
    out["v07_battery_alarms_30d"] = count_since(al_ms[raised & np.isin(sigs, list(batt_sigs))], nL, "BATTERY")
    mean_bt = g.rolling(btemp, nL, "mean")
    out["v08_battery_temp_age_years"] = batt_age_d / 365.25 * np.power(2.0, (mean_bt - 25.0) / 10.0)

    lo_ms = int(al_ms.min()) if len(al_ms) else g.F0
    s_iv, e_iv = _alarm_intervals(al, ac_sig, lo_ms)
    on_batt = _intervals_integral(s_iv, e_iv, G) - _intervals_integral(s_iv, e_iv, G - nL * STEP_MS)
    out["v10_minutes_on_battery_30d"] = on_batt / 60_000.0

    module = set(vc["module_fault_signals"])
    out["v33_module_faults_30d"] = count_since(al_ms[raised & (cats == "MALFUNCTION") & np.isin(sigs, list(module))],
                                               nL, "CONTROLLER")
    excluded = module | batt_sigs | {ac_sig} | set(vc["switching_fail_signals"])
    diag = np.isin(cats, ["WARNING", "MALFUNCTION"]) & np.isin(states, ["RAISED", "EVENT"]) & \
        ~np.isin(sigs, list(excluded))
    out["v34_diag_warnings_30d"] = count_since(al_ms[diag], nL, "CONTROLLER")

    # ---------------- battery tests -> V9 and V3
    bt = _sel(inp.tests, aid)
    bt_ms = to_ms(bt["ts"])
    failed_auto = np.sort(bt_ms[((bt["result"] == "FAILED") & (bt["trigger"] == "AUTO")).to_numpy()])
    out["v09_auto_test_failed_count"] = (np.searchsorted(failed_auto, G, side="right")
                                         - np.searchsorted(failed_auto, batt_start, side="right"))

    rated_ah = float(asset.get("battery_rated_ah") or 26.0)
    ev_a_t, ev_a_v = _soh_from_discharges(g, cur, charge, rated_ah, vc)
    ok_t = ((bt["result"].isin(["PASSED", "FAILED"])) & bt["voltage_before"].notna()
            & bt["voltage_min_during_test"].notna()).to_numpy()
    ev_b_t = bt_ms[ok_t]
    if ok_t.any():
        r_app = ((bt["voltage_before"].to_numpy(dtype=float)[ok_t] - bt["voltage_min_during_test"].to_numpy(dtype=float)[ok_t])
                 / bt["test_load_current_a"].to_numpy(dtype=float)[ok_t])
        ev_b_v = 100.0 * np.clip((vc["soh_ref_resistance_ohm"] / np.maximum(r_app, 1e-4))
                                 ** vc["soh_resistance_exponent"], 0.0, 1.0)
        order = np.argsort(ev_b_t, kind="stable")
        ev_b_t, ev_b_v = ev_b_t[order], ev_b_v[order]
    else:
        ev_b_v = np.array([])
    out["v03_soh_est_pct"] = _latest_evidence(G, batt_start, nL, ev_a_t, ev_a_v, ev_b_t, ev_b_v,
                                              batt_age_d, vc)

    # ---------------- faults
    fl = _sel(inp.faults, aid)
    f_ms = to_ms(fl["start_ts"])
    ftype = fl["fault_type"].to_numpy()
    outcome = fl["outcome"].to_numpy()
    pos, keep = g.bucket(f_ms)
    lightning_at = np.zeros(len(f_ms), dtype=bool)
    lightning_at[keep] = np.nan_to_num(light[pos[keep]], nan=0.0) > 0
    out["v11_wear_max_pct"] = pd.Series(wear).ffill().to_numpy()[gp]
    out["v12_lightning_faults_30d"] = g.count(f_ms[lightning_at], nL)
    out["v13_permanent_faults_30d"] = g.count(f_ms[np.isin(outcome, PERMANENT)], nL)
    out["v14_temporary_faults_30d"] = g.count(f_ms[outcome == "CLEARED_BY_RECLOSE"], nL)
    out["v15_earth_faults_30d"] = g.count(f_ms[ftype == "EF"], nL)
    out["v16_sef_faults_30d"] = g.count(f_ms[ftype == "SEF"], nL)
    out["v17_oc_faults_30d"] = g.count(f_ms[np.isin(ftype, OC_TYPES)], nL)
    peak = np.zeros(len(g.F))
    if keep.any():
        np.maximum.at(peak, pos[keep], fl["max_fault_current_a"].to_numpy(dtype=float)[keep])
    out["v18_max_fault_current_30d_a"] = g.rolling(peak, nL, "max")
    out["v19_faults_24h"] = g.count(f_ms, nS)
    out["v20_lockouts_30d"] = g.count(f_ms[outcome == "LOCKOUT"], nL)

    # ---------------- switching operations
    op = _sel(inp.ops, aid)
    o_ms = to_ms(op["ts"])
    otype = op["op_type"].to_numpy()
    src = op["source"].to_numpy()
    res = op["result"].to_numpy()
    otime = op["operating_time_ms"].to_numpy(dtype=float)
    out["v21_co_total"] = pd.Series(co_total).ffill().to_numpy()[gp]
    out["v22_open_ops_30d"] = count_since(o_ms[otype == "OPEN"], nL, "SWITCHING")
    out["v23_close_ops_30d"] = count_since(o_ms[otype == "CLOSE"], nL, "SWITCHING")
    out["v24_protection_trips_30d"] = count_since(o_ms[(otype == "OPEN") & (src == "PROTECTION")], nL, "SWITCHING")
    out["v25_auto_reclose_ops_30d"] = count_since(o_ms[(otype == "CLOSE") & (src == "AUTO_RECLOSE")], nL, "SWITCHING")
    out["v26_switching_ops_24h"] = count_since(o_ms, nS, "SWITCHING")
    bad_op = (res == "FAIL") | ((otype == "OPEN") & (otime > sc["excessive_to_ms"])) | \
             ((otype == "CLOSE") & (otime > sc["excessive_tc_ms"]))
    out["v27_unsuccessful_ops_30d"] = count_since(o_ms[bad_op], nL, "SWITCHING")

    # ---------------- communications
    cm = _sel(inp.comms, aid)
    c_ms = to_ms(cm["ts"])
    cev = cm["event"].to_numpy()
    # expected polls = grid points in (max(t - 24 h, reset, data start), t]; none yet -> unknown (NaN)
    lo28 = np.maximum(G - nS * STEP_MS, R["COMMS"])
    if a_start is not None:
        lo28 = np.maximum(lo28, a_start)
    expected = (np.searchsorted(g.F, G, side="right") - np.searchsorted(g.F, lo28, side="right")).astype(float)
    avail = rows_sum(conn, nS, "COMMS") / np.where(expected > 0, expected, np.nan) * 100.0
    out["v28_scada_availability_24h_pct"] = np.minimum(100.0, avail)
    out["v29_comm_failures_24h"] = count_since(c_ms[cev == "DISCONNECT"], nS, "COMMS")
    out["v30_modem_restarts_30d"] = count_since(c_ms[cev == "MODEM_RESTART"], nL, "COMMS")
    out["v31_signal_strength_24h_dbm"] = rows_rolling(sig, nS, "mean", "COMMS")
    out["v32_reconnects_24h"] = count_since(c_ms[cev == "RECONNECT"], nS, "COMMS")

    # ---------------- controller / firmware
    fw_now = attr("firmware_version")
    idx = {v: i for i, v in enumerate(fw_versions)}
    avail = np.searchsorted(fw_ms, G, side="right")
    behind = np.array([np.nan if v not in idx else max(0, int(n) - idx[v] - 1)
                       for v, n in zip(fw_now, avail)], dtype=float)
    out["v35_firmware_versions_behind"] = behind
    rs485 = g.count(c_ms[cev == "RS485_ERROR"], nL).astype(float)
    out["v36_rs485_errors_30d"] = rs485 if bool(asset.get("has_rs485_link")) else np.full(len(G), np.nan)

    # ---------------- ageing and environment
    def years_since(name):
        ms = np.array([date_ms(x) if x is not None else np.nan for x in attr(name)], dtype=float)
        return (G - ms) / 86_400_000.0

    out["v37_sg_age_years"] = years_since("manufactured_date") / 365.25
    out["v38_years_in_service"] = years_since("installed_date") / 365.25
    last_maint = np.array([date_ms(x) if x is not None else np.nan for x in attr("last_maintenance_date")],
                          dtype=float)
    last_maint = np.fmax(last_maint, np.where(R_any > NEG_INF, R_any, np.nan).astype(float))
    out["v39_days_since_maintenance"] = (G - last_maint) / 86_400_000.0
    out["v40_ambient_temp_c"] = amb[gp]
    out["v41_coastal_flag"] = np.full(len(G), int(bool(asset.get("coastal_flag"))))
    out["v42_weather"] = weather[gp]

    df = pd.DataFrame({"asset_id": aid, "ts": ms_to_ts(G), **out})
    return df[keep_g].reset_index(drop=True)


def _soh_from_discharges(g: _Grid, cur: np.ndarray, charge: np.ndarray, rated_ah: float, vc: dict):
    """Coulomb-counting capacity estimates from completed discharge episodes.

    Evidence time = first telemetry row after the episode (when its end is known).
    """
    disc = np.nan_to_num(cur, nan=0.0) < vc["discharge_current_a"]
    valid = ~np.isnan(cur)
    d = np.diff(np.concatenate([[0], disc.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    times, vals = [], []
    for s, e in zip(starts, ends):
        if s == 0 or e >= len(cur) or not valid[s - 1] or not valid[e]:
            continue
        if np.isnan(charge[s - 1]) or np.isnan(charge[e - 1]):
            continue
        ah = float(np.sum(-cur[s:e])) * 0.25 + cur[s] * 0.125
        delta = charge[s - 1] - charge[e - 1]
        if delta >= vc["soh_min_delta_pct"] and ah > 0:
            times.append(int(g.F[e]))
            vals.append(min(100.0, ah / (delta / 100.0) / rated_ah * 100.0))
    return np.array(times, dtype=np.int64), np.array(vals, dtype=float)


def _latest_evidence(G, batt_start, n_long, ta, va, tb, vb, batt_age_d, vc) -> np.ndarray:
    def latest(times, vals, max_age=None):
        if len(times) == 0:
            return np.full(len(G), NEG_INF, dtype=np.int64), np.full(len(G), np.nan)
        k = np.searchsorted(times, G, side="right") - 1
        t = np.where(k >= 0, times[np.maximum(k, 0)], NEG_INF)
        v = np.where(k >= 0, vals[np.maximum(k, 0)], np.nan)
        ok = (k >= 0) & (t > batt_start)
        if max_age is not None:
            ok &= t > G - max_age
        return np.where(ok, t, NEG_INF), np.where(ok, v, np.nan)

    t_a, v_a = latest(ta, va, n_long * STEP_MS)
    t_b, v_b = latest(tb, vb)
    est = np.where(t_a > t_b, v_a, v_b)
    no_evidence = (t_a == NEG_INF) & (t_b == NEG_INF)
    new_batt = batt_age_d < vc["soh_new_battery_days"]
    return np.where(no_evidence, np.where(new_batt, 100.0, np.nan), est)
