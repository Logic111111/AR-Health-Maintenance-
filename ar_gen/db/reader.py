"""Read raw data for the variable calculator and convert DataFrames to insertable rows."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import sqlalchemy as sa

from ..sim.timeutil import DAY_MS, MIN_MS, ms_to_dt
from ..variables import RawInputs, to_ms
from . import schema

EXTRA_LOOKBACK_DAYS = 7      # alarms raised before the 30-day window (e.g. a long AC-off)


def read_df(conn, stmt) -> pd.DataFrame:
    res = conn.execute(stmt)
    return pd.DataFrame(res.fetchall(), columns=list(res.keys()))


def asset_frame(conn) -> pd.DataFrame:
    return read_df(conn, sa.select(schema.asset_master))


def asset_start_ms(conn, step_ms: int) -> dict:
    # One index seek per asset (a GROUP BY would scan the whole telemetry index).
    t = schema.telemetry_15min
    out = {}
    for aid in conn.execute(sa.select(schema.asset_master.c.asset_id)).scalars():
        ts = conn.execute(sa.select(sa.func.min(t.c.ts)).where(t.c.asset_id == aid)).scalar()
        if ts is not None:
            out[aid] = int(to_ms([ts])[0]) - step_ms
    return out


def load_inputs(conn, start_ms: int, end_ms: int, cfg: dict, asset_start: dict | None = None) -> RawInputs:
    """Raw rows needed to compute variables for interval ends in (start_ms, end_ms]."""
    vc = cfg["variables"]
    step = int(cfg["interval_minutes"]) * MIN_MS
    tel_lo = ms_to_dt(start_ms - int(vc["long_window_days"]) * DAY_MS - 2 * step)
    ev_lo = ms_to_dt(start_ms - (int(vc["long_window_days"]) + EXTRA_LOOKBACK_DAYS) * DAY_MS)
    hi = ms_to_dt(end_ms)
    t = schema.TABLES

    def window(table, col="ts", lo=ev_lo, cols=None):
        tb = t[table]
        sel = sa.select(*(tb.c[c] for c in cols)) if cols else sa.select(tb)
        return read_df(conn, sel.where(tb.c[col] > lo, tb.c[col] <= hi))

    tel_cols = ["asset_id", "ts", "battery_voltage", "battery_current", "battery_charge_pct", "battery_temp_c",
                "ambient_temp_c", "lightning_strikes_10km", "comms_connected", "signal_strength_dbm", "weather",
                "co_total", "mechanical_wear_pct", "contact_wear_pct_a", "contact_wear_pct_b", "contact_wear_pct_c"]
    bt = t["battery_tests"]
    rl = t["hi_reset_log"]
    resets = read_df(conn, sa.select(rl.c.asset_id, rl.c.ts, rl.c.part_groups).where(rl.c.ts <= hi))
    return RawInputs(
        resets=resets,
        assets=asset_frame(conn),
        telemetry=window("telemetry_15min", lo=tel_lo, cols=tel_cols),
        faults=window("fault_events", col="start_ts"),
        ops=window("co_operations"),
        alarms=window("alarm_events"),
        tests=read_df(conn, sa.select(bt).where(bt.c.ts <= hi)),
        comms=window("comms_events"),
        maintenance=read_df(conn, sa.select(t["maintenance_log"])),     # full history for attribute timelines
        firmware=read_df(conn, sa.select(t["firmware_release"])),
        asset_start=asset_start if asset_start is not None else asset_start_ms(conn, step),
    )


def df_to_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with plain Python values and None for missing (fast, column-wise)."""
    if df.empty:
        return []
    columns = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            vals = [None if pd.isna(x) else x.to_pydatetime() for x in s.tolist()]
        elif pd.api.types.is_float_dtype(s):
            vals = [None if v != v else v for v in s.to_numpy().tolist()]
        else:
            vals = [None if (isinstance(v, float) and v != v) or v is pd.NaT else
                    (v.item() if hasattr(v, "item") and not isinstance(v, (str, bytes, datetime)) else v)
                    for v in s.tolist()]
        columns.append(vals)
    names = list(df.columns)
    return [dict(zip(names, row)) for row in zip(*columns)]
