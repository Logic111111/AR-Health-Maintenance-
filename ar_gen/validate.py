"""Data validation: gaps, ranges, counters vs operations, fault sequences, alarms, coverage."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.reader import asset_frame, asset_start_ms, read_df
from .pipeline import get_engine, step_ms
from .sources.simulated import SimulatedNojaSource
from .variables import to_ms

log = logging.getLogger(__name__)


TELEMETRY_COLS = ("ts", "co_total", "mechanical_wear_pct", "battery_voltage", "battery_charge_pct",
                  "humidity_pct", "ia", "battery_temp_c", "contact_wear_pct_a")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _table_counts(conn) -> dict:
    return {name: conn.execute(sa.select(sa.func.count()).select_from(t)).scalar()
            for name, t in schema.TABLES.items()}


def validate(cfg: dict) -> tuple[list[Check], dict]:
    engine = get_engine(cfg)
    step = step_ms(cfg)
    checks: list[Check] = []
    t = schema.TABLES
    with engine.connect() as conn:
        counts = _table_counts(conn)
        assets = asset_frame(conn)
        a_start = asset_start_ms(conn, step)
        ops = read_df(conn, sa.select(t["co_operations"]))
        faults = read_df(conn, sa.select(t["fault_events"]))
        alarms = read_df(conn, sa.select(t["alarm_events"]).order_by(t["alarm_events"].c.ts,
                                                                      t["alarm_events"].c.event_id))
        maint = read_df(conn, sa.select(t["maintenance_log"]))
        failures = read_df(conn, sa.select(t["failure_log"]))
        cov_v = dict(conn.execute(sa.select(t["ar_variables_15min"].c.asset_id, sa.func.count())
                                  .group_by(t["ar_variables_15min"].c.asset_id)).fetchall())
        cov_h = dict(conn.execute(sa.select(t["ar_health_index_15min"].c.asset_id, sa.func.count())
                                  .group_by(t["ar_health_index_15min"].c.asset_id)).fetchall())
        hi_stats = conn.execute(sa.select(sa.func.min(t["ar_health_index_15min"].c.health_index),
                                          sa.func.max(t["ar_health_index_15min"].c.health_index),
                                          sa.func.avg(t["ar_health_index_15min"].c.health_index))).fetchone()

    source = SimulatedNojaSource(cfg)
    last_ms = source.last_collected_ms() if source.exists() else None
    if last_ms is None or assets.empty:
        return [Check("data present", False, "no seeded fleet / no data")], counts

    ops_ms = to_ms(ops["ts"]) if len(ops) else np.array([], dtype=np.int64)
    al_ms = to_ms(alarms["ts"]) if len(alarms) else np.array([], dtype=np.int64)
    resets = defaultdict(list)
    for aid, ms, det in zip(maint["asset_id"], to_ms(maint["ts"]), maint["details"]):
        if isinstance(det, dict) and det.get("counters_reset"):
            resets[aid].append(int(ms))

    gaps_total = gaps_unexplained = 0
    range_bad = Counter()
    counter_bad = counter_checked = wear_bad = 0
    for aid in assets["asset_id"]:
        with engine.connect() as conn:
            tt = t["telemetry_15min"]
            tl = read_df(conn, sa.select(*(tt.c[c] for c in TELEMETRY_COLS)).where(tt.c.asset_id == aid)
                         .order_by(tt.c.ts))
        if tl.empty:
            continue
        ts = to_ms(tl["ts"])
        # ---- gaps must be explained by a controller power-down (Shutdown ... Power Restart)
        expected = np.arange(a_start[aid] + step, last_ms + 1, step, dtype=np.int64)
        missing = np.setdiff1d(expected, ts)
        gaps_total += len(missing)
        if len(missing):
            sel = (alarms["asset_id"] == aid).to_numpy()
            ev = alarms[sel]
            ev_ms = al_ms[sel]
            off = []
            start = None
            for ms, sig in zip(ev_ms, ev["signal_name"]):
                if sig == "Shutdown" and start is None:
                    start = ms
                elif sig == "Power Restart" and start is not None:
                    off.append((start, ms))
                    start = None
            if start is not None:
                off.append((start, np.iinfo(np.int64).max))
            explained = np.zeros(len(missing), dtype=bool)
            for s, e in off:
                explained |= (missing >= s) & (missing < e)
            gaps_unexplained += int((~explained).sum())
        # ---- physical ranges
        rng_checks = {"battery_voltage": (9.6, 14.8), "battery_charge_pct": (0, 100), "humidity_pct": (0, 100),
                      "ia": (0, 20000), "battery_temp_c": (5, 75), "contact_wear_pct_a": (0, 150)}
        for c, (lo, hi) in rng_checks.items():
            x = tl[c].dropna()
            range_bad[c] += int(((x < lo) | (x > hi)).sum())
        # ---- lifetime counter == logged successful OPEN operations
        sel = ((ops["asset_id"] == aid) & (ops["op_type"] == "OPEN") & (ops["result"] == "SUCCESS")).to_numpy()
        opens = np.sort(ops_ms[sel])
        co = tl["co_total"].to_numpy(dtype=np.int64)
        rs = np.array(sorted(resets.get(aid, [])), dtype=np.int64)
        prev, cur = ts[:-2], ts[1:-1]            # last row may hold a fault still in progress
        n_ops = np.searchsorted(opens, cur, "right") - np.searchsorted(opens, prev, "right")
        no_reset = (np.searchsorted(rs, cur, "right") - np.searchsorted(rs, prev, "right")) == 0
        counter_checked += int(no_reset.sum())
        counter_bad += int(((np.diff(co)[:-1] != n_ops) & no_reset).sum())
        rated = float(assets.loc[assets["asset_id"] == aid, "rated_mech_ops"].iloc[0])
        wear_bad += int((np.abs(tl["mechanical_wear_pct"] - tl["co_total"] / rated * 100) > 1e-3).sum())

    checks.append(Check("telemetry gaps explained by controller power-down", gaps_unexplained == 0,
                        f"{gaps_total} missing rows, {gaps_unexplained} unexplained"))
    checks.append(Check("physical ranges", sum(range_bad.values()) == 0,
                        ", ".join(f"{k}: {v}" for k, v in range_bad.items() if v) or "all within limits"))
    checks.append(Check("co_total matches logged OPEN operations", counter_bad == 0,
                        f"{counter_checked} interval pairs checked, {counter_bad} mismatches"))
    checks.append(Check("mechanical wear = co_total / rated ops", wear_bad == 0, f"{wear_bad} mismatches"))

    # ---- fault <-> operation sequences
    by_fault = {fid: g for fid, g in ops[ops["fault_id"].notna()].groupby("fault_id")}
    bad_seq = []
    for f in faults.itertuples():
        g = by_fault.get(f.fault_id)
        if g is None:
            bad_seq.append((f.fault_id, "no operations"))
            continue
        g = g.sort_values("ts")
        closes = g[g["op_type"] == "CLOSE"]
        trips = g[g["op_type"] == "OPEN"]
        if len(closes) != f.reclose_attempts or not (g["ts"] >= f.start_ts).all() or not (g["ts"] <= f.end_ts).all():
            bad_seq.append((f.fault_id, "count/time"))
        elif (trips["source"] != "PROTECTION").any() or (closes["source"] != "AUTO_RECLOSE").any():
            bad_seq.append((f.fault_id, "source"))
        elif f.outcome == "CLEARED_BY_RECLOSE" and not (g.iloc[-1]["op_type"] == "CLOSE" and g.iloc[-1]["result"] == "SUCCESS"):
            bad_seq.append((f.fault_id, "outcome"))
    checks.append(Check("fault -> operation sequences", not bad_seq,
                        f"{len(faults)} faults, {len(bad_seq)} inconsistent {bad_seq[:3]}"))

    # ---- alarm RAISED/CLEARED pairing
    active, bad_pair = set(), 0
    for aid, sig, state in zip(alarms["asset_id"], alarms["signal_name"], alarms["state"]):
        key = (aid, sig)
        if state == "RAISED":
            bad_pair += key in active
            active.add(key)
        elif state == "CLEARED":
            bad_pair += key not in active
            active.discard(key)
    checks.append(Check("alarm RAISED/CLEARED pairing", bad_pair == 0,
                        f"{len(alarms)} events, {bad_pair} violations, {len(active)} still active"))

    # ---- failures followed by corrective maintenance
    repaired = set(maint["failure_id"].dropna().astype(int))
    max_delay = max(hi for _, hi in cfg["maintenance"]["corrective_delay_days"].values()) + 3
    late = pending = 0
    for f in failures.itertuples():
        if f.failure_id in repaired:
            continue
        if int(to_ms([f.ts])[0]) < last_ms - max_delay * 86_400_000:
            late += 1
        else:
            pending += 1
    checks.append(Check("failures repaired by corrective maintenance", late == 0,
                        f"{len(failures)} failures, {len(failures) - late - pending} repaired, "
                        f"{pending} repair pending, {late} overdue"))

    # ---- calculated tables cover every asset and interval
    exp_rows = {aid: (last_ms - s) // step for aid, s in a_start.items()}
    miss_v = sum(max(0, exp_rows[a] - cov_v.get(a, 0)) for a in exp_rows)
    miss_h = sum(max(0, exp_rows[a] - cov_h.get(a, 0)) for a in exp_rows)
    checks.append(Check("V1-V42 rows for every asset and interval", miss_v == 0,
                        f"{sum(cov_v.values())} rows, {miss_v} missing"))
    checks.append(Check("Health Index rows for every asset and interval", miss_h == 0,
                        f"{sum(cov_h.values())} rows, {miss_h} missing"))
    if hi_stats and hi_stats[0] is not None:
        checks.append(Check("Health Index within 0-100", 0 <= hi_stats[0] and hi_stats[1] <= 100,
                            f"min {hi_stats[0]:.1f}, max {hi_stats[1]:.1f}, mean {hi_stats[2]:.1f}"))
    return checks, counts


def hi_before_failures(cfg: dict) -> pd.DataFrame:
    """Mean HI in the 7 days before each failure compared with the asset's overall mean."""
    engine = get_engine(cfg)
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        failures = read_df(conn, sa.select(schema.failure_log))
        overall = dict(conn.execute(sa.select(h.c.asset_id, sa.func.avg(h.c.health_index))
                                    .group_by(h.c.asset_id)).fetchall())
        rows = []
        for f in failures.itertuples():
            lo = f.ts - pd.Timedelta(days=7)
            avg = conn.execute(sa.select(sa.func.avg(h.c.health_index), sa.func.min(h.c.health_index))
                               .where(h.c.asset_id == f.asset_id, h.c.ts > lo, h.c.ts <= f.ts)).fetchone()
            rows.append(dict(asset_id=f.asset_id, ts=f.ts, component=f.component,
                             hi_7d_before=avg[0], hi_min_7d=avg[1], hi_asset_mean=overall.get(f.asset_id)))
    return pd.DataFrame(rows)
