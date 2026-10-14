"""Data for the 3D dashboard: recloser status, positions, HI history and recent events.

Plain functions over the SQLAlchemy engine (SQLite or MySQL). Latest rows are read with
one index seek per recloser instead of GROUP BY scans, so the page stays fast on
multi-million-row tables.
"""
from __future__ import annotations

import math

import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.reader import read_df
from .locations import ensure_locations

STATUS_ORDER = ["OFFLINE", "FAULTY", "LOCKOUT", "CRITICAL", "POOR", "FAIR", "GOOD"]
STATUS_COLORS = {                      # neon "control room" palette (RGB, used by the map and the UI)
    "OFFLINE": [120, 134, 156], "FAULTY": [255, 0, 128], "LOCKOUT": [170, 90, 255],
    "CRITICAL": [255, 59, 48], "POOR": [255, 149, 0], "FAIR": [255, 214, 10], "GOOD": [0, 230, 118],
}
STATUS_MEANING = {
    "OFFLINE": "no data for 60 min (controller down or no comms)",
    "FAULTY": "failure recorded, waiting for repair",
    "LOCKOUT": "switch locked out after a fault",
    "CRITICAL": "HI below 40 - act now",
    "POOR": "HI 40-60 - plan maintenance",
    "FAIR": "HI 60-80 - monitor",
    "GOOD": "HI 80 or more",
}
OFFLINE_AFTER_MIN = 60


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x is pd.NaT


HI_CLASSES = {"GOOD": 80, "FAIR": 60, "POOR": 40, "CRITICAL": 0}     # default of config health_index.classes


def status_of(minutes_since_telemetry, open_failure: bool, switch_state, health_index, classes=None) -> str:
    """Priority: OFFLINE > FAULTY > LOCKOUT > HI class (lower limits from config health_index.classes)."""
    if _missing(minutes_since_telemetry) or minutes_since_telemetry >= OFFLINE_AFTER_MIN:
        return "OFFLINE"
    if open_failure:
        return "FAULTY"
    if switch_state == "LOCKOUT":
        return "LOCKOUT"
    if _missing(health_index):
        return "OFFLINE"
    for name, lower in sorted((classes or HI_CLASSES).items(), key=lambda kv: -kv[1]):
        if health_index >= lower:
            return name
    return "CRITICAL"


def _latest_row(conn, table, asset_id: str, cols):
    stmt = (sa.select(*(table.c[c] for c in cols)).where(table.c.asset_id == asset_id)
            .order_by(table.c.ts.desc()).limit(1))
    row = conn.execute(stmt).mappings().first()
    return dict(row) if row else {}


def open_failures(conn) -> dict:
    """asset_id -> list of components with a failure that has no corrective repair yet."""
    f, m = schema.failure_log, schema.maintenance_log
    stmt = (sa.select(f.c.asset_id, f.c.component)
            .select_from(f.outerjoin(m, m.c.failure_id == f.c.failure_id))
            .where(m.c.maint_id.is_(None)))
    out: dict = {}
    for aid, comp in conn.execute(stmt):
        out.setdefault(aid, []).append(comp)
    return out


def fleet_snapshot(engine: sa.Engine, cfg: dict) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    """One row per recloser at the newest calculated interval, with status, colour and position."""
    ensure_locations(engine, cfg)
    am, loc = schema.asset_master, schema.asset_location
    h, t, p = schema.ar_health_index_15min, schema.telemetry_15min, schema.ar_ml_prediction_15min
    hi_cols = ["ts", "health_index", "health_class", "worst_variables", "cap_applied", "f_battery", "f_faults",
               "f_switching", "f_comms", "f_controlling", "f_aging", "f_environment"]
    tel_cols = ["ts", "switch_state", "battery_voltage", "ac_supply_on", "comms_connected", "signal_strength_dbm",
                "ia", "weather"]
    with engine.connect() as conn:
        assets = read_df(conn, sa.select(am.c.asset_id, am.c.model, am.c.area_name, am.c.feeder_name,
                                         am.c.comms_channel, loc.c.latitude, loc.c.longitude)
                         .select_from(am.outerjoin(loc, loc.c.asset_id == am.c.asset_id)).order_by(am.c.asset_id))
        latest = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
        failures = open_failures(conn)
        rows = []
        for aid in assets["asset_id"]:
            hi = _latest_row(conn, h, aid, hi_cols)
            tel = _latest_row(conn, t, aid, tel_cols)
            ml = _latest_row(conn, p, aid, ["ts", "p_fail_30d", "ml_risk_class", "ml_healthy", "top_component"])
            rows.append({"asset_id": aid,
                         **{c: hi.get(c) for c in hi_cols if c != "ts"},
                         **{c: tel.get(c) for c in tel_cols if c != "ts"},
                         "hi_ts": hi.get("ts"), "last_telemetry_ts": tel.get("ts"),
                         "ml_p": ml.get("p_fail_30d"), "ml_class": ml.get("ml_risk_class"),
                         "ml_healthy": ml.get("ml_healthy"), "ml_component": ml.get("top_component"),
                         "ml_ts": ml.get("ts")})
    snap = assets.merge(pd.DataFrame(rows, columns=["asset_id", *[c for c in hi_cols if c != "ts"],
                                                    *[c for c in tel_cols if c != "ts"], "hi_ts",
                                                    "last_telemetry_ts", "ml_p", "ml_class", "ml_healthy",
                                                    "ml_component", "ml_ts"]), on="asset_id", how="left")
    latest_ts = pd.Timestamp(latest) if latest is not None else None

    def minutes_since(x):
        if latest_ts is None or _missing(x):
            return None
        return (latest_ts - pd.Timestamp(x)).total_seconds() / 60.0

    snap["open_failures"] = snap["asset_id"].map(lambda a: ",".join(sorted(set(failures.get(a, [])))))
    snap["minutes_since_data"] = snap["last_telemetry_ts"].map(minutes_since)
    classes = cfg.get("health_index", {}).get("classes")
    snap["status"] = [status_of(m, bool(f), s, hi, classes) for m, f, s, hi in
                      zip(snap["minutes_since_data"], snap["open_failures"], snap["switch_state"],
                          snap["health_index"])]
    # ML values more than one interval behind the HI are old (model removed/broken, scoring failed):
    # they are not shown as current; ml_ts still tells how old they are.
    lag = pd.to_datetime(snap["hi_ts"]) - pd.to_datetime(snap["ml_ts"])
    snap["ml_stale"] = (lag > pd.Timedelta(minutes=int(cfg.get("interval_minutes", 15)))).fillna(False).astype(bool)
    if snap["ml_stale"].any():
        ml_cols = ["ml_p", "ml_class", "ml_healthy", "ml_component"]
        snap[ml_cols] = snap[ml_cols].astype(object)
        snap.loc[snap["ml_stale"], ml_cols] = None
    snap["ml_p"] = pd.to_numeric(snap["ml_p"], errors="coerce")
    snap["color"] = snap["status"].map(lambda s: list(STATUS_COLORS[s]))
    snap["risk"] = snap["health_index"].map(lambda v: 100.0 if _missing(v) else max(0.0, 100.0 - float(v)))
    return latest_ts, snap


def feeder_paths(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Thin lines joining the reclosers of each feeder (in asset-id order)."""
    rows = []
    placed = snapshot.dropna(subset=["latitude", "longitude"]).sort_values("asset_id")
    for feeder, g in placed.groupby("feeder_name", sort=True):
        rows.append({"feeder_name": feeder, "path": [[float(lo), float(la)] for lo, la in
                                                     zip(g["longitude"], g["latitude"])],
                     "color": [90, 90, 90]})
    return pd.DataFrame(rows, columns=["feeder_name", "path", "color"])


def _local(ts: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(ts).dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)


def hi_history(engine: sa.Engine, asset_id: str, days: int = 30,
               tz: str = "Asia/Colombo") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(15-min HI rows with local time, daily average/min HI per local day)."""
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        last = conn.execute(sa.select(sa.func.max(h.c.ts)).where(h.c.asset_id == asset_id)).scalar()
        if last is None:
            return (pd.DataFrame(columns=["ts_local", "health_index"]),
                    pd.DataFrame(columns=["date", "avg_hi", "min_hi"]))
        lo = pd.Timestamp(last) - pd.Timedelta(days=days)
        rows = read_df(conn, sa.select(h.c.ts, h.c.health_index)
                       .where(h.c.asset_id == asset_id, h.c.ts > lo.to_pydatetime()).order_by(h.c.ts))
    rows["ts_local"] = _local(rows["ts"], tz)
    rows = rows[["ts_local", "health_index"]]
    daily = (rows.assign(date=rows["ts_local"].dt.date).groupby("date")["health_index"]
             .agg(avg_hi="mean", min_hi="min").reset_index())
    return rows, daily


def recent_events(engine: sa.Engine, asset_id: str, days: int = 30, tz: str = "Asia/Colombo") -> pd.DataFrame:
    """Failures, repairs/maintenance, manual HI resets and lockouts, newest first."""
    f, m, r, fe = schema.failure_log, schema.maintenance_log, schema.hi_reset_log, schema.fault_events
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        last = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
        lo = (pd.Timestamp(last) - pd.Timedelta(days=days)).to_pydatetime() if last is not None else None

        def q(table, cols, col="ts", extra=None):
            stmt = sa.select(*cols).where(table.c.asset_id == asset_id)
            if lo is not None:
                stmt = stmt.where(table.c[col] > lo)
            if extra is not None:
                stmt = stmt.where(extra)
            return read_df(conn, stmt)

        fails = q(f, [f.c.ts, f.c.component, f.c.description])
        maint = q(m, [m.c.ts, m.c.component, m.c.maint_type, m.c.description])
        resets = q(r, [r.c.ts, r.c.part_groups, r.c.note])
        locks = q(fe, [fe.c.start_ts, fe.c.fault_type, fe.c.max_fault_current_a], col="start_ts",
                  extra=fe.c.outcome == "LOCKOUT")
    parts = []                                       # build each part only when it has rows
    if not fails.empty:
        parts.append(pd.DataFrame({"ts": fails["ts"], "kind": "FAILURE",
                                   "detail": fails["component"] + ": " + fails["description"]}))
    if not maint.empty:
        parts.append(pd.DataFrame({"ts": maint["ts"], "kind": maint["maint_type"].map(
            lambda x: "REPAIR" if x == "CORRECTIVE" else "MAINTENANCE"),
                                   "detail": maint["component"] + ": " + maint["description"]}))
    if not resets.empty:
        parts.append(pd.DataFrame({"ts": resets["ts"], "kind": "RESET",
                                   "detail": "HI reset: " + resets["part_groups"] + resets["note"].fillna("")
                                   .map(lambda n: f" ({n})" if n else "")}))
    if not locks.empty:
        amps = pd.to_numeric(locks["max_fault_current_a"], errors="coerce").round(0).astype("Int64").astype(str)
        parts.append(pd.DataFrame({"ts": locks["start_ts"], "kind": "LOCKOUT",
                                   "detail": locks["fault_type"] + " fault, " + amps + " A"}))
    if not parts:
        return pd.DataFrame(columns=["ts_local", "kind", "detail"])
    ev = pd.concat(parts, ignore_index=True)
    ev["ts_local"] = _local(ev["ts"], tz)
    return ev.sort_values("ts_local", ascending=False, kind="stable")[["ts_local", "kind", "detail"]] \
        .reset_index(drop=True)


# ------------------------------------------------------------------ futuristic dashboard helpers
def branches(cfg: dict) -> pd.DataFrame:
    """LECO branch offices (from config.yaml areas with an anchor)."""
    rows = [{"name": a["name"], "latitude": float(a["anchor"][0]), "longitude": float(a["anchor"][1])}
            for a in cfg.get("areas", []) if a.get("anchor")]
    return pd.DataFrame(rows, columns=["name", "latitude", "longitude"])


def branch_summary(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["area_name", "reclosers", "avg_hi", "alerts", "ml_high"])
    alerts = snapshot["status"].isin(["OFFLINE", "FAULTY", "LOCKOUT", "CRITICAL", "POOR"])
    ml_high = snapshot.get("ml_class", pd.Series(index=snapshot.index, dtype=object)).eq("HIGH")
    g = snapshot.assign(_a=alerts, _m=ml_high).groupby("area_name")
    out = g.agg(reclosers=("asset_id", "size"), avg_hi=("health_index", "mean"), alerts=("_a", "sum"),
                ml_high=("_m", "sum")).reset_index()
    return out.sort_values("avg_hi")


def alert_feed(engine: sa.Engine, hours: int = 24, tz: str = "Asia/Colombo", limit: int = 40) -> pd.DataFrame:
    """Fleet-wide recent events: failures, repairs, maintenance, HI resets and lockouts."""
    f, m, r, fe = schema.failure_log, schema.maintenance_log, schema.hi_reset_log, schema.fault_events
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        last = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
        if last is None:
            return pd.DataFrame(columns=["ts_local", "kind", "asset_id", "detail"])
        lo = (pd.Timestamp(last) - pd.Timedelta(hours=hours)).to_pydatetime()
        fails = read_df(conn, sa.select(f.c.ts, f.c.asset_id, f.c.component, f.c.description).where(f.c.ts > lo))
        maint = read_df(conn, sa.select(m.c.ts, m.c.asset_id, m.c.component, m.c.maint_type, m.c.description)
                        .where(m.c.ts > lo))
        resets = read_df(conn, sa.select(r.c.ts, r.c.asset_id, r.c.part_groups, r.c.note).where(r.c.ts > lo))
        locks = read_df(conn, sa.select(fe.c.start_ts, fe.c.asset_id, fe.c.fault_type, fe.c.max_fault_current_a)
                        .where(fe.c.start_ts > lo, fe.c.outcome == "LOCKOUT"))
    parts = []
    if not fails.empty:
        parts.append(pd.DataFrame({"ts": fails["ts"], "kind": "FAILURE", "asset_id": fails["asset_id"],
                                   "detail": fails["component"] + ": " + fails["description"]}))
    if not maint.empty:
        parts.append(pd.DataFrame({"ts": maint["ts"], "kind": maint["maint_type"].map(
            lambda x: "REPAIR" if x == "CORRECTIVE" else "MAINTENANCE"), "asset_id": maint["asset_id"],
            "detail": maint["component"] + ": " + maint["description"]}))
    if not resets.empty:
        parts.append(pd.DataFrame({"ts": resets["ts"], "kind": "RESET", "asset_id": resets["asset_id"],
                                   "detail": "HI reset: " + resets["part_groups"]}))
    if not locks.empty:
        amps = pd.to_numeric(locks["max_fault_current_a"], errors="coerce").round(0).astype("Int64").astype(str)
        parts.append(pd.DataFrame({"ts": locks["start_ts"], "kind": "LOCKOUT", "asset_id": locks["asset_id"],
                                   "detail": locks["fault_type"] + " fault, " + amps + " A"}))
    if not parts:
        return pd.DataFrame(columns=["ts_local", "kind", "asset_id", "detail"])
    ev = pd.concat(parts, ignore_index=True)
    ev["ts_local"] = _local(ev["ts"], tz)
    return ev.sort_values("ts_local", ascending=False).head(limit)[["ts_local", "kind", "asset_id", "detail"]] \
        .reset_index(drop=True)


def daily_summary_frame(engine: sa.Engine, days: int = 30) -> pd.DataFrame:
    d = schema.ar_daily_summary
    with engine.connect() as conn:
        last = conn.execute(sa.select(sa.func.max(d.c.day))).scalar()
        if last is None:
            return pd.DataFrame(columns=[c.name for c in d.columns])
        lo = pd.Timestamp(last) - pd.Timedelta(days=days)
        return read_df(conn, sa.select(d).where(d.c.day > lo.date()).order_by(d.c.asset_id, d.c.day))


def ml_history(engine: sa.Engine, asset_id: str, days: int = 30, tz: str = "Asia/Colombo") -> pd.DataFrame:
    return _prediction_history(engine, schema.ar_ml_prediction_15min, asset_id, days, tz)


def dl_history(engine: sa.Engine, asset_id: str, days: int = 30, tz: str = "Asia/Colombo") -> pd.DataFrame:
    return _prediction_history(engine, schema.ar_dl_prediction_15min, asset_id, days, tz)


def _prediction_history(engine: sa.Engine, p: sa.Table, asset_id: str, days: int, tz: str) -> pd.DataFrame:
    with engine.connect() as conn:
        last = conn.execute(sa.select(sa.func.max(p.c.ts)).where(p.c.asset_id == asset_id)).scalar()
        if last is None:
            return pd.DataFrame(columns=["ts_local", "p_fail_30d"])
        lo = (pd.Timestamp(last) - pd.Timedelta(days=days)).to_pydatetime()
        rows = read_df(conn, sa.select(p.c.ts, p.c.p_fail_30d).where(p.c.asset_id == asset_id, p.c.ts > lo)
                       .order_by(p.c.ts))
    rows["ts_local"] = _local(rows["ts"], tz)
    return rows[["ts_local", "p_fail_30d"]]


def latest_variables(engine: sa.Engine, asset_id: str) -> dict:
    v = schema.ar_variables_15min
    with engine.connect() as conn:
        return _latest_row(conn, v, asset_id, [c.name for c in v.columns if c.name not in ("asset_id", "computed_at")])


def reset_log(engine: sa.Engine, limit: int = 50, tz: str = "Asia/Colombo") -> pd.DataFrame:
    r = schema.hi_reset_log
    with engine.connect() as conn:
        df = read_df(conn, sa.select(r.c.ts, r.c.asset_id, r.c.part_groups, r.c.note, r.c.created_at)
                     .order_by(r.c.created_at.desc()).limit(limit))
    if df.empty:
        return pd.DataFrame(columns=["maintenance time", "recloser", "parts", "note"])
    return pd.DataFrame({"maintenance time": _local(df["ts"], tz), "recloser": df["asset_id"],
                         "parts": df["part_groups"], "note": df["note"].fillna("")})


def model_status(cfg: dict) -> dict:
    """Installed ML model info, after the same checks the pipeline makes (without loading the model).

    installed=False with an "error" means a model is there but cannot be used (shown in the UI).
    """
    from pathlib import Path

    from .ml_scoring import read_meta
    folder = Path(cfg["paths"]["models_dir"])
    try:
        meta = read_meta(folder)
        if meta is None:
            return {"installed": False, "folder": str(folder)}
        return {"installed": True, "folder": str(folder),
                "version": str(meta.get("model_version", "unknown")), "threshold": float(meta["threshold"]),
                "metrics": meta.get("metrics") or {}, "trained": meta.get("trained_utc"),
                "data_until": meta.get("data_until_utc")}
    except Exception as exc:                             # noqa: BLE001 - shown in the UI
        return {"installed": False, "folder": str(folder), "error": str(exc)}


# ------------------------------------------------------------------ deep-learning model (models/dl)
def dl_model_status(cfg: dict) -> dict:
    """Installed deep-learning model info, after the same checks the pipeline makes (without loading it)."""
    from pathlib import Path

    from .dl_scoring import read_dl_meta
    folder = Path(cfg["paths"]["dl_models_dir"])
    try:
        meta = read_dl_meta(folder)
        if meta is None:
            return {"installed": False, "folder": str(folder)}
        return {"installed": True, "folder": str(folder), "version": str(meta.get("model_version", "unknown")),
                "arch": str(meta.get("arch", "?")), "threshold": float(meta["threshold"]),
                "window": f"{meta['window_steps']} steps x {meta['step_hours']} h",
                "metrics": meta.get("metrics") or {}, "trained": meta.get("trained_utc"),
                "data_until": meta.get("data_until_utc")}
    except Exception as exc:                             # noqa: BLE001 - shown in the UI
        return {"installed": False, "folder": str(folder), "error": str(exc)}


DL_COLUMNS = ["dl_p", "dl_class", "dl_healthy", "dl_component"]


def dl_snapshot(engine: sa.Engine) -> pd.DataFrame:
    """Newest deep-learning prediction of every recloser (one index seek per recloser)."""
    am, p = schema.asset_master, schema.ar_dl_prediction_15min
    cols = ["ts", "p_fail_30d", "dl_risk_class", "dl_healthy", "top_component"]
    with engine.connect() as conn:
        ids = list(conn.execute(sa.select(am.c.asset_id).order_by(am.c.asset_id)).scalars())
        rows = [{"asset_id": aid, **_latest_row(conn, p, aid, cols)} for aid in ids]
    out = pd.DataFrame(rows, columns=["asset_id", *cols]).rename(columns={
        "ts": "dl_ts", "p_fail_30d": "dl_p", "dl_risk_class": "dl_class", "top_component": "dl_component"})
    out["dl_p"] = pd.to_numeric(out["dl_p"], errors="coerce")
    return out


def with_dl(snapshot: pd.DataFrame, dl: pd.DataFrame, interval_minutes: int = 15) -> pd.DataFrame:
    """fleet_snapshot rows + their newest deep-learning values.

    Values more than one interval behind the HI (model removed or broken, scoring failed) are not shown
    as current: dl_stale marks them and they are blanked.
    """
    out = snapshot.merge(dl, on="asset_id", how="left")
    lag = pd.to_datetime(out["hi_ts"]) - pd.to_datetime(out["dl_ts"])
    out["dl_stale"] = (lag > pd.Timedelta(minutes=int(interval_minutes))).fillna(False).astype(bool)
    if out["dl_stale"].any():
        out[DL_COLUMNS] = out[DL_COLUMNS].astype(object)
        out.loc[out["dl_stale"], DL_COLUMNS] = None
    out["dl_p"] = pd.to_numeric(out["dl_p"], errors="coerce")
    return out
