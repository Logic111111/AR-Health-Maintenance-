"""Phase 1 pipeline: collect raw data -> store -> V1-V42 -> Health Index.

All entry points are idempotent: they continue from what is already in the database
and the simulator state, so the scheduler, Windows Task Scheduler and manual runs can
be mixed freely (a lock file prevents two runs at the same time).
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.engine import create_all, drop_all, ensure_database, make_engine, with_retry
from .db.reader import asset_frame, asset_start_ms, df_to_rows, load_inputs, read_df
from .db.writer import insert_rows, write_chunk
from .health_index import compute_health_index
from .locations import ensure_locations
from .sim.timeutil import DAY_MS, MIN_MS, dt_to_ms, local_date, local_to_ms, ms_to_dt, parse_date, to_local
from .sources.simulated import SimulatedNojaSource
from .variables import compute_variables, to_ms

log = logging.getLogger(__name__)


LOCK_WAIT_S = 120            # how long collectors and recalculations wait for another run to finish


def step_ms(cfg: dict) -> int:
    return int(cfg["interval_minutes"]) * MIN_MS


def now_ms(cfg: dict) -> int:
    t = dt_to_ms(datetime.now(timezone.utc))
    return t - t % step_ms(cfg)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


# ------------------------------------------------------------------ locking
def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


@contextmanager
def run_lock(cfg: dict, wait_s: float = 0.0):
    """Exclusive OS-level lock so collectors and HI resets never write at the same time.

    Waits up to ``wait_s`` seconds (retrying every 5 s) before giving up.
    """
    path = cfg["paths"]["state_dir"] / "ar_gen.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        deadline = time.monotonic() + wait_s
        while not _try_lock(fh):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"collector busy - please try again (lock {path})")
            time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        fh.close()


# ------------------------------------------------------------------ database setup
def get_engine(cfg: dict) -> sa.Engine:
    return make_engine(cfg["database_url"])


def init_db(cfg: dict, reset: bool = False) -> sa.Engine:
    ensure_database(cfg["database_url"])
    engine = get_engine(cfg)
    if reset:
        drop_all(engine)
        state = cfg["paths"]["state_dir"] / "sim_state.json"
        gt = cfg["paths"]["output_dir"] / "ground_truth_daily.csv"
        for p in (state, gt):
            if p.exists():
                p.unlink()
        log.info("Dropped all tables and removed the simulator state")
    create_all(engine)
    n = ensure_locations(engine, cfg)
    if n:
        log.info("Placed %d reclosers on the map (simulated positions)", n)
    log.info("Database ready: %s", engine.url.render_as_string(hide_password=True))
    return engine


def default_start_ms(cfg: dict, days: int | None = None) -> int:
    days = cfg["default_backfill_days"] if days is None else days
    today = to_local(now_ms(cfg), cfg["timezone"]).date()
    return local_to_ms(today - timedelta(days=days), 0.0, cfg["timezone"])


def seed(cfg: dict, start: str | None = None, force: bool = False) -> SimulatedNojaSource:
    engine = get_engine(cfg)
    create_all(engine)
    source = SimulatedNojaSource(cfg)
    with engine.connect() as conn:
        n_existing = conn.execute(sa.select(sa.func.count()).select_from(schema.asset_master)).scalar()
    if (source.exists() or n_existing) and not force:
        raise RuntimeError("Fleet already seeded. Use 'init-db --reset' first to start again.")
    start_ms = local_to_ms(parse_date(start), 0.0, cfg["timezone"]) if start else default_start_ms(cfg)
    start_ms -= start_ms % step_ms(cfg)
    rows = source.seed(start_ms)
    with engine.begin() as conn:
        write_chunk(conn, rows, (), cfg["db"]["batch_size"])
    source.commit_seed()
    ensure_locations(engine, cfg)
    log.info("Seeded %d reclosers starting %s UTC (hidden profiles: %s)", len(rows["asset_master"]),
             ms_to_dt(start_ms), cfg["paths"]["output_dir"] / "asset_profiles.csv")
    return source


# ------------------------------------------------------------------ collection
def collect(cfg: dict, engine: sa.Engine, source, until_ms: int, chunk_intervals: int = 96) -> int:
    """Collect raw data up to ``until_ms`` and store it; returns the number of intervals."""
    step = step_ms(cfg)
    first = source.last_collected_ms()
    total = max(0, (until_ms - first) // step)
    if total == 0:
        return 0
    write = with_retry(cfg["db"].get("retries", 5))(_write_batch)
    t_start, done, last_log = time.time(), 0, time.time()
    for batch in source.collect(until_ms, chunk_intervals):
        write(engine, batch, cfg["db"]["batch_size"])
        source.commit(batch)
        done = (batch.last_ms - first) // step
        if time.time() - last_log > 10 or done == total:
            rate = done / max(1e-6, time.time() - t_start)
            log.info("collected %s (%d/%d intervals, %.0f intervals/s)", ms_to_dt(batch.last_ms), done, total, rate)
            last_log = time.time()
    return done


def _write_batch(engine, batch, batch_size):
    with engine.begin() as conn:
        write_chunk(conn, batch.rows, batch.asset_updates, batch_size)


# ------------------------------------------------------------------ variables + HI
def _round_frame(df: pd.DataFrame, nd: int = 4) -> pd.DataFrame:
    floats = df.select_dtypes(include="float").columns
    df[floats] = df[floats].round(nd)
    return df


_LOAD = object()        # compute_range(model=...): load the installed model itself


def compute_range(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model=_LOAD) -> int:
    """(Re)compute V1-V42 and the HI (and ML risk, daily summary) for interval ends in (start_ms, end_ms]."""
    step = step_ms(cfg)
    start_ms -= start_ms % step
    chunk = int(cfg["variables"]["calc_chunk_days"]) * DAY_MS
    with engine.connect() as conn:
        a_start = asset_start_ms(conn, step)
    if not a_start:
        return 0
    start_ms = max(start_ms, min(a_start.values()))
    if model is _LOAD:
        model = _try_load_model(cfg)
    retry = with_retry(cfg["db"].get("retries", 5))
    total, s, t_start = 0, start_ms, time.time()
    while s < end_ms:
        e = min(end_ms, s + chunk)
        with engine.connect() as conn:
            inp = load_inputs(conn, s, e, cfg, a_start)
        variables = compute_variables(inp, s, e, cfg)
        hi = compute_health_index(variables, inp.assets, cfg["health_index"])
        stamp = utc_now()
        v_rows = df_to_rows(_round_frame(variables.assign(computed_at=stamp)))
        h_rows = df_to_rows(hi.assign(computed_at=stamp))
        retry(_replace_calculated)(engine, s, e, v_rows, h_rows, cfg)
        if model is not None and not _score_safely(cfg, engine, s, e, model):
            model = None            # keeps the ML gap at the end, where the next run catches it up
        retry(update_daily_summary)(cfg, engine, s, e)
        total += len(v_rows)
        log.info("variables + HI up to %s (%d rows, %.0f s)", ms_to_dt(e), total, time.time() - t_start)
        s = e
    return total


def _replace_calculated(engine, s, e, v_rows, h_rows, cfg):
    lo, hi = ms_to_dt(s), ms_to_dt(e)
    with engine.begin() as conn:
        for table in (schema.ar_health_index_15min, schema.ar_variables_15min):
            conn.execute(table.delete().where(table.c.ts > lo, table.c.ts <= hi))
        insert_rows(conn, schema.ar_variables_15min, v_rows, batch_size=cfg["db"]["batch_size"])
        insert_rows(conn, schema.ar_health_index_15min, h_rows, batch_size=cfg["db"]["batch_size"])


# ------------------------------------------------------------------ ML scoring + daily summary
def _try_load_model(cfg: dict):
    """The installed Colab model, or None (a broken/incompatible bundle never stops the pipeline)."""
    from .ml_scoring import load_model
    try:
        return load_model(cfg["paths"]["models_dir"])
    except Exception as exc:                           # noqa: BLE001 - report and carry on without ML
        log.warning("ML model not used: %s", exc)
        return None


def _score_safely(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model) -> bool:
    """ML scoring never stops the Health Index: a failure is logged and caught up by the next run."""
    try:
        ml_score_range(cfg, engine, start_ms, end_ms, model)
        return True
    except Exception as exc:                           # noqa: BLE001 - report, HI goes on
        log.warning("ML scoring failed for %s - %s (the next run catches it up): %s",
                    ms_to_dt(start_ms), ms_to_dt(end_ms), exc)
        return False


def _ml_catch_up(cfg: dict, engine: sa.Engine, last_hi_ms: int, model):
    """Score HI intervals an earlier run left without ML (failed write, killed process, model added back).

    Returns the model, or None when scoring still fails (this run then skips ML; the next one retries).
    A database never scored before is left to ``ml-score`` (see docs/ML_COLAB_GUIDE.md).
    """
    p = schema.ar_ml_prediction_15min
    with engine.connect() as conn:
        ts = conn.execute(sa.select(sa.func.max(p.c.ts))).scalar()
    if ts is None:
        return model
    done = int(to_ms([ts])[0])
    if done >= last_hi_ms:
        return model
    log.info("ML catch-up: scoring %s - %s", ms_to_dt(done), ms_to_dt(last_hi_ms))
    try:
        score_and_summarise(cfg, engine, done, last_hi_ms, model)
    except Exception as exc:                           # noqa: BLE001 - report, HI goes on
        log.warning("ML catch-up failed (retried next run): %s", exc)
        return None
    return model


def ml_score_range(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model) -> int:
    """Write ML predictions for interval ends in (start_ms, end_ms] (8 days of history for trends)."""
    from .ml_features import build_features
    v, am, p = schema.ar_variables_15min, schema.asset_master, schema.ar_ml_prediction_15min
    with engine.connect() as conn:
        df = read_df(conn, sa.select(v, am.c.model, am.c.comms_channel)
                     .select_from(v.join(am, am.c.asset_id == v.c.asset_id))
                     .where(v.c.ts > ms_to_dt(start_ms - 8 * DAY_MS), v.c.ts <= ms_to_dt(end_ms)))
    if df.empty:
        return 0
    X = build_features(df, int(cfg["interval_minutes"]))
    ts_ms = to_ms(df["ts"])
    sel = (ts_ms > start_ms) & (ts_ms <= end_ms)
    if not sel.any():
        return 0
    pred = model.predict(X[sel])
    out = df.loc[sel, ["asset_id", "ts"]].join(pred[["p_fail_30d", "ml_risk_class", "ml_healthy", "top_component",
                                                     "top_component_p"]])
    out["model_version"] = model.model_version
    out["computed_at"] = utc_now()
    rows = df_to_rows(out)
    with engine.begin() as conn:
        conn.execute(p.delete().where(p.c.ts > ms_to_dt(start_ms), p.c.ts <= ms_to_dt(end_ms)))
        insert_rows(conn, p, rows, batch_size=cfg["db"]["batch_size"])
    return len(rows)


def update_daily_summary(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int) -> int:
    """Recompute the daily summary of every local day that overlaps (start_ms, end_ms].

    A day is only replaced by a summary of at least as many 15-min rows, so a day whose first rows
    were deleted by prune-history keeps the summary it had when it was complete.
    """
    tz, step = cfg["timezone"], step_ms(cfg)
    if end_ms <= start_ms:
        return 0
    first_day, last_day = local_date(start_ms + step, tz), local_date(end_ms, tz)
    lo = ms_to_dt(local_to_ms(first_day, 0.0, tz))
    hi = ms_to_dt(local_to_ms(last_day + timedelta(days=1), 0.0, tz))
    h, p, d = schema.ar_health_index_15min, schema.ar_ml_prediction_15min, schema.ar_daily_summary
    with engine.connect() as conn:
        hi_df = read_df(conn, sa.select(h.c.asset_id, h.c.ts, h.c.health_index).where(h.c.ts >= lo, h.c.ts < hi))
        ml_df = read_df(conn, sa.select(p.c.asset_id, p.c.ts, p.c.p_fail_30d, p.c.ml_healthy)
                        .where(p.c.ts >= lo, p.c.ts < hi))
    if hi_df.empty:
        return 0
    healthy_min = float(cfg["health_index"]["healthy_min"])
    hi_df["day"] = pd.to_datetime(hi_df["ts"]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.date
    hi_df["unhealthy"] = (hi_df["health_index"] < healthy_min).astype(float) * step / 3_600_000
    daily = (hi_df.groupby(["asset_id", "day"])
             .agg(intervals=("health_index", "size"), avg_hi=("health_index", "mean"),
                  min_hi=("health_index", "min"), max_hi=("health_index", "max"),
                  unhealthy_hours=("unhealthy", "sum")).reset_index())
    daily["hi_healthy"] = daily["avg_hi"] >= healthy_min
    if not ml_df.empty:
        ml_df["day"] = pd.to_datetime(ml_df["ts"]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.date
        ml_df["ml_healthy"] = ml_df["ml_healthy"].astype(bool)
        ml = (ml_df.groupby(["asset_id", "day"])
              .agg(avg_p_fail=("p_fail_30d", "mean"), max_p_fail=("p_fail_30d", "max"),
                   ml_healthy=("ml_healthy", "all")).reset_index())
        daily = daily.merge(ml, on=["asset_id", "day"], how="left")
    else:
        daily["avg_p_fail"] = daily["max_p_fail"] = float("nan")
        daily["ml_healthy"] = None
    has_ml = daily["ml_healthy"].notna()
    daily["ml_healthy"] = daily["ml_healthy"].astype(object).where(has_ml, None)
    daily["agree"] = [bool(a == b) if m else None for a, b, m in
                      zip(daily["hi_healthy"], daily["ml_healthy"], has_ml)]
    daily["updated_at"] = utc_now()
    with engine.connect() as conn:
        old = read_df(conn, sa.select(d).where(d.c.day >= first_day, d.c.day <= last_day))
    if not old.empty:
        old["day"] = pd.to_datetime(old["day"]).dt.date
        key = ["asset_id", "day"]
        old_n = old.set_index(key)["intervals"]
        new_n = daily.set_index(key)["intervals"].reindex(old_n.index).fillna(0)
        cut = old_n.index[old_n > new_n]
        if len(cut):
            daily = pd.concat([daily[~daily.set_index(key).index.isin(cut)],
                               old.set_index(key).loc[cut].reset_index()[daily.columns]], ignore_index=True)
    rows = df_to_rows(daily)
    with engine.begin() as conn:
        conn.execute(d.delete().where(d.c.day >= first_day, d.c.day <= last_day))
        insert_rows(conn, d, rows, batch_size=cfg["db"]["batch_size"])
    return len(rows)


def score_and_summarise(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model) -> int:
    """ml-score: predictions + daily summary for a stored range, chunk by chunk."""
    chunk = int(cfg["variables"]["calc_chunk_days"]) * DAY_MS
    total, s = 0, start_ms - start_ms % step_ms(cfg)
    while s < end_ms:
        e = min(end_ms, s + chunk)
        total += ml_score_range(cfg, engine, s, e, model)
        update_daily_summary(cfg, engine, s, e)
        log.info("ML scores up to %s (%d rows)", ms_to_dt(e), total)
        s = e
    return total


def recompute_hi(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int) -> int:
    """Re-score stored variables (e.g. after changing HI weights/thresholds)."""
    chunk = int(cfg["variables"]["calc_chunk_days"]) * DAY_MS
    v, h = schema.ar_variables_15min, schema.ar_health_index_15min
    total, s = 0, start_ms
    with engine.connect() as conn:
        assets = asset_frame(conn)
    while s < end_ms:
        e = min(end_ms, s + chunk)
        with engine.connect() as conn:
            variables = read_df(conn, sa.select(v).where(v.c.ts > ms_to_dt(s), v.c.ts <= ms_to_dt(e)))
        if not variables.empty:
            hi = compute_health_index(variables, assets, cfg["health_index"])
            rows = df_to_rows(hi.assign(computed_at=utc_now()))
            with engine.begin() as conn:
                conn.execute(h.delete().where(h.c.ts > ms_to_dt(s), h.c.ts <= ms_to_dt(e)))
                insert_rows(conn, h, rows, batch_size=cfg["db"]["batch_size"])
            update_daily_summary(cfg, engine, s, e)
            total += len(rows)
            log.info("HI up to %s (%d rows)", ms_to_dt(e), total)
        s = e
    return total


def last_calculated_ms(engine: sa.Engine) -> int | None:
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        ts = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
    return None if ts is None else int(to_ms([ts])[0])


def compute_pending(cfg: dict, engine: sa.Engine, until_ms: int) -> int:
    last = last_calculated_ms(engine)
    if last is None:
        with engine.connect() as conn:
            starts = asset_start_ms(conn, step_ms(cfg))
        if not starts:
            return 0
        last = min(starts.values())
    model = _try_load_model(cfg)
    if model is not None:
        model = _ml_catch_up(cfg, engine, last, model)
    if until_ms <= last:
        return 0
    return compute_range(cfg, engine, last, until_ms, model=model)


# ------------------------------------------------------------------ top-level runs
def update(cfg: dict, until_ms: int | None = None, compute: bool = True, chunk_intervals: int = 96) -> dict:
    """Collect up to ``until_ms`` (default: the interval that just ended) and update variables + HI."""
    with run_lock(cfg, wait_s=LOCK_WAIT_S):      # waits for a dashboard reset in progress
        engine = get_engine(cfg)
        create_all(engine)
        source = SimulatedNojaSource(cfg)
        source.load()
        _check_fleet_in_db(engine, source)
        source.recover(engine)
        until = (until_ms if until_ms is not None else now_ms(cfg))
        until -= until % step_ms(cfg)
        n = collect(cfg, engine, source, until, chunk_intervals)
        rows = compute_pending(cfg, engine, source.last_collected_ms()) if compute else 0
        engine.dispose()
        return {"intervals": n, "variable_rows": rows, "last_ts": ms_to_dt(source.last_collected_ms())}


def _check_fleet_in_db(engine: sa.Engine, source: SimulatedNojaSource) -> None:
    with engine.connect() as conn:
        ids = set(conn.execute(sa.select(schema.asset_master.c.asset_id)).scalars())
    missing = [a.asset_id for a in source.state.assets if a.asset_id not in ids]
    if missing:
        raise RuntimeError(
            f"The database {engine.url.render_as_string(hide_password=True)} does not contain the seeded fleet "
            f"({len(missing)} reclosers missing, e.g. {missing[0]}). Either copy the existing data with "
            "'copy-db --from <old DATABASE_URL>', or start again with 'init-db --reset' and 'seed'.")


def copy_db(cfg: dict, source_url: str, batch_size: int = 20000) -> dict:
    """Copy every table from ``source_url`` into the configured (empty) database."""
    ensure_database(cfg["database_url"])
    src, dst = make_engine(source_url), get_engine(cfg)
    create_all(dst)
    with dst.connect() as conn:
        if conn.execute(sa.select(sa.func.count()).select_from(schema.asset_master)).scalar():
            raise RuntimeError("Target database is not empty - run 'init-db --reset' on it first")
    counts = {}
    for name in schema.INSERT_ORDER:
        table = schema.TABLES[name]
        n = 0
        with src.connect() as sconn:
            result = sconn.execution_options(yield_per=batch_size).execute(sa.select(table))
            for part in result.partitions():
                rows = [dict(r._mapping) for r in part]
                with dst.begin() as dconn:
                    insert_rows(dconn, table, rows, batch_size=cfg["db"]["batch_size"])
                n += len(rows)
                log.info("copy %s: %d rows", name, n)
        counts[name] = n
    return counts


def live_snapshot(cfg: dict, engine: sa.Engine | None = None):
    """Newest interval: telemetry + Health Index of every recloser (left-joined, so a
    recloser whose controller is powered down still appears)."""
    engine = engine or get_engine(cfg)
    h, t = schema.ar_health_index_15min, schema.telemetry_15min
    with engine.connect() as conn:
        ts = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
        if ts is None:
            return None, pd.DataFrame()
        stmt = (sa.select(h.c.asset_id, t.c.switch_state, t.c.ia, t.c.uab, t.c.battery_voltage,
                          t.c.battery_current, t.c.battery_charge_pct, t.c.ac_supply_on, t.c.comms_connected,
                          t.c.signal_strength_dbm, t.c.weather, t.c.co_total, h.c.health_index, h.c.health_class,
                          h.c.is_healthy, h.c.worst_variables)
                .select_from(h.outerjoin(t, sa.and_(t.c.asset_id == h.c.asset_id, t.c.ts == h.c.ts)))
                .where(h.c.ts == ts).order_by(h.c.asset_id))
        return ts, read_df(conn, stmt)


# ------------------------------------------------------------------ prune old history
def backup_sqlite(cfg: dict, dest) -> Path:
    """Consistent copy of the SQLite database (safe while other programs read it)."""
    import sqlite3

    from sqlalchemy.engine import make_url
    url = make_url(cfg["database_url"])
    if url.get_backend_name() != "sqlite":
        raise RuntimeError("the built-in backup is for SQLite only - back up MySQL/MariaDB with mysqldump, "
                           "then use --no-backup")
    dest = Path(dest)
    if dest.exists():
        raise RuntimeError(f"backup file {dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src, dst = sqlite3.connect(url.database), sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def _alarm_ids_still_needed(conn, before) -> set:
    """Old alarm rows that later rows depend on: conditions still RAISED and controllers still
    powered down at the cut-off (keeps RAISED/CLEARED pairing and telemetry-gap explanations valid)."""
    a = schema.alarm_events
    df = read_df(conn, sa.select(a.c.event_id, a.c.asset_id, a.c.signal_name, a.c.state)
                 .where(a.c.ts <= before).order_by(a.c.asset_id, a.c.signal_name, a.c.ts, a.c.event_id))
    keep: set = set()
    rc = df[df["state"].isin(["RAISED", "CLEARED"])]
    last = rc.groupby(["asset_id", "signal_name"]).tail(1)
    keep |= set(last.loc[last["state"] == "RAISED", "event_id"])
    power = df[(df["state"] == "EVENT") & df["signal_name"].isin(["Shutdown", "Power Restart"])]
    lastp = power.groupby("asset_id").tail(1)
    keep |= set(lastp.loc[lastp["signal_name"] == "Shutdown", "event_id"])
    return {int(x) for x in keep}


def _vacuum_sqlite(cfg: dict) -> float | None:
    """Give the freed space back to the disk (SQLite); returns the file size in MB."""
    import sqlite3

    from sqlalchemy.engine import make_url
    url = make_url(cfg["database_url"])
    if url.get_backend_name() != "sqlite":
        return None
    con = sqlite3.connect(url.database, timeout=60)
    try:
        for attempt in range(5):
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.execute("VACUUM")
                break
            except sqlite3.OperationalError as exc:
                if attempt == 4:
                    log.warning("could not compact the database file now (%s); it will reuse the space", exc)
                time.sleep(3)
    finally:
        con.close()
    return round(os.path.getsize(url.database) / 1e6, 1)


def prune_history(cfg: dict, keep_days: int = 30, event_margin_days: int = 7, backup_to=None,
                  wait_s: float | None = None) -> dict:
    """Delete data older than ``keep_days`` (optionally after a full backup).

    Readings, V1-V42 and HI rows older than the cut-off are deleted. Events are kept
    ``event_margin_days`` longer, because the 30-day calculations still read them. Also kept:
    the current battery's tests, failures still waiting for repair (and failures whose repair
    is kept), alarms still raised and power-downs still in progress at the cut-off.
    """
    if keep_days < 1:
        raise ValueError("keep at least 1 day (the Health Index needs recent history)")
    engine = get_engine(cfg)
    create_all(engine)
    t = schema.TABLES
    with run_lock(cfg, LOCK_WAIT_S if wait_s is None else wait_s):
        latest = last_calculated_ms(engine)
        if latest is None:
            raise ValueError("no data in the database yet")
        cutoff = latest - keep_days * DAY_MS
        ev_cut = cutoff - event_margin_days * DAY_MS
        c_dt, e_dt = ms_to_dt(cutoff), ms_to_dt(ev_cut)
        if backup_to:
            backup_sqlite(cfg, backup_to)
            log.info("full backup written to %s", backup_to)
        deleted: dict = {}
        with engine.begin() as conn:
            keep_alarms = _alarm_ids_still_needed(conn, e_dt)
            m, f, o = t["maintenance_log"], t["failure_log"], t["co_operations"]
            repaired = sorted({int(x) for x in conn.execute(
                sa.select(m.c.failure_id).where(m.c.failure_id.is_not(None))).scalars()})
            for name in ("ar_ml_prediction_15min", "ar_health_index_15min", "ar_variables_15min", "telemetry_15min"):
                deleted[name] = conn.execute(t[name].delete().where(t[name].c.ts <= c_dt)).rowcount
            ds = t["ar_daily_summary"]
            deleted["ar_daily_summary"] = conn.execute(
                ds.delete().where(ds.c.day < local_date(cutoff, cfg["timezone"]))).rowcount
            deleted["co_operations"] = conn.execute(o.delete().where(o.c.ts <= e_dt)).rowcount
            fe = t["fault_events"]
            used = sa.select(o.c.fault_id).where(o.c.fault_id.is_not(None))
            deleted["fault_events"] = conn.execute(
                fe.delete().where(fe.c.start_ts <= e_dt, fe.c.fault_id.not_in(used))).rowcount
            ce = t["comms_events"]
            deleted["comms_events"] = conn.execute(ce.delete().where(ce.c.ts <= e_dt)).rowcount
            a = t["alarm_events"]
            stmt = a.delete().where(a.c.ts <= e_dt)
            if keep_alarms:
                stmt = stmt.where(a.c.event_id.not_in(sorted(keep_alarms)))
            deleted["alarm_events"] = conn.execute(stmt).rowcount
            bt, am = t["battery_tests"], schema.asset_master
            n = 0
            for aid, installed in conn.execute(sa.select(am.c.asset_id, am.c.battery_install_date)).fetchall():
                limit = ev_cut if installed is None else min(ev_cut, local_to_ms(installed, 0.0, cfg["timezone"]))
                n += conn.execute(bt.delete().where(bt.c.asset_id == aid, bt.c.ts < ms_to_dt(limit))).rowcount
            deleted["battery_tests"] = n
            deleted["maintenance_log"] = conn.execute(m.delete().where(m.c.ts <= e_dt)).rowcount
            still_linked = sa.select(m.c.failure_id).where(m.c.failure_id.is_not(None))
            deleted["failure_log"] = conn.execute(
                f.delete().where(f.c.ts <= e_dt, f.c.failure_id.in_(repaired), f.c.failure_id.not_in(still_linked))
            ).rowcount if repaired else 0
        engine.dispose()
        size_mb = _vacuum_sqlite(cfg)
    log.info("pruned data before %s UTC (events before %s UTC): %s", c_dt, e_dt,
             ", ".join(f"{k} {v:,}" for k, v in deleted.items()))
    return {"cutoff": c_dt, "events_cutoff": e_dt, "deleted": deleted, "backup": backup_to, "db_size_mb": size_mb}


# ------------------------------------------------------------------ maintenance done -> HI reset
def _hi_at(engine: sa.Engine, asset_id: str, ts_ms: int) -> float | None:
    h = schema.ar_health_index_15min
    with engine.connect() as conn:
        return conn.execute(sa.select(h.c.health_index)
                            .where(h.c.asset_id == asset_id, h.c.ts == ms_to_dt(ts_ms))).scalar()


def maintenance_done(cfg: dict, asset_id: str, groups: list[str], at_ms: int | None = None, note: str = "",
                     wait_s: float = 60.0) -> dict:
    """Record maintenance on part group(s) and recalculate V1-V42 + HI from that moment.

    groups: any of BATTERY, SWITCHING, COMMS, CONTROLLER, or ALL. at_ms defaults to the
    latest calculated interval. Saving and recalculating happen under the run lock, so a
    reset is never stored without its recalculation.
    """
    from .variables import RESET_GROUPS

    names = [str(g).strip().upper() for g in groups if str(g).strip()]
    bad = [n for n in names if n != "ALL" and n not in RESET_GROUPS]
    if not names or bad:
        raise ValueError(f"unknown part(s) {', '.join(bad) or '(none)'}: use BATTERY, SWITCHING, COMMS, "
                         f"CONTROLLER or ALL")
    stored = "ALL" if "ALL" in names else ",".join(g for g in RESET_GROUPS if g in names)
    engine = get_engine(cfg)
    create_all(engine)
    asset_id = str(asset_id).strip().upper()
    with engine.connect() as conn:
        known = conn.execute(sa.select(schema.asset_master.c.asset_id)
                             .where(schema.asset_master.c.asset_id == asset_id)).scalar()
    if known is None:
        raise ValueError(f"unknown asset {asset_id}")
    asset_id = known                                        # canonical id (MySQL compares case-insensitively)
    if last_calculated_ms(engine) is None:
        raise ValueError("no Health Index data yet - run backfill or run-once first")
    tz = cfg["timezone"]
    at_ms = last_calculated_ms(engine) if at_ms is None else int(at_ms)
    if at_ms > dt_to_ms(datetime.now(timezone.utc)):
        raise ValueError(f"maintenance time {to_local(at_ms, tz):%Y-%m-%d %H:%M} ({tz}) is in the future")
    step = step_ms(cfg)
    first_ts = at_ms + (-at_ms) % step                      # first interval end at or after the reset
    with run_lock(cfg, wait_s):
        last = last_calculated_ms(engine)                   # re-read: the collector may have just added rows
        hi_before = _hi_at(engine, asset_id, first_ts)
        with engine.begin() as conn:
            res = conn.execute(schema.hi_reset_log.insert().values(
                ts=ms_to_dt(at_ms), asset_id=asset_id, part_groups=stored, note=(note or None) and note[:255],
                created_at=utc_now()))
            reset_id = res.inserted_primary_key[0]
        try:
            # A reset after the latest data needs no recalculation: the next collection applies it.
            rows = compute_range(cfg, engine, at_ms - 1, last) if at_ms <= last else 0
        except Exception as exc:
            with engine.begin() as conn:                    # never keep a reset without its recalculation
                conn.execute(schema.hi_reset_log.delete().where(schema.hi_reset_log.c.reset_id == reset_id))
            try:
                compute_range(cfg, engine, at_ms - 1, last)  # put the rows back as they were
            except Exception:
                log.error("could not restore the recalculated rows - run: python main.py compute-variables "
                          "--start %s", to_local(at_ms, tz).date())
            raise RuntimeError(f"recalculation failed, maintenance not saved: {exc}") from exc
        hi_after = _hi_at(engine, asset_id, first_ts)
    engine.dispose()
    log.info("maintenance done on %s (%s) at %s UTC: HI %s -> %s, %d rows recalculated",
             asset_id, stored, ms_to_dt(at_ms), hi_before, hi_after, rows)
    return {"reset_id": reset_id, "recomputed_rows": rows, "hi_before": hi_before, "hi_after": hi_after}


def status(cfg: dict) -> pd.DataFrame:
    engine = get_engine(cfg)
    h = schema.ar_health_index_15min
    latest = sa.select(h.c.asset_id, sa.func.max(h.c.ts).label("ts")).group_by(h.c.asset_id).subquery()
    stmt = (sa.select(h.c.asset_id, h.c.ts, h.c.health_index, h.c.health_class, h.c.is_healthy,
                      h.c.cap_applied, h.c.worst_variables, *[h.c[c] for c in schema.FACTOR_COLUMNS])
            .join(latest, sa.and_(h.c.asset_id == latest.c.asset_id, h.c.ts == latest.c.ts))
            .order_by(h.c.health_index))
    with engine.connect() as conn:
        return read_df(conn, stmt)
