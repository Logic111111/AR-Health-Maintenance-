"""Batch writes and crash recovery."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import sqlalchemy as sa

from . import schema

# Event tables with simulator-assigned ids: table -> (id column, state counter key)
ID_TABLES = {
    "co_operations": ("op_id", "op"),
    "maintenance_log": ("maint_id", "maint"),
    "alarm_events": ("event_id", "alarm"),
    "battery_tests": ("test_id", "test"),
    "comms_events": ("event_id", "comms"),
    "failure_log": ("failure_id", "failure"),
    "fault_events": ("fault_id", "fault"),
}


def insert_rows(conn: sa.Connection, table: sa.Table, rows: list[dict],
                ignore_duplicates: bool = False, batch_size: int = 5000) -> int:
    if not rows:
        return 0
    stmt = table.insert()
    if ignore_duplicates:
        stmt = stmt.prefix_with("IGNORE", dialect=("mysql", "mariadb")).prefix_with(
            "OR IGNORE", dialect="sqlite")
    for i in range(0, len(rows), batch_size):
        conn.execute(stmt, rows[i:i + batch_size])
    return len(rows)


def write_chunk(conn: sa.Connection, rows_by_table: dict[str, list[dict]],
                asset_updates: Iterable[dict] = (), batch_size: int = 5000) -> dict[str, int]:
    """Insert one chunk of rows in foreign-key order (the caller owns the transaction)."""
    counts: dict[str, int] = {}
    if rows_by_table.get("asset_master"):
        counts["asset_master"] = insert_rows(conn, schema.asset_master, rows_by_table["asset_master"])
    for upd in asset_updates:
        values = {k: v for k, v in upd.items() if k != "asset_id"}
        if values:
            conn.execute(schema.asset_master.update()
                         .where(schema.asset_master.c.asset_id == upd["asset_id"]).values(**values))
    for name in schema.INSERT_ORDER:
        if name == "asset_master":
            continue
        rows = rows_by_table.get(name)
        if rows:
            ignore = name in ("telemetry_15min", "firmware_release")
            counts[name] = insert_rows(conn, schema.TABLES[name], rows, ignore, batch_size)
    return counts


def rollback_after(engine: sa.Engine, last_ts: datetime, next_ids: dict[str, int]) -> None:
    """Delete raw and calculated rows written after the last saved simulator state.

    Called on start-up so a crash between a DB commit and the state save is
    regenerated identically (generation is deterministic).
    """
    t = schema.TABLES
    with engine.begin() as conn:
        for name in ("ar_ml_prediction_15min", "ar_health_index_15min", "ar_variables_15min", "telemetry_15min"):
            conn.execute(t[name].delete().where(t[name].c.ts > last_ts))
        for name in ("co_operations", "maintenance_log", "alarm_events", "battery_tests",
                     "comms_events", "failure_log"):
            id_col, key = ID_TABLES[name]
            conn.execute(t[name].delete().where(t[name].c[id_col] >= next_ids[key]))
        fe = t["fault_events"]
        conn.execute(fe.delete().where(sa.or_(fe.c.fault_id >= next_ids["fault"],
                                              fe.c.end_ts > last_ts)))
