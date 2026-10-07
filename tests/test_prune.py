"""prune-history: keep only recent data (plus what the 30-day calculations still need)."""
import sqlite3

import pandas as pd
import sqlalchemy as sa

from ar_gen import pipeline
from ar_gen.db import schema
from ar_gen.sim.timeutil import DAY_MS, MIN_MS, ms_to_dt
from ar_gen.validate import validate
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)


def _count(cfg, table, where=None):
    eng = pipeline.get_engine(cfg)
    stmt = sa.select(sa.func.count()).select_from(table)
    if where is not None:
        stmt = stmt.where(where)
    with eng.connect() as conn:
        return conn.execute(stmt).scalar()


def _hi_after(cfg, cutoff):
    h = schema.ar_health_index_15min
    with pipeline.get_engine(cfg).connect() as conn:
        return pd.read_sql(sa.select(h.c.asset_id, h.c.ts, h.c.health_index).where(h.c.ts > ms_to_dt(cutoff))
                           .order_by(h.c.asset_id, h.c.ts), conn)


def test_prune_keeps_recent_data_and_a_full_backup(db, tmp_path):
    cfg, start = db
    eng = pipeline.get_engine(cfg)
    latest = pipeline.last_calculated_ms(eng)
    old_failure_ts = ms_to_dt(start + 2 * 60 * MIN_MS)
    with eng.begin() as conn:                                 # an old failure still waiting for repair
        fid = (conn.execute(sa.select(sa.func.max(schema.failure_log.c.failure_id))).scalar() or 0) + 1000
        conn.execute(schema.failure_log.insert().values(failure_id=fid, ts=old_failure_ts, asset_id="AR-003",
                                                        component="COMMS", detected_by="SCADA",
                                                        description="old open failure"))
    before = {name: _count(cfg, t) for name, t in schema.TABLES.items()}
    cutoff = latest - DAY_MS
    hi_kept_before = _hi_after(cfg, cutoff)

    backup = tmp_path / "backup.sqlite"
    res = pipeline.prune_history(cfg, keep_days=1, event_margin_days=1, backup_to=backup)

    with sqlite3.connect(backup) as b:                       # the backup has everything
        for name, n in before.items():
            assert b.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == n, name
    t = schema.telemetry_15min
    assert _count(cfg, t, t.c.ts <= ms_to_dt(cutoff)) == 0 and _count(cfg, t) > 0
    o = schema.co_operations
    assert _count(cfg, o, o.c.ts <= ms_to_dt(cutoff - DAY_MS)) == 0
    assert res["deleted"]["telemetry_15min"] > 0
    f = schema.failure_log
    assert _count(cfg, f, f.c.failure_id == fid) == 1                   # unrepaired failure kept
    pd.testing.assert_frame_equal(_hi_after(cfg, cutoff), hi_kept_before)  # kept rows untouched
    checks = [c for c in validate(cfg)[0] if not c.ok]
    assert not checks, checks


def test_collection_continues_normally_after_prune(db, tmp_path):
    cfg, start = db
    pipeline.prune_history(cfg, keep_days=1, event_margin_days=1, backup_to=tmp_path / "b.sqlite")
    latest = pipeline.last_calculated_ms(pipeline.get_engine(cfg))
    res = pipeline.update(cfg, until_ms=latest + DAY_MS)
    assert res["intervals"] == 96 and res["variable_rows"] == 3 * 96
    failed = [c for c in validate(cfg)[0] if not c.ok]
    assert not failed, failed


def test_cli_prune_history(db, tmp_path):
    from ar_gen import cli
    cfg, _ = db
    backup = tmp_path / "cli_backup.sqlite"
    assert cli.main(["prune-history", "--keep-days", "2", "--backup", str(backup)]) == 0
    assert backup.exists()
    assert cli.main(["prune-history", "--keep-days", "0"]) == 2          # refuses to delete everything
