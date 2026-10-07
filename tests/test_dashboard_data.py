import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen import pipeline
from ar_gen.dashboard_data import (STATUS_COLORS, STATUS_ORDER, feeder_paths, fleet_snapshot, hi_history,
                                   recent_events, status_of)
from ar_gen.db import schema
from tests.sim_helpers import BUSY, make_cfg
from tests.test_maintenance_done import START, db  # noqa: F401  (shared pipeline-database fixture)


@pytest.mark.parametrize("minutes,open_failure,switch,hi,expected", [
    (None, False, "CLOSED", 95, "OFFLINE"),
    (90, False, "CLOSED", 95, "OFFLINE"),
    (0, True, "LOCKOUT", 95, "FAULTY"),
    (0, False, "LOCKOUT", 95, "LOCKOUT"),
    (0, False, "CLOSED", 39.9, "CRITICAL"),
    (0, False, "CLOSED", 59.9, "POOR"),
    (0, False, "CLOSED", 79.9, "FAIR"),
    (0, False, "CLOSED", 80, "GOOD"),
    (0, False, "CLOSED", None, "OFFLINE"),
])
def test_status_priority(minutes, open_failure, switch, hi, expected):
    assert status_of(minutes, open_failure, switch, hi) == expected
    assert set(STATUS_COLORS) == set(STATUS_ORDER)


def test_fleet_snapshot_on_pipeline_db(db):
    cfg, _ = db
    eng = pipeline.get_engine(cfg)
    ts, snap = fleet_snapshot(eng, cfg)
    assert ts is not None and len(snap) == 3
    assert set(snap["status"]) <= set(STATUS_ORDER)
    assert snap["latitude"].between(5.9, 9.9).all() and snap["longitude"].between(79.6, 81.95).all()
    assert (snap["risk"] >= 0).all() and snap["color"].map(len).eq(3).all()

    target = snap[snap["status"] != "OFFLINE"]["asset_id"].iloc[0]
    with eng.begin() as conn:
        next_id = (conn.execute(sa.select(sa.func.max(schema.failure_log.c.failure_id))).scalar() or 0) + 1000
        conn.execute(schema.failure_log.insert().values(failure_id=next_id, ts=ts, asset_id=target,
                                                        component="COMMS", detected_by="SCADA",
                                                        description="test failure"))
    _, snap2 = fleet_snapshot(eng, cfg)
    row = snap2.set_index("asset_id").loc[target]
    assert row["status"] == "FAULTY" and "COMMS" in row["open_failures"]


def test_snapshot_empty_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'e.sqlite').as_posix()}")
    for env, sub in (("AR_STATE_DIR", "state"), ("AR_OUTPUT_DIR", "output"), ("AR_LOG_DIR", "logs")):
        monkeypatch.setenv(env, str(tmp_path / sub))
    cfg = make_cfg(3, BUSY)
    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    ts, snap = fleet_snapshot(pipeline.get_engine(cfg), cfg)
    assert ts is None
    assert snap.empty or set(snap["status"]) == {"OFFLINE"}


def test_hi_history_daily_average(db):
    cfg, _ = db
    rows, daily = hi_history(pipeline.get_engine(cfg), "AR-001", days=30, tz=cfg["timezone"])
    assert len(rows) > 96 and not daily.empty
    d = daily.iloc[1]
    expected = rows[rows["ts_local"].dt.date == d["date"]]["health_index"].mean()
    assert d["avg_hi"] == pytest.approx(expected)
    assert d["min_hi"] <= d["avg_hi"]


def test_recent_events_include_resets(db):
    cfg, _ = db
    pipeline.maintenance_done(cfg, "AR-001", ["BATTERY"], note="new battery")
    ev = recent_events(pipeline.get_engine(cfg), "AR-001", days=30, tz=cfg["timezone"])
    assert {"ts_local", "kind", "detail"} <= set(ev.columns)
    resets = ev[ev["kind"] == "RESET"]
    assert len(resets) == 1 and "BATTERY" in resets["detail"].iloc[0]
    assert ev["ts_local"].is_monotonic_decreasing


def test_feeder_paths(db):
    cfg, _ = db
    _, snap = fleet_snapshot(pipeline.get_engine(cfg), cfg)
    paths = feeder_paths(snap)
    assert set(paths["feeder_name"]) == set(snap["feeder_name"])
    assert all(len(p) >= 1 and len(p[0]) == 2 for p in paths["path"])


def test_snapshot_on_database_made_before_the_new_tables(db):
    cfg, _ = db
    eng = pipeline.get_engine(cfg)
    schema.hi_reset_log.drop(eng)
    schema.asset_location.drop(eng)
    ts, snap = fleet_snapshot(eng, cfg)
    assert ts is not None and len(snap) == 3 and snap["latitude"].notna().all()
    assert recent_events(eng, "AR-001", tz=cfg["timezone"]) is not None


def test_recent_events_with_nothing_in_the_window(db):
    cfg, _ = db
    ev = recent_events(pipeline.get_engine(cfg), "AR-001", days=0, tz=cfg["timezone"])
    assert ev.empty and list(ev.columns) == ["ts_local", "kind", "detail"]


def test_status_follows_the_configured_hi_classes():
    classes = {"GOOD": 90, "FAIR": 70, "POOR": 50, "CRITICAL": 0}
    assert status_of(0, False, "CLOSED", 85, classes) == "FAIR"
    assert status_of(0, False, "CLOSED", 65, classes) == "POOR"
    assert status_of(0, False, "CLOSED", 49, classes) == "CRITICAL"


def test_ml_values_older_than_the_health_index_are_not_shown_as_current(db):
    import shutil

    from ar_gen.dashboard_data import model_status
    from tests.ml_helpers import write_tiny_bundle
    cfg, start = db
    models = cfg["paths"]["models_dir"]
    write_tiny_bundle(models)
    eng = pipeline.get_engine(cfg)
    last = pipeline.last_calculated_ms(eng)
    pipeline.compute_range(cfg, eng, start, last)
    _, snap = fleet_snapshot(eng, cfg)
    assert snap["ml_p"].notna().all() and not snap["ml_stale"].any()
    assert model_status(cfg)["installed"]
    shutil.rmtree(models)                                   # model removed; the HI goes on without ML
    pipeline.update(cfg, until_ms=last + 4 * 15 * 60_000)
    _, snap = fleet_snapshot(eng, cfg)
    assert snap["ml_p"].isna().all() and snap["ml_stale"].all()
    assert not model_status(cfg)["installed"]


def test_model_status_explains_an_unusable_model(db):
    from ar_gen.dashboard_data import model_status
    from tests.ml_helpers import write_tiny_bundle
    cfg, _ = db
    write_tiny_bundle(cfg["paths"]["models_dir"], version=999)
    st = model_status(cfg)
    assert not st["installed"] and "feature version" in st["error"]
