"""'Maintenance done': save a reset, recalculate from it, then normal calculation continues."""
import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen import cli, pipeline
from ar_gen.db import schema
from ar_gen.sim.timeutil import DAY_MS, MIN_MS, local_to_ms, parse_date
from ar_gen.variables import to_ms
from tests.sim_helpers import BUSY, make_cfg

START = "2025-03-01"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    for env, sub in (("AR_STATE_DIR", "state"), ("AR_OUTPUT_DIR", "output"), ("AR_LOG_DIR", "logs"),
                     ("AR_MODELS_DIR", "models")):
        monkeypatch.setenv(env, str(tmp_path / sub))
    cfg = make_cfg(3, BUSY)
    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    start = local_to_ms(parse_date(START), 0.0, cfg["timezone"])
    pipeline.update(cfg, until_ms=start + 4 * DAY_MS)
    return cfg, start


def _df(cfg, stmt):
    eng = pipeline.get_engine(cfg)
    with eng.connect() as conn:
        out = pd.read_sql(stmt, conn)
    eng.dispose()
    return out


def _open_ops_between(cfg, asset, lo_ms, hi_ms):
    o = schema.co_operations
    ops = _df(cfg, sa.select(o.c.ts).where(o.c.asset_id == asset, o.c.op_type == "OPEN"))
    ms = to_ms(ops["ts"])
    return int(((ms > lo_ms) & (ms <= hi_ms)).sum())


def _v22(cfg, asset, ts_ms):
    v = schema.ar_variables_15min
    from ar_gen.sim.timeutil import ms_to_dt
    return int(_df(cfg, sa.select(v.c.v22_open_ops_30d).where(v.c.asset_id == asset, v.c.ts == ms_to_dt(ts_ms))).iloc[0, 0])


def test_reset_raises_hi_and_then_normal_calculation_continues(db):
    cfg, start = db
    v = schema.ar_variables_15min
    best = _df(cfg, sa.select(v.c.asset_id, v.c.ts, v.c.v22_open_ops_30d).order_by(v.c.v22_open_ops_30d.desc())
               .limit(1))
    asset, r_ms = best.loc[0, "asset_id"], int(to_ms(best["ts"])[0])
    assert best.loc[0, "v22_open_ops_30d"] > 0

    res = pipeline.maintenance_done(cfg, asset, ["SWITCHING"], at_ms=r_ms, note="OSM serviced")
    assert res["hi_after"] >= res["hi_before"] and res["recomputed_rows"] > 0
    nxt = r_ms + 15 * MIN_MS
    assert _v22(cfg, asset, nxt) == _open_ops_between(cfg, asset, r_ms, nxt)

    pipeline.update(cfg, until_ms=start + 5 * DAY_MS)          # normal calculation continues
    last = start + 5 * DAY_MS
    assert _v22(cfg, asset, last) == _open_ops_between(cfg, asset, max(r_ms, last - 30 * DAY_MS), last)


def test_reset_rejects_bad_input(db):
    cfg, start = db
    with pytest.raises(ValueError, match="unknown asset"):
        pipeline.maintenance_done(cfg, "AR-999", ["ALL"])
    with pytest.raises(ValueError, match="part"):
        pipeline.maintenance_done(cfg, "AR-001", ["FOO"])
    with pytest.raises(ValueError, match="future"):
        pipeline.maintenance_done(cfg, "AR-001", ["ALL"], at_ms=pipeline.now_ms(cfg) + DAY_MS)
    assert _df(cfg, sa.select(sa.func.count()).select_from(schema.hi_reset_log)).iloc[0, 0] == 0


def test_reset_just_after_the_latest_data_is_kept_for_the_next_interval(db):
    cfg, start = db
    last = pipeline.last_calculated_ms(pipeline.get_engine(cfg))
    at = last + 10 * MIN_MS                                    # technician finished 10 min after the last poll
    res = pipeline.maintenance_done(cfg, "AR-001", ["SWITCHING"], at_ms=at)
    assert res["reset_id"] is not None
    pipeline.update(cfg, until_ms=last + 30 * MIN_MS)
    t = last + 30 * MIN_MS
    assert _v22(cfg, "AR-001", t) == _open_ops_between(cfg, "AR-001", at, t)


def test_reset_waits_for_collector_and_covers_its_new_interval(db):
    import threading

    from ar_gen.sources.simulated import SimulatedNojaSource
    cfg, _ = db
    eng = pipeline.get_engine(cfg)
    last0 = pipeline.last_calculated_ms(eng)
    holding = threading.Event()

    def collector():
        with pipeline.run_lock(cfg):
            holding.set()
            src = SimulatedNojaSource(cfg)
            src.load()
            pipeline.collect(cfg, eng, src, last0 + 15 * MIN_MS)
            pipeline.compute_pending(cfg, eng, src.last_collected_ms())

    th = threading.Thread(target=collector)
    th.start()
    assert holding.wait(10)
    pipeline.maintenance_done(cfg, "AR-001", ["SWITCHING"], at_ms=last0, wait_s=60)
    th.join()
    new = last0 + 15 * MIN_MS
    assert _v22(cfg, "AR-001", new) == _open_ops_between(cfg, "AR-001", last0, new)


def test_reset_is_removed_when_recalculation_fails(db, monkeypatch):
    cfg, _ = db

    def broken(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr(pipeline, "compute_range", broken)
    with pytest.raises(RuntimeError, match="not saved"):
        pipeline.maintenance_done(cfg, "AR-001", ["ALL"])
    assert _df(cfg, sa.select(sa.func.count()).select_from(schema.hi_reset_log)).iloc[0, 0] == 0


def test_lower_case_asset_id_is_stored_canonical(db):
    cfg, _ = db
    pipeline.maintenance_done(cfg, " ar-001 ", ["all"])
    rows = _df(cfg, sa.select(schema.hi_reset_log))
    assert list(rows["asset_id"]) == ["AR-001"] and list(rows["part_groups"]) == ["ALL"]


def test_compute_variables_waits_for_the_lock(db, monkeypatch):
    cfg, _ = db
    monkeypatch.setattr(pipeline, "LOCK_WAIT_S", 0)
    with pipeline.run_lock(cfg):
        assert cli.main(["compute-variables", "--start", "2025-03-03"]) == 2
        assert cli.main(["compute-hi", "--start", "2025-03-03"]) == 2


def test_reset_waits_for_lock(db):
    cfg, _ = db
    with pipeline.run_lock(cfg):
        with pytest.raises(RuntimeError, match="collector busy"):
            pipeline.maintenance_done(cfg, "AR-001", ["ALL"], wait_s=0)
    assert _df(cfg, sa.select(sa.func.count()).select_from(schema.hi_reset_log)).iloc[0, 0] == 0


def test_cli_maintenance_done(db):
    cfg, _ = db
    assert cli.main(["maintenance-done", "--asset", "AR-001", "--part", "ALL", "--note", "test"]) == 0
    rows = _df(cfg, sa.select(schema.hi_reset_log))
    assert len(rows) == 1 and rows.loc[0, "part_groups"] == "ALL" and rows.loc[0, "note"] == "test"
