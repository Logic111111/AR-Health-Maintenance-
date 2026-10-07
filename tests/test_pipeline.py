"""End-to-end: seed -> collect -> V1-V42 -> HI on SQLite, validation, idempotency, crash recovery."""
import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen import pipeline
from ar_gen.db import schema
from ar_gen.db.writer import write_chunk
from ar_gen.sim.timeutil import DAY_MS, local_to_ms, parse_date
from ar_gen.sources.simulated import SimulatedNojaSource
from ar_gen.validate import validate
from tests.sim_helpers import BUSY, make_cfg

START = "2025-03-01"


def cfg_in(tmp_path, monkeypatch, name):
    d = tmp_path / name
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(d / 'db.sqlite').as_posix()}")
    for env, sub in (("AR_STATE_DIR", "state"), ("AR_OUTPUT_DIR", "output"), ("AR_LOG_DIR", "logs")):
        monkeypatch.setenv(env, str(d / sub))
    return make_cfg(3, BUSY)


def start_ms(cfg):
    return local_to_ms(parse_date(START), 0.0, cfg["timezone"])


def dump(cfg) -> dict:
    eng = pipeline.get_engine(cfg)
    out = {}
    with eng.connect() as conn:
        for name, table in schema.TABLES.items():
            df = pd.read_sql(sa.select(table), conn)
            df = df.drop(columns=[c for c in ("computed_at", "updated_at") if c in df])
            out[name] = df.sort_values(list(df.columns[:2])).reset_index(drop=True)
    eng.dispose()
    return out


def assert_same(a: dict, b: dict):
    assert a.keys() == b.keys()
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name], check_dtype=False, obj=name)


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    cfg = cfg_in(tmp_path_factory.mktemp("single"), mp, "run")
    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    res = pipeline.update(cfg, until_ms=start_ms(cfg) + 3 * DAY_MS)
    data = dump(cfg)
    yield cfg, res, data
    mp.undo()


def test_full_pipeline_fills_every_table_and_validates(single):
    cfg, res, data = single
    assert res["intervals"] == 3 * 96
    n = 3 * 3 * 96
    assert len(data["ar_variables_15min"]) == n and len(data["ar_health_index_15min"]) == n
    assert len(data["telemetry_15min"]) > 0.9 * n
    hi = data["ar_health_index_15min"]
    assert hi["health_index"].between(0, 100).all()
    assert set(hi["health_class"]) <= {"GOOD", "FAIR", "POOR", "CRITICAL"}
    checks, counts = validate(cfg)
    failed = [c for c in checks if not c.ok]
    assert not failed, failed


def test_second_run_is_a_no_op(single):
    cfg, _, data = single
    res = pipeline.update(cfg, until_ms=start_ms(cfg) + 3 * DAY_MS)
    assert res["intervals"] == 0 and res["variable_rows"] == 0
    assert_same(data, dump(cfg))


def test_interval_by_interval_equals_one_backfill(single, tmp_path, monkeypatch):
    _, _, data = single
    cfg = cfg_in(tmp_path, monkeypatch, "steps")
    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    s = start_ms(cfg)
    pipeline.update(cfg, until_ms=s + 2 * DAY_MS + 7 * 15 * 60_000)
    for k in range(1, 4):                                  # a few real-time style 15-min runs
        pipeline.update(cfg, until_ms=s + 2 * DAY_MS + (7 + k) * 15 * 60_000)
    pipeline.update(cfg, until_ms=s + 3 * DAY_MS)
    assert_same(data, dump(cfg))


def test_crash_between_commit_and_state_save_is_recovered(single, tmp_path, monkeypatch):
    _, _, data = single
    cfg = cfg_in(tmp_path, monkeypatch, "crash")
    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    s = start_ms(cfg)
    pipeline.update(cfg, until_ms=s + DAY_MS)
    # Simulate a crash: the next day reaches the database but the state file is not saved.
    src = SimulatedNojaSource(cfg)
    eng = pipeline.get_engine(cfg)
    for batch in src.collect(s + 2 * DAY_MS, 96):
        with eng.begin() as conn:
            write_chunk(conn, batch.rows, batch.asset_updates)
    eng.dispose()
    pipeline.update(cfg, until_ms=s + 3 * DAY_MS)           # recovery + regeneration
    assert_same(data, dump(cfg))


def test_copy_db_moves_everything_and_wrong_database_is_refused(single, tmp_path, monkeypatch):
    src_cfg, _, data = single
    cfg = cfg_in(tmp_path, monkeypatch, "copy")
    # state of the source run, but an empty target database -> refused with a clear message
    monkeypatch.setenv("AR_STATE_DIR", str(src_cfg["paths"]["state_dir"]))
    cfg = make_cfg(3, BUSY)
    pipeline.init_db(cfg)
    with pytest.raises(RuntimeError, match="does not contain the seeded fleet"):
        pipeline.update(cfg)
    pipeline.copy_db(cfg, src_cfg["database_url"])
    assert_same(data, dump(cfg))


def test_validate_detects_corrupted_data(single, tmp_path, monkeypatch):
    src_cfg, _, _ = single
    cfg = cfg_in(tmp_path, monkeypatch, "corrupt")
    monkeypatch.setenv("AR_STATE_DIR", str(src_cfg["paths"]["state_dir"]))
    cfg = make_cfg(3, BUSY)
    pipeline.init_db(cfg)
    pipeline.copy_db(cfg, src_cfg["database_url"])
    eng = pipeline.get_engine(cfg)
    t = schema.telemetry_15min
    with eng.begin() as conn:
        ts = conn.execute(sa.select(t.c.ts).where(t.c.asset_id == "AR-001").order_by(t.c.ts)
                          .offset(50).limit(1)).scalar()
        conn.execute(t.update().where(t.c.asset_id == "AR-001", t.c.ts == ts).values(co_total=t.c.co_total + 5))
        conn.execute(t.delete().where(t.c.asset_id == "AR-002", t.c.ts == ts))
    eng.dispose()
    checks = {c.name: c for c in validate(cfg)[0]}
    assert not checks["co_total matches logged OPEN operations"].ok
    assert not checks["telemetry gaps explained by controller power-down"].ok
