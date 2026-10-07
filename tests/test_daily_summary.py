"""Daily summary: per recloser per local day HI statistics (and ML, when a model exists)."""
import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen import pipeline
from ar_gen.db import schema
from ar_gen.sim.timeutil import HOUR_MS
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)


def _frames(cfg):
    eng = pipeline.get_engine(cfg)
    with eng.connect() as conn:
        hi = pd.read_sql(sa.select(schema.ar_health_index_15min.c.asset_id, schema.ar_health_index_15min.c.ts,
                                   schema.ar_health_index_15min.c.health_index), conn)
        daily = pd.read_sql(sa.select(schema.ar_daily_summary), conn)
    hi["day"] = pd.to_datetime(hi["ts"]).dt.tz_localize("UTC").dt.tz_convert(cfg["timezone"]).dt.date
    return hi, daily


def _expected(hi, healthy_min):
    g = hi.groupby(["asset_id", "day"])["health_index"]
    exp = g.agg(intervals="size", avg_hi="mean", min_hi="min", max_hi="max").reset_index()
    unhealthy = hi.assign(u=hi["health_index"] < healthy_min).groupby(["asset_id", "day"])["u"].sum() * 0.25
    exp["unhealthy_hours"] = unhealthy.to_numpy()
    exp["hi_healthy"] = exp["avg_hi"] >= healthy_min
    return exp.sort_values(["asset_id", "day"]).reset_index(drop=True)


def _check(cfg):
    hi, daily = _frames(cfg)
    daily = daily.assign(day=pd.to_datetime(daily["day"]).dt.date).sort_values(["asset_id", "day"])
    exp = _expected(hi, cfg["health_index"]["healthy_min"])
    got = daily[["asset_id", "day", "intervals", "avg_hi", "min_hi", "max_hi", "unhealthy_hours", "hi_healthy"]]
    pd.testing.assert_frame_equal(got.reset_index(drop=True), exp, check_dtype=False, atol=1e-6)


def test_daily_summary_matches_the_15_minute_rows(db):
    cfg, _ = db
    _check(cfg)


def test_a_day_split_over_two_calculation_runs_is_summarised_completely(db):
    cfg, start = db
    eng = pipeline.get_engine(cfg)
    last = pipeline.last_calculated_ms(eng)
    mid = start + 2 * 24 * HOUR_MS + 6 * HOUR_MS + 30 * 60_000          # local midday, mid-interval boundary
    mid -= mid % (15 * 60_000)
    with eng.begin() as conn:
        conn.execute(schema.ar_daily_summary.delete())
    pipeline.compute_range(cfg, eng, start, mid)
    pipeline.compute_range(cfg, eng, mid, last)
    _check(cfg)


def test_compute_hi_refreshes_the_daily_summary(db):
    cfg, start = db
    eng = pipeline.get_engine(cfg)
    cfg["health_index"]["weights"]["v11"] *= 5
    cfg["health_index"]["healthy_min"] = 95
    pipeline.recompute_hi(cfg, eng, start, pipeline.last_calculated_ms(eng))
    _check(cfg)


def test_a_day_cut_by_prune_keeps_its_complete_summary(db, tmp_path):
    cfg, start = db
    eng = pipeline.get_engine(cfg)
    _, before = _frames(cfg)
    res = pipeline.prune_history(cfg, keep_days=2, event_margin_days=1, backup_to=tmp_path / "b.sqlite")
    cut_ms = int(pd.Timestamp(res["cutoff"]).value // 1_000_000)
    pipeline.compute_range(cfg, eng, cut_ms, pipeline.last_calculated_ms(eng))   # recompute from the cut-off
    _, after = _frames(cfg)
    first = after["day"].min()
    keep = before[before["day"] == first].set_index("asset_id")["intervals"]
    assert (after[after["day"] == first].set_index("asset_id")["intervals"] == keep).all()
