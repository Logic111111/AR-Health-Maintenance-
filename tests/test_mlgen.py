"""generate-ml-data (no database) must give the same dataset as the database route + export-dataset."""
import pandas as pd
import pytest

from ar_gen import pipeline
from ar_gen.dataset import export_dataset
from ar_gen.mlgen import generate_ml_dataset
from ar_gen.sim.timeutil import DAY_MS, local_to_ms, parse_date
from tests.sim_helpers import BUSY, make_cfg

START, DAYS = "2025-03-01", 12


def _read(path):
    df = pd.read_csv(path, parse_dates=["ts"], low_memory=False)
    return df.sort_values(["asset_id", "ts"]).reset_index(drop=True)


def test_direct_generation_equals_database_route(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    for env, sub in (("AR_STATE_DIR", "state"), ("AR_OUTPUT_DIR", "output"), ("AR_LOG_DIR", "logs")):
        monkeypatch.setenv(env, str(tmp_path / sub))
    cfg = make_cfg(3, BUSY)
    start_ms = local_to_ms(parse_date(START), 0.0, cfg["timezone"])

    pipeline.init_db(cfg)
    pipeline.seed(cfg, START)
    pipeline.update(cfg, until_ms=start_ms + DAYS * DAY_MS)
    db_files = export_dataset(cfg, tmp_path / "via_db")

    res = generate_ml_dataset(cfg, start_ms, DAYS, tmp_path / "direct", chunk_days=3)

    via_db, direct = _read(db_files["dataset"]), _read(res["dataset"])
    assert list(direct.columns) == list(via_db.columns)
    assert len(direct) == 3 * DAYS * 96
    pd.testing.assert_frame_equal(direct, via_db, check_dtype=False)
    assert res["failures"] >= 0 and res["reclosers"] == 3

    hourly = _read(res["hourly_dataset"])
    pd.testing.assert_frame_equal(hourly, direct[direct["ts"].dt.minute == 0].reset_index(drop=True),
                                  check_dtype=False)
    gt_db = pd.read_csv(db_files["ground_truth"]).reset_index(drop=True)
    gt_direct = pd.read_csv(res["ground_truth"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(gt_direct, gt_db, check_dtype=False)
