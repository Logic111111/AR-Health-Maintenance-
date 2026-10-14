"""The ML switch (config.yaml ml.enabled, off as shipped): with it off nothing ML runs or shows, models or not."""
import importlib.util

import pandas as pd
import sqlalchemy as sa

from ar_gen import cli, pipeline
from ar_gen.config import load_config, ml_enabled
from ar_gen.dashboard_data import branches, fleet_snapshot
from ar_gen.db import schema
from ar_gen.ui.map3d import build_deck
from tests.ml_helpers import write_tiny_bundle, write_tiny_dl_bundle
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _install_models(cfg):
    write_tiny_bundle(cfg["paths"]["models_dir"])
    if HAS_TORCH:
        write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])


def _prediction_rows(cfg):
    with pipeline.get_engine(cfg).connect() as conn:
        count = lambda t: conn.execute(sa.select(sa.func.count()).select_from(t)).scalar()  # noqa: E731
        daily = pd.read_sql(sa.select(schema.ar_daily_summary), conn)
        return count(schema.ar_ml_prediction_15min), count(schema.ar_dl_prediction_15min), daily


def test_ml_is_off_as_shipped_and_the_switch_does_not_touch_the_simulation(monkeypatch):
    monkeypatch.delenv("AR_ML_ENABLED", raising=False)
    off = load_config(load_dotenv_file=False)
    assert off["ml"]["enabled"] is False and not ml_enabled(off)
    monkeypatch.setenv("AR_ML_ENABLED", "1")
    on = load_config(load_dotenv_file=False)
    assert ml_enabled(on) and on["config_hash"] == off["config_hash"]     # the saved simulator state stays valid


def test_with_ml_off_the_pipeline_scores_nothing_even_with_models_installed(db):
    cfg, start = db
    _install_models(cfg)
    eng = pipeline.get_engine(cfg)
    assert pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng)) > 0
    pipeline.update(cfg, until_ms=pipeline.last_calculated_ms(eng) + 4 * 15 * 60_000)
    n_ml, n_dl, daily = _prediction_rows(cfg)
    assert n_ml == 0 and n_dl == 0
    assert len(daily) > 0 and daily["avg_hi"].notna().all() and daily["max_p_fail"].isna().all()


def test_ml_commands_say_ml_is_switched_off(db, capsys):
    cfg, _ = db
    _install_models(cfg)
    assert cli.main(["ml-status"]) == 0 and cli.main(["dl-status"]) == 0
    assert cli.main(["ml-score"]) == 2 and cli.main(["dl-score"]) == 2
    assert capsys.readouterr().out.count("ML is switched off") == 4
    assert _prediction_rows(cfg)[:2] == (0, 0)


def test_map_tooltip_has_no_ml_when_switched_off(db):
    cfg, _ = db
    _, snap = fleet_snapshot(pipeline.get_engine(cfg), cfg)
    assert "ML risk" in build_deck(snap, branches(cfg))._tooltip["html"]
    off = build_deck(snap, branches(cfg), show_ml=False)._tooltip["html"]
    assert "ML" not in off and "Likely part" not in off and "Health Index" in off and "Worst" in off
