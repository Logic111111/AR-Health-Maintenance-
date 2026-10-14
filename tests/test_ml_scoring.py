"""Live ML scoring: shared feature recipe, model bundle loading, pipeline predictions."""
import json

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen import cli, pipeline
from ar_gen.db import schema
from ar_gen.db.schema import V_COLUMNS
from ar_gen.ml_features import CATEGORIES, TREND_VARS, V_FEATURES, build_features, feature_names
from ar_gen.ml_scoring import load_model
from ar_gen.sim.timeutil import DAY_MS
from tests.ml_helpers import synthetic_variables, write_tiny_bundle
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)

pytestmark = pytest.mark.usefixtures("ml_on")       # these tests use the ML models


def test_feature_recipe_matches_the_database_columns():
    assert V_FEATURES == [c for c in V_COLUMNS if c != "v42_weather"]
    names = feature_names()
    assert names[0] == "telemetry_present" and len(names) == len(set(names))
    assert all(f"{v}_chg7d" in names and f"{v}_mean7d" in names for v in TREND_VARS)
    assert all(f"{col}__{cat}" in names for col, cats in CATEGORIES.items() for cat in cats)


def test_features_have_fixed_columns_and_no_trend_in_the_first_week():
    df = synthetic_variables(days=3)
    X = build_features(df, step_minutes=15)
    assert list(X.columns) == feature_names() and len(X) == len(df)
    assert X[f"{TREND_VARS[0]}_chg7d"].isna().all()
    assert X["v42_weather__SUNNY"].isin([0.0, 1.0]).all() and X["model__OSM27-310"].eq(0).all()


def test_hourly_and_15min_features_agree_at_the_hour():
    quarter = synthetic_variables(days=10, step_minutes=15)
    hourly = quarter[quarter["ts"].dt.minute == 0].reset_index(drop=True)
    Xq = build_features(quarter, 15).set_index(quarter["ts"])
    Xh = build_features(hourly, 60).set_index(hourly["ts"])
    common = Xh.index[Xh[f"{TREND_VARS[0]}_chg7d"].notna()]
    assert len(common) > 24
    for v in TREND_VARS[:3]:
        np.testing.assert_allclose(Xq.loc[common, f"{v}_chg7d"], Xh.loc[common, f"{v}_chg7d"], rtol=1e-9)
        np.testing.assert_allclose(Xq.loc[common, f"{v}_mean7d"], Xh.loc[common, f"{v}_mean7d"], rtol=0.01)


def test_load_model_returns_none_without_a_bundle(tmp_path):
    assert load_model(tmp_path) is None


def test_load_model_rejects_another_feature_version(tmp_path):
    write_tiny_bundle(tmp_path, version=999)
    with pytest.raises(ValueError, match="feature"):
        load_model(tmp_path)


def test_predictions_are_calibrated_probabilities(tmp_path):
    write_tiny_bundle(tmp_path)
    model = load_model(tmp_path)
    X = build_features(synthetic_variables(days=9), 15)
    pred = model.predict(X)
    assert len(pred) == len(X) and pred["p_fail_30d"].between(0, 1).all()
    assert (pred["ml_healthy"] == (pred["p_fail_30d"] < model.threshold)).all()
    assert set(pred["top_component"].dropna()) <= {"BATTERY", "COMMS", "CONTROLLER", "MECHANISM"}
    assert pred["top_component_p"].between(0, 1).all()


def _counts(cfg):
    eng = pipeline.get_engine(cfg)
    with eng.connect() as conn:
        n_var = conn.execute(sa.select(sa.func.count()).select_from(schema.ar_variables_15min)).scalar()
        preds = pd.read_sql(sa.select(schema.ar_ml_prediction_15min), conn)
        daily = pd.read_sql(sa.select(schema.ar_daily_summary), conn)
    return n_var, preds, daily


def test_pipeline_scores_every_interval_when_a_model_is_installed(db):
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    n_var, preds, daily = _counts(cfg)
    assert len(preds) == n_var and preds["p_fail_30d"].between(0, 1).all()
    assert set(preds["ml_risk_class"]) <= {"LOW", "ELEVATED", "HIGH"} and (preds["model_version"] == "test-1").all()
    assert daily["max_p_fail"].notna().all() and daily["agree"].notna().all()


def test_pipeline_without_a_model_still_fills_the_daily_hi_summary(db):
    cfg, _ = db
    n_var, preds, daily = _counts(cfg)
    assert len(preds) == 0 and len(daily) > 0
    assert daily["avg_hi"].notna().all() and daily["ml_healthy"].isna().all()


def test_incompatible_model_does_not_stop_the_pipeline(db):
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"], version=999)
    eng = pipeline.get_engine(cfg)
    assert pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng)) > 0
    assert len(_counts(cfg)[1]) == 0


def test_cli_ml_status_and_ml_score(db, capsys):
    cfg, _ = db
    assert cli.main(["ml-status"]) == 0
    assert "No ML model" in capsys.readouterr().out
    write_tiny_bundle(cfg["paths"]["models_dir"])
    assert cli.main(["ml-status"]) == 0
    out = capsys.readouterr().out
    assert "test-1" in out and "threshold" in out
    assert cli.main(["ml-score"]) == 0
    assert len(_counts(cfg)[1]) > 0


def test_a_failed_ml_write_does_not_stop_hi_and_is_caught_up_next_run(db, monkeypatch):
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    eng = pipeline.get_engine(cfg)
    last = pipeline.last_calculated_ms(eng)
    pipeline.score_and_summarise(cfg, eng, start, last, load_model(cfg["paths"]["models_dir"]))
    real = pipeline.ml_score_range

    def locked(*a, **k):
        raise sa.exc.OperationalError("INSERT", {}, Exception("database is locked"))

    monkeypatch.setattr(pipeline, "ml_score_range", locked)
    res = pipeline.update(cfg, until_ms=last + DAY_MS)        # HI goes on, ML fails for this day
    assert res["variable_rows"] == 3 * 96
    monkeypatch.setattr(pipeline, "ml_score_range", real)
    pipeline.update(cfg, until_ms=last + DAY_MS + 4 * 15 * 60_000)   # next run catches the missed day up
    n_var, preds, daily = _counts(cfg)
    assert len(preds) == n_var
    assert daily["max_p_fail"].notna().all()


def test_read_meta_reports_problems_without_loading_lightgbm(tmp_path, monkeypatch):
    from ar_gen.ml_scoring import read_meta
    assert read_meta(tmp_path / "none") is None
    write_tiny_bundle(tmp_path / "m")
    assert read_meta(tmp_path / "m")["model_version"] == "test-1"
    meta = json.loads((tmp_path / "m" / "ml_meta.json").read_text())
    (tmp_path / "m" / "ml_meta.json").write_text(json.dumps({**meta, "features": meta["features"] + ["v99_new"]}))
    with pytest.raises(ValueError, match="v99_new"):
        read_meta(tmp_path / "m")
    (tmp_path / "m" / "ml_meta.json").write_text(json.dumps(meta))
    monkeypatch.setitem(__import__("sys").modules, "lightgbm", None)
    with pytest.raises(RuntimeError, match="LightGBM"):
        read_meta(tmp_path / "m")
