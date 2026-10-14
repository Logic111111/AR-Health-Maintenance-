"""Deep-learning model: shared network code, bundle loading, window rule, pipeline / CLI / dashboard data."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

torch = pytest.importorskip("torch")

from ar_gen import cli, pipeline  # noqa: E402
from ar_gen.dashboard_data import dl_model_status, dl_snapshot, fleet_snapshot, with_dl  # noqa: E402
from ar_gen.db import schema  # noqa: E402
from ar_gen.dl_model import ARCHS, scale_inputs, window_rows  # noqa: E402
from ar_gen.dl_scoring import DLModel, load_dl_model, read_dl_meta  # noqa: E402
from ar_gen.ml_features import build_features  # noqa: E402
from ar_gen.sim.timeutil import DAY_MS  # noqa: E402
from tests.ml_helpers import COMPONENTS, synthetic_variables, write_tiny_dl_bundle  # noqa: E402
from tests.test_maintenance_done import db  # noqa: E402, F401  (shared pipeline-database fixture)

pytestmark = pytest.mark.usefixtures("ml_on")       # these tests use the ML models


def _two_assets(step_minutes, days=4):
    return pd.concat([synthetic_variables(days=days, step_minutes=step_minutes, asset=a) for a in ("AR-001", "AR-002")],
                     ignore_index=True)


def test_window_rows_are_oldest_first_and_repeat_the_first_row():
    np.testing.assert_array_equal(window_rows([0, 5, 9], [0, 0, 6], steps=4, stride=2),
                                  [[0, 0, 0, 0], [0, 1, 3, 5], [6, 6, 7, 9]])


def test_scale_inputs_logs_standardises_fills_and_clips():
    raw = np.array([[np.e - 1, 5.0, np.nan], [-3.0, 1e9, 2.0]])
    out = scale_inputs(raw, [0], mean=[0.0, 1.0, 0.0], std=[1.0, 2.0, 1.0], clip=8.0)
    np.testing.assert_allclose(out, [[1.0, 2.0, 0.0], [0.0, 8.0, 2.0]], rtol=1e-6)
    assert out.dtype == np.float32 and np.isnan(raw[0, 2])            # input left untouched


@pytest.mark.parametrize("step_minutes", [60, 15])
def test_scorer_windows_follow_the_notebook_rule_on_regular_rows(step_minutes):
    frame = _two_assets(step_minutes)
    model = DLModel(net=None, meta={"window_steps": 6, "step_hours": 2})
    first = frame.groupby("asset_id").cumcount().rsub(np.arange(len(frame))).to_numpy()
    targets = np.flatnonzero(frame["ts"].dt.minute.eq(0).to_numpy())[::7]
    expected = window_rows(targets, first[targets], 6, 2 * 60 // step_minutes)
    np.testing.assert_array_equal(model.window_rows(frame, targets), expected)


def test_scorer_windows_take_the_latest_row_at_or_before_each_step_when_rows_are_missing():
    frame = synthetic_variables(days=2, step_minutes=60).drop(index=[20, 21]).reset_index(drop=True)
    model = DLModel(net=None, meta={"window_steps": 3, "step_hours": 1})
    end = int(np.flatnonzero(frame["ts"] == frame["ts"].iloc[0] + pd.Timedelta(hours=22))[0])
    rows = model.window_rows(frame, np.array([end]))[0]
    hours = ((frame["ts"].iloc[rows] - frame["ts"].iloc[0]) / pd.Timedelta(hours=1)).tolist()
    assert hours == [19, 19, 22]                                      # 20 h and 21 h are missing -> 19 h


def test_read_dl_meta_reports_problems(tmp_path, monkeypatch):
    assert read_dl_meta(tmp_path / "none") is None
    write_tiny_dl_bundle(tmp_path / "ok")
    assert read_dl_meta(tmp_path / "ok")["model_version"] == "dl-test-1"
    write_tiny_dl_bundle(tmp_path / "fv", version=999)
    with pytest.raises(ValueError, match="feature version"):
        read_dl_meta(tmp_path / "fv")
    write_tiny_dl_bundle(tmp_path / "nv", net_version=999)
    with pytest.raises(ValueError, match="network version"):
        read_dl_meta(tmp_path / "nv")
    (tmp_path / "ok" / "model_dl.pt").unlink()
    with pytest.raises(ValueError, match="model_dl.pt"):
        read_dl_meta(tmp_path / "ok")
    write_tiny_dl_bundle(tmp_path / "nt")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="PyTorch"):
        read_dl_meta(tmp_path / "nt")


@pytest.mark.parametrize("arch", ARCHS)
def test_every_network_type_loads_and_gives_calibrated_probabilities(tmp_path, arch):
    write_tiny_dl_bundle(tmp_path, arch=arch)
    model = load_dl_model(tmp_path)
    df = synthetic_variables(days=9)
    X = build_features(df, 15)
    targets = (df["ts"] >= df["ts"].iloc[-1] - pd.Timedelta(days=1)).to_numpy()
    pred = model.predict(df, X, targets)
    assert len(pred) == targets.sum() and list(pred.index) == list(df.index[targets])
    assert pred["p_fail_30d"].between(0, 1).all()
    assert (pred["dl_healthy"] == (pred["p_fail_30d"] < model.threshold)).all()
    assert set(pred["dl_risk_class"]) <= {"LOW", "ELEVATED", "HIGH"}
    assert set(pred["top_component"]) <= set(COMPONENTS) and pred["top_component_p"].between(0, 1).all()


def test_scores_equal_a_direct_forward_pass_of_the_saved_network(tmp_path):
    meta = write_tiny_dl_bundle(tmp_path, arch="tcn")
    model = load_dl_model(tmp_path)
    df = synthetic_variables(days=9, step_minutes=60)
    X = build_features(df, 60)
    end = len(df) - 1
    pred = model.predict(df, X, np.arange(len(df)) == end)
    s = meta["scaling"]
    x = scale_inputs(X.to_numpy(np.float32)[window_rows([end], [0], meta["window_steps"], meta["step_hours"])[0]],
                     [meta["features"].index(c) for c in s["log1p"]], s["mean"], s["std"], s["clip"])
    saved = torch.load(tmp_path / "model_dl.pt", weights_only=True)
    from ar_gen.dl_model import ARNet
    net = ARNet(**saved["config"]).eval()
    net.load_state_dict(saved["state_dict"])
    with torch.no_grad():
        raw = float(torch.sigmoid(net(torch.from_numpy(x)[None])[0])[0])
    assert pred["p_fail_30d"].iloc[0] == pytest.approx(np.interp(raw, [0, 0.5, 1], [0, 0.4, 1]), abs=1e-6)


def _dl_rows(cfg):
    eng = pipeline.get_engine(cfg)
    with eng.connect() as conn:
        n_var = conn.execute(sa.select(sa.func.count()).select_from(schema.ar_variables_15min)).scalar()
        dl = pd.read_sql(sa.select(schema.ar_dl_prediction_15min), conn)
        ml = pd.read_sql(sa.select(schema.ar_ml_prediction_15min), conn)
    return n_var, dl, ml


def test_pipeline_scores_every_interval_with_a_dl_model_and_no_lightgbm_model(db):
    cfg, start = db
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    n_var, dl, ml = _dl_rows(cfg)
    assert len(dl) == n_var and len(ml) == 0
    assert dl["p_fail_30d"].between(0, 1).all() and (dl["model_version"] == "dl-test-1").all()


def test_an_unusable_dl_model_does_not_stop_the_pipeline(db):
    cfg, start = db
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"], net_version=999)
    eng = pipeline.get_engine(cfg)
    assert pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng)) > 0
    assert len(_dl_rows(cfg)[1]) == 0


def test_a_failed_dl_write_does_not_stop_hi_and_is_caught_up_next_run(db, monkeypatch):
    from ar_gen import dl_scoring
    cfg, start = db
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])
    eng = pipeline.get_engine(cfg)
    last = pipeline.last_calculated_ms(eng)
    dl_scoring.score_stored(cfg, eng, start, last, load_dl_model(cfg["paths"]["dl_models_dir"]))
    real = dl_scoring.score_range

    def locked(*a, **k):
        raise sa.exc.OperationalError("INSERT", {}, Exception("database is locked"))

    monkeypatch.setattr(dl_scoring, "score_range", locked)
    res = pipeline.update(cfg, until_ms=last + DAY_MS)              # HI goes on, deep learning fails for this day
    assert res["variable_rows"] == 3 * 96
    monkeypatch.setattr(dl_scoring, "score_range", real)
    pipeline.update(cfg, until_ms=last + DAY_MS + 4 * 15 * 60_000)   # the next run catches the missed day up
    n_var, dl, _ = _dl_rows(cfg)
    assert len(dl) == n_var


def test_cli_dl_status_and_dl_score(db, capsys):
    cfg, _ = db
    assert cli.main(["dl-status"]) == 0
    assert "No deep-learning model" in capsys.readouterr().out
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])
    assert cli.main(["dl-status"]) == 0
    out = capsys.readouterr().out
    assert "dl-test-1" in out and "threshold" in out and "12 steps x 2 h" in out
    assert cli.main(["dl-score"]) == 0
    n_var, dl, _ = _dl_rows(cfg)
    assert len(dl) == n_var
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"], version=999)
    assert cli.main(["dl-status"]) == 2


def test_dashboard_data_for_the_dl_tab(db):
    cfg, start = db
    assert dl_model_status(cfg) == {"installed": False, "folder": str(cfg["paths"]["dl_models_dir"])}
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])
    status = dl_model_status(cfg)
    assert status["installed"] and status["version"] == "dl-test-1" and status["window"] == "12 steps x 2 h"
    eng = pipeline.get_engine(cfg)
    last = pipeline.last_calculated_ms(eng)
    from ar_gen.dl_scoring import score_stored
    score_stored(cfg, eng, start, last - DAY_MS, load_dl_model(cfg["paths"]["dl_models_dir"]))
    _, snap = fleet_snapshot(eng, cfg)
    view = with_dl(snap, dl_snapshot(eng))
    assert view["dl_stale"].all() and view["dl_p"].isna().all()       # a day behind the HI: not shown as current
    score_stored(cfg, eng, last - DAY_MS, last, load_dl_model(cfg["paths"]["dl_models_dir"]))
    view = with_dl(snap, dl_snapshot(eng))
    assert not view["dl_stale"].any() and view["dl_p"].between(0, 1).all()
    meta_path = cfg["paths"]["dl_models_dir"] / "dl_meta.json"
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "net_version": 999}))
    assert "network version" in dl_model_status(cfg)["error"]
