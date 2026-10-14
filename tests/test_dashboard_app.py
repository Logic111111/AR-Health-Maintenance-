"""Smoke tests for the futuristic Streamlit dashboard (runs the real script in-process)."""
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from streamlit.testing.v1 import AppTest

from ar_gen import pipeline
from ar_gen.db import schema
from tests.ml_helpers import write_tiny_bundle, write_tiny_dl_bundle
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)

APP = str(Path(__file__).resolve().parent.parent / "ar_gen" / "dashboard_app.py")
TABS = ["Command Centre", "Fleet", "Recloser", "Maintenance"]
ML_TABS = ["Command Centre", "Fleet", "Recloser", "ML vs HI", "Deep Learning", "Maintenance"]


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


def test_dashboard_without_ml_shows_no_ml_anywhere(db):
    import re
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])                   # installed, but ML is switched off
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    at = _app()
    assert [t.label for t in at.tabs][:4] == TABS
    text = _all_text(at)
    assert "AR HEALTH COMMAND CENTRE" in text.upper()
    assert not re.search(r"\bML\b|Deep Learning|deep-learning|test-1", text), text
    assert not [r for r in at.sidebar.radio if r.label == "Column height"]
    fleet = at.dataframe[0].value
    assert len(fleet) == 3 and not [c for c in fleet.columns if "ML" in c or c == "Likely part"]


def test_dashboard_renders_every_tab_without_a_model(ml_on, db):
    at = _app()
    assert [t.label for t in at.tabs][:6] == ML_TABS
    text = _all_text(at)
    assert "AR HEALTH COMMAND CENTRE" in text.upper()
    assert "No ML model" in text                                    # the ML tab explains how to add one
    assert "No deep-learning model" in text and "AR_DL_Colab.ipynb" in text
    fleet = at.dataframe[0].value
    assert len(fleet) == 3


def test_dashboard_with_a_model_shows_ml_risk(ml_on, db):
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    at = _app()
    text = _all_text(at)
    assert "test-1" in text                                          # model chip / ML tab
    assert "No ML model" not in text


def test_filters_that_match_nothing_show_an_empty_state(db):
    at = _app()
    at.sidebar.multiselect(key="f_status").set_value([]).run()
    assert not at.exception, [e.message for e in at.exception]
    assert "No reclosers match" in _all_text(at)


def test_reset_form_still_saves(db):
    cfg, _ = db
    at = _app()
    at.selectbox(key="asset").set_value("AR-001").run()
    at.multiselect(key="parts").set_value(["SWITCHING"])
    [b for b in at.button if b.label == "Save & recalculate"][0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("HI" in s.value for s in at.success), [e.value for e in at.error]
    with pipeline.get_engine(cfg).connect() as conn:
        rows = pd.read_sql(sa.select(schema.hi_reset_log), conn)
    assert len(rows) == 1 and rows.loc[0, "part_groups"] == "SWITCHING"


def test_dashboard_on_an_empty_database(tmp_path, monkeypatch):
    from tests.sim_helpers import BUSY, make_cfg
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'e.sqlite').as_posix()}")
    for env, sub in (("AR_STATE_DIR", "state"), ("AR_OUTPUT_DIR", "output"), ("AR_LOG_DIR", "logs"),
                     ("AR_MODELS_DIR", "models"), ("AR_DL_MODELS_DIR", "models_dl")):
        monkeypatch.setenv(env, str(tmp_path / sub))
    pipeline.init_db(make_cfg(3, BUSY))
    at = _app()
    assert "No data yet" in _all_text(at)


def test_dashboard_explains_an_unusable_model_and_hides_its_predictions(ml_on, db):
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    write_tiny_bundle(cfg["paths"]["models_dir"], version=999)     # e.g. an old model after a program update
    at = _app()
    text = _all_text(at)
    assert "MODEL ERROR" in text and "feature version" in text
    assert at.dataframe[0].value["ML risk %"].isna().all()         # no old ML values shown as current


def test_model_text_is_escaped_in_the_page(ml_on, db):
    import json
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    meta_path = Path(cfg["paths"]["models_dir"]) / "ml_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update(model_version="<b>v</b>", metrics={"<i>auc</i>": "<script>x</script>"})
    meta_path.write_text(json.dumps(meta))
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    at = _app()
    raw = "\n".join(str(m.value) for m in at.markdown)
    assert "<b>v</b>" not in raw and "<script>" not in raw and "&lt;b&gt;v&lt;/b&gt;" in raw


def test_deep_learning_tab_shows_the_network_next_to_lightgbm(ml_on, db):
    import pytest
    pytest.importorskip("torch")
    cfg, start = db
    write_tiny_bundle(cfg["paths"]["models_dir"])
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"])
    eng = pipeline.get_engine(cfg)
    pipeline.compute_range(cfg, eng, start, pipeline.last_calculated_ms(eng))
    at = _app()
    text = _all_text(at)
    assert "No deep-learning model" not in text
    assert "DL · dl-test-1" in text and "DEEP LEARNING dl-test-1" in text and "DEEP LEARNING VS LIGHTGBM" in text
    ranking = [d.value for d in at.dataframe if "DL risk %" in d.value.columns][0]
    assert len(ranking) == 3 and ranking["DL risk %"].between(0, 100).all()
    assert ranking["LightGBM risk %"].notna().all()


def test_deep_learning_tab_explains_an_unusable_model(ml_on, db):
    import pytest
    pytest.importorskip("torch")
    cfg, _ = db
    write_tiny_dl_bundle(cfg["paths"]["dl_models_dir"], net_version=999)
    at = _app()
    text = _all_text(at)
    assert "DL · MODEL ERROR" in text and "network version" in text
