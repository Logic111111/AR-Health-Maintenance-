"""The deep-learning notebook must reuse the LightGBM notebook's data/split/alarm cells and export a complete bundle."""
import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
SHARED = ("FIND_DATA_CODE", "LOAD_DATA_CODE", "FAILURES_CODE", "FEATURES_CODE", "SPLIT_CODE", "CALIBRATE_CODE",
          "THRESHOLD_CODE", "DAILY_CODE")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "notebooks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(tmp_path):
    out = tmp_path / "dl.ipynb"
    _load("build_dl_notebook").build(out)
    return nbformat.read(out, as_version=4)


def _code(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_notebook_is_valid_and_asks_for_a_gpu(tmp_path):
    nb = _build(tmp_path)
    nbformat.validate(nb)
    assert len(_code(nb)) >= 20
    assert nb.metadata["accelerator"] == "GPU"


def test_notebook_shares_data_split_and_alarm_cells_with_the_lightgbm_notebook(tmp_path):
    lgbm = _load("build_colab_notebook")
    dl_cells, lgbm_cells = _code(_build(tmp_path)), _code(nbformat.read(ROOT / "notebooks" / "AR_ML_Colab.ipynb", as_version=4))
    for name in SHARED:
        source = getattr(lgbm, name).strip()
        assert source in dl_cells and source in lgbm_cells, name
    embedded = [c for c in dl_cells if c.startswith("%%writefile ar_ml_features.py")]
    assert len(embedded) == 1
    assert embedded[0].split("\n", 1)[1].strip() == (ROOT / "ar_gen" / "ml_features.py").read_text(encoding="utf-8").strip()


def test_notebook_embeds_the_network_code_the_dashboard_uses(tmp_path):
    cells = [c for c in _code(_build(tmp_path)) if c.startswith("%%writefile ar_dl_model.py")]
    assert len(cells) == 1
    assert cells[0].split("\n", 1)[1].strip() == (ROOT / "ar_gen" / "dl_model.py").read_text(encoding="utf-8").strip()


def test_notebook_exports_a_self_contained_bundle(tmp_path):
    code = "\n".join(_code(_build(tmp_path)))
    for name in ("ar_dl_model.zip", "model_dl.pt", "dl_meta.json"):
        assert name in code, name
    from ar_gen.dl_scoring import REQUIRED_META
    for key in REQUIRED_META + ("component_classes", "net_config", "model_version", "data_until_utc", "metrics"):
        assert f'"{key}"' in code, key


def test_committed_notebook_is_up_to_date(tmp_path):
    fresh = _build(tmp_path)
    committed = nbformat.read(ROOT / "notebooks" / "AR_DL_Colab.ipynb", as_version=4)
    assert [c.source for c in committed.cells] == [c.source for c in fresh.cells]
