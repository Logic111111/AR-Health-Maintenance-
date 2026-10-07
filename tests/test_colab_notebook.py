"""The Colab notebook must embed the project's feature recipe verbatim and export a loadable bundle."""
import importlib.util
from pathlib import Path

import nbformat

from ar_gen.ml_scoring import COMPONENT_FILE, META_FILE, MODEL_FILE, REQUIRED_META

ROOT = Path(__file__).resolve().parent.parent


def _build(tmp_path):
    spec = importlib.util.spec_from_file_location("build_colab_notebook", ROOT / "notebooks" / "build_colab_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "nb.ipynb"
    mod.build(out)
    return nbformat.read(out, as_version=4)


def test_notebook_is_valid(tmp_path):
    nb = _build(tmp_path)
    nbformat.validate(nb)
    assert len([c for c in nb.cells if c.cell_type == "code"]) >= 15


def test_notebook_embeds_the_feature_recipe_verbatim(tmp_path):
    nb = _build(tmp_path)
    cells = [c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("%%writefile ar_ml_features.py")]
    assert len(cells) == 1
    embedded = cells[0].split("\n", 1)[1]
    assert embedded.strip() == (ROOT / "ar_gen" / "ml_features.py").read_text(encoding="utf-8").strip()


def test_notebook_exports_the_bundle_the_project_loads(tmp_path):
    nb = _build(tmp_path)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for name in (META_FILE, MODEL_FILE, COMPONENT_FILE, "ar_ml_model.zip"):
        assert name in code, name
    for key in REQUIRED_META + ("component_classes", "model_version", "data_until_utc", "metrics"):
        assert f'"{key}"' in code, key


def test_committed_notebook_is_up_to_date(tmp_path):
    fresh = _build(tmp_path)
    committed = nbformat.read(ROOT / "notebooks" / "AR_ML_Colab.ipynb", as_version=4)
    assert [c.source for c in committed.cells] == [c.source for c in fresh.cells]
