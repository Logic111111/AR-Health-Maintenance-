"""Test-wide isolation: tests never read the real models/ or models/dl/ folders, and the ML switch is off
(as shipped in config.yaml) unless a test asks for ``ml_on``."""
import os

import pytest

MODEL_DIRS = {"AR_MODELS_DIR": "no_models", "AR_DL_MODELS_DIR": "no_dl_models"}


@pytest.fixture(scope="session", autouse=True)
def _isolated_models_dir(tmp_path_factory):
    old = {env: os.environ.get(env) for env in (*MODEL_DIRS, "AR_ML_ENABLED")}
    for env, name in MODEL_DIRS.items():
        os.environ[env] = str(tmp_path_factory.mktemp(name))
    os.environ["AR_ML_ENABLED"] = "0"
    yield
    for env, value in old.items():
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value


@pytest.fixture()
def ml_on(monkeypatch):
    """Turn the ML switch on (LightGBM + deep-learning scoring and the ML parts of the dashboard).

    Request it before ``db`` so the database's config is made with ML on."""
    monkeypatch.setenv("AR_ML_ENABLED", "1")
