"""Test-wide isolation: tests never read the real models/ folder."""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_models_dir(tmp_path_factory):
    old = os.environ.get("AR_MODELS_DIR")
    os.environ["AR_MODELS_DIR"] = str(tmp_path_factory.mktemp("no_models"))
    yield
    if old is None:
        os.environ.pop("AR_MODELS_DIR", None)
    else:
        os.environ["AR_MODELS_DIR"] = old
