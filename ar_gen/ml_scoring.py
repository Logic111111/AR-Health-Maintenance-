"""Load a model trained in Google Colab (folder models/) and score V1-V42 rows.

The bundle written by the Colab notebook:
  ml_meta.json           feature_version, features, threshold, calibration {x, y}, component_classes,
                         metrics, trained_utc, data_until_utc, model_version
  model_failure30d.txt   LightGBM model: probability of a failure within 30 days
  model_component.txt    optional LightGBM multiclass model: which part is most likely to fail
LightGBM text files and JSON are used (not pickles) so a model trained with Colab's library
versions loads on this PC.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ml_features import FEATURE_VERSION, feature_names

META_FILE, MODEL_FILE, COMPONENT_FILE = "ml_meta.json", "model_failure30d.txt", "model_component.txt"
REQUIRED_META = ("feature_version", "features", "threshold", "calibration")


def risk_class(p: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p >= threshold, "HIGH", np.where(p >= 0.5 * threshold, "ELEVATED", "LOW"))


@dataclass
class MLModel:
    booster: object
    meta: dict
    component_booster: object | None = None
    features: list = field(init=False)

    def __post_init__(self):
        self.features = list(self.meta["features"])

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def model_version(self) -> str:
        return str(self.meta.get("model_version") or self.meta.get("trained_utc") or "unknown")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        values = X[self.features].to_numpy(dtype=float)
        raw = np.asarray(self.booster.predict(values), dtype=float)
        cal = self.meta.get("calibration") or {}
        if cal.get("x") and cal.get("y"):
            p = np.interp(raw, np.asarray(cal["x"], float), np.asarray(cal["y"], float))
        else:
            p = raw
        p = np.clip(p, 0.0, 1.0)
        out = pd.DataFrame({"p_fail_30d": p, "ml_healthy": p < self.threshold,
                            "ml_risk_class": risk_class(p, self.threshold)}, index=X.index)
        classes = list(self.meta.get("component_classes") or [])
        if self.component_booster is not None and classes:
            probs = np.asarray(self.component_booster.predict(values), dtype=float).reshape(len(X), -1)
            top = probs.argmax(axis=1)
            out["top_component"] = [classes[i] if i < len(classes) else None for i in top]
            out["top_component_p"] = probs[np.arange(len(X)), top]
        else:
            out["top_component"] = None
            out["top_component_p"] = np.nan
        return out


def read_meta(models_dir) -> dict | None:
    """ml_meta.json after every check that needs no model loading; None when no model is installed.

    Raises ValueError for an incompatible bundle and RuntimeError when LightGBM is missing.
    Used by the dashboard (cheap) and by load_model.
    """
    models_dir = Path(models_dir)
    meta_path = models_dir / META_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"{meta_path} is missing {', '.join(missing)}")
    if int(meta["feature_version"]) != FEATURE_VERSION:
        raise ValueError(f"model was trained with feature version {meta['feature_version']}, this program uses "
                         f"feature version {FEATURE_VERSION} - retrain it with the current Colab notebook")
    unknown = [f for f in meta["features"] if f not in set(feature_names())]
    if unknown:
        raise ValueError(f"model uses features this program does not build ({', '.join(unknown[:5])}) - "
                         "retrain it with the current Colab notebook")
    if importlib.util.find_spec("lightgbm") is None:
        raise RuntimeError("LightGBM is not installed - run: python -m pip install lightgbm")
    if not (models_dir / MODEL_FILE).exists():
        raise ValueError(f"{models_dir / MODEL_FILE} is missing")
    return meta


def load_model(models_dir) -> MLModel | None:
    """The installed model, or None when models_dir has no ml_meta.json.

    Raises ValueError for an incompatible bundle and RuntimeError when LightGBM is missing.
    """
    models_dir = Path(models_dir)
    meta = read_meta(models_dir)
    if meta is None:
        return None
    import lightgbm
    booster = lightgbm.Booster(model_file=str(models_dir / MODEL_FILE))
    names = booster.feature_name()
    if names and list(names) != list(meta["features"]):
        raise ValueError("model features do not match ml_meta.json - re-export the model from the notebook")
    comp_path = models_dir / COMPONENT_FILE
    comp = lightgbm.Booster(model_file=str(comp_path)) if comp_path.exists() else None
    return MLModel(booster=booster, meta=meta, component_booster=comp)
