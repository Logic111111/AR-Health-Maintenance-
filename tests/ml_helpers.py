"""Helpers for ML tests: synthetic V1-V42 rows and a tiny model bundle (trains in < 1 s)."""
import json

import numpy as np
import pandas as pd

from ar_gen.ml_features import FEATURE_VERSION, V_FEATURES, feature_names

COMPONENTS = ["BATTERY", "COMMS", "CONTROLLER", "MECHANISM"]


def synthetic_variables(days=10, step_minutes=15, asset="AR-001", start="2026-01-01"):
    ts = pd.date_range(start, periods=int(days * 24 * 60 / step_minutes), freq=f"{step_minutes}min")
    hours = (ts - ts[0]) / pd.Timedelta(hours=1)
    df = pd.DataFrame({"asset_id": asset, "ts": ts, "telemetry_present": True,
                       "model": "OSM15-310", "comms_channel": "MOBILE"})
    for i, c in enumerate(V_FEATURES):
        df[c] = 50 + 10 * np.sin(2 * np.pi * hours / 24 + i) + 0.05 * hours
    df["v42_weather"] = np.where(np.asarray(hours) % 24 < 12, "SUNNY", "RAINY")
    return df


def write_tiny_bundle(models_dir, version=FEATURE_VERSION, with_component=True, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    names = feature_names()
    X = rng.normal(size=(400, len(names)))
    signal = X[:, names.index("v27_unsuccessful_ops_30d")]
    y = (signal + rng.normal(scale=0.5, size=400) > 0.8).astype(int)
    params = dict(objective="binary", num_leaves=4, learning_rate=0.3, min_data_in_leaf=10, verbose=-1)
    booster = lgb.train(params, lgb.Dataset(X, y, feature_name=names), num_boost_round=10)
    models_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models_dir / "model_failure30d.txt"))
    if with_component:
        yc = rng.integers(0, len(COMPONENTS), size=400)
        comp = lgb.train(dict(objective="multiclass", num_class=len(COMPONENTS), num_leaves=4, min_data_in_leaf=10,
                              verbose=-1), lgb.Dataset(X, yc, feature_name=names), num_boost_round=5)
        comp.save_model(str(models_dir / "model_component.txt"))
    meta = {"feature_version": version, "features": names, "threshold": 0.5,
            "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.4, 1.0]},
            "component_classes": COMPONENTS if with_component else [],
            "metrics": {"test_pr_auc": 0.5}, "trained_utc": "2026-10-07T00:00:00",
            "data_until_utc": "2026-10-06T00:00:00", "model_version": "test-1"}
    (models_dir / "ml_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def write_tiny_dl_bundle(folder, arch="gru", version=FEATURE_VERSION, net_version=None, steps=12, step_hours=2,
                         seed=0):
    """A deep-learning bundle with random weights: enough to test loading and scoring (nothing is trained)."""
    import torch

    from ar_gen.dl_model import NET_VERSION, ARNet
    names = feature_names()
    torch.manual_seed(seed)
    config = {"arch": arch, "n_inputs": len(names), "n_components": len(COMPONENTS), "width": 8, "steps": steps}
    net = ARNet(**config).eval()
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "state_dict": net.state_dict()}, folder / "model_dl.pt")
    meta = {"model_type": "pytorch-sequence", "net_version": NET_VERSION if net_version is None else net_version,
            "arch": arch, "net_config": config, "feature_version": version, "features": names,
            "window_steps": steps, "step_hours": step_hours,
            "scaling": {"log1p": ["v21_co_total", "v10_minutes_on_battery_30d"], "mean": [50.0] * len(names),
                        "std": [10.0] * len(names), "clip": 8.0},
            "threshold": 0.5, "calibration": {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.4, 1.0]},
            "component_classes": COMPONENTS, "metrics": {"test_pr_auc": 0.5, "lightgbm_failures_caught": "1 / 2"},
            "trained_utc": "2026-10-12T00:00:00", "data_until_utc": "2026-10-06 18:00:00", "model_version": "dl-test-1"}
    (folder / "dl_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta
