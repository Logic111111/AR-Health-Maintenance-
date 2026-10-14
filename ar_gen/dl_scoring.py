"""Load the deep-learning model trained in Google Colab (folder models/dl) and score stored V1-V42 rows.

The bundle written by notebooks/AR_DL_Colab.ipynb (ar_dl_model.zip):
  dl_meta.json   net_version, feature_version, features, window_steps, step_hours, scaling {log1p, mean, std,
                 clip}, threshold, calibration {x, y}, component_classes, metrics, trained_utc, model_version
  model_dl.pt    the network's size settings and weights (a PyTorch state dict, loaded with weights_only)
It is separate from the LightGBM model in models/: both can be installed side by side, and a missing or
broken deep-learning bundle never stops the Health Index or the LightGBM scores. Only this module needs PyTorch.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.reader import df_to_rows, read_df
from .db.writer import insert_rows
from .ml_features import FEATURE_VERSION, TREND_HOURS, build_features, feature_names
from .ml_scoring import risk_class
from .sim.timeutil import DAY_MS, ms_to_dt
from .variables import to_ms

log = logging.getLogger(__name__)

META_FILE, WEIGHTS_FILE = "dl_meta.json", "model_dl.pt"
REQUIRED_META = ("net_version", "feature_version", "features", "window_steps", "step_hours", "scaling",
                 "threshold", "calibration")
BATCH = 2048                                  # windows per forward pass (keeps memory small on the PC)
TORCH_HINT = "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"


def read_dl_meta(folder) -> dict | None:
    """dl_meta.json after every check that needs no model loading; None when no model is installed.

    Raises ValueError for an incompatible bundle and RuntimeError when PyTorch is missing.
    Used by the dashboard (cheap) and by load_dl_model.
    """
    folder = Path(folder)
    meta_path = folder / META_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"{meta_path} is missing {', '.join(missing)}")
    if int(meta["feature_version"]) != FEATURE_VERSION:
        raise ValueError(f"model was trained with feature version {meta['feature_version']}, this program uses "
                         f"feature version {FEATURE_VERSION} - retrain it with the current deep-learning notebook")
    unknown = [f for f in meta["features"] if f not in set(feature_names())]
    if unknown:
        raise ValueError(f"model uses features this program does not build ({', '.join(unknown[:5])}) - "
                         "retrain it with the current deep-learning notebook")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(f"PyTorch is not installed - run: {TORCH_HINT}")
    from .dl_model import NET_VERSION
    if int(meta["net_version"]) != NET_VERSION:
        raise ValueError(f"model was exported with network version {meta['net_version']}, this program uses "
                         f"network version {NET_VERSION} - retrain it with the current deep-learning notebook")
    if not (folder / WEIGHTS_FILE).exists():
        raise ValueError(f"{folder / WEIGHTS_FILE} is missing")
    return meta


@dataclass
class DLModel:
    net: object
    meta: dict

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    @property
    def model_version(self) -> str:
        return str(self.meta.get("model_version") or self.meta.get("trained_utc") or "unknown")

    @property
    def history_days(self) -> int:
        """Days of V1-V42 rows needed before a scored interval: the window plus the 7-day trend inputs."""
        window_h = (int(self.meta["window_steps"]) - 1) * float(self.meta["step_hours"])
        return int(np.ceil((window_h + TREND_HOURS) / 24)) + 1

    def window_rows(self, frame: pd.DataFrame, targets: np.ndarray) -> np.ndarray:
        """(len(targets), window_steps) row positions of frame: for each target row, the latest row of the same
        recloser at or before ts - k x step_hours (oldest first). Before the recloser's first row, its first
        row is repeated. On a regular series this is ar_dl_model.window_rows."""
        steps = int(self.meta["window_steps"])
        offsets = np.arange(-(steps - 1), 1, dtype=np.int64) * int(round(float(self.meta["step_hours"]) * 3.6e12))
        ts = pd.to_datetime(frame["ts"]).to_numpy("datetime64[ns]").astype(np.int64)
        ids = frame["asset_id"].to_numpy()
        out = np.empty((len(targets), steps), dtype=np.int64)
        tgt_ids = ids[targets]
        for aid in pd.unique(tgt_ids):
            rows = np.flatnonzero(ids == aid)
            rows = rows[np.argsort(ts[rows], kind="stable")]
            sel = np.flatnonzero(tgt_ids == aid)
            want = ts[targets[sel]][:, None] + offsets[None, :]
            out[sel] = rows[np.clip(np.searchsorted(ts[rows], want, side="right") - 1, 0, None)]
        return out

    def predict(self, frame: pd.DataFrame, X: pd.DataFrame, targets) -> pd.DataFrame:
        """Scores for the rows of ``frame`` selected by the boolean mask ``targets``.

        frame: asset_id and ts of every row (enough history before the targets); X: build_features(frame).
        """
        import torch

        from .dl_model import scale_inputs
        pos = np.flatnonzero(np.asarray(targets, dtype=bool))
        s, feats = self.meta["scaling"], list(self.meta["features"])
        scaled = scale_inputs(X[feats].to_numpy(dtype=np.float32), [feats.index(c) for c in s["log1p"]],
                              s["mean"], s["std"], float(s["clip"]))
        win = self.window_rows(frame.reset_index(drop=True), pos)
        raw, parts = [np.empty(0, dtype=np.float32)], []
        with torch.no_grad():
            for i in range(0, len(win), BATCH):
                o30, _, opart = self.net(torch.from_numpy(scaled[win[i:i + BATCH]]))
                raw.append(torch.sigmoid(o30).numpy())
                parts.append(torch.softmax(opart, dim=1).numpy())
        raw = np.concatenate(raw).astype(float)
        cal = self.meta.get("calibration") or {}
        p = np.interp(raw, np.asarray(cal["x"], float), np.asarray(cal["y"], float)) if cal.get("x") and cal.get("y") \
            else raw
        p = np.clip(p, 0.0, 1.0)
        out = pd.DataFrame({"p_fail_30d": p, "dl_healthy": p < self.threshold,
                            "dl_risk_class": risk_class(p, self.threshold)}, index=frame.index[pos])
        classes = list(self.meta.get("component_classes") or [])
        if parts and len(classes) >= 2:
            probs = np.concatenate(parts)
            top = probs.argmax(axis=1)
            out["top_component"] = [classes[i] if i < len(classes) else None for i in top]
            out["top_component_p"] = probs[np.arange(len(probs)), top].astype(float)
        else:
            out["top_component"] = None
            out["top_component_p"] = np.nan
        return out


def load_dl_model(folder) -> DLModel | None:
    """The installed deep-learning model, or None when the folder has no dl_meta.json.

    Raises ValueError for an incompatible bundle and RuntimeError when PyTorch is missing.
    """
    folder = Path(folder)
    meta = read_dl_meta(folder)
    if meta is None:
        return None
    import torch

    from .dl_model import ARNet
    saved = torch.load(folder / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    if int(config.get("n_inputs", -1)) != len(meta["features"]):
        raise ValueError("network inputs do not match dl_meta.json - re-export the model from the notebook")
    net = ARNet(**config)
    net.load_state_dict(saved["state_dict"])
    net.eval()
    return DLModel(net=net, meta=meta)


# ------------------------------------------------------------------ scoring stored V1-V42 rows
def score_range(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model: DLModel) -> int:
    """Write deep-learning predictions for interval ends in (start_ms, end_ms] (reads the window's history too)."""
    v, am, p = schema.ar_variables_15min, schema.asset_master, schema.ar_dl_prediction_15min
    with engine.connect() as conn:
        df = read_df(conn, sa.select(v, am.c.model, am.c.comms_channel)
                     .select_from(v.join(am, am.c.asset_id == v.c.asset_id))
                     .where(v.c.ts > ms_to_dt(start_ms - model.history_days * DAY_MS), v.c.ts <= ms_to_dt(end_ms))
                     .order_by(v.c.asset_id, v.c.ts))
    if df.empty:
        return 0
    X = build_features(df, int(cfg["interval_minutes"]))
    ts_ms = to_ms(df["ts"])
    sel = (ts_ms > start_ms) & (ts_ms <= end_ms)
    if not sel.any():
        return 0
    out = df.loc[sel, ["asset_id", "ts"]].join(model.predict(df, X, sel))
    out["model_version"] = model.model_version
    out["computed_at"] = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    rows = df_to_rows(out)
    with engine.begin() as conn:
        conn.execute(p.delete().where(p.c.ts > ms_to_dt(start_ms), p.c.ts <= ms_to_dt(end_ms)))
        insert_rows(conn, p, rows, batch_size=cfg["db"]["batch_size"])
    return len(rows)


def score_stored(cfg: dict, engine: sa.Engine, start_ms: int, end_ms: int, model: DLModel) -> int:
    """dl-score: predictions for a stored range, chunk by chunk."""
    step = int(cfg["interval_minutes"]) * 60_000
    chunk = int(cfg["variables"]["calc_chunk_days"]) * DAY_MS
    total, s = 0, start_ms - start_ms % step
    while s < end_ms:
        e = min(end_ms, s + chunk)
        total += score_range(cfg, engine, s, e, model)
        log.info("deep-learning scores up to %s (%d rows)", ms_to_dt(e), total)
        s = e
    return total


def last_scored_ms(engine: sa.Engine) -> int | None:
    p = schema.ar_dl_prediction_15min
    with engine.connect() as conn:
        ts = conn.execute(sa.select(sa.func.max(p.c.ts))).scalar()
    return None if ts is None else int(to_ms([ts])[0])
