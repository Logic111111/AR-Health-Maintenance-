"""ML dataset export for Phase 2 (spec section 8).

One row per recloser per 15 minutes: V1-V42, sub-scores, factor scores, HI, class and
failure labels taken from failure_log. Rows near the end of the data whose label
horizon is not fully observed are flagged ``censored_7d`` / ``censored_30d``.
"""
from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.reader import asset_frame, read_df
from .pipeline import get_engine
from .variables import to_ms

log = logging.getLogger(__name__)
DAY_MS = 86_400_000


def failure_labels(ts_ms: np.ndarray, fail_ms: np.ndarray, fail_comp: np.ndarray, data_end_ms: int) -> dict:
    """Labels for one asset: time to the next failure strictly after each row."""
    order = np.argsort(fail_ms, kind="stable")
    fail_ms, fail_comp = fail_ms[order], fail_comp[order]
    k = np.searchsorted(fail_ms, ts_ms, side="right")
    has = k < len(fail_ms)
    nxt = np.where(has, fail_ms[np.minimum(k, max(len(fail_ms) - 1, 0))] if len(fail_ms) else 0, 0)
    days = np.where(has, (nxt - ts_ms) / DAY_MS, np.nan)
    comp = np.where(has, fail_comp[np.minimum(k, max(len(fail_comp) - 1, 0))] if len(fail_comp) else None, None)
    remaining = (data_end_ms - ts_ms) / DAY_MS
    f7 = has & (days <= 7)
    f30 = has & (days <= 30)
    return {"failure_next_7d": f7, "failure_next_30d": f30, "next_failure_component": comp,
            "days_to_next_failure": np.round(days, 4),
            "censored_7d": ~f7 & (remaining < 7), "censored_30d": ~f30 & (remaining < 30)}


def open_failure_flags(ts_ms: np.ndarray, failures: pd.DataFrame, repairs: dict) -> np.ndarray:
    """True while a recorded failure of this asset is waiting for its corrective repair."""
    flag = np.zeros(len(ts_ms), dtype=bool)
    for fid, f_ms in zip(failures["failure_id"], to_ms(failures["ts"])):
        r_ms = repairs.get(fid, np.iinfo(np.int64).max)
        flag |= (ts_ms >= f_ms) & (ts_ms < r_ms)
    return flag


def export_dataset(cfg: dict, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else cfg["paths"]["output_dir"] / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = get_engine(cfg)
    v, h = schema.ar_variables_15min, schema.ar_health_index_15min
    with engine.connect() as conn:
        assets = asset_frame(conn)
        failures = read_df(conn, sa.select(schema.failure_log))
        maint = read_df(conn, sa.select(schema.maintenance_log.c.failure_id, schema.maintenance_log.c.ts)
                        .where(schema.maintenance_log.c.failure_id.is_not(None)))
        data_end = conn.execute(sa.select(sa.func.max(h.c.ts))).scalar()
    if data_end is None:
        raise RuntimeError("No Health Index rows yet - run backfill first")
    data_end_ms = int(to_ms([data_end])[0])
    repairs = dict(zip(maint["failure_id"], to_ms(maint["ts"])))

    path = out_dir / "ar_ml_dataset.csv.gz"
    n_rows, first = 0, True
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=5) as fh:
        for asset in assets.to_dict("records"):
            aid = asset["asset_id"]
            with engine.connect() as conn:
                vars_df = read_df(conn, sa.select(v).where(v.c.asset_id == aid).order_by(v.c.ts))
                hi_df = read_df(conn, sa.select(h).where(h.c.asset_id == aid).order_by(h.c.ts))
            df = vars_df.drop(columns=["computed_at"]).merge(
                hi_df.drop(columns=["computed_at"]), on=["asset_id", "ts"], how="inner")
            if df.empty:
                continue
            ts_ms = to_ms(df["ts"])
            fa = failures[failures["asset_id"] == aid]
            labels = failure_labels(ts_ms, to_ms(fa["ts"]), fa["component"].to_numpy(dtype=object), data_end_ms)
            df.insert(2, "model", asset["model"])
            df.insert(3, "comms_channel", asset["comms_channel"])
            df["failed_now"] = open_failure_flags(ts_ms, fa, repairs)
            for k, val in labels.items():
                df[k] = val
            df.to_csv(fh, header=first, index=False, date_format="%Y-%m-%d %H:%M:%S")
            first = False
            n_rows += len(df)
            log.info("exported %s (%d rows so far)", aid, n_rows)

    result = {"dataset": path, "rows": n_rows}
    gt = cfg["paths"]["output_dir"] / "ground_truth_daily.csv"
    if gt.exists():
        g = pd.read_csv(gt).drop_duplicates(["date", "asset_id"], keep="last").sort_values(["asset_id", "date"])
        gt_path = out_dir / "ar_ml_ground_truth.csv.gz"
        g.to_csv(gt_path, index=False, compression="gzip")
        result["ground_truth"] = gt_path
    prof = cfg["paths"]["output_dir"] / "asset_profiles.csv"
    if prof.exists():
        shutil.copyfile(prof, out_dir / "ar_ml_asset_profiles.csv")
        result["profiles"] = out_dir / "ar_ml_asset_profiles.csv"
    return result
