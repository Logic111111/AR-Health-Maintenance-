"""Generate an ML training dataset directly, without a database.

Simulates a fleet in memory and, piece by piece, calculates V1-V42, the Health Index
and the failure labels, writing the same columns as ``export-dataset``. Memory stays
small: only the last ~40 days of raw data are kept, finished pieces go to temporary
files and the labels (which need the future failure log) are added in a final pass.
"""
from __future__ import annotations

import gzip
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import failure_labels, open_failure_flags
from .db import schema
from .health_index import compute_health_index
from .pipeline import _round_frame
from .sim.engine import Simulator
from .sim.fleet import asset_master_rows, create_fleet, firmware_rows, write_profiles
from .sim.timeutil import DAY_MS, MIN_MS, ms_to_dt
from .variables import RawInputs, compute_variables, to_ms

log = logging.getLogger(__name__)

EVENT_TABLES = {"fault_events": "start_ts", "co_operations": "ts", "alarm_events": "ts", "comms_events": "ts"}
HISTORY_TABLES = ("battery_tests", "maintenance_log", "failure_log")
HI_COLUMNS = [c.name for c in schema.ar_health_index_15min.columns if c.name != "computed_at"]


def _frame(rows: list[dict], table: str) -> pd.DataFrame:
    cols = [c.name for c in schema.TABLES[table].columns]
    return pd.DataFrame(rows, columns=cols)


class _Buffer:
    """Raw rows needed for the variable windows (recent events + full small histories)."""

    def __init__(self):
        self.telemetry = _frame([], "telemetry_15min")
        self.events = {t: _frame([], t) for t in EVENT_TABLES}
        self.history = {t: _frame([], t) for t in HISTORY_TABLES}

    def add(self, rows: dict) -> None:
        def cat(old, new):
            return new if old.empty else (old if new.empty else pd.concat([old, new], ignore_index=True))
        self.telemetry = cat(self.telemetry, _frame(rows.get("telemetry_15min", []), "telemetry_15min"))
        for t in EVENT_TABLES:
            self.events[t] = cat(self.events[t], _frame(rows.get(t, []), t))
        for t in HISTORY_TABLES:
            self.history[t] = cat(self.history[t], _frame(rows.get(t, []), t))

    def trim(self, keep_from) -> None:
        self.telemetry = self.telemetry[self.telemetry["ts"] > keep_from].reset_index(drop=True)
        for t, col in EVENT_TABLES.items():
            self.events[t] = self.events[t][self.events[t][col] > keep_from].reset_index(drop=True)

    def inputs(self, cfg: dict, assets: pd.DataFrame, firmware: pd.DataFrame, s: int, e: int,
               asset_start: dict) -> RawInputs:
        """Same windows as db.reader.load_inputs."""
        vc, step = cfg["variables"], int(cfg["interval_minutes"]) * MIN_MS
        tel_lo = ms_to_dt(s - int(vc["long_window_days"]) * DAY_MS - 2 * step)
        ev_lo = ms_to_dt(s - (int(vc["long_window_days"]) + 7) * DAY_MS)
        hi = ms_to_dt(e)
        tl = self.telemetry

        def win(t, col):
            df = self.events[t]
            return df[(df[col] > ev_lo) & (df[col] <= hi)]

        tests = self.history["battery_tests"]
        return RawInputs(assets=assets, telemetry=tl[(tl["ts"] > tel_lo) & (tl["ts"] <= hi)],
                         faults=win("fault_events", "start_ts"), ops=win("co_operations", "ts"),
                         alarms=win("alarm_events", "ts"), tests=tests[tests["ts"] <= hi],
                         comms=win("comms_events", "ts"), maintenance=self.history["maintenance_log"],
                         firmware=firmware, asset_start=asset_start)


def generate_ml_dataset(cfg: dict, start_ms: int, days: int, out_dir: Path, hourly_copy: bool = True,
                        chunk_days: int = 10) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(cfg["interval_minutes"]) * MIN_MS
    start_ms -= start_ms % step
    end_ms = start_ms + days * DAY_MS
    t_start = time.time()

    state = create_fleet(cfg, start_ms)
    write_profiles(state, cfg, out_dir)
    shutil.move(out_dir / "asset_profiles.csv", out_dir / "ar_ml_asset_profiles.csv")
    shutil.move(out_dir / "asset_profiles.json", out_dir / "ar_ml_asset_profiles.json")
    asset_start = {a.asset_id: start_ms for a in state.assets}
    firmware = pd.DataFrame(firmware_rows(cfg))
    buf = _Buffer()
    ground_truth: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ar_mlgen_", dir=out_dir))
    pieces: list[Path] = []
    total = (end_ms - start_ms) // step

    def compute(s: int, e: int) -> None:
        assets = pd.DataFrame(asset_master_rows(state))
        inp = buf.inputs(cfg, assets, firmware, s, e, asset_start)
        variables = compute_variables(inp, s, e, cfg)
        hi = compute_health_index(variables, assets, cfg["health_index"])   # scored before rounding, like the DB route
        variables = _round_frame(variables)
        df = variables.merge(hi[HI_COLUMNS], on=["asset_id", "ts"], how="inner")
        info = assets.set_index("asset_id")
        df.insert(2, "model", df["asset_id"].map(info["model"]))
        df.insert(3, "comms_channel", df["asset_id"].map(info["comms_channel"]))
        path = tmp_dir / f"piece_{len(pieces):04d}.pkl"
        df.to_pickle(path)
        pieces.append(path)

    # Pass 1: simulate + V1-V42 + HI, one piece behind the simulation so faults that
    # finish in the next piece are already in the logs (same as a DB backfill).
    pending, prev_end = None, start_ms
    sim = Simulator(cfg, state)
    for batch in sim.iter_chunks(end_ms, chunk_days * 96):
        buf.add(batch.rows)
        ground_truth.extend(batch.ground_truth)
        if pending:
            compute(*pending)
            buf.trim(ms_to_dt(pending[1] - (int(cfg["variables"]["long_window_days"]) + 8) * DAY_MS))
        pending, prev_end = (prev_end, batch.last_ms), batch.last_ms
        done = (batch.last_ms - start_ms) // step
        log.info("simulated + calculated up to %s (%d%%, %.0f s)", ms_to_dt(batch.last_ms).date(),
                 100 * done // total, time.time() - t_start)
    if pending:
        compute(*pending)

    # Pass 2: labels from the complete failure log, then write the CSV file(s).
    failures = buf.history["failure_log"]
    maint = buf.history["maintenance_log"]
    repairs = dict(zip(maint.loc[maint["failure_id"].notna(), "failure_id"],
                       to_ms(maint.loc[maint["failure_id"].notna(), "ts"])))
    data_end_ms = state.last_ms
    path = out_dir / "ar_ml_dataset.csv.gz"
    hourly_path = out_dir / "ar_ml_dataset_hourly.csv.gz"
    n_rows = n_hourly = 0
    fh_h = gzip.open(hourly_path, "wt", newline="", encoding="utf-8", compresslevel=5) if hourly_copy else None
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=5) as fh:
        for k, piece in enumerate(pieces):
            df = pd.read_pickle(piece)
            parts = []
            for aid, part in df.groupby("asset_id", sort=True):
                part = part.sort_values("ts")
                ts_ms = to_ms(part["ts"])
                fa = failures[failures["asset_id"] == aid]
                labels = failure_labels(ts_ms, to_ms(fa["ts"]), fa["component"].to_numpy(dtype=object), data_end_ms)
                part = part.assign(failed_now=open_failure_flags(ts_ms, fa, repairs))
                for name, val in labels.items():
                    part[name] = val
                parts.append(part)
            out = pd.concat(parts, ignore_index=True)
            out.to_csv(fh, header=(k == 0), index=False, date_format="%Y-%m-%d %H:%M:%S")
            n_rows += len(out)
            if fh_h is not None:
                hourly = out[out["ts"].dt.minute == 0]
                hourly.to_csv(fh_h, header=(k == 0), index=False, date_format="%Y-%m-%d %H:%M:%S")
                n_hourly += len(hourly)
            piece.unlink()
    if fh_h is not None:
        fh_h.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    gt = (pd.DataFrame(ground_truth).drop_duplicates(["date", "asset_id"], keep="last")
          .sort_values(["asset_id", "date"]))
    gt_path = out_dir / "ar_ml_ground_truth.csv.gz"
    gt.to_csv(gt_path, index=False, compression="gzip")

    comp = failures["component"].value_counts().to_dict()
    result = {"dataset": path, "rows": n_rows, "reclosers": len(state.assets), "days": days,
              "failures": len(failures), "failures_by_component": comp, "ground_truth": gt_path,
              "profiles": out_dir / "ar_ml_asset_profiles.csv", "minutes": round((time.time() - t_start) / 60, 1)}
    if hourly_copy:
        result.update(hourly_dataset=hourly_path, hourly_rows=n_hourly)
    return result
