"""Simulated NOJA OSM15-310 / OSM27-310 + RC cubicle fleet (deterministic)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

import sqlalchemy as sa

from ..db.writer import rollback_after
from ..sim.engine import Simulator
from ..sim.fleet import asset_master_rows, create_fleet, firmware_rows, write_profiles
from ..sim.state import SimState
from ..sim.timeutil import ms_to_dt
from .base import RawBatch, RecloserSource

log = logging.getLogger(__name__)


class SimulatedNojaSource(RecloserSource):
    name = "simulated"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state_path: Path = cfg["paths"]["state_dir"] / "sim_state.json"
        self.state: SimState | None = None

    # ------------------------------------------------------------ lifecycle
    def exists(self) -> bool:
        return self.state_path.exists()

    def seed(self, start_ms: int) -> dict:
        """Create the fleet; returns the rows to insert (state is saved by :meth:`commit_seed`)."""
        self.state = create_fleet(self.cfg, start_ms)
        write_profiles(self.state, self.cfg, self.cfg["paths"]["output_dir"])
        return {"firmware_release": firmware_rows(self.cfg), "asset_master": asset_master_rows(self.state)}

    def commit_seed(self) -> None:
        self.state.save(self.state_path)

    def load(self) -> SimState:
        if self.state is None:
            if not self.exists():
                raise FileNotFoundError(f"No simulator state at {self.state_path} - run 'seed' first")
            self.state = SimState.load(self.state_path)
            if self.state.config_hash != self.cfg["config_hash"]:
                log.warning("config.yaml changed since seeding (hash %s -> %s); new data follows the "
                            "current settings", self.state.config_hash, self.cfg["config_hash"])
        return self.state

    # ------------------------------------------------------------ RecloserSource
    def last_collected_ms(self) -> int:
        return self.load().last_ms

    def collect(self, until_ms: int, chunk_intervals: int = 96) -> Iterator[RawBatch]:
        yield from Simulator(self.cfg, self.load()).iter_chunks(until_ms, chunk_intervals)

    def commit(self, batch: RawBatch) -> None:
        self.state.save(self.state_path)
        if batch.ground_truth:
            self._append_ground_truth(batch.ground_truth)

    def recover(self, engine: sa.Engine) -> None:
        st = self.load()
        rollback_after(engine, ms_to_dt(st.last_ms), st.next_ids)

    # ------------------------------------------------------------ hidden truth file
    def ground_truth_path(self) -> Path:
        return self.cfg["paths"]["output_dir"] / "ground_truth_daily.csv"

    def _append_ground_truth(self, rows: list[dict]) -> None:
        path = self.ground_truth_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
