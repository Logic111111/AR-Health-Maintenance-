"""Shared helpers for simulator tests."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from ar_gen.config import load_config
from ar_gen.sim.engine import Simulator
from ar_gen.sim.fleet import create_fleet
from ar_gen.sim.timeutil import dt_to_ms

START = datetime(2025, 3, 1)

# Make rare events frequent so a few simulated days exercise every code path.
BUSY = {
    "protection": {"fault_rate_per_year": 600.0},
    "outages": {"area_rate_per_year": 250, "local_aux_rate_per_year": 25},
    "switching": {"manual_ops_per_year": 150, "mech_trip_per_year": 20},
    "comms": {"rs485_errors_per_month": 20},
}


def make_cfg(n_assets=3, overrides=None):
    o = {"fleet": {"n_assets": n_assets}}
    if overrides:
        for k, v in overrides.items():
            o.setdefault(k, {})
            if isinstance(v, dict):
                o[k].update(v)
            else:
                o[k] = v
    return load_config(overrides=o, load_dotenv_file=False)


def run(cfg, days, start=START, chunk_intervals=96, state=None):
    """Simulate ``days`` and return (rows_by_table, asset_rows, final_state)."""
    start_ms = dt_to_ms(start)
    if state is None:
        state = create_fleet(cfg, start_ms)
    sim = Simulator(cfg, state)
    rows = defaultdict(list)
    for chunk in sim.iter_chunks(start_ms + days * 86_400_000, chunk_intervals):
        for name, lst in chunk.rows.items():
            rows[name].extend(lst)
    return rows, sim.state
