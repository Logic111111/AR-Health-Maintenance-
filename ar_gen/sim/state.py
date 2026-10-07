"""Simulator state: everything needed to continue the simulation exactly where it stopped.

The state contains the hidden "truth" (true battery capacity, mechanism wear, ...).
It is persisted to ``state/sim_state.json`` and is never written to the database.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEGRADATION_PROCESSES = ("BATTERY_DEGRADING", "MECHANISM_WEAR", "COMMS_PROBLEM", "CONTROLLER_PROBLEM")


@dataclass
class AreaState:
    idx: int
    name: str
    monsoon: str
    coastal: bool
    temp_offset: float
    diurnal_amp: float
    weather: str = "SUNNY"
    weather_hour: int = -1
    outage_start: int | None = None
    outage_until: int | None = None


@dataclass
class AssetState:
    # ---- identity and static configuration
    idx: int
    asset_id: str
    area_idx: int
    feeder_name: str
    model: str
    rated_voltage_kv: float
    system_kv: float
    rated_break_ka: float
    rated_fault_ops: int
    rated_mech_ops: int
    feeder_type: str
    peak_a: float
    imbalance: list
    v_offset: float
    exposure: float
    prospective_fault_ka: float
    oc_pickup_a: float
    ef_pickup_a: float
    coastal: bool
    distance_coast_km: float
    harsh: bool
    solar_gain: float
    channel: str
    base_signal_dbm: float
    has_rs485: bool
    ctrl_load_a: float
    ext_load_a: float
    ext_load_time_min: int
    to_base_ms: float
    tc_base_ms: float
    rc_model: str
    relay_module: str
    profiles: list
    notes: dict
    # ---- asset-master fields that change over time
    serial: str
    manufactured: str
    installed: str
    last_maint: str
    firmware: str
    battery_model: str
    battery_type: str
    battery_interval_y: int
    battery_install: str
    # ---- electrical / UPS state
    switch: str = "CLOSED"
    open_reason: str | None = None
    ctrl: str = "RUN"                      # RUN / STANDBY / OFF / BOOT
    standby_since: int | None = None
    boot_at: int | None = None
    ac_on: bool = True
    ac_off_since: int | None = None
    aux_from: int | None = None
    aux_until: int | None = None
    dead_from: int | None = None
    dead_until: int | None = None
    ext_on: bool = True
    ext_overload_until: int | None = None
    shutdown_hours: float | None = None    # hours on battery before the last shutdown
    # ---- battery (hidden truth)
    soc: float = 1.0
    init_cap: float = 1.0
    age_eff_days: float = 0.0
    cycle_loss: float = 0.0
    extra_loss: float = 0.0
    shorted: bool = False
    charger_fault: bool = False
    charger_fault_since: int | None = None
    batt_temp: float = 30.0
    cub_temp: float = 30.0
    dis_start_soc: float | None = None
    dis_min_soc: float = 1.0
    deep_penalized: bool = False
    next_test_ms: int = 0
    last_test_ms: int | None = None
    consecutive_failed: int = 0
    # ---- lifetime counters (as the RC reports them)
    co_total: int = 0
    wear: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # ---- hidden degradation
    mech_deg: float = 0.0
    comms_deg: float = 0.0
    ctrl_deg: float = 0.0
    pollution: float = 0.0
    onsets: dict = field(default_factory=dict)
    rates: dict = field(default_factory=dict)
    persistent_fault: str | None = None
    # ---- communications
    connected: bool = True
    modem_on: bool = True
    reconnect_at: int | None = None
    restart_at: int | None = None
    disconnected_since: int | None = None
    # ---- misc dynamic
    load_noise: float = 0.0
    alarms: dict = field(default_factory=dict)      # signal -> {"cat", "since", "clear_at"}
    queue: list = field(default_factory=list)       # heap of [t_ms, seq, kind, payload]
    active_fault: dict | None = None
    failed: dict = field(default_factory=dict)      # component -> open failure_id
    next_pm_ms: int | None = None
    in_maintenance: bool = False


@dataclass
class SimState:
    seed: int
    config_hash: str
    sim_start_ms: int
    last_ms: int
    next_ids: dict
    seq: int
    last_gt_date: str
    areas: list
    assets: list

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimState":
        d = dict(d)
        d["areas"] = [AreaState(**x) for x in d["areas"]]
        d["assets"] = [AssetState(**x) for x in d["assets"]]
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "SimState":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def new_ids() -> dict[str, int]:
    return {"op": 1, "fault": 1, "alarm": 1, "test": 1, "comms": 1, "failure": 1, "maint": 1}
