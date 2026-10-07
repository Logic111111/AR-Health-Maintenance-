"""ML feature recipe shared by the Google Colab notebook and the live scorer.

KEEP THIS FILE SELF-CONTAINED (only numpy and pandas): the Colab notebook embeds it verbatim,
so training and live scoring build exactly the same features.

Input: V1-V42 rows (one per recloser per interval, every interval present) with the columns
asset_id, ts, telemetry_present, v01..v42, model, comms_channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_VERSION = 1

V_FEATURES = [
    "v01_battery_voltage_v", "v02_discharge_episodes_24h", "v03_soh_est_pct", "v04_battery_life_used_pct",
    "v05_ac_failures_30d", "v06_max_dod_30d_pct", "v07_battery_alarms_30d", "v08_battery_temp_age_years",
    "v09_auto_test_failed_count", "v10_minutes_on_battery_30d", "v11_wear_max_pct", "v12_lightning_faults_30d",
    "v13_permanent_faults_30d", "v14_temporary_faults_30d", "v15_earth_faults_30d", "v16_sef_faults_30d",
    "v17_oc_faults_30d", "v18_max_fault_current_30d_a", "v19_faults_24h", "v20_lockouts_30d", "v21_co_total",
    "v22_open_ops_30d", "v23_close_ops_30d", "v24_protection_trips_30d", "v25_auto_reclose_ops_30d",
    "v26_switching_ops_24h", "v27_unsuccessful_ops_30d", "v28_scada_availability_24h_pct",
    "v29_comm_failures_24h", "v30_modem_restarts_30d", "v31_signal_strength_24h_dbm", "v32_reconnects_24h",
    "v33_module_faults_30d", "v34_diag_warnings_30d", "v35_firmware_versions_behind", "v36_rs485_errors_30d",
    "v37_sg_age_years", "v38_years_in_service", "v39_days_since_maintenance", "v40_ambient_temp_c",
    "v41_coastal_flag",
]

# Variables whose 7-day change and 7-day mean are added: failures show up as things getting worse.
TREND_VARS = [
    "v01_battery_voltage_v", "v03_soh_est_pct", "v06_max_dod_30d_pct", "v07_battery_alarms_30d",
    "v10_minutes_on_battery_30d", "v11_wear_max_pct", "v27_unsuccessful_ops_30d",
    "v28_scada_availability_24h_pct", "v29_comm_failures_24h", "v30_modem_restarts_30d",
    "v31_signal_strength_24h_dbm", "v33_module_faults_30d", "v34_diag_warnings_30d",
]
TREND_HOURS = 168

# Fixed category lists -> the one-hot columns never change between training and scoring.
CATEGORIES = {
    "v42_weather": ["SUNNY", "CLOUDY", "RAINY", "STORM"],
    "model": ["OSM15-310", "OSM27-310"],
    "comms_channel": ["MOBILE", "FIBRE"],
}


def feature_names() -> list[str]:
    names = ["telemetry_present", *V_FEATURES]
    names += [f"{v}_{kind}" for v in TREND_VARS for kind in ("chg7d", "mean7d")]
    names += [f"{col}__{cat}" for col, cats in CATEGORIES.items() for cat in cats]
    return names


def build_features(df: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
    """Model inputs for every row of ``df`` (same row order and index).

    step_minutes: spacing of the rows of one recloser (60 for the hourly training file,
    15 for live data). The 7-day trend windows are defined in time, so both agree.
    """
    rows = max(1, int(round(TREND_HOURS * 60 / step_minutes)))
    order = df.sort_values(["asset_id", "ts"], kind="stable").index
    d = df.loc[order]
    out = pd.DataFrame(index=d.index)
    present = d["telemetry_present"]
    if present.dtype == object:
        present = present.astype(str).str.lower().isin(["true", "1"])
    out["telemetry_present"] = present.astype(float)
    for c in V_FEATURES:
        out[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    groups = d["asset_id"]
    for v in TREND_VARS:
        x = out[v]
        out[f"{v}_chg7d"] = x - x.groupby(groups).shift(rows)
        out[f"{v}_mean7d"] = x.groupby(groups).transform(lambda s: s.rolling(rows, min_periods=1).mean())
    for col, cats in CATEGORIES.items():
        values = d[col].astype(str) if col in d.columns else pd.Series("", index=d.index)
        for cat in cats:
            out[f"{col}__{cat}"] = (values == cat).astype(float)
    out = out[feature_names()]
    return out.loc[df.index].astype(np.float64)
