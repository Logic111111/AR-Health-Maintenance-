"""Database schema (SQLAlchemy Core; portable across MySQL 8, MariaDB 10.6 and SQLite).

All timestamps are UTC, stored as DATETIME(3) on MySQL/MariaDB.
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

metadata = sa.MetaData()

DT3 = sa.DateTime().with_variant(mysql.DATETIME(fsp=3), "mysql", "mariadb")


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=True, create_constraint=False,
                   validate_strings=False, length=max(len(v) for v in values))


def _num(precision: int, scale: int) -> sa.Numeric:
    return sa.Numeric(precision, scale, asdecimal=False)


def _asset_fk() -> sa.Column:
    return sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), nullable=False)


SWITCH_STATES = ("CLOSED", "OPEN", "LOCKOUT")
WEATHER = ("SUNNY", "CLOUDY", "RAINY", "STORM")
OP_SOURCES = ("PROTECTION", "AUTO_RECLOSE", "HMI", "CMS", "SCADA", "IO", "MECH_TRIP")
FAULT_TYPES = ("OC_A", "OC_B", "OC_C", "EF", "SEF", "NPS", "UV", "OV", "UF", "OF")
FAULT_OUTCOMES = ("CLEARED_BY_RECLOSE", "LOCKOUT", "UPSTREAM_CLEARED")
COMPONENTS = ("BATTERY", "MECHANISM", "CONTACTS", "COMMS", "CONTROLLER")

# --------------------------------------------------------------------- raw tables
asset_master = sa.Table(
    "asset_master", metadata,
    sa.Column("asset_id", sa.String(16), primary_key=True),
    sa.Column("osm_serial_number", sa.String(13), nullable=False),
    sa.Column("model", sa.String(16), nullable=False),
    sa.Column("rated_voltage_kv", _num(4, 1), nullable=False),
    sa.Column("system_voltage_kv", _num(4, 1), nullable=False),
    sa.Column("rated_continuous_a", sa.SmallInteger, nullable=False),
    sa.Column("rated_break_ka", _num(4, 1), nullable=False),
    sa.Column("rated_fault_break_ops", sa.SmallInteger, nullable=False),
    sa.Column("rated_mech_ops", sa.Integer, nullable=False),
    sa.Column("manufactured_date", sa.Date, nullable=False),
    sa.Column("installed_date", sa.Date, nullable=False),
    sa.Column("last_maintenance_date", sa.Date),
    sa.Column("rc_model", sa.String(8), nullable=False),
    sa.Column("relay_module", sa.String(16), nullable=False),
    sa.Column("firmware_version", sa.String(16), nullable=False),
    sa.Column("battery_model", sa.String(8), nullable=False),
    sa.Column("battery_type", sa.String(4), nullable=False),
    sa.Column("battery_rated_ah", _num(4, 1), nullable=False),
    sa.Column("battery_service_interval_years", sa.SmallInteger, nullable=False),
    sa.Column("battery_install_date", sa.Date, nullable=False),
    sa.Column("comms_channel", _enum("comms_channel", "MOBILE", "FIBRE"), nullable=False),
    sa.Column("has_rs485_link", sa.Boolean, nullable=False),
    sa.Column("coastal_flag", sa.Boolean, nullable=False),
    sa.Column("distance_to_coast_km", _num(5, 1)),
    sa.Column("area_name", sa.String(32), nullable=False),
    sa.Column("feeder_name", sa.String(32), nullable=False),
    sa.Column("external_load_time_min", sa.SmallInteger, nullable=False),
)

telemetry_15min = sa.Table(
    "telemetry_15min", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("ts", DT3, primary_key=True),
    sa.Column("ia", _num(7, 1)), sa.Column("ib", _num(7, 1)), sa.Column("ic", _num(7, 1)),
    sa.Column("in_residual", _num(7, 1)),
    sa.Column("uab", _num(5, 2)), sa.Column("ubc", _num(5, 2)), sa.Column("uca", _num(5, 2)),
    sa.Column("frequency_hz", _num(6, 3)),
    sa.Column("switch_state", _enum("switch_state", *SWITCH_STATES), nullable=False),
    sa.Column("ac_supply_on", sa.Boolean, nullable=False),
    sa.Column("external_load_on", sa.Boolean, nullable=False),
    sa.Column("battery_voltage", _num(5, 2)),
    sa.Column("battery_current", _num(5, 2)),
    sa.Column("battery_charge_pct", sa.SmallInteger),
    sa.Column("battery_temp_c", _num(4, 1)),
    sa.Column("cubicle_temp_c", _num(4, 1)),
    sa.Column("ambient_temp_c", _num(4, 1)),
    sa.Column("humidity_pct", sa.SmallInteger),
    sa.Column("weather", _enum("weather", *WEATHER)),
    sa.Column("lightning_strikes_10km", sa.SmallInteger),
    sa.Column("comms_connected", sa.Boolean, nullable=False),
    sa.Column("signal_strength_dbm", _num(5, 1)),
    sa.Column("co_total", sa.Integer, nullable=False),
    sa.Column("mechanical_wear_pct", _num(7, 3), nullable=False),
    sa.Column("contact_wear_pct_a", _num(7, 3), nullable=False),
    sa.Column("contact_wear_pct_b", _num(7, 3), nullable=False),
    sa.Column("contact_wear_pct_c", _num(7, 3), nullable=False),
    sa.Index("ix_telemetry_ts", "ts"),
)

fault_events = sa.Table(
    "fault_events", metadata,
    sa.Column("fault_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("start_ts", DT3, nullable=False),
    sa.Column("end_ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("fault_type", _enum("fault_type", *FAULT_TYPES), nullable=False),
    sa.Column("phases", sa.String(4), nullable=False),
    sa.Column("max_fault_current_a", _num(8, 1), nullable=False),
    sa.Column("duration_ms", sa.Integer, nullable=False),
    sa.Column("reclose_attempts", sa.SmallInteger, nullable=False),
    sa.Column("outcome", _enum("fault_outcome", *FAULT_OUTCOMES), nullable=False),
    sa.Index("ix_fault_asset_ts", "asset_id", "start_ts"),
)

co_operations = sa.Table(
    "co_operations", metadata,
    sa.Column("op_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("op_type", _enum("op_type", "OPEN", "CLOSE"), nullable=False),
    sa.Column("source", _enum("op_source", *OP_SOURCES), nullable=False),
    sa.Column("ar_state", sa.String(16)),
    sa.Column("trip_ia", _num(8, 1)), sa.Column("trip_ib", _num(8, 1)),
    sa.Column("trip_ic", _num(8, 1)), sa.Column("trip_in", _num(8, 1)),
    sa.Column("operating_time_ms", _num(6, 1)),
    sa.Column("result", _enum("op_result", "SUCCESS", "FAIL"), nullable=False),
    sa.Column("fault_id", sa.BigInteger, sa.ForeignKey("fault_events.fault_id")),
    sa.Index("ix_co_asset_ts", "asset_id", "ts"),
)

alarm_events = sa.Table(
    "alarm_events", metadata,
    sa.Column("event_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("category", _enum("alarm_category", "WARNING", "MALFUNCTION", "ALERT"), nullable=False),
    sa.Column("signal_name", sa.String(48), nullable=False),
    sa.Column("state", _enum("alarm_state", "RAISED", "CLEARED", "EVENT"), nullable=False),
    sa.Column("detail", sa.String(128)),
    sa.Index("ix_alarm_asset_ts", "asset_id", "ts"),
)

battery_tests = sa.Table(
    "battery_tests", metadata,
    sa.Column("test_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("trigger", _enum("test_trigger", "AUTO", "MANUAL"), nullable=False),
    sa.Column("result", _enum("test_result", "PASSED", "FAILED", "NOT_PERFORMED"), nullable=False),
    sa.Column("not_performed_reason", sa.String(16)),
    sa.Column("voltage_before", _num(5, 2)),
    sa.Column("voltage_min_during_test", _num(5, 2)),
    sa.Column("test_load_current_a", _num(5, 1)),
    sa.Index("ix_btest_asset_ts", "asset_id", "ts"),
)

comms_events = sa.Table(
    "comms_events", metadata,
    sa.Column("event_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("channel", _enum("comms_event_channel", "MOBILE", "FIBRE", "RS485"), nullable=False),
    sa.Column("event", _enum("comms_event", "DISCONNECT", "RECONNECT", "MODEM_RESTART", "RS485_ERROR"),
              nullable=False),
    sa.Index("ix_comms_asset_ts", "asset_id", "ts"),
)

failure_log = sa.Table(
    "failure_log", metadata,
    sa.Column("failure_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("component", _enum("failure_component", *COMPONENTS), nullable=False),
    sa.Column("detected_by", _enum("detected_by", "ALARM", "BATTERY_TEST", "PATROL", "SCADA"),
              nullable=False),
    sa.Column("description", sa.String(255), nullable=False),
    sa.Index("ix_failure_asset_ts", "asset_id", "ts"),
)

maintenance_log = sa.Table(
    "maintenance_log", metadata,
    sa.Column("maint_id", sa.BigInteger, primary_key=True, autoincrement=False),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("component", _enum("maint_component", *COMPONENTS), nullable=False),
    sa.Column("maint_type", _enum("maint_type", "PREVENTIVE", "CORRECTIVE"), nullable=False),
    sa.Column("failure_id", sa.BigInteger, sa.ForeignKey("failure_log.failure_id")),
    sa.Column("description", sa.String(255), nullable=False),
    sa.Column("details", sa.JSON),
    sa.Index("ix_maint_asset_ts", "asset_id", "ts"),
)

firmware_release = sa.Table(
    "firmware_release", metadata,
    sa.Column("version", sa.String(16), primary_key=True),
    sa.Column("release_date", sa.Date, nullable=False),
)

# --------------------------------------------------------------------- dashboard / maintenance input
asset_location = sa.Table(
    "asset_location", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("latitude", sa.Double(), nullable=False),
    sa.Column("longitude", sa.Double(), nullable=False),
    sa.Column("source", sa.String(16), nullable=False),          # SIMULATED or GPS
)

# Manual "Maintenance done" resets of the Health Index (automatic resets come from
# CORRECTIVE rows in maintenance_log). part_groups: comma list of BATTERY, SWITCHING,
# COMMS, CONTROLLER, or ALL.
hi_reset_log = sa.Table(
    "hi_reset_log", metadata,
    sa.Column("reset_id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("ts", DT3, nullable=False),
    _asset_fk(),
    sa.Column("part_groups", sa.String(64), nullable=False),
    sa.Column("note", sa.String(255)),
    sa.Column("created_at", DT3, nullable=False),
    sa.Index("ix_reset_asset_ts", "asset_id", "ts"),
)

# --------------------------------------------------------------------- calculated tables
# V1-V42 final data, one row per asset per 15 minutes.
VARIABLE_COLUMNS: list[tuple[str, sa.types.TypeEngine]] = [
    ("telemetry_present", sa.Boolean()),
    ("v01_battery_voltage_v", sa.Double()),
    ("v02_discharge_episodes_24h", sa.Integer()),
    ("v03_soh_est_pct", sa.Double()),
    ("v04_battery_life_used_pct", sa.Double()),
    ("v05_ac_failures_30d", sa.Integer()),
    ("v06_max_dod_30d_pct", sa.Double()),
    ("v07_battery_alarms_30d", sa.Integer()),
    ("v08_battery_temp_age_years", sa.Double()),
    ("v09_auto_test_failed_count", sa.Integer()),
    ("v10_minutes_on_battery_30d", sa.Double()),
    ("v11_wear_max_pct", sa.Double()),
    ("v12_lightning_faults_30d", sa.Integer()),
    ("v13_permanent_faults_30d", sa.Integer()),
    ("v14_temporary_faults_30d", sa.Integer()),
    ("v15_earth_faults_30d", sa.Integer()),
    ("v16_sef_faults_30d", sa.Integer()),
    ("v17_oc_faults_30d", sa.Integer()),
    ("v18_max_fault_current_30d_a", sa.Double()),
    ("v19_faults_24h", sa.Integer()),
    ("v20_lockouts_30d", sa.Integer()),
    ("v21_co_total", sa.Integer()),
    ("v22_open_ops_30d", sa.Integer()),
    ("v23_close_ops_30d", sa.Integer()),
    ("v24_protection_trips_30d", sa.Integer()),
    ("v25_auto_reclose_ops_30d", sa.Integer()),
    ("v26_switching_ops_24h", sa.Integer()),
    ("v27_unsuccessful_ops_30d", sa.Integer()),
    ("v28_scada_availability_24h_pct", sa.Double()),
    ("v29_comm_failures_24h", sa.Integer()),
    ("v30_modem_restarts_30d", sa.Integer()),
    ("v31_signal_strength_24h_dbm", sa.Double()),
    ("v32_reconnects_24h", sa.Integer()),
    ("v33_module_faults_30d", sa.Integer()),
    ("v34_diag_warnings_30d", sa.Integer()),
    ("v35_firmware_versions_behind", sa.Integer()),
    ("v36_rs485_errors_30d", sa.Integer()),
    ("v37_sg_age_years", sa.Double()),
    ("v38_years_in_service", sa.Double()),
    ("v39_days_since_maintenance", sa.Double()),
    ("v40_ambient_temp_c", sa.Double()),
    ("v41_coastal_flag", sa.SmallInteger()),
    ("v42_weather", sa.String(8)),
]
VARIABLE_NAMES = [name for name, _ in VARIABLE_COLUMNS]
V_COLUMNS = [n for n in VARIABLE_NAMES if n.startswith("v")]          # the 42 variables
V_SHORT = {n[:3]: n for n in V_COLUMNS}                                  # "v01" -> full name

ar_variables_15min = sa.Table(
    "ar_variables_15min", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("ts", DT3, primary_key=True),
    *[sa.Column(name, typ) for name, typ in VARIABLE_COLUMNS],
    sa.Column("computed_at", DT3, nullable=False),
    sa.Index("ix_variables_ts", "ts"),
)

FACTOR_COLUMNS = ["f_battery", "f_faults", "f_switching", "f_comms", "f_controlling",
                  "f_aging", "f_environment"]
SCORE_COLUMNS = [f"s{i:02d}" for i in range(1, 43)]

ar_health_index_15min = sa.Table(
    "ar_health_index_15min", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("ts", DT3, primary_key=True),
    sa.Column("health_index", sa.Double(), nullable=False),
    sa.Column("health_class", sa.String(8), nullable=False),
    sa.Column("is_healthy", sa.Boolean, nullable=False),
    sa.Column("cap_applied", sa.String(8)),
    *[sa.Column(c, sa.Double()) for c in FACTOR_COLUMNS],
    sa.Column("worst_variables", sa.String(64)),
    *[sa.Column(c, sa.Double()) for c in SCORE_COLUMNS],
    sa.Column("computed_at", DT3, nullable=False),
    sa.Index("ix_hi_ts", "ts"),
)

# ML risk per recloser per 15 min (only when a model trained in Colab is installed in models/).
ar_ml_prediction_15min = sa.Table(
    "ar_ml_prediction_15min", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("ts", DT3, primary_key=True),
    sa.Column("p_fail_30d", sa.Double(), nullable=False),
    sa.Column("ml_risk_class", sa.String(10), nullable=False),
    sa.Column("ml_healthy", sa.Boolean, nullable=False),
    sa.Column("top_component", sa.String(16)),
    sa.Column("top_component_p", sa.Double()),
    sa.Column("model_version", sa.String(40), nullable=False),
    sa.Column("computed_at", DT3, nullable=False),
    sa.Index("ix_mlpred_ts", "ts"),
)

# Daily Health Index (and ML) summary per recloser per local day (Asia/Colombo).
# Deep-learning model (models/dl, notebooks/AR_DL_Colab.ipynb): separate from the LightGBM predictions.
ar_dl_prediction_15min = sa.Table(
    "ar_dl_prediction_15min", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("ts", DT3, primary_key=True),
    sa.Column("p_fail_30d", sa.Double(), nullable=False),
    sa.Column("dl_risk_class", sa.String(10), nullable=False),
    sa.Column("dl_healthy", sa.Boolean, nullable=False),
    sa.Column("top_component", sa.String(16)),
    sa.Column("top_component_p", sa.Double()),
    sa.Column("model_version", sa.String(40), nullable=False),
    sa.Column("computed_at", DT3, nullable=False),
    sa.Index("ix_dlpred_ts", "ts"),
)

ar_daily_summary = sa.Table(
    "ar_daily_summary", metadata,
    sa.Column("asset_id", sa.String(16), sa.ForeignKey("asset_master.asset_id"), primary_key=True),
    sa.Column("day", sa.Date, primary_key=True),
    sa.Column("intervals", sa.Integer, nullable=False),
    sa.Column("avg_hi", sa.Double()),
    sa.Column("min_hi", sa.Double()),
    sa.Column("max_hi", sa.Double()),
    sa.Column("unhealthy_hours", sa.Double()),
    sa.Column("avg_p_fail", sa.Double()),
    sa.Column("max_p_fail", sa.Double()),
    sa.Column("hi_healthy", sa.Boolean),
    sa.Column("ml_healthy", sa.Boolean),
    sa.Column("agree", sa.Boolean),
    sa.Column("updated_at", DT3, nullable=False),
    sa.Index("ix_daily_day", "day"),
)

# Insert order that satisfies the foreign keys.
INSERT_ORDER = [
    "firmware_release", "asset_master", "asset_location", "hi_reset_log", "fault_events", "failure_log", "co_operations",
    "maintenance_log", "alarm_events", "battery_tests", "comms_events", "telemetry_15min",
    "ar_variables_15min", "ar_health_index_15min", "ar_ml_prediction_15min", "ar_dl_prediction_15min",
    "ar_daily_summary",
]
TABLES = {t.name: t for t in metadata.sorted_tables}
