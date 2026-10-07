from datetime import datetime, date

import sqlalchemy as sa
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from ar_gen.db import schema
from ar_gen.db.engine import make_engine, create_all
from ar_gen.db.writer import insert_rows, rollback_after

EXPECTED_TABLES = {
    "asset_master", "telemetry_15min", "co_operations", "fault_events", "alarm_events",
    "battery_tests", "comms_events", "failure_log", "maintenance_log", "firmware_release",
    "ar_variables_15min", "ar_health_index_15min",
}


def _engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 't.sqlite'}")
    create_all(eng)
    return eng


def asset_row(asset_id="AR-001"):
    return dict(asset_id=asset_id, osm_serial_number="0211501000001", model="OSM15-310",
                rated_voltage_kv=15.5, system_voltage_kv=11.0, rated_continuous_a=800,
                rated_break_ka=16.0, rated_fault_break_ops=70, rated_mech_ops=30000,
                manufactured_date=date(2015, 1, 1), installed_date=date(2015, 6, 1),
                last_maintenance_date=date(2025, 6, 1), rc_model="RC-10ES", relay_module="REL-02",
                firmware_version="1.20.0", battery_model="BAT-14", battery_type="AGM",
                battery_rated_ah=26.0, battery_service_interval_years=2,
                battery_install_date=date(2025, 1, 1), comms_channel="MOBILE", has_rs485_link=False,
                coastal_flag=True, distance_to_coast_km=1.5, area_name="A", feeder_name="F1",
                external_load_time_min=120)


def tele(ts, asset_id="AR-001"):
    return dict(ts=ts, asset_id=asset_id, ia=100.0, ib=101.0, ic=99.0, in_residual=2.0,
                uab=11.0, ubc=11.0, uca=11.0, frequency_hz=50.0, switch_state="CLOSED",
                ac_supply_on=True, external_load_on=True, battery_voltage=13.7,
                battery_current=0.06, battery_charge_pct=100, battery_temp_c=30.0,
                cubicle_temp_c=31.0, ambient_temp_c=29.0, humidity_pct=80, weather="SUNNY",
                lightning_strikes_10km=0, comms_connected=True, signal_strength_dbm=-70.0,
                co_total=10, mechanical_wear_pct=0.033, contact_wear_pct_a=1.0,
                contact_wear_pct_b=1.0, contact_wear_pct_c=1.0)


def test_create_all_creates_every_table(tmp_path):
    eng = _engine(tmp_path)
    assert EXPECTED_TABLES <= set(sa.inspect(eng).get_table_names())


def test_create_all_is_idempotent(tmp_path):
    eng = _engine(tmp_path)
    create_all(eng)
    assert EXPECTED_TABLES <= set(sa.inspect(eng).get_table_names())


def test_mysql_ddl_uses_millisecond_datetime():
    ddl = str(CreateTable(schema.telemetry_15min).compile(dialect=mysql.dialect()))
    assert "DATETIME(3)" in ddl
    assert "PRIMARY KEY (asset_id, ts)" in ddl


def test_health_index_table_has_scores_for_all_42_variables():
    cols = set(schema.ar_health_index_15min.c.keys())
    assert {"health_index", "health_class", "is_healthy", "cap_applied", "worst_variables"} <= cols
    assert {f"s{i:02d}" for i in range(1, 43)} <= cols
    assert len(schema.VARIABLE_NAMES) == 43  # telemetry_present + v01..v42


def test_telemetry_duplicates_are_skipped(tmp_path):
    eng = _engine(tmp_path)
    ts = datetime(2026, 1, 1, 0, 15)
    with eng.begin() as conn:
        insert_rows(conn, schema.asset_master, [asset_row()])
        insert_rows(conn, schema.telemetry_15min, [tele(ts)], ignore_duplicates=True)
        insert_rows(conn, schema.telemetry_15min, [tele(ts)], ignore_duplicates=True)
        n = conn.execute(sa.select(sa.func.count()).select_from(schema.telemetry_15min)).scalar()
    assert n == 1


def test_millisecond_timestamps_round_trip(tmp_path):
    eng = _engine(tmp_path)
    ts = datetime(2026, 1, 1, 0, 7, 12, 345000)
    with eng.begin() as conn:
        insert_rows(conn, schema.asset_master, [asset_row()])
        insert_rows(conn, schema.alarm_events, [dict(event_id=1, ts=ts, asset_id="AR-001",
                    category="WARNING", signal_name="AC Off (On Battery Supply)",
                    state="RAISED", detail=None)])
        back = conn.execute(sa.select(schema.alarm_events.c.ts)).scalar()
    assert back == ts


def test_rollback_after_removes_rows_newer_than_state(tmp_path):
    eng = _engine(tmp_path)
    t1, t2 = datetime(2026, 1, 1, 0, 15), datetime(2026, 1, 1, 0, 30)
    with eng.begin() as conn:
        insert_rows(conn, schema.asset_master, [asset_row()])
        insert_rows(conn, schema.telemetry_15min, [tele(t1), tele(t2)])
        insert_rows(conn, schema.alarm_events, [
            dict(event_id=1, ts=t1, asset_id="AR-001", category="WARNING", signal_name="x",
                 state="EVENT", detail=None),
            dict(event_id=2, ts=t2, asset_id="AR-001", category="WARNING", signal_name="x",
                 state="EVENT", detail=None)])
    next_ids = {"op": 1, "fault": 1, "alarm": 2, "test": 1, "comms": 1, "failure": 1, "maint": 1}
    rollback_after(eng, last_ts=t1, next_ids=next_ids)
    with eng.connect() as conn:
        tele_ts = conn.execute(sa.select(schema.telemetry_15min.c.ts)).scalars().all()
        alarms = conn.execute(sa.select(schema.alarm_events.c.event_id)).scalars().all()
    assert tele_ts == [t1]
    assert alarms == [1]
