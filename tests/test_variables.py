from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ar_gen.config import load_config
from ar_gen.variables import RawInputs, compute_variables, to_ms

CFG = load_config(load_dotenv_file=False)
T0 = datetime(2026, 1, 1)
DAY = 24 * 60
ASSET = dict(asset_id="AR-001", battery_rated_ah=26.0, battery_install_date=date(2025, 1, 1),
             battery_service_interval_years=2, manufactured_date=date(2016, 1, 1),
             installed_date=date(2016, 6, 1), last_maintenance_date=date(2025, 12, 1),
             firmware_version="1.20.0", has_rs485_link=False, coastal_flag=True,
             comms_channel="MOBILE", rated_break_ka=16.0, rated_mech_ops=30000)

COLS = {
    "faults": ["fault_id", "asset_id", "start_ts", "end_ts", "fault_type", "max_fault_current_a", "outcome"],
    "ops": ["op_id", "asset_id", "ts", "op_type", "source", "operating_time_ms", "result"],
    "alarms": ["event_id", "asset_id", "ts", "category", "signal_name", "state"],
    "tests": ["test_id", "asset_id", "ts", "trigger", "result", "voltage_before",
              "voltage_min_during_test", "test_load_current_a"],
    "comms": ["event_id", "asset_id", "ts", "channel", "event"],
    "maintenance": ["maint_id", "asset_id", "ts", "details"],
}


def at(minutes):
    return T0 + timedelta(minutes=minutes)


def telemetry(days=3):
    rows = []
    for i in range(1, days * 96 + 1):
        rows.append(dict(asset_id="AR-001", ts=at(15 * i), battery_voltage=13.6, battery_current=0.06,
                         battery_charge_pct=100.0, battery_temp_c=35.0, ambient_temp_c=28.0,
                         lightning_strikes_10km=0, comms_connected=True, signal_strength_dbm=-70.0,
                         weather="SUNNY", co_total=1000 + i // 96, mechanical_wear_pct=3.3,
                         contact_wear_pct_a=10.0, contact_wear_pct_b=11.0, contact_wear_pct_c=12.5))
    return pd.DataFrame(rows)


def inputs(tele=None, asset=None, **events):
    fw = pd.DataFrame([{"version": x["version"], "release_date": x["release_date"]}
                       for x in CFG["firmware_releases"]])
    frames = {k: pd.DataFrame(events.get(k, []), columns=c) for k, c in COLS.items()}
    return RawInputs(assets=pd.DataFrame([asset or ASSET]),
                     telemetry=telemetry() if tele is None else tele,
                     firmware=fw, asset_start={"AR-001": int(to_ms([T0])[0])}, **frames)


def compute(inp, start_min=DAY, end_min=3 * DAY):
    s, e = to_ms([at(start_min), at(end_min)])
    return compute_variables(inp, int(s), int(e), CFG).set_index("ts")


def row(df, minutes):
    return df.loc[pd.Timestamp(at(minutes))]


# ---------------------------------------------------------------- tests
def test_one_row_per_interval_and_snapshots():
    tele = telemetry()
    tele = tele[tele["ts"] != at(2 * DAY)]
    df = compute(inputs(tele))
    assert len(df) == 2 * 96
    gap = row(df, 2 * DAY)
    assert not gap["telemetry_present"]
    assert np.isnan(gap["v01_battery_voltage_v"]) and gap["v42_weather"] is None
    assert gap["v21_co_total"] == 1001                     # lifetime counter carried forward
    ok = row(df, 2 * DAY + 15)
    assert ok["telemetry_present"] and ok["v01_battery_voltage_v"] == 13.6
    assert ok["v11_wear_max_pct"] == 12.5
    assert ok["v40_ambient_temp_c"] == 28.0 and ok["v42_weather"] == "SUNNY" and ok["v41_coastal_flag"] == 1


def test_discharge_episodes_and_depth_of_discharge():
    tele = telemetry()
    for m in range(2 * DAY + 60, 2 * DAY + 121, 15):
        tele.loc[tele["ts"] == at(m), ["battery_current", "battery_charge_pct"]] = (-0.6, 90)
    tele.loc[tele["ts"] == at(2 * DAY + 300), ["battery_current", "battery_charge_pct"]] = (-0.4, 80)
    df = compute(inputs(tele))
    r = row(df, 2 * DAY + 360)
    assert r["v02_discharge_episodes_24h"] == 2
    assert r["v06_max_dod_30d_pct"] == pytest.approx(20)
    assert row(df, 3 * DAY)["v02_discharge_episodes_24h"] == 2


def test_minutes_on_battery_and_ac_failures():
    ev = [dict(event_id=1, asset_id="AR-001", ts=at(2 * DAY + 60), category="WARNING",
               signal_name="AC Off (On Battery Supply)", state="RAISED"),
          dict(event_id=2, asset_id="AR-001", ts=at(2 * DAY + 150), category="WARNING",
               signal_name="AC Off (On Battery Supply)", state="CLEARED")]
    df = compute(inputs(alarms=ev))
    assert row(df, 2 * DAY + 45)["v10_minutes_on_battery_30d"] == 0
    assert row(df, 2 * DAY + 120)["v10_minutes_on_battery_30d"] == pytest.approx(60)
    assert row(df, 3 * DAY)["v10_minutes_on_battery_30d"] == pytest.approx(90)
    assert row(df, 3 * DAY)["v05_ac_failures_30d"] == 1
    assert row(df, 3 * DAY)["v07_battery_alarms_30d"] == 0


def test_fault_counts():
    tele = telemetry()
    tele.loc[tele["ts"] == at(2 * DAY + 15), "lightning_strikes_10km"] = 3
    f = lambda i, m, t, cur, out: dict(fault_id=i, asset_id="AR-001", start_ts=at(m), end_ts=at(m + 1),
                                       fault_type=t, max_fault_current_a=cur, outcome=out)
    faults = [f(1, 2 * DAY + 10, "OC_A", 3000.0, "LOCKOUT"),
              f(2, 2 * DAY + 20, "EF", 500.0, "CLEARED_BY_RECLOSE"),
              f(3, 2 * DAY + 40, "SEF", 30.0, "UPSTREAM_CLEARED")]
    df = compute(inputs(tele, faults=faults))
    before = row(df, 2 * DAY)
    assert before["v19_faults_24h"] == 0 and before["v18_max_fault_current_30d_a"] == 0
    r = row(df, 3 * DAY)
    assert (r["v13_permanent_faults_30d"], r["v14_temporary_faults_30d"]) == (2, 1)
    assert (r["v15_earth_faults_30d"], r["v16_sef_faults_30d"], r["v17_oc_faults_30d"]) == (1, 1, 1)
    assert r["v18_max_fault_current_30d_a"] == 3000.0
    assert r["v19_faults_24h"] == 3 and r["v20_lockouts_30d"] == 1
    assert r["v12_lightning_faults_30d"] == 1
    assert row(df, 2 * DAY + 15)["v19_faults_24h"] == 1


def test_switching_counts():
    o = lambda i, m, typ, src, t, res: dict(op_id=i, asset_id="AR-001", ts=at(m), op_type=typ, source=src,
                                            operating_time_ms=t, result=res)
    ops = [o(1, 2 * DAY + 5, "OPEN", "PROTECTION", 25.0, "SUCCESS"),
           o(2, 2 * DAY + 6, "CLOSE", "AUTO_RECLOSE", 50.0, "SUCCESS"),
           o(3, 2 * DAY + 100, "OPEN", "HMI", None, "FAIL"),
           o(4, 2 * DAY + 200, "CLOSE", "HMI", 120.0, "SUCCESS")]
    r = row(compute(inputs(ops=ops)), 3 * DAY)
    assert (r["v22_open_ops_30d"], r["v23_close_ops_30d"]) == (2, 2)
    assert (r["v24_protection_trips_30d"], r["v25_auto_reclose_ops_30d"]) == (1, 1)
    assert r["v26_switching_ops_24h"] == 4 and r["v27_unsuccessful_ops_30d"] == 2


def test_comms_variables():
    tele = telemetry()
    last_day = tele["ts"] > at(2 * DAY)
    tele.loc[last_day & (tele.index % 2 == 0), "comms_connected"] = False
    tele.loc[last_day, "signal_strength_dbm"] = -90.0
    c = lambda i, m, e: dict(event_id=i, asset_id="AR-001", ts=at(m), channel="MOBILE", event=e)
    ev = [c(1, 2 * DAY + 30, "DISCONNECT"), c(2, 2 * DAY + 35, "RECONNECT"),
          c(3, 2 * DAY + 90, "DISCONNECT"), c(4, 2 * DAY + 110, "MODEM_RESTART"),
          c(5, 2 * DAY + 115, "RECONNECT")]
    r = row(compute(inputs(tele, comms=ev)), 3 * DAY)
    assert r["v28_scada_availability_24h_pct"] == pytest.approx(50)
    assert (r["v29_comm_failures_24h"], r["v32_reconnects_24h"], r["v30_modem_restarts_30d"]) == (2, 2, 1)
    assert r["v31_signal_strength_24h_dbm"] == pytest.approx(-90)
    assert np.isnan(r["v36_rs485_errors_30d"])                 # no RS485 link


def test_soh_estimate_from_battery_tests_and_failed_count():
    t = lambda i, m, res, vb, vm: dict(test_id=i, asset_id="AR-001", ts=at(m), trigger="AUTO", result=res,
                                       voltage_before=vb, voltage_min_during_test=vm, test_load_current_a=10.0)
    tests = [t(1, 2 * DAY, "PASSED", 12.88, 12.53), t(2, 2 * DAY + 720, "FAILED", 12.85, 11.74)]
    df = compute(inputs(tests=tests))
    assert np.isnan(row(df, 2 * DAY - 15)["v03_soh_est_pct"])     # old battery, no evidence yet
    assert row(df, 2 * DAY)["v03_soh_est_pct"] == pytest.approx(100, abs=0.5)
    after = row(df, 2 * DAY + 720)
    assert after["v03_soh_est_pct"] == pytest.approx(51.2, abs=0.5)
    assert after["v09_auto_test_failed_count"] == 1
    assert row(df, 2 * DAY + 705)["v09_auto_test_failed_count"] == 0


def test_soh_estimate_from_a_long_discharge():
    tele = telemetry()
    for k in range(16):
        tele.loc[tele["ts"] == at(2 * DAY + 15 * (k + 1)), ["battery_current", "battery_charge_pct"]] = \
            (-0.6, 100 - 9.2 * (k + 1) / 16)
    df = compute(inputs(tele))
    assert np.isnan(row(df, 2 * DAY + 15 * 16)["v03_soh_est_pct"])   # episode not finished yet
    est = row(df, 2 * DAY + 15 * 17)["v03_soh_est_pct"]
    assert est == pytest.approx(2.325 / 0.092 / 26 * 100, abs=0.1)


def test_battery_replacement_resets_battery_variables():
    asset = dict(ASSET, battery_install_date=date(2026, 1, 3), battery_service_interval_years=4)
    maint = [dict(maint_id=1, asset_id="AR-001", ts=at(2 * DAY + 600),
                  details={"changes": {"battery_install_date": ["2025-01-01", "2026-01-03"],
                                       "battery_service_interval_years": [2, 4],
                                       "last_maintenance_date": ["2025-12-01", "2026-01-03"]}})]
    tests = [dict(test_id=1, asset_id="AR-001", ts=at(2 * DAY), trigger="AUTO", result="FAILED",
                  voltage_before=12.8, voltage_min_during_test=11.5, test_load_current_a=10.0)]
    df = compute(inputs(asset=asset, maintenance=maint, tests=tests))
    old = row(df, 2 * DAY + 585)
    assert old["v04_battery_life_used_pct"] == pytest.approx((365 + 2 + 585 / 1440) / 730.5 * 100, abs=0.01)
    assert old["v09_auto_test_failed_count"] == 1 and old["v03_soh_est_pct"] < 50
    assert old["v39_days_since_maintenance"] == pytest.approx(33 + 585 / 1440, abs=0.01)
    new = row(df, 2 * DAY + 615)
    assert new["v04_battery_life_used_pct"] < 0.1
    assert new["v09_auto_test_failed_count"] == 0 and new["v03_soh_est_pct"] == 100
    assert new["v39_days_since_maintenance"] < 1


def test_firmware_and_ageing():
    r = row(compute(inputs()), 3 * DAY)
    assert r["v35_firmware_versions_behind"] == 2               # 1.21.0 and 1.22.0 are newer
    assert r["v37_sg_age_years"] == pytest.approx((date(2026, 1, 4) - date(2016, 1, 1)).days / 365.25, abs=1e-6)
    assert r["v38_years_in_service"] == pytest.approx((date(2026, 1, 4) - date(2016, 6, 1)).days / 365.25, abs=1e-6)


def test_incremental_equals_full_and_uses_no_future_data():
    tele = telemetry()
    tele.loc[tele.index % 7 == 0, "battery_current"] = -0.5
    tele.loc[tele.index % 11 == 0, "comms_connected"] = False
    ops = [dict(op_id=i, asset_id="AR-001", ts=at(DAY + 37 * i), op_type="OPEN", source="HMI",
                operating_time_ms=70.0 if i % 3 == 0 else 25.0, result="SUCCESS") for i in range(1, 60)]
    alarms = [dict(event_id=i, asset_id="AR-001", ts=at(DAY + 50 * i), category="WARNING",
                   signal_name="AC Off (On Battery Supply)", state="RAISED" if i % 2 else "CLEARED")
              for i in range(1, 50)]
    full = compute(inputs(tele, ops=ops, alarms=alarms))
    cut = at(2 * DAY)
    part1 = compute(inputs(tele[tele["ts"] <= cut], ops=[o for o in ops if o["ts"] <= cut],
                           alarms=[a for a in alarms if a["ts"] <= cut]), DAY, 2 * DAY)
    part2 = compute(inputs(tele, ops=ops, alarms=alarms), 2 * DAY, 3 * DAY)
    pd.testing.assert_frame_equal(full, pd.concat([part1, part2]), check_dtype=False)
