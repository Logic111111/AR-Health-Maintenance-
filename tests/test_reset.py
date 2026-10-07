"""Health-Index reset after maintenance: windows of the repaired part start at the reset."""
import pandas as pd

from tests.test_variables import DAY, at, compute, inputs, row, telemetry

R = 2 * DAY + 600                     # reset time (minutes after T0), on the 15-min grid


def op(i, m, typ="OPEN", result="FAIL", op_time=None):
    return dict(op_id=i, asset_id="AR-001", ts=at(m), op_type=typ, source="HMI",
                operating_time_ms=op_time, result=result)


def alarm(i, m, signal, category="WARNING", state="RAISED"):
    return dict(event_id=i, asset_id="AR-001", ts=at(m), category=category, signal_name=signal, state=state)


def resets(*items):
    return pd.DataFrame([dict(asset_id="AR-001", ts=at(m), part_groups=g) for m, g in items],
                        columns=["asset_id", "ts", "part_groups"])


def maint(m, component, maint_type):
    return pd.DataFrame([dict(maint_id=1, asset_id="AR-001", ts=at(m), component=component,
                              maint_type=maint_type, failure_id=None, details=None)])


OPS = [op(1, R - 300), op(2, R - 200), op(3, R - 100), op(4, R + 100)]


def test_switching_reset_drops_operations_before_reset():
    inp = inputs(ops=OPS)
    inp.resets = resets((R, "SWITCHING"))
    r = row(compute(inp), 3 * DAY)
    assert (r["v27_unsuccessful_ops_30d"], r["v22_open_ops_30d"]) == (1, 1)
    plain = row(compute(inputs(ops=OPS)), 3 * DAY)
    assert (plain["v27_unsuccessful_ops_30d"], plain["v22_open_ops_30d"]) == (4, 4)


def test_corrective_maintenance_resets_its_group_only():
    inp = inputs(ops=OPS, alarms=[alarm(1, R - 50, "Check Battery")])
    inp.maintenance = maint(R, "MECHANISM", "CORRECTIVE")
    r = row(compute(inp), 3 * DAY)
    assert r["v27_unsuccessful_ops_30d"] == 1
    assert r["v07_battery_alarms_30d"] == 1            # battery history is not touched


def test_preventive_maintenance_does_not_reset():
    inp = inputs(ops=OPS)
    inp.maintenance = maint(R, "MECHANISM", "PREVENTIVE")
    assert row(compute(inp), 3 * DAY)["v27_unsuccessful_ops_30d"] == 4


def test_reset_all_and_v39():
    comms = [dict(event_id=1, asset_id="AR-001", ts=at(R - 60), channel="MOBILE", event="DISCONNECT")]
    alarms = [alarm(1, R - 50, "Check Battery"), alarm(2, R - 40, "SIM Module Fault", "MALFUNCTION")]
    plain = row(compute(inputs(ops=OPS, alarms=alarms, comms=comms)), R + 60)
    assert min(plain["v07_battery_alarms_30d"], plain["v27_unsuccessful_ops_30d"],
               plain["v29_comm_failures_24h"], plain["v33_module_faults_30d"]) >= 1
    inp = inputs(ops=OPS, alarms=alarms, comms=comms)
    inp.resets = resets((R, "ALL"))
    df = compute(inp)
    r = row(df, R + 60)
    assert (r["v07_battery_alarms_30d"], r["v27_unsuccessful_ops_30d"],
            r["v29_comm_failures_24h"], r["v33_module_faults_30d"]) == (0, 0, 0, 0)
    assert row(df, R + 15)["v39_days_since_maintenance"] < 0.02


def test_event_exactly_at_reset_is_excluded():
    inp = inputs(ops=[op(1, R)])
    inp.resets = resets((R, "SWITCHING"))
    assert row(compute(inp), R + 15)["v27_unsuccessful_ops_30d"] == 0
    assert row(compute(inputs(ops=[op(1, R)])), R + 15)["v27_unsuccessful_ops_30d"] == 1


def test_latest_of_two_resets_wins():
    inp = inputs(ops=[op(1, R - 400), op(2, R - 100)])
    inp.resets = resets((R - 300, "SWITCHING"), (R, "SWITCHING"))
    df = compute(inp)
    assert row(df, R - 60)["v27_unsuccessful_ops_30d"] == 1      # only the op after the first reset
    assert row(df, R + 15)["v27_unsuccessful_ops_30d"] == 0


def test_comms_reset_restarts_availability():
    tele = telemetry()
    tele.loc[(tele["ts"] > at(R - DAY)) & (tele["ts"] <= at(R)), "comms_connected"] = False
    inp = inputs(tele)
    inp.resets = resets((R, "COMMS"))
    assert row(compute(inp), R + 120)["v28_scada_availability_24h_pct"] == 100
    assert row(compute(inputs(tele)), R + 120)["v28_scada_availability_24h_pct"] < 100


def test_v28_at_the_reset_instant_is_missing_not_zero():
    inp = inputs()
    inp.resets = resets((R, "COMMS"))
    df = compute(inp)
    assert pd.isna(row(df, R)["v28_scada_availability_24h_pct"])        # no poll after the reset yet
    assert row(df, R + 15)["v28_scada_availability_24h_pct"] == 100


def test_v28_counts_whole_polls_after_a_mid_interval_reset():
    tele = telemetry()
    tele.loc[tele["ts"] == at(R + 30), "comms_connected"] = False
    inp = inputs(tele)
    inp.resets = resets((R + 5, "COMMS"))                                 # polls after it: R+15 (up), R+30 (down)
    assert row(compute(inp), R + 30)["v28_scada_availability_24h_pct"] == 50


def test_mid_interval_reset_is_exact():
    inp = inputs(ops=[op(1, R + 2), op(2, R + 10)])
    inp.resets = resets((R + 5, "SWITCHING"))
    assert row(compute(inp), R + 15)["v27_unsuccessful_ops_30d"] == 1


def test_battery_reset_restarts_dod_and_episodes():
    tele = telemetry()
    before = (tele["ts"] >= at(R - 480)) & (tele["ts"] <= at(R))
    tele.loc[before, ["battery_charge_pct", "battery_current"]] = (70.0, -0.5)
    plain = row(compute(inputs(tele)), 3 * DAY)
    assert (plain["v06_max_dod_30d_pct"], plain["v02_discharge_episodes_24h"]) == (30, 1)
    inp = inputs(tele)
    inp.resets = resets((R, "BATTERY"))
    r = row(compute(inp), 3 * DAY)
    assert (r["v06_max_dod_30d_pct"], r["v02_discharge_episodes_24h"]) == (0, 0)


def test_incremental_equals_full_with_resets():
    tele = telemetry()
    tele.loc[tele.index % 7 == 0, "battery_current"] = -0.5
    tele.loc[tele.index % 11 == 0, "comms_connected"] = False
    ops = [op(i, DAY + 37 * i, result="FAIL" if i % 3 else "SUCCESS", op_time=25.0) for i in range(1, 60)]
    alarms = [alarm(i, DAY + 50 * i, "Check Battery") for i in range(1, 50)]
    rs = resets((DAY + 500, "SWITCHING"), (2 * DAY + 300, "ALL"))
    full_inp = inputs(tele, ops=ops, alarms=alarms)
    full_inp.resets = rs
    full = compute(full_inp)
    cut = at(2 * DAY)
    p1 = inputs(tele[tele["ts"] <= cut], ops=[o for o in ops if o["ts"] <= cut],
                alarms=[a for a in alarms if a["ts"] <= cut])
    p1.resets = rs[rs["ts"] <= cut]
    p2 = inputs(tele, ops=ops, alarms=alarms)
    p2.resets = rs
    pd.testing.assert_frame_equal(full, pd.concat([compute(p1, DAY, 2 * DAY), compute(p2, 2 * DAY, 3 * DAY)]),
                                  check_dtype=False)
