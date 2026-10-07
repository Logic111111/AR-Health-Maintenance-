import json
from collections import Counter, defaultdict

import pytest

from ar_gen.sim.fleet import create_fleet, asset_master_rows
from ar_gen.sim.state import SimState
from ar_gen.sim.switching import contact_wear_increment
from ar_gen.sim.timeutil import dt_to_ms, ms_to_dt
from tests.sim_helpers import BUSY, START, make_cfg, run


@pytest.fixture(scope="module")
def busy():
    cfg = make_cfg(4, BUSY)
    state0 = create_fleet(cfg, dt_to_ms(START))
    initial = {a.asset_id: (a.co_total, list(a.wear)) for a in state0.assets}
    rows, state = run(cfg, days=10)
    return cfg, rows, state, initial


# ---------------- fleet ----------------
def test_fleet_composition():
    cfg = make_cfg(30)
    st = create_fleet(cfg, dt_to_ms(START))
    assert [a.asset_id for a in st.assets][:2] == ["AR-001", "AR-002"]
    assert sum(a.coastal for a in st.assets) == round(cfg["fleet"]["coastal_fraction"] * 30)
    profiles = Counter(p for a in st.assets for p in a.profiles)
    for p in ("BATTERY_DEGRADING", "MECHANISM_WEAR", "COMMS_PROBLEM", "CONTROLLER_PROBLEM",
              "HARSH_ENVIRONMENT", "HEALTHY"):
        assert profiles[p] >= 1
    rows = asset_master_rows(st)
    assert {r["model"] for r in rows} == {"OSM15-310", "OSM27-310"}
    for a in st.assets:
        assert a.co_total >= 0 and all(0 <= w < 100 for w in a.wear)


# ---------------- telemetry ----------------
def test_telemetry_one_row_per_asset_interval_and_physical(busy):
    cfg, rows, state, _ = busy
    tele = rows["telemetry_15min"]
    keys = Counter((r["asset_id"], r["ts"]) for r in tele)
    assert max(keys.values()) == 1
    assert len(tele) <= 4 * 10 * 96
    assert len(tele) > 0.9 * 4 * 10 * 96          # gaps only while the controller is down
    for r in tele:
        assert 9.6 <= r["battery_voltage"] <= 14.8
        assert 0 <= r["battery_charge_pct"] <= 100
        assert r["switch_state"] in ("CLOSED", "OPEN", "LOCKOUT")
        assert 0 <= r["ia"] <= 800 * 1.3
        assert r["weather"] in ("SUNNY", "CLOUDY", "RAINY", "STORM")
        if not r["ac_supply_on"]:
            assert r["battery_current"] < 0
        if r["switch_state"] != "CLOSED":
            assert r["ia"] < 800


# ---------------- protection sequences ----------------
def test_every_fault_has_a_consistent_operation_sequence(busy):
    _, rows, _, _ = busy
    faults = rows["fault_events"]
    assert len(faults) > 20
    ops_by_fault = defaultdict(list)
    for op in rows["co_operations"]:
        if op["fault_id"] is not None:
            ops_by_fault[op["fault_id"]].append(op)
    outcomes = Counter(f["outcome"] for f in faults)
    assert outcomes["CLEARED_BY_RECLOSE"] > 0 and outcomes["LOCKOUT"] > 0
    for f in faults:
        ops = sorted(ops_by_fault[f["fault_id"]], key=lambda o: o["ts"])
        assert ops, f
        trips = [o for o in ops if o["op_type"] == "OPEN"]
        closes = [o for o in ops if o["op_type"] == "CLOSE"]
        assert len(closes) == f["reclose_attempts"]
        assert all(o["source"] == "PROTECTION" for o in trips)
        assert all(o["source"] == "AUTO_RECLOSE" for o in closes)
        assert f["start_ts"] <= ops[0]["ts"] <= ops[-1]["ts"] <= f["end_ts"]
        if f["outcome"] == "CLEARED_BY_RECLOSE":
            assert ops[-1]["op_type"] == "CLOSE" and ops[-1]["result"] == "SUCCESS"
            assert len(trips) == len(closes)
        elif f["outcome"] == "LOCKOUT":
            last = ops[-1]
            assert (last["op_type"] == "OPEN" and last["ar_state"] == "OPEN1_LOCKOUT") or \
                   (last["op_type"] == "CLOSE" and last["result"] == "FAIL")
        else:
            assert trips[0]["result"] == "FAIL"


# ---------------- counters ----------------
def test_lifetime_counters_match_logged_operations(busy):
    cfg, rows, state, initial = busy
    models = {a.asset_id: a for a in state.assets}
    resets = {(m["asset_id"]) for m in rows["maintenance_log"]
              if (m["details"] or {}).get("counters_reset")}
    opens = defaultdict(list)
    for op in rows["co_operations"]:
        if op["op_type"] == "OPEN" and op["result"] == "SUCCESS":
            opens[op["asset_id"]].append(op)
    for aid, (co0, wear0) in initial.items():
        if aid in resets:
            continue
        a = models[aid]
        ops = sorted(opens[aid], key=lambda o: o["ts"])
        snaps = sorted((r for r in rows["telemetry_15min"] if r["asset_id"] == aid), key=lambda r: r["ts"])
        last = snaps[-1]
        done = [o for o in ops if o["ts"] <= last["ts"]]
        assert last["co_total"] == co0 + len(done)
        assert last["mechanical_wear_pct"] == pytest.approx(last["co_total"] / 300.0, abs=1e-3)
        expect_a = wear0[0] + sum(contact_wear_increment(o["trip_ia"] or 0, a.rated_break_ka,
                                                         a.rated_fault_ops) for o in done)
        assert last["contact_wear_pct_a"] == pytest.approx(expect_a, abs=0.01)


# ---------------- battery / supply ----------------
def test_battery_discharges_on_ac_loss_and_floats_on_ac(busy):
    _, rows, _, _ = busy
    off = [r for r in rows["telemetry_15min"] if not r["ac_supply_on"]]
    on_full = [r for r in rows["telemetry_15min"] if r["ac_supply_on"] and r["battery_charge_pct"] >= 99]
    assert off and on_full
    assert all(r["battery_current"] < 0 for r in off)
    assert all(13.2 <= r["battery_voltage"] <= 14.2 for r in on_full)
    raised = [a for a in rows["alarm_events"]
              if a["signal_name"] == "AC Off (On Battery Supply)" and a["state"] == "RAISED"]
    assert raised


def test_alarm_clear_always_follows_a_raise(busy):
    _, rows, _, _ = busy
    active = set()
    for a in sorted(rows["alarm_events"], key=lambda e: (e["ts"], e["event_id"])):
        key = (a["asset_id"], a["signal_name"])
        if a["state"] == "RAISED":
            assert key not in active, a
            active.add(key)
        elif a["state"] == "CLEARED":
            assert key in active, a
            active.discard(key)


def test_battery_tests_are_logged(busy):
    _, rows, _, _ = busy
    tests = rows["battery_tests"]
    assert tests
    for t in tests:
        if t["result"] == "NOT_PERFORMED":
            assert t["not_performed_reason"] in ("AC_OFF", "LOW_VOLTAGE", "DISCHARGING", "RESTING")
        else:
            assert t["voltage_min_during_test"] < t["voltage_before"]


# ---------------- failures and repairs ----------------
def test_degrading_battery_fails_and_is_replaced():
    cfg = make_cfg(2, {
        "fleet": {"profiles": {"weights": {"BATTERY_DEGRADING": 1.0, "HEALTHY": 0.0,
                                           "MECHANISM_WEAR": 0.0, "COMMS_PROBLEM": 0.0,
                                           "CONTROLLER_PROBLEM": 0.0, "HARSH_ENVIRONMENT": 0.0},
                               "secondary_probability": 0.0, "onset_window_days": [0, 1]}},
        "battery": {"degrading_extra_fade_per_day": [0.02, 0.02],
                    "test": {"interval_days": 5}},
    })
    rows, state = run(cfg, days=60)
    failures = [f for f in rows["failure_log"] if f["component"] == "BATTERY"]
    assert failures
    by_id = {f["failure_id"]: f for f in failures}
    fixes = [m for m in rows["maintenance_log"]
             if m["component"] == "BATTERY" and m["maint_type"] == "CORRECTIVE"]
    assert fixes
    for fix in fixes:
        failure = by_id[fix["failure_id"]]
        assert failure["asset_id"] == fix["asset_id"]
        assert fix["ts"] > failure["ts"]
        assert "battery_install_date" in fix["details"]["changes"]


# ---------------- determinism ----------------
def test_same_seed_same_data():
    cfg = make_cfg(2, BUSY)
    r1, _ = run(cfg, days=2)
    r2, _ = run(cfg, days=2)
    assert r1 == r2


def test_split_run_equals_single_run():
    cfg = make_cfg(2, BUSY)
    single, _ = run(cfg, days=3)
    first, mid_state = run(cfg, days=1)
    restored = SimState.from_dict(json.loads(json.dumps(mid_state.to_dict())))
    second, _ = run(cfg, days=2, start=ms_to_dt(restored.last_ms), state=restored, chunk_intervals=7)
    combined = defaultdict(list)
    for part in (first, second):
        for k, v in part.items():
            combined[k].extend(v)
    for table in single:
        key = lambda r: tuple(str(v) for v in r.values())
        assert sorted(single[table], key=key) == sorted(combined[table], key=key), table
