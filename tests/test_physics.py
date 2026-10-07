import math

import pytest

from ar_gen.sim import rng as rngmod
from ar_gen.sim.switching import ops_at_current, contact_wear_increment
from ar_gen.sim.battery import ocv, battery_rint, terminal_voltage
from ar_gen.sim.protection import iec_standard_inverse


# ---------------- deterministic random streams ----------------
def test_same_keys_give_same_sequence():
    a = rngmod.rng(1, 2, 3)
    b = rngmod.rng(1, 2, 3)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_different_keys_give_different_sequences():
    a = rngmod.rng(1, 2, 3).random()
    b = rngmod.rng(1, 2, 4).random()
    c = rngmod.rng(1, 3, 3).random()
    assert len({a, b, c}) == 3


def test_derive_seed_is_64_bit():
    s = rngmod.derive_seed(20261005, 7, 123456)
    assert 0 <= s < 2 ** 64


# ---------------- contact wear curve (manual 2.1.4) ----------------
def test_ops_flat_below_800a():
    assert ops_at_current(100, 16.0, 70) == pytest.approx(30000)
    assert ops_at_current(800, 16.0, 70) == pytest.approx(30000)


def test_ops_at_rated_break_current():
    assert ops_at_current(16000, 16.0, 70) == pytest.approx(70)
    assert ops_at_current(12500, 12.5, 140) == pytest.approx(140)


def test_ops_log_log_interpolation_midpoint():
    i_mid = math.sqrt(800 * 16000)
    n_mid = math.sqrt(30000 * 70)
    assert ops_at_current(i_mid, 16.0, 70) == pytest.approx(n_mid, rel=1e-6)


def test_ops_clamped_above_rating():
    assert ops_at_current(20000, 16.0, 70) == pytest.approx(70)


def test_contact_wear_increment_percent():
    assert contact_wear_increment(500, 16.0, 70) == pytest.approx(100 / 30000)
    assert contact_wear_increment(0, 16.0, 70) == 0.0


# ---------------- battery electrical model ----------------
def test_ocv_table_endpoints_and_monotonic():
    assert ocv(1.0) == pytest.approx(12.73)
    assert ocv(0.0) == pytest.approx(10.5)
    values = [ocv(x / 20) for x in range(21)]
    assert values == sorted(values)


def test_internal_resistance_grows_as_soh_falls():
    assert battery_rint(1.0, 0.012, 2.5) == pytest.approx(0.012)
    assert battery_rint(0.6, 0.012, 2.5) > 3 * 0.012


def test_discharge_voltage_below_ocv():
    v = terminal_voltage(soc=0.5, current_a=-0.7, rint=0.05, mode="DISCHARGE", temp_c=25)
    assert v == pytest.approx(ocv(0.5) - 0.035)


def test_float_voltage_is_temperature_compensated():
    v25 = terminal_voltage(1.0, 0.06, 0.012, "FLOAT", 25)
    v35 = terminal_voltage(1.0, 0.06, 0.012, "FLOAT", 35)
    assert 13.6 <= v25 <= 13.8
    assert v35 < v25


# ---------------- protection timing ----------------
def test_iec_standard_inverse_formula():
    t = iec_standard_inverse(3000, 300, 0.5)
    assert t == pytest.approx(0.5 * 0.14 / ((3000 / 300) ** 0.02 - 1), rel=1e-9)


def test_iec_no_trip_below_pickup():
    assert iec_standard_inverse(250, 300, 0.5) == math.inf
