import math

import numpy as np
import pandas as pd
import pytest

from ar_gen.config import load_config
from ar_gen.db.schema import V_COLUMNS, V_SHORT
from ar_gen.health_index import sub_score, compute_health_index, classify

HI_CFG = load_config(load_dotenv_file=False)["health_index"]
ASSET = {"asset_id": "AR-001", "rated_break_ka": 16.0, "rated_mech_ops": 30000,
         "comms_channel": "MOBILE"}

# Values that give a sub-score of 100 for every variable.
PERFECT = {
    "v01": 13.7, "v02": 0, "v03": 98, "v04": 10, "v05": 0, "v06": 0, "v07": 0, "v08": 0.5,
    "v09": 0, "v10": 0, "v11": 5, "v12": 0, "v13": 0, "v14": 0, "v15": 0, "v16": 0, "v17": 0,
    "v18": 0, "v19": 0, "v20": 0, "v21": 100, "v22": 2, "v23": 2, "v24": 0, "v25": 0, "v26": 0,
    "v27": 0, "v28": 100, "v29": 0, "v30": 0, "v31": -60, "v32": 0, "v33": 0, "v34": 0, "v35": 0,
    "v36": 0, "v37": 2, "v38": 2, "v39": 30, "v40": 28, "v41": 0, "v42": "SUNNY",
}


def frame(**changes):
    row = {V_SHORT[k]: v for k, v in PERFECT.items()}
    for k, v in changes.items():
        row[V_SHORT[k]] = v
    row.update(asset_id="AR-001", ts=pd.Timestamp("2026-01-01 00:15"), telemetry_present=True)
    return pd.DataFrame([row])


def hi_of(**changes):
    return compute_health_index(frame(**changes), pd.DataFrame([ASSET]), HI_CFG).iloc[0]


# ---------------- sub-scores ----------------
def test_linear_score_lower_is_better():
    thr = HI_CFG["thresholds"]["v02"]          # good 0 -> bad 4
    assert sub_score("v02", 0, thr, ASSET) == 100
    assert sub_score("v02", 2, thr, ASSET) == pytest.approx(50)
    assert sub_score("v02", 10, thr, ASSET) == 0


def test_linear_score_higher_is_better():
    thr = HI_CFG["thresholds"]["v03"]          # good 90 -> bad 50
    assert sub_score("v03", 95, thr, ASSET) == 100
    assert sub_score("v03", 70, thr, ASSET) == pytest.approx(50)


def test_two_sided_band_for_battery_voltage():
    thr = HI_CFG["thresholds"]["v01"]          # band 12.6-14.4, bad 11.6 / 14.8
    assert sub_score("v01", 13.7, thr, ASSET) == 100
    assert sub_score("v01", 12.1, thr, ASSET) == pytest.approx(50)
    assert sub_score("v01", 14.6, thr, ASSET) == pytest.approx(50)
    assert sub_score("v01", 11.0, thr, ASSET) == 0


def test_mapped_scores():
    assert sub_score("v42", "STORM", HI_CFG["thresholds"]["v42"], ASSET) == 20
    assert sub_score("v41", 1, HI_CFG["thresholds"]["v41"], ASSET) == 40


def test_fault_current_relative_to_rated_break():
    thr = HI_CFG["thresholds"]["v18"]          # 20 % -> 80 % of 16 kA
    assert sub_score("v18", 8000, thr, ASSET) == pytest.approx(50)


def test_signal_threshold_depends_on_channel():
    thr = HI_CFG["thresholds"]["v31"]
    assert sub_score("v31", -90, thr, ASSET) == pytest.approx(50)
    assert sub_score("v31", -22.5, thr, dict(ASSET, comms_channel="FIBRE")) == pytest.approx(50)


def test_missing_value_gives_nan():
    assert math.isnan(sub_score("v02", np.nan, HI_CFG["thresholds"]["v02"], ASSET))


# ---------------- index ----------------
def test_perfect_asset_scores_100_and_is_healthy():
    r = hi_of()
    assert r["health_index"] == pytest.approx(100)
    assert r["health_class"] == "GOOD" and bool(r["is_healthy"])


def test_index_is_weighted_sum_of_sub_scores():
    # v11 weight 6 -> score 50 removes 3 points; v24 weight 4 -> score 0 removes 4 points
    r = hi_of(v11=60, v24=25)
    assert r["s11"] == pytest.approx(50) and r["s24"] == 0
    assert r["health_index"] == pytest.approx(100 - 3 - 4)


def test_missing_variable_weight_is_redistributed_within_factor():
    r = hi_of(v36=np.nan)
    assert r["health_index"] == pytest.approx(100)
    assert np.isnan(r["s36"])


def test_critical_variable_caps_the_index():
    r = hi_of(v27=3)
    assert r["health_index"] == pytest.approx(45)
    assert r["cap_applied"] == "v27"
    assert r["health_class"] == "POOR" and not bool(r["is_healthy"])


def test_worst_variables_lists_largest_weighted_deficits():
    r = hi_of(v11=60, v24=25, v33=1)
    assert r["worst_variables"].split(",")[0] in {"v24", "v33"}
    assert set(r["worst_variables"].split(",")) == {"v11", "v24", "v33"}


def test_factor_scores_are_reported():
    r = hi_of(v28=89.5)      # availability halfway -> comms factor drops
    assert r["f_comms"] == pytest.approx((3 * 50 + 7 * 100) / 10)
    assert r["f_battery"] == pytest.approx(100)


@pytest.mark.parametrize("hi,cls", [(85, "GOOD"), (80, "GOOD"), (70, "FAIR"), (50, "POOR"), (30, "CRITICAL")])
def test_classification_limits(hi, cls):
    assert classify(hi, HI_CFG) == cls


def test_all_42_variables_have_weight_and_threshold():
    for col in V_COLUMNS:
        short = col[:3]
        assert short in HI_CFG["weights"] and short in HI_CFG["thresholds"]
