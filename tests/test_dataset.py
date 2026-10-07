import numpy as np

from ar_gen.dataset import failure_labels

DAY = 86_400_000


def test_failure_labels_and_censoring():
    ts = np.arange(0, 41) * DAY
    lab = failure_labels(ts, np.array([10 * DAY]), np.array(["BATTERY"], dtype=object), 40 * DAY)
    assert lab["days_to_next_failure"][0] == 10 and lab["next_failure_component"][0] == "BATTERY"
    assert not lab["failure_next_7d"][0] and lab["failure_next_30d"][0]
    assert lab["failure_next_7d"][5] and lab["failure_next_7d"][9]
    assert not lab["failure_next_30d"][10] and np.isnan(lab["days_to_next_failure"][10])   # strictly after
    assert not lab["censored_30d"][10] and lab["censored_30d"][15]                          # 25 days left
    assert not lab["censored_30d"][5]                                                        # failure seen
    assert lab["censored_7d"][34] and not lab["censored_7d"][33]


def test_no_failures():
    ts = np.arange(0, 5) * DAY
    lab = failure_labels(ts, np.array([], dtype=np.int64), np.array([], dtype=object), 100 * DAY)
    assert not lab["failure_next_30d"].any() and np.isnan(lab["days_to_next_failure"]).all()
