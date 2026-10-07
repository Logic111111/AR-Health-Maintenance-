import math

import pandas as pd
import pytest
import sqlalchemy as sa

from ar_gen.db import schema
from ar_gen.db.engine import create_all, make_engine
from ar_gen.db.writer import insert_rows
from ar_gen.locations import ensure_locations, load_gps_csv, simulated_locations
from ar_gen.sim.fleet import asset_master_rows, create_fleet
from ar_gen.sim.timeutil import dt_to_ms
from tests.sim_helpers import START, make_cfg

LAT_BOX, LON_BOX = (5.9, 9.9), (79.6, 81.95)


@pytest.fixture(scope="module")
def fleet():
    cfg = make_cfg(30)
    rows = asset_master_rows(create_fleet(cfg, dt_to_ms(START)))
    return cfg, rows, pd.DataFrame(rows)


def _km(a, b):
    dlat = (a[0] - b[0]) * 111.0
    dlon = (a[1] - b[1]) * 111.0 * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dlat, dlon)


def test_simulated_locations_are_deterministic_and_inside_sri_lanka(fleet):
    cfg, _, assets = fleet
    a = simulated_locations(assets, cfg["seed"], cfg["areas"])
    b = simulated_locations(assets, cfg["seed"], cfg["areas"])
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30 and set(a["source"]) == {"SIMULATED"}
    assert a["latitude"].between(*LAT_BOX).all() and a["longitude"].between(*LON_BOX).all()


def _dist_to_line_km(p, line):
    best = float("inf")
    for a, b in zip(line, line[1:]):
        ax, ay = 0.0, 0.0
        bx, by = (b[1] - a[1]) * 111.0 * math.cos(math.radians(a[0])), (b[0] - a[0]) * 111.0
        px, py = (p[1] - a[1]) * 111.0 * math.cos(math.radians(a[0])), (p[0] - a[0]) * 111.0
        t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
        best = min(best, math.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay))))
    return best


def _dist_to_outline_km(p):
    from ar_gen.locations import SRI_LANKA_OUTLINE
    ring = SRI_LANKA_OUTLINE + SRI_LANKA_OUTLINE[:1]
    return _dist_to_line_km(p, ring)


def test_coastal_points_are_near_their_coast(fleet):
    cfg, _, assets = fleet
    loc = simulated_locations(assets, cfg["seed"], cfg["areas"]).merge(assets, on="asset_id")
    coastal = loc[loc["coastal_flag"].astype(bool)]
    assert len(coastal) >= 5
    for r in coastal.itertuples():
        assert _dist_to_outline_km((r.latitude, r.longitude)) <= r.distance_to_coast_km + 3.0, r.asset_id


def test_branch_clusters_near_their_anchor():
    cfg = make_cfg(100)
    assets = pd.DataFrame(asset_master_rows(create_fleet(cfg, dt_to_ms(START))))
    loc = simulated_locations(assets, cfg["seed"], cfg["areas"]).merge(assets, on="asset_id")
    anchors = {a["name"]: a["anchor"] for a in cfg["areas"]}
    assert len(anchors) == 7 and set(loc["area_name"]) == set(anchors)
    for name, g in loc.groupby("area_name"):
        centre = (g["latitude"].mean(), g["longitude"].mean())
        assert _km(centre, tuple(anchors[name])) <= 20.0, name


def test_fleet_distances_follow_the_config(fleet):
    cfg, _, assets = fleet
    lo_c, hi_c = cfg["fleet"]["coastal_distance_km"]
    lo_i, hi_i = cfg["fleet"]["inland_distance_km"]
    for r in assets.itertuples():
        lo, hi = (lo_c, hi_c) if r.coastal_flag else (lo_i, hi_i)
        assert lo - 0.05 <= r.distance_to_coast_km <= hi + 0.05, r.asset_id


def test_feeders_are_spatially_grouped(fleet):
    cfg, _, assets = fleet
    loc = simulated_locations(assets, cfg["seed"], cfg["areas"]).merge(assets, on="asset_id")
    pts = {r.asset_id: (r.latitude, r.longitude) for r in loc.itertuples()}
    same, other = [], []
    for i, a in loc.iterrows():
        for j, b in loc.iterrows():
            if i >= j:
                continue
            d = _km(pts[a.asset_id], pts[b.asset_id])
            if a.feeder_name == b.feeder_name:
                same.append(d)
            elif a.area_name != b.area_name:
                other.append(d)
    assert sum(same) / len(same) < sum(other) / len(other)


def _db_with_assets(tmp_path, rows):
    eng = make_engine(f"sqlite:///{(tmp_path / 'loc.sqlite').as_posix()}")
    create_all(eng)
    with eng.begin() as conn:
        insert_rows(conn, schema.asset_master, rows)
    return eng


def test_ensure_locations_fills_missing_and_keeps_gps(fleet, tmp_path):
    cfg, rows, _ = fleet
    eng = _db_with_assets(tmp_path, rows)
    assert ensure_locations(eng, cfg) == 30
    csv = tmp_path / "gps.csv"
    csv.write_text("asset_id,latitude,longitude\nAR-001,7.0,80.0\n", encoding="utf-8")
    assert load_gps_csv(eng, csv) == 1
    assert ensure_locations(eng, cfg) == 0
    t = schema.asset_location
    with eng.connect() as conn:
        row = conn.execute(sa.select(t).where(t.c.asset_id == "AR-001")).mappings().one()
        n = conn.execute(sa.select(sa.func.count()).select_from(t)).scalar()
    assert (row["latitude"], row["longitude"], row["source"]) == (7.0, 80.0, "GPS") and n == 30


def test_gps_csv_rejects_unknown_asset(fleet, tmp_path):
    _, rows, _ = fleet
    eng = _db_with_assets(tmp_path, rows)
    csv = tmp_path / "bad.csv"
    csv.write_text("asset_id,latitude,longitude\nAR-999,7.0,80.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AR-999"):
        load_gps_csv(eng, csv)


def _inside(lat, lon, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        (y1, x1), (y2, x2) = poly[i], poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def test_simulated_points_are_on_land():
    from ar_gen.locations import SRI_LANKA_OUTLINE
    cfg = make_cfg(100)
    assets = pd.DataFrame(asset_master_rows(create_fleet(cfg, dt_to_ms(START))))
    loc = simulated_locations(assets, cfg["seed"], cfg["areas"])
    at_sea = [r.asset_id for r in loc.itertuples() if not _inside(r.latitude, r.longitude, SRI_LANKA_OUTLINE)]
    assert not at_sea, at_sea


def test_dashboard_listens_on_this_pc_only_unless_lan(monkeypatch):
    from ar_gen import cli
    calls = []
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert cli.main(["dashboard", "--no-browser"]) == 0
    assert cli.main(["dashboard", "--no-browser", "--lan"]) == 0
    first, second = calls
    assert first[first.index("--server.address") + 1] == "localhost"
    assert second[second.index("--server.address") + 1] == "0.0.0.0"
