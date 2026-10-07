"""Recloser map positions: simulated Sri Lanka placement, or real GPS from a CSV."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from .db import schema
from .db.engine import create_all
from .db.reader import df_to_rows, read_df
from .db.writer import insert_rows
from .sim import rng as rngmod

# Simplified Sri Lanka coastline (lat, lon), clockwise from Point Pedro. Denser along the
# LECO coastal belt (Galle - Negombo).
SRI_LANKA_OUTLINE = [
    (9.82, 80.25), (9.27, 80.82), (8.57, 81.23), (8.30, 81.40), (7.92, 81.56), (7.72, 81.70), (7.22, 81.85),
    (6.87, 81.83), (6.40, 81.62), (6.12, 81.12), (6.02, 80.80), (5.94, 80.55), (5.97, 80.43), (6.03, 80.22),
    (6.14, 80.10), (6.24, 80.05), (6.42, 80.00), (6.47, 79.98), (6.58, 79.96), (6.71, 79.90), (6.83, 79.865),
    (6.93, 79.85), (7.00, 79.865), (7.21, 79.84), (7.58, 79.80), (8.23, 79.76), (8.55, 79.92), (8.98, 79.91),
    (9.38, 80.15), (9.66, 80.02),
]
# Coast stretch of each coastal area (part of the outline): reclosers sit inland of it.
COASTLINES = {
    "Western-Coastal": [(6.42, 80.00), (6.58, 79.96), (6.93, 79.85), (7.21, 79.84), (7.58, 79.80)],  # Bentota-Chilaw
    "Southern-Coastal": [(6.14, 80.10), (6.03, 80.22), (5.97, 80.43), (5.94, 80.55), (6.02, 80.80)],  # Hikkaduwa-Tangalle
    "Eastern-Coastal": [(6.87, 81.83), (7.22, 81.85), (7.72, 81.70), (7.92, 81.56), (8.30, 81.40)],  # Pottuvil-Vakarai
}
# Inland areas: anchor town; reclosers fan out along their feeders.
AREA_GEOMETRY = {
    "Western-Coastal": {"anchor": (6.93, 79.86)},       # Colombo
    "Southern-Coastal": {"anchor": (6.05, 80.22)},      # Galle
    "Eastern-Coastal": {"anchor": (7.72, 81.70)},       # Batticaloa
    "Central-Hills": {"anchor": (7.29, 80.63)},         # Kandy
    "NorthCentral-Dry": {"anchor": (8.31, 80.40)},      # Anuradhapura
}
ISLAND_CENTRE = (7.87, 80.77)
DEFAULT_GEOMETRY = {"anchor": ISLAND_CENTRE}
LAT_RANGE, LON_RANGE = (5.9, 9.9), (79.6, 81.95)
KM_PER_DEG = 111.0
LOCATION_STREAM = 5
COAST_MARGIN_KM = 0.3          # keeps reclosers recorded "0.2 km from the coast" on land


def _move(lat: float, lon: float, north_km: float, east_km: float) -> tuple[float, float]:
    lat2 = lat + north_km / KM_PER_DEG
    lon2 = lon + east_km / (KM_PER_DEG * math.cos(math.radians(lat2)))
    return (min(max(lat2, LAT_RANGE[0]), LAT_RANGE[1]), min(max(lon2, LON_RANGE[0]), LON_RANGE[1]))


def _km_vector(a: tuple, b: tuple) -> tuple[float, float]:
    """(north_km, east_km) from point a to point b (local flat approximation)."""
    lat_mid = math.radians((a[0] + b[0]) / 2)
    return (b[0] - a[0]) * KM_PER_DEG, (b[1] - a[1]) * KM_PER_DEG * math.cos(lat_mid)


def _coast_point(line: list, u: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Point at arc-length fraction u of the coastline and the unit inland normal there."""
    segs = [_km_vector(p, q) for p, q in zip(line, line[1:])]
    lengths = [math.hypot(*s) for s in segs]
    target = u * sum(lengths)
    for (p, q), (dn, de), length in zip(zip(line, line[1:]), segs, lengths):
        if target <= length or (p, q) == (line[-2], line[-1]):
            f = min(1.0, target / length) if length else 0.0
            point = (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
            n1 = (-de / length, dn / length)
            to_centre = _km_vector(point, ISLAND_CENTRE)
            inland = n1 if n1[0] * to_centre[0] + n1[1] * to_centre[1] > 0 else (-n1[0], -n1[1])
            return point, inland
        target -= length
    raise ValueError("empty coastline")


def inside_sri_lanka(lat: float, lon: float) -> bool:
    """Point-in-polygon test against the simplified outline."""
    inside, poly = False, SRI_LANKA_OUTLINE
    for i in range(len(poly)):
        (y1, x1), (y2, x2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _ring_segments():
    ring = SRI_LANKA_OUTLINE + SRI_LANKA_OUTLINE[:1]
    segs = []
    for p, q in zip(ring, ring[1:]):
        dn, de = _km_vector(p, q)
        segs.append((p, q, dn, de, math.hypot(dn, de)))
    return segs


_SEGMENTS = _ring_segments()
_RING_KM = sum(s[4] for s in _SEGMENTS)


def _nearest_on_coast(point: tuple) -> float:
    """Arc-length position (km along the outline) of the coast point nearest to ``point``."""
    best, best_s, start = float("inf"), 0.0, 0.0
    for p, q, dn, de, length in _SEGMENTS:
        vn, ve = _km_vector(p, point)
        t = 0.0 if length == 0 else max(0.0, min(1.0, (vn * dn + ve * de) / (length * length)))
        d = math.hypot(vn - t * dn, ve - t * de)
        if d < best:
            best, best_s = d, start + t * length
        start += length
    return best_s


def _coast_at(s_km: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Coast point at arc-length ``s_km`` and the unit normal pointing inland there."""
    s_km %= _RING_KM
    for p, q, dn, de, length in _SEGMENTS:
        if s_km <= length:
            f = s_km / length if length else 0.0
            point = (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)
            n = (-de / length, dn / length)
            probe = _move(point[0], point[1], n[0] * 0.5, n[1] * 0.5)
            return point, (n if inside_sri_lanka(*probe) else (-n[0], -n[1]))
        s_km -= length
    return _coast_at(0.0)


def _anchored_coastal(anchor, spread_km, fno, nf, inland_km, rnd) -> tuple[float, float]:
    """Along the coast near the branch anchor, then inland by the recloser's coast distance."""
    s = _nearest_on_coast(tuple(anchor)) + (fno - (nf + 1) / 2) * spread_km / nf \
        + rnd.uniform(-spread_km / 2, spread_km / 2)
    (lat, lon), (nn, ne) = _coast_at(s)
    lat, lon = _move(lat, lon, nn * inland_km, ne * inland_km)
    for _ in range(60):                      # safety net at sharp corners: step towards the island centre
        if inside_sri_lanka(lat, lon):
            break
        dn, de = _km_vector((lat, lon), ISLAND_CENTRE)
        norm = math.hypot(dn, de) or 1.0
        lat, lon = _move(lat, lon, dn / norm, de / norm)
    return lat, lon


def _feeder_no(name) -> int:
    tail = str(name).rsplit("-F", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 1


def simulated_locations(assets: pd.DataFrame, seed: int, areas: list | None = None) -> pd.DataFrame:
    """Deterministic positions along feeder lines in each recloser's area.

    ``areas`` is ``cfg["areas"]``: an area with an ``anchor`` (LECO branches) is placed inland of
    the coast nearest its anchor (coastal areas) or around it (inland areas); areas without an
    anchor use the built-in legacy geometry.
    """
    n_feeders = assets.groupby("area_name")["feeder_name"].nunique().to_dict()
    area_cfg = {a["name"]: a for a in (areas or []) if a.get("anchor")}
    rows = []
    for r in assets.itertuples(index=False):
        idx = int(str(r.asset_id).split("-")[-1]) - 1
        rnd = rngmod.rng(seed, LOCATION_STREAM, idx)
        geo = AREA_GEOMETRY.get(r.area_name, DEFAULT_GEOMETRY)
        lat0, lon0 = geo["anchor"]
        fno = _feeder_no(r.feeder_name)
        nf = max(1, int(n_feeders.get(r.area_name, 1)))
        d = r.distance_to_coast_km
        inland = (0.0 if d is None or pd.isna(d) else float(d)) + COAST_MARGIN_KM
        coast = COASTLINES.get(r.area_name)
        branch = area_cfg.get(r.area_name)
        if branch is not None and branch.get("coastal", True):
            lat, lon = _anchored_coastal(branch["anchor"], float(branch.get("spread_km", 5.0)), fno, nf, inland, rnd)
            rows.append({"asset_id": r.asset_id, "latitude": round(lat, 6), "longitude": round(lon, 6),
                         "source": "SIMULATED"})
            continue
        if branch is not None:
            lat0, lon0 = branch["anchor"]
            coast = None
        if coast:
            u = min(0.98, max(0.02, 0.5 + (fno - 1.5) * 0.35 + rnd.uniform(-0.3, 0.3)))   # F1 south/west, F2 north/east
            (lat0, lon0), (nn, ne) = _coast_point(coast, u)
            north, east = nn * inland, ne * inland
        else:
            bearing = 2 * math.pi * (fno - 1) / nf + rnd.uniform(-0.25, 0.25)
            dist = rnd.uniform(3.0, 25.0)
            north, east = dist * math.cos(bearing), dist * math.sin(bearing)
        lat, lon = _move(lat0, lon0, north, east)
        rows.append({"asset_id": r.asset_id, "latitude": round(lat, 6), "longitude": round(lon, 6),
                     "source": "SIMULATED"})
    return pd.DataFrame(rows, columns=["asset_id", "latitude", "longitude", "source"])


def ensure_locations(engine: sa.Engine, cfg: dict) -> int:
    """Create simulated positions for reclosers without a location; returns how many were added.

    Also creates any missing tables, so databases made before this feature keep working.
    """
    create_all(engine)
    am, loc = schema.asset_master, schema.asset_location
    with engine.begin() as conn:
        assets = read_df(conn, sa.select(am.c.asset_id, am.c.area_name, am.c.feeder_name,
                                         am.c.coastal_flag, am.c.distance_to_coast_km))
        if assets.empty:
            return 0
        have = set(conn.execute(sa.select(loc.c.asset_id)).scalars())
        new = simulated_locations(assets, cfg["seed"], cfg.get("areas"))
        new = new[~new["asset_id"].isin(have)]
        insert_rows(conn, loc, df_to_rows(new))
    return len(new)


def load_gps_csv(engine: sa.Engine, path: str | Path) -> int:
    """Load real positions (columns asset_id, latitude, longitude); they replace simulated ones."""
    df = pd.read_csv(path)
    missing = {"asset_id", "latitude", "longitude"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV needs columns asset_id, latitude, longitude (missing: {', '.join(sorted(missing))})")
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="raise")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="raise")
    am, loc = schema.asset_master, schema.asset_location
    with engine.begin() as conn:
        known = set(conn.execute(sa.select(am.c.asset_id)).scalars())
        bad = sorted(set(df["asset_id"]) - known)
        if bad:
            raise ValueError(f"unknown asset_id(s) in CSV: {', '.join(bad)}")
        conn.execute(loc.delete().where(loc.c.asset_id.in_(list(df["asset_id"]))))
        rows = df[["asset_id", "latitude", "longitude"]].assign(source="GPS")
        insert_rows(conn, loc, df_to_rows(rows))
    return len(df)
