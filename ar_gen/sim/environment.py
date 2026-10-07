"""Area environment: weather, temperature, humidity, lightning and upstream grid outages."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from . import rng as rngmod
from .state import AreaState
from .timeutil import HOUR_MS, INTERVALS_PER_YEAR, MIN_MS


@dataclass
class AreaCond:
    weather: str
    ambient_c: float
    humidity_pct: int
    lightning: int
    sun: float
    month: int
    local_hour: float
    weekday: int
    outage_start: int | None
    outage_until: int | None


def _diurnal_shape(h: float) -> float:
    """-1 at 06:00 (minimum), +1 at 14:00 (maximum)."""
    if 6.0 <= h < 14.0:
        return -math.cos(math.pi * (h - 6.0) / 8.0)
    d = (h - 14.0) % 24.0
    return math.cos(math.pi * d / 16.0)


def weather_distribution(cfg: dict, monsoon: str, month: int, local_hour: float) -> dict:
    c = cfg["climate"]
    rain = c["rain_hour_fraction"][monsoon][month - 1]
    share = c["storm_share_of_rain"][month - 1]
    if 14.0 <= local_hour < 19.0:
        share *= c["storm_afternoon_boost"]
    elif local_hour < 8.0 or local_hour >= 22.0:
        share *= 0.5
    share = min(0.9, share)
    dry = 1.0 - rain
    return {"SUNNY": dry * (1 - c["cloudy_share_of_dry"]), "CLOUDY": dry * c["cloudy_share_of_dry"],
            "RAINY": rain * (1 - share), "STORM": rain * share}


def step_area(area: AreaState, cfg: dict, t0: int, t1: int, local_mid: datetime, k: int) -> AreaCond:
    seed = cfg["seed"]
    c = cfg["climate"]
    month = local_mid.month
    lh = local_mid.hour + local_mid.minute / 60.0

    # Weather changes at most once per hour (Markov chain with persistence).
    hour = t0 // HOUR_MS
    if hour != area.weather_hour:
        rh = rngmod.rng(seed, rngmod.AREA_HOUR, area.idx, hour)
        if area.weather_hour == -1 or rh.random() > c["weather_persistence"]:
            area.weather = rngmod.weighted_choice(rh, weather_distribution(cfg, area.monsoon, month, lh))
        area.weather_hour = hour
    w = area.weather

    r = rngmod.rng(seed, rngmod.AREA_INTERVAL, area.idx, k)
    mean = c["mean_temp_c"][month - 1] + area.temp_offset
    ambient = mean + area.diurnal_amp * _diurnal_shape(lh) + c["temp_delta_c"][w] + r.gauss(0, c["temp_noise_c"])
    humidity = (c["base_humidity_pct"][month - 1] + (3 if area.coastal else -3)
                - 2.0 * (ambient - mean) + c["humidity_delta_pct"][w] + r.gauss(0, 2.0))
    humidity = int(round(min(100.0, max(35.0, humidity))))
    elev = max(0.0, math.sin(math.pi * (lh - 6.0) / 12.0)) if 6.0 <= lh <= 18.0 else 0.0
    sun = elev * c["sun_factor"][w]
    lightning = rngmod.poisson(r, c["lightning_per_interval"][w])

    # Upstream grid outage (line and auxiliary AC lost for every recloser in the area).
    o = cfg["outages"]
    if area.outage_until is not None and area.outage_until <= t0:
        area.outage_start = area.outage_until = None
    if area.outage_until is None:
        lam = o["area_rate_per_year"] / INTERVALS_PER_YEAR * o["area_weather_mult"][w]
        if r.random() < lam:
            start = t0 + int(r.random() * (t1 - t0))
            dur_min = min(o["area_duration_cap_h"] * 60.0,
                          o["area_duration_median_min"] * math.exp(r.gauss(0, o["area_duration_sigma"])))
            area.outage_start = start
            area.outage_until = start + max(MIN_MS, int(dur_min * MIN_MS))

    return AreaCond(weather=w, ambient_c=ambient, humidity_pct=humidity, lightning=lightning, sun=sun,
                    month=month, local_hour=lh, weekday=local_mid.weekday(),
                    outage_start=area.outage_start, outage_until=area.outage_until)
