"""Create the auto-recloser fleet and its hidden profiles (deterministic from the seed)."""
from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

from . import rng as rngmod
from .battery import calendar_soh
from .state import DEGRADATION_PROCESSES, AreaState, AssetState, SimState, new_ids
from .timeutil import DAY_MS, date_str, local_date, ms_to_dt, parse_date

PROFILE_NAMES = ("HEALTHY", "BATTERY_DEGRADING", "MECHANISM_WEAR", "COMMS_PROBLEM",
                 "CONTROLLER_PROBLEM", "HARSH_ENVIRONMENT")


def _largest_remainder(weights: dict, n: int) -> dict:
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:rest]:
        counts[k] += 1
    return counts


def _pick_count(r, items: list, k: int) -> set:
    pool = list(items)
    r.shuffle(pool)
    return set(pool[:k])


def _assign_profiles(n: int, cfg: dict, r, coastal: list[bool]) -> list[list[str]]:
    pcfg = cfg["fleet"]["profiles"]
    counts = _largest_remainder(pcfg["weights"], n)
    if n >= 6:
        for name in PROFILE_NAMES:
            if pcfg["weights"].get(name, 0) > 0 and counts.get(name, 0) == 0:
                donor = max(counts, key=counts.get)
                counts[donor] -= 1
                counts[name] = 1
    pool = [name for name, c in counts.items() for _ in range(c)]
    r.shuffle(pool)
    # Harsh environments go to coastal sites where possible.
    for i, p in enumerate(pool):
        if p == "HARSH_ENVIRONMENT" and not coastal[i]:
            for j in range(n):
                if coastal[j] and pool[j] != "HARSH_ENVIRONMENT":
                    pool[i], pool[j] = pool[j], pool[i]
                    break
    result = []
    non_healthy = [p for p in PROFILE_NAMES if p != "HEALTHY" and pcfg["weights"].get(p, 0) > 0]
    for p in pool:
        profs = [p]
        if p != "HEALTHY" and r.random() < pcfg["secondary_probability"]:
            options = [q for q in non_healthy if q != p]
            if options:
                profs.append(options[r.randrange(len(options))])
        result.append(profs)
    return result


def _firmware_at(releases: list, d: date) -> str:
    valid = [x for x in releases if parse_date(str(x["release_date"])) <= d]
    return (valid[-1] if valid else releases[0])["version"]


def _lognormal(r, median: float, sigma: float) -> float:
    return median * math.exp(r.gauss(0.0, sigma))


def create_fleet(cfg: dict, start_ms: int) -> SimState:
    seed = cfg["seed"]
    tz = cfg["timezone"]
    n = cfg["fleet"]["n_assets"]
    r = rngmod.rng(seed, rngmod.FLEET)
    start_date = local_date(start_ms, tz)

    areas = [AreaState(idx=i, name=a["name"], monsoon=a["monsoon"], coastal=bool(a["coastal"]),
                       temp_offset=float(a["temp_offset_c"]), diurnal_amp=float(a["diurnal_amp_c"]))
             for i, a in enumerate(cfg["areas"])]

    # Feeders per area
    lcfg = cfg["load"]
    feeders = {}
    for area in areas:
        for j in range(cfg["feeders_per_area"]):
            name = f"{area.name}-F{j + 1}"
            imb = [r.gauss(0, lcfg["phase_imbalance_sigma"]) for _ in range(3)]
            mean = sum(imb) / 3
            feeders[name] = dict(
                type=rngmod.weighted_choice(r, lcfg["feeder_types"]),
                peak=r.uniform(*lcfg["feeder_peak_a"]),
                imbalance=[x - mean for x in imb],
                v_offset=r.uniform(*lcfg["voltage_feeder_offset"]),
                exposure=r.uniform(*cfg["protection"]["exposure_range"]))

    area_of = [i % len(areas) for i in range(n)]
    feeder_of = [f"{areas[area_of[i]].name}-F{(i // len(areas)) % cfg['feeders_per_area'] + 1}"
                 for i in range(n)]

    coastal_candidates = [i for i in range(n) if areas[area_of[i]].coastal]
    n_coastal = min(len(coastal_candidates), int(round(cfg["fleet"]["coastal_fraction"] * n)))
    coastal_set = _pick_count(r, coastal_candidates, n_coastal)
    coastal = [i in coastal_set for i in range(n)]

    fibre_set = _pick_count(r, range(n), int(round(cfg["fleet"]["fibre_fraction"] * n)))
    rs485_set = _pick_count(r, range(n), int(round(cfg["fleet"]["rs485_fraction"] * n)))
    profiles = _assign_profiles(n, cfg, r, coastal)
    releases = cfg["firmware_releases"]
    bcfg, scfg, ccfg, mcfg = cfg["battery"], cfg["switching"], cfg["comms"], cfg["maintenance"]
    coast_km = cfg["fleet"].get("coastal_distance_km", [0.2, 5.0])
    inland_km = cfg["fleet"].get("inland_distance_km", [10.0, 80.0])

    assets = []
    for i in range(n):
        ar = rngmod.rng(seed, rngmod.FLEET, 1000 + i)
        model = rngmod.weighted_choice(ar, cfg["fleet"]["model_mix"])
        mc = cfg["models"][model]
        fd = feeders[feeder_of[i]]
        harsh = "HARSH_ENVIRONMENT" in profiles[i]

        age_years = ar.uniform(*cfg["fleet"]["age_years"])
        manufactured = start_date - timedelta(days=int(age_years * 365.25))
        installed = min(manufactured + timedelta(days=ar.randint(30, 240)), start_date - timedelta(days=7))
        if installed < manufactured:
            installed = manufactured

        firmware = _firmware_at(releases, manufactured)
        if ar.random() < cfg["firmware_upgrade_at_seed_prob"]:
            later = [x["version"] for x in releases
                     if parse_date(str(x["release_date"])) <= start_date and
                     parse_date(str(x["release_date"])) > manufactured]
            if later:
                firmware = later[ar.randrange(len(later))]

        rc_model = "RC-15" if manufactured >= date(2016, 2, 19) and ar.random() < 0.8 else "RC-10ES"
        relay = "REL-15-4GA" if rc_model == "RC-15" else ("REL-02" if manufactured >= date(2013, 11, 11)
                                                          else "REL-01")
        channel = "FIBRE" if i in fibre_set else "MOBILE"
        weak = "COMMS_PROBLEM" in profiles[i] and ar.random() < cfg["fleet"]["profiles"]["weak_coverage_share"]
        if channel == "FIBRE":
            base_signal = ar.uniform(*ccfg["fibre_rx_dbm"])
        elif weak:
            base_signal = ar.uniform(*ccfg["weak_coverage_dbm"])
        else:
            base_signal = ar.uniform(*ccfg["mobile_signal_dbm"])

        bmodel = rngmod.weighted_choice(ar, bcfg["model_mix"])
        interval_y = bcfg["models"][bmodel]["service_interval_years"]
        batt_age_days = ar.uniform(*bcfg["initial_age_fraction_of_interval"]) * interval_y * 365.25
        batt_install = start_date - timedelta(days=int(batt_age_days))

        peak = fd["peak"] * ar.uniform(*cfg["fleet"]["position_load_factor"])
        pcfg = cfg["protection"]
        oc_pickup = min(1280.0, max(300.0, round(pcfg["oc_pickup_factor"] * peak / 10.0) * 10.0))

        last_maint = start_date - timedelta(days=int(ar.uniform(*mcfg["initial_last_maintenance_months"]) * 30.4))
        if last_maint < installed:
            last_maint = installed
        next_pm = last_maint + timedelta(days=mcfg["preventive_interval_days"]
                                         + ar.randint(-mcfg["preventive_jitter_days"], mcfg["preventive_jitter_days"]))
        if next_pm <= start_date:
            next_pm = start_date + timedelta(days=ar.randint(1, 60))
        next_pm_ms = start_ms + (next_pm - start_date).days * DAY_MS

        # lifetime counters consistent with age
        co_total = int(age_years * ar.uniform(*scfg["initial_ops_per_year"]))
        base_wear = age_years * ar.uniform(*scfg["initial_contact_wear_per_year"])
        wear = [min(scfg["initial_contact_wear_cap"], base_wear * ar.uniform(0.85, 1.15)) for _ in range(3)]

        # hidden degradation onsets (profile processes start inside the window,
        # others follow the background hazard)
        onsets, rates = {}, {}
        for proc in DEGRADATION_PROCESSES:
            if proc in profiles[i] and not (proc == "COMMS_PROBLEM" and weak):
                onsets[proc] = start_ms + int(ar.uniform(*cfg["fleet"]["profiles"]["onset_window_days"]) * DAY_MS)
            else:
                years = ar.expovariate(1.0 / mcfg["next_onset_mean_years"][proc])
                onsets[proc] = start_ms + int(years * 365.25 * DAY_MS)
        rates["BATTERY_DEGRADING"] = ar.uniform(*bcfg["degrading_extra_fade_per_day"])
        rates["MECHANISM_WEAR"] = ar.uniform(*scfg["mech_deg_growth_per_day"])
        rates["COMMS_PROBLEM"] = ar.uniform(*ccfg["comms_deg_growth_per_day"])
        rates["CONTROLLER_PROBLEM"] = ar.uniform(*cfg["controller"]["ctrl_deg_growth_per_day"])

        serial = f"0{mc['serial_code']}{manufactured.year % 100:02d}{manufactured.month:02d}{i + 1:07d}"
        a = AssetState(
            idx=i, asset_id=f"AR-{i + 1:03d}", area_idx=area_of[i], feeder_name=feeder_of[i], model=model,
            rated_voltage_kv=float(mc["rated_voltage_kv"]), system_kv=float(mc["system_voltage_kv"]),
            rated_break_ka=float(mc["rated_break_ka"]), rated_fault_ops=int(mc["rated_fault_break_ops"]),
            rated_mech_ops=int(mc["rated_mech_ops"]), feeder_type=fd["type"], peak_a=peak,
            imbalance=list(fd["imbalance"]), v_offset=fd["v_offset"], exposure=fd["exposure"],
            prospective_fault_ka=ar.uniform(*mc["prospective_fault_ka"]), oc_pickup_a=oc_pickup,
            ef_pickup_a=float(ar.randint(*pcfg["ef_pickup_a"])), coastal=coastal[i],
            distance_coast_km=round(ar.uniform(*coast_km) if coastal[i] else ar.uniform(*inland_km), 1),
            harsh=harsh, solar_gain=ar.uniform(6.0, 12.0) * (1.4 if harsh else 1.0),
            channel=channel, base_signal_dbm=base_signal, has_rs485=i in rs485_set,
            ctrl_load_a=ar.uniform(*bcfg["controller_load_a"]), ext_load_a=ar.uniform(*bcfg["external_load_a"]),
            ext_load_time_min=int(_weighted_list(ar, cfg["fleet"]["external_load_time_min"])),
            to_base_ms=ar.uniform(*scfg["open_time_ms"]), tc_base_ms=ar.uniform(*scfg["close_time_ms"]),
            rc_model=rc_model, relay_module=relay, profiles=profiles[i],
            notes={"weak_coverage": weak},
            serial=serial, manufactured=date_str(manufactured), installed=date_str(installed),
            last_maint=date_str(last_maint), firmware=firmware, battery_model=bmodel,
            battery_type="GEL" if ar.random() < bcfg["gel_fraction"] else "AGM",
            battery_interval_y=int(interval_y), battery_install=date_str(batt_install),
            init_cap=min(1.05, max(0.9, ar.gauss(1.0, bcfg["capacity_tolerance_sigma"]))),
            age_eff_days=batt_age_days * bcfg["initial_temp_age_factor"],
            next_test_ms=start_ms + int(ar.uniform(0.0, bcfg["test"]["interval_days"]) * DAY_MS),
            co_total=co_total, wear=wear,
            mech_deg=min(0.4, scfg["mech_deg_initial_per_year"] * age_years),
            pollution=ar.uniform(0.0, 0.3) if coastal[i] else ar.uniform(0.0, 0.05),
            onsets=onsets, rates=rates, next_pm_ms=next_pm_ms,
        )
        assets.append(a)

    return SimState(seed=seed, config_hash=cfg["config_hash"], sim_start_ms=start_ms, last_ms=start_ms,
                    next_ids=new_ids(), seq=0, last_gt_date=date_str(start_date), areas=areas, assets=assets)


def _weighted_list(r, spec: dict):
    return rngmod.weighted_choice(r, dict(zip(spec["values"], spec["weights"])))


# ------------------------------------------------------------------ DB rows
def asset_master_rows(state: SimState) -> list[dict]:
    rows = []
    for a in state.assets:
        rows.append(dict(
            asset_id=a.asset_id, model=a.model,
            rated_voltage_kv=a.rated_voltage_kv, system_voltage_kv=a.system_kv,
            rated_continuous_a=800, rated_break_ka=a.rated_break_ka,
            rated_fault_break_ops=a.rated_fault_ops, rated_mech_ops=a.rated_mech_ops,
            rc_model=a.rc_model, battery_rated_ah=26.0, comms_channel=a.channel, has_rs485_link=a.has_rs485,
            coastal_flag=a.coastal, distance_to_coast_km=a.distance_coast_km,
            area_name=state.areas[a.area_idx].name, feeder_name=a.feeder_name,
            external_load_time_min=a.ext_load_time_min, **asset_dynamic_fields(a)))
    return rows


def asset_dynamic_fields(a: AssetState) -> dict:
    return dict(manufactured_date=parse_date(a.manufactured), installed_date=parse_date(a.installed),
                last_maintenance_date=parse_date(a.last_maint), firmware_version=a.firmware,
                battery_model=a.battery_model, battery_type=a.battery_type,
                battery_service_interval_years=a.battery_interval_y,
                battery_install_date=parse_date(a.battery_install), osm_serial_number=a.serial,
                relay_module=a.relay_module)


def asset_update_row(a: AssetState) -> dict:
    return dict(asset_id=a.asset_id, **asset_dynamic_fields(a))


def firmware_rows(cfg: dict) -> list[dict]:
    return [dict(version=x["version"], release_date=parse_date(str(x["release_date"])))
            for x in cfg["firmware_releases"]]


def write_profiles(state: SimState, cfg: dict, out_dir: Path) -> Path:
    """Hidden profile assignment (the 'answer key') - never written to the database."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "asset_profiles.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["asset_id", "profiles", "weak_coverage", "coastal", "harsh",
                    *[f"onset_{p.lower()}" for p in DEGRADATION_PROCESSES]])
        for a in state.assets:
            w.writerow([a.asset_id, "+".join(a.profiles), a.notes.get("weak_coverage", False),
                        a.coastal, a.harsh,
                        *[ms_to_dt(a.onsets[p]).date().isoformat() for p in DEGRADATION_PROCESSES]])
    with open(out_dir / "asset_profiles.json", "w", encoding="utf-8") as fh:
        json.dump({a.asset_id: {"profiles": a.profiles, "notes": a.notes,
                                "onsets": {p: ms_to_dt(a.onsets[p]).isoformat() for p in DEGRADATION_PROCESSES}}
                   for a in state.assets}, fh, indent=2)
    return path
