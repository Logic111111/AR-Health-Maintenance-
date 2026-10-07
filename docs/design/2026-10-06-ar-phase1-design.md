# NOJA Auto Recloser Preventive Maintenance — Phase 1 Design

Date: 2026-10-06 · Status: **APPROVED 2026-10-06** (HI model as written; dataset 30 ARs × 2 years) · Supersedes the 2026-10-05 spec
References: NOJA-5002-14 OSM User Manual (firmware 1.22); your 42-variable table (V1–V42 + weightages)

---

## 1. Goal

**Phase 1 (this build)**
1. **Real-time data getter**: every 15 minutes it collects one interval of raw data per auto recloser (AR), shaped like a real NOJA OSM + RC cubicle: load-profile telemetry, Close/Open log, fault/event logs, battery tests, comms events, lifetime counters, maintenance records.
2. **Final data**: calculates the 42 Excel variables (V1–V42) for every AR every 15 minutes.
3. **Health Index (HI)**: a mathematical model turns the 42 variables into a 0–100 HI, health class and healthy / not-healthy flag, every 15 minutes.
4. **ML-ready dataset**: enough history, plus failure labels, to train a model in a notebook.

Everything is stored in **MySQL** (your server is MariaDB 10.6; `DATABASE_URL` in `.env`). It also runs on SQLite for tests.

**Phase 2 (later, not built now)**: your ML model reads the same 15-min variables, predicts health, and is compared live with the HI.

**Success criteria**
- `scheduler` keeps every AR up to date at :00/:15/:30/:45 (raw → variables → HI), filling any missed intervals.
- `validate` passes: no unexplained gaps, physics/counter/event consistency.
- HI falls before recorded failures (checked against the hidden ground truth file), so the data is useful for ML.
- Same seed + config ⇒ identical data; one long backfill == many 15-min runs.

---

## 2. Architecture

```
            every 15 min (scheduler / run-once / backfill)
┌──────────────────────┐   raw rows    ┌────────────┐   ┌──────────────────┐   ┌───────────────────┐
│ RecloserSource       │ ───────────▶ │ raw tables │──▶│ VariableCalculator│──▶│ HealthIndexModel  │
│  SimulatedNojaSource │               │ (MySQL)    │   │ ar_variables_15min│   │ ar_health_index_  │
│  (later: real NOJA   │               └────────────┘   └──────────────────┘   │ 15min             │
│   via DNP3/CMS logs) │                                                        └───────────────────┘
└──────────────────────┘                                   status / export-dataset read from the tables
```
- `RecloserSource` is an interface: `collect(until_ts) -> RawBatch`. Phase 1 implements `SimulatedNojaSource`, a deterministic physics-based simulator of each OSM + RC. A real source (SCADA/DNP3 or CMS log import) can replace it later without touching the variable or HI code.
- The simulator keeps a hidden per-AR state (true battery capacity, mechanism wear, modem and controller health, insulator pollution) in `state/sim_state.json`. This is never stored in the database. Random numbers come from `(seed, asset, interval)`, so the run is deterministic and can be resumed. If a crash happens between the DB commit and the state save, the rows written after the last save are deleted and regenerated identically.

---

## 3. Database (MySQL; UTC `DATETIME(3)`)

**Raw (what a NOJA RC / SCADA records)**
| table | grain | key columns |
|---|---|---|
| asset_master | 1 row / AR | model, ratings (16 / 12.5 kA, 70 / 140 fault ops, 30 000 mech ops), manufactured / installed / last maintenance dates, RC model, relay module, firmware, battery model / type / Ah / service interval / install date, comms channel, RS485 link, coastal flag, area, feeder, External Load Time |
| telemetry_15min | 1 row / AR / 15 min, PK (asset_id, ts) | ia, ib, ic, in, uab, ubc, uca, frequency, switch_state, ac_supply_on, external_load_on, battery V / A / charge %, battery / cubicle / ambient temp, humidity, weather, lightning_strikes_10km, comms_connected, signal_strength_dbm, co_total, mechanical_wear_pct, contact_wear_pct_a/b/c |
| co_operations | per operation | ts (ms), OPEN/CLOSE, source (PROTECTION / AUTO_RECLOSE / HMI / CMS / SCADA / IO / MECH_TRIP), ar_state (OPEN1_LOCKOUT, OPEN2–4, CLOSE1–4), trip currents, operating_time_ms, SUCCESS/FAIL, fault_id |
| fault_events | per fault | start/end, type (OC_A/B/C, EF, SEF, NPS, UV, OV, UF, OF), phases, max current, fault-on duration, reclose attempts, outcome (CLEARED_BY_RECLOSE / LOCKOUT / UPSTREAM_CLEARED) |
| alarm_events | per event | WARNING / MALFUNCTION / ALERT, Appendix F signal name, RAISED / CLEARED / EVENT, detail |
| battery_tests | per test | AUTO/MANUAL, PASSED / FAILED / NOT_PERFORMED (+reason), voltage before, min voltage, test load current |
| comms_events | per event | MOBILE / FIBRE / RS485; DISCONNECT / RECONNECT / MODEM_RESTART / RS485_ERROR |
| failure_log, maintenance_log | per record | component (BATTERY / MECHANISM / CONTACTS / COMMS / CONTROLLER), PREVENTIVE / CORRECTIVE, linked failure, `details.changes = {field: [old, new]}` |
| firmware_release | reference | version, release date (manual revision history) |

**Calculated (every 15 min)**
| table | grain | content |
|---|---|---|
| ar_variables_15min | PK (asset_id, ts) | `telemetry_present`, v01 … v42 in natural units (§5) |
| ar_health_index_15min | PK (asset_id, ts) | health_index, health_class, is_healthy, cap_applied, 7 factor scores, worst_variables, s01 … s42 sub-scores (§6) |

---

## 4. Simulator assumptions (all in `config.yaml`)
- **Fleet:** default 30 ARs (≈60 % OSM15-310 on 11 kV, 40 % OSM27-310 on 22 kV), ~30 % coastal, switchgear ages 0–15 yr, 5 Sri Lanka-like areas (SW / NE monsoon zones, Asia/Colombo), 2 feeders each. RC-10ES / REL-02 for older units, RC-15 / REL-15-4GA for newer. ~15 % fibre, ~20 % with an RS485 device. Firmware drawn from the manual's release history.
- **Environment (shared per area):** monthly temperature/humidity, daily cycle, hourly weather Markov chain (SUNNY/CLOUDY/RAINY/STORM) with monsoon and inter-monsoon thunderstorms, lightning counts in storms. Cubicle temperature = ambient + solar gain + electronics.
- **Load:** feeder peak 120–550 A, residential / commercial / industrial / mixed daily curves, weekend factors, temperature effect, noise, phase imbalance. Residual current from the phasor sum plus leakage on polluted coastal sites.
- **Supply:** area-wide grid outages (line + aux AC lost, worse in storms) and local aux failures (VT fuse/MCB; line live, hours–days).
- **Battery/UPS (manual §4.8):** float 13.6–13.8 V, bulk charge ≤ 2.6 A. On battery: controller ~0.2 A + comms 0.25–0.5 A until External Load Time (120 / 480 / 1440 min) expires, which also drops comms. Standby at 20 % charge or 10.5 V; power-down 5 min later (telemetry gap); Power Restart when AC returns. True capacity fades with temperature-weighted age (halves per +10 °C), faster after the BAT-14 2 yr / BAT-11 4 yr service interval, plus cycle wear. Internal resistance = R0 / SOH^2.5. Auto battery test every 30 days with a 10 A / 30 s load; NOT_PERFORMED on AC off / < 12.5 V; FAILED below 12.0 V.
- **Protection (manual §6):** fault rate × weather × coastal pollution × feeder exposure. Factory AR map R-R-L: trip → 10 s → reclose → trip → 20 s → reclose → trip = lockout; crew restores after 1–8 h. IEC standard-inverse timing. Fault current lognormal, capped at the rated break current.
- **Switching:** open 20–30 ms / close 40–60 ms (manual §2.1.2), drifting up with mechanism wear. Excessive To > 60 ms, Excessive Tc > 100 ms, Trip/Close Request Fail; a failed protection trip ⇒ UPSTREAM_CLEARED + MECH_TRIP isolation. Lifetime counters per manual §7.7.1. Contact wear += 100 / N(I) on a log-log curve between (800 A, 30 000) and (rated kA, rated fault ops).
- **Comms:** cellular RSSI −55 … −105 dBm or fibre optical power −12 … −22 dBm, rain fade, disconnect hazard rising with weak signal, rain, AC loss and modem degradation. Watchdog modem restarts; RS485 error bursts.
- **Controller:** Appendix F malfunctions at base rates, raised by degradation, heat and humidity; mostly transient, persistent near failure.
- **Hidden degradation profiles** (file only, not DB): HEALTHY, BATTERY_DEGRADING, MECHANISM_WEAR, COMMS_PROBLEM, CONTROLLER_PROBLEM, HARSH_ENVIRONMENT. Onsets are spread over the run; new episodes start after repairs, so data keeps varying.
- **Failures → maintenance:** a threshold crossing writes `failure_log`, then CORRECTIVE maintenance 1–21 days later resets the component (new battery; OSM tank replacement ⇒ counters reset + new serial; modem/module swap). Annual PREVENTIVE inspection: test operations, bushing cleaning, battery replacement when due (70 % compliance), firmware upgrade (30 %).

---

## 5. Final data: V1–V42 every 15 min (`ar_variables_15min`)
W24 = last 24 h, W30 = last 30 days (configurable). Snapshot = value at ts. A row is written for every AR and interval. When the controller was powered down, `telemetry_present = 0` and the snapshot columns are NULL.

| V | column | definition |
|---|---|---|
| V1 | v01_battery_voltage_v | battery voltage at ts |
| V2 | v02_discharge_episodes_24h | discharge episode starts (battery current < −0.05 A) in W24 |
| V3 | v03_soh_est_pct | SOH **estimate**: latest of (a) Ah discharged ÷ Δcharge% ÷ rated Ah over a W30 episode with Δ ≥ 8 %, (b) last battery test of the current battery: 100·(R_ref ÷ apparent R)^0.58; 100 for a battery < 60 days old with no evidence, otherwise NULL |
| V4 | v04_battery_life_used_pct | battery age ÷ service interval × 100 |
| V5 | v05_ac_failures_30d | `AC Off (On Battery Supply)` raised, W30 |
| V6 | v06_max_dod_30d_pct | 100 − min charge %, W30 |
| V7 | v07_battery_alarms_30d | Check Battery, Critical Battery Level, Battery Status Abnormal, Battery Off, Battery Charger Fault, W30 |
| V8 | v08_battery_temp_age_years | battery age (yr) × 2^((mean battery temp W30 − 25)/10) |
| V9 | v09_auto_test_failed_count | FAILED AUTO tests of the current battery |
| V10 | v10_minutes_on_battery_30d | minutes with AC Off, W30 |
| V11 | v11_wear_max_pct | max(mechanical, contact A/B/C) |
| V12 | v12_lightning_faults_30d | faults starting with lightning_strikes_10km > 0, W30 |
| V13–V17 | v13 … v17 (_30d) | permanent (LOCKOUT + UPSTREAM_CLEARED), temporary, EF, SEF, OC fault counts |
| V18 | v18_max_fault_current_30d_a | max fault current, W30 (0 if none) |
| V19 | v19_faults_24h | faults, W24 |
| V20 | v20_lockouts_30d | lockouts, W30 |
| V21 | v21_co_total | lifetime Close/Open count |
| V22–V25 | v22 … v25 (_30d) | open ops, close ops, protection trips, auto-reclose closes |
| V26 | v26_switching_ops_24h | all ops, W24 |
| V27 | v27_unsuccessful_ops_30d | FAIL ops + opens > 60 ms + closes > 100 ms, W30 |
| V28 | v28_scada_availability_24h_pct | connected rows ÷ expected rows, W24 |
| V29 / V32 | v29_comm_failures_24h / v32_reconnects_24h | DISCONNECT / RECONNECT, W24 |
| V30 | v30_modem_restarts_30d | MODEM_RESTART, W30 |
| V31 | v31_signal_strength_24h_dbm | mean signal, W24 |
| V33 | v33_module_faults_30d | module MALFUNCTIONs, W30 |
| V34 | v34_diag_warnings_30d | other WARNING/MALFUNCTION, W30 |
| V35 | v35_firmware_versions_behind | newer releases available |
| V36 | v36_rs485_errors_30d | RS485 errors, W30 (NULL if no RS485 link) |
| V37 / V38 | v37_sg_age_years / v38_years_in_service | from OSM manufacture / install date |
| V39 | v39_days_since_maintenance | days since last maintenance |
| V40 | v40_ambient_temp_c | ambient temperature at ts |
| V41 | v41_coastal_flag | 0/1 |
| V42 | v42_weather | weather at ts |

---

## 6. Health Index mathematical model (`ar_health_index_15min`)

**Step 1 — sub-score.** Each variable maps to s_i ∈ [0, 100] (100 = healthy) by a piecewise-linear function between a *good* threshold g_i (s = 100) and a *bad* threshold b_i (s = 0):
s_i = 100 · clamp((b_i − x_i) / (b_i − g_i), 0, 1). This works in both directions; V1 uses a two-sided band.

**Step 2 — factor score.** F_k = Σ_{i∈k} w_i s_i / Σ_{i∈k} w_i over the variables that are available (missing ones, e.g. V36 with no RS485 link, have their weight redistributed within the factor).

**Step 3 — Health Index.** HI = Σ_k W_k · F_k / 100, where W_k = Σ w_i of the factor = your percentages (Battery 20, Faults 24, Switching 20, Comms 10, Controlling 13, Aging 8, Environment 5 = 100). When nothing is missing, HI = Σ_i w_i · s_i / 100.

**Step 4 — critical-condition cap.** A weighted average can hide one dangerous condition, so if a critical sub-score hits 0 the HI is capped: V27 unsuccessful ops → ≤ 45, V11 wear ≥ 100 % → ≤ 40, V3 SOH ≤ 50 % → ≤ 50, V9 ≥ 2 failed tests → ≤ 50, V33 module faults → ≤ 55. `cap_applied` names the variable.

**Step 5 — classification.** HI ≥ 80 **GOOD**, 60–80 **FAIR** (monitor), 40–60 **POOR** (plan maintenance), < 40 **CRITICAL** (act now). `is_healthy` = HI ≥ 60. `worst_variables` = the 3 largest w_i·(100 − s_i).

**Thresholds (good g → bad b; all in `config.yaml`)**
| V | g → b | V | g → b | V | g → b |
|---|---|---|---|---|---|
| V1 V | band 12.6–14.4 → 11.6 / 14.8 | V15 EF | 0 → 8 | V29 fails/24h | 0 → 10 |
| V2 /24h | 0 → 4 | V16 SEF | 0 → 4 | V30 restarts | 0 → 10 |
| V3 % | 90 → 50 | V17 OC | 0 → 8 | V31 dBm | −75 → −105 (fibre −18 → −27) |
| V4 % | 50 → 120 | V18 % of rated kA | 20 → 80 | V32 /24h | 1 → 10 |
| V5 | 0 → 10 | V19 /24h | 0 → 4 | V33 | 0 → 3 |
| V6 % | 20 → 80 | V20 | 0 → 4 | V34 | 3 → 30 |
| V7 | 0 → 5 | V21 % of 30 000 | 10 → 100 | V35 releases | 0 → 6 |
| V8 yr | 1 → 4 | V22 / V23 | 10 → 60 | V36 | 0 → 20 |
| V9 | 0 → 2 | V24 | 3 → 25 | V37 / V38 yr | 5 → 30 (30-yr design life, manual §3.1) |
| V10 min | 60 → 2880 | V25 | 3 → 20 | V39 days | 365 → 1095 |
| V11 % | 20 → 100 | V26 /24h | 2 → 15 | V40 °C | 30 → 45 |
| V12 | 0 → 5 | V27 | 0 → 3 | V41 | non-coastal 100, coastal 40 |
| V13 | 0 → 4 | V28 % | 99 → 80 | V42 | SUNNY 100, CLOUDY 90, RAINY 60, STORM 20 |
| V14 | 2 → 15 | | | | |

---

## 7. Commands (`python main.py <command>`)
| command | what it does |
|---|---|
| `init-db [--reset]` | create MySQL database, tables, indexes |
| `seed [--assets N] [--start]` | create the AR fleet and hidden profiles |
| `backfill [--end]` | generate history up to now (raw → variables → HI), committed per day |
| `run-once` | collect the latest 15-min interval(s) and update variables + HI |
| `scheduler` | run-once at :00/:15/:30/:45 (Windows Task Scheduler alternative in README) |
| `status` | print every AR's latest HI, class, healthy flag, worst variables |
| `validate` | gaps, ranges, counters vs operations, fault ↔ operation sequences, alarm pairing |
| `export-dataset [--out]` | ML dataset (§8) as CSV |
| `compute-variables` / `compute-hi [--start --end]` | recompute a range |

---

## 8. ML dataset for Phase 2
- **Size (default):** 30 ARs × 2 years ≈ 2.1 M rows per 15-min table. It includes several dozen failures across all components and every profile at least once (adjustable).
- `export-dataset` writes `output/ml/ar_ml_dataset.csv.gz` with one row per AR per 15 min: v01–v42, s01–s42, factor scores, HI, class, plus **labels** from `failure_log`: `failure_next_7d`, `failure_next_30d`, `next_failure_component`, `days_to_next_failure`. It also writes `ar_ml_ground_truth.csv.gz` (hidden true SOH, mechanism / comms / controller degradation per AR per day) for evaluating the model honestly. Loads directly in Jupyter or Google Colab with pandas.

---

## 9. Variables that raw recloser data cannot give directly (flagged)
- **V3 SOH:** the RC reports charge %, not SOH, so V3 is estimated (above). The hidden true SOH is in the ground-truth file.
- **V12 Surge/Lightning:** the RC does not classify lightning. Proxy: `lightning_strikes_10km` (an external lightning-detection feed) + `AC Status (Surge)`. Real-world better source: utility outage cause codes.
- **V31:** cellular RSSI and fibre optical power are different scales, hence channel-specific thresholds.
- **V35:** needs the `firmware_release` reference table.
- **V36:** the RC-10/RC-15 has no native RS485 port (RS-232 / USB / LAN). Errors come from an external RS485 converter/RTU; NULL for ARs without one.

## 10. Decisions log
- 2026-10-05: design approved; climate Sri Lanka-like; variables every 15 min with 24 h / 30 d windows; target DB MySQL via `.env`.
- 2026-10-06: scope = Phase 1 (getter + variables + HI); ML in Phase 2. HI model §6 approved as written (caps on). Dataset 30 ARs × 2 years approved.
- 2026-10-06 (build): telemetry columns are instantaneous values at the poll time `ts` (what a DNP3 poll returns). V3 test exponent calibrated to 0.58 against the battery model (was 0.4); V3 is NULL without evidence unless the battery is < 60 days old. Coulomb-counting evidence only from completed episodes (no look-ahead). Modem watchdog restarts back off (20 min → half the outage, max 4 h). `config_hash` covers simulation settings only. Added `copy-db` (SQLite → MySQL) and a guard that refuses to collect into a database without the seeded fleet. CONTACTS failures are rare in 2 years (realistic); the other four components all fail several times.
