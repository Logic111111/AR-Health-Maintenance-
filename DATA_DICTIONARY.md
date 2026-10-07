# Data dictionary

NOJA OSM15-310 / OSM27-310 auto reclosers with RC-10ES / RC-15 cubicles. Behaviour is based on the NOJA-5002-14 OSM User Manual (firmware 1.22). Section numbers below refer to that manual.

## Conventions

- **Time.** All timestamps are **UTC**, stored as `DATETIME(3)` (milliseconds) on MySQL/MariaDB. Daily and weekly patterns use local time (`Asia/Colombo`, UTC+05:30).
- **Intervals.** An interval is identified by its **end** time `ts`, and it covers `(ts − 15 min, ts]`. Rows exist at :00, :15, :30 and :45 UTC.
- **Windows.** `W24` = `(ts − 24 h, ts]` and `W30` = `(ts − 30 days, ts]`. Both lengths are set in `config.yaml → variables`.
- **No look-ahead.** Every calculated value at `ts` uses only raw rows with timestamps ≤ `ts`.
- **Hidden truth is never in the database.** The simulator's true battery capacity, mechanism wear, modem and controller health, and pollution stay in `state/sim_state.json`. The assigned degradation profiles are in `output/asset_profiles.csv` and the daily truth is in `output/ground_truth_daily.csv`.

---

## 1. Raw tables (what an RC / SCADA records)

### asset_master: one row per recloser
Current values. Every change is also recorded in `maintenance_log.details.changes`.

| column | type | description |
|---|---|---|
| asset_id | VARCHAR(16) PK | `AR-001` … |
| osm_serial_number | VARCHAR(13) | OSM tank serial (`0` + model code + YYMM + number); changes when the tank is replaced |
| model | VARCHAR(16) | `OSM15-310` (15.5 kV, used on 11 kV) or `OSM27-310` (27 kV, used on 22 kV) |
| rated_voltage_kv / system_voltage_kv | DECIMAL(4,1) | rating / network voltage |
| rated_continuous_a | SMALLINT | 800 A (§2.1) |
| rated_break_ka | DECIMAL(4,1) | 16 kA (OSM15) / 12.5 kA (OSM27) |
| rated_fault_break_ops | SMALLINT | 70 / 140 operations at the rated break current (§2.1.4) |
| rated_mech_ops | INT | 30 000 mechanical operations |
| manufactured_date / installed_date / last_maintenance_date | DATE | |
| rc_model / relay_module | VARCHAR | `RC-10ES` + `REL-01/02` or `RC-15` + `REL-15-4GA` |
| firmware_version | VARCHAR(16) | e.g. `1.20.0` |
| battery_model / battery_type | VARCHAR | `BAT-14` (2-year service interval) or `BAT-11` (4-year); `AGM` / `GEL` |
| battery_rated_ah | DECIMAL(4,1) | 26 Ah |
| battery_service_interval_years | SMALLINT | 2 or 4 |
| battery_install_date | DATE | |
| comms_channel | ENUM | `MOBILE` (cellular modem) or `FIBRE` |
| has_rs485_link | BOOL | an external RS485 device/RTU is attached (see V36) |
| coastal_flag / distance_to_coast_km | BOOL / DECIMAL | |
| area_name / feeder_name | VARCHAR | LECO branch (Negombo, Kelaniya, Kotte, Nugegoda, Moratuwa, Kalutara, Galle), three feeders each (e.g. `Galle-F2`) |
| external_load_time_min | SMALLINT | UPS "External Load Time" (§4.8.3): 120, 480 or 1440 min on battery before the radio/modem supply is switched off |

### telemetry_15min: one row per recloser per interval (SCADA poll at `ts`)
Values are **instantaneous at `ts`**, as a DNP3 poll returns them. A row is missing only while the controller is powered down (between a `Shutdown` and the following `Power Restart` event); `validate` checks this.

| column | unit | description |
|---|---|---|
| asset_id, ts | | PK |
| ia, ib, ic | A | phase currents (0 when the switch is open or the line is dead) |
| in_residual | A | residual current: phasor sum, plus leakage on polluted insulators |
| uab, ubc, uca | kV | source-side line voltages (0 when the line is dead) |
| frequency_hz | Hz | NULL when the line is dead |
| switch_state | | `CLOSED`, `OPEN`, `LOCKOUT` |
| ac_supply_on | bool | auxiliary AC supply present |
| external_load_on | bool | 12 V external (radio/modem) supply on |
| battery_voltage | V | battery terminal voltage: float 13.6–13.8 V (temperature compensated), lower on battery |
| battery_current | A | + charging / − discharging |
| battery_charge_pct | % | RC charge estimate |
| battery_temp_c, cubicle_temp_c | °C | inside the RC cubicle |
| ambient_temp_c, humidity_pct, weather | °C, %, `SUNNY/CLOUDY/RAINY/STORM` | external weather feed |
| lightning_strikes_10km | count | strikes within 10 km during the interval (external lightning-detection feed) |
| comms_connected | bool | SCADA link up at `ts` |
| signal_strength_dbm | dBm | cellular RSSI (−55 … −110) or fibre optical power (−12 … −27); NULL when the modem is off |
| co_total | count | lifetime Close/Open counter (§7.7.1); +1 per successful OPEN; reset when the OSM tank is replaced |
| mechanical_wear_pct | % | co_total ÷ rated_mech_ops × 100 |
| contact_wear_pct_a/b/c | % | vacuum interrupter contact wear per phase: +100 ÷ N(I) per interruption, with N(I) on a log-log curve from (800 A, 30 000) to (rated kA, rated fault ops) |

### co_operations: Close/Open log, one row per operation

| column | description |
|---|---|
| op_id (PK), ts (ms), asset_id | |
| op_type | `OPEN` / `CLOSE` |
| source | `PROTECTION`, `AUTO_RECLOSE`, `HMI`, `CMS`, `SCADA`, `IO`, `MECH_TRIP` |
| ar_state | `OPEN2`–`OPEN4`, `OPEN1_LOCKOUT` for trips; `CLOSE1`–`CLOSE4` for closes |
| trip_ia/ib/ic/in | interrupted currents (A) |
| operating_time_ms | opening time (20–30 ms new) or closing time (40–60 ms new) (§2.1.2); grows with mechanism wear; NULL when the operation failed |
| result | `SUCCESS` / `FAIL` (Trip/Close Request Fail) |
| fault_id | links protection trips and auto-recloses to `fault_events` |

### fault_events: one row per protection sequence

| column | description |
|---|---|
| fault_id (PK), start_ts, end_ts, asset_id | |
| fault_type | `OC_A/B/C` (overcurrent, dominant phase), `EF`, `SEF`, `NPS`, `UV`, `OV`, `UF`, `OF` |
| phases | e.g. `AB`, `ABC`, `BN` |
| max_fault_current_a | A |
| duration_ms | total fault-on time over all trips |
| reclose_attempts | number of auto-reclose closes |
| outcome | `CLEARED_BY_RECLOSE` (temporary), `LOCKOUT` (permanent), `UPSTREAM_CLEARED` (recloser failed to trip; backup protection cleared it) |

Protection uses the factory auto-reclose map R-R-L: trip → 10 s → reclose → trip → 20 s → reclose → trip → lockout. Trip times follow the IEC standard-inverse curve.

### alarm_events: RC event log (Appendix F names)

| column | description |
|---|---|
| event_id (PK), ts (ms), asset_id | |
| category | `WARNING`, `MALFUNCTION`, `ALERT` |
| signal_name | e.g. `AC Off (On Battery Supply)`, `Critical Battery Level`, `Battery Status Abnormal`, `Check Battery`, `Battery Off (On AC Supply)`, `Battery Charger Fault`, `Shutdown`, `Power Restart`, `Excessive To`, `Excessive Tc`, `Trip Request Fail`, `Close Request Fail`, `Mechanically Locked`, `OSM Disconnected`, `Open(OC)`, `Open(EF)`, module faults, comms errors, … |
| state | `RAISED` / `CLEARED` for conditions; `EVENT` for one-off events |
| detail | free text |

### battery_tests: UPS battery test log (§4.8.2)

| column | description |
|---|---|
| test_id (PK), ts, asset_id | |
| trigger | `AUTO` (every 30 days) / `MANUAL` (maintenance) |
| result | `PASSED`, `FAILED` (minimum voltage < 12.0 V), `NOT_PERFORMED` |
| not_performed_reason | `AC_OFF`, `LOW_VOLTAGE` (< 12.5 V), `DISCHARGING`, `RESTING` |
| voltage_before, voltage_min_during_test | V |
| test_load_current_a | 10 A for 30 s |

### comms_events

| column | description |
|---|---|
| event_id (PK), ts, asset_id | |
| channel | `MOBILE`, `FIBRE`, `RS485` |
| event | `DISCONNECT`, `RECONNECT`, `MODEM_RESTART` (watchdog), `RS485_ERROR` |

### failure_log / maintenance_log

| table | columns |
|---|---|
| failure_log | failure_id (PK), ts, asset_id, component (`BATTERY`, `MECHANISM`, `CONTACTS`, `COMMS`, `CONTROLLER`), detected_by (`ALARM`, `BATTERY_TEST`, `PATROL`, `SCADA`), description |
| maintenance_log | maint_id (PK), ts, asset_id, component, maint_type (`PREVENTIVE` / `CORRECTIVE`), failure_id (the failure a corrective job repaired), description, details (JSON) |

`details.changes` is `{field: [old, new]}` for every asset_master field the job changed: battery model, type and install date, firmware, OSM serial, manufactured and installed dates, and last maintenance date. When the OSM tank is replaced, `details.counters_reset = true` and `details.old_counters` holds the old lifetime counters.

### asset_location: map position of each recloser (dashboard)

| column | description |
|---|---|
| asset_id | PK, FK to asset_master |
| latitude, longitude | WGS84 degrees |
| source | `SIMULATED` (placed along feeders in the recloser's area) or `GPS` (loaded with `set-locations --csv`) |

### ar_ml_prediction_15min: ML failure risk per recloser per 15 min
Filled only when a model trained in Google Colab is installed in `models/` (see `docs/ML_COLAB_GUIDE.md`).

| column | description |
|---|---|
| asset_id, ts | PK |
| p_fail_30d | calibrated probability of a failure within 30 days (0–1) |
| ml_risk_class | `LOW` (< half the alarm level), `ELEVATED`, `HIGH` (≥ alarm level) |
| ml_healthy | p_fail_30d below the model's alarm level |
| top_component, top_component_p | most likely failing part and its probability (component model) |
| model_version | model that produced the row (e.g. `lgbm-20261007-1030`) |
| computed_at | when the row was calculated |

### ar_daily_summary: daily HI (and ML) per recloser per local day (Asia/Colombo)

| column | description |
|---|---|
| asset_id, day | PK |
| intervals | number of 15-min HI values in the day |
| avg_hi, min_hi, max_hi | daily average / minimum / maximum Health Index |
| unhealthy_hours | hours with HI < 60 |
| hi_healthy | avg_hi ≥ 60 |
| avg_p_fail, max_p_fail | daily average / maximum ML failure probability (NULL without a model) |
| ml_healthy | no 15-min ML warning in the day (NULL without a model) |
| agree | hi_healthy = ml_healthy (NULL without a model) |
| updated_at | when the row was recalculated |

The daily rows are recalculated whenever any interval of that day is (re)calculated, including by `compute-hi`.
So a day is always summarised from all its rows.

A day is never replaced by a summary of fewer rows. As a result, the first day kept by `prune-history` keeps the
summary it had when it was complete.

### hi_reset_log: manual "Maintenance done" Health-Index resets

| column | description |
|---|---|
| reset_id | PK (autoincrement) |
| ts | maintenance time (UTC); the reset applies from this moment |
| asset_id | FK |
| part_groups | `BATTERY`, `SWITCHING`, `COMMS`, `CONTROLLER` (comma list), or `ALL` |
| note | technician note |
| created_at | when the row was entered (UTC) |

### firmware_release
`version` (PK), `release_date`: the manual's revision history, 1.8.0 (2013) to 1.22.0 (2018). Used for V35.

---

## 2. Calculated tables (every 15 min)

### ar_variables_15min: V1–V42 "final data"
PK (`asset_id`, `ts`); one row for every recloser and every interval. `telemetry_present = 0` marks intervals with no telemetry row; telemetry snapshots (V1, V40, V42) are NULL then. Lifetime counters (V11, V21) carry the last known value forward. `computed_at` records when the row was calculated.

| V | column | unit | raw source | definition |
|---|---|---|---|---|
| V1 | v01_battery_voltage_v | V | telemetry.battery_voltage | value at ts |
| V2 | v02_discharge_episodes_24h | count | telemetry.battery_current | starts of discharge (current < −0.05 A after a non-discharging row) in W24 |
| V3 | v03_soh_est_pct | % | telemetry + battery_tests + maintenance_log | **estimated** state of health (see note A) |
| V4 | v04_battery_life_used_pct | % | battery_install_date, service interval (history) | battery age ÷ (service interval × 365.25 d) × 100 |
| V5 | v05_ac_failures_30d | count | alarm_events | `AC Off (On Battery Supply)` RAISED in W30 |
| V6 | v06_max_dod_30d_pct | % | telemetry.battery_charge_pct | 100 − minimum charge % in W30 |
| V7 | v07_battery_alarms_30d | count | alarm_events | RAISED `Battery Status Abnormal`, `Check Battery`, `Critical Battery Level`, `Battery Off (On AC Supply)`, `Battery Charger Fault` in W30 |
| V8 | v08_battery_temp_age_years | years | battery age + telemetry.battery_temp_c | age (y) × 2^((mean battery temperature in W30 − 25) / 10): temperature-equivalent age |
| V9 | v09_auto_test_failed_count | count | battery_tests | FAILED AUTO tests since the current battery was installed |
| V10 | v10_minutes_on_battery_30d | min | alarm_events | time inside AC Off RAISED→CLEARED periods within W30 |
| V11 | v11_wear_max_pct | % | telemetry | max(mechanical_wear_pct, contact_wear_pct_a/b/c) |
| V12 | v12_lightning_faults_30d | count | fault_events + telemetry.lightning_strikes_10km | faults starting in an interval with lightning > 0, in W30 (proxy, note B) |
| V13 | v13_permanent_faults_30d | count | fault_events | outcome LOCKOUT or UPSTREAM_CLEARED, W30 |
| V14 | v14_temporary_faults_30d | count | fault_events | outcome CLEARED_BY_RECLOSE, W30 |
| V15 | v15_earth_faults_30d | count | fault_events | fault_type EF, W30 |
| V16 | v16_sef_faults_30d | count | fault_events | fault_type SEF, W30 |
| V17 | v17_oc_faults_30d | count | fault_events | fault_type OC_A/B/C, W30 |
| V18 | v18_max_fault_current_30d_a | A | fault_events | max max_fault_current_a in W30 (0 if no fault) |
| V19 | v19_faults_24h | count | fault_events | all faults, W24 |
| V20 | v20_lockouts_30d | count | fault_events | outcome LOCKOUT, W30 |
| V21 | v21_co_total | count | telemetry.co_total | lifetime Close/Open count |
| V22 | v22_open_ops_30d | count | co_operations | OPEN operations, W30 |
| V23 | v23_close_ops_30d | count | co_operations | CLOSE operations, W30 |
| V24 | v24_protection_trips_30d | count | co_operations | OPEN with source PROTECTION, W30 |
| V25 | v25_auto_reclose_ops_30d | count | co_operations | CLOSE with source AUTO_RECLOSE, W30 |
| V26 | v26_switching_ops_24h | count | co_operations | all operations, W24 |
| V27 | v27_unsuccessful_ops_30d | count | co_operations | result FAIL, opening time > 60 ms or closing time > 100 ms, W30 |
| V28 | v28_scada_availability_24h_pct | % | telemetry.comms_connected | connected polls ÷ expected polls (96, fewer in the first day) in W24; missing rows count as unavailable |
| V29 | v29_comm_failures_24h | count | comms_events | DISCONNECT, W24 |
| V30 | v30_modem_restarts_30d | count | comms_events | MODEM_RESTART, W30 |
| V31 | v31_signal_strength_24h_dbm | dBm | telemetry.signal_strength_dbm | mean in W24 (note C) |
| V32 | v32_reconnects_24h | count | comms_events | RECONNECT, W24 |
| V33 | v33_module_faults_30d | count | alarm_events | MALFUNCTION RAISED for SIM/Relay/Panel/I/O module faults, RTC, CAN bus, module/SIM/panel/I/O comms errors, GPS, in W30 |
| V34 | v34_diag_warnings_30d | count | alarm_events | other WARNING/MALFUNCTION RAISED or EVENT rows in W30, excluding those counted in V5, V7, V27 and V33 (e.g. Shutdown, Power Restart, Mechanically Locked, OSM Disconnected, External Load Overload, Capacitor Voltage Abnormal, USB Overcurrent) |
| V35 | v35_firmware_versions_behind | count | firmware_version (history) + firmware_release | releases available at ts that are newer than the installed version (note D) |
| V36 | v36_rs485_errors_30d | count | comms_events | RS485_ERROR in W30; NULL without an RS485 link (note E) |
| V37 | v37_sg_age_years | years | manufactured_date (history) | age of the OSM switchgear |
| V38 | v38_years_in_service | years | installed_date (history) | time in service |
| V39 | v39_days_since_maintenance | days | last_maintenance_date (history) | days since the last maintenance |
| V40 | v40_ambient_temp_c | °C | telemetry.ambient_temp_c | value at ts |
| V41 | v41_coastal_flag | 0/1 | asset_master.coastal_flag | |
| V42 | v42_weather | text | telemetry.weather | value at ts |

"History" means the value valid at `ts`. It is rebuilt from `asset_master` plus `maintenance_log.details.changes`, so recomputing old intervals after a replacement gives the right answer.

**Health-Index reset after maintenance.** A reset at time `R` for a part group makes that group's windows start at `R`. For `t ≥ R`, W24 / W30 become `(max(t − W, R), t]`, and events at exactly `R` belong to the time before the reset.

Resets come from two sources:
- every `maintenance_log` row with `maint_type = CORRECTIVE`, which resets its component's group (BATTERY → BATTERY; MECHANISM and CONTACTS → SWITCHING; COMMS → COMMS; CONTROLLER → CONTROLLER);
- every `hi_reset_log` row.

| group | variables affected |
|---|---|
| BATTERY | V2, V6, V7 |
| SWITCHING | V22–V27 |
| COMMS | V28 (expected polls also count from `R`), V29–V32 |
| CONTROLLER | V33, V34 |
| any | V39 = days since max(last maintenance date, latest reset) |

The other variables are never reset; they describe the line, the supply, age or environment.

**Notes: variables that raw recloser data cannot give directly**
- **A. V3 SOH.** The RC reports charge %, not state of health, so V3 is estimated. It uses the most recent of two kinds of evidence:
  - coulomb counting over a completed discharge episode in W30 with a charge drop of at least 8 %: Ah discharged ÷ (Δcharge % / 100) ÷ rated Ah × 100;
  - the last battery test of the current battery: 100 · (0.035 Ω ÷ apparent R)^0.58, where apparent R = (voltage_before − voltage_min) ÷ test current.

  A battery younger than 60 days with no evidence scores 100. Otherwise, with no evidence, V3 is NULL; the HI then redistributes the weight. The true SOH is in the ground-truth file.
- **B. V12.** The RC does not classify lightning. The proxy is an external lightning-detection feed (`lightning_strikes_10km`). In the field, utility outage cause codes are a better source.
- **C. V31.** Cellular RSSI and fibre optical power use different scales, so the HI uses channel-specific thresholds.
- **D. V35.** Needs the `firmware_release` reference table, which you keep up to date.
- **E. V36.** RC-10/RC-15 have no native RS485 port (they have RS-232, USB and LAN). The errors come from an external RS485 converter or RTU.

### ar_health_index_15min: Health Index
PK (`asset_id`, `ts`). The model is in `ar_gen/health_index.py` and all parameters are under `config.yaml → health_index`.

1. **Sub-score** `s_i = 100 · clamp((bad − x) / (bad − good), 0, 1)`; it works for higher-is-better and lower-is-better variables.
   - V1 uses a band: 100 inside 12.6–14.4 V, falling to 0 at 11.6 V and at 14.8 V.
   - V18 is relative to the rated break current (20 % → 80 %); V21 is relative to 30 000 operations (10 % → 100 %).
   - V31 has separate thresholds for MOBILE (−75 → −105 dBm) and FIBRE (−18 → −27 dBm).
   - V41 maps coastal → 40, inland → 100. V42 maps SUNNY 100, CLOUDY 90, RAINY 60, STORM 20.
2. **Factor score** `F_k = Σ w_i s_i / Σ w_i` over the factor's available variables. A missing variable, such as V36 without RS485, has its weight redistributed within the factor.
3. **HI** `= Σ W_k F_k / Σ W_k`, with `W_k` = Battery 20, Faults 24, Switching 20, Comms 10, Controlling 13, Aging 8, Environment 5 (the Excel weightages).
4. **Caps.** If a critical sub-score is 0, the HI is limited:

   | variable | condition for a sub-score of 0 | HI cap |
   |---|---|---|
   | V27 | ≥ 3 unsuccessful operations | 45 |
   | V11 | wear ≥ 100 % | 40 |
   | V3 | SOH ≤ 50 % | 50 |
   | V9 | ≥ 2 failed tests | 50 |
   | V33 | ≥ 3 module faults | 55 |
5. **Class.** GOOD ≥ 80, FAIR ≥ 60, POOR ≥ 40, otherwise CRITICAL. `is_healthy` = HI ≥ 60.

| column | description |
|---|---|
| health_index | 0–100 |
| health_class | GOOD / FAIR / POOR / CRITICAL |
| is_healthy | HI ≥ 60 |
| cap_applied | variable that capped the HI (e.g. `v09`), else NULL |
| f_battery … f_environment | the seven factor scores (0–100) |
| worst_variables | the three variables with the largest `w_i · (100 − s_i)` |
| s01 … s42 | sub-scores (NULL when the variable is NULL) |

Thresholds per variable (good → bad):

| V | g → b | V | g → b | V | g → b |
|---|---|---|---|---|---|
| V1 | band 12.6–14.4 → 11.6 / 14.8 V | V15 | 0 → 8 | V29 | 0 → 10 |
| V2 | 0 → 4 | V16 | 0 → 4 | V30 | 0 → 10 |
| V3 | 90 → 50 % | V17 | 0 → 8 | V31 | −75 → −105 dBm (fibre −18 → −27) |
| V4 | 50 → 120 % | V18 | 20 → 80 % of rated kA | V32 | 1 → 10 |
| V5 | 0 → 10 | V19 | 0 → 4 | V33 | 0 → 3 |
| V6 | 20 → 80 % | V20 | 0 → 4 | V34 | 3 → 30 |
| V7 | 0 → 5 | V21 | 10 → 100 % of 30 000 | V35 | 0 → 6 |
| V8 | 1 → 4 years | V22, V23 | 10 → 60 | V36 | 0 → 20 |
| V9 | 0 → 2 | V24 | 3 → 25 | V37, V38 | 5 → 30 years |
| V10 | 60 → 2880 min | V25 | 3 → 20 | V39 | 365 → 1095 days |
| V11 | 20 → 100 % | V26 | 2 → 15 | V40 | 30 → 45 °C |
| V12 | 0 → 5 | V27 | 0 → 3 | V41 | inland 100, coastal 40 |
| V13 | 0 → 4 | V28 | 99 → 80 % | V42 | SUNNY 100 … STORM 20 |
| V14 | 2 → 15 | | | | |

---

## 3. ML dataset (`python main.py export-dataset`)

`output/ml/ar_ml_dataset.csv.gz` has one row per recloser per 15 minutes.

| columns | description |
|---|---|
| asset_id, ts, model, comms_channel | identifiers |
| telemetry_present, v01 … v42 | features (the final data) |
| health_index, health_class, is_healthy, cap_applied, worst_variables, f_*, s01 … s42 | the mathematical HI, to compare with your model |
| failed_now | a recorded failure is waiting for its corrective repair at ts |
| failure_next_7d, failure_next_30d | a failure_log entry occurs in (ts, ts + 7 d] or (ts, ts + 30 d] |
| next_failure_component | BATTERY / MECHANISM / CONTACTS / COMMS / CONTROLLER (empty if none later) |
| days_to_next_failure | days until the next failure (empty if none later) |
| censored_7d, censored_30d | the label horizon goes past the end of the data and no failure was seen, so the label is unknown; exclude these rows when training |

Two companion files support evaluation only; do not use them as features:
- `ar_ml_ground_truth.csv.gz`: daily hidden truth per recloser (true battery SOH, mechanism / comms / controller degradation, pollution, open failures).
- `ar_ml_asset_profiles.csv`: hidden profile per recloser.
