# LECO Auto Recloser Health: real-time data getter, V1–V42, Health Index, 3D dashboard and ML

Every 15 minutes this project:

1. **Collects** one interval of raw data per auto recloser (AR), shaped like a real NOJA OSM15-310 / OSM27-310 with an RC-10/RC-15 cubicle (NOJA-5002-14 manual): telemetry poll, Close/Open log, fault and event logs, battery tests, comms events, lifetime counters and maintenance records.
2. **Calculates the final data**: the 42 Excel variables V1–V42.
3. **Calculates the Health Index (HI)** with a mathematical model, giving 0–100, GOOD / FAIR / POOR / CRITICAL and healthy / not healthy.
4. **Stores** everything in MySQL / MariaDB (or SQLite) and can export an ML-ready dataset with failure labels.

The fleet is **100 NOJA reclosers across 7 LECO branches**: Negombo, Kelaniya, Kotte, Nugegoda, Moratuwa, Kalutara and Galle. The branches are set in `config.yaml → areas`.

On top of that:
- a **futuristic 3D dashboard** (`python main.py dashboard`) shows the fleet live;
- **maintenance resets** restart the repaired part's Health Index;
- a **Windows service** keeps the real-time data generation running in the background, from boot (section 5);
- **machine learning is switched off for now** (`config.yaml` → `ml: enabled: false`), so the collector and the
  dashboard show the Health Index only. The models stay in the project, and one setting turns them back on
  (section 7):
  - a **LightGBM model trained in Google Colab** (`docs/ML_COLAB_GUIDE.md`);
  - a **deep-learning model** (GRU / TCN / Transformer over the last 14 days, trained on a Colab GPU).

The data source is a deterministic, physics-based simulator of each recloser (`SimulatedNojaSource`). A real source (DNP3 / SCADA historian / CMS log import) can replace it later through the `RecloserSource` interface (`ar_gen/sources/base.py`) without changing the variable or HI code.

- Design: [docs/design/2026-10-06-ar-phase1-design.md](docs/design/2026-10-06-ar-phase1-design.md)
- Every table, column and formula: [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
- Train the ML model in Google Colab: [docs/ML_COLAB_GUIDE.md](docs/ML_COLAB_GUIDE.md)

---

## 1. Install

Requires Python 3.11+ (tested with 3.13).

```powershell
cd "D:\AR Preventive Analyze"
python -m pip install -r requirements.txt
```

## 2. Database (MariaDB 10.6 / MySQL 8)

In HeidiSQL, connected as root, run:

```sql
CREATE DATABASE IF NOT EXISTS ar_maintenance CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'ar_gen'@'localhost' IDENTIFIED BY 'choose-a-password';
CREATE USER IF NOT EXISTS 'ar_gen'@'127.0.0.1' IDENTIFIED BY 'choose-a-password';
GRANT ALL PRIVILEGES ON ar_maintenance.* TO 'ar_gen'@'localhost';
GRANT ALL PRIVILEGES ON ar_maintenance.* TO 'ar_gen'@'127.0.0.1';
FLUSH PRIVILEGES;
```

Then copy `.env.example` to `.env` and put your password in it. `.env` is never committed.

```
DATABASE_URL=mysql+pymysql://ar_gen:choose-a-password@127.0.0.1:3306/ar_maintenance
```

If the password contains `@ : / ? # %`, URL-encode those characters (for example `@` becomes `%40`). Without a `.env` the project uses SQLite at `data/ar_maintenance.sqlite`.

## 3. Quick start

```powershell
python main.py init-db          # create tables and indexes
python main.py seed --days 30   # 100 LECO reclosers, 30 days of history (hidden profiles -> output/asset_profiles.csv)
python main.py backfill         # raw -> V1-V42 -> HI -> daily summary  (about 8 min for 30 days)
python main.py validate         # consistency checks; must print "Validation PASSED"
python main.py status           # latest HI of every recloser
python main.py service install # admin PowerShell, once: the real-time generator as a Windows service (section 5)
python main.py dashboard        # the 3D command centre at http://localhost:8501
```

The live database keeps recent data only. Old history for ML training is produced separately with
`generate-ml-data` (no database), and `prune-history --keep-days 30` trims an existing database after a backup.

For a fast trial, run `python main.py seed --assets 3 --days 7` followed by `python main.py backfill`.

### Data already generated into SQLite, now moving to MySQL

Create `.env` (section 2), then run:

```powershell
python main.py copy-db --from sqlite:///data/ar_maintenance.sqlite
python main.py validate
```

Everything continues from there. If the configured database does not contain the seeded fleet, the getter refuses to run rather than write orphan rows. You can also regenerate the identical dataset directly in MySQL with `init-db --reset`, `seed --start <same first day>` and `backfill`, because generation is deterministic.

## 4. Commands

| command | what it does |
|---|---|
| `init-db [--reset] [--yes]` | create the database, tables and indexes. `--reset` drops everything and deletes the simulator state |
| `seed [--assets N] [--start YYYY-MM-DD \| --days N]` | create the fleet and its hidden degradation profiles |
| `backfill [--end DATE] [--no-compute]` | generate history up to now (or `--end`), committed per day, then V1–V42 and HI |
| `run-once` | collect the interval(s) that just ended, then update V1–V42 and HI. Missed intervals are filled automatically |
| `service install \| start \| stop \| restart \| status \| remove` | the real-time generator as a Windows service (section 5). All but `status` need an admin PowerShell |
| `package-service [--out DIR]` | zip of **just the generator service** to send to someone: `dist/LECO_AR_Generator_Service_<date>.zip` (no dashboard, ML, notebooks or data), with install steps in its README |
| `service run` (alias `scheduler`) | the real-time generator in this console: updates at :00, :15, :30 and :45 (+10 s), catches up on start; Ctrl+C stops it |
| `status` | latest HI, class, healthy flag, worst variables and factor scores per recloser |
| `dashboard [--port 8501] [--lan]` | open the **3D recloser health map** in your browser (section 5b). It listens on this PC only; `--lan` also allows other computers on your network, and anyone there can save resets |
| `maintenance-done --asset AR-008 --part SWITCHING [--part ...] [--at "YYYY-MM-DD HH:MM"] [--note ...]` | record maintenance and reset the repaired part's Health Index, then recalculate. Parts: BATTERY, SWITCHING, COMMS, CONTROLLER, ALL; `--at` is local time, default the latest interval; any time up to now is accepted |
| `ml-status`, `dl-status` | whether a trained LightGBM / deep-learning model is installed (version, alarm level, test results). Only while ML is switched on (section 7) |
| `ml-score`, `dl-score [--start --end]` | score stored V1–V42 rows with the installed model. Only while ML is switched on |
| `prune-history [--keep-days 30] [--backup FILE \| --no-backup]` | deletes older data, after a full backup by default, and keeps what the 30-day calculations still need |
| `set-locations --csv file.csv` | load real GPS positions (`asset_id,latitude,longitude`), replacing the simulated ones |
| `validate` | gaps, ranges, counters vs operations, fault ↔ operation sequences, alarm pairing, repairs, coverage, and HI before failures |
| `export-dataset [--out DIR]` | ML dataset `output/ml/ar_ml_dataset.csv.gz` with labels, plus ground truth and profiles |
| `compute-variables [--start --end]` | recompute V1–V42 and HI for a range |
| `compute-hi [--start --end]` | re-score stored variables after changing HI weights or thresholds in `config.yaml` |
| `copy-db --from URL` | copy all data from another database into `DATABASE_URL` |
| `generate-ml-data [--assets 100] [--days 730] [--seed N]` | generate an ML training dataset **directly, without the database**: a separate simulated fleet, written to `output/ml_generated/` |

Logs go to the console and to `logs/ar_gen.log` (rotating); the Windows service writes `logs/ar_service.log`. A lock file (`state/ar_gen.lock`) stops two runs from overlapping, for example the service and a manual `run-once`.

## 5. Real-time data generator service

The real-time generator collects every interval and updates V1–V42 and the Health Index:
- It runs 10 s after :00, :15, :30 and :45.
- After downtime it first catches up, one day per step, so the data stays continuous.

On Windows it runs as the service **LECO AR Data Generator** (`LecoArDataGenerator`):
- it starts at boot without anyone logged in (delayed automatic start, so MariaDB is up first);
- Windows restarts it 60 s after a crash;
- it keeps generating after the PowerShell window is closed.

**Install it once in an admin PowerShell** (Start menu → right-click *Windows PowerShell* → *Run as administrator*):

```powershell
cd "D:\AR Preventive Analyze"
python main.py service install     # test start, register, start now
python main.py service status      # Windows service state + the generator's last run (any PowerShell)
```

Before registering, `install` makes a test start of the service exactly as Windows will run it.

| command | what it does |
|---|---|
| `service install [--no-start]` | register (or update) and start the service. Run it again after moving the project folder or Python |
| `service start` / `stop` / `restart` | control it, as in `services.msc`. A stop waits for the update step that is running |
| `service status` | `RUNNING` / `STOPPED` / `NOT INSTALLED`, plus the generator's last run, data time, errors and next run (from `state/generator_status.json`) |
| `service remove` | stop and unregister it |
| `service run` | run the generator in this console instead (no admin needed); Ctrl+C stops it. `scheduler` is the same |

Notes:
- **Account.** The service runs `C:\Python313\python.exe ar_service.py` as the **LocalSystem** account.
  - Your packages (pandas, SQLAlchemy, …) are in your own user site-packages. `install` puts that folder on the
    service's command line, so after `pip install`, reinstalling the service is not needed.
  - The database (`.env` or SQLite), `state/` and `logs/` are the same as for your own commands.
- **Log.** `logs\ar_service.log`. Start and stop are also written to the Windows **Event Viewer** (Windows Logs →
  Application).
- **Old Task Scheduler task.** If you created the "AR data getter" task from an earlier version, delete it with
  `schtasks /Delete /TN "AR data getter" /F`. `install` reminds you. The lock file stops both from writing at the
  same time anyway.
- **Without the service.** Run `python main.py service run` in a terminal; it stops when the terminal closes.
- **Other systems.** On Linux, run `python main.py service run` under systemd.
- **Sending the service to someone:** `python main.py package-service` writes a ~110 kB zip to `dist/`. The receiver
  unzips it and follows its `README.md` (from `docs/SERVICE_PACKAGE_README.md`): pip install, `init-db`, `seed`,
  `backfill`, `service install`. Don't run `service install` from an unzipped copy on your own PC: it would point
  your existing service at that copy.

## 5b. 3D dashboard and resetting the HI after maintenance

```powershell
python main.py dashboard          # opens http://localhost:8501
```

**What the page shows:**
- **A 3D map of Sri Lanka.** Each recloser is a column:
  - **height = risk (100 − HI)**, so problem reclosers stand out;
  - **colour = status**, in this priority order:

    | Status | Colour | Meaning |
    |---|---|---|
    | OFFLINE | grey | no data for 60 min |
    | FAULTY | dark red | failure waiting for repair |
    | LOCKOUT | purple | switch locked out after a fault |
    | CRITICAL | red | HI < 40 |
    | POOR | orange | HI 40–60 |
    | FAIR | yellow | HI 60–80 |
    | GOOD | green | HI ≥ 80 |

  - Hovering over a column shows the HI, worst variables, battery, AC, comms and switch state.
  - Thin lines join the reclosers on the same feeder.
- **Above the map:** one count tile per status. The page refreshes every 60 s.
- **Below the map:** the fleet table, worst first.
- **Detail panel** for the chosen recloser:
  - the HI over the last 30 days, with the **daily average**;
  - the factor scores;
  - recent failures, repairs, resets and lockouts;
  - the **Maintenance done** form.

**Maintenance done = HI reset of the repaired part.** From the maintenance time onward, events from before the repair stop counting for that part. Everything else still counts. The normal 15-minute calculation then continues, so if the problem is really still there the HI drops again.

| Part | Variables that restart |
|---|---|
| Battery | V2, V6, V7 (V3, V4, V8, V9 already follow the battery) |
| SWITCHING (OSM mechanism / contacts) | V22–V27 (V11, V21 reset with a new tank) |
| Comms | V28–V32 |
| Controller | V33, V34 |
| Any reset | V39 days since maintenance |

- **Never reset:** faults (V12–V20), AC supply (V5, V10), age and environment.
- **Automatic resets:** every CORRECTIVE repair in `maintenance_log` resets its part.
- **Manual resets:** the dashboard form or `maintenance-done` adds a row to `hi_reset_log` and recalculates from that moment, waiting for the collector if it is busy.

**Map positions:** reclosers are placed along their feeders in the 5 areas (`asset_location`, source SIMULATED). Load real positions with `python main.py set-locations --csv gps.csv`, where the CSV has the columns `asset_id,latitude,longitude`.

The map background comes from an online map service, so the browser needs internet.

## 6. HeidiSQL queries

```sql
-- Latest Health Index of every recloser (worst first)
SELECT h.asset_id, h.ts, h.health_index, h.health_class, h.is_healthy, h.cap_applied, h.worst_variables
FROM ar_health_index_15min h
JOIN (SELECT asset_id, MAX(ts) ts FROM ar_health_index_15min GROUP BY asset_id) m USING (asset_id, ts)
ORDER BY h.health_index;

-- Reclosers that are NOT healthy right now
SELECT h.asset_id, h.health_index, h.health_class, h.worst_variables, h.f_battery, h.f_switching, h.f_comms
FROM ar_health_index_15min h
JOIN (SELECT asset_id, MAX(ts) ts FROM ar_health_index_15min GROUP BY asset_id) m USING (asset_id, ts)
WHERE h.is_healthy = 0;

-- Daily HI trend for one recloser
SELECT DATE(ts) AS day, ROUND(AVG(health_index), 1) AS hi, MIN(health_index) AS hi_min
FROM ar_health_index_15min WHERE asset_id = 'AR-005'
GROUP BY DATE(ts) ORDER BY day;

-- Latest V1-V42 row of one recloser
SELECT * FROM ar_variables_15min WHERE asset_id = 'AR-005' ORDER BY ts DESC LIMIT 1;

-- Battery behaviour over the last 24 h
SELECT ts, battery_voltage, battery_current, battery_charge_pct, ac_supply_on, battery_temp_c
FROM telemetry_15min WHERE asset_id = 'AR-005' AND ts > NOW() - INTERVAL 1 DAY ORDER BY ts;

-- Failures, the repair and the HI during the 7 days before each failure
SELECT f.failure_id, f.asset_id, f.ts, f.component, f.description, m.ts AS repaired_at,
       (SELECT ROUND(AVG(h.health_index), 1) FROM ar_health_index_15min h
         WHERE h.asset_id = f.asset_id AND h.ts BETWEEN f.ts - INTERVAL 7 DAY AND f.ts) AS hi_7d_before
FROM failure_log f LEFT JOIN maintenance_log m ON m.failure_id = f.failure_id
ORDER BY f.ts;

-- Faults per month and outcome
SELECT DATE_FORMAT(start_ts, '%Y-%m') AS month, outcome, COUNT(*) AS n
FROM fault_events GROUP BY month, outcome ORDER BY month;

-- A fault and its trip / reclose sequence
SELECT o.ts, o.op_type, o.source, o.ar_state, o.trip_ia, o.trip_ib, o.trip_ic, o.operating_time_ms, o.result
FROM co_operations o WHERE o.fault_id = 100 ORDER BY o.ts;

-- Asset history: what maintenance changed
SELECT ts, asset_id, component, maint_type, description, details FROM maintenance_log ORDER BY ts;
```

## 7. Machine learning (switched off for now) and the ML dataset

**The ML switch.** In `config.yaml`:

```yaml
ml:
  enabled: false
```

With `false` (the current setting):
- the collector does no LightGBM or deep-learning scoring;
- the dashboard has no ML tabs, gauges, columns, KPIs, map option or tooltip lines;
- `ml-status`, `ml-score`, `dl-status` and `dl-score` only say that ML is switched off.

The notebooks, the model files in `models/` and `models/dl/` and the predictions already stored are kept.

To bring ML back:
1. Set `enabled: true`.
2. Run `python main.py service restart` (admin) and restart the dashboard.
3. Run `python main.py ml-score` / `dl-score` once to fill the gap.

The environment variable `AR_ML_ENABLED=1` / `0` overrides the setting. The switch does not change the simulated
data.

### ML dataset

```powershell
python main.py export-dataset
```

**More training data without the database:**

```powershell
python main.py generate-ml-data --assets 100 --days 730
```

This simulates a separate fleet in memory (seed = config seed + 1, so it is independent of the live reclosers) and writes the same columns as `export-dataset` to `output/ml_generated/`:
- `ar_ml_dataset.csv.gz`: every 15 min;
- `ar_ml_dataset_hourly.csv.gz`: hourly, smaller, best for Colab;
- the ground truth and profiles files.

Nothing is written to the database or the live simulator state. A test checks that it produces exactly the same rows as the database route.

**Training in Google Colab (recommended):** `notebooks/AR_ML_Colab.ipynb`, with step-by-step instructions in **[docs/ML_COLAB_GUIDE.md](docs/ML_COLAB_GUIDE.md)**.

The notebook:
- tunes LightGBM with Optuna;
- calibrates the probabilities;
- chooses the alarm level by a false-alarm budget;
- compares ML with the HI, explains predictions with SHAP;
- adds a "which part will fail" model;
- exports `ar_ml_model.zip`.

Unzip it into `models/` and run `python main.py ml-score` (it scores all stored data; `--start YYYY-MM-DD` limits it). The collector then stores the ML risk every 15 minutes (`ar_ml_prediction_15min`) together with a daily HI-vs-ML summary (`ar_daily_summary`), and the dashboard shows both.

The notebook embeds the project's feature code (`ar_gen/ml_features.py`), so training and live scoring always build the same inputs.

**Deep-learning model (Colab GPU, works next to LightGBM):** `notebooks/AR_DL_Colab.ipynb`, built by
`notebooks/build_dl_notebook.py`. The notebook:
- reads the last 14 days of each recloser as a sequence;
- trains GRU, TCN and Transformer networks and keeps the best;
- compares it with LightGBM and the HI on the same test months;
- exports `ar_dl_model.zip`.

To use it:
1. Unzip `ar_dl_model.zip` into `models/dl/`.
2. Install PyTorch (CPU) once, if it is missing.
3. Run `python main.py dl-status`, then `python main.py dl-score`.

The collector then stores the network's risk every 15 minutes in `ar_dl_prediction_15min`. The dashboard's
**Deep Learning** tab shows it next to LightGBM and the HI. The network code is `ar_gen/dl_model.py`, which the
notebook embeds verbatim; scoring is `ar_gen/dl_scoring.py`. Step-by-step: the deep-learning section of
[docs/ML_COLAB_GUIDE.md](docs/ML_COLAB_GUIDE.md).

The older notebook `notebooks/AR_failure_prediction.ipynb` (with its `_results` copy) is a simpler first version, kept for reference.

This writes the following files to `output/ml/`:
- `ar_ml_dataset.csv.gz`: one row per AR per 15 min: v01–v42, s01–s42, factor scores, HI and class, plus the labels `failure_next_7d`, `failure_next_30d`, `next_failure_component`, `days_to_next_failure`, `failed_now` and `censored_7d/30d`. Exclude censored rows when training.
- `ar_ml_ground_truth.csv.gz`: hidden daily truth (true battery SOH, mechanism / comms / controller degradation). Use it only for evaluation.
- `ar_ml_asset_profiles.csv`: the hidden profile of each AR.

Load it in a notebook:

```python
import pandas as pd
df = pd.read_csv("ar_ml_dataset.csv.gz", parse_dates=["ts"])
train = df[~df["censored_30d"] & ~df["failed_now"]]
X = train.filter(regex=r"^v\d\d_").drop(columns=["v42_weather"]).join(pd.get_dummies(train["v42_weather"]))
y = train["failure_next_30d"]
```

To avoid leakage, split train and test by **asset** or by **time**, not at random.

## 8. Configuration

All settings are in `config.yaml`:
- seed and fleet size;
- climate, load, outages;
- battery and UPS, protection, comms, controller and maintenance rates;
- V1–V42 windows;
- HI weights, thresholds, caps and classes.

`ml: enabled` switches the machine-learning models on or off (section 7).

Environment variables override `DATABASE_URL`, `AR_CONFIG`, `AR_STATE_DIR`, `AR_OUTPUT_DIR`, `AR_LOG_DIR`, `AR_MODELS_DIR`, `AR_DL_MODELS_DIR` and `AR_ML_ENABLED`, which suits cloud or container runs.

- Changing **HI** settings: run `python main.py compute-hi`.
- Changing **variable** windows: run `python main.py compute-variables`.
- Changing **simulation** settings: affects only data generated from then on; the log warns about it. For a clean dataset, run `init-db --reset`, then `seed` and `backfill`.

## 9. How realistic the data is, and its limits

- **Physics-based and deterministic.** Battery capacity fades with temperature-weighted age and cycling; internal resistance is R0 / SOH^2.5; the UPS follows the manual's thresholds (§4.8). Mechanism wear lengthens opening and closing times and causes Trip/Close Request Fail. Contact wear follows the manual's breaking-duty curve (§2.1.4). Weather drives faults, outages and comms fades.
- **Failures and repairs.** Hidden thresholds generate failures; corrective maintenance follows 1–21 days later (new battery, OSM tank swap with counter reset, modem or module replacement). Annual preventive inspections happen as well.
- **The same seed and config give identical data.** One long backfill equals thousands of 15-minute runs. Crash recovery deletes partially written rows and regenerates them identically.
- **Telemetry is a poll at the interval end** (instantaneous values). Short events between polls appear in the event logs, not in telemetry.
- **Faults that start just before a poll** are written when they finish, possibly one interval later. Real-time V-values for that interval can therefore differ slightly from a later `compute-variables`.
- **Variables raw RC data cannot give directly:** V3 (estimated SOH), V12 (lightning proxy), V31 (two scales), V35 (needs the release table) and V36 (external RS485 device). See DATA_DICTIONARY.md, notes A–E.
- **This is simulated data.** Use it to build and test the pipeline and ML workflow; validate any model on real recloser data before trusting it.

## 10. Tests

```powershell
python -m pytest
```

The tests cover:
- config;
- physics (battery, contact wear, protection timing);
- database schema;
- the HI model;
- the simulator: telemetry ranges, fault ↔ operation consistency, counters vs logged operations, alarm pairing, battery failure and repair, determinism, split run = single run;
- V1–V42 (each variable, no look-ahead, incremental = full);
- the end-to-end pipeline: validate passes, idempotent re-run, interval-by-interval = backfill, crash recovery, copy-db;
- the real-time generator (catch-up, errors, status) and the Windows service set-up (test start without your
  user site-packages, recovery settings);
- the ML switch: with it off, nothing ML is scored or shown even when models are installed.

## 11. Project layout

```
main.py                     CLI entry point
ar_service.py               Windows service host (started by Windows; use `python main.py service ...`)
config.yaml                 every tunable value
ar_gen/
  cli.py, pipeline.py       commands; collect -> store -> variables -> HI
  generator.py, service.py  real-time generator loop; Windows service install / start / stop / status
  sources/                  RecloserSource interface + SimulatedNojaSource
  sim/                      simulator: fleet, environment, load, UPS/battery, protection, switching,
                            comms, controller, maintenance, engine
  variables.py              V1-V42 calculator (pure pandas/numpy)
  health_index.py           HI mathematical model
  dataset.py, validate.py   ML export, data validation
  db/                       schema, engine, reader, writer
tests/                      pytest suite
state/                      simulator state (hidden truth) - not in the DB
output/                     hidden profiles, ground truth, ML exports
logs/                       ar_gen.log, ar_service.log
```
