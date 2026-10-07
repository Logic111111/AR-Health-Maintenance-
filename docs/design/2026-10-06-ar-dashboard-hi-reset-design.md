# 3D Recloser Dashboard and Health-Index Reset after Maintenance: Design

Date: 2026-10-06 · Status: design approved in chat, written spec awaiting review
Builds on: [2026-10-06-ar-phase1-design.md](2026-10-06-ar-phase1-design.md)

## 1. Goal and understanding

**The user asked for:**
- a 3D geographical dashboard showing every recloser with its status (faulty, good, and so on);
- when maintenance is done, a way to reset that recloser's Health Index, after which the normal 15-minute HI calculation continues.

**Decisions taken with the user:**

| Topic | Decision |
|---|---|
| Reset scope | Reset **only the repaired part**. It happens automatically for recorded corrective repairs and also from a "Maintenance done" button. |
| Locations | Sri Lanka, **simulated** positions in the 5 areas, replaceable later with real GPS from a CSV. |
| Technology | **Streamlit** app with a pydeck 3D map, started with `python main.py dashboard`. |

**Why the reset is needed:** the HI counts events in 24-hour and 30-day windows. After a repair, the pre-repair events keep the HI low for up to 30 days. Example: AR-008 had its OSM tank replaced on 2026-10-04, but its 5 failed operations still cap the HI at 45 until early November.

**Success criteria:**
1. The dashboard shows all reclosers on a 3D map of Sri Lanka, coloured by status, and refreshes by itself.
2. After a recorded repair, or a "Maintenance done" entry, the repaired part's history stops counting from that moment. The HI rises accordingly, then follows the normal calculation.
3. Recalculating any range gives the same result as calculating interval by interval, so the existing no-look-ahead guarantee still holds.

## 2. Status rules

Every recloser shows one status, checked in this order (first match wins):

| Status | Colour (RGB) | Rule at the latest interval `ts` |
|---|---|---|
| OFFLINE | grey (150,150,150) | no telemetry row in the last 60 min before the newest `ts` |
| FAULTY | dark red (139,0,0) | a `failure_log` row exists with no CORRECTIVE `maintenance_log` row linked to it by `failure_id` |
| LOCKOUT | purple (128,0,128) | latest telemetry `switch_state = 'LOCKOUT'` |
| CRITICAL | red (220,40,40) | HI < 40 |
| POOR | orange (245,140,30) | 40 ≤ HI < 60 |
| FAIR | yellow (240,200,40) | 60 ≤ HI < 80 |
| GOOD | green (40,170,80) | HI ≥ 80 |

The status logic is a pure function in `ar_gen/dashboard_data.py`, so it can be tested without Streamlit.

## 3. Recloser locations

**New table `asset_location`:**

| column | type | notes |
|---|---|---|
| asset_id | VARCHAR(16), PK, FK to asset_master | |
| latitude | DOUBLE | |
| longitude | DOUBLE | |
| source | VARCHAR(16) | `SIMULATED` or `GPS` |

**Simulated placement** (`ar_gen/locations.py`) is deterministic from the seed and the asset index:

| Area | Anchor (lat, lon) | Coast direction | Inland direction |
|---|---|---|---|
| Western-Coastal | Colombo (6.93, 79.86) | north–south | east |
| Southern-Coastal | Galle (6.05, 80.22) | east–west | north |
| Eastern-Coastal | Batticaloa (7.72, 81.70) | north–south | west |
| Central-Hills | Kandy (7.29, 80.63) | n/a (inland) | n/a |
| NorthCentral-Dry | Anuradhapura (8.31, 80.40) | n/a (inland) | n/a |

- **Coastal areas (revised after review):** a point is chosen along a simplified real coastline for the area (Bentota–Chilaw, Hikkaduwa–Tangalle, Pottuvil–Vakarai), and the recloser is moved inland along the normal by `distance_to_coast_km` + 0.3 km. All points fall inside a simplified Sri Lanka outline, so none lands in the sea. (The original design used a straight coast axis.) The inland offset equals the recloser's `distance_to_coast_km`, and the point is also kept inside a Sri Lanka bounding box (lat 5.9–9.9, lon 79.6–81.95).
- **Inland areas:** each feeder runs in its own compass direction from the anchor, and the reclosers sit 3–25 km out along it.
- **Real GPS:** `python main.py set-locations --csv file.csv` (columns `asset_id, latitude, longitude`) overwrites positions and sets `source = 'GPS'`.

Missing locations are created by `init-db`, `seed`, `set-locations` and on dashboard start. Existing GPS rows are never overwritten by simulated ones.

## 4. HI reset after maintenance

### 4.1 Reset events
A reset is a pair (asset, part group, time `R`). The **part groups** are BATTERY, SWITCHING, COMMS and CONTROLLER. Resets come from two places:

1. **Automatic:** every `maintenance_log` row with `maint_type = 'CORRECTIVE'` resets its component's group at its `ts`:

   | maintenance component | group reset |
   |---|---|
   | BATTERY | BATTERY |
   | MECHANISM, CONTACTS | SWITCHING |
   | COMMS | COMMS |
   | CONTROLLER | CONTROLLER |

2. **Manual:** a new table `hi_reset_log`:

   | column | type |
   |---|---|
   | reset_id | autoincrement PK |
   | ts | DATETIME(3), the time of the maintenance |
   | asset_id | FK |
   | groups | VARCHAR(64), comma list of groups or `ALL` |
   | note | VARCHAR(255) |
   | created_at | DATETIME(3) |

   It is written by the dashboard form and by `python main.py maintenance-done --asset AR-008 --part SWITCHING [--at ...] [--note ...]`. It is kept separate from `maintenance_log`, so there are no ID clashes with the simulator's IDs and crash rollback is unaffected.

### 4.2 Effect on the variables
For time `t`, let `R_g(t)` be the latest reset of group `g` with `R ≤ t` (minus infinity if none). Each variable in group `g` uses the window **(max(t − W, R_g(t)), t]** instead of (t − W, t]:

| Group | Variables restarted by the reset |
|---|---|
| BATTERY | V2, V6, V7 |
| SWITCHING | V22, V23, V24, V25, V26, V27 |
| COMMS | V28, V29, V30, V31, V32 |
| CONTROLLER | V33, V34 |
| any reset (all groups) | V39 days since maintenance = t − max(last_maintenance_date history, latest reset of any group) |

- **V28 availability:** expected polls = the number of 15-min grid points in (max(t − 24 h, R, asset_start), t]. When that is 0 (t exactly at a reset), V28 is NULL, not 0 %. This was changed after review: the earlier formula gave 0 % at a boundary reset and overstated availability after a mid-interval reset.
- **Already tied to repairs, unchanged:**
  - V3 and V9 (count only the current battery);
  - V4 and V8 (battery install date);
  - V11 and V21 (counters reset by the OSM swap);
  - V35 (firmware history).
- **Never reset:** V5 and V10 (AC supply), V12–V20 (line faults), V1, V37, V38, V40–V42 (snapshots, age and environment).

**Exactness:** event counts use exact timestamps, i.e. the count of events with `max(t − W, R) < ts ≤ t`, done with sorted arrays and `searchsorted`. Without resets this gives exactly the current bucket counts. Telemetry statistics (V2 episode starts, V6 minimum, V28 sum, V31 mean) are computed within **reset segments**: a new segment starts at the first telemetry row after each reset, and a rolling window cannot cross a segment start.

**No look-ahead:** a reset at `R` only affects `t ≥ R`, and only reset rows with `ts ≤ t` are read.

### 4.3 Recalculation after a manual reset
`maintenance-done` (CLI and dashboard) does three things:
1. takes the run lock (the same lock file as the collector, which holds it for about 10 s per run; waits up to 60 s, retrying every 5 s);
2. inserts the `hi_reset_log` row;
3. runs `compute_range(R − 15 min, last collected)`, so V1–V42 and the HI are rewritten from the reset onwards.

All three happen under the lock, so a reset is never saved without its recalculation.

Rules added after review:
- **End of the recalculation:** it is read again inside the lock, so an interval the collector added while the reset waited is covered.
- **Failed recalculation:** the reset row is deleted and the error reported ("maintenance not saved").
- **Maintenance time:** may be any moment up to now. If it is after the latest data, nothing is recalculated, and the next collection applies it.
- **Asset ids:** stored in canonical form (`AR-008`), whatever case was typed.
- **Recalculation commands:** `compute-variables` and `compute-hi` also take the lock.

The collector then continues normally.

**One-time rollout step:** `python main.py compute-variables` recalculates the stored 2-year history (about 10 min), so the automatic corrective resets also apply to the past.

## 5. Dashboard (`python main.py dashboard`, file `ar_gen/dashboard_app.py`)

**Page layout:**
- **Top bar:**
  - title;
  - data up to `<ts>` in Colombo time;
  - one count tile per status;
  - auto-refresh every 60 s (`st.fragment(run_every=60)`).
- **3D map (pydeck):**
  - background: Carto light basemap, no token needed; the browser needs internet;
  - view: pitch 50°, centred on Sri Lanka;
  - `ColumnLayer` per recloser: radius about 1.5 km, height = (100 − HI) × 400 m, plus a 2 km minimum so healthy ones stay visible, colour = status;
  - `PathLayer` for each feeder, joining its reclosers in index order;
  - tooltip: asset, status, HI, class, worst variables, battery V, AC, comms, switch, data time.
- **Fleet table**, worst first: asset, status, HI, class, worst variables, area, feeder, open failure. A select box chooses a recloser.
- **Detail panel for the selected recloser:**
  - HI over the last 30 days: the 15-min line plus the daily average (local days);
  - the 7 factor scores (bar chart);
  - recent events in the last 30 days: failures, maintenance, manual resets, lockouts;
  - the **"Maintenance done" form**: part groups (multi-select, or All), date/time (default now), note → **Save & recalculate**, then a confirmation with the new HI.

**Data access:**
- `ar_gen/dashboard_data.py` holds plain functions that return DataFrames: fleet snapshot, status, locations, HI history, events.
- They read through the existing SQLAlchemy engine, so SQLite and MySQL both work, and are cached for 30 s with `st.cache_data`.

**Dependency:** add `streamlit>=1.37` to requirements.txt (pydeck comes with it).

**Starting:** `python main.py dashboard` runs `streamlit run ar_gen/dashboard_app.py` with the project root on the path, and the browser opens at http://localhost:8501.

## 6. Error handling
- **Empty database or no HI rows:** the page shows "No data yet - run backfill / scheduler".
- **Reclosers with no location:** locations are created on start; if that fails, they are listed under the map as "not placed".
- **Run lock busy for 60 s:** nothing is saved; the message says "collector busy - please press Save again".
- **Invalid input:** an unknown asset, part or future date is refused with a message.

## 7. Testing
- `tests/test_reset.py`:
  - events before a reset don't count; events after it do;
  - a part that wasn't reset is unaffected;
  - a mid-interval reset is exact;
  - automatic resets come from CORRECTIVE maintenance rows;
  - `ALL` resets every group;
  - V28 expected polls restart;
  - V39 restarts;
  - incremental equals full recalculation, with resets present.
- `tests/test_variables.py` and `tests/test_mlgen.py` must stay green: no resets means identical results.
- `tests/test_locations.py`: deterministic; inside the Sri Lanka box; coastal points within `distance_to_coast_km` + 3 km of the area's coast line; a GPS CSV overrides and is kept.
- `tests/test_dashboard_data.py`: status priority (OFFLINE > FAULTY > LOCKOUT > HI class) and the fleet snapshot on a small pipeline database.
- `maintenance-done` end to end on SQLite: the HI rises at the reset and later rows follow the normal calculation.
- Manual check: start the dashboard and look at the map, tooltips, table, detail panel and reset form.

## 8. Out of scope (YAGNI)
- user logins or permissions;
- editing maintenance records;
- showing ML predictions on the map (possible later, once the model runs in real time);
- offline map tiles;
- mobile layout.
