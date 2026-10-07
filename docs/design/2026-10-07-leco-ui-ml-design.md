# LECO Fleet, Futuristic Dashboard and Colab ML: Design

Date: 2026-10-07 · Status: approved in chat ("Yes, build it")
Builds on: [2026-10-06-ar-dashboard-hi-reset-design.md](2026-10-06-ar-dashboard-hi-reset-design.md)

## 1. Goal and decisions

**The user asked for:**
- finishing the interrupted clean-up;
- a more futuristic UI;
- LECO branches with 100 reclosers across them;
- a thorough way to train the ML model in Google Colab. Nothing is trained on this PC.

**Decisions taken with the user:**

| Topic | Decision |
|---|---|
| Branches | 7 LECO branches: Negombo, Kelaniya, Kotte, Nugegoda, Moratuwa, Kalutara, Galle |
| Live data | Replace the old 30-recloser live database with the 100-recloser LECO fleet plus 30 days of history. The old 2-year data stays in `data/ar_maintenance_history_2y.sqlite`, so the interrupted prune is no longer needed. |
| ML data | Regenerate with LECO settings: 100 reclosers × 3 years (1095 days), separate seed |
| ML live | After Colab training, the model files go into `models/`. Every 15 min the system writes an ML risk per recloser plus a daily HI-vs-ML summary, and shows both in the dashboard. |
| UI technology | Streamlit with a custom dark "control room" theme |

## 2. LECO fleet and geography
- **Branches (`config.yaml` → `areas`):** each has `name, monsoon: SW, coastal: true, temp_offset_c, diurnal_amp_c, anchor: [lat, lon], spread_km`. Approximate anchors:

  | Branch | Anchor (lat, lon) |
  |---|---|
  | Negombo | (7.209, 79.838) |
  | Kelaniya | (6.955, 79.922) |
  | Kotte | (6.890, 79.902) |
  | Nugegoda | (6.865, 79.889) |
  | Moratuwa | (6.773, 79.882) |
  | Kalutara | (6.583, 79.960) |
  | Galle | (6.053, 80.221) |

- **Fleet settings:**
  - `feeders_per_area: 3`
  - `fleet.n_assets: 100`
  - `fleet.coastal_fraction: 0.5`
  - new `fleet.coastal_distance_km: [0.2, 5.0]` and `fleet.inland_distance_km: [5.5, 15.0]`. Defaults [0.2, 5] and [10, 80] keep the old behaviour when the keys are absent.
- **Outline:** `SRI_LANKA_OUTLINE` gets 5 more coastal points: Ambalangoda, Beruwala, Panadura, Mount Lavinia and Hendala.
- **Placement for areas with an `anchor` and `coastal: true`:**
  1. Find the point on the outline nearest the anchor.
  2. Move along the coast by `uniform(-spread, +spread)` km, plus a feeder offset.
  3. Take the inland normal (the side that is inside the outline) and step inland by `distance_to_coast_km + 0.3 km`.
- **Other areas:** areas with an `anchor` but not coastal are placed radially, 3–25 km from the anchor. Legacy area names keep their old geometry.
- **Tests:**
  - every point is inside the outline (100 assets);
  - coastal points are within d + 3 km of the outline;
  - reclosers on the same feeder are closer together than reclosers in different branches;
  - each branch's reclosers have their centroid within 20 km of the branch anchor.

## 3. ML live scoring and daily summary
- **`ar_gen/ml_features.py`:** the single source of the feature recipe. The notebook embeds this file verbatim, and a test checks they are identical.
  - `FEATURE_VERSION`, `TREND_VARS`, `TREND_HOURS = 168`, `CATEGORIES` (weather, model, comms channel).
  - `build_features(df: DataFrame, step_minutes: int) -> DataFrame`. The input has `asset_id, ts, telemetry_present, v01..v42, model, comms_channel`, sorted per asset and time. The output has these features:
    - `telemetry_present`;
    - v01–v41;
    - `<v>_chg7d` (value minus the value 7 days earlier) and `<v>_mean7d` (rolling 7-day mean), with the window in rows = 168 h · 60 / step;
    - one-hot columns for every category in `CATEGORIES` (fixed list, so the columns never change).
- **Model bundle in `models/`:**
  - `ml_meta.json`: `feature_version`, `features` (ordered list), `threshold`, `calibration` (isotonic x/y arrays), `component_classes`, `metrics`, `trained_utc`, `data_until_utc`, `model_version`;
  - `model_failure30d.txt` (LightGBM text);
  - optional `model_component.txt` (LightGBM multiclass text).
- **`ar_gen/ml_scoring.py`:**
  - `load_model(models_dir) -> MLModel | None` returns None when there is no bundle, and raises `ValueError` when `feature_version` doesn't match.
  - `MLModel.predict(features) -> DataFrame` returns `p_fail_30d` (calibrated through `numpy.interp` on the isotonic arrays), `ml_healthy` (p < threshold), `top_component` and `top_component_p`.
- **New table `ar_ml_prediction_15min`:**
  - key: asset_id, ts;
  - columns: `p_fail_30d`, `ml_risk_class` (LOW < 0.5·threshold ≤ ELEVATED < threshold ≤ HIGH), `ml_healthy`, `top_component`, `top_component_p`, `model_version`, `computed_at`.
- **New table `ar_daily_summary`:**
  - key: asset_id, day (Asia/Colombo date);
  - columns: `intervals`, `avg_hi`, `min_hi`, `max_hi`, `unhealthy_hours` (HI < 60 intervals × 0.25), `avg_p_fail`, `max_p_fail`, `hi_healthy` (avg_hi ≥ 60), `ml_healthy` (max_p < threshold, NULL without a model), `agree` (NULL without a model), `updated_at`.
- **Pipeline:** `compute_range` (and therefore `update`, `maintenance_done` and `compute-variables`) runs two extra steps after writing the HI rows of a chunk:
  1. if a model is loaded, score the chunk, using variables from 8 days before the chunk for the trends, and replace the prediction rows for that chunk;
  2. recompute the daily summary rows for every local day that overlaps the chunk.
- **Commands:**
  - `ml-status`: model installed? version, threshold, metrics;
  - `ml-score [--start --end]`: re-score stored variables, under the lock.
- **Dependency:** `requirements.txt` gets `lightgbm>=4.0`. Scoring works without it, but `load_model` then reports that LightGBM is missing.

## 4. Colab notebook and guide
- **Files:**
  - `notebooks/build_colab_notebook.py` generates `notebooks/AR_ML_Colab.ipynb`, embedding `ar_gen/ml_features.py` through a `%%writefile` cell;
  - `docs/ML_COLAB_GUIDE.md`.
- **Notebook sections:**
  - **0 Setup:** pip install lightgbm, optuna and shap if missing.
  - **1 Data:** Drive mount or upload of `ar_ml_dataset_hourly.csv.gz`, with memory-light dtypes.
  - **2 Explore:** failures by component, label rate, HI distribution by label.
  - **3 Features:** shared module.
  - **4 Split:** fit / valid / test by time, with 30-day gaps; censored and failed rows dropped.
  - **5 Baseline:** logistic regression.
  - **6 LightGBM:** with Optuna. The number of trials is time-boxed and set in the `CONFIG` cell.
  - **7 Calibration:** isotonic, on the validation part.
  - **8 Threshold:** the best F1 on the validation part, or a false-alarm budget.
  - **9 Test evaluation:** ROC-AUC, PR-AUC, failures caught, warning-day distribution, false-alarm days per AR-month, per component.
  - **10 Comparison:** HI < 60 and "ML or HI".
  - **11 SHAP:** global importance plus one example recloser.
  - **12 Component model:** multiclass, trained on the positive rows.
  - **13 Daily HI vs ML:** agreement.
  - **14 Export:** zip the model trained on the fit months (with the calibration and alarm level from the later months; refitting on all data would invalidate the calibration), then download.
- **`SMOKE=True`:** a switch for a tiny run. It is used here only to prove the code runs.

## 5. Futuristic UI
- **Files:** `.streamlit/config.toml` (dark theme) and `ar_gen/ui/` (theme CSS, SVG components, map builder, data helpers). `ar_gen/dashboard_app.py` becomes the layout.
- **Look:**
  - background #050A14, panels #0B1426 with glass blur, accent cyan #00E5FF;
  - glow in the status colours, pulse animation on FAULTY and CRITICAL;
  - monospace numbers;
  - all text inserted into HTML is escaped.
- **Header:** title "LECO · AR Health Command Centre", local clock and data time, data age / collector heartbeat (green < 20 min, amber < 60, red otherwise), ML model chip.
- **Sidebar:** branch multiselect, status multiselect, asset search, map height mode (risk or ML risk).
- **Tabs:**
  1. **Command Centre:** KPI glow tiles (7 statuses, fleet average HI, ML high-risk count); dark pydeck map with columns, halos for FAULTY/CRITICAL/LOCKOUT, neon feeder paths and branch labels; alert feed for the last 24 h; branch health bars.
  2. **Fleet:** a table with ProgressColumns for HI and ML risk, plus branch cards.
  3. **Recloser:** SVG gauges for HI and ML risk; SVG radar of the 7 factors; Altair HI line with 40/60/80 bands and the ML risk line; daily-average bars; events timeline; latest V1–V42 by factor.
  4. **ML vs HI:** Altair scatter (HI against ML risk, with quadrants), daily agreement heat-map from `ar_daily_summary`, disagreement list; without a model, the guide.
  5. **Maintenance:** the existing reset form plus recent resets.
- **Verification:**
  - AppTest renders every tab with no exception, both with and without a model;
  - unit tests for the SVG helpers;
  - a headless Edge screenshot of the running page.

## 6. Rollout
1. Stop nothing (no processes are running) and delete `data/ar_maintenance.sqlite*`; the backup stays.
2. `init-db`, then `seed --days 30` (100 LECO reclosers), then `backfill`, then `validate`.
3. Start the collector and the dashboard.
4. Move the old generated ML data to `output/ml_generated_v1`, then run `generate-ml-data --assets 100 --days 1095` in the background.
5. Update the docs (README, DATA_DICTIONARY, guide), then a final independent review.

## 7. Out of scope
Training the real model on this PC; logins; offline map tiles; a mobile layout.
