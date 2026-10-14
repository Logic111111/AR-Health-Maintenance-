"""Build notebooks/AR_ML_Colab.ipynb (run: python notebooks/build_colab_notebook.py).

The notebook embeds ar_gen/ml_features.py verbatim, so the model trained in Google Colab and
the live scorer on the PC build exactly the same features.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
FEATURES_SRC = ROOT / "ar_gen" / "ml_features.py"

# Cells shared with the deep-learning notebook (build_dl_notebook.py), so both notebooks load the data,
# split the months and judge alarms in exactly the same way.
FIND_DATA_CODE = """
candidates = [os.environ.get("AR_ML_DATA", ""), DATA_FILE, f"/content/drive/MyDrive/{DRIVE_FOLDER}/{DATA_FILE}",
              f"/content/drive/MyDrive/{DATA_FILE}", f"/content/{DATA_FILE}",
              os.path.join("..", "output", "ml_generated", DATA_FILE)]
path = next((p for p in candidates if p and os.path.exists(p)), None)
if path is None and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if path is None:
        from google.colab import files
        print(f"{DATA_FILE} is not in Drive/{DRIVE_FOLDER} - choose it from your computer instead")
        path = next(iter(files.upload()))
if path is None:
    raise FileNotFoundError(f"Put {DATA_FILE} in Google Drive/{DRIVE_FOLDER}, next to this notebook, or set AR_ML_DATA")
print("Using", path)
"""

LOAD_DATA_CODE = """
t0 = time.time()
BOOL_COLS = ["telemetry_present", "is_healthy", "failed_now", "failure_next_7d", "failure_next_30d",
             "censored_7d", "censored_30d"]
parts, keep_ids = [], None
for chunk in pd.read_csv(path, chunksize=300_000, parse_dates=["ts"], low_memory=False):
    chunk = chunk[chunk["ts"].dt.minute == 0]                  # one row per recloser per hour
    if SMOKE:                                                  # tiny run: only 15 reclosers
        keep_ids = keep_ids or sorted(chunk["asset_id"].unique())[:15]
        chunk = chunk[chunk["asset_id"].isin(keep_ids)]
    floats = chunk.select_dtypes("float64").columns.drop("days_to_next_failure", errors="ignore")
    chunk[floats] = chunk[floats].astype("float32")            # halves the memory
    parts.append(chunk)
df = pd.concat(parts, ignore_index=True).sort_values(["asset_id", "ts"]).reset_index(drop=True)
del parts
for c in BOOL_COLS:
    df[c] = df[c].astype(str).str.lower().eq("true")
print(f"{len(df):,} hourly rows | {df['asset_id'].nunique()} reclosers | "
      f"{df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d} (UTC) | loaded in {time.time() - t0:.0f} s")
"""

FAILURES_CODE = """
f = df.loc[df["days_to_next_failure"].notna(), ["asset_id", "ts", "days_to_next_failure", "next_failure_component"]].copy()
f["failure_time"] = f["ts"] + pd.to_timedelta(f["days_to_next_failure"].astype(float), unit="D")
f = f.sort_values(["asset_id", "failure_time"])
new = (f["asset_id"] != f["asset_id"].shift()) | (f["failure_time"].diff() > pd.Timedelta(hours=2))
failures = (f.loc[new, ["asset_id", "failure_time", "next_failure_component"]]
            .rename(columns={"next_failure_component": "component"}).reset_index(drop=True))
failures["failure_time"] = failures["failure_time"].dt.round("min")
print(f"{len(failures)} failures in {failures['asset_id'].nunique()} reclosers | "
      f"rows labelled 'fails within 30 days': {df['failure_next_30d'].mean():.1%}")
display(failures["component"].value_counts().rename_axis("component").to_frame("failures"))
"""

FEATURES_CODE = """
import ar_ml_features
importlib.reload(ar_ml_features)
from ar_ml_features import FEATURE_VERSION, build_features, feature_names

t0 = time.time()
X_all = build_features(df, step_minutes=60).astype("float32")
y_all = df["failure_next_30d"].astype(int).to_numpy()
FEATURES = feature_names()
print(f"feature version {FEATURE_VERSION}: {len(FEATURES)} inputs for {len(X_all):,} rows in {time.time() - t0:.0f} s")
"""

SPLIT_CODE = """
def safe_ap(y, s):
    return average_precision_score(y, s) if 0 < np.sum(y) < len(y) else float("nan")

def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < np.sum(y) < len(y) else float("nan")

usable = (~df["censored_30d"] & ~df["failed_now"]).to_numpy()
t_first, t_last = df.loc[usable, "ts"].min(), df.loc[usable, "ts"].max()
GAP = pd.Timedelta(days=HORIZON_DAYS)
TEST_START = (t_first + (t_last - t_first) * 0.65).normalize()
VALID_START = TEST_START - GAP - pd.Timedelta(days=90)
ts = df["ts"]
fit_mask = usable & (ts < VALID_START - GAP).to_numpy()
valid_mask = usable & ((ts >= VALID_START) & (ts < TEST_START - GAP)).to_numpy()
test_mask = usable & (ts >= TEST_START).to_numpy()

def failures_between(lo, hi):
    return failures[(failures["failure_time"] > lo) & (failures["failure_time"] <= hi)]

split = pd.DataFrame({
    "from": [t_first, VALID_START, TEST_START], "to": [VALID_START - GAP, TEST_START - GAP, t_last],
    "rows": [fit_mask.sum(), valid_mask.sum(), test_mask.sum()],
    "positive rows %": [100 * y_all[m].mean() for m in (fit_mask, valid_mask, test_mask)],
    "failures": [len(failures_between(t_first, VALID_START)), len(failures_between(VALID_START, TEST_START)),
                 len(failures_between(TEST_START, t_last + GAP))]}, index=["fit", "valid", "test"])
display(split.round(1))
for part, m in (("fit", fit_mask), ("valid", valid_mask), ("test", test_mask)):
    if y_all[m].sum() == 0:
        raise RuntimeError(f"The {part} months contain no failures, so the model cannot learn or be checked. "
                           "Generate more data on the PC (more reclosers or more days) and run again.")
"""

CALIBRATE_CODE = """
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, y_val)
CAL_X, CAL_Y = [float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_]

def calibrate(raw):
    return np.clip(np.interp(raw, CAL_X, CAL_Y), 0.0, 1.0)

p_val = calibrate(raw_val)
print(f"{len(CAL_X)} calibration points | validation: mean predicted {p_val.mean():.3f} vs observed {y_val.mean():.3f}")
"""

THRESHOLD_CODE = """
def event_metrics(frame, warn, fails):
    warn = pd.Series(np.asarray(warn, dtype=bool), index=frame.index)
    by_asset = {a: g for a, g in frame.assign(_w=warn).groupby("asset_id")}
    rows = []
    for fl in fails.itertuples():
        g = by_asset.get(fl.asset_id)
        hit = pd.Series(dtype="datetime64[ns]") if g is None else g.loc[
            g["_w"] & (g["ts"] < fl.failure_time) & (g["ts"] > fl.failure_time - pd.Timedelta(days=HORIZON_DAYS)), "ts"]
        lead = (fl.failure_time - hit.min()).total_seconds() / 86400 if len(hit) else np.nan
        rows.append((fl.asset_id, fl.failure_time, fl.component, len(hit) > 0, lead))
    per = pd.DataFrame(rows, columns=["asset_id", "failure_time", "component", "caught", "warning_days"])
    alarms = frame[warn.to_numpy() & ~frame["failure_next_30d"].to_numpy()]
    false_days = alarms.assign(day=alarms["ts"].dt.floor("D")).drop_duplicates(["asset_id", "day"]).shape[0]
    ar_months = frame.groupby("asset_id")["ts"].agg(lambda s: (s.max() - s.min()).days / 30.44).sum()
    return per, false_days / max(ar_months, 1e-9)

val_frame = df.loc[valid_mask, ["asset_id", "ts", "failure_next_30d"]].assign(p=p_val)
val_fails = failures_between(VALID_START, TEST_START)
grid = []
for thr in np.unique(np.quantile(p_val, np.linspace(0.5, 0.999, 80))):
    per, fa = event_metrics(val_frame, val_frame["p"] >= thr, val_fails)
    grid.append((float(thr), per["caught"].mean() if len(per) else np.nan, fa))
grid = pd.DataFrame(grid, columns=["threshold", "share_of_failures_caught", "false_alarm_days_per_ar_month"])
ok = grid[(grid["false_alarm_days_per_ar_month"] <= FALSE_ALARM_BUDGET) & grid["share_of_failures_caught"].notna()]
if len(ok) and ok["share_of_failures_caught"].max() > 0:
    THRESHOLD = float(ok.sort_values(["share_of_failures_caught", "threshold"], ascending=[False, False]).iloc[0]["threshold"])
    how = f"most failures caught within {FALSE_ALARM_BUDGET} false-alarm day(s) per AR-month"
else:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    THRESHOLD = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5
    how = "best F1 (no level met the false-alarm budget)"
print(f"alarm level: probability >= {THRESHOLD:.3f} ({how})")
"""

DAILY_CODE = """
d = test.assign(day=test["ts"].dt.tz_localize("UTC").dt.tz_convert("Asia/Colombo").dt.date)
daily = d.groupby(["asset_id", "day"]).agg(avg_hi=("health_index", "mean"), max_p=("p", "max")).reset_index()
daily["hi_healthy"] = daily["avg_hi"] >= HI_HEALTHY_MIN
daily["ml_healthy"] = daily["max_p"] < THRESHOLD
daily["agree"] = daily["hi_healthy"] == daily["ml_healthy"]
print(f"{len(daily):,} recloser-days | HI and ML agree on {daily['agree'].mean():.1%}")
display(pd.crosstab(daily["hi_healthy"].map({True: "HI: healthy", False: "HI: NOT healthy"}),
                    daily["ml_healthy"].map({True: "ML: healthy", False: "ML: NOT healthy"})))
top = daily.groupby("asset_id")["max_p"].max().sort_values(ascending=False).head(20).index
heat = daily[daily["asset_id"].isin(top)].pivot(index="asset_id", columns="day", values="max_p")
plt.figure(figsize=(16, 6))
plt.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="inferno", vmin=0, vmax=max(THRESHOLD * 2, 0.05))
plt.yticks(range(len(heat)), heat.index)
plt.colorbar(label="daily max ML probability")
plt.title("Daily ML risk for the 20 highest-risk reclosers (test months)")
plt.show()
"""


def features_writefile_code() -> str:
    """The cell that writes ar_gen/ml_features.py, verbatim, into the Colab session."""
    return "%%writefile ar_ml_features.py\n" + FEATURES_SRC.read_text(encoding="utf-8").strip()


def _cells() -> list:
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # ⚡ LECO Auto-Recloser Failure Prediction (Google Colab)

    This notebook trains a machine-learning model that answers, for every auto recloser (AR) and every hour:

    > **Will this recloser fail within the next 30 days?**

    It then compares the model with the mathematical **Health Index (HI)** and exports the model, so the
    project's dashboard can show **ML risk next to the HI every 15 minutes**.

    **How to run it:** step-by-step instructions are in `docs/ML_COLAB_GUIDE.md`. In short:
    1. Upload `ar_ml_dataset_hourly.csv.gz` (from `output/ml_generated/`) to a Google Drive folder named **AR_ML**.
    2. In Colab: **File → Upload notebook** → this file, then **Runtime → Run all**. A CPU runtime is enough.
    3. At the end, the browser downloads **`ar_ml_model.zip`**. Unzip it into the project's `models/` folder.

    | Section | What happens |
    |---|---|
    | 0–1 | Settings, packages, load the data from Drive |
    | 2 | Explore failures and labels |
    | 3 | Build the inputs with the project's feature recipe |
    | 4 | Split by time (train on the past, test on the newest months) |
    | 5–6 | Baseline model, then LightGBM tuned with Optuna |
    | 7–8 | Turn scores into realistic probabilities; choose the alarm level |
    | 9–10 | Honest test: failures caught, warning days, false alarms; compare with the HI |
    | 11 | Why the model warns (SHAP explanations) |
    | 12 | Which part is likely to fail (component model) |
    | 13 | Daily HI vs ML agreement |
    | 14 | Export `ar_ml_model.zip` for the dashboard |
    """)

    md("## 0. Settings and packages")
    code("""
    import os
    SMOKE = os.environ.get("AR_NB_SMOKE") == "1"   # True = tiny run that only checks the notebook works
    DATA_FILE = "ar_ml_dataset_hourly.csv.gz"     # made on the PC by: python main.py generate-ml-data
    DRIVE_FOLDER = "AR_ML"                        # Google Drive folder that holds DATA_FILE
    HORIZON_DAYS = 30                             # the question: failure within the next 30 days?
    OPTUNA_TRIALS = 3 if SMOKE else 40            # tuning budget (more trials = better model, longer run)
    OPTUNA_TIMEOUT_MIN = 1 if SMOKE else 25       # stop tuning after this many minutes anyway
    FALSE_ALARM_BUDGET = 1.0                      # max false-alarm days per recloser per month for the alarm level
    HI_HEALTHY_MIN = 60                           # Phase 1 rule: HI >= 60 means healthy
    SEED = 42
    """)
    code("""
    import importlib, subprocess, sys
    for pkg in ("lightgbm", "optuna"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    import datetime as dt, json, time, zipfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    import optuna
    from sklearn.calibration import calibration_curve
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 40)
    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})
    IN_COLAB = "google.colab" in sys.modules
    print("lightgbm", lgb.__version__, "| optuna", optuna.__version__, "| Colab:", IN_COLAB, "| SMOKE:", SMOKE)
    """)

    md("""
    ## 1. Load the data
    The cell first looks in Google Drive (`MyDrive/AR_ML/`) and asks for permission to mount it. If the file is not
    there, an upload box appears instead. The hourly file is ~140 MB, so uploading through Drive is much faster.
    """)
    code(FIND_DATA_CODE)
    code(LOAD_DATA_CODE)

    md("""
    ## 2. Explore the failures
    Every row carries the days until that recloser's next recorded failure. The list of failures is rebuilt from it.
    """)
    code(FAILURES_CODE)
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    monthly = failures.set_index("failure_time").resample("MS")["asset_id"].count()
    axes[0].bar(monthly.index, monthly.values, width=20, color="#00a3cc")
    axes[0].set_title("Failures per month")
    usable_rows = df[~df["censored_30d"] & ~df["failed_now"]]
    for label, color in [(False, "#2ecc71"), (True, "#e74c3c")]:
        usable_rows.loc[usable_rows["failure_next_30d"] == label, "health_index"].plot.hist(
            bins=50, alpha=0.6, density=True, ax=axes[1], color=color, label=f"failure within 30 d = {label}")
    axes[1].set_title("Health Index: rows before a failure vs normal rows")
    axes[1].legend()
    plt.show()
    """)

    md("""
    ## 3. Model inputs (features)
    The next cell writes the **same feature recipe the dashboard uses** (`ar_gen/ml_features.py`, copied in
    automatically). It contains:
    * V1–V41;
    * weather, recloser model and comms channel, as yes/no columns;
    * the 7-day change and 7-day average of 13 key variables.

    Not used as inputs: the HI, its sub-scores and factor scores (the model would just copy the HI formula), and
    the label columns, which are the answers.
    """)
    code("%%writefile ar_ml_features.py\n" + FEATURES_SRC.read_text(encoding="utf-8").strip())
    code(FEATURES_CODE)

    md("""
    ## 4. Split by time, never randomly
    Neighbouring hours are almost identical, so a random split would leak answers and look far too good.

    | part | used for |
    |---|---|
    | **fit** | the oldest data: trains the model |
    | **valid** | the next 3 months: tuning, calibration and the alarm level |
    | **test** | the newest ~35 %: the honest exam |

    A **30-day gap** separates the parts, because each label looks 30 days ahead. Rows whose outcome is unknown
    (`censored_30d`) and rows where the recloser is already broken (`failed_now`) are left out.
    """)
    code(SPLIT_CODE)

    md("## 5. Baseline: logistic regression\nA simple straight-line model. LightGBM has to beat it to be worth using.")
    code("""
    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=300, class_weight="balanced"))
    idx = np.flatnonzero(fit_mask)
    if len(idx) > 300_000:
        idx = np.random.default_rng(SEED).choice(idx, 300_000, replace=False)
    baseline.fit(X_all.iloc[idx], y_all[idx])
    p_base_valid = baseline.predict_proba(X_all[valid_mask])[:, 1]
    print(f"baseline validation PR-AUC {safe_ap(y_all[valid_mask], p_base_valid):.3f} "
          f"(a random guess scores {y_all[valid_mask].mean():.3f})")
    """)

    md("""
    ## 6. LightGBM tuned with Optuna
    LightGBM is gradient-boosted decision trees: fast, handles missing values, and strong on table data.
    Optuna tries different settings (`OPTUNA_TRIALS`) and keeps the ones with the best **validation PR-AUC**.
    Each trial stops adding trees when the validation score stops improving.
    """)
    code("""
    X_fit, y_fit = X_all[fit_mask], y_all[fit_mask]
    X_val, y_val = X_all[valid_mask], y_all[valid_mask]
    # feature_pre_filter=False: lets each Optuna trial use its own min_data_in_leaf on the same dataset
    dtrain = lgb.Dataset(X_fit, y_fit, feature_name=FEATURES, free_raw_data=False, params={"feature_pre_filter": False})
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
    neg_pos = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum())
    BASE = dict(objective="binary", metric="average_precision", verbosity=-1, seed=SEED, bagging_freq=1)

    def objective(trial):
        params = {**BASE,
                  "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20 if SMOKE else 100, 3000, log=True),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                  "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max(1.5, neg_pos))}
        booster = lgb.train(params, dtrain, num_boost_round=200 if SMOKE else 3000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(50 if SMOKE else 150, verbose=False)])
        trial.set_user_attr("best_iteration", int(booster.best_iteration or booster.current_iteration()))
        score = safe_ap(y_val, booster.predict(X_val, num_iteration=booster.best_iteration))
        return 0.0 if np.isnan(score) else score

    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT_MIN * 60, show_progress_bar=not SMOKE)
    best = study.best_trial
    BEST_PARAMS = {**BASE, **best.params}
    BEST_ITER = max(10, int(best.user_attrs.get("best_iteration", 100)))
    print(f"{len(study.trials)} trials in {time.time() - t0:.0f} s | best validation PR-AUC {best.value:.3f} "
          f"with {BEST_ITER} trees")
    display(pd.Series(best.params).to_frame("best setting"))
    model = lgb.train(BEST_PARAMS, dtrain, num_boost_round=BEST_ITER)
    raw_val = model.predict(X_val)
    """)

    md("""
    ## 7. Realistic probabilities (calibration)
    Class weighting pushes raw scores up. **Isotonic calibration** on the validation months maps them to probabilities
    that match how often failures really followed. The mapping is stored as numbers in the model file, and the
    dashboard applies exactly the same mapping.
    """)
    code(CALIBRATE_CODE)

    md("""
    ## 8. Alarm level (threshold)
    For each candidate level, the cell counts on the validation months:
    * how many failures got a warning in the 30 days before them;
    * how many **false-alarm days per recloser per month** there were (days with a warning but no failure within 30 days).

    The chosen level catches the most failures while staying within `FALSE_ALARM_BUDGET`. If no level fits the
    budget, the level with the best F1 is used.
    """)
    code(THRESHOLD_CODE)

    md("""
    ## 9. Honest test: ML vs Health Index
    All numbers below come from the **test months**, which the model never saw.
    * **ROC-AUC / PR-AUC:** how well risky rows are ranked above safe ones (PR-AUC is the stricter one for rare events).
    * **Failures caught:** a warning appeared in the 30 days before the failure.
    * **Warning days:** how early the first warning came.

    The HI's own rule is **HI < 60**. "ML or HI" warns when either one does.
    """)
    code("""
    test = df.loc[test_mask, ["asset_id", "ts", "health_index", "failure_next_30d"]].copy()
    test["p"] = calibrate(model.predict(X_all[test_mask]))
    test["ml_warn"] = test["p"] >= THRESHOLD
    test["hi_warn"] = test["health_index"] < HI_HEALTHY_MIN
    test["any_warn"] = test["ml_warn"] | test["hi_warn"]
    test_fails = failures_between(TEST_START, t_last + GAP)
    any_score = np.maximum(test["p"] / max(THRESHOLD, 1e-9), (100 - test["health_index"]) / (100 - HI_HEALTHY_MIN))
    results, per_failure = {}, {}
    for name, score, warn in [("ML model", test["p"], "ml_warn"),
                              ("Health Index (HI < 60)", 100 - test["health_index"], "hi_warn"),
                              ("ML or HI", any_score, "any_warn")]:
        per, fa = event_metrics(test, test[warn], test_fails)
        per_failure[name] = per
        results[name] = {"ROC-AUC": round(safe_auc(test["failure_next_30d"], score), 3),
                         "PR-AUC": round(safe_ap(test["failure_next_30d"], score), 3),
                         "failures caught": f"{int(per['caught'].sum())} / {len(per)}",
                         "median warning days": round(per["warning_days"].median(), 1) if len(per) else np.nan,
                         "false-alarm days per AR-month": round(fa, 2)}
    comparison = pd.DataFrame(results).T
    print(f"random-guess PR-AUC = {test['failure_next_30d'].mean():.3f} | "
          f"Brier score (lower = better) = {brier_score_loss(test['failure_next_30d'], test['p']):.4f}")
    display(comparison)
    by_comp = pd.DataFrame({name: per.groupby("component")["caught"].agg(lambda s: f"{int(s.sum())} / {len(s)}")
                            for name, per in per_failure.items()})
    by_comp.index.name = "failures caught by component"
    display(by_comp)
    """)
    code("""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for name, score in [("ML model", test["p"]), ("Health Index", 100 - test["health_index"])]:
        if 0 < test["failure_next_30d"].sum() < len(test):
            pr, rc, _ = precision_recall_curve(test["failure_next_30d"], score)
            axes[0].plot(rc, pr, label=f"{name} (PR-AUC {safe_ap(test['failure_next_30d'], score):.2f})")
    axes[0].axhline(test["failure_next_30d"].mean(), color="grey", ls=":", label="random guess")
    axes[0].set(xlabel="recall", ylabel="precision", title="Precision vs recall (test)")
    axes[0].legend()
    lead = per_failure["ML model"]["warning_days"].dropna()
    axes[1].hist(lead, bins=15, color="#00a3cc")
    axes[1].set(title="Warning days before caught failures (ML)", xlabel="days")
    if test["failure_next_30d"].sum() > 0:
        frac, mean_p = calibration_curve(test["failure_next_30d"], test["p"], n_bins=10, strategy="quantile")
        axes[2].plot(mean_p, frac, "o-", color="#e67e22", label="ML model")
    axes[2].plot([0, 1], [0, 1], ":", color="grey", label="perfect")
    axes[2].set(title="Calibration (test)", xlabel="predicted probability", ylabel="observed frequency")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    """)

    md("""
    ## 10. One recloser over time
    The HI (blue, left axis) and the ML failure probability (red, right axis) for the test recloser where the two
    together caught the most failures. Black lines are recorded failures; dashed lines are the two alarm levels.
    """)
    code("""
    if len(test_fails):
        both = sum(per_failure[k].groupby("asset_id")["caught"].sum() for k in ("ML model", "Health Index (HI < 60)"))
        aid = both.sort_values(kind="stable").index[-1]
        a = test[test["asset_id"] == aid].set_index("ts")[["health_index", "p"]].resample("1h").asfreq()
        fig, ax1 = plt.subplots(figsize=(15, 4))
        ax1.plot(a.index, a["health_index"], color="tab:blue", lw=1)
        ax1.axhline(HI_HEALTHY_MIN, color="tab:blue", ls="--", lw=0.8)
        ax1.set_ylim(0, 102)
        ax1.set_ylabel("Health Index", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(a.index, a["p"], color="tab:red", lw=1)
        ax2.axhline(THRESHOLD, color="tab:red", ls="--", lw=0.8)
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("ML failure probability", color="tab:red")
        for k, fl in enumerate(test_fails[test_fails["asset_id"] == aid].itertuples()):
            ax1.axvline(fl.failure_time, color="black", lw=1.5)
            ax1.text(fl.failure_time, 3 + 9 * (k % 2), f" {fl.component}", fontsize=9)
        ax1.set_title(f"{aid}: Health Index vs ML failure probability (test months)")
        plt.show()
    """)

    md("""
    ## 11. Why does the model warn? (SHAP explanations)
    LightGBM computes **SHAP values** itself (`pred_contrib=True`): how much each input pushed the risk up or down.
    The first chart shows the most important inputs overall. The second explains the first warning before one
    caught failure.
    """)
    code("""
    idx = np.flatnonzero(test_mask)
    if len(idx) > 20_000:
        idx = np.random.default_rng(SEED).choice(idx, 20_000, replace=False)
    contrib = model.predict(X_all.iloc[idx], pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False).head(20)
    importance[::-1].plot.barh(figsize=(9, 6), color="#00a3cc", title="Top 20 inputs (mean |SHAP|, log-odds)")
    plt.tight_layout()
    plt.show()
    caught = per_failure["ML model"][per_failure["ML model"]["caught"]]
    if len(caught):
        ex = caught.iloc[0]
        g = test[(test["asset_id"] == ex.asset_id) & test["ml_warn"] & (test["ts"] < ex.failure_time)
                 & (test["ts"] > ex.failure_time - pd.Timedelta(days=HORIZON_DAYS))]
        i = g.index[0]
        one = pd.Series(model.predict(X_all.loc[[i]], pred_contrib=True)[0, :-1], index=FEATURES)
        one = one.reindex(one.abs().sort_values(ascending=False).index).head(10)
        one[::-1].plot.barh(figsize=(9, 4), color=["#e74c3c" if v > 0 else "#2ecc71" for v in one[::-1]],
                            title=f"{ex.asset_id}: first warning {test.loc[i, 'ts']:%Y-%m-%d} before a {ex.component} "
                                  f"failure on {ex.failure_time:%Y-%m-%d} (red = raises risk)")
        plt.tight_layout()
        plt.show()
    """)

    md("""
    ## 12. Which part is likely to fail? (component model)
    A second LightGBM model is trained only on rows **before** failures, to say which component is most at risk.
    The dashboard shows it as the "likely part".
    """)
    code("""
    comp_rows = (fit_mask | valid_mask) & df["failure_next_30d"].to_numpy() & df["next_failure_component"].notna().to_numpy()
    COMPONENTS = sorted(df.loc[comp_rows, "next_failure_component"].unique())
    comp_model = None
    if len(COMPONENTS) >= 2:
        cls = {c: i for i, c in enumerate(COMPONENTS)}
        yc = df.loc[comp_rows, "next_failure_component"].map(cls).to_numpy()
        counts = np.bincount(yc, minlength=len(COMPONENTS))
        weights = (len(yc) / (len(COMPONENTS) * counts))[yc]
        comp_model = lgb.train(dict(objective="multiclass", num_class=len(COMPONENTS), learning_rate=0.05,
                                    num_leaves=31, min_data_in_leaf=10 if SMOKE else 50, feature_fraction=0.8,
                                    verbosity=-1, seed=SEED),
                               lgb.Dataset(X_all[comp_rows], yc, weight=weights, feature_name=FEATURES),
                               num_boost_round=60 if SMOKE else 400)
        test_pos = test_mask & df["failure_next_30d"].to_numpy() & df["next_failure_component"].isin(COMPONENTS).to_numpy()
        if test_pos.any():
            predicted = np.array(COMPONENTS)[comp_model.predict(X_all[test_pos]).argmax(axis=1)]
            actual = df.loc[test_pos, "next_failure_component"].to_numpy()
            print(f"component model: right part on {(predicted == actual).mean():.0%} of test rows before failures")
            display(pd.crosstab(pd.Series(actual, name="actual"), pd.Series(predicted, name="predicted")))
    else:
        print("Not enough different failing parts in the training months for a component model.")
    """)

    md("""
    ## 13. Daily view: HI vs ML per recloser per day
    This is the same daily comparison the project stores in `ar_daily_summary`. HI healthy = daily average HI ≥ 60;
    ML healthy = probability below the alarm level all day.
    """)
    code(DAILY_CODE)

    md("""
    ## 14. Export the model for the dashboard
    This writes `ar_ml_model.zip` with three files:
    * `model_failure30d.txt`: the LightGBM failure model;
    * `model_component.txt`: the "which part" model;
    * `ml_meta.json`: feature list, alarm level, calibration and test metrics.

    LightGBM text files and JSON load on any computer, whatever library versions it has. **Unzip it into the
    project's `models/` folder**; the next 15-minute run starts scoring.
    """)
    code("""
    os.makedirs("export", exist_ok=True)
    for old in os.listdir("export"):
        os.remove(os.path.join("export", old))
    model.save_model("export/model_failure30d.txt")
    if comp_model is not None:
        comp_model.save_model("export/model_component.txt")
    ml_res, hi_res = per_failure["ML model"], per_failure["Health Index (HI < 60)"]
    metrics = {"test_roc_auc": results["ML model"]["ROC-AUC"], "test_pr_auc": results["ML model"]["PR-AUC"],
               "test_failures_caught": results["ML model"]["failures caught"],
               "test_median_warning_days": results["ML model"]["median warning days"],
               "test_false_alarm_days_per_ar_month": results["ML model"]["false-alarm days per AR-month"],
               "hi_failures_caught": results["Health Index (HI < 60)"]["failures caught"],
               "ml_or_hi_failures_caught": results["ML or HI"]["failures caught"],
               "validation_pr_auc": round(float(best.value), 3)}
    now = dt.datetime.now(dt.timezone.utc)
    meta = {"feature_version": FEATURE_VERSION, "features": FEATURES, "threshold": THRESHOLD,
            "calibration": {"x": CAL_X, "y": CAL_Y},
            "component_classes": COMPONENTS if comp_model is not None else [],
            "metrics": metrics, "params": BEST_PARAMS, "trees": BEST_ITER, "horizon_days": HORIZON_DAYS,
            "trained_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "data_until_utc": str(df["ts"].max()),
            "model_version": "lgbm-" + now.strftime("%Y%m%d-%H%M")}
    with open("export/ml_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with zipfile.ZipFile("ar_ml_model.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir("export")):
            z.write(os.path.join("export", fn), fn)
    print("ar_ml_model.zip:", sorted(os.listdir("export")), "| model version", meta["model_version"])
    display(pd.Series(metrics).to_frame("test result"))
    if IN_COLAB and not SMOKE:
        from google.colab import files
        files.download("ar_ml_model.zip")
    """)
    code("""
    # Check: load the bundle exactly like the dashboard does, and score a few rows.
    check = lgb.Booster(model_file="export/model_failure30d.txt")
    with open("export/ml_meta.json") as fh:
        m = json.load(fh)
    assert check.feature_name() == m["features"], "feature names differ"
    raw = check.predict(X_all.iloc[:5][m["features"]].to_numpy(dtype=float))
    print("probabilities for 5 rows:", np.round(np.interp(raw, m["calibration"]["x"], m["calibration"]["y"]), 4))
    """)
    md("""
    ### Reading the results honestly
    * **The data is simulated.** The model learns the patterns the simulator built in. Test it on real NOJA data before relying on it.
    * **Check the test numbers, not the validation numbers.** Validation was used for tuning and is optimistic.
    * **"ML or HI" often catches the most failures.** The dashboard shows both side by side for that reason.
    * **Retrain** when more data is available: generate a new file on the PC, run this notebook again, and replace the files in `models/`.
    """)
    return cells


def build(path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = _cells()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "AR_ML_Colab.ipynb")
    print("wrote", out)
