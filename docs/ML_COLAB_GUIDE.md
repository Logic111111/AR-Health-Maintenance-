# Training the AR failure-prediction model in Google Colab

This guide trains the machine-learning (ML) model in **Google Colab**, Google's free computer in the browser, and
puts it back into the project. From then on, the dashboard shows each recloser's **ML failure risk next to its
Health Index** every 15 minutes.

**What the model predicts:** the probability that a recloser **fails within the next 30 days**, plus which part
is most likely to fail.

**Nothing is trained on your PC.** The PC only creates the data file and, afterwards, uses the finished model.

> **ML is switched off for now** (`config.yaml` → `ml: enabled: false`): the collector and the dashboard ignore
> installed models. Before Step 6, set `enabled: true`, then run `python main.py service restart` (admin) and
> restart the dashboard. Training in Colab (Steps 1–5) works either way. See README section 7.

Expected time: **5 minutes** of preparation, **25–45 minutes** of Colab running by itself, **2 minutes** to install
the result.

---

## Step 1: Get the training file (on the PC)

The file is created by the simulator: 100 LECO reclosers over 3 years, about 290 recorded failures.

```powershell
cd "D:\AR Preventive Analyze"
python main.py generate-ml-data --assets 100 --days 1095      # only if the file does not exist yet (~40 min)
```

The result is `D:\AR Preventive Analyze\output\ml_generated\ar_ml_dataset_hourly.csv.gz` (about 140 MB). It has
one row per recloser per hour: V1–V42, the Health Index and the failure labels.

To train on more history, increase `--days` or `--assets`. More failures give a better model.

## Step 2: Put the file in Google Drive

1. Open **https://drive.google.com** and sign in with your Google account.
2. Click **New → New folder**, name it exactly **`AR_ML`**, and open it.
3. Drag `ar_ml_dataset_hourly.csv.gz` into the folder and wait until the upload finishes.

Drive is used instead of uploading directly to Colab because Colab forgets uploaded files when it restarts,
while Drive keeps them.

## Step 3: Open the notebook in Colab

1. Open **https://colab.research.google.com**.
2. Choose **File → Upload notebook** and pick `D:\AR Preventive Analyze\notebooks\AR_ML_Colab.ipynb`.
3. Check **Runtime → Change runtime type**:
   - **CPU** is enough. LightGBM doesn't need a GPU.
   - If you have Colab Pro, **High-RAM** makes loading faster, but the free runtime works.

## Step 4: Run everything

1. Choose **Runtime → Run all**.
2. When Colab asks **"Permit this notebook to access your Google Drive files?"**, click **Connect to Google
   Drive** and allow access. The notebook reads the file from `MyDrive/AR_ML/`.
3. Let it run. The progress bar in section 6 shows the tuning; that is the longest part (about 25 minutes by
   default). You can close the laptop lid only if you keep the browser tab open.

Settings you can change in the first code cell:

| Setting | Default | Meaning |
|---|---|---|
| `OPTUNA_TRIALS` | 40 | How many model settings to try. More trials give a better model and take longer. |
| `OPTUNA_TIMEOUT_MIN` | 25 | Tuning stops after this many minutes in any case. |
| `FALSE_ALARM_BUDGET` | 1.0 | The most false-alarm days per recloser per month you accept when the alarm level is chosen. |
| `HORIZON_DAYS` | 30 | Prediction window. If you change it, regenerate the data and retrain. |

## Step 5: Read the results

| Section | What to look at |
|---|---|
| 2. Explore | Number of failures per part; the HI is lower before failures (red bars). |
| 4. Split | **fit / valid / test** periods. The **test** months are never used for training. |
| 5. Baseline | PR-AUC of a simple model. LightGBM should be clearly higher. |
| 6. LightGBM | Best validation PR-AUC and settings after tuning. |
| 8. Alarm level | The probability at which a recloser is flagged, and why it was chosen. |
| 9. Test | **The table to judge the model by:** see the next table. |
| 10. One recloser | HI and ML risk over time, with the failures marked. |
| 11. SHAP | Which inputs drive the risk, overall and for one real warning. |
| 12. Component | How often the model names the right failing part. |
| 13. Daily | How often HI and ML agree day by day; heat-map of the riskiest reclosers. |

The section 9 table compares ML, HI and "ML or HI" on:
- **PR-AUC:** higher is better; a random guess scores the share of failure rows;
- **failures caught**, out of all test failures;
- **median warning days** before a failure;
- **false-alarm days per AR-month.**

What to expect from the simulated data:
- The ML model catches **battery** failures well, because a failed battery test predicts failure.
- The HI catches **mechanism** and **controller** failures well, through its caps.
- **"ML or HI" usually catches the most**, which is why the dashboard shows both.

## Step 6: Download and install the model

1. The last section downloads **`ar_ml_model.zip`** automatically. If the browser blocks it, find it in the
   file panel on the left of Colab (folder icon), right-click it and choose **Download**.
2. Unzip it into **`D:\AR Preventive Analyze\models\`**. Create the folder if it doesn't exist. It must contain:
   - `ml_meta.json`: feature list, alarm level, calibration and test results;
   - `model_failure30d.txt`: the failure model (LightGBM text format);
   - `model_component.txt`: the "which part" model.
3. On the PC, run:

```powershell
cd "D:\AR Preventive Analyze"
python -m pip install -r requirements.txt     # installs lightgbm if missing (small, only for using the model)
python main.py ml-status                      # shows the model version, alarm level and test results
python main.py ml-score                        # scores all stored data once (optional: --start 2026-09-07)
```

After that, the collector (`scheduler` or Task Scheduler) scores every new 15-minute interval by itself. In the
dashboard:
- the header shows **"ML · lgbm-…"**;
- the map's column height can switch to **ML failure risk**;
- the **Recloser** tab shows the ML gauge and risk line;
- the **ML vs HI** tab shows the comparison and the daily agreement.

The results are also stored in the database:

| Table | Content |
|---|---|
| `ar_ml_prediction_15min` | Risk per recloser per 15 min: probability, LOW / ELEVATED / HIGH, likely part |
| `ar_daily_summary` | Per recloser per day: average / minimum / maximum HI, hours unhealthy, average and maximum ML risk, and whether HI and ML agree |

## Retraining later

Repeat the steps with a new data file whenever more data or real data is available, then replace the three files in
`models/` and run `python main.py ml-score` (it rescores all stored data with the new model).
- The model files include a **feature version**. If the project's feature recipe changes, the old model is not
  used:
  - the dashboard header shows **ML · MODEL ERROR**;
  - the ML vs HI tab gives the reason;
  - old ML values are hidden instead of being shown as current;
  - the Health Index carries on as normal.

  Retrain with the new notebook.
- If one 15-minute ML write fails (for example, the database is busy), the HI still goes on. The next collector run
  scores the missed intervals.
- To rebuild the notebook after changing `ar_gen/ml_features.py`, run `python notebooks/build_colab_notebook.py`. The
  notebook always embeds the current feature code.

## Deep-learning model (optional, works next to LightGBM)

`notebooks/AR_DL_Colab.ipynb` answers the same question with a neural network instead of LightGBM:
- **What it reads:** the **last 14 days of each recloser as a sequence** (168 steps, 2 hours apart).
- **What it learns:** the warning patterns over time, by itself.
- **What it tries:** three networks (**GRU**, **TCN** and **Transformer**). The one with the best validation score is kept.
- **What it answers:** failure within 30 days, failure within 7 days, and which part.

It reuses the LightGBM notebook's code for loading, inputs, the time split, calibration, the alarm level and the
daily view. Both models are therefore judged on the same test months.

In the app, the network is **separate from LightGBM**:
- its own folder, `models\dl\`;
- its own table, `ar_dl_prediction_15min`;
- its own commands, `dl-status` and `dl-score`;
- its own dashboard tab, **Deep Learning**.

Both models can run side by side. A missing or broken deep-learning model never stops the Health Index or LightGBM.

### DL step 1: Train in Colab (about 30–45 minutes, mostly unattended)

1. Keep `ar_ml_dataset_hourly.csv.gz` in `MyDrive/AR_ML/` (Step 2 above). It is the same file LightGBM uses.
2. Optional but recommended: also put the LightGBM result **`ar_ml_model.zip`** in `MyDrive/AR_ML/`. The notebook then
   scores LightGBM on the same test months and puts both in one table.
3. Open **https://colab.research.google.com**, choose **File → Upload notebook** and pick
   `D:\AR Preventive Analyze\notebooks\AR_DL_Colab.ipynb`.
4. Choose **Runtime → Change runtime type → T4 GPU** and click **Save**.
   - The free T4 is enough.
   - The notebook asks for a GPU itself, but check this.
   - On a CPU, training takes many hours. The first code cell then prints `WARNING: no GPU`.
5. Choose **Runtime → Run all**. When asked, click **Connect to Google Drive** and allow access.
6. Wait. Loading and inputs take about 5–10 minutes. **Section 7** then prints one line per network per epoch, for
   example `gru  epoch  3  loss 0.81  validation PR-AUC 0.62  (1.4 min)`, and finishes with a table showing which
   network was kept. Keep the browser tab open.
7. When it finishes, the browser downloads **`ar_dl_model.zip`**. A copy is also saved in `MyDrive/AR_ML/`, which
   is useful when the browser blocks the download.

### DL step 2: Read the results

| Section | What to look at |
|---|---|
| 7. Train | The network table: validation PR-AUC of GRU, TCN and Transformer, and which one was kept. The learning curves should rise, then level off. |
| 10. Test | **The table to judge it by.** Deep learning vs LightGBM vs Health Index: failures caught, warning days, false alarms. Also **"DL or LightGBM"**: what you gain by using both. |
| 11. One recloser | HI, LightGBM risk and deep-learning risk over time, with the failures marked. |
| 12. Which part | How often the network names the right failing part, compared with LightGBM. |
| 13. What it looks at | Which inputs and which days of the 14-day window push the risk up. |

Deep learning is not automatically better on table data like this. Install it and use it **next to** LightGBM.
If section 10 shows it clearly worse, you can leave it out.

### DL step 3: Install on the PC (2 minutes)

1. Unzip `ar_dl_model.zip` into **`D:\AR Preventive Analyze\models\dl\`**. Create the folder if needed.
   - It must contain `dl_meta.json` and `model_dl.pt` directly, not in a sub-folder.
   - Use `models\dl\`, **not** `models\`: `models\` holds the LightGBM files.
2. PyTorch is needed to use the network (not to train it). If `python main.py dl-status` says it is missing:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

3. Check the model and score the stored data once:

```powershell
cd "D:\AR Preventive Analyze"
python main.py dl-status     # model version, network type, window, alarm level and test results
python main.py dl-score      # scores all stored data once (optional: --start 2026-09-20)
```

After that, the collector (`scheduler` or Task Scheduler) also scores every new 15-minute interval with the network.
It reads about 3 weeks of V1–V42 history for each prediction (14-day window plus 7-day trends).

### DL step 4: Use it in the dashboard

Open `python main.py dashboard` and go to the **Deep Learning** tab:
- **Header:** a **DL · dl-…** chip shows the installed network.
- **KPIs:**
  - reclosers the network rates HIGH;
  - where it and LightGBM both warn (act now);
  - where only one of them warns.
- **Scatter:** the deep-learning risk of every recloser against its LightGBM risk, or against its HI when no
  LightGBM model is installed.
- **One recloser (sidebar selection):**
  - HI, LightGBM risk and deep-learning risk over 30 days.
- **Ranking table:** all reclosers by deep-learning risk, with the likely failing part.

### Notebook settings (first code cell)

| Setting | Default | Meaning |
|---|---|---|
| `ARCHS` | gru, tcn, transformer | Networks to try. Remove some for a shorter run. |
| `WINDOW_STEPS` × `STEP_HOURS` | 168 × 2 h | How much history the network reads (14 days). |
| `EPOCHS` / `PATIENCE` | 15 / 4 | Most training rounds, and when to stop if the validation score stops improving. |
| `POS_SHARE` | 0.3 | Share of "fails within 30 days" windows in training. Calibration corrects for it. |
| `BATCH` | 1024 | Windows per training step. Lower it (e.g. 512) if Colab runs out of GPU memory. |

The network code lives in `ar_gen/dl_model.py`. The notebook embeds it verbatim, so the dashboard rebuilds exactly the
trained network. After changing that file, `ar_gen/ml_features.py` or the shared cells, run
`python notebooks/build_dl_notebook.py` and retrain.

## Troubleshooting

| Problem | What to do |
|---|---|
| `FileNotFoundError ... AR_ML` | The file is not in `MyDrive/AR_ML/`. Check the folder name, or let the upload box appear and choose the file. |
| Colab "ran out of RAM" | Use **Runtime → Change runtime type → High-RAM** if available, or generate a smaller file (`--assets 60 --days 730`). |
| "The fit months contain no failures" | Too little data. Generate more reclosers or more days. |
| The session disconnects during tuning | Reduce `OPTUNA_TRIALS` (e.g. 15) and **Run all** again. |
| `ml-status` says "feature version" | The model was trained with an older notebook. Retrain with `notebooks/AR_ML_Colab.ipynb` from this project. |
| `ml-status` says "LightGBM is not installed" | Run `python -m pip install lightgbm`. |
| The dashboard shows "ML · MODEL ERROR" | Open the **ML vs HI** tab: it shows why the model cannot be used (also printed by `python main.py ml-status`). |
| The dashboard still shows "ML · NOT INSTALLED" | Check that the three files are directly inside `models\`, not in a sub-folder, and refresh the page. |
| Deep-learning notebook prints "WARNING: no GPU" | **Runtime → Change runtime type → T4 GPU**, then **Run all** again. |
| Deep-learning notebook: "CUDA out of memory" | Lower `BATCH` (e.g. 512) in the first code cell and run again. |
| No "LightGBM" row in the deep-learning test table | Put `ar_ml_model.zip` in `MyDrive/AR_ML/` and run again. |
| `dl-status` says "PyTorch is not installed" | `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `dl-status` says "feature version" or "network version" | The network was trained with an older notebook. Retrain with `notebooks/AR_DL_Colab.ipynb` from this project. |
| The dashboard shows "DL · MODEL ERROR" | Open the **Deep Learning** tab: it shows why (also printed by `python main.py dl-status`). |
| The Deep Learning tab says "No deep-learning model installed" | Check that `dl_meta.json` and `model_dl.pt` are directly inside `models\dl\`, then refresh. |

## Honest limits

- **The training data is simulated**, so the model learns the patterns the simulator produces. Before relying on it
  for maintenance decisions, collect real NOJA recloser data, train again on that, and check the test results.
- **Only the test months count.** Validation scores are optimistic because they were used for tuning.
