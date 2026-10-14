"""Build notebooks/AR_DL_Colab.ipynb (run: python notebooks/build_dl_notebook.py).

The deep-learning notebook answers the same question as AR_ML_Colab.ipynb (failure within 30 days) with a
sequence network trained on a Colab GPU. It reuses the LightGBM notebook's cells for loading, the feature
recipe, the time split, calibration, the alarm level and the daily view, so both models are judged on
exactly the same test months.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_colab_notebook import (CALIBRATE_CODE, DAILY_CODE, FAILURES_CODE, FEATURES_CODE,  # noqa: E402
                                  FIND_DATA_CODE, LOAD_DATA_CODE, ROOT, SPLIT_CODE, THRESHOLD_CODE,
                                  features_writefile_code)

DL_MODEL_SRC = ROOT / "ar_gen" / "dl_model.py"


def dl_model_writefile_code() -> str:
    """The cell that writes ar_gen/dl_model.py (network + input scaling), verbatim, into the Colab session."""
    return "%%writefile ar_dl_model.py\n" + DL_MODEL_SRC.read_text(encoding="utf-8").strip()


def _cells() -> list:
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # 🧠 LECO Auto-Recloser Failure Prediction: Deep Learning (Google Colab)

    The same question as `AR_ML_Colab.ipynb`, answered by a neural network instead of LightGBM:

    > **Will this recloser fail within the next 30 days?**

    **What is different?**
    * LightGBM looks at one hour at a time, plus a few hand-made 7-day trends.
    * The network reads the **last 14 days of each recloser as a sequence** and learns the warning patterns over
      time by itself.
    * One network answers three questions at once: failure within 30 days, failure within 7 days, and which part
      will fail.

    **A fair comparison.** This notebook loads the data, splits the months, calibrates and chooses the alarm level
    with **exactly the same code** as the LightGBM notebook. If `ar_ml_model.zip` (the LightGBM result) is in the
    same Drive folder, both models are scored on the same test months, next to the Health Index.

    **How to run it** (details in `docs/ML_COLAB_GUIDE.md`):
    1. Use the same data as before: `ar_ml_dataset_hourly.csv.gz` in the Google Drive folder **AR_ML**.
       Optionally, put `ar_ml_model.zip` next to it.
    2. In Colab, choose **File → Upload notebook** and pick this file.
    3. Choose **Runtime → Change runtime type → T4 GPU** (the free GPU is enough), then **Runtime → Run all**.
    4. At the end, the browser downloads **`ar_dl_model.zip`**. A copy is also saved in Drive/AR_ML.
    5. On the PC, unzip it into **`models/dl/`** and run `python main.py dl-score`. The dashboard's **Deep Learning**
       tab then shows the network's risk.

    | Section | What happens |
    |---|---|
    | 0 | Settings, packages, GPU check |
    | 1–4 | Same as the LightGBM notebook: load the data, list the failures, build the inputs, split by time |
    | 5 | Turn the hourly rows into 14-day sequences |
    | 6 | Three networks: GRU, TCN and Transformer |
    | 7 | Train each network; keep the best one on the validation months |
    | 8–9 | Turn scores into realistic probabilities; choose the alarm level |
    | 10 | Honest test: deep learning vs LightGBM vs Health Index |
    | 11 | One recloser over time |
    | 12 | Which part is likely to fail |
    | 13 | What the network looks at (gradients) |
    | 14 | Daily HI vs deep-learning agreement |
    | 15 | Export `ar_dl_model.zip` |
    """)

    md("## 0. Settings and packages")
    code("""
    import os
    SMOKE = os.environ.get("AR_NB_SMOKE") == "1"   # True = tiny run that only checks the notebook works
    DATA_FILE = "ar_ml_dataset_hourly.csv.gz"     # made on the PC by: python main.py generate-ml-data
    DRIVE_FOLDER = "AR_ML"                        # Google Drive folder that holds DATA_FILE
    LGBM_ZIP = "ar_ml_model.zip"                  # optional: LightGBM result of AR_ML_Colab.ipynb, for the comparison
    HORIZON_DAYS = 30                             # the question: failure within the next 30 days?
    WINDOW_STEPS = 168                            # steps of history the network reads for one prediction
    STEP_HOURS = 2                                # hours between steps: 168 x 2 h = the last 14 days
    ARCHS = ["gru", "tcn", "transformer"]         # networks to try; the best on the validation months is kept
    HIDDEN = 16 if SMOKE else 64                  # network width
    EPOCHS = 2 if SMOKE else 15                   # training rounds, each on a fresh sample of windows
    PATIENCE = 4                                  # stop after this many rounds without a better validation score
    SAMPLES_PER_EPOCH = 4_096 if SMOKE else 409_600
    BATCH = 256 if SMOKE else 1024
    POS_SHARE = 0.3                               # share of "fails within 30 days" windows in each round
    LEARNING_RATE = 2e-3
    FALSE_ALARM_BUDGET = 1.0                      # max false-alarm days per recloser per month for the alarm level
    HI_HEALTHY_MIN = 60                           # Phase 1 rule: HI >= 60 means healthy
    SEED = 42
    """)
    code("""
    import importlib, subprocess, sys
    try:
        importlib.import_module("lightgbm")        # only used to score the LightGBM bundle for the comparison
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lightgbm"])

    import datetime as dt, json, shutil, time, zipfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 40)
    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})
    IN_COLAB = "google.colab" in sys.modules
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP = DEVICE.type == "cuda"                   # mixed precision: about twice as fast on a GPU
    torch.manual_seed(SEED)
    gpu = torch.cuda.get_device_name(0) if AMP else "no GPU"
    print("torch", torch.__version__, "|", DEVICE, f"({gpu})", "| Colab:", IN_COLAB, "| SMOKE:", SMOKE)
    if DEVICE.type == "cpu" and not SMOKE:
        print("WARNING: no GPU. Choose Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all. "
              "On a CPU the training takes many hours.")
    """)

    md("""
    ## 1. Load the data
    Same as the LightGBM notebook. The cell first looks in Google Drive (`MyDrive/AR_ML/`) and asks for permission to
    mount it. If the file is not there, an upload box appears instead.
    """)
    code(FIND_DATA_CODE)
    code(LOAD_DATA_CODE)

    md("## 2. The failures\nEach row carries the days until that recloser's next recorded failure. The list of failures is rebuilt from it.")
    code(FAILURES_CODE)

    md("""
    ## 3. Model inputs (features)
    The **same feature recipe the dashboard uses** (`ar_gen/ml_features.py`, copied in automatically): V1–V41,
    weather / recloser model / comms channel as yes-no columns, and 7-day trends of 13 key variables. The HI and the
    label columns are not inputs. Here these inputs form **one step** of the sequence the network reads.
    """)
    code(features_writefile_code())
    code(FEATURES_CODE)

    md("""
    ## 4. Split by time, never randomly
    Exactly the same split as the LightGBM notebook: **fit** (oldest data, trains the network), **valid** (the next
    3 months: picks the best network and epoch, calibration and the alarm level) and **test** (the newest ~35 %, the
    honest exam), with a 30-day gap between the parts. Rows whose outcome is unknown, or where the recloser is
    already broken, are left out as prediction points. They still appear inside the 14-day histories.
    """)
    code(SPLIT_CODE)

    md("""
    ## 5. From table rows to sequences
    For every prediction, the network reads **168 snapshots, 2 hours apart: the last 14 days** of that recloser.
    * **One scale.** Counts with a long tail (operations, fault current, minutes on battery …) are log-compressed.
      Then every input is standardised with averages from the **fit months only**.
    * **Missing values** become 0, which is the average. `telemetry_present` still tells the network when data was missing.
    * **Short history.** For a recloser's first 14 days, the window is padded with its first hour.
    * **Memory.** Windows are cut on the fly from one array on the GPU, so memory stays small.

    The next cell writes the **project's deep-learning code** (`ar_gen/dl_model.py`, copied in automatically): the
    networks, the input scaling and the window rule. The dashboard rebuilds the trained network with the same file.
    """)
    code(dl_model_writefile_code())
    code("""
    import ar_dl_model
    importlib.reload(ar_dl_model)
    from ar_dl_model import NET_VERSION, ARNet, scale_inputs, window_rows

    t0 = time.time()
    CLIP = 8.0                                     # scaled inputs are limited to +-8 standard deviations
    rng = np.random.default_rng(SEED)
    fit_idx = np.flatnonzero(fit_mask)
    stat = pd.DataFrame(X_all.iloc[np.sort(rng.choice(fit_idx, min(len(fit_idx), 500_000), replace=False))]
                        .to_numpy(np.float32), columns=FEATURES)
    LOG_COLS = [c for c in FEATURES if stat[c].min() >= 0 and stat[c].quantile(0.99) > 50]
    LOG_IDX = [FEATURES.index(c) for c in LOG_COLS]
    stat[LOG_COLS] = np.log1p(stat[LOG_COLS])
    MU = stat.mean().fillna(0).to_numpy(np.float32)
    SD = stat.std().to_numpy(np.float32)
    SD = np.where(np.isfinite(SD) & (SD > 1e-6), SD, 1.0).astype(np.float32)
    del stat

    X_seq = np.empty((len(X_all), len(FEATURES)), dtype=np.float32)
    for i in range(0, len(X_all), 500_000):
        X_seq[i:i + 500_000] = scale_inputs(X_all.iloc[i:i + 500_000].to_numpy(np.float32), LOG_IDX, MU, SD, CLIP)

    # Rows are sorted by recloser and hour: a window is "this row and every STEP_HOURS-th row before it".
    ids = df["asset_id"].to_numpy()
    first = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ROW_START_NP = np.repeat(first, np.diff(np.r_[first, len(ids)]))
    hours = df["ts"].diff().dt.total_seconds().div(3600).to_numpy()
    odd = np.count_nonzero((hours != 1) & (np.arange(len(ids)) != ROW_START_NP))
    if odd:
        print(f"WARNING: {odd} rows are not exactly 1 hour after the previous row; windows assume hourly rows")

    X_dev = torch.from_numpy(X_seq).to(DEVICE)
    del X_seq
    ROW_START = torch.from_numpy(ROW_START_NP).to(DEVICE)
    OFFSETS = torch.arange(-(WINDOW_STEPS - 1) * STEP_HOURS, 1, STEP_HOURS, device=DEVICE)

    def windows(rows):
        # (batch, WINDOW_STEPS, inputs) histories ending at the given rows: ar_dl_model.window_rows, on the GPU.
        idx = rows[:, None] + OFFSETS[None, :]
        return X_dev[torch.maximum(idx, ROW_START[rows][:, None])]

    # Answers: failure within 30 days, within 7 days, and which part (only known for rows before a failure).
    comp_rows = (fit_mask | valid_mask) & df["failure_next_30d"].to_numpy() & df["next_failure_component"].notna().to_numpy()
    COMPONENTS = sorted(df.loc[comp_rows, "next_failure_component"].unique())
    part = df["next_failure_component"].map({c: i for i, c in enumerate(COMPONENTS)}).fillna(-1).to_numpy(np.int64)
    part[~df["failure_next_30d"].to_numpy()] = -1
    Y30 = torch.from_numpy(y_all.astype(np.float32)).to(DEVICE)
    Y7 = torch.from_numpy(df["failure_next_7d"].to_numpy(np.float32)).to(DEVICE)
    YC = torch.from_numpy(part).to(DEVICE)
    print(f"{len(FEATURES)} inputs x {WINDOW_STEPS} steps ({WINDOW_STEPS * STEP_HOURS / 24:.0f} days) per window | "
          f"{len(LOG_COLS)} inputs log-compressed | {X_dev.numel() * 4 / 1e9:.2f} GB on {DEVICE} | "
          f"parts: {COMPONENTS} | {time.time() - t0:.0f} s")
    """)
    code("""
    DAYS_AGO = (WINDOW_STEPS - 1 - np.arange(WINDOW_STEPS)) * STEP_HOURS / 24

    def day_axis(ax):
        # Label a window's step axis with days before the prediction.
        ticks = np.arange(WINDOW_STEPS - 1, -1, -max(1, 48 // STEP_HOURS))[::-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"-{d:.0f} d" if d else "now" for d in DAYS_AGO[ticks]])

    cand = np.flatnonzero(fit_mask & (y_all == 1) & df["days_to_next_failure"].between(1, 3).to_numpy())
    if len(cand):
        r = int(cand[len(cand) // 2])
        w = windows(torch.as_tensor([r], device=DEVICE))[0].cpu().numpy()
        top = np.argsort(w.std(axis=0))[::-1][:15]
        fig, ax = plt.subplots(figsize=(15, 5))
        im = ax.imshow(w[:, top].T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([FEATURES[i] for i in top])
        day_axis(ax)
        plt.colorbar(im, label="scaled value (0 = fit-month average)")
        ax.set_title(f"One window as the network sees it: {df.at[r, 'asset_id']}, "
                     f"{df.at[r, 'days_to_next_failure']:.1f} days before a {df.at[r, 'next_failure_component']} failure")
        plt.tight_layout()
        plt.show()
    """)

    md("""
    ## 6. Three networks
    Three common ways to read a sequence are tried:

    | network | how it reads the 14 days |
    |---|---|
    | **GRU** (gated recurrent unit) | Step by step, with a running memory, like an engineer scrolling through the log. A short convolution first smooths hour-to-hour noise. |
    | **TCN** (temporal convolutional network) | Slides pattern detectors along the sequence. Each layer looks twice as far back as the one before. Fast, and good at sudden changes. |
    | **Transformer** | Every step can look directly at every other step (self-attention), for example to compare today with 10 days ago. |

    All three end the same way:
    * The **latest step** and an **attention-weighted summary** of all steps are combined.
    * Three output "heads" read the result: failure within 30 days, failure within 7 days, and which part.

    Learning the 7-day and "which part" questions as well helps the network focus on what really comes before a failure.
    The code is in `ar_dl_model.py` (written in section 5).
    """)
    code("""
    # ARNet comes from ar_dl_model.py (section 5), the same code the dashboard uses.
    for arch in ARCHS:
        n = sum(p.numel() for p in ARNet(arch, len(FEATURES), len(COMPONENTS), HIDDEN, WINDOW_STEPS).parameters())
        print(f"{arch:<12} {n:>9,} weights")
    """)

    md("""
    ## 7. Train and pick the best network
    * Each epoch (training round) draws a fresh sample of windows from the fit months.
    * `POS_SHARE` of the windows end in the 30 days before a failure. Failures are rare, so without this the network
      would hardly ever see one. Section 8 corrects the probabilities afterwards.
    * The loss is: 30-day failure + ½ × 7-day failure + 0.3 × "which part" (counted only for windows before a failure).
    * After each epoch, the network is scored on validation windows (**PR-AUC**). The best epoch is kept.
    * Training stops when the score has not improved for `PATIENCE` epochs.
    * Every network in `ARCHS` is trained the same way. The one with the best validation PR-AUC is kept.
    """)
    code("""
    FIT_POS = np.flatnonzero(fit_mask & (y_all == 1))
    FIT_NEG = np.flatnonzero(fit_mask & (y_all == 0))
    VALID_ROWS, TEST_ROWS = np.flatnonzero(valid_mask), np.flatnonzero(test_mask)
    MONITOR = np.sort(np.random.default_rng(SEED).choice(VALID_ROWS, min(len(VALID_ROWS), 3_000 if SMOKE else 60_000),
                                                         replace=False))

    def sample_rows(rng):
        n_pos = int(SAMPLES_PER_EPOCH * POS_SHARE)
        rows = np.concatenate([rng.choice(FIT_POS, n_pos), rng.choice(FIT_NEG, SAMPLES_PER_EPOCH - n_pos)])
        return torch.from_numpy(rng.permutation(rows)).to(DEVICE)

    def loss_fn(out, rows):
        o30, o7, opart = out
        loss = (F.binary_cross_entropy_with_logits(o30.float(), Y30[rows])
                + 0.5 * F.binary_cross_entropy_with_logits(o7.float(), Y7[rows]))
        yc = YC[rows]
        known = yc >= 0
        if known.any():
            loss = loss + 0.3 * F.cross_entropy(opart[known].float(), yc[known])
        return loss

    @torch.no_grad()
    def predict(net, rows):
        # Raw 30-day failure probability and part probabilities for the windows that end at `rows`.
        net.eval()
        batch = 512 if SMOKE else 4096
        p30, parts = [], []
        for i in range(0, len(rows), batch):
            r = torch.as_tensor(rows[i:i + batch], device=DEVICE)
            with torch.autocast(DEVICE.type, enabled=AMP):
                o30, _, opart = net(windows(r))
            p30.append(torch.sigmoid(o30.float()).cpu())
            parts.append(torch.softmax(opart.float(), dim=1).cpu())
        return torch.cat(p30).numpy(), torch.cat(parts).numpy()

    def train(arch):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        net = ARNet(arch, len(FEATURES), len(COMPONENTS), HIDDEN, WINDOW_STEPS).to(DEVICE)
        steps = SAMPLES_PER_EPOCH // BATCH
        opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE, total_steps=EPOCHS * steps)
        scaler = torch.amp.GradScaler(DEVICE.type, enabled=AMP)
        best, best_state, history, waited, t0 = -1.0, None, [], 0, time.time()
        for epoch in range(1, EPOCHS + 1):
            net.train()
            rows, total = sample_rows(rng), 0.0
            for b in range(steps):
                r = rows[b * BATCH:(b + 1) * BATCH]
                with torch.autocast(DEVICE.type, enabled=AMP):
                    out = net(windows(r))
                loss = loss_fn(out, r)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.detach()
            score = safe_ap(y_all[MONITOR], predict(net, MONITOR)[0])
            score = 0.0 if np.isnan(score) else score
            history.append({"arch": arch, "epoch": epoch, "train_loss": float(total) / steps,
                            "valid_pr_auc": score, "minutes": (time.time() - t0) / 60})
            print(f"  {arch:<11} epoch {epoch:>2}  loss {history[-1]['train_loss']:.4f}  "
                  f"validation PR-AUC {score:.3f}  ({history[-1]['minutes']:.1f} min)")
            if score > best:
                best, waited = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        return net, history

    runs, history = {}, []
    for arch in ARCHS:
        net, hist = train(arch)
        h = pd.DataFrame(hist)
        best_row = h.loc[h["valid_pr_auc"].idxmax()]
        runs[arch] = {"net": net, "weights": sum(p.numel() for p in net.parameters()), "epochs run": len(h),
                      "best epoch": int(best_row["epoch"]), "best validation PR-AUC": round(float(best_row["valid_pr_auc"]), 3),
                      "minutes": round(float(h["minutes"].iloc[-1]), 1)}
        history += hist
    selection = pd.DataFrame([{"network": a, **{k: v for k, v in r.items() if k != "net"}} for a, r in runs.items()])
    selection = selection.set_index("network")
    BEST_ARCH = selection["best validation PR-AUC"].idxmax()
    model = runs[BEST_ARCH]["net"]
    for r in runs.values():
        r["net"] = None
    if AMP:
        torch.cuda.empty_cache()
    display(selection)
    print(f"kept: {BEST_ARCH} (random guess scores {y_all[MONITOR].mean():.3f})")
    """)
    code("""
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for arch, h in hist.groupby("arch"):
        axes[0].plot(h["epoch"], h["train_loss"], "o-", label=arch)
        axes[1].plot(h["epoch"], h["valid_pr_auc"], "o-", label=arch)
    axes[1].axhline(y_all[MONITOR].mean(), color="grey", ls=":", label="random guess")
    axes[0].set(title="Training loss (lower = better)", xlabel="epoch")
    axes[1].set(title="Validation PR-AUC (higher = better)", xlabel="epoch")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()
    raw_val, _ = predict(model, VALID_ROWS)
    y_val = y_all[valid_mask]
    print(f"{BEST_ARCH}: PR-AUC on all validation rows {safe_ap(y_val, raw_val):.3f}")
    """)

    md("""
    ## 8. Realistic probabilities (calibration)
    Training used 30 % failure windows, far more than in reality (about 8 %), so the raw network scores are too high.
    **Isotonic calibration** on the validation months maps them to probabilities that match how often failures
    really followed. This is the same code as in the LightGBM notebook.
    """)
    code(CALIBRATE_CODE)

    md("""
    ## 9. Alarm level (threshold)
    The same rule as the LightGBM notebook. On the validation months, it picks the level that catches the most
    failures while staying within `FALSE_ALARM_BUDGET` false-alarm days per recloser per month. If no level fits the
    budget, the level with the best F1 is used.
    """)
    code(THRESHOLD_CODE)

    md("""
    ## 10. Honest test: deep learning vs LightGBM vs Health Index
    All numbers below come from the **test months**, which no model saw.
    * **LightGBM** appears when `ar_ml_model.zip` is found in Drive/AR_ML, or when the notebook runs inside the
      project, where it uses `models/`.
    * **ROC-AUC / PR-AUC:** how well risky rows are ranked above safe ones. PR-AUC is the stricter one for rare events.
    * **Failures caught:** a warning appeared in the 30 days before the failure.
    * **Warning days:** how early the first warning came.
    * **"DL or …":** warns when either one does. It shows what you gain by using both.
    """)
    code("""
    test = df.loc[test_mask, ["asset_id", "ts", "health_index", "failure_next_30d", "next_failure_component"]].copy()
    raw_test, part_test = predict(model, TEST_ROWS)
    test["p"] = calibrate(raw_test)
    test_fails = failures_between(TEST_START, t_last + GAP)

    def load_lgbm():
        for z in (LGBM_ZIP, f"/content/drive/MyDrive/{DRIVE_FOLDER}/{LGBM_ZIP}", f"/content/{LGBM_ZIP}"):
            if os.path.exists(z):
                with zipfile.ZipFile(z) as fh:
                    fh.extractall("lgbm_bundle")
                break
        folder = next((d for d in ("lgbm_bundle", os.path.join("..", "models"))
                       if os.path.exists(os.path.join(d, "ml_meta.json"))
                       and os.path.exists(os.path.join(d, "model_failure30d.txt"))), None)
        if folder is None:
            print(f"No LightGBM bundle found: put {LGBM_ZIP} in Drive/{DRIVE_FOLDER} to add it to the comparison.")
            return None
        with open(os.path.join(folder, "ml_meta.json")) as fh:
            meta = json.load(fh)
        if meta.get("feature_version") != FEATURE_VERSION or not set(meta["features"]) <= set(FEATURES):
            print("The LightGBM bundle uses another feature recipe; retrain it with AR_ML_Colab.ipynb. Skipped.")
            return None
        if str(meta.get("data_until_utc")) != str(df["ts"].max()):
            print(f"NOTE: the LightGBM bundle was trained on data up to {meta.get('data_until_utc')}, this file ends "
                  f"{df['ts'].max()}. The comparison is only fair when both notebooks used the same data file.")
        comp_path = os.path.join(folder, "model_component.txt")
        comp = lgb.Booster(model_file=comp_path) if os.path.exists(comp_path) and meta.get("component_classes") else None
        print(f"LightGBM bundle {meta.get('model_version')} from {folder}")
        return {"booster": lgb.Booster(model_file=os.path.join(folder, "model_failure30d.txt")),
                "meta": meta, "component": comp}

    LGBM = load_lgbm()
    if LGBM:
        X_test_lgbm = X_all.loc[test_mask, LGBM["meta"]["features"]].to_numpy(np.float32)
        cal = LGBM["meta"]["calibration"]
        test["p_lgbm"] = np.clip(np.interp(LGBM["booster"].predict(X_test_lgbm), cal["x"], cal["y"]), 0.0, 1.0)
        LGBM_THRESHOLD = float(LGBM["meta"]["threshold"])

    DL = f"Deep learning ({BEST_ARCH})"
    HI = "Health Index (HI < 60)"
    dl_warn, hi_warn = test["p"] >= THRESHOLD, test["health_index"] < HI_HEALTHY_MIN
    dl_rel = test["p"] / max(THRESHOLD, 1e-9)                       # >= 1 means the network warns
    hi_rel = (100 - test["health_index"]) / (100 - HI_HEALTHY_MIN)  # > 1 means the HI warns
    systems = {DL: (test["p"], dl_warn)}
    if LGBM:
        lgbm_warn = test["p_lgbm"] >= LGBM_THRESHOLD
        systems["LightGBM"] = (test["p_lgbm"], lgbm_warn)
    systems[HI] = (100 - test["health_index"], hi_warn)
    if LGBM:
        systems["DL or LightGBM"] = (np.maximum(dl_rel, test["p_lgbm"] / max(LGBM_THRESHOLD, 1e-9)), dl_warn | lgbm_warn)
    systems["DL or HI"] = (np.maximum(dl_rel, hi_rel), dl_warn | hi_warn)

    results, per_failure = {}, {}
    for name, (score, warn) in systems.items():
        per, fa = event_metrics(test, warn, test_fails)
        per_failure[name] = per
        results[name] = {"ROC-AUC": round(safe_auc(test["failure_next_30d"], score), 3),
                         "PR-AUC": round(safe_ap(test["failure_next_30d"], score), 3),
                         "failures caught": f"{int(per['caught'].sum())} / {len(per)}",
                         "median warning days": round(per["warning_days"].median(), 1) if len(per) else np.nan,
                         "false-alarm days per AR-month": round(fa, 2)}
    comparison = pd.DataFrame(results).T
    print(f"random-guess PR-AUC = {test['failure_next_30d'].mean():.3f} | "
          f"Brier score of the network (lower = better) = {brier_score_loss(test['failure_next_30d'], test['p']):.4f}")
    display(comparison)
    by_comp = pd.DataFrame({name: per.groupby("component")["caught"].agg(lambda s: f"{int(s.sum())} / {len(s)}")
                            for name, per in per_failure.items()})
    by_comp.index.name = "failures caught by component"
    display(by_comp)
    """)
    code("""
    y_test = test["failure_next_30d"].to_numpy()
    probs = [(DL, "p", "#e67e22")] + ([("LightGBM", "p_lgbm", "#00a3cc")] if LGBM else [])
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    if 0 < y_test.sum() < len(y_test):
        for name, score in [(n, test[c]) for n, c, _ in probs] + [("Health Index", 100 - test["health_index"])]:
            pr, rc, _ = precision_recall_curve(y_test, score)
            axes[0].plot(rc, pr, label=f"{name} (PR-AUC {safe_ap(y_test, score):.2f})")
    axes[0].axhline(y_test.mean(), color="grey", ls=":", label="random guess")
    axes[0].set(xlabel="recall", ylabel="precision", title="Precision vs recall (test)")
    axes[0].legend()
    for name, _, color in probs:
        axes[1].hist(per_failure[name]["warning_days"].dropna(), bins=15, alpha=0.6, color=color, label=name)
    axes[1].set(title="Warning days before caught failures", xlabel="days")
    axes[1].legend()
    if y_test.sum() > 0:
        for name, col, color in probs:
            frac, mean_p = calibration_curve(y_test, test[col], n_bins=10, strategy="quantile")
            axes[2].plot(mean_p, frac, "o-", color=color, label=name)
    axes[2].plot([0, 1], [0, 1], ":", color="grey", label="perfect")
    axes[2].set(title="Calibration (test)", xlabel="predicted probability", ylabel="observed frequency")
    axes[2].legend()
    plt.tight_layout()
    plt.show()
    """)

    md("""
    ## 11. One recloser over time
    The HI (blue, left axis) and the failure probabilities (right axis) for the test recloser where the network and
    the HI together caught the most failures. Black lines are recorded failures; dashed lines are the alarm levels.
    """)
    code("""
    if len(test_fails):
        both = per_failure[DL].groupby("asset_id")["caught"].sum() + per_failure[HI].groupby("asset_id")["caught"].sum()
        aid = both.sort_values(kind="stable").index[-1]
        cols = ["health_index", "p"] + (["p_lgbm"] if LGBM else [])
        a = test[test["asset_id"] == aid].set_index("ts")[cols].resample("1h").asfreq()
        fig, ax1 = plt.subplots(figsize=(15, 4))
        ax1.plot(a.index, a["health_index"], color="tab:blue", lw=1)
        ax1.axhline(HI_HEALTHY_MIN, color="tab:blue", ls="--", lw=0.8)
        ax1.set_ylim(0, 102)
        ax1.set_ylabel("Health Index", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(a.index, a["p"], color="tab:red", lw=1, label=DL)
        ax2.axhline(THRESHOLD, color="tab:red", ls="--", lw=0.8)
        if LGBM:
            ax2.plot(a.index, a["p_lgbm"], color="tab:orange", lw=0.8, alpha=0.8, label="LightGBM")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("failure probability (30 days)", color="tab:red")
        ax2.legend(loc="upper left")
        for k, fl in enumerate(test_fails[test_fails["asset_id"] == aid].itertuples()):
            ax1.axvline(fl.failure_time, color="black", lw=1.5)
            ax1.text(fl.failure_time, 3 + 9 * (k % 2), f" {fl.component}", fontsize=9)
        ax1.set_title(f"{aid}: Health Index vs failure probability (test months)")
        plt.show()
    """)

    md("""
    ## 12. Which part is likely to fail?
    The network's third head names the most likely part. It is checked on the test rows in the 30 days before a
    failure, and compared with LightGBM's separate component model when that is available.
    """)
    code("""
    pos = y_test & test["next_failure_component"].isin(COMPONENTS).to_numpy()
    if len(COMPONENTS) >= 2 and pos.any():
        predicted = np.array(COMPONENTS)[part_test[pos].argmax(axis=1)]
        actual = test.loc[pos, "next_failure_component"].to_numpy()
        print(f"{DL}: right part on {(predicted == actual).mean():.0%} of test rows before failures")
        if LGBM and LGBM["component"] is not None:
            lgbm_parts = np.array(LGBM["meta"]["component_classes"])[LGBM["component"].predict(X_test_lgbm[pos]).argmax(axis=1)]
            print(f"LightGBM component model: right part on {(lgbm_parts == actual).mean():.0%} of the same rows")
        display(pd.crosstab(pd.Series(actual, name="actual"), pd.Series(predicted, name="predicted (deep learning)")))
    else:
        print("Not enough different failing parts in the training months to name the part.")
    """)

    md("""
    ## 13. What does the network look at?
    Neural networks have no built-in SHAP values like LightGBM, so **gradients** are used. They measure how much each
    input, at each step, pushes the 30-day risk (log-odds) up or down. The reference is an "average recloser": all
    inputs at their fit-month average.
    * **Left:** the inputs that move the risk most, averaged over test windows (gradient × input).
    * **Right:** the same, per step. It shows how far back in the 14 days the network looks.
    * **Next chart:** **integrated gradients** for the first warning before one caught failure. Red pushed the risk
      up, blue pushed it down.
    """)
    code("""
    def grad_x_input(net, x):
        net.eval()
        x = x.detach().clone().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):             # cuDNN cannot back-propagate a GRU in eval mode
            net(x)[0].sum().backward()
        return (x.grad * x).detach()

    def integrated_gradients(net, x, n=32):
        # For one window x (1 x steps x inputs), starting from the all-average window.
        net.eval()
        path = (torch.linspace(1 / n, 1, n, device=x.device)[:, None, None] * x).requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            net(path)[0].sum().backward()
        return (path.grad.mean(dim=0) * x[0]).detach().cpu().numpy()

    sample = np.random.default_rng(SEED).choice(TEST_ROWS, min(len(TEST_ROWS), 256 if SMOKE else 4096), replace=False)
    total = 0
    for i in range(0, len(sample), 256):
        total = total + grad_x_input(model, windows(torch.as_tensor(sample[i:i + 256], device=DEVICE))).abs().sum(dim=0)
    total = (total / len(sample)).cpu().numpy()                     # steps x inputs
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.3]})
    by_input = pd.Series(total.sum(axis=0), index=FEATURES).sort_values(ascending=False).head(20)
    by_input[::-1].plot.barh(ax=axes[0], color="#00a3cc", title="Top 20 inputs (mean |gradient x input|)")
    axes[1].plot(np.arange(WINDOW_STEPS), total.sum(axis=1), color="#e67e22")
    day_axis(axes[1])
    axes[1].set(title="How far back the network looks", ylabel="mean |gradient x input|")
    plt.tight_layout()
    plt.show()
    """)
    code("""
    caught = per_failure[DL][per_failure[DL]["caught"]]
    if len(caught):
        ex = caught.iloc[0]
        g = test[(test["asset_id"] == ex.asset_id) & (test["p"] >= THRESHOLD) & (test["ts"] < ex.failure_time)
                 & (test["ts"] > ex.failure_time - pd.Timedelta(days=HORIZON_DAYS))]
        row = int(g.index[0])
        ig = integrated_gradients(model, windows(torch.as_tensor([row], device=DEVICE)))
        top = np.argsort(np.abs(ig).sum(axis=0))[::-1][:12]
        lim = np.abs(ig[:, top]).max() or 1.0
        fig, ax = plt.subplots(figsize=(15, 5))
        im = ax.imshow(ig[:, top].T, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([FEATURES[i] for i in top])
        day_axis(ax)
        plt.colorbar(im, label="push on the risk (red = up)")
        ax.set_title(f"{ex.asset_id}: first warning {test.loc[row, 'ts']:%Y-%m-%d %H:%M} before a {ex.component} "
                     f"failure on {ex.failure_time:%Y-%m-%d}")
        plt.tight_layout()
        plt.show()
    """)

    md("""
    ## 14. Daily view: HI vs deep learning per recloser per day
    The same daily comparison the project stores in `ar_daily_summary`, here with the network's probability:
    * HI healthy = daily average HI ≥ 60;
    * ML healthy = probability below the alarm level all day.
    """)
    code(DAILY_CODE)

    md("""
    ## 15. Export the network
    This writes `ar_dl_model.zip` with:
    * `model_dl.pt`: the network weights (PyTorch) and its size settings;
    * `dl_meta.json`: everything needed to rebuild the inputs and read the output. That is the feature list, scaling,
      window length, alarm level, calibration, part names and the test results.

    **Install it on the PC:** unzip it into the project's **`models/dl/`** folder (not `models/`, which holds the
    LightGBM model), then run `python main.py dl-score`. The dashboard's **Deep Learning** tab then shows the network's
    risk next to LightGBM and the HI, and the collector scores every 15 minutes. A copy is saved in Drive/AR_ML when
    Drive is connected.
    """)
    code("""
    os.makedirs("export_dl", exist_ok=True)
    for old in os.listdir("export_dl"):
        os.remove(os.path.join("export_dl", old))
    NET_CONFIG = {"arch": BEST_ARCH, "n_inputs": len(FEATURES), "n_components": len(COMPONENTS),
                  "width": HIDDEN, "steps": WINDOW_STEPS}
    torch.save({"config": NET_CONFIG, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
               "export_dl/model_dl.pt")
    dl_res = results[DL]
    metrics = {"test_roc_auc": dl_res["ROC-AUC"], "test_pr_auc": dl_res["PR-AUC"],
               "test_failures_caught": dl_res["failures caught"],
               "test_median_warning_days": dl_res["median warning days"],
               "test_false_alarm_days_per_ar_month": dl_res["false-alarm days per AR-month"],
               "hi_failures_caught": results[HI]["failures caught"],
               "dl_or_hi_failures_caught": results["DL or HI"]["failures caught"],
               "validation_pr_auc": float(selection.loc[BEST_ARCH, "best validation PR-AUC"])}
    if LGBM:
        metrics["lightgbm_failures_caught"] = results["LightGBM"]["failures caught"]
        metrics["dl_or_lightgbm_failures_caught"] = results["DL or LightGBM"]["failures caught"]
    now = dt.datetime.now(dt.timezone.utc)
    meta = {"model_type": "pytorch-sequence", "net_version": NET_VERSION, "arch": BEST_ARCH, "net_config": NET_CONFIG,
            "feature_version": FEATURE_VERSION, "features": FEATURES,
            "window_steps": WINDOW_STEPS, "step_hours": STEP_HOURS,
            "scaling": {"log1p": LOG_COLS, "mean": MU.tolist(), "std": SD.tolist(), "clip": CLIP},
            "threshold": THRESHOLD, "calibration": {"x": CAL_X, "y": CAL_Y},
            "component_classes": COMPONENTS, "metrics": metrics,
            "selection": selection.reset_index().to_dict("records"), "horizon_days": HORIZON_DAYS,
            "trained_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "data_until_utc": str(df["ts"].max()),
            "torch_version": torch.__version__, "model_version": f"dl-{BEST_ARCH}-" + now.strftime("%Y%m%d-%H%M")}
    with open("export_dl/dl_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with zipfile.ZipFile("ar_dl_model.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir("export_dl")):
            z.write(os.path.join("export_dl", fn), fn)
    print("ar_dl_model.zip:", sorted(os.listdir("export_dl")), "| model version", meta["model_version"])
    display(pd.Series(metrics).to_frame("test result"))
    drive_dir = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
    if IN_COLAB and not SMOKE:
        if os.path.isdir(drive_dir):
            shutil.copy("ar_dl_model.zip", drive_dir)
            print("copy saved in Drive:", drive_dir)
        from google.colab import files
        files.download("ar_dl_model.zip")
    """)
    code("""
    # Check: rebuild the network from the exported files alone and score a few test rows from the raw inputs,
    # with the functions the dashboard uses (ar_dl_model.window_rows and scale_inputs).
    saved = torch.load("export_dl/model_dl.pt", map_location="cpu", weights_only=True)
    with open("export_dl/dl_meta.json") as fh:
        m = json.load(fh)
    check = ARNet(**saved["config"]).eval()
    check.load_state_dict(saved["state_dict"])
    s = m["scaling"]
    log_idx = [m["features"].index(c) for c in s["log1p"]]
    rows = TEST_ROWS[-5:]
    win = window_rows(rows, ROW_START_NP[rows], m["window_steps"], m["step_hours"])     # hourly rows
    x = np.stack([scale_inputs(X_all.iloc[w][m["features"]].to_numpy(np.float32), log_idx, s["mean"], s["std"],
                               s["clip"]) for w in win])
    with torch.no_grad():
        raw_check = torch.sigmoid(check(torch.from_numpy(x))[0]).numpy()
    print("raw scores from the bundle:", np.round(raw_check, 4), "| in the notebook:", np.round(raw_test[-5:], 4))
    assert np.allclose(raw_check, raw_test[-5:], atol=0.01), "the exported bundle does not reproduce the notebook"
    print("probabilities:", np.round(np.interp(raw_check, m["calibration"]["x"], m["calibration"]["y"]), 4))
    """)
    md("""
    ### Reading the results honestly
    * **The data is simulated.** The network learns the patterns the simulator built in. Test it on real NOJA data
      before relying on it.
    * **Deep learning is not automatically better.** On table data like this, gradient-boosted trees (LightGBM) are
      often as good or better. Judge only by the section 10 test table, not by the validation scores.
    * **Different strengths.** If the network and LightGBM catch different failures, "DL or LightGBM" shows the gain
      of using both.
    * **Results vary a little between runs** on a GPU. For a firm conclusion, run the notebook twice and compare.
    * **Retrain** when more data is available: generate a new file on the PC, then run both notebooks again on the same file.
      Replace the files in `models/dl/` and run `python main.py dl-score`.
    """)
    return cells


def build(path: Path) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = _cells()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}, "accelerator": "GPU",
                      "colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "AR_DL_Colab.ipynb")
    print("wrote", out)
