"""Command line interface: python main.py <command> [options]."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd

from .config import load_config, ml_enabled
from .logging_setup import setup_logging
from .sim.timeutil import dt_to_ms, local_to_ms, parse_date

log = logging.getLogger("ar_gen")


def _parse_when(text: str | None, cfg: dict) -> int | None:
    """YYYY-MM-DD (local midnight) or an ISO datetime in UTC -> ms."""
    if not text:
        return None
    if len(text) == 10:
        return local_to_ms(parse_date(text), 0.0, cfg["timezone"])
    return dt_to_ms(datetime.fromisoformat(text))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python main.py",
                                description="NOJA OSM auto recloser data getter, V1-V42 and Health Index")
    p.add_argument("--config", help="path to config.yaml (default: project config.yaml or $AR_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-db", help="create the database, tables and indexes")
    s.add_argument("--reset", action="store_true", help="DROP all tables and delete the simulator state")
    s.add_argument("--yes", action="store_true", help="do not ask for confirmation with --reset")

    s = sub.add_parser("seed", help="create the recloser fleet and hidden profiles")
    s.add_argument("--assets", type=int, help="number of reclosers (default: fleet.n_assets)")
    s.add_argument("--start", help="first day YYYY-MM-DD (default: today - default_backfill_days)")
    s.add_argument("--days", type=int, help="start this many days before today")

    s = sub.add_parser("backfill", help="generate history up to now: raw -> V1-V42 -> HI")
    s.add_argument("--end", help="stop at YYYY-MM-DD or ISO UTC datetime (default: now)")
    s.add_argument("--no-compute", action="store_true", help="raw data only")

    sub.add_parser("run-once", help="collect the interval(s) that just ended and update V1-V42 + HI")
    sub.add_parser("scheduler", help="real-time generator in this console (same as 'service run')")
    s = sub.add_parser("service", help="real-time data generator as a Windows service: install, start, stop, "
                                      "restart, status, remove; or run it in this console")
    s.add_argument("action", choices=["install", "start", "stop", "restart", "status", "remove", "run"])
    s.add_argument("--no-start", action="store_true", help="install: register the service but do not start it now")
    s = sub.add_parser("package-service", help="zip of just the real-time generator service, to send to someone "
                                              "(no dashboard, ML, notebooks or data)")
    s.add_argument("--out", help="output folder (default: dist)")
    sub.add_parser("status", help="latest Health Index of every recloser")
    s = sub.add_parser("live", help="show the newest 15-min interval of every recloser; refreshes when new data arrives")
    s.add_argument("--once", action="store_true", help="print once and exit")
    s.add_argument("--refresh", type=int, default=30, help="seconds between database checks (default 30)")
    sub.add_parser("validate", help="check gaps, ranges, counters, sequences, coverage")

    s = sub.add_parser("export-dataset", help="ML dataset (CSV.gz) with failure labels")
    s.add_argument("--out", help="output directory (default: output/ml)")

    s = sub.add_parser("dashboard", help="open the 3D recloser health map in your browser")
    s.add_argument("--port", type=int, default=8501, help="web port (default 8501)")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    s.add_argument("--lan", action="store_true",
                   help="also allow other computers on your network (anyone there can save resets)")

    s = sub.add_parser("maintenance-done", help="record maintenance and reset the repaired part's Health Index")
    s.add_argument("--asset", required=True, help="recloser, e.g. AR-008")
    s.add_argument("--part", action="append", required=True,
                   help="BATTERY, SWITCHING, COMMS, CONTROLLER or ALL (repeat for several)")
    s.add_argument("--at", help='local time "YYYY-MM-DD HH:MM" (Asia/Colombo); default: latest interval')
    s.add_argument("--note", default="", help="technician note")

    s = sub.add_parser("set-locations", help="load real GPS positions (CSV: asset_id,latitude,longitude)")
    s.add_argument("--csv", required=True, help="path to the CSV file")

    sub.add_parser("ml-status", help="show whether a trained ML model is installed (models folder)")
    s = sub.add_parser("ml-score", help="score stored V1-V42 rows with the installed ML model")
    s.add_argument("--start", help="YYYY-MM-DD or ISO UTC (default: beginning)")
    s.add_argument("--end", help="YYYY-MM-DD or ISO UTC (default: last collected)")

    sub.add_parser("dl-status", help="show whether a deep-learning model is installed (models/dl folder)")
    s = sub.add_parser("dl-score", help="score stored V1-V42 rows with the installed deep-learning model")
    s.add_argument("--start", help="YYYY-MM-DD or ISO UTC (default: beginning)")
    s.add_argument("--end", help="YYYY-MM-DD or ISO UTC (default: last collected)")

    s = sub.add_parser("prune-history", help="delete old data, keeping the last N days (full backup first)")
    s.add_argument("--keep-days", type=int, default=30, help="days of data to keep (default 30)")
    s.add_argument("--backup", help="backup file (default: data/ar_maintenance_backup_<date>.sqlite)")
    s.add_argument("--no-backup", action="store_true", help="do not write a backup first")

    s = sub.add_parser("generate-ml-data", help="generate an ML training dataset directly (no database)")
    s.add_argument("--assets", type=int, default=100, help="number of reclosers (default 100)")
    s.add_argument("--days", type=int, default=730, help="days of history (default 730 = 2 years)")
    s.add_argument("--seed", type=int, help="random seed (default: config seed + 1, independent of the live fleet)")
    s.add_argument("--start", help="first day YYYY-MM-DD (default: today - days)")
    s.add_argument("--out", help="output folder (default: output/ml_generated)")
    s.add_argument("--no-hourly", action="store_true", help="skip the smaller hourly copy of the dataset")

    s = sub.add_parser("copy-db", help="copy all data from another database into DATABASE_URL")
    s.add_argument("--from", dest="source_url", required=True,
                   help="source SQLAlchemy URL, e.g. sqlite:///data/ar_maintenance.sqlite")

    for name, text in (("compute-variables", "recompute V1-V42 + HI for a range"),
                       ("compute-hi", "re-score stored variables (after changing HI settings)")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--start", help="YYYY-MM-DD or ISO UTC (default: beginning)")
        s.add_argument("--end", help="YYYY-MM-DD or ISO UTC (default: last collected)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    if getattr(args, "assets", None):
        overrides = {"fleet": {"n_assets": args.assets}}
    cfg = load_config(args.config, overrides=overrides or None)
    setup_logging(cfg, args.verbose)
    try:
        return COMMANDS[args.command](cfg, args) or 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    except ModuleNotFoundError as exc:             # e.g. 'dashboard' in the service-only package
        log.error("'%s' needs %s, which is not installed in this copy of the project", args.command, exc.name)
        return 2


# ------------------------------------------------------------------ commands
def cmd_init_db(cfg, args):
    from . import pipeline
    if args.reset and not args.yes and sys.stdin.isatty():
        answer = input("This deletes ALL recloser data in the database. Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            print("Cancelled.")
            return 1
    pipeline.init_db(cfg, reset=args.reset)


def cmd_seed(cfg, args):
    from . import pipeline
    from .sim.timeutil import local_date
    start = args.start
    if start is None and args.days is not None:
        start = local_date(pipeline.default_start_ms(cfg, args.days), cfg["timezone"]).isoformat()
    pipeline.seed(cfg, start)


def cmd_backfill(cfg, args):
    from . import pipeline
    end = _parse_when(args.end, cfg)
    res = pipeline.update(cfg, until_ms=end, compute=not args.no_compute)
    log.info("backfill done: %d intervals collected, %d variable rows, data up to %s UTC",
             res["intervals"], res["variable_rows"], res["last_ts"])


def cmd_run_once(cfg, args):
    from . import pipeline
    res = pipeline.update(cfg)
    log.info("run-once: %d interval(s) collected, %d variable rows, data up to %s UTC",
             res["intervals"], res["variable_rows"], res["last_ts"])
    _log_status(cfg)


def _log_status(cfg):
    from .generator import log_health
    log_health(cfg)


def cmd_scheduler(cfg, args):
    """The real-time generator in this console: updates at :00 :15 :30 :45 (+10 s) until Ctrl+C."""
    import threading

    from .generator import run_generator
    stop = threading.Event()
    log.info("Real-time generator in this console. Ctrl+C to stop. "
             "(To run it in the background at every boot instead: python main.py service install)")
    try:
        run_generator(cfg, stop, mode="console")
    except KeyboardInterrupt:
        stop.set()


def cmd_package_service(cfg, args):
    from pathlib import Path

    from .packaging import build_service_package
    path = build_service_package(Path(args.out) if args.out else None)
    log.info("service package written: %s (%.0f kB). Send this zip; the receiver follows README.md inside it.",
             path, path.stat().st_size / 1024)


def cmd_service(cfg, args):
    if args.action == "run":
        return cmd_scheduler(cfg, args)
    from .service import manage
    return manage(args.action, cfg, start=not args.no_start)


def cmd_status(cfg, args):
    from . import pipeline
    df = pipeline.status(cfg)
    if df.empty:
        print("No Health Index rows yet - run backfill or run-once.")
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 500):
        print(df[["asset_id", "ts", "health_index", "health_class", "is_healthy", "cap_applied",
                  "worst_variables", "f_battery", "f_faults", "f_switching", "f_comms", "f_controlling",
                  "f_aging", "f_environment"]].round(1).to_string(index=False))


def cmd_live(cfg, args):
    import time

    from . import pipeline
    from .sim.timeutil import to_local
    from .variables import to_ms
    engine = pipeline.get_engine(cfg)
    shown = None
    try:
        while True:
            ts, df = pipeline.live_snapshot(cfg, engine)
            if ts is not None and ts != shown:
                local = to_local(int(to_ms([ts])[0]), cfg["timezone"])
                n_bad = int((~df["is_healthy"].astype(bool)).sum())
                print(f"\n=== interval ending {local:%Y-%m-%d %H:%M} {cfg['timezone']} ({ts:%H:%M} UTC) | "
                      f"{len(df)} reclosers, {n_bad} not healthy | shown at {datetime.now():%H:%M:%S} ===")
                view = df.rename(columns={"switch_state": "switch", "battery_voltage": "batt_V",
                                          "battery_current": "batt_A", "battery_charge_pct": "charge%",
                                          "ac_supply_on": "AC", "comms_connected": "comms",
                                          "signal_strength_dbm": "signal", "health_index": "HI",
                                          "health_class": "class", "worst_variables": "worst"})
                view = view.drop(columns=["is_healthy"])
                with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 500):
                    print(view.round(2).to_string(index=False, na_rep="-"))
                shown = ts
                sys.stdout.flush()
            if args.once:
                return 0
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        return 0


def cmd_validate(cfg, args):
    from .validate import hi_before_failures, validate
    checks, counts = validate(cfg)
    print("\nRow counts")
    for name, n in counts.items():
        print(f"  {name:24s} {n:>12,}")
    print("\nChecks")
    for c in checks:
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
    hb = hi_before_failures(cfg)
    if not hb.empty:
        hb = hb.dropna(subset=["hi_7d_before"])
        print("\nHealth Index in the 7 days before recorded failures (vs the asset's mean HI)")
        summary = hb.groupby("component").agg(failures=("asset_id", "size"), hi_7d_before=("hi_7d_before", "mean"),
                                              hi_min_7d=("hi_min_7d", "mean"), hi_asset_mean=("hi_asset_mean", "mean"))
        print(summary.round(1).to_string())
        print(f"  all failures: HI 7 d before = {hb['hi_7d_before'].mean():.1f}, "
              f"asset mean = {hb['hi_asset_mean'].mean():.1f}")
    ok = all(c.ok for c in checks)
    print(f"\nValidation {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def cmd_export_dataset(cfg, args):
    from .dataset import export_dataset
    res = export_dataset(cfg, args.out)
    for k, v in res.items():
        log.info("%s: %s", k, v)


def cmd_dashboard(cfg, args):
    import os
    import subprocess
    import threading
    import webbrowser

    from .config import ROOT
    app = ROOT / "ar_gen" / "dashboard_app.py"
    if not app.exists():
        log.error("The dashboard is not part of this copy of the project (service-only package).")
        return 2
    url = f"http://localhost:{args.port}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.headless", "true", "--browser.gatherUsageStats", "false",
           "--server.address", "0.0.0.0" if args.lan else "localhost"]
    log.info("Dashboard starting at %s (Ctrl+C to stop)", url)
    if not args.no_browser:
        threading.Timer(4.0, webbrowser.open, args=(url,)).start()
    try:
        return subprocess.call(cmd, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        return 0


def cmd_maintenance_done(cfg, args):
    from datetime import datetime

    from . import pipeline
    from .sim.timeutil import dt_to_ms, zone
    at_ms = None
    if args.at:
        try:
            local = datetime.strptime(args.at.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=zone(cfg["timezone"]))
        except ValueError:
            log.error('--at must look like "2026-10-06 14:30" (local time)')
            return 2
        at_ms = dt_to_ms(local)
    res = pipeline.maintenance_done(cfg, args.asset, args.part, at_ms, args.note)
    log.info("reset %s saved for %s: HI %s -> %s (%d rows recalculated)", res["reset_id"], args.asset,
             res["hi_before"], res["hi_after"], res["recomputed_rows"])


def _ml_switched_off(cfg) -> bool:
    if ml_enabled(cfg):
        return False
    print("ML is switched off (config.yaml: ml: enabled: false). The collector and the dashboard run without it; "
          "set it to true to use the ML models again.")
    return True


def cmd_ml_status(cfg, args):
    if _ml_switched_off(cfg):
        return 0
    from .ml_scoring import load_model
    folder = cfg["paths"]["models_dir"]
    try:
        model = load_model(folder)
    except (ValueError, RuntimeError) as exc:
        print(f"The ML model in {folder} cannot be used: {exc}")
        return 2
    if model is None:
        print(f"No ML model installed in {folder}.\n"
              "Train it in Google Colab (docs/ML_COLAB_GUIDE.md) and unzip ar_ml_model.zip into that folder.")
        return 0
    meta = model.meta
    print(f"ML model {model.model_version} - trained {meta.get('trained_utc')}, data until {meta.get('data_until_utc')}")
    print(f"  alarm threshold: {model.threshold:.3f} | features: {len(model.features)} | "
          f"component model: {'yes' if model.component_booster is not None else 'no'}")
    for key, value in (meta.get("metrics") or {}).items():
        print(f"  {key}: {value}")


def cmd_ml_score(cfg, args):
    if _ml_switched_off(cfg):
        return 2
    from . import pipeline
    from .ml_scoring import load_model
    model = load_model(cfg["paths"]["models_dir"])
    if model is None:
        log.error("No ML model installed in %s - see docs/ML_COLAB_GUIDE.md", cfg["paths"]["models_dir"])
        return 2
    with pipeline.run_lock(cfg, wait_s=pipeline.LOCK_WAIT_S):
        engine, start, end = _range(cfg, args)
        n = pipeline.score_and_summarise(cfg, engine, start, end, model)
    log.info("scored %d rows with model %s", n, model.model_version)


def cmd_dl_status(cfg, args):
    if _ml_switched_off(cfg):
        return 0
    from .dl_scoring import load_dl_model
    folder = cfg["paths"]["dl_models_dir"]
    try:
        model = load_dl_model(folder)
    except (ValueError, RuntimeError) as exc:
        print(f"The deep-learning model in {folder} cannot be used: {exc}")
        return 2
    if model is None:
        print(f"No deep-learning model installed in {folder}.\n"
              "Train it in Google Colab with notebooks/AR_DL_Colab.ipynb (docs/ML_COLAB_GUIDE.md) and unzip "
              "ar_dl_model.zip into that folder.")
        return 0
    meta = model.meta
    print(f"Deep-learning model {model.model_version} ({meta.get('arch', '?')}) - trained {meta.get('trained_utc')}, "
          f"data until {meta.get('data_until_utc')}")
    print(f"  alarm threshold: {model.threshold:.3f} | inputs: {len(meta['features'])} | window: "
          f"{meta['window_steps']} steps x {meta['step_hours']} h | history read: {model.history_days} days")
    for key, value in (meta.get("metrics") or {}).items():
        print(f"  {key}: {value}")


def cmd_dl_score(cfg, args):
    if _ml_switched_off(cfg):
        return 2
    from . import pipeline
    from .dl_scoring import load_dl_model, score_stored
    model = load_dl_model(cfg["paths"]["dl_models_dir"])
    if model is None:
        log.error("No deep-learning model installed in %s - see docs/ML_COLAB_GUIDE.md", cfg["paths"]["dl_models_dir"])
        return 2
    with pipeline.run_lock(cfg, wait_s=pipeline.LOCK_WAIT_S):
        engine, start, end = _range(cfg, args)
        n = score_stored(cfg, engine, start, end, model)
    log.info("scored %d rows with deep-learning model %s", n, model.model_version)


def cmd_prune_history(cfg, args):
    from datetime import datetime

    from . import pipeline
    from .config import ROOT
    backup = None
    if not args.no_backup:
        backup = args.backup or str(ROOT / "data" / f"ar_maintenance_backup_{datetime.now():%Y%m%d_%H%M}.sqlite")
    res = pipeline.prune_history(cfg, keep_days=args.keep_days, backup_to=backup)
    log.info("kept data after %s UTC; deleted %s rows; database now %s MB%s", res["cutoff"],
             f"{sum(res['deleted'].values()):,}", res["db_size_mb"],
             f"; backup: {res['backup']}" if res["backup"] else "")


def cmd_set_locations(cfg, args):
    from . import pipeline
    from .locations import ensure_locations, load_gps_csv
    engine = pipeline.get_engine(cfg)
    ensure_locations(engine, cfg)
    try:
        n = load_gps_csv(engine, args.csv)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    log.info("loaded GPS positions for %d reclosers", n)


def cmd_generate_ml_data(cfg, args):
    from pathlib import Path

    from . import pipeline
    from .mlgen import generate_ml_dataset
    seed = args.seed if args.seed is not None else cfg["seed"] + 1
    cfg = load_config(args.config, overrides={"fleet": {"n_assets": args.assets}, "seed": seed},
                      load_dotenv_file=False)
    start_ms = _parse_when(args.start, cfg) if args.start else pipeline.default_start_ms(cfg, args.days)
    out = Path(args.out) if args.out else cfg["paths"]["output_dir"] / "ml_generated"
    log.info("Generating %d reclosers x %d days (seed %d) -> %s (no database)", args.assets, args.days, seed, out)
    res = generate_ml_dataset(cfg, start_ms, args.days, out, hourly_copy=not args.no_hourly)
    for k, v in res.items():
        log.info("%s: %s", k, v)


def cmd_copy_db(cfg, args):
    from . import pipeline
    from .config import _sqlite_absolute
    counts = pipeline.copy_db(cfg, _sqlite_absolute(args.source_url))
    for name, n in counts.items():
        log.info("copied %-24s %10d rows", name, n)


def _range(cfg, args):
    from . import pipeline
    from .db.reader import asset_start_ms
    from .sources.simulated import SimulatedNojaSource
    engine = pipeline.get_engine(cfg)
    start = _parse_when(args.start, cfg)
    end = _parse_when(args.end, cfg)
    if start is None:
        with engine.connect() as conn:
            starts = asset_start_ms(conn, pipeline.step_ms(cfg))
        if not starts:
            raise RuntimeError("No telemetry yet")
        start = min(starts.values())
    if end is None:
        end = SimulatedNojaSource(cfg).last_collected_ms()
    return engine, start, end


def cmd_compute_variables(cfg, args):
    from . import pipeline
    with pipeline.run_lock(cfg, wait_s=pipeline.LOCK_WAIT_S):     # never race the collector or a reset
        engine, start, end = _range(cfg, args)
        n = pipeline.compute_range(cfg, engine, start, end)
    log.info("recomputed %d variable + HI rows", n)


def cmd_compute_hi(cfg, args):
    from . import pipeline
    with pipeline.run_lock(cfg, wait_s=pipeline.LOCK_WAIT_S):
        engine, start, end = _range(cfg, args)
        n = pipeline.recompute_hi(cfg, engine, start, end)
    log.info("re-scored %d HI rows", n)


COMMANDS = {
    "init-db": cmd_init_db, "seed": cmd_seed, "backfill": cmd_backfill, "run-once": cmd_run_once,
    "scheduler": cmd_scheduler, "status": cmd_status, "live": cmd_live, "validate": cmd_validate,
    "export-dataset": cmd_export_dataset, "copy-db": cmd_copy_db, "generate-ml-data": cmd_generate_ml_data, "set-locations": cmd_set_locations, "prune-history": cmd_prune_history, "service": cmd_service, "package-service": cmd_package_service, "ml-status": cmd_ml_status, "ml-score": cmd_ml_score, "dl-status": cmd_dl_status, "dl-score": cmd_dl_score, "maintenance-done": cmd_maintenance_done, "dashboard": cmd_dashboard, "compute-variables": cmd_compute_variables,
    "compute-hi": cmd_compute_hi,
}
