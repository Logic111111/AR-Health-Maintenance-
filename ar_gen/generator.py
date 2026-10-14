"""Real-time data generator: the 15-minute update (collect -> V1-V42 -> Health Index), forever.

Runs ``pipeline.update`` 10 s after every interval end (:00 :15 :30 :45 for 15-min intervals). After downtime it
first catches up, one day per step, so a stop request never waits for a long catch-up. Every run is recorded in
``state/generator_status.json`` (read by ``python main.py service status``).

Used in a console by ``python main.py service run`` (alias ``scheduler``) and by the Windows service
(``ar_service.py``, see ar_gen/service.py). The stop event ends the loop between steps.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATUS_FILE = "generator_status.json"
RUN_OFFSET_S = 10                 # run 10 s after the interval end, when the interval is complete
CATCH_UP_STEP_DAYS = 1            # after downtime: collect + calculate at most one day per step


def _utc(ms: int | float | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _clock_ms() -> int:
    return int(time.time() * 1000)


def next_run_ms(now: int, step_ms: int) -> int:
    """First run time after ``now``: the next interval end + RUN_OFFSET_S."""
    base = now - RUN_OFFSET_S * 1000
    return (base // step_ms + 1) * step_ms + RUN_OFFSET_S * 1000


def status_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["state_dir"]) / STATUS_FILE


def read_status(cfg: dict) -> dict | None:
    try:
        return json.loads(status_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_status(cfg: dict, status: dict) -> None:
    path = status_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:                       # e.g. a reader has the file open for a moment
        log.debug("could not write %s: %s", path, exc)


def pid_alive(pid) -> bool:
    """Whether a process with this id is running (never signals it: os.kill would end it on Windows)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                      # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _last_collected_ms(cfg: dict) -> int:
    from .sources.simulated import SimulatedNojaSource
    return SimulatedNojaSource(cfg).last_collected_ms()


def log_health(cfg: dict) -> None:
    """One log line with the number of unhealthy reclosers (and which)."""
    from . import pipeline
    df = pipeline.status(cfg)
    if df.empty:
        return
    unhealthy = df[~df["is_healthy"].astype(bool)]
    log.info("Health Index: %d reclosers, %d not healthy%s", len(df), len(unhealthy),
             (": " + ", ".join(f"{r.asset_id} {r.health_index:.0f} ({r.health_class})" for r in unhealthy.itertuples()))
             if len(unhealthy) else "")


def update_now(cfg: dict, stop: threading.Event, clock=_clock_ms) -> dict:
    """Collect and calculate up to the interval that just ended, in day-sized steps; returns a run summary."""
    from . import pipeline
    from .sim.timeutil import DAY_MS
    step = pipeline.step_ms(cfg)
    target = clock()
    target -= target % step
    intervals, last_ts = 0, None
    while not stop.is_set():
        until = min(target, _last_collected_ms(cfg) + CATCH_UP_STEP_DAYS * DAY_MS)
        res = pipeline.update(cfg, until_ms=until)
        intervals += res["intervals"]
        last_ts = res["last_ts"]
        if until >= target:
            break
        log.info("catching up: data up to %s UTC", last_ts)
    return {"intervals": intervals, "last_ts": last_ts}


def _sleep_until(stop: threading.Event, deadline_ms: int, clock=_clock_ms) -> None:
    # Short slices: Ctrl+C reaches a console run quickly, and a clock change or sleep/resume is noticed.
    while not stop.is_set():
        left = deadline_ms - clock()
        if left <= 0:
            return
        stop.wait(min(left / 1000, 1.0))


def run_generator(cfg: dict, stop: threading.Event, mode: str = "console", clock=_clock_ms) -> None:
    """Run the update every interval until ``stop`` is set. Failures are logged and retried next interval."""
    from . import pipeline
    step = pipeline.step_ms(cfg)
    status = {"mode": mode, "pid": os.getpid(), "started_utc": _utc(clock()), "runs": 0, "failures_in_a_row": 0,
              "last_run_utc": None, "last_ok": None, "last_error": None, "intervals": 0, "data_up_to_utc": None,
              "duration_s": None, "next_run_utc": None, "stopped_utc": None}
    log.info("real-time generator started (%s): updates %d s after every %d-minute interval", mode,
             RUN_OFFSET_S, step // 60_000)
    try:
        while not stop.is_set():
            t0 = clock()
            status.update(last_run_utc=_utc(t0), runs=status["runs"] + 1)
            try:
                res = update_now(cfg, stop, clock)
                status.update(last_ok=True, last_error=None, failures_in_a_row=0, intervals=res["intervals"],
                              data_up_to_utc=str(res["last_ts"]) if res["last_ts"] is not None else None)
                log.info("collected %d interval(s); data up to %s UTC", res["intervals"], res["last_ts"])
                log_health(cfg)
            except Exception as exc:              # keep running; the next run catches up
                status.update(last_ok=False, last_error=f"{type(exc).__name__}: {exc}"[:500],
                              failures_in_a_row=status["failures_in_a_row"] + 1)
                log.exception("update failed - will retry at the next interval")
            status["duration_s"] = round((clock() - t0) / 1000, 1)
            nxt = next_run_ms(clock(), step)
            status["next_run_utc"] = _utc(nxt)
            _write_status(cfg, status)
            _sleep_until(stop, nxt, clock)
    finally:
        status.update(stopped_utc=_utc(clock()), next_run_utc=None)
        _write_status(cfg, status)
        log.info("real-time generator stopped")
