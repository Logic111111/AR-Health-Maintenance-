"""Windows service "LECO AR Data Generator": the real-time generator (ar_gen.generator) as a Windows service.

    python main.py service install    register it and start it (admin PowerShell): starts automatically at boot
                                      (delayed start), no login needed, restarted by Windows if it crashes
    python main.py service start | stop | restart | remove                       (admin PowerShell)
    python main.py service status     Windows service state + the generator's last run   (any PowerShell)
    python main.py service run        run the generator in this console instead (Ctrl+C stops it)

The service runs this Python (python.exe) with ar_service.py as the LocalSystem account. That account does not
see the installing user's own site-packages (where pip put pandas, SQLAlchemy, ...), so their path is passed on
the service command line (--site) and added at start-up. The service logs to logs/ar_service.log.
"""
from __future__ import annotations

import logging
import os
import site
import subprocess
import sys
import threading
from pathlib import Path

from .config import ROOT, load_config

log = logging.getLogger(__name__)

SERVICE_NAME = "LecoArDataGenerator"
DISPLAY_NAME = "LECO AR Data Generator"
DESCRIPTION = ("Real-time NOJA auto-recloser data for the LECO AR Health dashboard: collects every 15 minutes and "
               f"updates V1-V42 and the Health Index ({ROOT}). Log: logs\\ar_service.log")
HOST_SCRIPT = ROOT / "ar_service.py"
LOG_FILE = "ar_service.log"
OLD_TASK = "AR data getter"                    # the Task Scheduler task suggested before this service existed
RESTART_DELAY_MS = 60_000
STOP_WAIT_S = 180                              # a stop waits for the running update step (at most one day)
ADMIN_HINT = ("This needs administrator rights. Open PowerShell as administrator (Start menu -> right-click "
              "Windows PowerShell -> Run as administrator), then run:\n"
              f'  cd "{ROOT}"\n  python main.py service {{action}}')


class ServiceError(RuntimeError):
    pass


# ------------------------------------------------------------------ run inside the service process
def serve(stop: threading.Event) -> None:
    """Body of the Windows service (called by ar_service.py): file logging, then the generator until stopped."""
    from .generator import run_generator
    from .logging_setup import setup_logging
    cfg = load_config()
    setup_logging(cfg, console=False, filename=LOG_FILE)
    log.info("%s service started (python %s, %s)", DISPLAY_NAME, sys.version.split()[0], sys.executable)
    run_generator(cfg, stop, mode="service")


# ------------------------------------------------------------------ helpers
def is_windows() -> bool:
    return os.name == "nt"


def is_admin() -> bool:
    if not is_windows():
        return os.geteuid() == 0
    import ctypes
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def extra_site_dirs() -> list[str]:
    """Site-packages of this Python that the LocalSystem account would not see (the user's own)."""
    user_site = site.getusersitepackages() if site.ENABLE_USER_SITE else None
    return [user_site] if user_site and os.path.isdir(user_site) else []


def service_command_line() -> tuple[str, str]:
    """(program, arguments) that Windows runs for the service."""
    args = [f'"{HOST_SCRIPT}"', "--scm"]
    for path in extra_site_dirs():
        args += ["--site", f'"{path}"']
    return sys.executable, " ".join(args)


def preflight() -> str:
    """Start the host script the way the service will (no user site-packages) and check it can load the app."""
    exe, args = service_command_line()
    cmd = [exe, "-s", str(HOST_SCRIPT), "--check"]
    for path in extra_site_dirs():
        cmd += ["--site", path]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=os.environ.get("SystemRoot", None))
    if res.returncode != 0:
        raise ServiceError("the service could not load the app in a test start:\n" + (res.stderr or res.stdout)[-2000:])
    return res.stdout.strip()


def _win():
    try:
        import pywintypes
        import win32service
        import win32serviceutil
    except ImportError as exc:
        raise ServiceError("pywin32 is not installed - run: python -m pip install pywin32") from exc
    return pywintypes, win32service, win32serviceutil


STATES = {1: "STOPPED", 2: "START PENDING", 3: "STOP PENDING", 4: "RUNNING", 5: "CONTINUE PENDING",
          6: "PAUSE PENDING", 7: "PAUSED"}


def query_state(name: str = SERVICE_NAME) -> str:
    """RUNNING, STOPPED, ... or NOT INSTALLED (no admin rights needed)."""
    pywintypes, _, win32serviceutil = _win()
    try:
        return STATES.get(win32serviceutil.QueryServiceStatus(name)[1], "UNKNOWN")
    except pywintypes.error as exc:
        if exc.winerror == 1060:                    # ERROR_SERVICE_DOES_NOT_EXIST
            return "NOT INSTALLED"
        raise


def query_command_line(name: str = SERVICE_NAME) -> str | None:
    pywintypes, win32service, _ = _win()
    scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    try:
        h = win32service.OpenService(scm, name, win32service.SERVICE_QUERY_CONFIG)
    except pywintypes.error:
        return None
    try:
        return win32service.QueryServiceConfig(h)[3]
    finally:
        win32service.CloseServiceHandle(h)
        win32service.CloseServiceHandle(scm)


def _old_task_exists() -> bool:
    try:
        return subprocess.run(["schtasks", "/Query", "/TN", OLD_TASK], capture_output=True).returncode == 0
    except OSError:
        return False


# ------------------------------------------------------------------ actions (admin)
def install(start: bool = True) -> str:
    pywintypes, win32service, _ = _win()
    checked = preflight()
    exe, args = service_command_line()
    binary = f'"{exe}" {args}'
    scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ALL_ACCESS)
    try:
        try:
            h = win32service.OpenService(scm, SERVICE_NAME, win32service.SERVICE_ALL_ACCESS)
            win32service.ChangeServiceConfig(h, win32service.SERVICE_WIN32_OWN_PROCESS,
                                             win32service.SERVICE_AUTO_START, win32service.SERVICE_ERROR_NORMAL,
                                             binary, None, 0, None, None, None, DISPLAY_NAME)
            verb = "updated"
        except pywintypes.error as exc:
            if exc.winerror != 1060:
                raise
            h = win32service.CreateService(scm, SERVICE_NAME, DISPLAY_NAME, win32service.SERVICE_ALL_ACCESS,
                                           win32service.SERVICE_WIN32_OWN_PROCESS, win32service.SERVICE_AUTO_START,
                                           win32service.SERVICE_ERROR_NORMAL, binary, None, 0, None, None, None)
            verb = "installed"
        try:
            win32service.ChangeServiceConfig2(h, win32service.SERVICE_CONFIG_DESCRIPTION, DESCRIPTION)
            win32service.ChangeServiceConfig2(h, win32service.SERVICE_CONFIG_DELAYED_AUTO_START_INFO, True)
            win32service.ChangeServiceConfig2(h, win32service.SERVICE_CONFIG_FAILURE_ACTIONS, {
                "ResetPeriod": 86400, "RebootMsg": None, "Command": None,
                "Actions": [(win32service.SC_ACTION_RESTART, RESTART_DELAY_MS)] * 3})
            win32service.ChangeServiceConfig2(h, win32service.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG, True)
        finally:
            win32service.CloseServiceHandle(h)
    finally:
        win32service.CloseServiceHandle(scm)
    lines = [f"{DISPLAY_NAME} ({SERVICE_NAME}) {verb}: starts automatically at boot, restarted after a crash.",
             f"  test start: {checked}", f"  command: {binary}"]
    if _old_task_exists():
        lines.append(f'  NOTE: the old Task Scheduler task "{OLD_TASK}" still exists. The service replaces it; '
                     f'remove it with: schtasks /Delete /TN "{OLD_TASK}" /F')
    if start:
        lines.append(start_service())
    return "\n".join(lines)


def start_service() -> str:
    _, win32service, win32serviceutil = _win()
    if query_state() == "RUNNING":
        return f"{DISPLAY_NAME} is already running."
    win32serviceutil.StartService(SERVICE_NAME)
    win32serviceutil.WaitForServiceStatus(SERVICE_NAME, win32service.SERVICE_RUNNING, 60)
    return f"{DISPLAY_NAME} started. It catches up first if data is missing (log: logs\\{LOG_FILE})."


def stop_service() -> str:
    _, win32service, win32serviceutil = _win()
    if query_state() in ("STOPPED", "NOT INSTALLED"):
        return f"{DISPLAY_NAME} is not running."
    win32serviceutil.StopService(SERVICE_NAME)
    win32serviceutil.WaitForServiceStatus(SERVICE_NAME, win32service.SERVICE_STOPPED, STOP_WAIT_S)
    return f"{DISPLAY_NAME} stopped."


def remove() -> str:
    _, _, win32serviceutil = _win()
    if query_state() == "NOT INSTALLED":
        return f"{DISPLAY_NAME} is not installed."
    lines = [stop_service()]
    win32serviceutil.RemoveService(SERVICE_NAME)
    lines.append(f"{DISPLAY_NAME} removed. Run the generator by hand with: python main.py service run")
    return "\n".join(lines)


# ------------------------------------------------------------------ status (no admin)
def status_text(cfg: dict) -> str:
    from .generator import pid_alive, read_status
    lines = []
    if is_windows():
        state = query_state()
        lines.append(f"{DISPLAY_NAME} ({SERVICE_NAME}): {state}")
        if state != "NOT INSTALLED":
            lines.append(f"  command: {query_command_line()}")
    st = read_status(cfg)
    if not st:
        lines.append("Generator: no run recorded yet.")
    else:
        running = st.get("stopped_utc") is None and pid_alive(st.get("pid"))
        if running:
            state = f"running since {st.get('started_utc')} UTC"
        elif st.get("stopped_utc"):
            state = f"stopped at {st['stopped_utc']} UTC"
        else:
            state = "not running (the process ended without a clean stop)"
        lines.append(f"Generator ({st.get('mode')}, pid {st.get('pid')}): {state}")
        ok = {True: "OK", False: "FAILED", None: "-"}[st.get("last_ok")]
        lines.append(f"  last run {st.get('last_run_utc')} UTC: {ok}, {st.get('intervals')} interval(s), "
                     f"{st.get('duration_s')} s; data up to {st.get('data_up_to_utc')} UTC")
        if st.get("last_error"):
            lines.append(f"  last error ({st.get('failures_in_a_row')} in a row): {st['last_error']}")
        if running and st.get("next_run_utc"):
            lines.append(f"  next run {st['next_run_utc']} UTC")
    return "\n".join(lines)


ADMIN_ACTIONS = {"install", "start", "stop", "restart", "remove"}


def manage(action: str, cfg: dict, start: bool = True) -> int:
    """`python main.py service <action>` (except run). Prints the outcome; returns the exit code."""
    if not is_windows() and action != "status":
        print("The Windows service is for Windows. Elsewhere run `python main.py service run` under systemd or "
              "similar.")
        return 2
    if action in ADMIN_ACTIONS and not is_admin():
        print(ADMIN_HINT.format(action=action))
        return 2
    try:
        if action == "status":
            print(status_text(cfg))
        elif action == "install":
            print(install(start=start))
        elif action == "start":
            print(start_service())
        elif action == "stop":
            print(stop_service())
        elif action == "restart":
            print(stop_service())
            print(start_service())
        elif action == "remove":
            print(remove())
        else:
            raise ValueError(f"unknown service action {action!r}")
    except ServiceError as exc:
        print(f"Service {action} failed: {exc}")
        return 2
    except Exception as exc:                       # pywin32 errors: show the Windows message
        print(f"Service {action} failed: {getattr(exc, 'strerror', None) or exc}")
        return 2
    return 0
