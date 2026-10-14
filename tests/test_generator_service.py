"""Real-time generator loop and the Windows service that runs it."""
import json
import os
import sys
import threading
import types

import pytest

from ar_gen import cli, generator, pipeline, service
from ar_gen.sim.timeutil import DAY_MS, MIN_MS
from tests.test_maintenance_done import db  # noqa: F401  (shared pipeline-database fixture)

STEP = 15 * MIN_MS


def test_next_run_is_ten_seconds_after_the_next_interval_end():
    t = 1_000_000 * STEP                                       # an interval end
    assert generator.next_run_ms(t, STEP) == t + 10_000         # its own run is still to come
    assert generator.next_run_ms(t + 10_000, STEP) == t + STEP + 10_000
    assert generator.next_run_ms(t + 7 * MIN_MS, STEP) == t + STEP + 10_000


def test_catch_up_runs_in_day_steps_and_stops_between_steps(monkeypatch):
    last = {"ms": 1_000_000 * STEP}
    target = last["ms"] + 2 * DAY_MS + 5 * STEP
    calls, stop = [], threading.Event()

    def fake_update(cfg, until_ms):
        calls.append(until_ms)
        last["ms"] = until_ms
        return {"intervals": 1, "last_ts": until_ms}

    monkeypatch.setattr(pipeline, "update", fake_update)
    monkeypatch.setattr(generator, "_last_collected_ms", lambda cfg: last["ms"])
    cfg = {"interval_minutes": 15}
    res = generator.update_now(cfg, stop, clock=lambda: target + 20_000)
    start = target - 2 * DAY_MS - 5 * STEP
    assert calls == [start + DAY_MS, start + 2 * DAY_MS, target] and res["intervals"] == 3

    calls.clear()
    last["ms"] = start

    def update_then_stop(cfg, until_ms):
        stop.set()                                              # e.g. the service is being stopped
        return fake_update(cfg, until_ms)

    monkeypatch.setattr(pipeline, "update", update_then_stop)
    generator.update_now(cfg, stop, clock=lambda: target)
    assert calls == [start + DAY_MS]


def test_generator_collects_records_its_status_and_stops_cleanly(db, monkeypatch):
    cfg, _ = db
    last = pipeline.last_calculated_ms(pipeline.get_engine(cfg))
    stop, real_update = threading.Event(), pipeline.update

    def update_once(cfg, until_ms=None):
        res = real_update(cfg, until_ms=until_ms)
        stop.set()
        return res

    monkeypatch.setattr(pipeline, "update", update_once)
    generator.run_generator(cfg, stop, mode="console", clock=lambda: last + 2 * STEP + 10_000)
    st = generator.read_status(cfg)
    assert st["mode"] == "console" and st["pid"] == os.getpid() and st["runs"] == 1
    assert st["last_ok"] is True and st["intervals"] == 2 and st["stopped_utc"] is not None
    assert pipeline.last_calculated_ms(pipeline.get_engine(cfg)) == last + 2 * STEP


def test_a_failed_update_is_recorded_and_the_generator_keeps_running(monkeypatch, tmp_path):
    cfg = {"interval_minutes": 15, "paths": {"state_dir": tmp_path}}
    stop, seen, calls = threading.Event(), [], []

    def flaky(cfg, stop_, clock):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        stop.set()
        return {"intervals": 1, "last_ts": "2026-10-14 09:00:00"}

    monkeypatch.setattr(generator, "update_now", flaky)
    monkeypatch.setattr(generator, "log_health", lambda cfg: None)
    monkeypatch.setattr(generator, "_sleep_until", lambda *a, **k: None)
    monkeypatch.setattr(generator, "_write_status", lambda cfg, st: seen.append(dict(st)))
    generator.run_generator(cfg, stop, clock=lambda: 1_000_000 * STEP)
    assert seen[0]["last_ok"] is False and "database is locked" in seen[0]["last_error"]
    assert seen[0]["failures_in_a_row"] == 1 and seen[0]["next_run_utc"]
    assert seen[1]["last_ok"] is True and seen[1]["failures_in_a_row"] == 0 and seen[-1]["stopped_utc"]


def test_pid_alive():
    assert generator.pid_alive(os.getpid())
    assert not generator.pid_alive(None) and not generator.pid_alive("x")


def test_service_runs_this_python_with_the_host_script_and_the_users_site_packages(monkeypatch):
    monkeypatch.setattr(service, "extra_site_dirs", lambda: [r"C:\Users\u\AppData\Roaming\Python\Python313\site-packages"])
    exe, args = service.service_command_line()
    assert exe == sys.executable
    assert args == (f'"{service.HOST_SCRIPT}" --scm --site '
                    '"C:\\Users\\u\\AppData\\Roaming\\Python\\Python313\\site-packages"')
    assert service.HOST_SCRIPT.exists()


@pytest.mark.parametrize("action", sorted(service.ADMIN_ACTIONS))
def test_service_changes_need_an_admin_powershell(action, monkeypatch, capsys):
    monkeypatch.setattr(service, "is_windows", lambda: True)
    monkeypatch.setattr(service, "is_admin", lambda: False)
    assert service.manage(action, {}) == 2
    out = capsys.readouterr().out
    assert "administrator" in out and f"python main.py service {action}" in out


def _fake_win(calls, existing=False):
    class WinError(Exception):
        def __init__(self, winerror):
            super().__init__(winerror)
            self.winerror = winerror

    def record(name, result=None):
        return lambda *a: calls.append((name, a)) or result

    def open_service(scm, name, access):
        calls.append(("OpenService", (name,)))
        if not existing:
            raise WinError(1060)
        return "h"

    ws = types.SimpleNamespace(
        SC_MANAGER_ALL_ACCESS=1, SERVICE_ALL_ACCESS=2, SERVICE_WIN32_OWN_PROCESS=16, SERVICE_AUTO_START=2,
        SERVICE_ERROR_NORMAL=1, SERVICE_CONFIG_DESCRIPTION=1, SERVICE_CONFIG_DELAYED_AUTO_START_INFO=3,
        SERVICE_CONFIG_FAILURE_ACTIONS=2, SERVICE_CONFIG_FAILURE_ACTIONS_FLAG=4, SC_ACTION_RESTART=1,
        SERVICE_RUNNING=4, SERVICE_STOPPED=1,
        OpenSCManager=record("OpenSCManager", "scm"), OpenService=open_service,
        CreateService=record("CreateService", "h"), ChangeServiceConfig=record("ChangeServiceConfig"),
        ChangeServiceConfig2=record("ChangeServiceConfig2"), CloseServiceHandle=record("CloseServiceHandle"))
    wsu = types.SimpleNamespace(StartService=record("StartService"), WaitForServiceStatus=record("Wait"),
                                QueryServiceStatus=lambda name: (16, 1))
    return types.SimpleNamespace(error=WinError), ws, wsu


@pytest.mark.parametrize("existing", [False, True])
def test_install_registers_auto_start_with_restart_on_failure_and_starts_it(monkeypatch, existing):
    calls = []
    monkeypatch.setattr(service, "_win", lambda: _fake_win(calls, existing))
    monkeypatch.setattr(service, "preflight", lambda: "app loads")
    monkeypatch.setattr(service, "_old_task_exists", lambda: True)
    text = service.install(start=True)
    names = [c[0] for c in calls]
    created = [c for c in calls if c[0] == ("ChangeServiceConfig" if existing else "CreateService")][0][1]
    exe, args = service.service_command_line()
    assert f'"{exe}" {args}' in created
    assert existing or service.SERVICE_NAME in created
    config2 = {c[1][1]: c[1][2] for c in calls if c[0] == "ChangeServiceConfig2"}
    assert config2[3] is True                                              # delayed automatic start
    assert config2[2]["Actions"] == [(1, service.RESTART_DELAY_MS)] * 3     # restart after a crash
    assert config2[4] is True and "StartService" in names
    assert ("updated" if existing else "installed") in text and "schtasks /Delete" in text


@pytest.mark.skipif(os.name != "nt", reason="Windows service manager")
def test_status_of_a_service_that_is_not_installed():
    assert service.query_state("LecoArDataGenerator-test-does-not-exist") == "NOT INSTALLED"


def test_status_text_tells_a_crashed_generator_from_a_stopped_one(monkeypatch, tmp_path):
    cfg = {"paths": {"state_dir": tmp_path}}
    monkeypatch.setattr(service, "is_windows", lambda: False)
    base = {"mode": "service", "pid": 999999, "started_utc": "2026-10-14 08:00:00", "last_run_utc": "x",
            "last_ok": False, "last_error": "OperationalError: database is locked", "failures_in_a_row": 2,
            "intervals": 0, "duration_s": 0.1, "data_up_to_utc": "y", "next_run_utc": "z", "stopped_utc": None}
    (tmp_path / generator.STATUS_FILE).write_text(json.dumps(base))
    text = service.status_text(cfg)
    assert "not running (the process ended without a clean stop)" in text and "2 in a row" in text
    (tmp_path / generator.STATUS_FILE).write_text(json.dumps({**base, "pid": os.getpid()}))
    assert "running since 2026-10-14 08:00:00 UTC" in service.status_text(cfg)
    (tmp_path / generator.STATUS_FILE).write_text(json.dumps({**base, "stopped_utc": "2026-10-14 09:00:00"}))
    assert "stopped at 2026-10-14 09:00:00 UTC" in service.status_text(cfg)


def test_cli_service_run_and_scheduler_start_the_generator_in_the_console(db, monkeypatch):
    modes = []
    monkeypatch.setattr(generator, "run_generator", lambda cfg, stop, mode="console": modes.append(mode))
    assert cli.main(["service", "run"]) == 0 and cli.main(["scheduler"]) == 0
    assert modes == ["console", "console"]


@pytest.mark.skipif(os.name != "nt", reason="Windows service host")
def test_the_service_host_loads_the_app_without_the_users_site_packages_on_the_path():
    assert "app loads" in service.preflight()
