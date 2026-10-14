"""Windows service host for the real-time data generator ("LECO AR Data Generator").

Do not run this file yourself. Install and control the service with:
    python main.py service install | start | stop | restart | status | remove
(admin PowerShell for all but status; see README section "Real-time data generator service").

Windows starts it as:  python.exe ar_service.py --scm [--site <user site-packages>] ...
``--check`` loads the app the same way and exits (used by ``service install`` as a test start).
"""
import os
import site
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _prepare_paths(argv):
    # The LocalSystem account does not see the installing user's site-packages: add the ones passed by --site.
    for i, arg in enumerate(argv[:-1]):
        if arg == "--site":
            site.addsitedir(argv[i + 1])
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def _check():
    import servicemanager  # noqa: F401  (pywin32 must load too)

    from ar_gen.config import load_config
    from ar_gen.pipeline import get_engine
    cfg = load_config()
    url = get_engine(cfg).url.render_as_string(hide_password=True)
    print(f"app loads (python {sys.version.split()[0]}), database {url}")


def _host():
    import threading

    import servicemanager
    import win32service
    import win32serviceutil

    class ARGeneratorService(win32serviceutil.ServiceFramework):
        _svc_name_ = "LecoArDataGenerator"
        _svc_display_name_ = "LECO AR Data Generator"

        def __init__(self, args):
            super().__init__(args)
            self.stop_event = threading.Event()

        def SvcStop(self):
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING, waitHint=180_000)
            self.stop_event.set()

        def SvcDoRun(self):
            servicemanager.LogInfoMsg(f"{self._svc_display_name_}: starting")
            try:
                from ar_gen.service import serve
                serve(self.stop_event)
            except Exception as exc:                 # noqa: BLE001 - report, then let Windows restart the service
                servicemanager.LogErrorMsg(f"{self._svc_display_name_} failed: {exc!r}")
                os._exit(1)                          # an abnormal end triggers the service's restart-on-failure
            servicemanager.LogInfoMsg(f"{self._svc_display_name_}: stopped")

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(ARGeneratorService)
    servicemanager.StartServiceCtrlDispatcher()


if __name__ == "__main__":
    _prepare_paths(sys.argv)
    if "--check" in sys.argv:
        _check()
    elif "--scm" in sys.argv:
        _host()
    else:
        print(__doc__)
