"""Console + rotating file logging (logs/ar_gen.log)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(cfg: dict, verbose: bool = False) -> None:
    log_dir = cfg["paths"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    fh = RotatingFileHandler(log_dir / "ar_gen.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    for noisy in ("apscheduler.executors.default", "apscheduler.scheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
