"""Console + rotating file logging (logs/ar_gen.log; the Windows service uses logs/ar_service.log)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler


def setup_logging(cfg: dict, verbose: bool = False, console: bool = True, filename: str = "ar_gen.log") -> None:
    log_dir = cfg["paths"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)
    fh = RotatingFileHandler(log_dir / filename, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
