"""Configuration loading: config.yaml + environment variables (+ optional .env).

Environment variables (cloud friendly):
    DATABASE_URL   SQLAlchemy URL, e.g. mysql+pymysql://user:pw@host:3306/ar_maintenance
    AR_CONFIG      path to the YAML file (default: <project>/config.yaml)
    AR_STATE_DIR / AR_OUTPUT_DIR / AR_LOG_DIR   directory overrides
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

_NOT_HASHED = ("paths", "db", "health_index", "variables", "default_backfill_days")
_PATH_ENV = {"state_dir": "AR_STATE_DIR", "output_dir": "AR_OUTPUT_DIR", "log_dir": "AR_LOG_DIR",
             "models_dir": "AR_MODELS_DIR"}


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _sqlite_absolute(url: str) -> str:
    """Make a relative sqlite:/// URL absolute (relative to the project root)."""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        rest = url[len(prefix):]
        if rest and rest != ":memory:" and not Path(rest).is_absolute():
            return prefix + str(_resolve(rest)).replace("\\", "/")
    return url


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None,
                load_dotenv_file: bool = True) -> dict[str, Any]:
    """Load the YAML config, apply overrides and environment variables.

    Adds ``database_url`` and ``config_hash`` and resolves ``paths`` to absolute
    ``Path`` objects. ``config_hash`` covers only simulation settings, so moving
    directories or databases does not invalidate the saved simulator state.
    """
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:  # pragma: no cover
            pass

    cfg_path = Path(path or os.environ.get("AR_CONFIG") or ROOT / "config.yaml")
    with open(_resolve(cfg_path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        _deep_merge(cfg, copy.deepcopy(overrides))

    # Only settings that change the simulated raw data (not paths, DB, HI or variable windows).
    hash_src = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in _NOT_HASHED}
    hash_src.get("fleet", {}).pop("n_assets", None)
    cfg["config_hash"] = hashlib.sha256(
        json.dumps(hash_src, sort_keys=True, default=str).encode()).hexdigest()[:16]

    url = os.environ.get("DATABASE_URL") or cfg["db"]["default_url"]
    cfg["database_url"] = _sqlite_absolute(url)

    paths = {}
    for key, default in cfg["paths"].items():
        env_name = _PATH_ENV.get(key)
        paths[key] = _resolve(os.environ.get(env_name) or default) if env_name else _resolve(default)
    cfg["paths"] = paths
    return cfg
