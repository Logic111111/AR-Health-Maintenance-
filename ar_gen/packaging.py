"""Build a service-only package: the real-time data generator as a Windows service, without the dashboard,
the ML parts, notebooks, data or tests (``python main.py package-service``).

The zip unpacks to one folder that runs on its own: ``python main.py init-db / seed / backfill`` and then
``python main.py service install``. Its README is docs/SERVICE_PACKAGE_README.md.
"""
from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

from .config import ROOT

PACKAGE_NAME = "LECO_AR_Generator_Service"

# Everything the generator, the service and its commands (init-db, seed, backfill, run-once, status, validate,
# maintenance-done, compute-*, prune-history, copy-db, service ...) import from the project.
FILES = [
    "main.py", "ar_service.py", "config.yaml", ".env.example", "DATA_DICTIONARY.md",
    "ar_gen/__init__.py", "ar_gen/cli.py", "ar_gen/config.py", "ar_gen/generator.py", "ar_gen/health_index.py",
    "ar_gen/locations.py", "ar_gen/logging_setup.py", "ar_gen/pipeline.py", "ar_gen/service.py",
    "ar_gen/validate.py", "ar_gen/variables.py",
]
PACKAGES = ["ar_gen/db", "ar_gen/sim", "ar_gen/sources"]
README_SRC = "docs/SERVICE_PACKAGE_README.md"
REQUIREMENTS = """\
# Python packages for the LECO AR Data Generator service (python -m pip install -r requirements-service.txt)
pandas>=2.0
numpy>=1.24
SQLAlchemy>=2.0,<2.1
PyMySQL>=1.1
pywin32>=306; sys_platform == "win32"
python-dotenv>=1.0
PyYAML>=6.0
tzdata
"""


def package_files() -> list[str]:
    """Project-relative paths of every file in the package (README and requirements are generated)."""
    files = list(FILES)
    for pkg in PACKAGES:
        files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / pkg).glob("*.py"))
    return files


def build_service_package(out_dir: Path | None = None) -> Path:
    """Write <out_dir>/LECO_AR_Generator_Service_<date>.zip (default out_dir: dist/) and return its path."""
    out_dir = Path(out_dir) if out_dir else ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{PACKAGE_NAME}_{datetime.now():%Y%m%d}.zip"
    missing = [f for f in package_files() + [README_SRC] if not (ROOT / f).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot build the service package, missing: {', '.join(missing)}")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in package_files():
            z.write(ROOT / rel, f"{PACKAGE_NAME}/{rel}")
        z.writestr(f"{PACKAGE_NAME}/README.md", (ROOT / README_SRC).read_text(encoding="utf-8"))
        z.writestr(f"{PACKAGE_NAME}/requirements-service.txt", REQUIREMENTS)
    return path
