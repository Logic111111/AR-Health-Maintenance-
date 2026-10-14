"""The service-only package unzips to a folder that works on its own (no dashboard, ML, notebooks or data)."""
import os
import subprocess
import sys
import zipfile

from ar_gen import cli
from ar_gen.packaging import PACKAGE_NAME, build_service_package

LEFT_OUT = ("ar_gen/dashboard_app.py", "ar_gen/ui/", "ar_gen/ml_scoring.py", "ar_gen/dl_scoring.py", "notebooks/",
            "models/", "data/", "state/", "tests/", ".env")


def test_package_has_the_service_and_nothing_else(tmp_path):
    names = zipfile.ZipFile(build_service_package(tmp_path)).namelist()
    assert all(n.startswith(f"{PACKAGE_NAME}/") for n in names)
    rel = {n.split("/", 1)[1] for n in names}
    for need in ("main.py", "ar_service.py", "config.yaml", "README.md", "requirements-service.txt",
                 "ar_gen/service.py", "ar_gen/generator.py", "ar_gen/pipeline.py", "ar_gen/sim/engine.py"):
        assert need in rel, need
    assert not [n for n in rel for out in LEFT_OUT if n.startswith(out) and n != ".env.example"]
    assert not [n for n in rel if "__pycache__" in n]


def _run(folder, *argv):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AR_") and k not in ("DATABASE_URL", "PYTHONPATH")}
    return subprocess.run([sys.executable, "main.py", *argv], cwd=folder, env=env, capture_output=True, text=True,
                          timeout=600)


def test_unzipped_package_creates_data_and_runs_the_generator_on_its_own(tmp_path):
    zipfile.ZipFile(build_service_package(tmp_path / "dist")).extractall(tmp_path)
    folder = tmp_path / PACKAGE_NAME
    found = _run(folder, "--help")
    assert found.returncode == 0 and "service" in found.stdout
    probe = subprocess.run([sys.executable, "-c", "import ar_gen; print(ar_gen.__file__)"], cwd=folder,
                           capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert str(folder) in probe.stdout                       # the package's own code, not this project's
    for argv in (["init-db"], ["seed", "--assets", "2", "--days", "1"], ["backfill"], ["validate"], ["status"]):
        res = _run(folder, *argv)
        assert res.returncode == 0, (argv, res.stderr[-1500:])
    assert "Validation PASSED" in _run(folder, "validate").stdout
    assert (folder / "data" / "ar_maintenance.sqlite").exists() and (folder / "state" / "sim_state.json").exists()
    dash = _run(folder, "dashboard", "--no-browser")
    assert dash.returncode == 2 and "not part of this copy" in dash.stderr


def test_cli_package_service(tmp_path, capsys):
    assert cli.main(["package-service", "--out", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob(f"{PACKAGE_NAME}_*.zip"))) == 1
