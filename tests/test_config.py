from pathlib import Path

from ar_gen.config import load_config, ROOT


def test_loads_yaml_values():
    cfg = load_config(load_dotenv_file=False)
    assert cfg["seed"] == 20261005
    assert cfg["models"]["OSM15-310"]["rated_break_ka"] == 16.0
    assert cfg["interval_minutes"] == 15


def test_excel_weights_sum_to_100_and_match_factor_percentages():
    hi = load_config(load_dotenv_file=False)["health_index"]
    assert sum(hi["weights"].values()) == 100
    factor_pct = {k: sum(hi["weights"][v] for v in vs) for k, vs in hi["factors"].items()}
    assert factor_pct == {"BATTERY": 20, "FAULTS": 24, "SWITCHING": 20, "COMMS": 10,
                          "CONTROLLING": 13, "AGING": 8, "ENVIRONMENT": 5}


def test_database_url_comes_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@127.0.0.1:3306/ar_maintenance")
    cfg = load_config(load_dotenv_file=False)
    assert cfg["database_url"] == "mysql+pymysql://u:p@127.0.0.1:3306/ar_maintenance"


def test_default_sqlite_url_is_absolute_under_project(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cfg = load_config(load_dotenv_file=False)
    assert cfg["database_url"].startswith("sqlite:///")
    db_path = Path(cfg["database_url"][len("sqlite:///"):])
    assert db_path.is_absolute()
    assert ROOT in db_path.parents


def test_paths_are_absolute_and_env_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("AR_STATE_DIR", str(tmp_path / "st"))
    cfg = load_config(load_dotenv_file=False)
    assert cfg["paths"]["state_dir"] == tmp_path / "st"
    assert cfg["paths"]["output_dir"].is_absolute()


def test_overrides_are_deep_merged():
    base = load_config(load_dotenv_file=False)
    cfg = load_config(overrides={"fleet": {"n_assets": 3}}, load_dotenv_file=False)
    assert cfg["fleet"]["n_assets"] == 3
    assert cfg["fleet"]["coastal_fraction"] == base["fleet"]["coastal_fraction"]   # siblings kept


def test_hash_ignores_paths_and_url(monkeypatch, tmp_path):
    a = load_config(load_dotenv_file=False)
    monkeypatch.setenv("AR_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
    b = load_config(load_dotenv_file=False)
    assert a["config_hash"] == b["config_hash"]
    c = load_config(overrides={"seed": 1}, load_dotenv_file=False)
    assert c["config_hash"] != a["config_hash"]
