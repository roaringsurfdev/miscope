"""Tests for dashboard deployment configuration (REQ_124)."""

import tomllib
from pathlib import Path

import pytest

from dashboard.config import DashboardConfig, load_dashboard_config


def _write_toml(path: Path, text: str) -> None:
    path.write_text(text)


class TestLoadDashboardConfig:
    def test_default_path_loads(self):
        """Default config path resolves to apps/dashboard/config.toml and parses."""
        cfg = load_dashboard_config()
        assert isinstance(cfg, DashboardConfig)
        assert isinstance(cfg.server.host, str)
        assert isinstance(cfg.server.port, int)
        assert isinstance(cfg.server.debug, bool)

    def test_load_from_explicit_path(self, tmp_path):
        config_file = tmp_path / "custom.toml"
        _write_toml(
            config_file,
            '[server]\nhost = "127.0.0.1"\nport = 9000\ndebug = true\n',
        )
        cfg = load_dashboard_config(config_file)
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 9000
        assert cfg.server.debug is True

    def test_frozen_dataclass(self, tmp_path):
        config_file = tmp_path / "c.toml"
        _write_toml(
            config_file, '[server]\nhost = "x"\nport = 1\ndebug = false\n'
        )
        cfg = load_dashboard_config(config_file)
        with pytest.raises(AttributeError):
            cfg.server.host = "other"  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dashboard_config(tmp_path / "nope.toml")

    def test_missing_server_table_raises(self, tmp_path):
        config_file = tmp_path / "c.toml"
        _write_toml(config_file, "")
        with pytest.raises(ValueError, match=r"\[server\] table"):
            load_dashboard_config(config_file)

    def test_missing_required_key_raises(self, tmp_path):
        config_file = tmp_path / "c.toml"
        _write_toml(config_file, '[server]\nhost = "x"\nport = 1\n')  # debug missing
        with pytest.raises(ValueError, match="debug"):
            load_dashboard_config(config_file)

    def test_invalid_toml_raises(self, tmp_path):
        config_file = tmp_path / "c.toml"
        _write_toml(config_file, "this is not toml = =")
        with pytest.raises(tomllib.TOMLDecodeError):
            load_dashboard_config(config_file)

    def test_unknown_keys_ignored(self, tmp_path):
        """Forward-compatible: extra keys under [server] are silently ignored."""
        config_file = tmp_path / "c.toml"
        _write_toml(
            config_file,
            '[server]\nhost = "x"\nport = 1\ndebug = false\nfuture_flag = "y"\n',
        )
        cfg = load_dashboard_config(config_file)
        assert cfg.server.host == "x"
