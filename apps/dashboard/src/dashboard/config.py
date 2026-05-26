"""Dashboard deployment configuration (REQ_124).

Loads ``apps/dashboard/config.toml`` into a frozen ``DashboardConfig`` dataclass.
Distinct from ``miscope.config.AppConfig``, which carries library-level data
paths (``data_root``). This module carries dashboard-only deployment values
(host, port, debug) — values an operator might want to vary per deployment
without touching code.

Usage:
    from dashboard.config import load_dashboard_config

    cfg = load_dashboard_config()
    app.run(host=cfg.server.host, port=cfg.server.port, debug=cfg.server.debug)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# Default config path: <repo>/apps/dashboard/config.toml. This module lives at
# apps/dashboard/src/dashboard/config.py, so the project file is three parents up.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


@dataclass(frozen=True)
class ServerConfig:
    """Dashboard HTTP server settings."""

    host: str
    port: int
    debug: bool


@dataclass(frozen=True)
class DashboardConfig:
    """Parsed dashboard deployment configuration."""

    server: ServerConfig


def load_dashboard_config(path: Path | str | None = None) -> DashboardConfig:
    """Load and validate the dashboard config from a TOML file.

    Args:
        path: Optional override of the config file location. Defaults to
            ``apps/dashboard/config.toml`` relative to this module.

    Returns:
        Frozen ``DashboardConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing.
        tomllib.TOMLDecodeError: If the file is not valid TOML.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Dashboard config not found at {config_path}")

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    server_raw = raw.get("server")
    if not isinstance(server_raw, dict):
        raise ValueError(
            f"Missing required [server] table in {config_path}. "
            "Add `[server]` with host, port, debug."
        )

    missing = [k for k in ("host", "port", "debug") if k not in server_raw]
    if missing:
        raise ValueError(f"Missing required keys under [server] in {config_path}: {missing}")

    return DashboardConfig(
        server=ServerConfig(
            host=str(server_raw["host"]),
            port=int(server_raw["port"]),
            debug=bool(server_raw["debug"]),
        ),
    )
