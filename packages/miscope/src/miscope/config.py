"""Application configuration for MIScope.

Provides the resolved path to the unified data root, with an environment
variable override for non-standard layouts.

Usage:
    from miscope.config import get_config

    cfg = get_config()
    cfg.data_root          # Path to data/
    cfg.project_root       # Resolved project root

Environment variable overrides:
    MISCOPE_DATA_ROOT      Override data root path
    MISCOPE_PROJECT_ROOT   Override project root (data_root defaults under this)

Legacy aliases (still accepted, lower priority than MISCOPE_DATA_ROOT):
    TDW_DATA_ROOT, TDW_PROJECT_ROOT

Deprecated (emit DeprecationWarning, removed in a follow-up release):
    MISCOPE_RESULTS_DIR, MISCOPE_MODEL_FAMILIES_DIR
    TDW_RESULTS_DIR, TDW_MODEL_FAMILIES_DIR
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

_DEPRECATED_ROOT_VARS = (
    "MISCOPE_RESULTS_DIR",
    "MISCOPE_MODEL_FAMILIES_DIR",
    "TDW_RESULTS_DIR",
    "TDW_MODEL_FAMILIES_DIR",
)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration paths."""

    project_root: Path
    data_root: Path


def get_config() -> AppConfig:
    """Get application configuration.

    Resolves the data root in this order:
    1. ``MISCOPE_DATA_ROOT`` (or legacy ``TDW_DATA_ROOT``) environment variable.
    2. Default: ``{project_root}/data``.

    If either of the deprecated ``MISCOPE_RESULTS_DIR`` /
    ``MISCOPE_MODEL_FAMILIES_DIR`` env vars (or their ``TDW_*`` aliases) is set
    without the new ``MISCOPE_DATA_ROOT``, a :class:`DeprecationWarning` is
    emitted; the deprecated vars are otherwise ignored.

    Project root is resolved from ``MISCOPE_PROJECT_ROOT`` (or
    ``TDW_PROJECT_ROOT``), or by walking up from this file to find the uv
    workspace pyproject.toml.

    Returns:
        AppConfig with resolved paths.
    """
    project_root = _resolve_project_root()

    explicit_root = os.environ.get("MISCOPE_DATA_ROOT") or os.environ.get("TDW_DATA_ROOT")
    if explicit_root is None:
        _warn_if_deprecated_vars_set()
        data_root = project_root / "data"
    else:
        data_root = Path(explicit_root)

    return AppConfig(project_root=project_root, data_root=data_root)


def _warn_if_deprecated_vars_set() -> None:
    """Emit a DeprecationWarning if any retired env var is still set."""
    leftover = [name for name in _DEPRECATED_ROOT_VARS if os.environ.get(name)]
    if not leftover:
        return
    warnings.warn(
        f"Ignoring deprecated environment variable(s): {', '.join(leftover)}. "
        "MIScope now reads paths from MISCOPE_DATA_ROOT only (default: "
        "{project_root}/data). Unset the deprecated vars to silence this warning.",
        DeprecationWarning,
        stacklevel=3,
    )


def _resolve_project_root() -> Path:
    """Resolve the project root directory.

    Strategy:
    1. MISCOPE_PROJECT_ROOT environment variable (explicit override)
    2. TDW_PROJECT_ROOT environment variable (legacy alias)
    3. Walk up from this file looking for the uv workspace root
       (a pyproject.toml containing [tool.uv.workspace]). The package's own
       pyproject.toml is skipped; we want the repo root, where data/ lives.
    4. Fall back to the outermost pyproject.toml (non-workspace layouts).
    5. Fall back to current working directory.
    """
    env_root = os.environ.get("MISCOPE_PROJECT_ROOT") or os.environ.get("TDW_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    outermost: Path | None = None
    for ancestor in here.parents:
        pyproject = ancestor / "pyproject.toml"
        if not pyproject.exists():
            continue
        outermost = ancestor
        try:
            if "[tool.uv.workspace]" in pyproject.read_text():
                return ancestor
        except OSError:
            continue

    if outermost is not None:
        return outermost

    return Path.cwd()
