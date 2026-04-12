"""Application configuration for MIScope.

Provides default paths for results and model families directories,
with environment variable overrides for non-standard layouts.

Usage:
    from miscope.config import get_config

    cfg = get_config()
    cfg.results_dir        # Path to results/
    cfg.model_families_dir # Path to model_families/
    cfg.project_root       # Resolved project root

Environment variable overrides:
    MISCOPE_RESULTS_DIR         Override results directory path
    MISCOPE_MODEL_FAMILIES_DIR  Override model families directory path
    MISCOPE_PROJECT_ROOT        Override project root (all defaults resolve from this)

Legacy aliases (still accepted, lower priority than MISCOPE_* vars):
    TDW_RESULTS_DIR, TDW_MODEL_FAMILIES_DIR, TDW_PROJECT_ROOT
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration paths."""

    project_root: Path
    results_dir: Path
    model_families_dir: Path


def get_config() -> AppConfig:
    """Get application configuration.

    Resolves paths in this order:
    1. MISCOPE_* environment variable override
    2. TDW_* legacy environment variable (backwards compatibility)
    3. Default relative to project root

    Project root is resolved from MISCOPE_PROJECT_ROOT (or TDW_PROJECT_ROOT),
    or by walking up from this file to find pyproject.toml.

    Returns:
        AppConfig with resolved paths
    """
    project_root = _resolve_project_root()

    results_dir = Path(
        os.environ.get("MISCOPE_RESULTS_DIR")
        or os.environ.get("TDW_RESULTS_DIR")
        or str(project_root / "results")
    )
    model_families_dir = Path(
        os.environ.get("MISCOPE_MODEL_FAMILIES_DIR")
        or os.environ.get("TDW_MODEL_FAMILIES_DIR")
        or str(project_root / "model_families")
    )

    return AppConfig(
        project_root=project_root,
        results_dir=results_dir,
        model_families_dir=model_families_dir,
    )


def _resolve_project_root() -> Path:
    """Resolve the project root directory.

    Strategy:
    1. MISCOPE_PROJECT_ROOT environment variable (explicit override)
    2. TDW_PROJECT_ROOT environment variable (legacy alias)
    3. Walk up from this file looking for pyproject.toml
    4. Fall back to current working directory
    """
    env_root = os.environ.get("MISCOPE_PROJECT_ROOT") or os.environ.get("TDW_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Walk up from src/miscope/config.py to find pyproject.toml
    current = Path(__file__).resolve().parent.parent.parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor

    # Fallback: current working directory
    return Path.cwd()
