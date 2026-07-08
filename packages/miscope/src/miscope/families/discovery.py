"""Filesystem helpers for locating and loading model families.

This module replaces the former ``FamilyRegistry`` class. Variant discovery
is now a property of the family itself; what remains here are two narrow
responsibilities:

1. Locating ``family.json`` files under the unified data root.
2. Mapping a family name to its implementation class so the right class can
   be constructed at load time.

Family implementations register themselves at import time via
``register_family_implementation`` (see each ``implementations/*.py`` module).
"""

from __future__ import annotations

import json
from pathlib import Path

from miscope.families.base_model_family import BaseModelFamily
from miscope.families.protocols import ModelFamily

# Mapping of family names to their implementation classes. Family modules
# call ``register_family_implementation`` at import time; families that omit
# registration fall back to ``BaseModelFamily`` (config-only).
_FAMILY_IMPLEMENTATIONS: dict[str, type[BaseModelFamily]] = {}


def register_family_implementation(name: str, cls: type[BaseModelFamily]) -> None:
    """Register a family implementation class.

    Called at import time from each implementation module.

    Args:
        name: Family name (must match family.json "name" field)
        cls: Implementation class (must be subclass of BaseModelFamily)
    """
    _FAMILY_IMPLEMENTATIONS[name] = cls


def list_family_dirs(data_root: Path | str) -> list[Path]:
    """Return subdirectories of ``data_root`` containing ``family.json``."""
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "family.json").exists())


def load_family_from_dir(family_dir: Path | str, data_root: Path | str) -> ModelFamily:
    """Load a ModelFamily from a directory containing ``family.json``.

    Args:
        family_dir: Path to the family's directory under ``data_root``.
        data_root: Unified data root the family should anchor on (typically
            ``cfg.data_root``).

    Returns:
        A ``ModelFamily`` instance — the registered implementation class if
        one exists for the family name, otherwise ``BaseModelFamily``.
    """
    family_json = Path(family_dir) / "family.json"
    with open(family_json) as f:
        config = json.load(f)
    family_name = config.get("name", "")
    impl_class = _FAMILY_IMPLEMENTATIONS.get(family_name, BaseModelFamily)
    return impl_class(config, config_path=family_json, data_root=Path(data_root))


def discover_families(data_root: Path | str) -> dict[str, ModelFamily]:
    """Discover all families under ``data_root`` and return them keyed by name.

    Returns:
        Dict mapping family name to ``ModelFamily`` instance. Families whose
        ``family.json`` fails to parse are skipped with a warning.
    """
    families: dict[str, ModelFamily] = {}
    for family_dir in list_family_dirs(data_root):
        try:
            family = load_family_from_dir(family_dir, data_root)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load family from {family_dir / 'family.json'}: {e}")
            continue
        families[family.name] = family
    return families
