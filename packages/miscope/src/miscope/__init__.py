"""MIScope — notebook research API.

Entry point for ad-hoc analysis in notebooks. Provides access to
model families, variants, checkpoints, artifacts, and visualizations
without requiring knowledge of file paths or internal module structure.

Quick start:
    from miscope import load_family

    family = load_family("modulo_addition_1layer")
    variant = family.get_variant(prime=113, seed=999)

    # Inspect activations
    probe = variant.make_probe([[3, 29]])
    logits, cache = variant.run_with_cache(probe, epoch=26400)

    # Access artifacts
    epoch_data = variant.artifacts.load_epoch("dominant_frequencies", 26400)

    # Render visualizations
    from miscope.visualization import render_dominant_frequencies
    fig = render_dominant_frequencies(epoch_data, 26400)
    fig.show()
"""

from __future__ import annotations

from miscope import query, registry
from miscope.config import AppConfig, get_config
from miscope.families.protocols import ModelFamily
from miscope.views import BoundView, EpochContext, ViewCatalog, ViewDefinition, catalog

__all__ = [
    "load_family",
    "list_families",
    "get_config",
    "AppConfig",
    "ModelFamily",
    # REQ_047: View Catalog
    "BoundView",
    "EpochContext",
    "ViewCatalog",
    "ViewDefinition",
    "catalog",
    # REQ_107: Discoverability Registry (INFORMATION_SCHEMA for analysis)
    "registry",
    # REQ_110C: DuckDB query surface over the warehouse
    "query",
]


def load_family(name: str, *, config: AppConfig | None = None) -> ModelFamily:
    """Load a model family by name.

    Args:
        name: Family identifier (e.g., "modulo_addition_1layer")
        config: Optional config override. Uses default config if not provided.

    Returns:
        ModelFamily with variant lookup methods.

    Raises:
        KeyError: If family name not found.
    """
    from miscope.families.discovery import discover_families

    cfg = config or get_config()
    families = discover_families(cfg.data_root)
    if name not in families:
        raise KeyError(f"Family '{name}' not found. Available: {list(families.keys())}")
    return families[name]


def list_families(*, config: AppConfig | None = None) -> list[str]:
    """List available model family names.

    Args:
        config: Optional config override. Uses default config if not provided.

    Returns:
        List of family name strings.
    """
    from miscope.families.discovery import discover_families

    cfg = config or get_config()
    return list(discover_families(cfg.data_root).keys())
