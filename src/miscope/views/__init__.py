"""REQ_047: View Catalog — Universal presentation layer.

Provides the unified interface between data sources and renderers.
The primary research interface is EpochContext, created via Variant.at(epoch).

Example usage:
    from miscope import load_family

    family = load_family("modulo_addition_1layer")
    variant = family.get_variant(prime=113, seed=999)

    # Pin a training moment and access multiple views
    ctx = variant.at(epoch=26400)
    ctx.view("training.metadata.loss_curves").show()
    ctx.view("parameters.embeddings.fourier_coefficients").show()
    ctx.view("parameter_trajectory").show()

    # Convenience shortcut (uses first available epoch for per-epoch views)
    variant.view("training.metadata.loss_curves").show()

    # Inspect available views
    from miscope.views import catalog
    print(catalog.names())
"""

# Import universal registrations to populate the catalog as a side effect.
import miscope.views.universal  # noqa: F401
from miscope.views.catalog import BoundView, EpochContext, ViewCatalog, ViewDefinition, _catalog

# Public catalog instance — use catalog.names() to inspect available views.
catalog = _catalog

__all__ = [
    "BoundView",
    "EpochContext",
    "ViewCatalog",
    "ViewDefinition",
    "catalog",
]
