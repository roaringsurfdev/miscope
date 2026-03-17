"""REQ_047: View Catalog — Core types and registry.

ViewDefinition pairs data loading + rendering for a named view.
ViewCatalog is the registry of all available views.
EpochContext is the training moment cursor and primary research interface.
BoundView is the Presenter — epoch fixed, exposes show/figure/export.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

if TYPE_CHECKING:
    from miscope.families.variant import Variant
    from miscope.views.dataview_catalog import BoundDataView, DataViewCatalog


class ArtifactKind(Enum):
    """Artifact storage pattern — determines which ArtifactLoader check to use.

    EPOCH: per-epoch .npz files exist (get_available_analyzers).
    SUMMARY: summary.npz exists (has_summary).
    CROSS_EPOCH: cross_epoch.npz exists (has_cross_epoch).
    CROSS_VARIANT: reserved for future cross-variant views; raises NotImplementedError.
    """

    EPOCH = "epoch"
    SUMMARY = "summary"
    CROSS_EPOCH = "cross_epoch"
    CROSS_VARIANT = "cross_variant"


@dataclass
class AnalyzerRequirement:
    """Declares that a view requires a specific analyzer artifact to exist.

    Attributes:
        name: Analyzer name (e.g., "dominant_frequencies").
        kind: Storage pattern that determines which availability check to use.
    """

    name: str
    kind: ArtifactKind


@dataclass
class ViewDefinition:
    """Pairs data loading and rendering for a named view.

    Attributes:
        name: Unique view identifier (e.g., "parameters.embeddings.fourier_coefficients").
        load_data: Callable(variant, epoch) -> Any. Loads all data needed
            for rendering. Epoch may be None for cross-epoch/metadata views.
        renderer: Callable(data, epoch, **kwargs) -> Figure. Renders the
            loaded data. Epoch is the snapshot epoch for per-epoch views
            or the cursor position for cross-epoch views.
        epoch_source_analyzer: Analyzer name used to resolve a None epoch
            to the first available artifact epoch. None for cross-epoch and
            metadata-based views where no resolution is needed.
        required_analyzers: Artifact requirements that must be satisfied for
            this view to be available. Empty list means always available
            (e.g., metadata-based views).
    """

    name: str
    load_data: Callable[..., Any]
    renderer: Callable[..., go.Figure]
    epoch_source_analyzer: str | None = field(default=None)
    required_analyzers: list[AnalyzerRequirement] = field(default_factory=list)

    def is_available_for(self, variant: Variant) -> bool:
        """Return True if all required artifacts exist for this variant."""
        for req in self.required_analyzers:
            if not _check_requirement(req, variant):
                return False
        return True


def _check_requirement(req: AnalyzerRequirement, variant: Variant) -> bool:
    """Check a single AnalyzerRequirement against a variant's artifacts."""
    if req.kind == ArtifactKind.EPOCH:
        return req.name in variant.artifacts.get_available_analyzers()
    if req.kind == ArtifactKind.SUMMARY:
        return variant.artifacts.has_summary(req.name)
    if req.kind == ArtifactKind.CROSS_EPOCH:
        return variant.artifacts.has_cross_epoch(req.name)
    if req.kind == ArtifactKind.CROSS_VARIANT:
        raise NotImplementedError(
            "CROSS_VARIANT availability cannot be checked against a single variant"
        )
    raise ValueError(f"Unknown ArtifactKind: {req.kind}")  # pragma: no cover


class ViewCatalog:
    """Registry of named views available in miscope."""

    def __init__(self) -> None:
        self._views: dict[str, ViewDefinition] = {}

    def register(self, view_def: ViewDefinition) -> None:
        """Register a view definition."""
        self._views[view_def.name] = view_def

    def get(self, name: str) -> ViewDefinition:
        """Look up a view by name.

        Raises:
            KeyError: If name not found, with message listing available views.
        """
        if name not in self._views:
            available = sorted(self._views.keys())
            raise KeyError(f"Unknown view '{name}'. Available views: {available}")
        return self._views[name]

    def names(self) -> list[str]:
        """Return sorted list of all registered view names."""
        return sorted(self._views.keys())

    def available_names_for(self, variant: Variant) -> list[str]:
        """Return sorted list of view names whose requirements are met for this variant."""
        return sorted(
            name for name, view_def in self._views.items() if view_def.is_available_for(variant)
        )


class BoundView:
    """A view definition with variant and epoch already bound.

    Returned by EpochContext.view(). Callers do not pass epoch — it was
    fixed at EpochContext creation time.
    """

    def __init__(
        self,
        view_def: ViewDefinition,
        variant: Variant,
        epoch: int | None,
    ) -> None:
        self._view_def = view_def
        self._variant = variant
        self._epoch = epoch

    def figure(self, **kwargs: Any) -> go.Figure:
        """Render and return the Plotly Figure."""
        data = self._view_def.load_data(self._variant, self._epoch)
        return self._view_def.renderer(data, self._epoch, **kwargs)

    def show(self, **kwargs: Any) -> None:
        """Render the figure inline in a Jupyter notebook."""
        self.figure(**kwargs).show()

    def export(self, format: str, path: str | Path | None = None, **kwargs: Any) -> Path:
        """Write figure to a static file.

        When path is omitted, a canonical path is derived from the variant name,
        view name, epoch, and any domain kwargs (e.g. site="attn_out"):
            results/exports/{variant}__{view}[__epoch{NNNNN}][__{key}_{val}...].{fmt}

        Args:
            format: "png", "svg", "pdf", or "html".
            path: Destination file path. Derived from context if not provided.
            **kwargs: Passed to figure() for renderer-specific options.

        Returns:
            Path to the written file.
        """
        from miscope.visualization.export import export_figure

        if path is None:
            path = self._default_export_path(format, **kwargs)
        fig = self.figure(**kwargs)
        return export_figure(fig, path, format=format)

    def _default_export_path(self, format: str, **kwargs: Any) -> Path:
        """Derive a canonical export path from variant, view, epoch, and kwargs."""
        from miscope.config import get_config

        parts = [self._variant.name, self._view_def.name]
        if self._epoch is not None:
            parts.append(f"epoch{self._epoch:05d}")
        for key in sorted(kwargs):
            parts.append(f"{key}_{kwargs[key]}")
        filename = "__".join(parts) + f".{format}"

        export_dir = get_config().results_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir / filename


class EpochContext:
    """Training moment cursor — holds a variant + resolved epoch.

    Created by Variant.at(epoch). Set the epoch once, then access any
    view without re-specifying it. Cross-epoch views receive the epoch
    as a highlight cursor rather than a data slice.
    """

    def __init__(
        self,
        variant: Variant,
        epoch: int | None,
        catalog: ViewCatalog | None = None,
        dataview_catalog: DataViewCatalog | None = None,
    ) -> None:
        self._variant = variant
        self._epoch = epoch
        self._catalog = catalog if catalog is not None else _catalog
        self._dataview_catalog = dataview_catalog

    def available_views(self) -> list[str]:
        """Return sorted list of view names whose requirements are met for this variant."""
        return self._catalog.available_names_for(self._variant)

    def available_dataviews(self) -> list[str]:
        """Return sorted list of dataview names whose requirements are met for this variant."""
        from miscope.views.dataview_catalog import _dataview_catalog

        catalog = (
            self._dataview_catalog if self._dataview_catalog is not None else _dataview_catalog
        )
        return catalog.available_names_for(self._variant)

    def view(self, name: str) -> BoundView:
        """Look up a view and bind it to this epoch.

        For per-epoch views (epoch_source_analyzer is set), resolves a None
        epoch to the first available artifact epoch. Cross-epoch and
        metadata-based views receive None as-is.

        Args:
            name: View identifier (e.g., "parameters.embeddings.fourier_coefficients").

        Returns:
            BoundView with epoch fixed.

        Raises:
            KeyError: If view name is not found in the catalog.
        """
        view_def = self._catalog.get(name)
        epoch = _resolve_epoch(self._epoch, view_def, self._variant)
        return BoundView(view_def=view_def, variant=self._variant, epoch=epoch)

    def dataview(self, name: str) -> BoundDataView:
        """Look up a dataview and bind it to this epoch.

        For per-epoch dataviews (epoch_source_analyzer is set), resolves a None
        epoch to the first available artifact epoch. Cross-epoch and
        metadata-based dataviews receive None as-is.

        Args:
            name: DataView identifier (e.g., "training.metadata.loss_curves").

        Returns:
            BoundDataView with epoch fixed.

        Raises:
            KeyError: If dataview name is not found in the catalog.
        """
        from miscope.views.dataview_catalog import (
            BoundDataView as _BoundDataView,
        )
        from miscope.views.dataview_catalog import (
            _dataview_catalog,
            _resolve_dataview_epoch,
        )

        catalog = (
            self._dataview_catalog if self._dataview_catalog is not None else _dataview_catalog
        )
        dataview_def = catalog.get(name)
        epoch = _resolve_dataview_epoch(self._epoch, dataview_def, self._variant)
        return _BoundDataView(dataview_def=dataview_def, variant=self._variant, epoch=epoch)


def _resolve_epoch(
    epoch: int | None,
    view_def: ViewDefinition,
    variant: Variant,
) -> int | None:
    """Resolve None epoch to first available artifact epoch for per-epoch views."""
    if epoch is not None or view_def.epoch_source_analyzer is None:
        return epoch
    epochs = variant.artifacts.get_epochs(view_def.epoch_source_analyzer)
    return epochs[0] if epochs else None


# Module-level default catalog. Populated by miscope.views.universal on import.
_catalog = ViewCatalog()
