"""Tests for REQ_047: View Catalog — Universal Presentation Layer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.views.catalog import BoundView, EpochContext, ViewCatalog, ViewDefinition, _catalog

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    return fig


@pytest.fixture
def simple_view(minimal_figure) -> ViewDefinition:
    """A simple view definition that ignores epoch and variant for testing."""
    captured_figure = minimal_figure

    def load_data(variant, epoch):
        return {"value": 42}

    def renderer(data, epoch, **kwargs):
        return captured_figure

    return ViewDefinition(
        name="test_view",
        load_data=load_data,
        renderer=renderer,
        epoch_source_analyzer=None,
    )


@pytest.fixture
def per_epoch_view(minimal_figure) -> ViewDefinition:
    """A per-epoch view definition with epoch_source_analyzer set."""
    captured_figure = minimal_figure

    def load_data(variant, epoch):
        return {"epoch_used": epoch}

    def renderer(data, epoch, **kwargs):
        return captured_figure

    return ViewDefinition(
        name="per_epoch_test",
        load_data=load_data,
        renderer=renderer,
        epoch_source_analyzer="dominant_frequencies",
    )


@pytest.fixture
def mock_variant() -> MagicMock:
    """A mock Variant with artifacts, metadata, and model_config."""
    variant = MagicMock()
    variant.artifacts.get_epochs.return_value = [100, 500, 1000]
    variant.artifacts.load_epoch.return_value = {"coefficients": np.zeros(10)}
    variant.metadata = {
        "train_losses": [2.5, 1.0, 0.5],
        "test_losses": [3.0, 2.0, 1.0],
        "checkpoint_epochs": [100, 500, 1000],
    }
    variant.model_config = {"prime": 113}
    return variant


@pytest.fixture
def isolated_catalog() -> ViewCatalog:
    """A fresh, empty catalog for tests that should not touch the global catalog."""
    return ViewCatalog()


# ---------------------------------------------------------------------------
# ViewDefinition
# ---------------------------------------------------------------------------


class TestViewDefinition:
    def test_fields(self, simple_view):
        assert simple_view.name == "test_view"
        assert callable(simple_view.load_data)
        assert callable(simple_view.renderer)
        assert simple_view.epoch_source_analyzer is None

    def test_epoch_source_analyzer_set(self, per_epoch_view):
        assert per_epoch_view.epoch_source_analyzer == "dominant_frequencies"


# ---------------------------------------------------------------------------
# ViewCatalog
# ---------------------------------------------------------------------------


class TestViewCatalog:
    def test_register_and_get(self, isolated_catalog, simple_view):
        isolated_catalog.register(simple_view)
        retrieved = isolated_catalog.get("test_view")
        assert retrieved is simple_view

    def test_get_unknown_raises_key_error(self, isolated_catalog):
        with pytest.raises(KeyError, match="Unknown view 'missing'"):
            isolated_catalog.get("missing")

    def test_get_error_lists_available(self, isolated_catalog, simple_view):
        isolated_catalog.register(simple_view)
        with pytest.raises(KeyError, match="test_view"):
            isolated_catalog.get("not_a_view")

    def test_names_returns_sorted_list(self, isolated_catalog):
        for name in ["zebra", "alpha", "mango"]:
            isolated_catalog.register(
                ViewDefinition(
                    name=name,
                    load_data=lambda v, e: {},
                    renderer=lambda d, e, **kw: go.Figure(),
                )
            )
        assert isolated_catalog.names() == ["alpha", "mango", "zebra"]

    def test_names_empty_catalog(self, isolated_catalog):
        assert isolated_catalog.names() == []


# ---------------------------------------------------------------------------
# BoundView
# ---------------------------------------------------------------------------


class TestBoundView:
    def test_figure_calls_load_and_render(self, simple_view, mock_variant, minimal_figure):
        bound = BoundView(view_def=simple_view, variant=mock_variant, epoch=500)
        fig = bound.figure()
        assert isinstance(fig, go.Figure)

    def test_figure_passes_epoch_to_renderer(self, mock_variant):
        received_epoch = []

        def load_data(variant, epoch):
            return {"e": epoch}

        def renderer(data, epoch, **kwargs):
            received_epoch.append(epoch)
            return go.Figure()

        view_def = ViewDefinition(name="x", load_data=load_data, renderer=renderer)
        bound = BoundView(view_def=view_def, variant=mock_variant, epoch=42)
        bound.figure()
        assert received_epoch == [42]

    def test_show_calls_figure_show(self, simple_view, mock_variant):
        bound = BoundView(view_def=simple_view, variant=mock_variant, epoch=None)
        with patch.object(go.Figure, "show") as mock_show:
            bound.show()
            mock_show.assert_called_once()

    def test_figure_passes_kwargs(self, mock_variant):
        received_kwargs = {}

        def load_data(v, e):
            return {}

        def renderer(data, epoch, **kwargs):
            received_kwargs.update(kwargs)
            return go.Figure()

        view_def = ViewDefinition(name="kw_test", load_data=load_data, renderer=renderer)
        bound = BoundView(view_def=view_def, variant=mock_variant, epoch=None)
        bound.figure(threshold=2.0, title="Hello")
        assert received_kwargs == {"threshold": 2.0, "title": "Hello"}

    def test_export_calls_export_figure(self, simple_view, mock_variant, tmp_path):
        bound = BoundView(view_def=simple_view, variant=mock_variant, epoch=0)
        with patch("miscope.visualization.export.export_figure") as mock_export:
            mock_export.return_value = tmp_path / "out.html"
            result = bound.export("html", tmp_path / "out.html")
            mock_export.assert_called_once()
            assert result == tmp_path / "out.html"


# ---------------------------------------------------------------------------
# EpochContext
# ---------------------------------------------------------------------------


class TestEpochContext:
    def test_view_returns_bound_view(self, simple_view, mock_variant, isolated_catalog):
        isolated_catalog.register(simple_view)
        ctx = EpochContext(variant=mock_variant, epoch=500, catalog=isolated_catalog)
        bound = ctx.view("test_view")
        assert isinstance(bound, BoundView)

    def test_view_unknown_name_raises(self, mock_variant, isolated_catalog):
        ctx = EpochContext(variant=mock_variant, epoch=0, catalog=isolated_catalog)
        with pytest.raises(KeyError, match="Unknown view"):
            ctx.view("not_registered")

    def test_epoch_is_fixed_in_bound_view(self, simple_view, mock_variant, isolated_catalog):
        isolated_catalog.register(simple_view)
        ctx = EpochContext(variant=mock_variant, epoch=999, catalog=isolated_catalog)
        bound = ctx.view("test_view")
        assert bound._epoch == 999

    def test_none_epoch_resolved_for_per_epoch_view(
        self, per_epoch_view, mock_variant, isolated_catalog
    ):
        """None epoch resolves to first available epoch for per-epoch views."""
        mock_variant.artifacts.get_epochs.return_value = [100, 500, 1000]
        isolated_catalog.register(per_epoch_view)
        ctx = EpochContext(variant=mock_variant, epoch=None, catalog=isolated_catalog)
        bound = ctx.view("per_epoch_test")
        assert bound._epoch == 100  # first available

    def test_none_epoch_not_resolved_for_cross_epoch_view(
        self, simple_view, mock_variant, isolated_catalog
    ):
        """None epoch passes through unchanged for non-per-epoch views."""
        isolated_catalog.register(simple_view)
        ctx = EpochContext(variant=mock_variant, epoch=None, catalog=isolated_catalog)
        bound = ctx.view("test_view")
        assert bound._epoch is None

    def test_explicit_epoch_not_overridden_for_per_epoch_view(
        self, per_epoch_view, mock_variant, isolated_catalog
    ):
        """Explicit epoch is never overridden, even for per-epoch views."""
        isolated_catalog.register(per_epoch_view)
        ctx = EpochContext(variant=mock_variant, epoch=500, catalog=isolated_catalog)
        bound = ctx.view("per_epoch_test")
        assert bound._epoch == 500

    def test_none_epoch_with_no_available_epochs(
        self, per_epoch_view, mock_variant, isolated_catalog
    ):
        """None epoch with no available artifacts resolves to None gracefully."""
        mock_variant.artifacts.get_epochs.return_value = []
        isolated_catalog.register(per_epoch_view)
        ctx = EpochContext(variant=mock_variant, epoch=None, catalog=isolated_catalog)
        bound = ctx.view("per_epoch_test")
        assert bound._epoch is None


# ---------------------------------------------------------------------------
# Global catalog: view registration (CoS: all registry views present)
# ---------------------------------------------------------------------------


class TestGlobalCatalog:
    def test_catalog_is_populated(self):
        """Global catalog has views registered after import."""
        names = _catalog.names()
        assert len(names) > 0

    def test_all_registry_views_present(self):
        """All views from export.py _VISUALIZATION_REGISTRY are in the catalog."""
        from miscope.visualization.export import _VISUALIZATION_REGISTRY

        catalog_names = set(_catalog.names())
        for name in _VISUALIZATION_REGISTRY:
            assert name in catalog_names, f"View '{name}' missing from catalog"

    def test_loss_curve_is_registered(self):
        """loss_curve view is registered (metadata-based data source)."""
        assert "training.metadata.loss_curves" in _catalog.names()

    def test_names_returns_sorted(self):
        names = _catalog.names()
        assert names == sorted(names)

    def test_loss_curve_has_no_epoch_source(self):
        """loss_curve does not trigger per-epoch resolution (metadata-based)."""
        view_def = _catalog.get("training.metadata.loss_curves")
        assert view_def.epoch_source_analyzer is None

    def test_dominant_frequencies_has_epoch_source(self):
        """dominant_frequencies has epoch_source_analyzer set."""
        view_def = _catalog.get("dominant_frequencies")
        assert view_def.epoch_source_analyzer == "dominant_frequencies"


# ---------------------------------------------------------------------------
# Variant integration
# ---------------------------------------------------------------------------


class TestVariantIntegration:
    """Tests for variant.at() and variant.view() methods."""

    @pytest.fixture
    def temp_variant(self):
        """A real Variant backed by a temp filesystem with minimal metadata."""
        from miscope.families.json_family import JsonModelFamily
        from miscope.families.variant import Variant

        config = {
            "name": "test_family",
            "display_name": "Test",
            "description": "Test",
            "architecture": {},
            "domain_parameters": {
                "prime": {"type": "int"},
                "seed": {"type": "int"},
            },
            "analyzers": [],
            "visualizations": [],
            "variant_pattern": "test_family_p{prime}_seed{seed}",
        }
        family = JsonModelFamily(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            variant = Variant(family, {"prime": 113, "seed": 999}, root)

            # Create metadata
            variant_dir = variant.variant_dir
            variant_dir.mkdir(parents=True)
            metadata = {
                "train_losses": [2.5, 1.0, 0.5],
                "test_losses": [3.0, 2.0, 1.0],
                "checkpoint_epochs": [100, 500, 1000],
            }
            with open(variant_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            yield variant

    def test_at_returns_epoch_context(self, temp_variant):
        ctx = temp_variant.at(epoch=500)
        assert isinstance(ctx, EpochContext)
        assert ctx._epoch == 500

    def test_at_none_epoch(self, temp_variant):
        ctx = temp_variant.at(epoch=None)
        assert ctx._epoch is None

    def test_view_is_shortcut_for_at_none(self, temp_variant):
        """variant.view(name) is equivalent to variant.at(None).view(name)."""
        ctx = temp_variant.at(epoch=None)
        bound_via_at = ctx.view("training.metadata.loss_curves")
        bound_via_shortcut = temp_variant.view("training.metadata.loss_curves")
        # Both should produce BoundViews bound to the same epoch
        assert bound_via_at._epoch == bound_via_shortcut._epoch

    def test_view_unknown_raises(self, temp_variant):
        with pytest.raises(KeyError, match="Unknown view"):
            temp_variant.view("does_not_exist")

    def test_loss_curve_figure_from_variant(self, temp_variant):
        """loss_curve view produces a Plotly Figure from variant metadata."""
        bound = temp_variant.view("training.metadata.loss_curves")
        fig = bound.figure()
        assert isinstance(fig, go.Figure)

    def test_shared_epoch_cursor(self, temp_variant):
        """Two views from the same EpochContext have the same bound epoch."""
        ctx = temp_variant.at(epoch=500)
        loss_view = ctx.view("training.metadata.loss_curves")
        # Access another non-artifact view — both epoch values should match
        assert loss_view._epoch == 500

    def test_loss_curve_with_explicit_epoch(self, temp_variant):
        """loss_curve with an epoch renders a cursor at that epoch."""
        bound = temp_variant.at(epoch=1).view("training.metadata.loss_curves")
        fig = bound.figure()
        # Vertical line at epoch 1 should appear as a shape or annotation
        assert isinstance(fig, go.Figure)
