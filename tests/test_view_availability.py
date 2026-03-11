"""Tests for REQ_062: View Availability for Variant."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miscope.views.catalog import (
    AnalyzerRequirement,
    ArtifactKind,
    ViewCatalog,
    ViewDefinition,
    _check_requirement,
)
from miscope.views.dataview_catalog import DataViewCatalog, DataViewDefinition, DataViewSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_artifacts(
    available_analyzers: list[str] | None = None,
    has_summary: dict[str, bool] | None = None,
    has_cross_epoch: dict[str, bool] | None = None,
) -> MagicMock:
    """Build a mock ArtifactLoader with configurable checks."""
    artifacts = MagicMock()
    artifacts.get_available_analyzers.return_value = available_analyzers or []
    artifacts.has_summary.side_effect = lambda name: (has_summary or {}).get(name, False)
    artifacts.has_cross_epoch.side_effect = lambda name: (has_cross_epoch or {}).get(name, False)
    return artifacts


def _make_mock_variant(artifacts: MagicMock) -> MagicMock:
    variant = MagicMock()
    variant.artifacts = artifacts
    return variant


def _noop_load(variant, epoch):
    return {}


def _noop_renderer(data, epoch, **kwargs):
    return MagicMock()


def _make_view(name: str, required_analyzers: list[AnalyzerRequirement]) -> ViewDefinition:
    return ViewDefinition(
        name=name,
        load_data=_noop_load,
        renderer=_noop_renderer,
        required_analyzers=required_analyzers,
    )


# ---------------------------------------------------------------------------
# ArtifactKind / _check_requirement
# ---------------------------------------------------------------------------


class TestCheckRequirement:
    def test_epoch_present(self):
        artifacts = _make_mock_artifacts(available_analyzers=["dominant_frequencies"])
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)
        assert _check_requirement(req, variant) is True

    def test_epoch_absent(self):
        artifacts = _make_mock_artifacts(available_analyzers=[])
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)
        assert _check_requirement(req, variant) is False

    def test_summary_present(self):
        artifacts = _make_mock_artifacts(has_summary={"repr_geometry": True})
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)
        assert _check_requirement(req, variant) is True

    def test_summary_absent(self):
        artifacts = _make_mock_artifacts(has_summary={"repr_geometry": False})
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)
        assert _check_requirement(req, variant) is False

    def test_cross_epoch_present(self):
        artifacts = _make_mock_artifacts(has_cross_epoch={"neuron_dynamics": True})
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)
        assert _check_requirement(req, variant) is True

    def test_cross_epoch_absent(self):
        artifacts = _make_mock_artifacts(has_cross_epoch={"neuron_dynamics": False})
        variant = _make_mock_variant(artifacts)
        req = AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)
        assert _check_requirement(req, variant) is False

    def test_cross_variant_raises(self):
        variant = _make_mock_variant(MagicMock())
        req = AnalyzerRequirement("some_analyzer", ArtifactKind.CROSS_VARIANT)
        with pytest.raises(NotImplementedError):
            _check_requirement(req, variant)


# ---------------------------------------------------------------------------
# ViewDefinition.is_available_for
# ---------------------------------------------------------------------------


class TestViewDefinitionIsAvailableFor:
    def test_empty_requirements_always_available(self):
        view = _make_view("metadata_view", required_analyzers=[])
        # No artifacts at all — still available
        artifacts = _make_mock_artifacts()
        variant = _make_mock_variant(artifacts)
        assert view.is_available_for(variant) is True

    def test_single_epoch_requirement_met(self):
        view = _make_view("epoch_view", [AnalyzerRequirement("dom_freq", ArtifactKind.EPOCH)])
        artifacts = _make_mock_artifacts(available_analyzers=["dom_freq"])
        variant = _make_mock_variant(artifacts)
        assert view.is_available_for(variant) is True

    def test_single_epoch_requirement_not_met(self):
        view = _make_view("epoch_view", [AnalyzerRequirement("dom_freq", ArtifactKind.EPOCH)])
        artifacts = _make_mock_artifacts(available_analyzers=[])
        variant = _make_mock_variant(artifacts)
        assert view.is_available_for(variant) is False

    def test_multi_requirement_all_met(self):
        view = _make_view(
            "multi_view",
            [
                AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH),
                AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH),
            ],
        )
        artifacts = _make_mock_artifacts(
            available_analyzers=["dominant_frequencies"],
            has_cross_epoch={"neuron_dynamics": True},
        )
        variant = _make_mock_variant(artifacts)
        assert view.is_available_for(variant) is True

    def test_multi_requirement_partial_met(self):
        view = _make_view(
            "multi_view",
            [
                AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH),
                AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH),
            ],
        )
        # cross_epoch present but epoch files absent
        artifacts = _make_mock_artifacts(
            available_analyzers=[],
            has_cross_epoch={"neuron_dynamics": True},
        )
        variant = _make_mock_variant(artifacts)
        assert view.is_available_for(variant) is False


# ---------------------------------------------------------------------------
# ViewCatalog.available_names_for
# ---------------------------------------------------------------------------


class TestViewCatalogAvailableNamesFor:
    def test_excludes_unavailable_views(self):
        cat = ViewCatalog()
        cat.register(_make_view("always_available", []))
        cat.register(_make_view("needs_epoch", [AnalyzerRequirement("dom_freq", ArtifactKind.EPOCH)]))

        artifacts = _make_mock_artifacts(available_analyzers=[])
        variant = _make_mock_variant(artifacts)

        assert cat.available_names_for(variant) == ["always_available"]

    def test_includes_available_views(self):
        cat = ViewCatalog()
        cat.register(_make_view("always_available", []))
        cat.register(_make_view("needs_epoch", [AnalyzerRequirement("dom_freq", ArtifactKind.EPOCH)]))

        artifacts = _make_mock_artifacts(available_analyzers=["dom_freq"])
        variant = _make_mock_variant(artifacts)

        assert cat.available_names_for(variant) == ["always_available", "needs_epoch"]

    def test_returns_subset_of_names(self):
        cat = ViewCatalog()
        cat.register(_make_view("view_a", []))
        cat.register(_make_view("view_b", [AnalyzerRequirement("missing", ArtifactKind.SUMMARY)]))

        artifacts = _make_mock_artifacts()
        variant = _make_mock_variant(artifacts)

        available = cat.available_names_for(variant)
        assert set(available).issubset(set(cat.names()))


# ---------------------------------------------------------------------------
# DataViewCatalog.available_names_for
# ---------------------------------------------------------------------------


class TestDataViewCatalogAvailableNamesFor:
    def _make_dataview(self, name: str, required_analyzers: list[AnalyzerRequirement]) -> DataViewDefinition:
        return DataViewDefinition(
            name=name,
            load_data=_noop_load,
            schema=DataViewSchema(fields=[]),
            required_analyzers=required_analyzers,
        )

    def test_excludes_unavailable_dataviews(self):
        cat = DataViewCatalog()
        cat.register(self._make_dataview("loss_curve", []))
        cat.register(self._make_dataview("pca_traj", [AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)]))

        artifacts = _make_mock_artifacts(has_cross_epoch={"parameter_trajectory": False})
        variant = _make_mock_variant(artifacts)

        assert cat.available_names_for(variant) == ["loss_curve"]

    def test_includes_available_dataviews(self):
        cat = DataViewCatalog()
        cat.register(self._make_dataview("loss_curve", []))
        cat.register(self._make_dataview("pca_traj", [AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)]))

        artifacts = _make_mock_artifacts(has_cross_epoch={"parameter_trajectory": True})
        variant = _make_mock_variant(artifacts)

        assert cat.available_names_for(variant) == ["loss_curve", "pca_traj"]


# ---------------------------------------------------------------------------
# EpochContext.available_views / available_dataviews
# ---------------------------------------------------------------------------


class TestEpochContextAvailability:
    def test_available_views_subset_of_catalog_names(self):
        from miscope.views.catalog import EpochContext, _catalog

        artifacts = _make_mock_artifacts()
        variant = _make_mock_variant(artifacts)
        ctx = EpochContext(variant=variant, epoch=None, catalog=_catalog)

        available = ctx.available_views()
        assert isinstance(available, list)
        assert set(available).issubset(set(_catalog.names()))

    def test_available_dataviews_subset_of_catalog_names(self):
        from miscope.views.catalog import EpochContext
        from miscope.views.dataview_catalog import _dataview_catalog

        artifacts = _make_mock_artifacts()
        variant = _make_mock_variant(artifacts)
        ctx = EpochContext(variant=variant, epoch=None)

        available = ctx.available_dataviews()
        assert isinstance(available, list)
        assert set(available).issubset(set(_dataview_catalog.names()))

    def test_metadata_view_always_available(self):
        """Loss curve requires no artifacts — always in available_views."""
        from miscope.views.catalog import EpochContext, _catalog

        artifacts = _make_mock_artifacts()
        variant = _make_mock_variant(artifacts)
        ctx = EpochContext(variant=variant, epoch=None, catalog=_catalog)

        assert "training.metadata.loss_curves" in ctx.available_views()
