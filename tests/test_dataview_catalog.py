"""Tests for REQ_054: Data View Catalog."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from miscope.views.dataview_catalog import (
    BoundDataView,
    DataView,
    DataViewCatalog,
    DataViewDefinition,
    DataViewField,
    DataViewSchema,
    _dataview_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema() -> DataViewSchema:
    return DataViewSchema(
        fields=[
            DataViewField(
                name="losses",
                field_type="dataframe",
                description="Training loss over epochs.",
                shape_or_columns=["epoch", "train_loss"],
            ),
        ]
    )


@pytest.fixture
def simple_dataview_def(simple_schema) -> DataViewDefinition:
    def load_data(variant, epoch):
        df = pd.DataFrame({"epoch": [1, 2], "train_loss": [2.0, 1.0]})
        return DataView(schema=simple_schema, losses=df)

    return DataViewDefinition(
        name="test.simple",
        load_data=load_data,
        schema=simple_schema,
        epoch_source_analyzer=None,
    )


@pytest.fixture
def per_epoch_dataview_def(simple_schema) -> DataViewDefinition:
    def load_data(variant, epoch):
        arr = np.zeros((5,))
        return DataView(schema=simple_schema, coefficients=arr)

    return DataViewDefinition(
        name="test.per_epoch",
        load_data=load_data,
        schema=simple_schema,
        epoch_source_analyzer="dominant_frequencies",
    )


@pytest.fixture
def mock_variant() -> MagicMock:
    variant = MagicMock()
    variant.name = "test_variant"
    variant.artifacts.get_epochs.return_value = [100, 500, 1000]
    variant.artifacts.load_epoch.return_value = {"coefficients": np.zeros((10, 8))}
    variant.metadata = {
        "train_losses": [2.5, 1.0, 0.5],
        "test_losses": [3.0, 2.0, 1.0],
        "checkpoint_epochs": [100, 500, 1000],
    }
    return variant


@pytest.fixture
def isolated_catalog() -> DataViewCatalog:
    return DataViewCatalog()


# ---------------------------------------------------------------------------
# DataViewField
# ---------------------------------------------------------------------------


class TestDataViewField:
    def test_fields(self):
        f = DataViewField(
            name="losses",
            field_type="dataframe",
            description="Training loss.",
            shape_or_columns=["epoch", "train_loss"],
        )
        assert f.name == "losses"
        assert f.field_type == "dataframe"
        assert f.description == "Training loss."
        assert f.shape_or_columns == ["epoch", "train_loss"]


# ---------------------------------------------------------------------------
# DataViewSchema
# ---------------------------------------------------------------------------


class TestDataViewSchema:
    def test_field_names_sorted(self, simple_schema):
        schema = DataViewSchema(
            fields=[
                DataViewField("zebra", "ndarray", "z"),
                DataViewField("alpha", "dataframe", "a"),
            ]
        )
        assert schema.field_names() == ["alpha", "zebra"]

    def test_get_field_returns_descriptor(self, simple_schema):
        field = simple_schema.get_field("losses")
        assert field.name == "losses"
        assert field.field_type == "dataframe"

    def test_get_field_unknown_raises(self, simple_schema):
        with pytest.raises(KeyError, match="Unknown field 'missing'"):
            simple_schema.get_field("missing")


# ---------------------------------------------------------------------------
# DataView
# ---------------------------------------------------------------------------


class TestDataView:
    def test_attribute_access(self, simple_schema):
        df = pd.DataFrame({"epoch": [1], "train_loss": [2.0]})
        view = DataView(schema=simple_schema, losses=df)
        assert isinstance(view.losses, pd.DataFrame)

    def test_ndarray_field(self, simple_schema):
        arr = np.ones((5, 3))
        view = DataView(schema=simple_schema, coefficients=arr)
        assert isinstance(view.coefficients, np.ndarray)
        assert view.coefficients.shape == (5, 3)

    def test_schema_accessible(self, simple_schema):
        view = DataView(schema=simple_schema, losses=pd.DataFrame())
        assert view.schema is simple_schema

    def test_repr(self, simple_schema):
        view = DataView(schema=simple_schema, losses=pd.DataFrame())
        assert "DataView" in repr(view)
        assert "losses" in repr(view)


# ---------------------------------------------------------------------------
# DataViewDefinition
# ---------------------------------------------------------------------------


class TestDataViewDefinition:
    def test_fields(self, simple_dataview_def, simple_schema):
        assert simple_dataview_def.name == "test.simple"
        assert callable(simple_dataview_def.load_data)
        assert simple_dataview_def.schema is simple_schema
        assert simple_dataview_def.epoch_source_analyzer is None

    def test_epoch_source_analyzer_set(self, per_epoch_dataview_def):
        assert per_epoch_dataview_def.epoch_source_analyzer == "dominant_frequencies"


# ---------------------------------------------------------------------------
# DataViewCatalog
# ---------------------------------------------------------------------------


class TestDataViewCatalog:
    def test_register_and_get(self, isolated_catalog, simple_dataview_def):
        isolated_catalog.register(simple_dataview_def)
        retrieved = isolated_catalog.get("test.simple")
        assert retrieved is simple_dataview_def

    def test_get_unknown_raises_key_error(self, isolated_catalog):
        with pytest.raises(KeyError, match="Unknown dataview 'missing'"):
            isolated_catalog.get("missing")

    def test_get_error_lists_available(self, isolated_catalog, simple_dataview_def):
        isolated_catalog.register(simple_dataview_def)
        with pytest.raises(KeyError, match="test.simple"):
            isolated_catalog.get("not_registered")

    def test_names_sorted(self, isolated_catalog, simple_schema):
        for name in ["z.view", "a.view", "m.view"]:
            isolated_catalog.register(
                DataViewDefinition(
                    name=name,
                    load_data=lambda v, e: DataView(schema=simple_schema),
                    schema=simple_schema,
                )
            )
        assert isolated_catalog.names() == ["a.view", "m.view", "z.view"]

    def test_names_empty(self, isolated_catalog):
        assert isolated_catalog.names() == []


# ---------------------------------------------------------------------------
# BoundDataView
# ---------------------------------------------------------------------------


class TestBoundDataView:
    def test_data_returns_dataview(self, simple_dataview_def, mock_variant):
        bound = BoundDataView(
            dataview_def=simple_dataview_def, variant=mock_variant, epoch=None
        )
        result = bound.data()
        assert isinstance(result, DataView)

    def test_schema_no_io(self, simple_dataview_def, mock_variant, simple_schema):
        bound = BoundDataView(
            dataview_def=simple_dataview_def, variant=mock_variant, epoch=None
        )
        assert bound.schema is simple_schema
        mock_variant.artifacts.load_epoch.assert_not_called()
        mock_variant.artifacts.load_cross_epoch.assert_not_called()

    def test_epoch_bound(self, simple_dataview_def, mock_variant):
        bound = BoundDataView(
            dataview_def=simple_dataview_def, variant=mock_variant, epoch=500
        )
        assert bound._epoch == 500

    def test_repr(self, simple_dataview_def, mock_variant):
        bound = BoundDataView(
            dataview_def=simple_dataview_def, variant=mock_variant, epoch=100
        )
        r = repr(bound)
        assert "test.simple" in r
        assert "100" in r


# ---------------------------------------------------------------------------
# EpochContext.dataview() — integration with catalog.py
# ---------------------------------------------------------------------------


class TestEpochContextDataview:
    @pytest.fixture
    def ctx_with_catalog(self, mock_variant, isolated_catalog, simple_dataview_def):
        from miscope.views.catalog import EpochContext

        isolated_catalog.register(simple_dataview_def)
        ctx = EpochContext(
            variant=mock_variant,
            epoch=500,
            dataview_catalog=isolated_catalog,
        )
        return ctx

    def test_dataview_returns_bound_dataview(self, ctx_with_catalog):
        bound = ctx_with_catalog.dataview("test.simple")
        assert isinstance(bound, BoundDataView)

    def test_dataview_epoch_fixed(self, ctx_with_catalog):
        bound = ctx_with_catalog.dataview("test.simple")
        assert bound._epoch == 500

    def test_dataview_unknown_raises(self, mock_variant, isolated_catalog):
        from miscope.views.catalog import EpochContext

        ctx = EpochContext(
            variant=mock_variant,
            epoch=0,
            dataview_catalog=isolated_catalog,
        )
        with pytest.raises(KeyError, match="Unknown dataview"):
            ctx.dataview("not_registered")

    def test_none_epoch_resolves_for_per_epoch(
        self, mock_variant, isolated_catalog, per_epoch_dataview_def
    ):
        from miscope.views.catalog import EpochContext

        mock_variant.artifacts.get_epochs.return_value = [100, 500, 1000]
        isolated_catalog.register(per_epoch_dataview_def)
        ctx = EpochContext(
            variant=mock_variant,
            epoch=None,
            dataview_catalog=isolated_catalog,
        )
        bound = ctx.dataview("test.per_epoch")
        assert bound._epoch == 100

    def test_none_epoch_unchanged_for_cross_epoch(
        self, mock_variant, isolated_catalog, simple_dataview_def
    ):
        from miscope.views.catalog import EpochContext

        isolated_catalog.register(simple_dataview_def)
        ctx = EpochContext(
            variant=mock_variant,
            epoch=None,
            dataview_catalog=isolated_catalog,
        )
        bound = ctx.dataview("test.simple")
        assert bound._epoch is None

    def test_explicit_epoch_not_overridden(
        self, mock_variant, isolated_catalog, per_epoch_dataview_def
    ):
        from miscope.views.catalog import EpochContext

        isolated_catalog.register(per_epoch_dataview_def)
        ctx = EpochContext(
            variant=mock_variant,
            epoch=999,
            dataview_catalog=isolated_catalog,
        )
        bound = ctx.dataview("test.per_epoch")
        assert bound._epoch == 999


# ---------------------------------------------------------------------------
# Global catalog: universal registrations
# ---------------------------------------------------------------------------


class TestGlobalDataViewCatalog:
    def test_catalog_is_populated(self):
        """Global dataview catalog has entries after import."""
        assert len(_dataview_catalog.names()) > 0

    def test_names_sorted(self):
        names = _dataview_catalog.names()
        assert names == sorted(names)

    def test_loss_curve_registered(self):
        assert "training.metadata.loss_curves" in _dataview_catalog.names()

    def test_fourier_coefficients_registered(self):
        assert "parameters.embeddings.fourier_coefficients" in _dataview_catalog.names()

    def test_pca_trajectory_registered(self):
        assert "parameters.pca.trajectory" in _dataview_catalog.names()

    def test_loss_curve_has_no_epoch_source(self):
        dv = _dataview_catalog.get("training.metadata.loss_curves")
        assert dv.epoch_source_analyzer is None

    def test_fourier_has_epoch_source(self):
        dv = _dataview_catalog.get("parameters.embeddings.fourier_coefficients")
        assert dv.epoch_source_analyzer == "dominant_frequencies"

    def test_pca_trajectory_has_no_epoch_source(self):
        dv = _dataview_catalog.get("parameters.pca.trajectory")
        assert dv.epoch_source_analyzer is None


# ---------------------------------------------------------------------------
# Universal dataviews: load_data output shape validation
# ---------------------------------------------------------------------------


class TestUniversalDataViewLoaders:
    @pytest.fixture
    def temp_variant(self):
        """A real Variant with minimal metadata for integration testing."""
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
            variant.variant_dir.mkdir(parents=True)
            metadata = {
                "train_losses": [2.5, 1.0, 0.5],
                "test_losses": [3.0, 2.0, 1.0],
                "checkpoint_epochs": [100, 500, 1000],
            }
            with open(variant.variant_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            yield variant

    def test_loss_curve_returns_dataview_with_losses_df(self, temp_variant):
        dv_def = _dataview_catalog.get("training.metadata.loss_curves")
        result = dv_def.load_data(temp_variant, None)
        assert isinstance(result, DataView)
        assert isinstance(result.losses, pd.DataFrame)
        assert list(result.losses.columns) == ["epoch", "train_loss", "test_loss"]
        assert len(result.losses) == 3

    def test_loss_curve_schema_accessible_before_load(self):
        dv_def = _dataview_catalog.get("training.metadata.loss_curves")
        schema = dv_def.schema
        assert isinstance(schema, DataViewSchema)
        assert "losses" in schema.field_names()

    def test_loss_curve_dataview_schema_matches(self, temp_variant):
        dv_def = _dataview_catalog.get("training.metadata.loss_curves")
        result = dv_def.load_data(temp_variant, None)
        assert result.schema is dv_def.schema


# ---------------------------------------------------------------------------
# Variant.dataview() shortcut
# ---------------------------------------------------------------------------


class TestVariantDataviewShortcut:
    @pytest.fixture
    def temp_variant(self):
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
            variant.variant_dir.mkdir(parents=True)
            metadata = {
                "train_losses": [2.5, 1.0, 0.5],
                "test_losses": [3.0, 2.0, 1.0],
                "checkpoint_epochs": [100, 500, 1000],
            }
            with open(variant.variant_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            yield variant

    def test_variant_dataview_returns_bound_dataview(self, temp_variant):
        bound = temp_variant.dataview("training.metadata.loss_curves")
        assert isinstance(bound, BoundDataView)

    def test_variant_dataview_unknown_raises(self, temp_variant):
        with pytest.raises(KeyError, match="Unknown dataview"):
            temp_variant.dataview("does_not_exist")

    def test_shared_epoch_cursor(self, temp_variant):
        """Two dataviews from the same EpochContext share the epoch."""
        ctx = temp_variant.at(epoch=500)
        loss_dv = ctx.dataview("training.metadata.loss_curves")
        pca_dv = ctx.dataview("parameters.pca.trajectory")
        assert loss_dv._epoch == pca_dv._epoch == 500

    def test_variant_dataview_schema_no_io(self, temp_variant):
        """Accessing schema does not trigger data loading."""
        bound = temp_variant.dataview("training.metadata.loss_curves")
        schema = bound.schema
        assert isinstance(schema, DataViewSchema)

    def test_variant_dataview_data_returns_dataview(self, temp_variant):
        bound = temp_variant.dataview("training.metadata.loss_curves")
        result = bound.data()
        assert isinstance(result, DataView)
