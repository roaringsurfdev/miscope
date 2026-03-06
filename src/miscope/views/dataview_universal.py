"""REQ_054: Universal Data View registrations.

Registers the standard set of dataviews available in miscope.
All dataviews here are universal — they apply to any trained transformer.

Import side effect: populates the module-level _dataview_catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from miscope.views.dataview_catalog import (
    DataView,
    DataViewCatalog,
    DataViewDefinition,
    DataViewField,
    DataViewSchema,
    _dataview_catalog,
)

if TYPE_CHECKING:
    from miscope.families.variant import Variant


def _register_all(catalog: DataViewCatalog = _dataview_catalog) -> None:
    """Register all universal dataviews into the given catalog."""

    # --- Loss curve (metadata-based, no artifact loader involved) ---

    _loss_curve_schema = DataViewSchema(
        fields=[
            DataViewField(
                name="losses",
                field_type="dataframe",
                description="Training and test loss at each recorded epoch.",
                shape_or_columns=["epoch", "train_loss", "test_loss"],
            ),
        ]
    )

    def _load_loss_curve(variant: Variant, epoch: int | None) -> DataView:
        meta = variant.metadata
        train_losses = meta["train_losses"]
        test_losses = meta["test_losses"]
        n = min(len(train_losses), len(test_losses))
        losses_df = pd.DataFrame(
            {
                "epoch": list(range(n)),
                "train_loss": list(train_losses)[:n],
                "test_loss": list(test_losses)[:n],
            }
        )
        return DataView(schema=_loss_curve_schema, losses=losses_df)

    catalog.register(
        DataViewDefinition(
            name="training.metadata.loss_curves",
            load_data=_load_loss_curve,
            schema=_loss_curve_schema,
            epoch_source_analyzer=None,
        )
    )

    # --- Fourier coefficients (per-epoch artifact) ---

    _fourier_schema = DataViewSchema(
        fields=[
            DataViewField(
                name="coefficients",
                field_type="ndarray",
                description=(
                    "Fourier coefficients for each embedding dimension at this epoch. "
                    "Shape: (n_freqs, n_vocab)."
                ),
                shape_or_columns="(n_freqs, n_vocab)",
            ),
        ]
    )

    def _load_fourier_coefficients(variant: Variant, epoch: int | None) -> DataView:
        epoch_data = variant.artifacts.load_epoch("dominant_frequencies", epoch)
        return DataView(schema=_fourier_schema, coefficients=epoch_data["coefficients"])

    catalog.register(
        DataViewDefinition(
            name="parameters.embeddings.fourier_coefficients",
            load_data=_load_fourier_coefficients,
            schema=_fourier_schema,
            epoch_source_analyzer="dominant_frequencies",
        )
    )

    # --- Parameter trajectory PCA (cross-epoch artifact) ---

    _pca_trajectory_schema = DataViewSchema(
        fields=[
            DataViewField(
                name="epochs",
                field_type="ndarray",
                description="Epoch indices for each row of the PCA projections.",
                shape_or_columns="(n_epochs,)",
            ),
            DataViewField(
                name="projections",
                field_type="ndarray",
                description=(
                    "PCA projections of flattened parameter snapshots (all groups). "
                    "Shape: (n_epochs, n_components)."
                ),
                shape_or_columns="(n_epochs, n_components)",
            ),
            DataViewField(
                name="explained_variance_ratio",
                field_type="ndarray",
                description="Fraction of variance explained by each PC (all groups).",
                shape_or_columns="(n_components,)",
            ),
            DataViewField(
                name="explained_variance",
                field_type="ndarray",
                description="Absolute variance explained by each PC (all groups).",
                shape_or_columns="(n_components,)",
            ),
            DataViewField(
                name="velocity",
                field_type="ndarray",
                description=(
                    "Parameter update velocity per epoch (all groups). "
                    "Shape: (n_epochs,)."
                ),
                shape_or_columns="(n_epochs,)",
            ),
        ]
    )

    def _load_pca_trajectory(variant: Variant, epoch: int | None) -> DataView:
        data = variant.artifacts.load_cross_epoch("parameter_trajectory")
        return DataView(
            schema=_pca_trajectory_schema,
            epochs=data["epochs"],
            projections=data["all__projections"],
            explained_variance_ratio=data["all__explained_variance_ratio"],
            explained_variance=data["all__explained_variance"],
            velocity=data["all__velocity"],
        )

    catalog.register(
        DataViewDefinition(
            name="parameters.pca.trajectory",
            load_data=_load_pca_trajectory,
            schema=_pca_trajectory_schema,
            epoch_source_analyzer=None,
        )
    )


_register_all()
