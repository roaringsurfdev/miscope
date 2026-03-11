"""REQ_054: Universal Data View registrations.

Registers the standard set of dataviews available in miscope.
All dataviews here are universal — they apply to any trained transformer.

Import side effect: populates the module-level _dataview_catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from miscope.views.catalog import AnalyzerRequirement, ArtifactKind
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
        assert epoch is not None, "epoch must be resolved before loading per-epoch artifact"
        epoch_data = variant.artifacts.load_epoch("dominant_frequencies", epoch)
        return DataView(schema=_fourier_schema, coefficients=epoch_data["coefficients"])

    catalog.register(
        DataViewDefinition(
            name="parameters.embeddings.fourier_coefficients",
            load_data=_load_fourier_coefficients,
            schema=_fourier_schema,
            epoch_source_analyzer="dominant_frequencies",
            required_analyzers=[AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)],
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
            required_analyzers=[AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)],
        )
    )

    # --- Neuron dynamics raw data (cross-epoch) ---
    # Exposes dominant_freq and max_frac so consumers can compute per-band
    # specialization at any threshold without knowing artifact internals.

    _neuron_dynamics_schema = DataViewSchema(
        fields=[
            DataViewField(
                name="epochs",
                field_type="ndarray",
                description="Epoch indices for each row of dominant_freq and max_frac.",
                shape_or_columns="(n_epochs,)",
            ),
            DataViewField(
                name="dominant_freq",
                field_type="ndarray",
                description=(
                    "Dominant frequency index (0-indexed) per neuron per epoch. "
                    "Shape: (n_epochs, d_mlp)."
                ),
                shape_or_columns="(n_epochs, d_mlp)",
            ),
            DataViewField(
                name="max_frac",
                field_type="ndarray",
                description=(
                    "Fraction of Fourier norm in dominant frequency per neuron per epoch. "
                    "Shape: (n_epochs, d_mlp). Use this with a threshold to determine commitment."
                ),
                shape_or_columns="(n_epochs, d_mlp)",
            ),
            DataViewField(
                name="stored_threshold",
                field_type="ndarray",
                description=(
                    "Commitment threshold used at analysis time (3/n_freq). "
                    "Shape: (1,). Reference value — consumers may apply any threshold to max_frac."
                ),
                shape_or_columns="(1,)",
            ),
        ]
    )

    def _load_neuron_dynamics_raw(variant: Variant, epoch: int | None) -> DataView:
        data = variant.artifacts.load_cross_epoch("neuron_dynamics")
        return DataView(
            schema=_neuron_dynamics_schema,
            epochs=data["epochs"],
            dominant_freq=data["dominant_freq"],
            max_frac=data["max_frac"],
            stored_threshold=data["threshold"],
        )

    catalog.register(
        DataViewDefinition(
            name="neuron_dynamics.raw",
            load_data=_load_neuron_dynamics_raw,
            schema=_neuron_dynamics_schema,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)],
        )
    )


    # --- Attention Fourier spectrum (per-epoch) (REQ_055) ---

    _attn_fourier_schema = DataViewSchema(
        fields=[
            DataViewField(
                name="qk_freq_norms",
                field_type="ndarray",
                description=(
                    "Per-head QK^T energy fraction in each Fourier frequency. "
                    "Shape: (n_heads, n_freq). Rows sum to ~1."
                ),
                shape_or_columns="(n_heads, n_freq)",
            ),
            DataViewField(
                name="v_freq_norms",
                field_type="ndarray",
                description=(
                    "Per-head V energy fraction in each Fourier frequency. "
                    "Shape: (n_heads, n_freq). Rows sum to ~1."
                ),
                shape_or_columns="(n_heads, n_freq)",
            ),
        ]
    )

    def _load_attn_fourier(variant: Variant, epoch: int | None) -> DataView:
        assert epoch is not None, "epoch must be resolved before loading per-epoch artifact"
        epoch_data = variant.artifacts.load_epoch("attention_fourier", epoch)
        return DataView(
            schema=_attn_fourier_schema,
            qk_freq_norms=epoch_data["qk_freq_norms"],
            v_freq_norms=epoch_data["v_freq_norms"],
        )

    catalog.register(
        DataViewDefinition(
            name="attention.fourier.qk_spectrum",
            load_data=_load_attn_fourier,
            schema=_attn_fourier_schema,
            epoch_source_analyzer="attention_fourier",
            required_analyzers=[AnalyzerRequirement("attention_fourier", ArtifactKind.EPOCH)],
        )
    )


_register_all()
