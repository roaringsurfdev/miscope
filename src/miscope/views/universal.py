"""REQ_047: Universal view registrations.

All views in this module are universal instruments — they apply to any
transformer regardless of family. Families are context providers, not
view owners.

Adapter functions bridge the varied renderer signatures into the unified
ViewDefinition interface: load_data(variant, epoch) -> Any and
renderer(data, epoch, **kwargs) -> Figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

from miscope.views.catalog import ViewDefinition, _catalog

if TYPE_CHECKING:
    from miscope.families.variant import Variant


# ---------------------------------------------------------------------------
# Per-epoch views
# ---------------------------------------------------------------------------
# load_data loads a single epoch snapshot; epoch_source_analyzer enables
# None-epoch resolution to first available epoch.


def _make_per_epoch(
    name: str,
    analyzer_name: str,
    render_fn: Any,
) -> ViewDefinition:
    """Factory for per-epoch views with identical load/render patterns."""

    def load_data(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_epoch(analyzer_name, epoch)  # type: ignore[arg-type]

    def renderer(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return render_fn(data, epoch, **kwargs)

    return ViewDefinition(
        name=name,
        load_data=load_data,
        renderer=renderer,
        epoch_source_analyzer=analyzer_name,
    )


# ---------------------------------------------------------------------------
# Summary views
# ---------------------------------------------------------------------------
# load_data loads the full cross-epoch summary file; epoch is a cursor.
# Resolves None to 0 so renderers always receive an int.


def _make_summary(
    name: str,
    analyzer_name: str,
    render_fn: Any,
) -> ViewDefinition:
    """Factory for summary-based cross-epoch views."""

    def load_data(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_summary(analyzer_name)

    def renderer(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return render_fn(data, epoch if epoch is not None else 0, **kwargs)

    return ViewDefinition(
        name=name,
        load_data=load_data,
        renderer=renderer,
        epoch_source_analyzer=None,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register_all() -> None:
    """Register all universal views into the module-level catalog."""
    import miscope.visualization as viz
    from miscope.visualization.renderers.loss_curves import render_loss_curves_with_indicator

    # --- Per-epoch views ---

    for name, analyzer, renderer_name in [
        ("parameters.embeddings.fourier_coefficients", "dominant_frequencies", "render_dominant_frequencies"),
        ("activations.mlp.neuron_heatmap", "neuron_activations", "render_neuron_heatmap"),
        ("activations.mlp.neuron_frequency_clusters", "neuron_freq_norm", "render_freq_clusters"),
        ("activations.mlp.coarseness_distribution", "coarseness", "render_coarseness_distribution"),
        ("activations.mlp.coarseness_by_neuron", "coarseness", "render_coarseness_by_neuron"),
        ("activations.attention.head_heatmap", "attention_patterns", "render_attention_heads"),
        ("activations.attention.head_frequency_clusters", "attention_freq", "render_attention_freq_heatmap"),
        ("parameters.singular_value_spectrum", "effective_dimensionality", "render_singular_value_spectrum"),
        ("loss_landscape.perturbation_distribution", "landscape_flatness", "render_perturbation_distribution"),
        ("activations.mlp.neuron_fourier_heatmap", "neuron_fourier", "render_neuron_fourier_heatmap"),
        ("activations.mlp.neuron_fourier_heatmap_output", "neuron_fourier", "render_neuron_fourier_heatmap_output"),
    ]:
        _catalog.register(_make_per_epoch(name, analyzer, getattr(viz, renderer_name)))

    # --- Summary (cross-epoch aggregate) views ---

    for name, analyzer, renderer_name in [
        ("activations.mlp.coarseness_trajectory", "coarseness", "render_coarseness_trajectory"),
        ("activations.mlp.blob_count_trajectory", "coarseness", "render_blob_count_trajectory"),
        ("activations.mlp.neuron_frequency_range", "neuron_freq_norm", "render_specialization_trajectory"),
        ("activations.mlp.neuron_frequency_specialization", "neuron_freq_norm", "render_specialization_by_frequency"),
        (
            "parameters.effective_dimensionality",
            "effective_dimensionality",
            "render_dimensionality_trajectory",
        ),
        (
            "activations.attention.frequency_clusters",
            "attention_freq",
            "render_attention_specialization_trajectory",
        ),
        (
            "activations.attention.head_frequency_range",
            "attention_freq",
            "render_attention_dominant_frequencies",
        ),
        ("loss_landscape.flatness_trajectory", "landscape_flatness", "render_flatness_trajectory"),
        (
            "activations.mlp.fourier_quality_trajectory",
            "fourier_frequency_quality",
            "render_fourier_quality_trajectory",
        ),
    ]:
        _catalog.register(_make_summary(name, analyzer, getattr(viz, renderer_name)))

    # --- Cross-epoch stacked view ---
    # Loads all epochs stacked; no cursor.

    def _load_dominant_frequencies_over_time(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_epochs("dominant_frequencies")

    def _render_dominant_frequencies_over_time(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_dominant_frequencies_over_time(data, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="activations.mlp.dominant_frequencies_over_time",
            load_data=_load_dominant_frequencies_over_time,
            renderer=_render_dominant_frequencies_over_time,
            epoch_source_analyzer=None,
        )
    )

    # --- Parameter trajectory PCA views ---
    # Loads cross_epoch.npz; epoch is used as the cursor highlight.

    def _load_parameter_trajectory(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("parameter_trajectory")

    def _make_pca_renderer(render_fn: Any) -> Any:
        def renderer(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
            epochs_arr = data["epochs"].tolist()
            current_epoch = epoch if epoch is not None else epochs_arr[-1]
            group = kwargs.pop("group", "all")
            pca_result = {
                "projections": data[f"{group}__projections"],
                "explained_variance_ratio": data[f"{group}__explained_variance_ratio"],
                "explained_variance": data[f"{group}__explained_variance"],
            }
            return render_fn(pca_result, epochs_arr, current_epoch, **kwargs)

        return renderer

    for name, render_fn in [
        ("parameters.pca.pc1_pc2", viz.render_parameter_trajectory),
        ("parameters.pca.pc1_pc3", viz.render_trajectory_pc1_pc3),
        ("parameters.pca.pc2_pc3", viz.render_trajectory_pc2_pc3),
        ("parameters.pca.scatter_3d", viz.render_trajectory_3d),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_parameter_trajectory,
                renderer=_make_pca_renderer(render_fn),
                epoch_source_analyzer=None,
            )
        )

    # --- Explained variance (no cursor) ---

    def _render_explained_variance(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        group = kwargs.pop("group", "all")
        pca_result = {
            "projections": data[f"{group}__projections"],
            "explained_variance_ratio": data[f"{group}__explained_variance_ratio"],
            "explained_variance": data[f"{group}__explained_variance"],
        }
        return viz.render_explained_variance(pca_result, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="parameters.pca.explained_variance",
            load_data=_load_parameter_trajectory,
            renderer=_render_explained_variance,
            epoch_source_analyzer=None,
        )
    )

    # --- Parameter velocity ---

    def _render_parameter_velocity(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        epochs_arr = data["epochs"].tolist()
        current_epoch = epoch if epoch is not None else epochs_arr[-1]
        group = kwargs.pop("group", "all")
        velocity = data[f"{group}__velocity"]
        return viz.render_parameter_velocity(velocity, epochs_arr, current_epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="parameters.pca.velocity",
            load_data=_load_parameter_trajectory,
            renderer=_render_parameter_velocity,
            epoch_source_analyzer=None,
        )
    )

    # --- Component velocity ---

    def _render_component_velocity(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        epochs_arr = data["epochs"].tolist()
        current_epoch = epoch if epoch is not None else epochs_arr[-1]
        return viz.render_component_velocity(data, epochs_arr, current_epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="parameters.pca.component_velocity",
            load_data=_load_parameter_trajectory,
            renderer=_render_component_velocity,
            epoch_source_analyzer=None,
        )
    )

    # --- Neuron frequency trajectory views ---
    # Needs prime from model config alongside the cross-epoch artifact.

    def _load_neuron_dynamics(variant: Variant, epoch: int | None) -> dict:
        cross_epoch = variant.artifacts.load_cross_epoch("neuron_dynamics")
        prime = int(variant.model_config["prime"])
        seed = variant.model_config.get("seed")
        return {"cross_epoch": cross_epoch, "prime": prime, "seed": seed}

    def _render_neuron_freq_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            render_neuron_freq_trajectory,
        )

        sorted_by_final = kwargs.pop("sorted_by_final", False)
        return render_neuron_freq_trajectory(
            data["cross_epoch"], data["prime"], sorted_by_final=sorted_by_final, **kwargs
        )

    def _render_switch_count_distribution(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            render_switch_count_distribution,
        )

        return render_switch_count_distribution(
            data["cross_epoch"], data["prime"], seed=data["seed"], **kwargs
        )

    def _render_commitment_timeline(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            render_commitment_timeline,
        )

        return render_commitment_timeline(
            data["cross_epoch"], data["prime"], seed=data["seed"], **kwargs
        )

    for name, renderer in [
        ("activations.mlp.neuron_freq_trajectory", _render_neuron_freq_trajectory),
        ("activations.mlp.switch_count_distribution", _render_switch_count_distribution),
        ("activations.mlp.commitment_timeline", _render_commitment_timeline),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_neuron_dynamics,
                renderer=renderer,
                epoch_source_analyzer=None,
            )
        )

    # --- Representational geometry views ---
    # Summary view uses site kwarg; per-epoch views bundle prime from model config.

    def _load_centroid_pca_variance(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_summary("repr_geometry")

    def _render_centroid_pca_variance(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", None)
        return viz.render_centroid_pca_variance_summary(data, current_epoch=epoch, site=site)

    _catalog.register(
        ViewDefinition(
            name="geometry.centroid_pca_variance",
            load_data=_load_centroid_pca_variance,
            renderer=_render_centroid_pca_variance,
            epoch_source_analyzer=None,
        )
    )

    def _render_trajectory_pca_variance(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_trajectory_pca_variance(data, current_epoch=epoch)

    _catalog.register(
        ViewDefinition(
            name="parameters.pca.variance_explained",
            load_data=_load_parameter_trajectory,
            renderer=_render_trajectory_pca_variance,
            epoch_source_analyzer=None,
        )
    )

    def _load_repr_geometry_summary(variant: Variant, _epoch: int | None) -> dict:
        return variant.artifacts.load_summary("repr_geometry")

    def _render_geometry_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", None)
        return viz.render_geometry_timeseries(data, site=site, current_epoch=epoch)

    _catalog.register(
        ViewDefinition(
            name="geometry.timeseries",
            load_data=_load_repr_geometry_summary,
            renderer=_render_geometry_timeseries,
            epoch_source_analyzer=None,
        )
    )

    def _load_repr_geometry_epoch(variant: Variant, epoch: int | None) -> dict:
        return {
            "epoch_data": variant.artifacts.load_epoch("repr_geometry", epoch),  # type: ignore[arg-type]
            "prime": variant.model_config.get("prime"),
        }

    def _render_centroid_pca(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        return viz.render_centroid_pca(data["epoch_data"], epoch or 0, site=site, p=data["prime"])

    def _render_centroid_distances(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        return viz.render_centroid_distances(
            data["epoch_data"], epoch or 0, site=site, p=data["prime"]
        )

    def _render_fisher_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        return viz.render_fisher_heatmap(data["epoch_data"], epoch or 0, site=site, p=data["prime"])

    for name, renderer in [
        ("geometry.centroid_pca", _render_centroid_pca),
        ("geometry.centroid_distances", _render_centroid_distances),
        ("geometry.fisher_heatmap", _render_fisher_heatmap),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_repr_geometry_epoch,
                renderer=renderer,
                epoch_source_analyzer="repr_geometry",
            )
        )

    # --- Global centroid PCA (REQ_050) ---
    # Loads cross_epoch.npz from global_centroid_pca; epoch is a cursor
    # selecting which epoch's projections to display.

    def _load_global_centroid_pca(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("global_centroid_pca")

    def _render_centroid_global_pca(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        epochs = data["epochs"]
        resolved_epoch = epoch if epoch is not None else int(epochs[-1])
        return viz.render_centroid_global_pca(data, resolved_epoch, site=site)

    _catalog.register(
        ViewDefinition(
            name="geometry.global_centroid_pca",
            load_data=_load_global_centroid_pca,
            renderer=_render_centroid_global_pca,
            epoch_source_analyzer=None,
        )
    )

    # --- Centroid DMD views (REQ_051) ---
    # All load from centroid_dmd cross_epoch.npz; epoch is a cursor.

    def _load_centroid_dmd(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("centroid_dmd")

    def _render_dmd_eigenvalues(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        return viz.render_dmd_eigenvalues(data, site=site)

    def _render_dmd_residual(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", None)
        log_y = kwargs.pop("log_y", True)
        return viz.render_dmd_residual(data, site=site, current_epoch=epoch, log_y=log_y)

    def _render_dmd_reconstruction(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "resid_post")
        epochs_arr = data["epochs"]
        resolved_epoch = epoch if epoch is not None else int(epochs_arr[-1])
        return viz.render_dmd_reconstruction(data, resolved_epoch, site=site)

    for name, renderer in [
        ("geometry.dmd_eigenvalues", _render_dmd_eigenvalues),
        ("geometry.dmd_residual", _render_dmd_residual),
        ("geometry.dmd_reconstruction", _render_dmd_reconstruction),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_centroid_dmd,
                renderer=renderer,
                epoch_source_analyzer=None,
            )
        )

    # --- Loss curve (metadata-based, no artifact loader involved) ---
    # This is the canonical example of a non-artifact view source.

    def _load_loss_curve(variant: Variant, epoch: int | None) -> dict:
        meta = variant.metadata
        return {
            "train_losses": meta["train_losses"],
            "test_losses": meta["test_losses"],
            "checkpoint_epochs": meta.get("checkpoint_epochs", []),
        }

    def _render_loss_curve(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return render_loss_curves_with_indicator(
            data["train_losses"],
            data["test_losses"],
            current_epoch=epoch if epoch is not None else 0,
            checkpoint_epochs=data["checkpoint_epochs"],
            **kwargs,
        )

    _catalog.register(
        ViewDefinition(
            name="training.metadata.loss_curves",
            load_data=_load_loss_curve,
            renderer=_render_loss_curve,
            epoch_source_analyzer=None,
        )
    )


_register_all()
