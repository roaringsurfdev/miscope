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
        ("dominant_frequencies", "dominant_frequencies", "render_dominant_frequencies"),
        ("neuron_heatmap", "neuron_activations", "render_neuron_heatmap"),
        ("freq_clusters", "neuron_freq_norm", "render_freq_clusters"),
        ("coarseness_distribution", "coarseness", "render_coarseness_distribution"),
        ("coarseness_by_neuron", "coarseness", "render_coarseness_by_neuron"),
        ("attention_heads", "attention_patterns", "render_attention_heads"),
        ("attention_freq_heatmap", "attention_freq", "render_attention_freq_heatmap"),
        ("singular_value_spectrum", "effective_dimensionality", "render_singular_value_spectrum"),
        ("perturbation_distribution", "landscape_flatness", "render_perturbation_distribution"),
        ("neuron_fourier_heatmap", "neuron_fourier", "render_neuron_fourier_heatmap"),
        ("neuron_fourier_heatmap_output", "neuron_fourier", "render_neuron_fourier_heatmap_output"),
    ]:
        _catalog.register(_make_per_epoch(name, analyzer, getattr(viz, renderer_name)))

    # --- Summary (cross-epoch aggregate) views ---

    for name, analyzer, renderer_name in [
        ("coarseness_trajectory", "coarseness", "render_coarseness_trajectory"),
        ("blob_count_trajectory", "coarseness", "render_blob_count_trajectory"),
        ("specialization_trajectory", "neuron_freq_norm", "render_specialization_trajectory"),
        ("specialization_by_frequency", "neuron_freq_norm", "render_specialization_by_frequency"),
        ("dimensionality_trajectory", "effective_dimensionality", "render_dimensionality_trajectory"),
        (
            "attention_specialization_trajectory",
            "attention_freq",
            "render_attention_specialization_trajectory",
        ),
        (
            "attention_dominant_frequencies",
            "attention_freq",
            "render_attention_dominant_frequencies",
        ),
        ("flatness_trajectory", "landscape_flatness", "render_flatness_trajectory"),
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
            name="dominant_frequencies_over_time",
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
        ("parameter_trajectory", viz.render_parameter_trajectory),
        ("trajectory_3d", viz.render_trajectory_3d),
        ("trajectory_pc1_pc3", viz.render_trajectory_pc1_pc3),
        ("trajectory_pc2_pc3", viz.render_trajectory_pc2_pc3),
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
            name="explained_variance",
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
            name="parameter_velocity",
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
            name="component_velocity",
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

    def _render_switch_count_distribution(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
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
        ("neuron_freq_trajectory", _render_neuron_freq_trajectory),
        ("switch_count_distribution", _render_switch_count_distribution),
        ("commitment_timeline", _render_commitment_timeline),
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

    def _load_repr_geometry_summary(variant: Variant, _epoch: int | None) -> dict:
        return variant.artifacts.load_summary("repr_geometry")

    def _render_geometry_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", None)
        return viz.render_geometry_timeseries(data, site=site, current_epoch=epoch)

    _catalog.register(
        ViewDefinition(
            name="geometry_timeseries",
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
        return viz.render_fisher_heatmap(
            data["epoch_data"], epoch or 0, site=site, p=data["prime"]
        )

    for name, renderer in [
        ("centroid_pca", _render_centroid_pca),
        ("centroid_distances", _render_centroid_distances),
        ("fisher_heatmap", _render_fisher_heatmap),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_repr_geometry_epoch,
                renderer=renderer,
                epoch_source_analyzer="repr_geometry",
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
            name="loss_curve",
            load_data=_load_loss_curve,
            renderer=_render_loss_curve,
            epoch_source_analyzer=None,
        )
    )


_register_all()
