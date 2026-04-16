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

from miscope.views.catalog import AnalyzerRequirement, ArtifactKind, ViewDefinition, _catalog

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
        required_analyzers=[AnalyzerRequirement(analyzer_name, ArtifactKind.EPOCH)],
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
        required_analyzers=[AnalyzerRequirement(analyzer_name, ArtifactKind.SUMMARY)],
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
        (
            "parameters.embeddings.fourier_coefficients",
            "dominant_frequencies",
            "render_dominant_frequencies",
        ),
        ("activations.mlp.neuron_heatmap", "neuron_activations", "render_neuron_heatmap"),
        ("activations.mlp.neuron_frequency_clusters", "neuron_freq_norm", "render_freq_clusters"),
        (
            "activations.mlp.neuron_freq_distribution",
            "neuron_freq_norm",
            "render_neuron_freq_distribution",
        ),
        ("activations.mlp.coarseness_distribution", "coarseness", "render_coarseness_distribution"),
        ("activations.mlp.coarseness_by_neuron", "coarseness", "render_coarseness_by_neuron"),
        ("activations.attention.head_heatmap", "attention_patterns", "render_attention_heads"),
        (
            "activations.attention.head_frequency_clusters",
            "attention_freq",
            "render_attention_freq_heatmap",
        ),
        (
            "parameters.singular_value_spectrum",
            "effective_dimensionality",
            "render_singular_value_spectrum",
        ),
        (
            "loss_landscape.perturbation_distribution",
            "landscape_flatness",
            "render_perturbation_distribution",
        ),
        (
            "activations.mlp.neuron_fourier_heatmap",
            "neuron_fourier",
            "render_neuron_fourier_heatmap",
        ),
        (
            "activations.mlp.neuron_fourier_heatmap_output",
            "neuron_fourier",
            "render_neuron_fourier_heatmap_output",
        ),
    ]:
        _catalog.register(_make_per_epoch(name, analyzer, getattr(viz, renderer_name)))

    # --- Summary (cross-epoch aggregate) views ---

    for name, analyzer, renderer_name in [
        ("activations.mlp.coarseness_trajectory", "coarseness", "render_coarseness_trajectory"),
        ("activations.mlp.blob_count_trajectory", "coarseness", "render_blob_count_trajectory"),
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
            required_analyzers=[AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH)],
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

    _pca_req = [AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH)]

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
                required_analyzers=_pca_req,
            )
        )

    # --- Group overlay: normalized per-group trajectories on shared axes ---

    def _make_overlay_renderer(col_x: int, col_y: int) -> Any:
        def renderer(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
            epochs_arr = data["epochs"].tolist()
            current_epoch = epoch if epoch is not None else epochs_arr[-1]
            return viz.render_trajectory_group_overlay(
                data, epochs_arr, current_epoch, col_x=col_x, col_y=col_y, **kwargs
            )

        return renderer

    for name, col_x, col_y in [
        ("parameters.pca.group_overlay", 0, 1),
        ("parameters.pca.group_overlay_pc2_pc3", 1, 2),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_parameter_trajectory,
                renderer=_make_overlay_renderer(col_x, col_y),
                epoch_source_analyzer=None,
                required_analyzers=_pca_req,
            )
        )

    # --- Group proximity: pairwise L2 distance between normalized trajectories ---

    def _make_proximity_renderer(col_x: int, col_y: int) -> Any:
        def renderer(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
            epochs_arr = data["epochs"].tolist()
            current_epoch = epoch if epoch is not None else epochs_arr[-1]
            return viz.render_trajectory_proximity(
                data, epochs_arr, current_epoch, col_x=col_x, col_y=col_y, **kwargs
            )

        return renderer

    for name, col_x, col_y in [
        ("parameters.pca.proximity", 0, 1),
        ("parameters.pca.proximity_pc2_pc3", 1, 2),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_parameter_trajectory,
                renderer=_make_proximity_renderer(col_x, col_y),
                epoch_source_analyzer=None,
                required_analyzers=_pca_req,
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
            required_analyzers=_pca_req,
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
            required_analyzers=_pca_req,
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
            required_analyzers=_pca_req,
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

    def _render_per_band_specialization(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            render_per_band_specialization,
        )

        return render_per_band_specialization(data["cross_epoch"], data["prime"], **kwargs)

    # --- Threshold-parameterized specialization summary views ---
    # Recomputed from neuron_dynamics raw data at any threshold,
    # so the dashboard threshold slider drives all three views.

    def _render_specialization_trajectory_dynamic(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            compute_summary_from_dynamics,
            render_specialization_trajectory,
        )

        threshold = kwargs.pop("threshold", 0.9)
        summary = compute_summary_from_dynamics(data["cross_epoch"], data["prime"], threshold)
        return render_specialization_trajectory(
            summary, epoch if epoch is not None else 0, **kwargs
        )

    def _render_specialization_by_frequency_dynamic(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        from miscope.visualization.renderers.neuron_freq_clusters import (
            compute_summary_from_dynamics,
            render_specialization_by_frequency,
        )

        threshold = kwargs.pop("threshold", 0.9)
        summary = compute_summary_from_dynamics(data["cross_epoch"], data["prime"], threshold)
        return render_specialization_by_frequency(summary, epoch, **kwargs)

    _nd_req = [AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("activations.mlp.neuron_frequency_range", _render_specialization_trajectory_dynamic),
        (
            "activations.mlp.neuron_frequency_specialization",
            _render_specialization_by_frequency_dynamic,
        ),
        ("activations.mlp.neuron_freq_trajectory", _render_neuron_freq_trajectory),
        ("activations.mlp.switch_count_distribution", _render_switch_count_distribution),
        ("activations.mlp.commitment_timeline", _render_commitment_timeline),
        ("activations.mlp.per_band_specialization", _render_per_band_specialization),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_neuron_dynamics,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_nd_req,
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
            required_analyzers=[AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)],
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
            required_analyzers=_pca_req,
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
            required_analyzers=[AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)],
        )
    )

    def _render_pc_budget(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_pc_budget(data, current_epoch=epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="geometry.pc_budget",
            load_data=_load_repr_geometry_summary,
            renderer=_render_pc_budget,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)],
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
                required_analyzers=[AnalyzerRequirement("repr_geometry", ArtifactKind.EPOCH)],
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
            required_analyzers=[
                AnalyzerRequirement("global_centroid_pca", ArtifactKind.CROSS_EPOCH)
            ],
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

    _dmd_req = [AnalyzerRequirement("centroid_dmd", ArtifactKind.CROSS_EPOCH)]

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
                required_analyzers=_dmd_req,
            )
        )

    # --- Attention Fourier views (REQ_055) ---
    # Per-epoch heatmaps and stacked temporal alignment trajectory.

    for name, analyzer, renderer_name in [
        ("parameters.attention.qk_fourier_heatmap", "attention_fourier", "render_qk_freq_heatmap"),
        ("parameters.attention.v_fourier_heatmap", "attention_fourier", "render_v_freq_heatmap"),
    ]:
        _catalog.register(_make_per_epoch(name, analyzer, getattr(viz, renderer_name)))

    def _load_attention_fourier_stacked(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_epochs("attention_fourier")

    def _render_head_alignment_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_head_alignment_trajectory(data, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="parameters.attention.head_alignment_trajectory",
            load_data=_load_attention_fourier_stacked,
            renderer=_render_head_alignment_trajectory,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("attention_fourier", ArtifactKind.EPOCH)],
        )
    )

    # --- Band concentration views (REQ_058) ---
    # Concentration trajectory and rank alignment trajectory per variant.
    # Both load from neuron_dynamics cross_epoch; rank alignment also loads
    # dominant_frequencies to get embedding band magnitudes.

    def _load_band_concentration(variant: Variant, epoch: int | None) -> dict:
        from miscope.analysis.band_concentration import compute_band_concentration_trajectory

        cross_epoch = variant.artifacts.load_cross_epoch("neuron_dynamics")
        prime = int(variant.model_config["prime"])
        threshold = 0.75
        return compute_band_concentration_trajectory(cross_epoch, threshold, prime)

    def _render_concentration_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_concentration_trajectory(data, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="analysis.band_concentration.trajectory",
            load_data=_load_band_concentration,
            renderer=_render_concentration_trajectory,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH)],
        )
    )

    def _load_rank_alignment(variant: Variant, epoch: int | None) -> dict:
        from miscope.analysis.band_concentration import compute_rank_alignment_trajectory

        cross_epoch = variant.artifacts.load_cross_epoch("neuron_dynamics")
        coeff_epochs = variant.artifacts.load_epochs("dominant_frequencies")
        prime = int(variant.model_config["prime"])
        threshold = 0.75
        return compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, threshold, prime)

    def _render_rank_alignment_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_rank_alignment_trajectory(data, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="analysis.band_concentration.rank_alignment",
            load_data=_load_rank_alignment,
            renderer=_render_rank_alignment_trajectory,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH),
                AnalyzerRequirement("dominant_frequencies", ArtifactKind.EPOCH),
            ],
        )
    )

    # --- Fourier nucleation views (REQ_063) ---
    # Always load epoch 0 — these are initialization-anchored views.
    # The epoch slider is intentionally ignored; epoch 0 is the nucleation snapshot.

    def _load_nucleation(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_epoch("fourier_nucleation", 0)

    def _render_nucleation_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_nucleation_heatmap(data, epoch=0, **kwargs)

    def _render_nucleation_frequency_gains(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_nucleation_frequency_gains(data, epoch=0, **kwargs)

    _nucleation_req = [AnalyzerRequirement("fourier_nucleation", ArtifactKind.EPOCH)]

    for name, renderer in [
        ("parameters.mlp.nucleation_heatmap", _render_nucleation_heatmap),
        ("parameters.mlp.nucleation_frequency_gains", _render_nucleation_frequency_gains),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_nucleation,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_nucleation_req,
            )
        )

    # --- Data compatibility views (REQ_064) ---
    # Computed on demand from variant params (prime, data_seed).
    # No artifact required — always available for modular addition variants.
    # The overlap view additionally tries to load the nucleation artifact
    # (epoch 0) and degrades gracefully when it is absent.

    def _load_data_compatibility(variant: Variant, epoch: int | None) -> dict:
        from miscope.analysis.data_compatibility import compute_data_compatibility

        prime = int(variant.model_config["prime"])
        data_seed = int(variant.model_config["data_seed"])
        return compute_data_compatibility(prime, data_seed)

    def _render_data_compatibility_spectrum(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_data_compatibility_spectrum(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="analysis.data_compatibility.spectrum",
            load_data=_load_data_compatibility,
            renderer=_render_data_compatibility_spectrum,
            epoch_source_analyzer=None,
            required_analyzers=[],
        )
    )

    def _load_data_compatibility_overlap(variant: Variant, epoch: int | None) -> dict:
        from miscope.analysis.data_compatibility import compute_data_compatibility

        prime = int(variant.model_config["prime"])
        data_seed = int(variant.model_config["data_seed"])
        compat = compute_data_compatibility(prime, data_seed)

        nucleation = None
        try:
            nucleation = variant.artifacts.load_epoch("fourier_nucleation", 0)
        except (FileNotFoundError, KeyError, OSError):
            pass

        return {"compatibility": compat, "nucleation": nucleation}

    def _render_data_compatibility_overlap(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_data_compatibility_overlap(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="analysis.data_compatibility.overlap",
            load_data=_load_data_compatibility_overlap,
            renderer=_render_data_compatibility_overlap,
            epoch_source_analyzer=None,
            required_analyzers=[],
        )
    )

    # --- Multi-stream specialization (REQ_066) ---
    # Loads from four artifact sources; W_E loaded selectively to avoid
    # pulling all weight matrices from parameter_snapshot across all epochs.

    def _load_multi_stream_specialization(variant: Variant, _epoch: int | None) -> dict:
        return {
            "neuron_dynamics": variant.artifacts.load_cross_epoch("neuron_dynamics"),
            "attn_fourier_epochs": variant.artifacts.load_epochs("attention_fourier"),
            "embedding_w_e": variant.artifacts.load_epochs("parameter_snapshot", fields=["W_E"]),
            "eff_dim_summary": variant.artifacts.load_summary("effective_dimensionality"),
            "prime": int(variant.model_config["prime"]),
        }

    def _render_multi_stream_specialization(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        from miscope.visualization.renderers.multi_stream_specialization import (
            render_multi_stream_specialization,
        )

        threshold_mlp = kwargs.pop("threshold_mlp", 0.7)
        threshold_embedding = kwargs.pop("threshold_embedding", 0.2)
        attn_floor = kwargs.pop("attn_floor", 0.07)
        return render_multi_stream_specialization(
            data,
            epoch,
            threshold_mlp=threshold_mlp,
            threshold_embedding=threshold_embedding,
            attn_floor=attn_floor,
            **kwargs,
        )

    _catalog.register(
        ViewDefinition(
            name="multi_stream_specialization",
            load_data=_load_multi_stream_specialization,
            renderer=_render_multi_stream_specialization,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("neuron_dynamics", ArtifactKind.CROSS_EPOCH),
                AnalyzerRequirement("attention_fourier", ArtifactKind.EPOCH),
                AnalyzerRequirement("parameter_snapshot", ArtifactKind.EPOCH),
                AnalyzerRequirement("effective_dimensionality", ArtifactKind.SUMMARY),
            ],
        )
    )

    # --- Site gradient convergence (REQ_077) ---
    # Both views load from gradient_site cross_epoch.npz.
    # Epoch is unused (full training arc); kept in signature for catalog compatibility.

    def _load_gradient_site(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("gradient_site")

    def _render_site_gradient_convergence(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_site_gradient_convergence(data, epoch, **kwargs)

    def _render_site_gradient_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_site_gradient_heatmap(data, epoch, **kwargs)

    _gradient_site_req = [AnalyzerRequirement("gradient_site", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("analysis.gradient.site_convergence", _render_site_gradient_convergence),
        ("analysis.gradient.site_heatmap", _render_site_gradient_heatmap),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_gradient_site,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_gradient_site_req,
            )
        )

    # --- Input trace views (REQ_075) ---
    # accuracy_grid: per-epoch, needs prime from model_config
    # residue_class_timeline: summary, needs prime from model_config
    # graduation_heatmap: cross-epoch, needs prime from model_config

    def _load_input_trace_epoch(variant: Variant, epoch: int | None) -> dict:
        return {
            "epoch_data": variant.artifacts.load_epoch("input_trace", epoch),  # type: ignore[arg-type]
            "prime": int(variant.model_config["prime"]),
        }

    def _render_accuracy_grid(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.input_trace import render_accuracy_grid

        return render_accuracy_grid(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="input_trace.accuracy_grid",
            load_data=_load_input_trace_epoch,
            renderer=_render_accuracy_grid,
            epoch_source_analyzer="input_trace",
            required_analyzers=[AnalyzerRequirement("input_trace", ArtifactKind.EPOCH)],
        )
    )

    def _load_input_trace_summary(variant: Variant, epoch: int | None) -> dict:
        return {
            "summary": variant.artifacts.load_summary("input_trace"),
            "prime": int(variant.model_config["prime"]),
        }

    def _render_residue_class_timeline(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.input_trace import (
            render_residue_class_accuracy_timeline,
        )

        return render_residue_class_accuracy_timeline(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="input_trace.residue_class_timeline",
            load_data=_load_input_trace_summary,
            renderer=_render_residue_class_timeline,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("input_trace", ArtifactKind.SUMMARY)],
        )
    )

    def _load_input_trace_graduation(variant: Variant, epoch: int | None) -> dict:
        return {
            "graduation": variant.artifacts.load_cross_epoch("input_trace_graduation"),
            "prime": int(variant.model_config["prime"]),
        }

    def _render_graduation_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.input_trace import render_pair_graduation_heatmap

        return render_pair_graduation_heatmap(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="input_trace.graduation_heatmap",
            load_data=_load_input_trace_graduation,
            renderer=_render_graduation_heatmap,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("input_trace_graduation", ArtifactKind.CROSS_EPOCH)
            ],
        )
    )

    # --- Frequency quality vs accuracy (REQ_053) ---

    def _load_freq_quality_vs_accuracy(variant: Variant, epoch: int | None) -> dict:
        input_summary = variant.artifacts.load_summary("input_trace")
        quality = variant.artifacts.load_epochs(
            "fourier_frequency_quality", fields=["quality_score"]
        )
        return {"input_summary": input_summary, "quality": quality}

    def _render_freq_quality_vs_accuracy(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.input_trace import (
            render_frequency_quality_vs_accuracy,
        )

        return render_frequency_quality_vs_accuracy(data, epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="input_trace.frequency_quality_vs_accuracy",
            load_data=_load_freq_quality_vs_accuracy,
            renderer=_render_freq_quality_vs_accuracy,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("input_trace", ArtifactKind.SUMMARY),
                AnalyzerRequirement("fourier_frequency_quality", ArtifactKind.EPOCH),
            ],
        )
    )

    # --- Neuron group PCA coordination views ---

    def _load_neuron_group_pca(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("neuron_group_pca")

    def _render_group_pca_cohesion(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import (
            render_neuron_group_pca_cohesion,
        )

        return render_neuron_group_pca_cohesion(data, epoch, **kwargs)

    def _render_group_spread(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_neuron_group_spread

        return render_neuron_group_spread(data, epoch, **kwargs)

    def _load_neuron_group_scatter(variant: Variant, epoch: int | None) -> dict:
        cross = variant.artifacts.load_cross_epoch("neuron_group_pca")
        norm = variant.artifacts.load_epoch("neuron_freq_norm", epoch)  # type: ignore[arg-type]
        snap = variant.artifacts.load_epoch("parameter_snapshot", epoch)  # type: ignore[arg-type]
        return {
            "group_bases": cross["group_bases"],
            "group_freqs": cross["group_freqs"],
            "W_in": snap["W_in"],
            "norm_matrix": norm["norm_matrix"],
        }

    def _render_group_scatter(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_neuron_group_scatter

        return render_neuron_group_scatter(data, epoch, **kwargs)

    _ngpca_req = [AnalyzerRequirement("neuron_group_pca", ArtifactKind.CROSS_EPOCH)]
    _ngpca_scatter_req = [
        AnalyzerRequirement("neuron_group_pca", ArtifactKind.CROSS_EPOCH),
        AnalyzerRequirement("neuron_freq_norm", ArtifactKind.EPOCH),
        AnalyzerRequirement("parameter_snapshot", ArtifactKind.EPOCH),
    ]

    _catalog.register(
        ViewDefinition(
            name="neuron_group.scatter",
            load_data=_load_neuron_group_scatter,
            renderer=_render_group_scatter,
            epoch_source_analyzer="parameter_snapshot",
            required_analyzers=_ngpca_scatter_req,
        )
    )

    _catalog.register(
        ViewDefinition(
            name="neuron_group.pca_cohesion",
            load_data=_load_neuron_group_pca,
            renderer=_render_group_pca_cohesion,
            epoch_source_analyzer=None,
            required_analyzers=_ngpca_req,
        )
    )

    _catalog.register(
        ViewDefinition(
            name="neuron_group.spread",
            load_data=_load_neuron_group_pca,
            renderer=_render_group_spread,
            epoch_source_analyzer=None,
            required_analyzers=_ngpca_req,
        )
    )

    # --- Neuron group projection views (require 'projections' field from analyzer v2) ---

    def _render_group_scatter_3d(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_neuron_group_scatter_3d

        return render_neuron_group_scatter_3d(data, epoch, **kwargs)

    def _render_group_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_neuron_group_trajectory

        return render_neuron_group_trajectory(data, epoch, **kwargs)

    def _render_group_polar_histogram(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import (
            render_neuron_group_polar_histogram,
        )

        return render_neuron_group_polar_histogram(data, epoch, **kwargs)

    for name, renderer in [
        ("neuron_group.scatter_3d", _render_group_scatter_3d),
        ("neuron_group.trajectory", _render_group_trajectory),
        ("neuron_group.polar_histogram", _render_group_polar_histogram),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_neuron_group_pca,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_ngpca_req,
            )
        )

    # --- Group centroid trajectory views ---
    # Both load centroid_pca_coords and centroid_pca_var from neuron_group_pca.

    def _render_group_centroid_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import (
            render_group_centroid_timeseries,
        )

        return render_group_centroid_timeseries(data, epoch, **kwargs)

    def _render_group_centroid_paths(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_group_centroid_paths

        return render_group_centroid_paths(data, epoch, **kwargs)

    for name, renderer in [
        ("neuron_group.centroid_pc_timeseries", _render_group_centroid_timeseries),
        ("neuron_group.centroid_paths", _render_group_centroid_paths),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_neuron_group_pca,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_ngpca_req,
            )
        )

    # --- Neuron group purity views (cross_epoch + per-epoch neuron_freq_norm) ---

    def _load_neuron_group_with_purity(variant: Variant, epoch: int | None) -> dict:
        cross = variant.artifacts.load_cross_epoch("neuron_group_pca")
        norm = variant.artifacts.load_epoch("neuron_freq_norm", epoch)  # type: ignore[arg-type]
        return {**cross, "norm_matrix": norm["norm_matrix"]}

    def _render_group_scatter_purity(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import (
            render_neuron_group_scatter_purity,
        )

        return render_neuron_group_scatter_purity(data, epoch, **kwargs)

    def _render_group_all_panels(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.neuron_group_pca import render_neuron_group_all_panels

        return render_neuron_group_all_panels(data, epoch, **kwargs)

    _ngpca_purity_req = [
        AnalyzerRequirement("neuron_group_pca", ArtifactKind.CROSS_EPOCH),
        AnalyzerRequirement("neuron_freq_norm", ArtifactKind.EPOCH),
    ]

    for name, renderer in [
        ("neuron_group.scatter_purity", _render_group_scatter_purity),
        ("neuron_group.all_groups", _render_group_all_panels),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_neuron_group_with_purity,
                renderer=renderer,
                epoch_source_analyzer="neuron_freq_norm",
                required_analyzers=_ngpca_purity_req,
            )
        )

    # --- Residue class graduation views ---

    def _load_graduation(variant: Variant, epoch: int | None) -> dict:
        cross = variant.artifacts.load_cross_epoch("input_trace_graduation")
        prime = int(variant.model_config["prime"])
        return {"data": cross, "prime": prime}

    def _render_graduation_spread(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.graduation import render_graduation_spread

        return render_graduation_spread(data["data"], data["prime"], **kwargs)

    def _render_graduation_cohesion(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.graduation import render_graduation_cohesion

        return render_graduation_cohesion(data["data"], data["prime"], **kwargs)

    _grad_req = [AnalyzerRequirement("input_trace_graduation", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("neuron_group.graduation_spread", _render_graduation_spread),
        ("neuron_group.graduation_cohesion", _render_graduation_cohesion),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_graduation,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_grad_req,
            )
        )

    # --- Transient frequency views ---

    def _load_transient(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("transient_frequency")

    def _load_transient_with_win(variant: Variant, epoch: int | None) -> dict:
        """Load transient artifact plus W_in for all snapshot epochs.

        Loads W_in at every available snapshot epoch so that both peak_scatter
        (needs only peak epoch) and pc1_cohesion (needs every epoch) can share
        this loader.  The full set is typically 50-150 epochs — similar cost to
        loading a cross-epoch parameter_snapshot artifact.
        """
        tf = variant.artifacts.load_cross_epoch("transient_frequency")
        snap_epochs = variant.artifacts.get_epochs("parameter_snapshot")
        w_in_by_epoch = {}
        for ep in snap_epochs:
            snap = variant.artifacts.load_epoch("parameter_snapshot", ep)
            w_in_by_epoch[int(ep)] = snap["W_in"]
        return {"transient": tf, "w_in_by_epoch": w_in_by_epoch}

    def _render_transient_committed_counts(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        from miscope.visualization.renderers.transient_frequency import (
            render_transient_committed_counts,
        )

        return render_transient_committed_counts(data, epoch, **kwargs)

    def _render_transient_peak_scatter(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.transient_frequency import (
            render_transient_peak_scatter,
        )

        return render_transient_peak_scatter(
            data["transient"], data["w_in_by_epoch"], epoch, **kwargs
        )

    def _render_transient_pc1_cohesion(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.transient_frequency import (
            render_transient_pc1_cohesion,
        )

        return render_transient_pc1_cohesion(
            data["transient"], data["w_in_by_epoch"], epoch, **kwargs
        )

    _transient_req = [AnalyzerRequirement("transient_frequency", ArtifactKind.CROSS_EPOCH)]
    _transient_win_req = [
        AnalyzerRequirement("transient_frequency", ArtifactKind.CROSS_EPOCH),
        AnalyzerRequirement("parameter_snapshot", ArtifactKind.EPOCH),
    ]

    _catalog.register(
        ViewDefinition(
            name="transient.committed_counts",
            load_data=_load_transient,
            renderer=_render_transient_committed_counts,
            epoch_source_analyzer=None,
            required_analyzers=_transient_req,
        )
    )

    _catalog.register(
        ViewDefinition(
            name="transient.peak_scatter",
            load_data=_load_transient_with_win,
            renderer=_render_transient_peak_scatter,
            epoch_source_analyzer=None,
            required_analyzers=_transient_win_req,
        )
    )

    _catalog.register(
        ViewDefinition(
            name="transient.pc1_cohesion",
            load_data=_load_transient_with_win,
            renderer=_render_transient_pc1_cohesion,
            epoch_source_analyzer=None,
            required_analyzers=_transient_win_req,
        )
    )

    # --- Frequency group weight geometry (REQ_090) ---
    # Both views load from freq_group_weight_geometry cross_epoch.npz.
    # matrix kwarg selects Win or Wout.

    def _load_freq_group_weight_geometry(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("freq_group_weight_geometry")

    def _render_weight_geometry_timeseries(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        matrix = kwargs.pop("matrix", "Win")
        return viz.render_weight_geometry_timeseries(data, epoch=epoch, matrix=matrix, **kwargs)

    def _render_weight_geometry_group_snapshot(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        matrix = kwargs.pop("matrix", "Win")
        return viz.render_weight_geometry_group_snapshot(data, epoch=epoch, matrix=matrix, **kwargs)

    def _render_weight_geometry_centroid_pca(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        matrix = kwargs.pop("matrix", "Win")
        return viz.render_weight_geometry_centroid_pca(data, epoch=epoch, matrix=matrix, **kwargs)

    _fgwg_req = [AnalyzerRequirement("freq_group_weight_geometry", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("weight_geometry.timeseries", _render_weight_geometry_timeseries),
        ("weight_geometry.group_snapshot", _render_weight_geometry_group_snapshot),
        ("weight_geometry.centroid_pca", _render_weight_geometry_centroid_pca),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_freq_group_weight_geometry,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_fgwg_req,
            )
        )

    # --- Intra-group manifold geometry (REQ_092) ---
    # All three views load from intragroup_manifold cross_epoch.npz.

    def _load_intragroup_manifold(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("intragroup_manifold")

    def _render_intragroup_summary(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_intragroup_manifold_summary(data, epoch=epoch, **kwargs)

    def _render_intragroup_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_intragroup_manifold_timeseries(data, epoch=epoch, **kwargs)

    def _render_intragroup_surface_fit(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_intragroup_manifold_surface_fit(data, epoch=epoch, **kwargs)

    _im_req = [AnalyzerRequirement("intragroup_manifold", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("intragroup_manifold.summary", _render_intragroup_summary),
        ("intragroup_manifold.timeseries", _render_intragroup_timeseries),
        ("intragroup_manifold.surface_fit", _render_intragroup_surface_fit),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_intragroup_manifold,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_im_req,
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
