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


def _adapt_activation_freq_legacy(
    art: dict[str, Any], site: str, output_key: str
) -> dict[str, Any]:
    """Reconstruct legacy ``neuron_freq_norm`` / ``attention_freq`` shape
    ``(n_freq, d_unit)`` (per-frequency fraction matrix) from
    ``activation_basis_projection`` per-site outputs (REQ_127).

    Computes ``(power_diag + axis_a_marginal_power + axis_b_marginal_power)``
    per ``(unit, freq)`` and normalizes across the frequency axis. The
    aggregation is structurally equivalent to the old 3x3-block-summation
    formula but uses a different basis normalization, so per-column scaling
    differs from the legacy artifact (Pearson ~0.99 on canon, Spearman
    ~0.83 — heatmap visuals are functionally identical under the
    renderer's ``zmin=0, zmax=1`` normalization; absolute values shift
    per neuron).

    Works on per-epoch ``(d_unit, n_freq, n_freq)`` and stacked
    ``(n_epochs, d_unit, n_freq, n_freq)`` shapes.
    """
    import numpy as _np

    power = art[f"{site}_power"]
    p_a = art[f"{site}_axis_a_marginal_power"]
    p_b = art[f"{site}_axis_b_marginal_power"]
    diag = _np.diagonal(power, axis1=-2, axis2=-1)
    block = diag + p_a + p_b
    block_sum = block.sum(axis=-1, keepdims=True)
    fractions = (block / _np.maximum(block_sum, 1e-10)).astype(_np.float32)
    # Swap to legacy (n_freq, d_unit) layout.
    fractions = _np.moveaxis(fractions, -1, -2)
    return {output_key: fractions}


def _adapt_attention_fourier_legacy(art: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the legacy ``attention_fourier`` shape
    (``qk_freq_norms`` + ``v_freq_norms``, both per-head per-frequency fractions
    summing to 1 across frequencies) from ``weight_basis_projection`` attn_qk
    and attn_v outputs (REQ_127).

    ``qk_freq_norms[h, k]`` is the diagonal entry of the joint 2D magnitude
    matrix, normalized; ``v_freq_norms[h, k]`` is the L2 norm of attn_v
    magnitudes across the value dimension, normalized.

    Works on both per-epoch ``(n_heads, ...)`` and stacked
    ``(n_epochs, n_heads, ...)`` artifact shapes — operates on the trailing
    axes.
    """
    import numpy as _np

    qk_mag = art["attn_qk_magnitudes"]
    qk_diag = _np.diagonal(qk_mag, axis1=-2, axis2=-1)
    qk_sum = qk_diag.sum(axis=-1, keepdims=True)
    qk_freq_norms = qk_diag / _np.maximum(qk_sum, 1e-10)

    v_mag = art["attn_v_magnitudes"]
    v_band = _np.linalg.norm(v_mag, axis=-1)
    v_sum = v_band.sum(axis=-1, keepdims=True)
    v_freq_norms = v_band / _np.maximum(v_sum, 1e-10)

    return {
        "qk_freq_norms": qk_freq_norms.astype(_np.float32),
        "v_freq_norms": v_freq_norms.astype(_np.float32),
    }


def _adapt_embedding_coefficients_legacy(
    cos_coeffs: Any, sin_coeffs: Any
) -> Any:
    """Reconstruct the legacy ``dominant_frequencies`` ``coefficients`` shape from
    ``weight_basis_projection`` embedding-site cos/sin coefficients (REQ_127).

    Legacy convention: 1D vector of L2 norms across ``d_model`` arranged as
    ``[const, sin_1, cos_1, sin_2, cos_2, ...]``. The new analyzer omits the
    constant term; this adapter leaves the DC slot at zero.

    Operates on the last two axes so both single-epoch and stacked-epoch
    inputs round-trip through one helper.
    """
    import numpy as _np

    cos_norms = _np.linalg.norm(cos_coeffs, axis=-1)
    sin_norms = _np.linalg.norm(sin_coeffs, axis=-1)
    n_freq = cos_norms.shape[-1]
    out_shape = cos_norms.shape[:-1] + (1 + 2 * n_freq,)
    out = _np.zeros(out_shape, dtype=_np.float32)
    out[..., 1::2] = sin_norms
    out[..., 2::2] = cos_norms
    return out


def _register_all() -> None:
    """Register all universal views into the module-level catalog."""
    import miscope.visualization as viz
    from miscope.visualization.renderers.loss_curves import render_loss_curves_with_indicator

    # --- Per-epoch views ---

    for name, analyzer, renderer_name in [
        ("activations.mlp.neuron_heatmap", "neuron_activations", "render_neuron_heatmap"),
        ("activations.attention.head_heatmap", "attention_patterns", "render_attention_heads"),
        (
            "parameters.singular_value_spectrum",
            "weight_spectra",
            "render_singular_value_spectrum",
        ),
        (
            "loss_landscape.perturbation_distribution",
            "landscape_flatness",
            "render_perturbation_distribution",
        ),
    ]:
        _catalog.register(_make_per_epoch(name, analyzer, getattr(viz, renderer_name)))

    # --- MLP + attention activation Fourier views (REQ_127 re-point) ---
    # Re-pointed from neuron_freq_norm / attention_freq to
    # activation_basis_projection. The adapter shapes the new
    # site-namespaced output back into the legacy norm_matrix /
    # freq_matrix layout the renderers expect. Per-column scaling
    # shifts (~0.99 Pearson) are absorbed by the renderer's
    # 0-1 colorscale normalization.

    _abp_per_epoch_req = [
        AnalyzerRequirement("activation_basis_projection", ArtifactKind.EPOCH)
    ]

    def _make_activation_freq_loader(site: str, output_key: str) -> Any:
        def loader(variant: Variant, epoch: int | None) -> dict:
            art = variant.artifacts.load_epoch("activation_basis_projection", epoch)
            return _adapt_activation_freq_legacy(art, site, output_key)

        return loader

    def _render_freq_clusters(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_freq_clusters(data, epoch=epoch or 0, **kwargs)

    def _render_neuron_freq_distribution(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_neuron_freq_distribution(data, epoch=epoch or 0, **kwargs)

    def _render_attention_freq_heatmap(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_attention_freq_heatmap(data, epoch=epoch or 0, **kwargs)

    for name, render_fn, site, output_key in [
        (
            "activations.mlp.neuron_frequency_clusters",
            _render_freq_clusters,
            "mlp_out",
            "norm_matrix",
        ),
        (
            "activations.mlp.neuron_freq_distribution",
            _render_neuron_freq_distribution,
            "mlp_out",
            "norm_matrix",
        ),
        (
            "activations.attention.head_frequency_clusters",
            _render_attention_freq_heatmap,
            "attn_pattern",
            "freq_matrix",
        ),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_make_activation_freq_loader(site, output_key),
                renderer=render_fn,
                epoch_source_analyzer="activation_basis_projection",
                required_analyzers=_abp_per_epoch_req,
            )
        )

    # --- Embedding Fourier coefficients (REQ_127 re-point) ---
    # Reads weight_basis_projection embedding-site cos/sin coefficients and
    # adapts to the legacy `coefficients` shape that render_dominant_frequencies
    # expects. DC term is zero (new analyzer omits it).

    def _load_embedding_fourier_coefficients(variant: Variant, epoch: int | None) -> dict:
        art = variant.artifacts.load_epoch("weight_basis_projection", epoch)
        return {
            "coefficients": _adapt_embedding_coefficients_legacy(
                art["embedding_cos_coeffs"], art["embedding_sin_coeffs"]
            )
        }

    def _render_embedding_fourier_coefficients(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_dominant_frequencies(data, epoch=epoch or 0, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="parameters.embeddings.fourier_coefficients",
            load_data=_load_embedding_fourier_coefficients,
            renderer=_render_embedding_fourier_coefficients,
            epoch_source_analyzer="weight_basis_projection",
            required_analyzers=[
                AnalyzerRequirement("weight_basis_projection", ArtifactKind.EPOCH)
            ],
        )
    )

    # --- Summary (cross-epoch aggregate) views ---

    for name, analyzer, renderer_name in [
        (
            "parameters.effective_dimensionality",
            "weight_spectra",
            "render_dimensionality_trajectory",
        ),
        ("loss_landscape.flatness_trajectory", "landscape_flatness", "render_flatness_trajectory"),
        (
            "activations.mlp.fourier_quality_trajectory",
            "fourier_frequency_quality",
            "render_fourier_quality_trajectory",
        ),
    ]:
        _catalog.register(_make_summary(name, analyzer, getattr(viz, renderer_name)))

    # --- Attention frequency clusters summary (REQ_127 re-point) ---
    # render_attention_specialization_trajectory expects per-epoch
    # (epochs, max_frac_per_head). Built on the fly from stacked
    # activation_basis_projection by computing max-across-frequency of
    # the legacy-shaped per-head fraction matrix.

    def _load_attention_freq_clusters_summary(variant: Variant, epoch: int | None) -> dict:
        import numpy as _np

        # Selective load: only the attn_pattern power keys the adapter reads.
        # Loading all keys would stack the (d_mlp, n_freq, n_freq) mlp_out
        # cubes across every epoch — multi-GB, OOM (REQ_127).
        art = variant.artifacts.load_epochs(
            "activation_basis_projection",
            fields=[
                "attn_pattern_power",
                "attn_pattern_axis_a_marginal_power",
                "attn_pattern_axis_b_marginal_power",
            ],
        )
        # _adapt_activation_freq_legacy returns (n_epochs, n_freq, n_heads)
        # for stacked input.
        adapted = _adapt_activation_freq_legacy(
            art, "attn_pattern", "freq_matrix"
        )["freq_matrix"]
        max_frac_per_head = adapted.max(axis=1)  # (n_epochs, n_heads)
        return {
            "epochs": art["epochs"],
            "max_frac_per_head": max_frac_per_head,
        }

    def _render_attention_specialization_trajectory(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_attention_specialization_trajectory(
            data, current_epoch=epoch or 0, **kwargs
        )

    _catalog.register(
        ViewDefinition(
            name="activations.attention.frequency_clusters",
            load_data=_load_attention_freq_clusters_summary,
            renderer=_render_attention_specialization_trajectory,
            epoch_source_analyzer=None,
            required_analyzers=_abp_per_epoch_req,
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

    # REQ_127 Phase B: the timeseries panel's Fourier-alignment trace was
    # defused out of repr_geometry into centroid_fourier_alignment (REQ_126
    # PR 3). Compose the two sources so the renderer sees one flat summary.
    def _load_geometry_timeseries(variant: Variant, _epoch: int | None) -> dict:
        from miscope.views.dataviews import RepresentationGeometryTimeseries

        return RepresentationGeometryTimeseries.from_variant(variant).as_summary_dict()

    def _render_geometry_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", None)
        return viz.render_geometry_timeseries(data, site=site, current_epoch=epoch)

    _catalog.register(
        ViewDefinition(
            name="geometry.timeseries",
            load_data=_load_geometry_timeseries,
            renderer=_render_geometry_timeseries,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY),
                AnalyzerRequirement("centroid_fourier_alignment", ArtifactKind.EPOCH),
            ],
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

    def _load_network_sync(variant: Variant, _epoch: int | None) -> dict:
        result: dict = {"repr_summary": variant.artifacts.load_summary("repr_geometry")}

        ngpca_path = variant.artifacts_dir / "neuron_group_pca" / "cross_epoch.npz"
        if ngpca_path.exists():
            cross = variant.artifacts.load_cross_epoch("neuron_group_pca")
            result["group_spread"] = cross["mean_spread"]
            result["spread_epochs"] = cross["epochs"]

        try:
            vs = variant.summary
        except FileNotFoundError:
            vs = None
        if vs is not None:
            result["markers"] = {
                "second_descent_onset_epoch": vs.get("second_descent_onset_epoch"),
                "effective_dimensionality_cross_over_epoch": vs.get(
                    "effective_dimensionality_cross_over_epoch"
                ),
            }

        return result

    def _render_network_sync(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        from miscope.visualization.renderers.network_sync import render_network_sync

        return render_network_sync(data, epoch=epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="geometry.network_sync",
            load_data=_load_network_sync,
            renderer=_render_network_sync,
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

    # --- Activation DMD views (REQ_117) ---
    # Per-site windowed DMD with peak-based regime detection and per-regime
    # recursive DMD. All four views load from activation_dmd cross_epoch.npz;
    # epoch is a cursor.

    def _load_activation_dmd(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("activation_dmd")

    def _render_activation_dmd_residuals(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_activation_dmd_residuals_with_regimes(data, current_epoch=epoch)

    def _render_activation_dmd_eigenvalues(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.render_activation_dmd_eigenvalue_migration(data)

    def _render_activation_dmd_tracks(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        site = kwargs.pop("site", "mlp_out")
        return viz.render_activation_dmd_track_trajectories(data, site=site, current_epoch=epoch)

    def _render_activation_dmd_per_regime(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_activation_dmd_per_regime_vs_windowed(data, current_epoch=epoch)

    _activation_dmd_req = [AnalyzerRequirement("activation_dmd", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("activation_dmd.residuals_with_regimes", _render_activation_dmd_residuals),
        ("activation_dmd.eigenvalue_migration", _render_activation_dmd_eigenvalues),
        ("activation_dmd.track_trajectories", _render_activation_dmd_tracks),
        ("activation_dmd.per_regime_vs_windowed", _render_activation_dmd_per_regime),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_activation_dmd,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_activation_dmd_req,
            )
        )

    # --- Parameter DMD views (REQ_117 phase 2) ---
    # Per-(group, matrix) windowed + per-regime DMD on weight slices
    # partitioned by neuron_grouping at the analyzer's reference_epoch.
    # All four views load from parameter_dmd cross_epoch.npz; epoch is a cursor.

    def _load_parameter_dmd(variant: Variant, epoch: int | None) -> dict:
        return variant.artifacts.load_cross_epoch("parameter_dmd")

    def _render_parameter_dmd_residuals(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_parameter_dmd_residuals_with_regimes(data, current_epoch=epoch)

    def _render_parameter_dmd_eigenvalues(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_parameter_dmd_eigenvalue_migration(data)

    def _render_parameter_dmd_tracks(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        # Caller (dashboard) supplies group_id + matrix via kwargs.
        # Defaults: first populated group, W_in.
        group_id = kwargs.pop("group_id", None)
        matrix = kwargs.pop("matrix", "W_in")
        if group_id is None:
            populated = data["populated_groups"]
            group_id = int(populated[0]) if len(populated) > 0 else 0
        return viz.render_parameter_dmd_track_trajectories(
            data, group_id=int(group_id), matrix=matrix, current_epoch=epoch
        )

    def _render_parameter_dmd_per_regime(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_parameter_dmd_per_regime_vs_windowed(data, current_epoch=epoch)

    _parameter_dmd_req = [AnalyzerRequirement("parameter_dmd", ArtifactKind.CROSS_EPOCH)]

    for name, renderer in [
        ("parameter_dmd.residuals_with_regimes", _render_parameter_dmd_residuals),
        ("parameter_dmd.eigenvalue_migration", _render_parameter_dmd_eigenvalues),
        ("parameter_dmd.track_trajectories", _render_parameter_dmd_tracks),
        ("parameter_dmd.per_regime_vs_windowed", _render_parameter_dmd_per_regime),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_parameter_dmd,
                renderer=renderer,
                epoch_source_analyzer=None,
                required_analyzers=_parameter_dmd_req,
            )
        )

    # --- Attention Fourier views (REQ_055; re-pointed to weight_basis_projection per REQ_127) ---
    # Per-epoch heatmaps and stacked temporal alignment trajectory. The legacy
    # `qk_freq_norms` / `v_freq_norms` shape is reconstructed from the new
    # analyzer's attn_qk / attn_v outputs via `_adapt_attention_fourier_legacy`.

    def _load_attention_fourier_per_epoch(variant: Variant, epoch: int | None) -> dict:
        art = variant.artifacts.load_epoch("weight_basis_projection", epoch)
        return _adapt_attention_fourier_legacy(art)

    def _load_attention_fourier_stacked(variant: Variant, epoch: int | None) -> dict:
        # Selective load: only the attn magnitude keys the adapter reads
        # (avoid stacking the full weight_basis_projection per epoch).
        art = variant.artifacts.load_epochs(
            "weight_basis_projection",
            fields=["attn_qk_magnitudes", "attn_v_magnitudes"],
        )
        out = _adapt_attention_fourier_legacy(art)
        out["epochs"] = art["epochs"]
        return out

    def _render_qk_freq_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_qk_freq_heatmap(data, epoch=epoch or 0, **kwargs)

    def _render_v_freq_heatmap(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_v_freq_heatmap(data, epoch=epoch or 0, **kwargs)

    def _render_head_alignment_trajectory(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.render_head_alignment_trajectory(data, **kwargs)

    _wbp_per_epoch_req = [AnalyzerRequirement("weight_basis_projection", ArtifactKind.EPOCH)]

    for name, render_fn in [
        ("parameters.attention.qk_fourier_heatmap", _render_qk_freq_heatmap),
        ("parameters.attention.v_fourier_heatmap", _render_v_freq_heatmap),
    ]:
        _catalog.register(
            ViewDefinition(
                name=name,
                load_data=_load_attention_fourier_per_epoch,
                renderer=render_fn,
                epoch_source_analyzer="weight_basis_projection",
                required_analyzers=_wbp_per_epoch_req,
            )
        )

    _catalog.register(
        ViewDefinition(
            name="parameters.attention.head_alignment_trajectory",
            load_data=_load_attention_fourier_stacked,
            renderer=_render_head_alignment_trajectory,
            epoch_source_analyzer=None,
            required_analyzers=_wbp_per_epoch_req,
        )
    )

    # --- Band concentration views (REQ_058) ---
    # Concentration trajectory and rank alignment trajectory per variant.
    # Both load from neuron_dynamics cross_epoch; rank alignment also loads
    # weight_basis_projection (embedding site) and adapts to the legacy
    # `coefficients` shape that compute_rank_alignment_trajectory expects.

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
        # Selective load: only the embedding cos/sin keys the adapter reads.
        wbp_epochs = variant.artifacts.load_epochs(
            "weight_basis_projection",
            fields=["embedding_cos_coeffs", "embedding_sin_coeffs"],
        )
        coeff_epochs = {
            "epochs": wbp_epochs["epochs"],
            "coefficients": _adapt_embedding_coefficients_legacy(
                wbp_epochs["embedding_cos_coeffs"],
                wbp_epochs["embedding_sin_coeffs"],
            ),
        }
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
                AnalyzerRequirement("weight_basis_projection", ArtifactKind.EPOCH),
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
    # REQ_127: re-pointed from attention_fourier → weight_basis_projection
    # (via legacy adapter) and effective_dimensionality → weight_spectra.

    def _load_multi_stream_specialization(variant: Variant, _epoch: int | None) -> dict:
        # Selective load: only the attn magnitude keys the adapter reads.
        wbp_stacked = variant.artifacts.load_epochs(
            "weight_basis_projection",
            fields=["attn_qk_magnitudes", "attn_v_magnitudes"],
        )
        attn_legacy = _adapt_attention_fourier_legacy(wbp_stacked)
        attn_legacy["epochs"] = wbp_stacked["epochs"]
        return {
            "neuron_dynamics": variant.artifacts.load_cross_epoch("neuron_dynamics"),
            "attn_fourier_epochs": attn_legacy,
            "embedding_w_e": variant.artifacts.load_epochs("parameter_snapshot", fields=["W_E"]),
            "eff_dim_summary": variant.artifacts.load_summary("weight_spectra"),
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
                AnalyzerRequirement("weight_basis_projection", ArtifactKind.EPOCH),
                AnalyzerRequirement("parameter_snapshot", ArtifactKind.EPOCH),
                AnalyzerRequirement("weight_spectra", ArtifactKind.SUMMARY),
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

    # --- Dimensionality dynamics views (REQ_095) ---
    # Two cross-epoch views measuring PR₃ and f_top3 across three domains.

    def _load_dimensionality_timeseries(variant: Variant, epoch: int | None) -> dict:
        pt = variant.artifacts.load_cross_epoch("parameter_trajectory")
        rg = variant.artifacts.load_summary("repr_geometry")
        wg = variant.artifacts.load_cross_epoch("freq_group_weight_geometry")

        markers: dict[str, Any] = {}
        try:
            vs = variant.summary
        except FileNotFoundError:
            vs = None
        if vs is not None:
            markers["onset"] = vs.get("second_descent_onset_epoch")
            markers["fd_end"] = (vs.get("first_descent_window") or {}).get("end_epoch")
            markers["eff_xover"] = vs.get("effective_dimensionality_cross_over_epoch")

        return {
            "parameter_trajectory": pt,
            "repr_geometry_summary": rg,
            "weight_geometry": wg,
            "markers": markers,
        }

    def _render_dimensionality_timeseries(data: Any, epoch: int | None, **kwargs: Any) -> go.Figure:
        return viz.build_dimensionality_timeseries(data, epoch=epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="dimensionality.timeseries",
            load_data=_load_dimensionality_timeseries,
            renderer=_render_dimensionality_timeseries,
            epoch_source_analyzer=None,
            required_analyzers=[
                AnalyzerRequirement("parameter_trajectory", ArtifactKind.CROSS_EPOCH),
                AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY),
                AnalyzerRequirement("freq_group_weight_geometry", ArtifactKind.CROSS_EPOCH),
            ],
        )
    )

    def _load_dimensionality_state_space(variant: Variant, epoch: int | None) -> dict:
        rg = variant.artifacts.load_summary("repr_geometry")

        markers: dict[str, Any] = {}
        try:
            vs = variant.summary
        except FileNotFoundError:
            vs = None
        if vs is not None:
            markers["onset"] = vs.get("second_descent_onset_epoch")
            markers["eff_xover"] = vs.get("effective_dimensionality_cross_over_epoch")

        return {
            "repr_geometry_summary": rg,
            "markers": markers,
        }

    def _render_dimensionality_state_space(
        data: Any, epoch: int | None, **kwargs: Any
    ) -> go.Figure:
        return viz.build_dimensionality_state_space(data, epoch=epoch, **kwargs)

    _catalog.register(
        ViewDefinition(
            name="dimensionality.state_space",
            load_data=_load_dimensionality_state_space,
            renderer=_render_dimensionality_state_space,
            epoch_source_analyzer=None,
            required_analyzers=[AnalyzerRequirement("repr_geometry", ArtifactKind.SUMMARY)],
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
