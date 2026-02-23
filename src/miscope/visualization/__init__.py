"""Visualization package for rendering analysis artifacts.

This package provides Plotly-based renderers for analysis artifacts
created by the analysis pipeline. Each renderer is dashboard-agnostic
and simply returns a plotly.graph_objects.Figure.

Per-epoch renderers accept single-epoch data + epoch number.
Cross-epoch renderers accept stacked data from load_epochs().

Usage:
    from miscope.analysis import ArtifactLoader
    from miscope.visualization import render_dominant_frequencies

    loader = ArtifactLoader(artifacts_dir)
    epoch_data = loader.load_epoch("dominant_frequencies", epoch=100)
    fig = render_dominant_frequencies(epoch_data, epoch=100)
    fig.show()  # or pass to Gradio
"""

from miscope.visualization.line_plot import line
from miscope.visualization.renderers.attention_freq import (
    render_attention_dominant_frequencies,
    render_attention_freq_heatmap,
    render_attention_specialization_trajectory,
)
from miscope.visualization.renderers.attention_patterns import (
    render_attention_heads,
    render_attention_single_head,
)
from miscope.visualization.renderers.coarseness import (
    render_blob_count_trajectory,
    render_coarseness_by_neuron,
    render_coarseness_distribution,
    render_coarseness_trajectory,
)
from miscope.visualization.renderers.dominant_frequencies import (
    get_dominant_indices,
    get_fourier_basis_names,
    render_dominant_frequencies,
    render_dominant_frequencies_over_time,
)
from miscope.visualization.renderers.effective_dimensionality import (
    render_dimensionality_trajectory,
    render_singular_value_spectrum,
)
from miscope.visualization.renderers.landscape_flatness import (
    FLATNESS_METRICS,
    render_flatness_trajectory,
    render_perturbation_distribution,
)
from miscope.visualization.renderers.neuron_activations import (
    get_most_active_neurons,
    render_neuron_across_epochs,
    render_neuron_grid,
    render_neuron_heatmap,
)
from miscope.visualization.renderers.neuron_freq_clusters import (
    get_neuron_specialization,
    get_specialized_neurons,
    render_freq_clusters,
    render_freq_clusters_comparison,
    render_specialization_by_frequency,
    render_specialization_trajectory,
)
from miscope.visualization.renderers.parameter_trajectory import (
    get_group_label,
    render_component_velocity,
    render_explained_variance,
    render_parameter_trajectory,
    render_parameter_velocity,
    render_trajectory_3d,
    render_trajectory_pc1_pc3,
    render_trajectory_pc2_pc3,
)
from miscope.visualization.renderers.neuron_fourier import (
    render_neuron_fourier_heatmap,
    render_neuron_fourier_heatmap_output,
)
from miscope.visualization.renderers.repr_geometry import (
    render_centroid_distances,
    render_centroid_pca,
    render_fisher_heatmap,
    render_geometry_timeseries,
)

__all__ = [
    # REQ_012: Line plot utility (replaces neel-plotly)
    "line",
    # REQ_004: Dominant frequencies
    "render_dominant_frequencies",
    "render_dominant_frequencies_over_time",
    "get_dominant_indices",
    "get_fourier_basis_names",
    # REQ_005: Neuron activations
    "render_neuron_heatmap",
    "render_neuron_grid",
    "render_neuron_across_epochs",
    "get_most_active_neurons",
    # REQ_006: Frequency clusters
    "render_freq_clusters",
    "render_freq_clusters_comparison",
    "get_specialized_neurons",
    "get_neuron_specialization",
    # REQ_027: Neuron specialization summary
    "render_specialization_trajectory",
    "render_specialization_by_frequency",
    # REQ_026: Attention frequency specialization
    "render_attention_freq_heatmap",
    "render_attention_specialization_trajectory",
    "render_attention_dominant_frequencies",
    # REQ_025: Attention patterns
    "render_attention_heads",
    "render_attention_single_head",
    # REQ_024: Coarseness
    "render_coarseness_trajectory",
    "render_coarseness_distribution",
    "render_blob_count_trajectory",
    "render_coarseness_by_neuron",
    # REQ_030: Effective dimensionality
    "render_dimensionality_trajectory",
    "render_singular_value_spectrum",
    # REQ_031: Landscape flatness
    "render_flatness_trajectory",
    "render_perturbation_distribution",
    "FLATNESS_METRICS",
    # REQ_029/REQ_038: Parameter trajectory
    "get_group_label",
    "render_parameter_trajectory",
    "render_explained_variance",
    "render_parameter_velocity",
    "render_component_velocity",
    # REQ_032: Trajectory PC3 projections
    "render_trajectory_3d",
    "render_trajectory_pc1_pc3",
    "render_trajectory_pc2_pc3",
    # REQ_044/045: Representational geometry
    "render_geometry_timeseries",
    "render_centroid_pca",
    "render_centroid_distances",
    "render_fisher_heatmap",
    # REQ_049: Neuron Fourier decomposition
    "render_neuron_fourier_heatmap",
    "render_neuron_fourier_heatmap_output",
]
