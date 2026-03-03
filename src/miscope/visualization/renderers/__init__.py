"""Renderer functions for visualization components."""

from miscope.visualization.renderers.dominant_frequencies import (
    get_dominant_indices,
    get_fourier_basis_names,
    render_dominant_frequencies,
    render_dominant_frequencies_over_time,
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
    render_commitment_timeline,
    render_freq_clusters,
    render_freq_clusters_comparison,
    render_neuron_freq_trajectory,
    render_switch_count_distribution,
)
from miscope.visualization.renderers.repr_geometry import (
    render_centroid_distances,
    render_centroid_pca,
    render_fisher_heatmap,
    render_geometry_timeseries,
)

__all__ = [
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
    # REQ_042: Neuron dynamics
    "render_neuron_freq_trajectory",
    "render_switch_count_distribution",
    "render_commitment_timeline",
    # REQ_044: Representational geometry
    "render_geometry_timeseries",
    "render_centroid_pca",
    "render_centroid_distances",
    # REQ_045: Fisher minimum pair
    "render_fisher_heatmap",
]
