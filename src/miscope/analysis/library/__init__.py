"""Generic, reusable analysis functions.

This module contains library functions that can be used by any analyzer
and are not family-specific. They provide building blocks for analysis.

Modules:
- fourier: Fourier basis generation and projection for modular arithmetic
- activations: Activation extraction and manipulation utilities
- weights: Weight matrix extraction utilities
- trajectory: PCA projection and velocity computation
"""

from miscope.analysis.library.activations import (
    compute_grid_size_from_dataset,
    extract_attention_patterns,
    extract_mlp_activations,
    extract_residual_stream,
    get_embedding_weights,
    reshape_to_grid,
    run_with_cache,
)
from miscope.analysis.library.clustering import (
    compute_center_spread,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
)
from miscope.analysis.library.fourier import (
    compose_neuron_fourier_weights,
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_neuron_coarseness,
    extract_frequency_pairs,
    get_dominant_frequency_indices,
    get_fourier_basis,
    project_onto_fourier_basis,
)
from miscope.analysis.library.geometry import find_circularity_crossovers
from miscope.analysis.library.landscape import compute_landscape_flatness
from miscope.analysis.library.manifold_geometry import fit_quadratic_surface
from miscope.analysis.library.shape import (
    characterize_circularity,
    characterize_fourier_alignment,
    compute_arc_length,
    compute_curvature_profile,
    compute_signed_loop_area,
    detect_self_intersection,
)
from miscope.analysis.library.trajectory import (
    compute_parameter_velocity,
    flatten_snapshot,
    normalize_per_group,
)
from miscope.analysis.library.weights import (
    COMPONENT_GROUPS,
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_singular_values,
    extract_neuron_weight_matrix,
    extract_parameter_snapshot,
)

__all__ = [
    # Fourier functions
    "get_fourier_basis",
    "project_onto_fourier_basis",
    "compute_2d_fourier_transform",
    "get_dominant_frequency_indices",
    "compute_frequency_variance_fractions",
    "compute_neuron_coarseness",
    "extract_frequency_pairs",
    # Activation functions
    "extract_attention_patterns",
    "extract_mlp_activations",
    "extract_residual_stream",
    "reshape_to_grid",
    "get_embedding_weights",
    "run_with_cache",
    "compute_grid_size_from_dataset",
    # Clustering metrics
    "compute_class_centroids",
    "compute_class_radii",
    "compute_class_dimensionality",
    "compute_center_spread",
    "compute_fisher_discriminant",
    # Shape characterization
    "characterize_circularity",
    "characterize_fourier_alignment",
    # Curve-shape helpers
    "compute_arc_length",
    "detect_self_intersection",
    "compute_signed_loop_area",
    "compute_curvature_profile",
    # Geometry helpers
    "find_circularity_crossovers",
    # Fourier weight composition
    "compose_neuron_fourier_weights",
    # Weight matrix functions
    "extract_parameter_snapshot",
    "extract_neuron_weight_matrix",
    "compute_weight_singular_values",
    "compute_participation_ratio",
    "WEIGHT_MATRIX_NAMES",
    "COMPONENT_GROUPS",
    # Trajectory functions
    "flatten_snapshot",
    "compute_parameter_velocity",
    # Group centroid helpers
    "normalize_per_group",
    # Landscape flatness functions
    "compute_landscape_flatness",
    # Manifold geometry functions
    "fit_quadratic_surface",
]
