"""Intra-Group Manifold Geometry Analyzer.

Cross-epoch analyzer that fits a quadratic surface to each frequency group's
distribution in weight-space PCA coordinates at every training epoch.

The key question: does manifold formation (transition from flat blob to saddle/bowl)
happen gradually or as a sharp phase-transition event?  Running the fit at every
epoch exposes the timing relative to neuron commitment and grokking.

Consumes the 'neuron_group_pca' cross-epoch artifact — specifically the
'projections' field (n_epochs, d_mlp, 3) — rather than re-loading weight matrices.
Group membership is read from 'neuron_group_idx'.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.library.manifold_geometry import fit_quadratic_surface

_SHAPE_TO_INT = {"flat/blob": 0, "bowl": 1, "saddle": 2}
_INT_TO_SHAPE = {v: k for k, v in _SHAPE_TO_INT.items()}


class IntraGroupManifoldAnalyzer:
    """Quadratic surface fit for each frequency group at every training epoch.

    Reads 'projections' from the neuron_group_pca cross-epoch artifact so no
    weight matrices need to be re-loaded.  Group membership is fixed at the
    final epoch (same convention as NeuronGroupPCAAnalyzer).

    Cross-epoch artifact keys:
        group_freqs   int32   (n_groups,)             frequency index per group
        group_sizes   int32   (n_groups,)             neuron count per group
        epochs        int32   (n_epochs,)
        r2_linear     float32 (n_epochs, n_groups)    R² of linear planar fit
        r2_quadratic  float32 (n_epochs, n_groups)    R² of full quadratic fit
        r2_curvature  float32 (n_epochs, n_groups)    r2_quadratic − r2_linear
        a             float32 (n_epochs, n_groups)    PC1² coefficient
        b             float32 (n_epochs, n_groups)    PC2² coefficient
        c             float32 (n_epochs, n_groups)    PC1·PC2 coefficient
        shape_int     int32   (n_groups,)             final-epoch shape label
                                                      (0=flat/blob, 1=bowl, 2=saddle)
    """

    name = "intragroup_manifold"
    requires = ["neuron_group_pca"]
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, np.ndarray]:
        """Fit quadratic surfaces to each group at every epoch."""
        loader = ArtifactLoader(artifacts_dir)
        ngpca = loader.load_cross_epoch("neuron_group_pca")

        group_freqs = ngpca["group_freqs"]
        group_sizes = ngpca["group_sizes"]
        neuron_group_idx = ngpca["neuron_group_idx"]
        projections = ngpca["projections"]  # (n_epochs, d_mlp, 3)
        artifact_epochs = ngpca["epochs"]   # (n_epochs,) int32

        n_groups = len(group_freqs)
        n_epochs = len(artifact_epochs)

        if n_groups == 0:
            return _empty_result(artifact_epochs)

        group_members = _build_group_members(neuron_group_idx, n_groups)

        r2_linear = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        r2_quadratic = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        r2_curvature = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        a_coeff = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        b_coeff = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        c_coeff = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)

        for ep_idx in range(n_epochs):
            for g_idx, members in enumerate(group_members):
                proj = projections[ep_idx, members, :]  # (n_members, 3)
                if np.any(np.isnan(proj)):
                    continue
                result = fit_quadratic_surface(proj)
                r2_linear[ep_idx, g_idx] = result["r2_linear"]
                r2_quadratic[ep_idx, g_idx] = result["r2_quadratic"]
                r2_curvature[ep_idx, g_idx] = result["r2_curvature"]
                a_coeff[ep_idx, g_idx] = result["a"]
                b_coeff[ep_idx, g_idx] = result["b"]
                c_coeff[ep_idx, g_idx] = result["c"]

        shape_int = _compute_final_shapes(r2_curvature, a_coeff, b_coeff, c_coeff, n_groups)

        return {
            "group_freqs": group_freqs.astype(np.int32),
            "group_sizes": group_sizes.astype(np.int32),
            "epochs": artifact_epochs.astype(np.int32),
            "r2_linear": r2_linear,
            "r2_quadratic": r2_quadratic,
            "r2_curvature": r2_curvature,
            "a": a_coeff,
            "b": b_coeff,
            "c": c_coeff,
            "shape_int": shape_int,
        }


def decode_shapes(shape_int: np.ndarray) -> list[str]:
    """Convert integer shape labels back to human-readable strings."""
    return [_INT_TO_SHAPE.get(int(v), "flat/blob") for v in shape_int]


def _build_group_members(
    neuron_group_idx: np.ndarray,
    n_groups: int,
) -> list[np.ndarray]:
    """Reconstruct per-group member index arrays from the flat label array."""
    return [np.where(neuron_group_idx == g)[0] for g in range(n_groups)]


def _compute_final_shapes(
    r2_curvature: np.ndarray,
    a_coeff: np.ndarray,
    b_coeff: np.ndarray,
    c_coeff: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Derive shape label from the final epoch's fitted coefficients."""
    from miscope.analysis.library.manifold_geometry import _classify_shape

    shape_int = np.zeros(n_groups, dtype=np.int32)
    for g_idx in range(n_groups):
        r2c = float(r2_curvature[-1, g_idx])
        a = float(a_coeff[-1, g_idx])
        b = float(b_coeff[-1, g_idx])
        c = float(c_coeff[-1, g_idx])
        if np.isnan(r2c):
            shape_int[g_idx] = _SHAPE_TO_INT["flat/blob"]
        else:
            shape_int[g_idx] = _SHAPE_TO_INT[_classify_shape(r2c, a, b, c)]
    return shape_int


def _empty_result(epochs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(epochs)
    return {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "epochs": epochs.astype(np.int32),
        "r2_linear": np.empty((n, 0), dtype=np.float32),
        "r2_quadratic": np.empty((n, 0), dtype=np.float32),
        "r2_curvature": np.empty((n, 0), dtype=np.float32),
        "a": np.empty((n, 0), dtype=np.float32),
        "b": np.empty((n, 0), dtype=np.float32),
        "c": np.empty((n, 0), dtype=np.float32),
        "shape_int": np.array([], dtype=np.int32),
    }
