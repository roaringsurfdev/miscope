"""REQ_126 PR 3: Centroid Fourier alignment.

Secondary analyzer that consumes per-site class centroids from
``representation_geometry`` (a.k.a. ``repr_geometry``), re-computes the
2D centroid PCA per site, and characterizes how well the angular order
of the projected residue classes matches a Fourier mode at the family's
modulus. This is the "fourier alignment" signal that used to live as a
fused field inside ``repr_geometry`` (REQ_126 PR 3 defused it out).

Per the REQ_126 Q2 decision, this is a small dedicated analyzer rather
than a field on a broader instrument — it composes cleanly through
analyzer chaining and audits independently. Scope: class centroids
only (activation side). Parameter-group-centroid Fourier alignment is
out of scope (the signal is less informative there).

The new analyzer recomputes the centroid PCA from the upstream
``{site}_centroids`` array — it does not re-derive on activations. The
centroid PCA matches the PCA ``repr_geometry`` ran during its own
analysis (same matrix, same primitive), so the projections are
identical up to PCA sign convention (which the Kåsa circle fit absorbs).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ALL, ArtifactInput, ResolvedInputs
from miscope.analysis.library.pca import pca
from miscope.analysis.library.shape import characterize_fourier_alignment
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# One scalar alignment per activation site; on-disk key is {site}_fourier_alignment.
SPEC = AnalyzerSpec(
    name="centroid_fourier_alignment",
    output_scope="per_epoch",
    inputs=(ArtifactInput("repr_geometry"),),
    produces_summary=True,
    outputs=(
        F.columnar(
            "fourier_alignment",
            "float64",
            ("variant", "epoch", "site"),
            "How well a site's class centroids align with a pure Fourier basis "
            "(1 = perfectly circular/Fourier, 0 = none).",
        ),
    ),
)


@register_analyzer(SPEC)
class CentroidFourierAlignmentAnalyzer:
    """Per-site Fourier alignment of class-centroid PCA projections."""

    name = "centroid_fourier_alignment"
    description = "Fourier alignment of class-centroid 2D PCA per site"
    depends_on = "repr_geometry"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute fourier alignment for each site whose centroids are present."""
        assert inputs.deps is not None and inputs.epoch is not None
        # Site set is discovered by scanning all keys for the `_centroids` suffix.
        upstream = inputs.deps.load_epoch("repr_geometry", inputs.epoch, fields=ALL)
        prime = int(context["params"]["prime"])

        result: dict[str, np.ndarray] = {}
        for key, value in upstream.items():
            if not key.endswith("_centroids"):
                continue
            site = key[: -len("_centroids")]
            alignment = _site_fourier_alignment(value, prime)
            result[f"{site}_fourier_alignment"] = np.float64(alignment)  # pyright: ignore[reportArgumentType]
        return result

    def get_summary_keys(self) -> list[str]:
        """Summary keys are populated dynamically from the per-epoch result."""
        return []

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Pass per-site scalars through unchanged for the summary collector."""
        return {k: float(v) for k, v in result.items()}


def _site_fourier_alignment(centroids: np.ndarray, prime: int) -> float:
    """2D PCA of class centroids → Fourier-alignment R²."""
    n_components = min(2, centroids.shape[0], centroids.shape[1])
    if n_components < 2:
        return 0.0
    projection_2d = pca(centroids, n_components=n_components).projections[:, :2]
    return characterize_fourier_alignment(projection_2d, prime)
