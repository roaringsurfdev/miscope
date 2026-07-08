"""Fourier Frequency Quality Analyzer (REQ_052, redefined REQ_130).

Secondary analyzer that scores how well the frequencies the model's neurons
actually organize around cover the mod-p addition task.

REQ_130 re-points this off the (retired) embedding-side ``dominant_frequencies``
onto ``neuron_grouping`` and **redefines** the metric. The old metric thresholded
embedding Fourier energy to a hard frequency set and scored the ideal tensor's R²
on that subspace; it yielded no real signal (see [[frequency-choice-frame]]).

The redefined metric is **neuron-weighted task coverage**: weight each Fourier
frequency by the fraction of neurons that compute with it (relative occupancy from
``neuron_grouping``), then compute the R² of the ideal p×p×p mod-p logit tensor
onto that *weighted* 2D Fourier subspace. With binary weights it reduces exactly
to the old hard-subspace R² (the ``coverage_hard`` companion output), so the
neuron-weighted ``quality_score`` is a smooth generalization that reflects what
the MLP actually does rather than embedding energy alone.

This metric is mod-p-task-specific and requires a frequency-indexed grouping
(the family Fourier override, ``feature_basis_name="fourier_w_in"``, where group
index g maps to frequency g+1). The universal kmeans grouping path produces
arbitrary clusters with no frequency meaning; the analyzer raises there.
"""

from typing import Any

import numpy as np

from miscope.analysis.inputs import ArtifactInput, ResolvedInputs
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec

# The grouping feature basis under which group index == frequency index.
_FREQUENCY_INDEXED_BASIS = "fourier_w_in"

SPEC = AnalyzerSpec(
    name="fourier_frequency_quality",
    output_scope="per_epoch",
    inputs=(ArtifactInput("neuron_grouping"),),
    produces_summary=True,
    outputs=(
        F.columnar(
            "quality_score",
            "float32",
            ("variant", "epoch"),
            "Overall frequency-quality score for the active frequency set at this epoch.",
        ),
        F.columnar(
            "coverage_hard",
            "float32",
            ("variant", "epoch"),
            "Fraction of neurons covered by the hard-thresholded active frequencies.",
        ),
        F.columnar(
            "active_frequencies",
            "int32",
            ("variant", "epoch", "frequency"),
            "Frequency indices judged active at this epoch (one row per frequency).",
        ),
        F.columnar(
            "k",
            "int32",
            ("variant", "epoch"),
            "Count of active frequencies.",
        ),
        F.columnar(
            "reconstruction_error",
            "float32",
            ("variant", "epoch"),
            "Residual error reconstructing the representation from the active frequencies.",
        ),
    ),
)


@register_analyzer(SPEC)
class FourierFrequencyQualityAnalyzer:
    """Scores neuron-weighted frequency coverage of the mod-p addition task.

    Reads per-epoch ``neuron_grouping`` artifacts (frequency-indexed via the
    modadd family override), weights each frequency by neuron occupancy, and
    computes the R² of the ideal mod-p logit tensor onto the neuron-weighted
    2D Fourier subspace.
    """

    name = "fourier_frequency_quality"
    depends_on = "neuron_grouping"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute neuron-weighted frequency quality for one epoch."""
        assert inputs.deps is not None and inputs.epoch is not None
        artifact = inputs.deps.load_epoch(
            "neuron_grouping", inputs.epoch, fields=["n_per_group", "feature_basis_name"]
        )
        _require_frequency_indexed(artifact)

        p = int(context["params"]["prime"])
        fourier_basis = context["fourier_basis"].cpu().numpy()  # (p, p)
        n_per_group = np.asarray(artifact["n_per_group"], dtype=np.float64)  # (K,)

        weights = _occupancy_weights(n_per_group)  # (K,) in [0, 1]
        w_rows = _frequency_weights_to_basis_rows(weights, p)  # (p,)

        quality_score = _weighted_quality_score(p, fourier_basis, w_rows)
        coverage_hard = _weighted_quality_score(p, fourier_basis, (w_rows > 0).astype(np.float64))

        active_frequencies = (np.where(n_per_group > 0)[0] + 1).astype(np.int32)

        return {
            "quality_score": np.float32(quality_score),
            "coverage_hard": np.float32(coverage_hard),
            "active_frequencies": active_frequencies,
            "k": np.int32(active_frequencies.size),
            "reconstruction_error": np.float32(1.0 - quality_score),
        }

    def get_summary_keys(self) -> list[str]:
        return ["quality_score", "coverage_hard", "reconstruction_error", "k"]

    def compute_summary(self, result: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
        return {
            "quality_score": float(result["quality_score"]),
            "coverage_hard": float(result["coverage_hard"]),
            "reconstruction_error": float(result["reconstruction_error"]),
            "k": float(result["k"]),
        }


def _require_frequency_indexed(artifact: dict[str, Any]) -> None:
    """Guard: the metric needs a grouping whose group index is a frequency.

    The modadd family override stores ``feature_basis_name="fourier_w_in"``.
    The universal kmeans path produces arbitrary clusters with no frequency
    meaning, so the metric is undefined there.
    """
    basis_name = str(artifact["feature_basis_name"])
    if basis_name != _FREQUENCY_INDEXED_BASIS:
        raise ValueError(
            f"fourier_frequency_quality requires a frequency-indexed neuron_grouping "
            f"(feature_basis_name='{_FREQUENCY_INDEXED_BASIS}'), got '{basis_name}'. "
            f"This metric is mod-p-task-specific and only applies when neurons are "
            f"grouped by Fourier frequency."
        )


def _occupancy_weights(n_per_group: np.ndarray) -> np.ndarray:
    """Relative neuron occupancy per frequency in [0, 1] (top frequency = 1).

    Max-normalization makes the weighted quality reduce to the hard-subspace R²
    in the clean limit (a frequency used by many neurons → ~1, unused → 0).
    """
    peak = float(n_per_group.max()) if n_per_group.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(n_per_group)
    return n_per_group / peak


def _frequency_weights_to_basis_rows(weights: np.ndarray, p: int) -> np.ndarray:
    """Map per-frequency weights (K,) onto Fourier basis-row weights (p,).

    Frequency g+1 occupies basis rows {2g+1, 2g+2} (sin, cos); the constant
    row 0 carries no frequency and stays 0.
    """
    w_rows = np.zeros(p, dtype=np.float64)
    for g, w in enumerate(weights):
        sin_row, cos_row = 2 * g + 1, 2 * g + 2
        if cos_row < p:
            w_rows[sin_row] = w
            w_rows[cos_row] = w
    return w_rows


def _weighted_quality_score(
    p: int,
    fourier_basis: np.ndarray,
    w_rows: np.ndarray,
) -> float:
    """Neuron-weighted R² of the ideal mod-p tensor onto the Fourier subspace.

    Generalizes the legacy hard-subspace score: each 2D frequency-pair component
    of the ideal tensor's projection is weighted by the product of the two
    frequencies' basis-row weights. With binary ``w_rows`` this recovers the
    legacy ``||T_F||² / ||T||²`` over the selected frequency set exactly.

    Exploits the structure ``T_Fa[i, b, c] = F[i, (c-b)%p]`` to avoid building
    the full p×p×p one-hot tensor, and restricts to active (nonzero-weight) rows.

    Args:
        p: Prime modulus.
        fourier_basis: Shape (p, p), rows are orthonormal basis vectors.
        w_rows: Shape (p,), per-basis-row weights in [0, 1].

    Returns:
        Weighted R² quality score in [0, 1].
    """
    active = np.where(w_rows > 0)[0]
    if active.size == 0:
        return 0.0

    F_r = fourier_basis[active, :]  # (m, p)
    w_r = w_rows[active]  # (m,)

    b_grid = np.arange(p)[:, None]  # (p, 1)
    c_grid = np.arange(p)[None, :]  # (1, p)
    a_idx = (c_grid - b_grid) % p  # (p, p): a_idx[b, c] = (c-b)%p

    T_Fa = F_r[:, a_idx]  # (m, p, p): T_Fa[i, b, c] = F_r[i, a_idx[b, c]]
    T_2D = np.einsum("jb,ibc->ijc", F_r, T_Fa)  # (m, m, p)

    pair_weights = np.outer(w_r, w_r)  # (m, m)
    weighted_energy = float(np.einsum("ij,ijc->", pair_weights, T_2D**2))
    total_energy = float(p**2)  # ||T_oh||_F^2 = p^2 (one 1 per input pair)

    return weighted_energy / total_energy
