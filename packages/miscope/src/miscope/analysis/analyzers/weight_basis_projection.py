"""REQ_126: Weight basis-projection analyzer.

Universal weight-side basis-projection analyzer that consumes the family's
declared :class:`BasisProjectionSite` set and projects each composed matrix
onto the family-supplied basis. The transform step uses only REQ_109's
``project_onto_fourier_basis`` primitive — there is no inline FFT,
einsum-rolled projection, or other ad-hoc basis math in this module.

Absorbed analyzers (target reproducibility on canon):
    - ``dominant_frequencies`` ← site ``embedding``
    - ``attention_fourier``    ← sites ``attn_v`` and ``attn_qk``
    - ``neuron_fourier``       ← sites ``mlp_in`` and ``mlp_out``
    - one-shot projection of ``fourier_nucleation`` ← site ``mlp_in``

The iterative-refinement portion of ``fourier_nucleation`` stays in its
own analyzer per the Atlas (retain bucket).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ALL, ArtifactInput, ResolvedInputs
from miscope.analysis.library.fourier_basis import (
    PeriodicFourierBasis,
    get_fourier_basis,
    project_onto_fourier_basis,
)
from miscope.analysis.output_schema import OutputField as F
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec
from miscope.core.basis_projection import BasisProjectionSite

# Dense Fourier-coefficient cubes, one set per weight site (on-disk key is
# {site}_{field}). 1D sites (embedding, attn_v, mlp_in, mlp_out) emit cos/sin
# coeffs + phases + dominant_frequency; the 2D site (attn_qk) emits the four
# cross-coeff cubes + dominant_frequency_pair. The union is declared here as
# logical fields keyed by `site`; absent fields for a given site are simply not
# written. All coefficient cubes are tensors; only the frequency-axis labels and
# per-unit dominant frequency flatten to columns. The per-head attention site
# (attn_v) gives `dominant_frequency` a `head` coordinate (REQ_136); it is emitted
# uniform-rank as (n_heads, n_units), n_heads=1 for non-attention 1D sites.
_WBP_TENSOR_FIELDS = (
    ("cos_coeffs", "Cosine Fourier coefficients (1D sites)."),
    ("sin_coeffs", "Sine Fourier coefficients (1D sites)."),
    ("cos_cos_coeffs", "Cos×cos cross-coefficients (2D attn_qk site)."),
    ("cos_sin_coeffs", "Cos×sin cross-coefficients (2D attn_qk site)."),
    ("sin_cos_coeffs", "Sin×cos cross-coefficients (2D attn_qk site)."),
    ("sin_sin_coeffs", "Sin×sin cross-coefficients (2D attn_qk site)."),
    ("magnitudes", "Per-frequency coefficient magnitudes."),
    ("phases", "Per-frequency phase angles (1D sites)."),
    ("power", "Per-frequency power (magnitude squared)."),
    ("fractional_power", "Power normalized to fraction of total per output unit."),
    ("dominant_frequency_pair", "Argmax (k_a, k_b) frequency pair per head (2D site)."),
)

SPEC = AnalyzerSpec(
    name="weight_basis_projection",
    output_scope="per_epoch",
    inputs=(ArtifactInput("parameter_snapshot"),),
    outputs=(
        *(
            F.tensor(name, "float64", ("variant", "epoch", "site"), desc)
            for name, desc in _WBP_TENSOR_FIELDS
        ),
        F.columnar(
            "dominant_frequency",
            "int32",
            ("variant", "epoch", "site", "head", "row_id"),
            "Argmax frequency index per output unit (1D sites; head = attention "
            "head, head=0 for non-attention sites; row_id = output dim).",
        ),
        F.columnar(
            "frequencies",
            "int32",
            ("variant", "epoch", "frequency"),
            "Frequency-index axis labels for the coefficient arrays.",
        ),
    ),
)


@register_analyzer(SPEC)
class WeightBasisProjectionAnalyzer:
    """Project family-declared weight sites onto the family's basis.

    For each :class:`BasisProjectionSite` the family declares in
    ``context["weight_basis_projection_sites"]``, runs the site's composer on the
    ``parameter_snapshot`` artifact and projects the result via REQ_109
    primitives. Outputs are namespaced by site (``{site_name}_*`` keys).
    """

    name = "weight_basis_projection"
    description = "Project family-declared weight sites onto a family-supplied basis"
    depends_on = "parameter_snapshot"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Project each declared site for the current epoch."""
        assert inputs.deps is not None and inputs.epoch is not None
        # Each site's composer may read any weight matrix → load all.
        snapshot = inputs.deps.load_epoch("parameter_snapshot", inputs.epoch, fields=ALL)
        sites: tuple[BasisProjectionSite, ...] = context.get("weight_basis_projection_sites", ())
        if not sites:
            return {}

        # The basis is family-parameterized; here the modadd family supplies
        # ``prime`` and the basis is constructed via the REQ_109 primitive.
        # Other families that supply a different basis would change this
        # construction step without touching the per-site projection loop.
        prime = int(context["params"]["prime"])
        basis = get_fourier_basis(prime)

        result: dict[str, np.ndarray] = {}
        for site in sites:
            matrix = site.compose(snapshot, context)
            site_result = _project_site(matrix, basis, site.period_axes)
            for key, value in site_result.items():
                result[f"{site.name}_{key}"] = value
        return result


def _project_site(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Project a composed matrix onto the basis along one or two period axes."""
    if len(period_axes) == 1:
        return _project_1d(matrix, basis, period_axes[0])
    if len(period_axes) == 2:
        return _project_2d(matrix, basis, period_axes)
    raise ValueError(f"weight_basis_projection supports 1 or 2 period axes; got {period_axes}")


def _project_1d(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axis: int,
) -> dict[str, np.ndarray]:
    """1D projection: returns cos/sin coeffs plus derived metrics."""
    result = project_onto_fourier_basis(matrix, basis, period_axis=period_axis)
    # Uniform-rank `dominant_frequency` (REQ_136): per-head sites (attn_v) yield a
    # (n_heads, n_units) argmax; give single-head sites a singleton head axis so the
    # field's declared `head` coordinate keys one shape. Tensors are left untouched.
    dominant_frequency = result.dominant_frequency.astype(np.int32)
    if dominant_frequency.ndim == 1:
        dominant_frequency = dominant_frequency[np.newaxis, :]
    return {
        "cos_coeffs": result.cos_coeffs.astype(np.float64),
        "sin_coeffs": result.sin_coeffs.astype(np.float64),
        "magnitudes": result.magnitudes.astype(np.float64),
        "phases": result.phases.astype(np.float64),
        "power": result.power.astype(np.float64),
        "fractional_power": result.fractional_power.astype(np.float64),
        "dominant_frequency": dominant_frequency,
        "frequencies": basis.frequencies.astype(np.int32),
    }


def _project_2d(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, int],
) -> dict[str, np.ndarray]:
    """2D projection: compose two 1D projections, derive joint metrics.

    Returns four coefficient cubes — one per ``(sin|cos) × (sin|cos)`` basis
    outer product — plus joint magnitudes / power / fractional power and the
    dominant ``(k_a, k_b)`` pair per non-period unit.
    """
    axis_a, axis_b = period_axes
    # Step 1: project along the first period axis.
    step1 = project_onto_fourier_basis(matrix, basis, period_axis=axis_a)
    # Step 2: project each first-axis component along the second period axis.
    # ``cos_coeffs`` and ``sin_coeffs`` have axis_a's period replaced by
    # frequency; axis_b is still period-shaped at its original position.
    step2_cos = project_onto_fourier_basis(step1.cos_coeffs, basis, period_axis=axis_b)
    step2_sin = project_onto_fourier_basis(step1.sin_coeffs, basis, period_axis=axis_b)

    cc = step2_cos.cos_coeffs.astype(np.float64)
    cs = step2_cos.sin_coeffs.astype(np.float64)
    sc = step2_sin.cos_coeffs.astype(np.float64)
    ss = step2_sin.sin_coeffs.astype(np.float64)

    power = cc**2 + cs**2 + sc**2 + ss**2
    magnitudes = np.sqrt(power)

    # Fractional power sums to 1 over the joint (k_a, k_b) plane per unit.
    total = power.sum(axis=period_axes, keepdims=True)
    fractional_power = np.where(total > 0, power / np.maximum(total, 1e-12), np.zeros_like(power))

    # Dominant (k_a, k_b) per non-period unit. argmax over the joint plane
    # by flattening period_axes; recover the pair via unravel_index.
    flat_axis_size = magnitudes.shape[axis_a] * magnitudes.shape[axis_b]
    moved = np.moveaxis(magnitudes, (axis_a, axis_b), (-2, -1))
    flat = moved.reshape(*moved.shape[:-2], flat_axis_size)
    flat_idx = np.argmax(flat, axis=-1)
    k_a_idx, k_b_idx = np.unravel_index(flat_idx, (basis.n_frequencies, basis.n_frequencies))
    dominant_freq_pair = np.stack(
        [basis.frequencies[k_a_idx], basis.frequencies[k_b_idx]],
        axis=-1,
    ).astype(np.int32)

    return {
        "cos_cos_coeffs": cc,
        "cos_sin_coeffs": cs,
        "sin_cos_coeffs": sc,
        "sin_sin_coeffs": ss,
        "magnitudes": magnitudes,
        "power": power,
        "fractional_power": fractional_power,
        "dominant_frequency_pair": dominant_freq_pair,
        "frequencies": basis.frequencies.astype(np.int32),
    }
