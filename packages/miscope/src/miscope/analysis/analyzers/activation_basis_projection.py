"""REQ_126: Activation basis-projection analyzer.

Universal activation-side basis-projection analyzer that consumes the
family's declared :class:`BasisProjectionSite` set and projects each
composed activation grid onto the family-supplied basis. The transform
step uses only REQ_109's ``project_onto_fourier_basis`` primitive.

Per the REQ_126 Q4 direction, this analyzer reads activations directly
from the hook cache (``ModelInput(needs_cache=True)``) rather than going
through an intermediate ``activation_snapshot`` artifact — hooks are the
long-term primary access path because larger models with smaller probes
may not be able to materialize a full activation snapshot.

Absorbed analyzers (target reproducibility on canon):
    - ``attention_freq`` ← site ``attn_pattern`` (joint diagonal + axis marginals)
    - ``neuron_freq_norm`` (a.k.a. ``neuron_freq_clusters``) ← site
      ``mlp_out`` (joint diagonal + axis marginals)
    - ``coarseness`` (REQ_102 gate) ← site ``mlp_out`` low-frequency
      energy ratio

For each 2D activation site, the analyzer emits both the 2D joint
projection (4 coefficient cubes, joint magnitudes/power/fractional
power, dominant frequency pair) **and** the per-axis 1D marginals
(project the mean-over-the-other-axis along each period axis). The
marginals are required for parity with the legacy "per-frequency
energy" aggregation, which mixed 2D diagonal energy with per-axis
marginal energy at the same frequency index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library.fourier_basis import (
    PeriodicFourierBasis,
    get_fourier_basis,
    project_onto_fourier_basis,
)
from miscope.analysis.registry import register_analyzer
from miscope.analysis.spec import AnalyzerSpec
from miscope.core.basis_projection import BasisProjectionSite

# Spec hooks: union across the modadd families' activation sites.
# Architectures that don't publish a hook simply don't populate the
# cache for it; the analyzer skips those sites at runtime.
_KNOWN_HOOKS: tuple[str, ...] = (
    "blocks.0.attn.hook_pattern",
    "blocks.0.mlp.hook_out",
)

SPEC = AnalyzerSpec(
    name="activation_basis_projection",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
    required_hooks=_KNOWN_HOOKS,
)


@register_analyzer(SPEC)
class ActivationBasisProjectionAnalyzer:
    """Project family-declared activation sites onto the family's basis."""

    name = "activation_basis_projection"
    description = "Project family-declared activation sites onto a family-supplied basis"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Project each declared site for the current epoch."""
        if inputs.cache is None:
            return {}

        sites: tuple[BasisProjectionSite, ...] = context.get(
            "activation_basis_projection_sites", ()
        )
        if not sites:
            return {}

        prime = int(context["params"]["prime"])
        basis = get_fourier_basis(prime)

        result: dict[str, np.ndarray] = {}
        for site in sites:
            if not _hooks_available(site, inputs.cache):
                continue
            matrix = site.compose(inputs.cache, context)
            site_result = _project_site(matrix, basis, site.period_axes)
            for key, value in site_result.items():
                result[f"{site.name}_{key}"] = value
        return result


def _hooks_available(site: BasisProjectionSite, cache: Any) -> bool:
    """Skip sites whose declared hooks aren't published by this architecture."""
    return all(hook in cache for hook in site.required_hooks)


def _project_site(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Project a composed activation matrix along one or two period axes."""
    if len(period_axes) == 1:
        return _project_1d(matrix, basis, period_axes[0])
    if len(period_axes) == 2:
        return _project_2d_with_marginals(matrix, basis, period_axes)
    raise ValueError(f"activation_basis_projection supports 1 or 2 period axes; got {period_axes}")


def _project_1d(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axis: int,
) -> dict[str, np.ndarray]:
    """1D projection — same shape as the weight-side analyzer's 1D output."""
    result = project_onto_fourier_basis(matrix, basis, period_axis=period_axis)
    return {
        "cos_coeffs": result.cos_coeffs.astype(np.float64),
        "sin_coeffs": result.sin_coeffs.astype(np.float64),
        "magnitudes": result.magnitudes.astype(np.float64),
        "phases": result.phases.astype(np.float64),
        "power": result.power.astype(np.float64),
        "fractional_power": result.fractional_power.astype(np.float64),
        "dominant_frequency": result.dominant_frequency.astype(np.int32),
        "frequencies": basis.frequencies.astype(np.int32),
    }


def _project_2d_with_marginals(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, int],
) -> dict[str, np.ndarray]:
    """2D projection with axis-marginal 1D projections.

    Marginals are 1D Fourier of the matrix averaged over one of the two
    period axes — they capture per-axis frequency content (e.g., ``mean
    over a, then Fourier on b``). Required for parity with the legacy
    "per-frequency energy" aggregation, which mixed 2D diagonal energy
    with per-axis marginal energy at the same frequency index.
    """
    axis_a, axis_b = period_axes
    # 2D joint projection: project axis_a, then axis_b of each component.
    step1 = project_onto_fourier_basis(matrix, basis, period_axis=axis_a)
    step2_cos = project_onto_fourier_basis(step1.cos_coeffs, basis, period_axis=axis_b)
    step2_sin = project_onto_fourier_basis(step1.sin_coeffs, basis, period_axis=axis_b)

    cc = step2_cos.cos_coeffs.astype(np.float64)
    cs = step2_cos.sin_coeffs.astype(np.float64)
    sc = step2_sin.cos_coeffs.astype(np.float64)
    ss = step2_sin.sin_coeffs.astype(np.float64)

    power = cc**2 + cs**2 + sc**2 + ss**2
    magnitudes = np.sqrt(power)

    total = power.sum(axis=period_axes, keepdims=True)
    fractional_power = np.where(total > 0, power / np.maximum(total, 1e-12), np.zeros_like(power))

    flat_axis_size = magnitudes.shape[axis_a] * magnitudes.shape[axis_b]
    moved = np.moveaxis(magnitudes, (axis_a, axis_b), (-2, -1))
    flat = moved.reshape(*moved.shape[:-2], flat_axis_size)
    flat_idx = np.argmax(flat, axis=-1)
    k_a_idx, k_b_idx = np.unravel_index(flat_idx, (basis.n_frequencies, basis.n_frequencies))
    dominant_freq_pair = np.stack(
        [basis.frequencies[k_a_idx], basis.frequencies[k_b_idx]],
        axis=-1,
    ).astype(np.int32)

    # Axis marginals: project the mean-over-the-other-axis along each period axis.
    marginal_a = matrix.mean(axis=axis_a)
    # After averaging over axis_a, axis_b's original position shifts down by 1
    # if axis_a < axis_b (which it is — period_axes is ordered (axis_a, axis_b)
    # with axis_a < axis_b by construction in our family declarations).
    marginal_a_axis_b = axis_b - 1 if axis_a < axis_b else axis_b
    marginal_a_result = project_onto_fourier_basis(marginal_a, basis, period_axis=marginal_a_axis_b)

    marginal_b = matrix.mean(axis=axis_b)
    marginal_b_result = project_onto_fourier_basis(marginal_b, basis, period_axis=axis_a)

    return {
        "cos_cos_coeffs": cc,
        "cos_sin_coeffs": cs,
        "sin_cos_coeffs": sc,
        "sin_sin_coeffs": ss,
        "magnitudes": magnitudes,
        "power": power,
        "fractional_power": fractional_power,
        "dominant_frequency_pair": dominant_freq_pair,
        # Axis marginals: 1D Fourier of mean-over-the-other-axis.
        # ``axis_a_marginal_*``: signal varies along axis a, axis b averaged out.
        # ``axis_b_marginal_*``: signal varies along axis b, axis a averaged out.
        "axis_b_marginal_cos_coeffs": marginal_a_result.cos_coeffs.astype(np.float64),
        "axis_b_marginal_sin_coeffs": marginal_a_result.sin_coeffs.astype(np.float64),
        "axis_b_marginal_power": marginal_a_result.power.astype(np.float64),
        "axis_a_marginal_cos_coeffs": marginal_b_result.cos_coeffs.astype(np.float64),
        "axis_a_marginal_sin_coeffs": marginal_b_result.sin_coeffs.astype(np.float64),
        "axis_a_marginal_power": marginal_b_result.power.astype(np.float64),
        "frequencies": basis.frequencies.astype(np.int32),
    }
