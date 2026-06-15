"""Activation per-frequency energy-norm analyzer.

Universal activation-side analyzer that consumes the family's declared
:class:`BasisProjectionSite` set, projects each composed activation grid onto
the family-supplied Fourier basis, and persists a single reduced field per site:
``{site}_freq_norm`` — the ``(n_freq, n_units)`` per-(unit, frequency) fraction
of non-DC Fourier variance (prime-scaled). It reads activations directly from
the hook cache (``ModelInput(needs_cache=True)``).

History (downsize): this analyzer was ``activation_basis_projection``, which
persisted seven ``(n_units, K, K)`` coefficient/power cubes plus two marginal
arrays per site. Every consumer — three downstream analyzers and the activation
Fourier views — collapsed those to the diagonal-plus-marginal per-frequency norm
and then argmaxed over frequency; nothing read the off-diagonal joint structure,
the raw coefficient cubes, ``magnitudes``, ``fractional_power``, or
``dominant_frequency_pair``. Under dense checkpointing those cubes were the
dominant on-disk cost (and were materialized a second time in the warehouse).
The analyzer now performs the K×K→K reduction in-memory and persists only the
``(n_freq, n_units)`` result — ~50× smaller per epoch per site.

Reduced fields it descends from:
    - ``neuron_freq_norm`` (a.k.a. ``neuron_freq_clusters``) ← site ``mlp_out``
    - ``attention_freq`` ← site ``attn_pattern``
    - ``coarseness`` (REQ_102 gate) ← low-frequency rows of the ``mlp_out`` norm

Re-add note: the raw per-axis marginal energy (``{site}_axis_a/b_marginal_power``,
each ``(n_units, K)``) is computed transiently here and discarded. If a future
analyzer needs it, re-declare those two fields on :data:`SPEC.outputs` and return
``marginal_*_power`` from :func:`_project_site` — they are small (``K`` per unit),
unlike the dropped ``K×K`` cubes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from miscope.analysis.inputs import ModelInput, ResolvedInputs
from miscope.analysis.library.basis_reconstruction import reduce_to_freq_norm
from miscope.analysis.library.fourier_basis import (
    PeriodicFourierBasis,
    get_fourier_basis,
    project_onto_fourier_basis,
)
from miscope.analysis.output_schema import OutputField as F
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
    name="activation_frequency_norm",
    output_scope="per_epoch",
    inputs=(ModelInput(needs_weights=False, needs_cache=True),),
    required_hooks=_KNOWN_HOOKS,
    version=1,  # renamed + narrowed from activation_basis_projection (was v2)
    outputs=(
        F.tensor(
            "freq_norm",
            "float64",
            ("variant", "epoch", "site"),
            "Per-frequency energy-fraction norm per unit (n_freq, n_units), prime-scaled.",
        ),
        F.columnar(
            "frequencies",
            "int32",
            ("variant", "epoch", "frequency"),
            "Frequency-index axis labels for the freq_norm rows.",
        ),
    ),
)


@register_analyzer(SPEC)
class ActivationFrequencyNormAnalyzer:
    """Project family-declared activation sites and persist the per-frequency norm."""

    name = "activation_frequency_norm"
    description = "Per-frequency activation energy-fraction norm per unit, per site"

    def analyze(
        self,
        inputs: ResolvedInputs,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Project each declared site for the current epoch and reduce to freq_norm."""
        if inputs.cache is None:
            return {}

        sites: tuple[BasisProjectionSite, ...] = context.get("activation_frequency_norm_sites", ())
        if not sites:
            return {}

        prime = int(context["params"]["prime"])
        basis = get_fourier_basis(prime)

        result: dict[str, np.ndarray] = {}
        for site in sites:
            if not _hooks_available(site, inputs.cache):
                continue
            matrix = site.compose(inputs.cache, context)
            site_result = _project_site(matrix, basis, site.period_axes, prime)
            for key, value in site_result.items():
                result[f"{site.name}_{key}"] = value
        result["frequencies"] = basis.frequencies.astype(np.int32)
        return result


def _hooks_available(site: BasisProjectionSite, cache: Any) -> bool:
    """Skip sites whose declared hooks aren't published by this architecture."""
    return all(hook in cache for hook in site.required_hooks)


def _project_site(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, ...],
    prime: int,
) -> dict[str, np.ndarray]:
    """Project a composed activation matrix and reduce to the per-frequency norm."""
    if len(period_axes) == 1:
        return _project_1d(matrix, basis, period_axes[0])
    if len(period_axes) == 2:
        return _project_2d(matrix, basis, period_axes, prime)
    raise ValueError(f"activation_frequency_norm supports 1 or 2 period axes; got {period_axes}")


def _project_1d(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axis: int,
) -> dict[str, np.ndarray]:
    """1D projection — freq_norm is per-unit power normalized across frequencies."""
    result = project_onto_fourier_basis(matrix, basis, period_axis=period_axis)
    # fractional_power is (n_units, K); transpose to legacy (n_freq, n_units) layout.
    return {"freq_norm": np.moveaxis(result.fractional_power.astype(np.float64), -1, 0)}


def _project_2d(
    matrix: np.ndarray,
    basis: PeriodicFourierBasis,
    period_axes: tuple[int, int],
    prime: int,
) -> dict[str, np.ndarray]:
    """2D projection reduced to the per-frequency norm.

    Computes the joint ``(n_units, K, K)`` power cube and the two per-axis
    marginal-power arrays transiently, then collapses them to the
    ``(n_freq, n_units)`` norm via the prime-scaled reduction. The heavy joint
    cube never leaves this function.
    """
    axis_a, axis_b = period_axes
    # 2D joint projection: project axis_a, then axis_b of each component.
    step1 = project_onto_fourier_basis(matrix, basis, period_axis=axis_a)
    step2_cos = project_onto_fourier_basis(step1.cos_coeffs, basis, period_axis=axis_b)
    step2_sin = project_onto_fourier_basis(step1.sin_coeffs, basis, period_axis=axis_b)

    cc = step2_cos.cos_coeffs
    cs = step2_cos.sin_coeffs
    sc = step2_sin.cos_coeffs
    ss = step2_sin.sin_coeffs
    joint_power = (cc**2 + cs**2 + sc**2 + ss**2).astype(np.float64)

    # Axis marginals: project the mean-over-the-other-axis along each period axis.
    # axis_a < axis_b by construction, so averaging over axis_a shifts axis_b down 1.
    marginal_a = matrix.mean(axis=axis_a)
    marginal_a_axis_b = axis_b - 1 if axis_a < axis_b else axis_b
    marginal_a_power = project_onto_fourier_basis(
        marginal_a, basis, period_axis=marginal_a_axis_b
    ).power.astype(np.float64)

    marginal_b = matrix.mean(axis=axis_b)
    marginal_b_power = project_onto_fourier_basis(
        marginal_b, basis, period_axis=axis_a
    ).power.astype(np.float64)

    # marginal_a varies along axis b (axis_a averaged out) and vice versa; the
    # reduction is symmetric in the two marginals, so the labelling is immaterial.
    freq_norm = reduce_to_freq_norm(joint_power, marginal_b_power, marginal_a_power, prime)
    return {"freq_norm": freq_norm}
