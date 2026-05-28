"""REQ_126 PR 2: ActivationBasisProjectionAnalyzer tests.

Three layers:

1. Unit tests on a synthetic activation cube verify shape/axis handling
   for 1D and 2D sites (with marginals) and the REQ_109-only-primitive
   audit.
2. Parity tests against canon ``p113/s999/ds598`` at epoch 24999 verify
   that ``attention_freq`` and ``neuron_freq_norm`` outputs are
   reproducible from the new analyzer's joint-2D and axis-marginal
   outputs. Parity tests load the canon checkpoint and run a forward
   pass — they are slow and gated on canon availability.
3. The REQ_102 gate test: verify that ``coarseness`` is reproducible as
   the sum of the first three frequencies of the reconstructed
   ``neuron_freq_norm``. Outcome recorded as evidence under REQ_102.

Per the float64-vs-float32 precision shift documented in
[feedback_req126_float64_parity], parity tolerances are loose by design
(rtol=1e-3, atol=1e-5). Discrepancies at the highest frequencies are
expected and are not regressions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers.activation_basis_projection import (
    ActivationBasisProjectionAnalyzer,
)
from miscope.analysis.inputs import ResolvedInputs

CANON_VARIANT_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "modulo_addition_1layer"
    / "variants"
    / "p113_seed999_dseed598"
)
PARITY_EPOCH = 24999
PRIME = 113
PARITY_RTOL = 1e-3
PARITY_ATOL = 1e-5


def _canon_available() -> bool:
    checkpoint = CANON_VARIANT_DIR / "checkpoints" / f"checkpoint_epoch_{PARITY_EPOCH:05d}.safetensors"
    legacy_neuron = (
        CANON_VARIANT_DIR / "artifacts" / "neuron_freq_norm" / f"epoch_{PARITY_EPOCH:05d}.npz"
    )
    return checkpoint.is_file() and legacy_neuron.is_file()


skip_no_canon = pytest.mark.skipif(
    not _canon_available(),
    reason="Canon variant checkpoint+artifacts not present on this machine",
)


# ---------------------------------------------------------------------------
# Unit tests on synthetic caches
# ---------------------------------------------------------------------------


def _synthetic_cache(p: int, n_heads: int = 4, d_mlp: int = 16, seed: int = 0) -> dict:
    """Build a minimal synthetic activation cache for the modadd transformer family."""
    rng = np.random.default_rng(seed)
    # Attention pattern: (p^2, n_heads, n_pos, n_pos) — we use n_pos=3 like canon.
    attn = rng.uniform(0.0, 1.0, size=(p * p, n_heads, 3, 3)).astype(np.float32)
    # MLP out: (p^2, seq_len, d_mlp). seq_len=3 for the transformer.
    mlp_out = rng.standard_normal((p * p, 3, d_mlp)).astype(np.float32)
    return {
        "blocks.0.attn.hook_pattern": torch.from_numpy(attn),
        "blocks.0.mlp.hook_out": torch.from_numpy(mlp_out),
    }


def _ctx(p: int, sites) -> dict:
    return {"params": {"prime": p}, "activation_basis_projection_sites": sites}


def _run(cache: dict, sites, p: int) -> dict[str, np.ndarray]:
    analyzer = ActivationBasisProjectionAnalyzer()
    return analyzer.analyze(ResolvedInputs(cache=cache), _ctx(p, sites))


def test_empty_sites_returns_empty():
    p = 13
    cache = _synthetic_cache(p)
    assert _run(cache, sites=(), p=p) == {}


def test_2d_shapes_keys_and_marginal_dimensions():
    """2D site emits joint outputs + per-axis marginals; check shapes."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = tuple(
        s for s in fam.activation_basis_projection_sites if s.name == "mlp_out"
    )
    p, d_mlp = 13, 16
    cache = _synthetic_cache(p, d_mlp=d_mlp)
    result = _run(cache, sites=sites, p=p)

    K = (p - 1) // 2
    # Joint outputs
    for piece in ("cos_cos", "cos_sin", "sin_cos", "sin_sin"):
        assert result[f"mlp_out_{piece}_coeffs"].shape == (d_mlp, K, K)
    assert result["mlp_out_power"].shape == (d_mlp, K, K)
    assert result["mlp_out_dominant_frequency_pair"].shape == (d_mlp, 2)
    # Marginals
    assert result["mlp_out_axis_a_marginal_power"].shape == (d_mlp, K)
    assert result["mlp_out_axis_b_marginal_power"].shape == (d_mlp, K)
    # Joint fractional power normalizes to 1 per neuron
    np.testing.assert_allclose(
        result["mlp_out_fractional_power"].sum(axis=(1, 2)),
        np.ones(d_mlp),
        atol=1e-10,
    )


def test_skips_sites_with_missing_hooks():
    """A site whose required hook is absent from the cache is silently skipped."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = fam.activation_basis_projection_sites
    p = 13
    cache = _synthetic_cache(p)
    # Remove the attention hook → only mlp_out site can run.
    cache.pop("blocks.0.attn.hook_pattern")
    result = _run(cache, sites=sites, p=p)
    assert any(k.startswith("mlp_out_") for k in result)
    assert not any(k.startswith("attn_pattern_") for k in result)


def test_analyzer_imports_only_req109_basis_primitives():
    """REQ_126 audit: the activation analyzer composes only REQ_109 primitives."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "miscope"
        / "analysis"
        / "analyzers"
        / "activation_basis_projection.py"
    )
    src = src_path.read_text()
    assert "np.fft" not in src
    assert "compute_2d_fourier_transform" not in src
    assert "compute_frequency_variance_fractions" not in src
    assert "from miscope.analysis.library.fourier_basis import" in src


# ---------------------------------------------------------------------------
# Parity fixtures (canon, slow — load checkpoint + forward pass)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canon_run():
    """Load canon model at PARITY_EPOCH, run probe, return cache + variant + family."""
    if not _canon_available():
        pytest.skip("Canon data not available")

    from miscope import load_family

    family = load_family("modulo_addition_1layer")
    variant = next(v for v in family.variants if v.name == "p113_seed999_dseed598")
    model = variant.load_model_at_checkpoint(PARITY_EPOCH)
    model.eval()
    probe = family.generate_analysis_dataset(variant.params)
    with torch.no_grad():
        _, cache = model.run_with_cache(probe)
    context = family.prepare_analysis_context(variant.params, device="cpu")
    return {"cache": cache, "context": context, "variant": variant, "family": family}


@pytest.fixture(scope="module")
def canon_new_result(canon_run):
    analyzer = ActivationBasisProjectionAnalyzer()
    inputs = ResolvedInputs(cache=canon_run["cache"])
    return analyzer.analyze(inputs, canon_run["context"])


def _load_legacy_npz(name: str) -> dict[str, np.ndarray]:
    path = CANON_VARIANT_DIR / "artifacts" / name / f"epoch_{PARITY_EPOCH:05d}.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _reconstruct_legacy_neuron_freq_norm(
    new_result: dict[str, np.ndarray],
    prime: int,
    site_prefix: str,
) -> np.ndarray:
    """Reproduce the legacy per-frequency variance fraction from new outputs.

    Legacy formula (sum of 8 cells in the (k+1)-th 3x3 cross of the
    2D Fourier transform, divided by total Fourier-space variance ≡
    total signal variance by Parseval):

        numerator_k[n] = power_joint[n, k, k]
                        + p * axis_a_marginal_power[n, k]
                        + p * axis_b_marginal_power[n, k]
        legacy_freq_norm[k, n] = numerator_k[n] / total_variance[n]

    Returns array shape (K, n_units) matching legacy norm_matrix.
    """
    joint_power = new_result[f"{site_prefix}_power"]  # (n_units, K, K)
    marginal_a_power = new_result[f"{site_prefix}_axis_a_marginal_power"]  # (n_units, K)
    marginal_b_power = new_result[f"{site_prefix}_axis_b_marginal_power"]  # (n_units, K)

    K = joint_power.shape[-1]
    diag_joint = joint_power[..., np.arange(K), np.arange(K)]  # (n_units, K)
    numerator = diag_joint + prime * (marginal_a_power + marginal_b_power)

    # total_variance per Parseval: joint + p*(marginal_a + marginal_b) + DC²
    # We don't materialize DC, but the joint sum + marginals reconstruct
    # the full non-DC Fourier energy. DC contributes the constant offset
    # (mean²·p²); the legacy zeroed DC before computing total variance
    # would conflict... wait, the legacy computes total_variance AFTER
    # zeroing DC. So:
    #     total_variance_legacy = ||legacy_fourier||² − DC²
    #                           = total non-DC Fourier energy
    #                           = joint_total + p*marginal_a_total + p*marginal_b_total
    total_non_dc = (
        joint_power.sum(axis=(-1, -2))
        + prime * marginal_a_power.sum(axis=-1)
        + prime * marginal_b_power.sum(axis=-1)
    )
    total_clipped = np.maximum(total_non_dc, 1e-10)
    reconstructed = numerator / total_clipped[..., None]  # (n_units, K)
    return reconstructed.T  # (K, n_units) to match legacy norm_matrix layout


@skip_no_canon
def test_parity_neuron_freq_norm(canon_run, canon_new_result):
    """Reconstruct ``neuron_freq_norm.norm_matrix`` from new mlp_out site."""
    legacy = _load_legacy_npz("neuron_freq_norm")["norm_matrix"]  # (K, d_mlp)
    reconstructed = _reconstruct_legacy_neuron_freq_norm(canon_new_result, PRIME, "mlp_out")
    np.testing.assert_allclose(reconstructed, legacy, rtol=PARITY_RTOL, atol=PARITY_ATOL)


@skip_no_canon
def test_parity_attention_freq(canon_run, canon_new_result):
    """Reconstruct ``attention_freq.freq_matrix`` from new attn_pattern site."""
    legacy = _load_legacy_npz("attention_freq")["freq_matrix"]  # (K, n_heads)
    reconstructed = _reconstruct_legacy_neuron_freq_norm(
        canon_new_result, PRIME, "attn_pattern"
    )  # (K, n_heads)
    np.testing.assert_allclose(reconstructed, legacy, rtol=PARITY_RTOL, atol=PARITY_ATOL)


# ---------------------------------------------------------------------------
# REQ_102 gate: coarseness reproducibility
# ---------------------------------------------------------------------------


@skip_no_canon
def test_req102_coarseness_recoverable_from_activation_basis_projection(
    canon_run, canon_new_result
):
    """REQ_102 gate condition.

    ``coarseness[n]`` is defined as the sum of the first ``n_low_freqs=3``
    legacy ``neuron_freq_norm`` rows. If the new analyzer reproduces
    ``neuron_freq_norm``, the coarseness signal (and thus the
    blob-vs-plaid classification) is preserved by transitivity. This
    test records that evidence under canonical conditions.
    """
    coarseness_path = (
        CANON_VARIANT_DIR / "artifacts" / "coarseness" / f"epoch_{PARITY_EPOCH:05d}.npz"
    )
    if not coarseness_path.is_file():
        pytest.skip(
            "Legacy coarseness artifact absent on canon — the REQ_102 "
            "blob-vs-plaid preservation gate needs canon refreshed with the "
            "coarseness analyzer to record this evidence."
        )
    legacy_coarseness = _load_legacy_npz("coarseness")["coarseness"]  # (d_mlp,)
    reconstructed_neuron_freq_norm = _reconstruct_legacy_neuron_freq_norm(
        canon_new_result, PRIME, "mlp_out"
    )  # (K, d_mlp)
    n_low_freqs = 3
    reconstructed_coarseness = reconstructed_neuron_freq_norm[:n_low_freqs].sum(axis=0)
    np.testing.assert_allclose(
        reconstructed_coarseness, legacy_coarseness, rtol=PARITY_RTOL, atol=PARITY_ATOL
    )

    # Blob classification (threshold 0.7) — confirm the boolean labelling
    # also matches. This is the practical signal coarseness is used for.
    blob_threshold = 0.7
    legacy_blob = legacy_coarseness >= blob_threshold
    reconstructed_blob = reconstructed_coarseness >= blob_threshold
    np.testing.assert_array_equal(reconstructed_blob, legacy_blob)
