"""ActivationFrequencyNormAnalyzer tests (downsized from activation_basis_projection).

Three layers:

1. Unit tests on a synthetic activation cache verify the narrowed output
   contract (only ``{site}_freq_norm`` + ``frequencies``), shapes, and the
   REQ_109-only-primitive audit.
2. Parity tests against canon ``p113/s999/ds598`` at epoch 24999 verify that the
   persisted ``{site}_freq_norm`` reproduces the retired ``neuron_freq_norm`` /
   ``attention_freq`` artifacts directly (no consumer-side reconstruction). These
   are the same parity bars the old analyzer met via ``reconstruct_neuron_freq_norm``;
   the reduction now happens inside the analyzer.
3. The REQ_102 gate test: ``coarseness`` is the sum of the first three frequency
   rows of ``mlp_out_freq_norm``.

Parity tolerances are loose by design (rtol=1e-3, atol=1e-5) per the
float64-vs-float32 precision shift documented in [feedback_req126_float64_parity].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers.activation_frequency_norm import (
    ActivationFrequencyNormAnalyzer,
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
    checkpoint = (
        CANON_VARIANT_DIR / "checkpoints" / f"checkpoint_epoch_{PARITY_EPOCH:05d}.safetensors"
    )
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
    attn = rng.uniform(0.0, 1.0, size=(p * p, n_heads, 3, 3)).astype(np.float32)
    mlp_out = rng.standard_normal((p * p, 3, d_mlp)).astype(np.float32)
    return {
        "blocks.0.attn.hook_pattern": torch.from_numpy(attn),
        "blocks.0.mlp.hook_out": torch.from_numpy(mlp_out),
    }


def _ctx(p: int, sites) -> dict:
    return {"params": {"prime": p}, "activation_frequency_norm_sites": sites}


def _run(cache: dict, sites, p: int) -> dict[str, np.ndarray]:
    analyzer = ActivationFrequencyNormAnalyzer()
    return analyzer.analyze(ResolvedInputs(cache=cache), _ctx(p, sites))  # pyright: ignore[reportArgumentType]


def test_empty_sites_returns_empty():
    p = 13
    cache = _synthetic_cache(p)
    assert _run(cache, sites=(), p=p) == {}


def test_narrowed_output_contract_and_shapes():
    """2D site emits only {site}_freq_norm (n_freq, n_units) + frequencies — no cubes."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = tuple(s for s in fam.activation_frequency_norm_sites if s.name == "mlp_out")
    p, d_mlp = 13, 16
    cache = _synthetic_cache(p, d_mlp=d_mlp)
    result = _run(cache, sites=sites, p=p)

    K = (p - 1) // 2
    assert set(result) == {"mlp_out_freq_norm", "frequencies"}
    assert result["mlp_out_freq_norm"].shape == (K, d_mlp)
    assert result["frequencies"].shape == (K,)
    # No heavy joint cubes or marginals survive.
    assert not any("coeffs" in k or "power" in k or "marginal" in k for k in result)


def test_skips_sites_with_missing_hooks():
    """A site whose required hook is absent from the cache is silently skipped."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = fam.activation_frequency_norm_sites
    p = 13
    cache = _synthetic_cache(p)
    cache.pop("blocks.0.attn.hook_pattern")
    result = _run(cache, sites=sites, p=p)
    assert any(k.startswith("mlp_out_") for k in result)
    assert not any(k.startswith("attn_pattern_") for k in result)


def test_analyzer_imports_only_req109_basis_primitives():
    """The activation analyzer composes only REQ_109 primitives."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "miscope"
        / "analysis"
        / "analyzers"
        / "activation_frequency_norm.py"
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
def canon_result(canon_run):
    analyzer = ActivationFrequencyNormAnalyzer()
    inputs = ResolvedInputs(cache=canon_run["cache"])
    return analyzer.analyze(inputs, canon_run["context"])


def _load_legacy_npz(name: str) -> dict[str, np.ndarray]:
    path = CANON_VARIANT_DIR / "artifacts" / name / f"epoch_{PARITY_EPOCH:05d}.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@skip_no_canon
def test_parity_neuron_freq_norm(canon_run, canon_result):
    """Persisted mlp_out_freq_norm reproduces legacy neuron_freq_norm.norm_matrix."""
    legacy = _load_legacy_npz("neuron_freq_norm")["norm_matrix"]  # (K, d_mlp)
    np.testing.assert_allclose(
        canon_result["mlp_out_freq_norm"], legacy, rtol=PARITY_RTOL, atol=PARITY_ATOL
    )


@skip_no_canon
def test_parity_attention_freq(canon_run, canon_result):
    """Persisted attn_pattern_freq_norm reproduces legacy attention_freq.freq_matrix."""
    legacy = _load_legacy_npz("attention_freq")["freq_matrix"]  # (K, n_heads)
    np.testing.assert_allclose(
        canon_result["attn_pattern_freq_norm"], legacy, rtol=PARITY_RTOL, atol=PARITY_ATOL
    )


@skip_no_canon
def test_req102_coarseness_recoverable(canon_run, canon_result):
    """coarseness[n] is the sum of the first n_low_freqs=3 rows of mlp_out_freq_norm."""
    coarseness_path = (
        CANON_VARIANT_DIR / "artifacts" / "coarseness" / f"epoch_{PARITY_EPOCH:05d}.npz"
    )
    if not coarseness_path.is_file():
        pytest.skip("Legacy coarseness artifact absent on canon.")
    legacy_coarseness = _load_legacy_npz("coarseness")["coarseness"]  # (d_mlp,)
    reconstructed = canon_result["mlp_out_freq_norm"][:3].sum(axis=0)
    np.testing.assert_allclose(reconstructed, legacy_coarseness, rtol=PARITY_RTOL, atol=PARITY_ATOL)

    blob_threshold = 0.7
    np.testing.assert_array_equal(
        reconstructed >= blob_threshold, legacy_coarseness >= blob_threshold
    )
