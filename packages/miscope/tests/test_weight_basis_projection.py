"""REQ_126: WeightBasisProjectionAnalyzer tests.

Two layers:

1. Unit tests on synthetic data verify shape/axis handling for 1D and 2D
   sites and the absence of ad-hoc basis math (only REQ_109 primitives are
   composed).
2. Parity tests against canon ``p113/s999/ds598`` at epoch 24999 verify that
   each of the four absorbed legacy analyzers' outputs are reproducible from
   the new analyzer's per-site outputs:

       - ``dominant_frequencies``       ← site ``embedding``
       - ``attention_fourier`` (V band) ← site ``attn_v``
       - ``attention_fourier`` (QK 2D)  ← site ``attn_qk``
       - ``neuron_fourier`` (theta)     ← site ``mlp_in``
       - ``neuron_fourier`` (xi)        ← site ``mlp_out``
       - ``fourier_nucleation`` (it=0)  ← site ``mlp_in``

Parity is reconstructive, not byte-identical — the new analyzer outputs
canonical per-site projection results; the legacy analyzers' specific
aggregations (e.g. ``||coeffs||`` summed across d_model, normalized
per-frequency fractions) are reproduced by post-processing the new outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _deps_fakes import deps_inputs

from miscope.analysis.analyzers.weight_basis_projection import (
    WeightBasisProjectionAnalyzer,
)

CANON_VARIANT_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "modulo_addition_1layer"
    / "variants"
    / "p113_seed999_dseed598"
)
PARITY_EPOCH = 24999
PRIME = 113

# Parity tolerance accommodates the precision mismatch: the legacy analyzers
# accumulate in float32 (torch default) while the new analyzer uses float64
# intermediates (REQ_109 ``project_onto_fourier_basis`` casts on entry). The
# largest discrepancies appear at the highest frequencies where the basis
# vectors oscillate fast enough that float32 dot-product accumulation loses
# the last few bits relative to float64.
PARITY_RTOL = 1e-3
PARITY_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Unit tests on synthetic snapshots
# ---------------------------------------------------------------------------


def _synthetic_snapshot(p: int, d_model: int, d_mlp: int, n_heads: int, d_head: int) -> dict:
    """Build a minimal synthetic parameter_snapshot dict for the transformer family."""
    rng = np.random.default_rng(0)
    return {
        "W_E": rng.standard_normal((p + 1, d_model)).astype(np.float32),
        "W_U": rng.standard_normal((d_model, p)).astype(np.float32),
        "W_Q": rng.standard_normal((n_heads, d_model, d_head)).astype(np.float32),
        "W_K": rng.standard_normal((n_heads, d_model, d_head)).astype(np.float32),
        "W_V": rng.standard_normal((n_heads, d_model, d_head)).astype(np.float32),
        "W_in": rng.standard_normal((d_model, d_mlp)).astype(np.float32),
        "W_out": rng.standard_normal((d_mlp, d_model)).astype(np.float32),
    }


def _synthetic_context(p: int, sites) -> dict:
    return {"params": {"prime": p}, "weight_basis_projection_sites": sites}


def _run_analyzer(snapshot: dict, sites, p: int) -> dict[str, np.ndarray]:
    analyzer = WeightBasisProjectionAnalyzer()
    inputs = deps_inputs({"parameter_snapshot": snapshot})
    return analyzer.analyze(inputs, _synthetic_context(p, sites))


def test_empty_sites_returns_empty():
    """No sites declared → analyzer is a no-op."""
    snapshot = _synthetic_snapshot(p=13, d_model=8, d_mlp=16, n_heads=2, d_head=4)
    result = _run_analyzer(snapshot, sites=(), p=13)
    assert result == {}


def test_1d_site_shapes_and_keys():
    """1D site emits 8 keys with frequency axis at the period-axis position."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = tuple(s for s in fam.weight_basis_projection_sites if s.name == "embedding")
    p, d_model = 13, 8
    snapshot = _synthetic_snapshot(p=p, d_model=d_model, d_mlp=16, n_heads=2, d_head=4)
    result = _run_analyzer(snapshot, sites=sites, p=p)

    n_freq = (p - 1) // 2
    expected_keys = {
        f"embedding_{k}"
        for k in (
            "cos_coeffs",
            "sin_coeffs",
            "magnitudes",
            "phases",
            "power",
            "fractional_power",
            "dominant_frequency",
            "frequencies",
        )
    }
    assert set(result.keys()) == expected_keys
    assert result["embedding_cos_coeffs"].shape == (n_freq, d_model)
    assert result["embedding_sin_coeffs"].shape == (n_freq, d_model)
    assert result["embedding_dominant_frequency"].shape == (d_model,)
    np.testing.assert_array_equal(result["embedding_frequencies"], np.arange(1, n_freq + 1))


def test_2d_site_shapes_and_keys():
    """2D site emits four coefficient cubes plus joint metrics."""
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    sites = tuple(s for s in fam.weight_basis_projection_sites if s.name == "attn_qk")
    p, n_heads, d_head = 13, 2, 4
    snapshot = _synthetic_snapshot(p=p, d_model=8, d_mlp=16, n_heads=n_heads, d_head=d_head)
    result = _run_analyzer(snapshot, sites=sites, p=p)

    n_freq = (p - 1) // 2
    for piece in ("cos_cos", "cos_sin", "sin_cos", "sin_sin"):
        assert result[f"attn_qk_{piece}_coeffs"].shape == (n_heads, n_freq, n_freq)
    assert result["attn_qk_magnitudes"].shape == (n_heads, n_freq, n_freq)
    assert result["attn_qk_dominant_frequency_pair"].shape == (n_heads, 2)
    # Fractional power normalizes over the joint frequency plane per head.
    np.testing.assert_allclose(
        result["attn_qk_fractional_power"].sum(axis=(1, 2)),
        np.ones(n_heads),
        atol=1e-10,
    )


def test_analyzer_imports_only_req109_basis_primitives():
    """REQ_126 audit: the analyzer module references only REQ_109 basis primitives.

    Guards against future drift where someone slips an ``np.fft`` or
    ``compute_2d_fourier_transform`` call into ``analyze()``.
    """
    src_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "miscope"
        / "analysis"
        / "analyzers"
        / "weight_basis_projection.py"
    )
    src = src_path.read_text()
    assert "np.fft" not in src
    assert "compute_2d_fourier_transform" not in src
    assert "compute_frequency_variance_fractions" not in src
    # Only REQ_109's project primitive should appear, and only via import:
    assert "from miscope.analysis.library.fourier_basis import" in src


# ---------------------------------------------------------------------------
# Parity tests against canon artifacts
# ---------------------------------------------------------------------------


def _canon_available() -> bool:
    snap = CANON_VARIANT_DIR / "artifacts" / "parameter_snapshot" / f"epoch_{PARITY_EPOCH:05d}.npz"
    return snap.is_file()


skip_no_canon = pytest.mark.skipif(
    not _canon_available(), reason="Canon variant data not present on this machine"
)


def _load_npz(name: str) -> dict[str, np.ndarray]:
    path = CANON_VARIANT_DIR / "artifacts" / name / f"epoch_{PARITY_EPOCH:05d}.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def canon_snapshot():
    if not _canon_available():
        pytest.skip("Canon data not available")
    return _load_npz("parameter_snapshot")


@pytest.fixture(scope="module")
def canon_new_result(canon_snapshot):
    from miscope.families.implementations.modulo_addition_1layer import (
        ModuloAddition1LayerFamily,
    )

    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    return _run_analyzer(canon_snapshot, sites=fam.weight_basis_projection_sites, p=PRIME)


# NOTE (REQ_102): the parity tests for the ``embedding`` (dominant_frequencies),
# ``mlp_in``/``mlp_out`` (neuron_fourier), and ``attn_v``/``attn_qk``
# (attention_fourier) sites were removed when those legacy analyzers were retired
# (REQ_126 recorded their absorption parity before retirement). Their on-disk
# artifacts no longer exist, so the tests could never run again. The
# ``fourier_nucleation`` parity test below is retained — that analyzer survives.


@skip_no_canon
def test_parity_fourier_nucleation_one_shot(canon_snapshot, canon_new_result):
    """``fourier_nucleation`` iteration 0 reproduced from the ``mlp_in`` site.

    Iteration 0's ``aggregate_energy`` is the per-frequency total power across
    neurons, then max-normalized. ``neuron_peak_freq`` is per-neuron argmax.
    """
    legacy = _load_npz("fourier_nucleation")
    legacy_agg = legacy["aggregate_energy"][0]  # (n_freqs,)
    legacy_peak = legacy["neuron_peak_freq"][0]  # (d_mlp,)

    # new ``mlp_in_power`` is (K, d_mlp); aggregate over neurons → (K,), then max-normalize.
    new_power = canon_new_result["mlp_in_power"]  # (K, d_mlp)
    new_agg_raw = new_power.sum(axis=-1)
    new_agg = new_agg_raw / new_agg_raw.max()
    new_peak = canon_new_result["mlp_in_dominant_frequency"]  # (d_mlp,)

    np.testing.assert_allclose(new_agg, legacy_agg, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    np.testing.assert_array_equal(new_peak, legacy_peak)
