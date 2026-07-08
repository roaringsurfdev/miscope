"""REQ_152 / REQ_154: CircuitSpectraAnalyzer tests.

Two layers:

1. Composition correctness — the family's ``full_ov`` and ``full_qk`` composers
   reproduce independent hand-rolled references per head.
2. Spectral-instrument correctness on synthetic snapshots — the analyzer's
   ``operator_norm`` / ``effective_rank`` / ``eigenvalues`` match independent
   reference computations, and ``copying_score`` is bounded in ``[0, 1]``.

No prior numbers exist for these objects (they are new), so the oracle is an
independent reference computation, not a parity baseline.
"""

from __future__ import annotations

import numpy as np
from _deps_fakes import deps_inputs

from miscope.analysis.analyzers.circuit_spectra import (
    CircuitSpectraAnalyzer,
    _circuit_spectra,
    _copying_score,
)
from miscope.analysis.library.weights import compute_participation_ratio
from miscope.families.implementations.modulo_addition_1layer import (
    ModuloAddition1LayerFamily,
    _compose_attn_qk,
    _compose_direct_path,
    _compose_full_ov,
    _compose_ov,
    _compose_qk,
)

P = 13
N_HEADS = 4
D_MODEL = 16
D_HEAD = 4


def _synthetic_snapshot() -> dict[str, np.ndarray]:
    """Minimal parameter_snapshot with W_U wider than p (exercises the [:, :p] slice)."""
    rng = np.random.default_rng(0)
    return {
        "W_E": rng.standard_normal((P + 1, D_MODEL)).astype(np.float32),
        "W_U": rng.standard_normal((D_MODEL, P + 1)).astype(np.float32),
        "W_Q": rng.standard_normal((N_HEADS, D_MODEL, D_HEAD)).astype(np.float32),
        "W_K": rng.standard_normal((N_HEADS, D_MODEL, D_HEAD)).astype(np.float32),
        "W_V": rng.standard_normal((N_HEADS, D_MODEL, D_HEAD)).astype(np.float32),
        "W_O": rng.standard_normal((N_HEADS, D_HEAD, D_MODEL)).astype(np.float32),
    }


def _context() -> dict:
    fam = ModuloAddition1LayerFamily.__new__(ModuloAddition1LayerFamily)
    return {"params": {"prime": P}, "circuit_spectra_sites": fam.circuit_spectra_sites}


# ---------------------------------------------------------------------------
# 1. Composition correctness
# ---------------------------------------------------------------------------


def test_compose_full_ov_matches_reference():
    """The OV composer equals an independent per-head W_E W_V W_O W_U reference."""
    snap = _synthetic_snapshot()
    matrix = _compose_full_ov(snap, {"params": {"prime": P}})
    assert matrix.shape == (N_HEADS, P, P)
    W_E, W_U = snap["W_E"][:P], snap["W_U"][:, :P]
    for h in range(N_HEADS):
        reference = W_E @ snap["W_V"][h] @ snap["W_O"][h] @ W_U
        np.testing.assert_allclose(matrix[h], reference, rtol=1e-5, atol=1e-5)


def test_compose_full_qk_matches_reference():
    """The QK composer equals an independent per-head (W_E W_Q)(W_E W_K)^T reference."""
    snap = _synthetic_snapshot()
    matrix = _compose_attn_qk(snap, {"params": {"prime": P}})
    assert matrix.shape == (N_HEADS, P, P)
    W_E = snap["W_E"][:P]
    for h in range(N_HEADS):
        reference = (W_E @ snap["W_Q"][h]) @ (W_E @ snap["W_K"][h]).T
        np.testing.assert_allclose(matrix[h], reference, rtol=1e-5, atol=1e-5)


def test_compose_residual_and_direct_path_references():
    """Residual OV/QK and the head-less direct path equal independent references."""
    snap = _synthetic_snapshot()
    ov = _compose_ov(snap, {})
    qk = _compose_qk(snap, {})
    dp = _compose_direct_path(snap, {"params": {"prime": P}})
    assert ov.shape == (N_HEADS, D_MODEL, D_MODEL)
    assert qk.shape == (N_HEADS, D_MODEL, D_MODEL)
    assert dp.shape == (P, P)
    for h in range(N_HEADS):
        np.testing.assert_allclose(ov[h], snap["W_V"][h] @ snap["W_O"][h], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(qk[h], snap["W_Q"][h] @ snap["W_K"][h].T, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dp, snap["W_E"][:P] @ snap["W_U"][:, :P], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Spectral-instrument correctness
# ---------------------------------------------------------------------------


def test_empty_sites_returns_empty():
    """No declared circuit sites → analyzer is a no-op."""
    analyzer = CircuitSpectraAnalyzer()
    inputs = deps_inputs({"parameter_snapshot": _synthetic_snapshot()})
    assert analyzer.analyze(inputs, {"params": {"prime": P}}) == {}


def test_analyzer_keys_and_shapes():
    """All five circuit sites emit site-prefixed keys; columnar per head, tensors full shape."""
    analyzer = CircuitSpectraAnalyzer()
    inputs = deps_inputs({"parameter_snapshot": _synthetic_snapshot()})
    result = analyzer.analyze(inputs, _context())

    fields = ("copying_score", "effective_rank", "operator_norm", "circuit_matrix", "eigenvalues")
    sites = ("full_ov", "full_qk", "ov", "qk", "direct_path")
    assert set(result) == {f"{site}_{f}" for site in sites for f in fields}

    # (n_heads, matrix_dim); direct_path is head-less → uniform-rank singleton head.
    expected = {
        "full_ov": (N_HEADS, P),
        "full_qk": (N_HEADS, P),
        "ov": (N_HEADS, D_MODEL),
        "qk": (N_HEADS, D_MODEL),
        "direct_path": (1, P),
    }
    for site, (n_heads, dim) in expected.items():
        assert result[f"{site}_copying_score"].shape == (n_heads,)
        assert result[f"{site}_effective_rank"].shape == (n_heads,)
        assert result[f"{site}_operator_norm"].shape == (n_heads,)
        assert result[f"{site}_circuit_matrix"].shape == (n_heads, dim, dim)
        assert result[f"{site}_eigenvalues"].shape == (n_heads, dim)
        assert result[f"{site}_eigenvalues"].dtype == np.complex128


def test_spectral_invariants_match_reference():
    """operator_norm / effective_rank / eigenvalues equal independent numpy refs."""
    matrix = _compose_full_ov(_synthetic_snapshot(), {"params": {"prime": P}})
    out = _circuit_spectra(matrix)
    for h in range(N_HEADS):
        s = np.linalg.svd(matrix[h], compute_uv=False)
        np.testing.assert_allclose(out["operator_norm"][h], s[0], rtol=1e-6)
        np.testing.assert_allclose(
            out["effective_rank"][h], compute_participation_ratio(s), rtol=1e-6
        )
        ref_eig = np.sort_complex(np.linalg.eigvals(matrix[h]))
        np.testing.assert_allclose(np.sort_complex(out["eigenvalues"][h]), ref_eig, rtol=1e-6)


def test_copying_score_bounded():
    """copying_score ∈ [0, 1] for every head."""
    out = _circuit_spectra(_compose_full_ov(_synthetic_snapshot(), {"params": {"prime": P}}))
    assert np.all(out["copying_score"] >= 0.0)
    assert np.all(out["copying_score"] <= 1.0)


def test_copying_score_extremes():
    """A positive-definite circuit scores 1.0; a sign-flip (negative) scores 0.0."""
    identity = np.eye(5)[np.newaxis]  # all eigenvalues +1 → pure copy
    assert _circuit_spectra(identity)["copying_score"][0] == 1.0
    assert _copying_score(np.array([-1.0, -2.0, -3.0])) == 0.0


def test_headless_circuit_gets_singleton_head_axis():
    """A 2D (p, p) circuit is treated uniform-rank with one head (DirectPath shape)."""
    out = _circuit_spectra(np.eye(5))
    assert out["copying_score"].shape == (1,)
    assert out["circuit_matrix"].shape == (1, 5, 5)
