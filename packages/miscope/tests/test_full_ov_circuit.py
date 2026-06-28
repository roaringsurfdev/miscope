"""REQ_152: FullOVCircuitAnalyzer tests.

Two layers:

1. Composition correctness — the family's ``full_ov`` composer reproduces an
   independent hand-rolled ``W_E[:p] W_V[h] W_O[h] W_U[:,:p]`` per head.
2. Spectral-instrument correctness on synthetic snapshots — the analyzer's
   ``operator_norm`` / ``effective_rank`` / ``eigenvalues`` match independent
   reference computations, and ``copying_score`` is bounded in ``[0, 1]``.

No prior numbers exist for this object (it is new), so the oracle is an
independent reference computation, not a parity baseline.
"""

from __future__ import annotations

import numpy as np
from _deps_fakes import deps_inputs

from miscope.analysis.analyzers.full_ov_circuit import (
    FullOVCircuitAnalyzer,
    _circuit_spectra,
    _copying_score,
)
from miscope.analysis.library.weights import compute_participation_ratio
from miscope.families.implementations.modulo_addition_1layer import (
    ModuloAddition1LayerFamily,
    _compose_full_ov,
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
    """The composer equals an independent per-head W_E W_V W_O W_U reference."""
    snap = _synthetic_snapshot()
    matrix = _compose_full_ov(snap, {"params": {"prime": P}})
    assert matrix.shape == (N_HEADS, P, P)

    W_E, W_U = snap["W_E"][:P], snap["W_U"][:, :P]
    for h in range(N_HEADS):
        reference = W_E @ snap["W_V"][h] @ snap["W_O"][h] @ W_U
        np.testing.assert_allclose(matrix[h], reference, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Spectral-instrument correctness
# ---------------------------------------------------------------------------


def test_empty_sites_returns_empty():
    """No declared circuit sites → analyzer is a no-op."""
    analyzer = FullOVCircuitAnalyzer()
    inputs = deps_inputs({"parameter_snapshot": _synthetic_snapshot()})
    assert analyzer.analyze(inputs, {"params": {"prime": P}}) == {}


def test_analyzer_keys_and_shapes():
    """Site-prefixed keys; columnar fields per head, tensors carry full shape."""
    analyzer = FullOVCircuitAnalyzer()
    inputs = deps_inputs({"parameter_snapshot": _synthetic_snapshot()})
    result = analyzer.analyze(inputs, _context())

    expected = {
        f"full_ov_{k}"
        for k in (
            "copying_score",
            "effective_rank",
            "operator_norm",
            "circuit_matrix",
            "eigenvalues",
        )
    }
    assert set(result) == expected
    assert result["full_ov_copying_score"].shape == (N_HEADS,)
    assert result["full_ov_effective_rank"].shape == (N_HEADS,)
    assert result["full_ov_operator_norm"].shape == (N_HEADS,)
    assert result["full_ov_circuit_matrix"].shape == (N_HEADS, P, P)
    assert result["full_ov_eigenvalues"].shape == (N_HEADS, P)
    assert result["full_ov_eigenvalues"].dtype == np.complex128


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
