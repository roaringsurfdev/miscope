"""Tests for the REQ_130 redefinition of FourierFrequencyQualityAnalyzer.

The metric is re-pointed off dominant_frequencies onto neuron_grouping and
redefined as neuron-weighted task coverage: weight each Fourier frequency by
relative neuron occupancy, then R² of the ideal mod-p tensor onto that weighted
subspace. With binary weights it reduces to the legacy hard-subspace R².
"""

from __future__ import annotations

import numpy as np
import pytest
from _deps_fakes import deps_inputs

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.analyzers.fourier_frequency_quality import (
    FourierFrequencyQualityAnalyzer,
    _frequency_weights_to_basis_rows,
    _weighted_quality_score,
)
from miscope.analysis.library.fourier import get_fourier_basis

PRIME = 7  # small odd prime; K = (p-1)//2 = 3 frequencies


def _ctx(p: int = PRIME) -> dict:
    basis, _ = get_fourier_basis(p)
    return {"params": {"prime": p}, "fourier_basis": basis}


def _grouping(n_per_group: list[int], basis_name: str = "fourier_w_in") -> dict:
    return {
        "n_per_group": np.asarray(n_per_group, dtype=np.int64),
        "feature_basis_name": np.array(basis_name, dtype="U64"),
    }


def _run(n_per_group: list[int], basis_name: str = "fourier_w_in") -> dict:
    analyzer = FourierFrequencyQualityAnalyzer()
    inputs = deps_inputs(
        epoch_artifacts={"neuron_grouping": _grouping(n_per_group, basis_name)}, epoch=0
    )
    return analyzer.analyze(inputs, _ctx())


# --- Registration ---


def test_registration_depends_on_neuron_grouping():
    spec = AnalyzerRegistry.get_spec("fourier_frequency_quality")
    declared = {i.analyzer_name for i in spec.inputs if hasattr(i, "analyzer_name")}
    assert "neuron_grouping" in declared
    assert "dominant_frequencies" not in declared


# --- Clean-limit reduction: equal occupancy → weighted == hard ---


def test_clean_limit_weighted_equals_hard():
    """Equal nonzero occupancy → all weights saturate to 1 → quality == coverage_hard."""
    result = _run([10, 0, 10])  # frequencies 1 and 3 used, equal counts
    assert result["quality_score"] == pytest.approx(result["coverage_hard"], abs=1e-6)


def test_clean_limit_matches_independent_hard_subspace():
    """The clean-limit score equals an independent hard-subspace R² over {1,3}."""
    result = _run([10, 0, 10])
    basis, _ = get_fourier_basis(PRIME)
    basis_np = basis.cpu().numpy()
    # Frequencies 1 and 3 → basis rows {1,2} and {5,6}.
    w_rows = np.zeros(PRIME)
    w_rows[[1, 2, 5, 6]] = 1.0
    expected = _weighted_quality_score(PRIME, basis_np, w_rows)
    assert result["quality_score"] == pytest.approx(expected, abs=1e-6)


# --- Monotonicity in occupancy ---


def test_occupancy_monotonic():
    """Adding a frequency and increasing its occupancy never decreases quality."""
    only_one = _run([10, 0, 0])["quality_score"]
    partial = _run([10, 0, 5])["quality_score"]  # freq 3 at half weight
    full = _run([10, 0, 10])["quality_score"]  # freq 3 at full weight
    assert only_one <= partial <= full
    assert full > only_one  # adding a populated frequency adds projected energy


# --- Guard: non-frequency-indexed grouping ---


def test_guard_rejects_universal_kmeans_grouping():
    """The metric is undefined for arbitrary kmeans clusters (not frequencies)."""
    with pytest.raises(ValueError, match="frequency-indexed"):
        _run([10, 5, 3], basis_name="weight_signature")


# --- Output contract & ranges ---


def test_output_keys_and_ranges():
    result = _run([8, 0, 4])
    assert set(result) == {
        "quality_score",
        "coverage_hard",
        "active_frequencies",
        "k",
        "reconstruction_error",
    }
    assert 0.0 <= float(result["quality_score"]) <= 1.0
    assert 0.0 <= float(result["coverage_hard"]) <= 1.0
    assert float(result["reconstruction_error"]) == pytest.approx(
        1.0 - float(result["quality_score"]), abs=1e-6
    )
    np.testing.assert_array_equal(result["active_frequencies"], np.array([1, 3], dtype=np.int32))
    assert int(result["k"]) == 2


def test_empty_grouping_yields_zero_quality():
    """No populated frequency → quality 0, no active frequencies."""
    result = _run([0, 0, 0])
    assert float(result["quality_score"]) == 0.0
    assert int(result["k"]) == 0
    assert result["active_frequencies"].size == 0


# --- Basis-row mapping helper ---


def test_frequency_weights_to_basis_rows_mapping():
    """Frequency g+1 maps to sin/cos rows {2g+1, 2g+2}; constant row 0 stays 0."""
    w_rows = _frequency_weights_to_basis_rows(np.array([1.0, 0.0, 0.5]), PRIME)
    expected = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5])
    np.testing.assert_allclose(w_rows, expected)
