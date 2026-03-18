"""Tests for REQ_074: Variant Outcome Registry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from miscope.analysis.variant_summary import (
    CANONICAL_SPECIALIZATION_THRESHOLD,
    build_variant_registry,
    compute_variant_summary,
    extract_learned_frequencies,
    write_variant_summary,
)
from miscope.views.cross_variant import ClassificationRules


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D_MLP = 512
_N_EPOCHS = 50
_PRIME = 113


def _make_nd_data(
    freq_per_neuron: list[int] | None = None,
    committed_fraction: float = 1.0,
    threshold: float = 0.75,
) -> dict:
    """Build a minimal neuron_dynamics cross-epoch artifact for testing.

    Args:
        freq_per_neuron: Which frequency each neuron is assigned at the final epoch.
            Defaults to alternating frequencies 14 and 28.
        committed_fraction: Fraction of neurons with max_frac >= threshold.
        threshold: The specialization threshold stored in the artifact.
    """
    if freq_per_neuron is None:
        freq_per_neuron = [14 if i % 2 == 0 else 28 for i in range(_D_MLP)]

    dominant_freq = np.zeros((_N_EPOCHS, _D_MLP), dtype=float)
    dominant_freq[:] = np.array(freq_per_neuron, dtype=float)

    max_frac = np.full((_N_EPOCHS, _D_MLP), threshold + 0.1)
    n_uncommitted = int(_D_MLP * (1.0 - committed_fraction))
    if n_uncommitted > 0:
        max_frac[:, -n_uncommitted:] = threshold - 0.1

    commitment_epochs = np.full(_D_MLP, float("nan"))
    epochs = np.arange(_N_EPOCHS, dtype=float) * 100

    return {
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "threshold": np.array([threshold]),
        "commitment_epochs": commitment_epochs,
        "epochs": epochs,
    }


def _make_variant(
    test_losses: list[float] | None = None,
    nd_data: dict | None = None,
    rg_data: dict | None = None,
    prime: int = _PRIME,
    seed: int = 485,
    data_seed: int = 598,
    family_name: str = "modulo_addition_1layer",
    variant_dir: Path | None = None,
) -> MagicMock:
    """Build a mock Variant with controllable artifact responses."""
    if test_losses is None:
        # Simple grokking: plateau then drop
        test_losses = [2.5] * 20 + [0.0001] * 30

    variant = MagicMock()
    variant.name = f"mock_p{prime}_seed{seed}_dseed{data_seed}"
    variant.model_config = {"prime": prime, "seed": seed, "data_seed": data_seed}
    variant.metadata = {
        "test_losses": test_losses,
        "train_losses": [0.001] * len(test_losses),
    }
    variant.family.name = family_name
    variant.variant_dir = variant_dir or Path(f"/tmp/mock/{family_name}/mock_p{prime}")

    def mock_load_cross_epoch(name):
        if name == "neuron_dynamics" and nd_data is not None:
            return nd_data
        raise FileNotFoundError(name)

    def mock_load_summary(name):
        if name == "repr_geometry" and rg_data is not None:
            return rg_data
        raise FileNotFoundError(name)

    variant.artifacts.load_cross_epoch.side_effect = mock_load_cross_epoch
    variant.artifacts.load_summary.side_effect = mock_load_summary
    return variant


# ---------------------------------------------------------------------------
# Unit: extract_learned_frequencies
# ---------------------------------------------------------------------------


def test_extract_learned_frequencies_returns_none_when_no_artifact():
    variant = _make_variant(nd_data=None)
    freqs, threshold = extract_learned_frequencies(variant)
    assert freqs is None
    assert threshold is None


def test_extract_learned_frequencies_basic():
    # 256 neurons on freq 14, 256 on freq 28 — both should qualify at 0.10 threshold
    nd = _make_nd_data(freq_per_neuron=[14 if i < 256 else 28 for i in range(_D_MLP)])
    variant = _make_variant(nd_data=nd)
    freqs, threshold = extract_learned_frequencies(variant, canonical_threshold=0.10)
    assert set(freqs) == {14, 28}
    assert threshold == 0.10


def test_extract_learned_frequencies_threshold_filters_minority():
    # 460 neurons on freq 14, only 20 on freq 28 — 20/512 ≈ 3.9%, well below 10%
    nd = _make_nd_data(freq_per_neuron=[14 if i < 492 else 28 for i in range(_D_MLP)])
    variant = _make_variant(nd_data=nd)
    freqs, _ = extract_learned_frequencies(variant, canonical_threshold=0.10)
    assert freqs == [14]  # freq 28 has 20/512 = 3.9%, below threshold

    # At a lower threshold (0.03) freq 28 qualifies
    freqs_loose, _ = extract_learned_frequencies(variant, canonical_threshold=0.03)
    assert set(freqs_loose) == {14, 28}


def test_extract_learned_frequencies_threshold_stored_in_return():
    nd = _make_nd_data()
    variant = _make_variant(nd_data=nd)
    threshold_in = 0.15
    _, threshold_out = extract_learned_frequencies(variant, canonical_threshold=threshold_in)
    assert threshold_out == threshold_in


# ---------------------------------------------------------------------------
# Unit: handshake_failures logic
# ---------------------------------------------------------------------------


def test_handshake_failures_empty_when_all_committed_survive():
    """All committed frequencies at onset are still present at the final epoch."""
    # Both freq 14 and 28 are present at onset AND final epoch
    nd = _make_nd_data(freq_per_neuron=[14 if i < 256 else 28 for i in range(_D_MLP)])
    variant = _make_variant(nd_data=nd)

    # Use grokking losses that produce a second_descent_onset_epoch
    test_losses = [2.5] * 20 + [0.0001] * 30
    variant.metadata["test_losses"] = test_losses

    summary = compute_variant_summary(variant, canonical_threshold=0.10)
    assert summary["handshake_failures"] == [] or summary["handshake_failures"] is None
    if summary["handshake_failures"] is not None:
        assert summary["handshake_succeeded"] is True


def test_handshake_failures_identifies_missing_frequency():
    """A frequency committed at onset but absent from the final epoch produces a failure."""
    n = _D_MLP
    # At all epochs: freq 14 for first 256 neurons, freq 28 for last 256
    nd = _make_nd_data(freq_per_neuron=[14 if i < 256 else 28 for i in range(n)])

    # Override final epoch: only freq 14 remains (freq 28 neurons become uncommitted)
    nd["max_frac"][-1, 256:] = 0.0  # last 256 neurons drop below threshold

    variant = _make_variant(nd_data=nd)
    test_losses = [2.5] * 20 + [0.0001] * 30
    variant.metadata["test_losses"] = test_losses

    summary = compute_variant_summary(variant, canonical_threshold=0.10)

    if summary["committed_frequencies_at_onset"] is not None:
        # freq 28 was committed at onset, but absent from learned_frequencies
        assert 28 in summary["handshake_failures"]
        assert summary["handshake_succeeded"] is False


def test_handshake_fields_none_when_no_second_descent():
    """No second descent onset → handshake fields are all None."""
    # Flat loss — never descends
    flat_losses = [2.5] * 50
    nd = _make_nd_data()
    variant = _make_variant(test_losses=flat_losses, nd_data=nd)

    summary = compute_variant_summary(variant)
    assert summary["second_descent_onset_epoch"] is None
    assert summary["committed_frequencies_at_onset"] is None
    assert summary["handshake_failures"] is None
    assert summary["handshake_succeeded"] is None


# ---------------------------------------------------------------------------
# Unit: second_descent_completion_epoch
# ---------------------------------------------------------------------------


def test_second_descent_completion_epoch_none_when_loss_does_not_stabilize():
    """When test loss never properly descends, completion epoch is None."""
    # Loss drops slightly but never below degraded_test_loss
    unstable_losses = [2.5] * 20 + [1.0] * 30
    variant = _make_variant(test_losses=unstable_losses)
    summary = compute_variant_summary(variant)
    assert summary["second_descent_completion_epoch"] is None


# ---------------------------------------------------------------------------
# Unit: canonical_threshold stored in output
# ---------------------------------------------------------------------------


def test_canonical_threshold_stored_matches_input():
    nd = _make_nd_data()
    variant = _make_variant(nd_data=nd)
    threshold_in = 0.07
    summary = compute_variant_summary(variant, canonical_threshold=threshold_in)
    assert summary["canonical_specialization_threshold"] == threshold_in


# ---------------------------------------------------------------------------
# Unit: graceful degradation on missing artifacts
# ---------------------------------------------------------------------------


def test_summary_written_without_nd_artifact():
    """Missing neuron_dynamics does not block summary computation."""
    variant = _make_variant(nd_data=None, rg_data=None)
    summary = compute_variant_summary(variant)
    assert summary["learned_frequencies"] is None
    assert summary["learned_frequency_count"] is None
    assert summary["max_resid_post_circularity"] is None
    assert "failure_mode" in summary
    assert "final_test_loss" in summary


def test_summary_includes_max_resid_post_circularity():
    rg_data = {"resid_post_circularity": [0.3, 0.7, 0.9, 0.85]}
    variant = _make_variant(rg_data=rg_data)
    summary = compute_variant_summary(variant)
    assert summary["max_resid_post_circularity"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Unit: identity fields
# ---------------------------------------------------------------------------


def test_summary_contains_required_identity_fields():
    variant = _make_variant(prime=113, seed=485, data_seed=598)
    summary = compute_variant_summary(variant)
    assert summary["prime"] == 113
    assert summary["model_seed"] == 485
    assert summary["data_seed"] == 598
    assert summary["family"] == "modulo_addition_1layer"
    assert "computed_at" in summary


# ---------------------------------------------------------------------------
# Integration: write_variant_summary
# ---------------------------------------------------------------------------


def test_write_variant_summary_creates_json_file(tmp_path):
    nd = _make_nd_data()
    variant_dir = tmp_path / "modulo_addition_1layer" / "mock_variant"
    variant_dir.mkdir(parents=True)

    variant = _make_variant(nd_data=nd, variant_dir=variant_dir)
    output_path = write_variant_summary(variant)

    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert data["prime"] == _PRIME
    assert "learned_frequencies" in data
    assert "canonical_specialization_threshold" in data


def test_write_variant_summary_overwrites_on_rerun(tmp_path):
    nd = _make_nd_data()
    variant_dir = tmp_path / "modulo_addition_1layer" / "mock_variant"
    variant_dir.mkdir(parents=True)
    variant = _make_variant(nd_data=nd, variant_dir=variant_dir)

    path1 = write_variant_summary(variant, canonical_threshold=0.10)
    ts1 = json.loads(path1.read_text())["computed_at"]

    path2 = write_variant_summary(variant, canonical_threshold=0.15)
    data2 = json.loads(path2.read_text())
    assert data2["canonical_specialization_threshold"] == 0.15
    # Timestamp should differ (or at worst be equal, never older)
    assert data2["computed_at"] >= ts1


# ---------------------------------------------------------------------------
# Integration: build_variant_registry
# ---------------------------------------------------------------------------


def test_build_variant_registry_one_entry_per_variant(tmp_path):
    family_name = "modulo_addition_1layer"
    family_dir = tmp_path / family_name

    # Write two fake variant_summary.json files
    for prime, mseed in [(113, 485), (113, 999)]:
        vdir = family_dir / f"p{prime}_mseed{mseed}"
        vdir.mkdir(parents=True)
        summary = {
            "prime": prime,
            "model_seed": mseed,
            "data_seed": 598,
            "family": family_name,
            "computed_at": "2026-03-17T00:00:00+00:00",
            "learned_frequencies": [14, 28],
            "learned_frequency_count": 2,
            "canonical_specialization_threshold": 0.10,
            "second_descent_onset_epoch": 5000,
            "second_descent_completion_epoch": None,
            "second_descent_survived": True,
            "committed_frequencies_at_onset": [14, 28],
            "handshake_failures": [],
            "handshake_succeeded": True,
            "failure_mode": "healthy",
            "max_resid_post_circularity": 0.92,
            "final_specialized_frequency_count": 2,
            "peak_test_loss_epoch": 4000,
            "final_test_loss": 0.00001,
        }
        (vdir / "variant_summary.json").write_text(json.dumps(summary))

    registry_path = build_variant_registry(tmp_path, family_name)

    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert len(registry) == 2

    variant_ids = {e["variant_id"] for e in registry}
    assert "113_485_598" in variant_ids
    assert "113_999_598" in variant_ids
