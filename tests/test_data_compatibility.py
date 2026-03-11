"""Tests for REQ_064: Fourier Data Compatibility Analyzer."""

import numpy as np
import pytest
import torch

from miscope.analysis.data_compatibility import compute_data_compatibility

# ---------------------------------------------------------------------------
# Unit: output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_returns_all_keys(self):
        result = compute_data_compatibility(prime=11, data_seed=42)
        expected_keys = {
            "frequencies",
            "condition_number",
            "condition_score",
            "phase_uniformity",
            "compatibility_score",
            "prime",
            "data_seed",
            "training_fraction",
            "n_training_pairs",
        }
        assert expected_keys == set(result.keys())

    def test_shapes(self):
        prime = 13
        result = compute_data_compatibility(prime=prime, data_seed=7)
        n_freqs = prime // 2
        assert result["frequencies"].shape == (n_freqs,)
        assert result["condition_number"].shape == (n_freqs,)
        assert result["condition_score"].shape == (n_freqs,)
        assert result["phase_uniformity"].shape == (n_freqs,)
        assert result["compatibility_score"].shape == (n_freqs,)

    def test_scalar_fields(self):
        prime = 11
        data_seed = 99
        result = compute_data_compatibility(prime=prime, data_seed=data_seed)
        assert result["prime"] == prime
        assert result["data_seed"] == data_seed
        assert result["training_fraction"] == pytest.approx(0.3)
        assert result["n_training_pairs"] == int(prime * prime * 0.3)

    def test_frequencies_values(self):
        prime = 17
        result = compute_data_compatibility(prime=prime, data_seed=1)
        expected = np.arange(1, prime // 2 + 1, dtype=np.int32)
        np.testing.assert_array_equal(result["frequencies"], expected)


# ---------------------------------------------------------------------------
# Unit: score range and properties
# ---------------------------------------------------------------------------


class TestScoreRanges:
    def test_compatibility_score_in_unit_interval(self):
        result = compute_data_compatibility(prime=11, data_seed=42)
        cs = result["compatibility_score"]
        assert cs.min() >= 0.0
        assert cs.max() <= 1.0

    def test_condition_score_in_unit_interval(self):
        result = compute_data_compatibility(prime=11, data_seed=42)
        cs = result["condition_score"]
        assert cs.min() >= 0.0
        assert cs.max() <= 1.0

    def test_phase_uniformity_in_unit_interval(self):
        result = compute_data_compatibility(prime=11, data_seed=42)
        pu = result["phase_uniformity"]
        assert pu.min() >= 0.0
        assert pu.max() <= 1.0

    def test_condition_number_ge_one(self):
        """Condition number of a PSD matrix is always ≥ 1."""
        result = compute_data_compatibility(prime=11, data_seed=42)
        assert (result["condition_number"] >= 1.0 - 1e-6).all()

    def test_full_grid_has_near_unit_condition_number(self):
        """When training on the full p×p grid, the Gram matrix is perfectly balanced.

        With all p² pairs present, the cos and sin components of every
        frequency are equally represented, so the condition number ≈ 1.
        """
        result = compute_data_compatibility(prime=11, data_seed=0, training_fraction=1.0)
        np.testing.assert_allclose(result["condition_number"], 1.0, atol=0.1)

    def test_full_grid_has_near_unit_compatibility(self):
        """Full grid should have near-perfect compatibility at all frequencies."""
        result = compute_data_compatibility(prime=11, data_seed=0, training_fraction=1.0)
        assert result["compatibility_score"].min() > 0.9


# ---------------------------------------------------------------------------
# Unit: splitting matches training pipeline exactly
# ---------------------------------------------------------------------------


class TestSplittingReproducibility:
    def test_matches_training_pipeline_split(self):
        """The reconstructed training s-values must match what the training pipeline produces.

        The training pipeline does:
            torch.manual_seed(data_seed)
            indices = torch.randperm(p * p)
            train_indices = indices[:int(p * p * training_fraction)]

        We verify the compatibility module reproduces the same s=(a+b)%p values.
        """
        prime = 11
        data_seed = 42
        training_fraction = 0.3

        # Ground truth: run the exact pipeline logic
        torch.manual_seed(data_seed)
        all_indices = torch.randperm(prime * prime)
        n_train = int(prime * prime * training_fraction)
        train_indices = all_indices[:n_train].numpy()
        a_vals = train_indices // prime
        b_vals = train_indices % prime
        expected_s = (a_vals + b_vals) % prime

        # Run the module
        result = compute_data_compatibility(prime=prime, data_seed=data_seed,
                                            training_fraction=training_fraction)
        assert result["n_training_pairs"] == n_train
        # The s-values are used internally; verify indirectly via n_training_pairs
        # and a direct comparison via the internal helper
        from miscope.analysis.data_compatibility import _reconstruct_training_sums
        actual_s = _reconstruct_training_sums(prime, data_seed, training_fraction)
        np.testing.assert_array_equal(actual_s, expected_s)

    def test_different_seeds_produce_different_splits(self):
        prime = 11
        r1 = compute_data_compatibility(prime=prime, data_seed=1)
        r2 = compute_data_compatibility(prime=prime, data_seed=2)
        # Not all frequencies are guaranteed to differ, but most will
        assert not np.allclose(r1["condition_number"], r2["condition_number"])

    def test_same_seed_is_deterministic(self):
        prime = 13
        r1 = compute_data_compatibility(prime=prime, data_seed=7)
        r2 = compute_data_compatibility(prime=prime, data_seed=7)
        np.testing.assert_array_equal(r1["condition_number"], r2["condition_number"])
        np.testing.assert_array_equal(r1["compatibility_score"], r2["compatibility_score"])


# ---------------------------------------------------------------------------
# Integration: views render without errors
# ---------------------------------------------------------------------------


class TestViewsRender:
    @pytest.fixture
    def compat_data(self):
        return compute_data_compatibility(prime=11, data_seed=42)

    def test_spectrum_renders(self, compat_data):
        from miscope.visualization.renderers.data_compatibility import (
            render_data_compatibility_spectrum,
        )

        fig = render_data_compatibility_spectrum(compat_data, epoch=None)
        assert fig is not None
        assert len(fig.data) > 0

    def test_overlap_renders_without_nucleation(self, compat_data):
        """Overlap view should work gracefully when nucleation artifact is absent."""
        from miscope.visualization.renderers.data_compatibility import (
            render_data_compatibility_overlap,
        )

        data = {"compatibility": compat_data, "nucleation": None}
        fig = render_data_compatibility_overlap(data, epoch=None)
        assert fig is not None
        assert len(fig.data) > 0

    def test_overlap_renders_with_nucleation(self, compat_data):
        """Overlap view should synthesize nucleation + compatibility when both present."""
        from miscope.visualization.renderers.data_compatibility import (
            render_data_compatibility_overlap,
        )

        prime = int(compat_data["prime"])
        n_freqs = prime // 2
        nucleation = {
            "aggregate_energy": np.random.rand(5, n_freqs).astype(np.float32),
            "frequencies": np.arange(1, n_freqs + 1, dtype=np.int32),
        }
        data = {"compatibility": compat_data, "nucleation": nucleation}
        fig = render_data_compatibility_overlap(data, epoch=None)
        assert fig is not None
        # Should have bars + compatibility line + overlap line
        assert len(fig.data) >= 3
