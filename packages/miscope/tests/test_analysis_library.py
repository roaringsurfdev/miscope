"""Tests for analysis library functions (REQ_021b)."""

import pytest
import torch

from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    get_dominant_frequency_indices,
    get_embedding_weights,
    get_fourier_basis,
    project_onto_fourier_basis,
    reshape_to_grid,
)

# --- Fourier Library Tests ---


class TestGetFourierBasis:
    """Tests for get_fourier_basis function."""

    def test_output_shape(self):
        """Test that basis has correct shape."""
        p = 113
        basis, names = get_fourier_basis(p)

        # Formula: 1 (constant) + 2 * (p // 2) (sin/cos pairs)
        # For p=113: 1 + 2*56 = 113
        expected_rows = 1 + 2 * (p // 2)
        assert basis.shape == (expected_rows, p)

    def test_basis_names_count(self):
        """Test correct number of basis names."""
        p = 113
        basis, names = get_fourier_basis(p)

        assert len(names) == basis.shape[0]
        assert names[0] == "Constant"
        assert names[1] == "sin k=1"
        assert names[2] == "cos k=1"

    def test_basis_is_normalized(self):
        """Test that each basis vector has unit norm."""
        p = 17
        basis, _ = get_fourier_basis(p)

        norms = basis.norm(dim=-1)
        expected = torch.ones(basis.shape[0])
        torch.testing.assert_close(norms, expected, rtol=1e-5, atol=1e-5)

    def test_device_placement(self):
        """Test basis can be placed on specified device."""
        p = 11
        basis, _ = get_fourier_basis(p, device="cpu")
        assert basis.device == torch.device("cpu")

    def test_small_prime(self):
        """Test with small prime."""
        p = 5
        basis, names = get_fourier_basis(p)

        # p=5: constant + 2 sin/cos pairs = 5 components
        assert basis.shape[0] == 5
        assert len(names) == 5


class TestProjectOntoFourierBasis:
    """Tests for project_onto_fourier_basis function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        p = 11
        d_model = 64
        basis, _ = get_fourier_basis(p)
        weights = torch.randn(p, d_model)

        coefficients = project_onto_fourier_basis(weights, basis)
        assert coefficients.shape == (basis.shape[0],)

    def test_coefficients_are_nonnegative(self):
        """Test that coefficient norms are non-negative."""
        p = 7
        basis, _ = get_fourier_basis(p)
        weights = torch.randn(p, 32)

        coefficients = project_onto_fourier_basis(weights, basis)
        assert (coefficients >= 0).all()


class TestCompute2DFourierTransform:
    """Tests for compute_2d_fourier_transform function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        p = 11
        n_neurons = 8
        basis, _ = get_fourier_basis(p)
        activations = torch.randn(n_neurons, p, p)

        result = compute_2d_fourier_transform(activations, basis)
        n_components = basis.shape[0]
        assert result.shape == (n_neurons, n_components, n_components)

    def test_batch_dimension_preserved(self):
        """Test that batch dimension is preserved."""
        p = 7
        batch_size = 16
        basis, _ = get_fourier_basis(p)
        activations = torch.randn(batch_size, p, p)

        result = compute_2d_fourier_transform(activations, basis)
        assert result.shape[0] == batch_size


class TestGetDominantFrequencyIndices:
    """Tests for get_dominant_frequency_indices function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        coefficients = torch.tensor([0.5, 2.0, 0.3, 1.5])
        result = get_dominant_frequency_indices(coefficients, threshold=1.0)
        assert isinstance(result, list)

    def test_threshold_filtering(self):
        """Test that threshold correctly filters indices."""
        coefficients = torch.tensor([0.5, 2.0, 0.3, 1.5])
        result = get_dominant_frequency_indices(coefficients, threshold=1.0)
        # Only indices 1 and 3 have values > 1.0
        assert set(result) == {1, 3}

    def test_no_dominant(self):
        """Test when no frequencies are dominant."""
        coefficients = torch.tensor([0.1, 0.2, 0.3])
        result = get_dominant_frequency_indices(coefficients, threshold=1.0)
        assert result == []


class TestComputeFrequencyVarianceFractions:
    """Tests for compute_frequency_variance_fractions function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        p = 11
        n_neurons = 8
        n_components = p + 1
        fourier_acts = torch.randn(n_neurons, n_components, n_components)

        result = compute_frequency_variance_fractions(fourier_acts, p)
        n_frequencies = p // 2
        assert result.shape == (n_frequencies, n_neurons)

    def test_values_are_nonnegative(self):
        """Test that variance fractions are non-negative."""
        p = 7
        n_neurons = 4
        n_components = p + 1
        fourier_acts = torch.randn(n_neurons, n_components, n_components)

        result = compute_frequency_variance_fractions(fourier_acts, p)
        assert (result >= 0).all()


# --- Activations Library Tests ---


class TestReshapeToGrid:
    """Tests for reshape_to_grid function."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        p = 11
        d = 32
        activations = torch.randn(p * p, d)

        result = reshape_to_grid(activations, p)
        assert result.shape == (d, p, p)

    def test_values_preserved(self):
        """Test that values are preserved during reshape."""
        p = 5
        d = 3
        activations = torch.arange(p * p * d, dtype=torch.float).reshape(p * p, d)

        result = reshape_to_grid(activations, p)

        # Check first element
        assert result[0, 0, 0] == activations[0, 0]


class TestComputeGridSizeFromDataset:
    """Tests for compute_grid_size_from_dataset function."""

    def test_correct_grid_size(self):
        """Test correct grid size is computed."""
        p = 11
        dataset = torch.zeros(p * p, 3)

        result = compute_grid_size_from_dataset(dataset)
        assert result == p

    def test_raises_for_non_square(self):
        """Test error for non-square dataset size."""
        dataset = torch.zeros(99, 3)  # 99 is not a perfect square

        with pytest.raises(ValueError):
            compute_grid_size_from_dataset(dataset)


class TestGetEmbeddingWeights:
    """Tests for get_embedding_weights function."""

    def test_excludes_special_tokens(self):
        """Test that special tokens are excluded."""
        from miscope.architectures import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            d_model=32,
            d_head=8,
            n_heads=2,
            n_layers=1,
            d_vocab=10,
            n_ctx=3,
            act_fn="relu",
        )
        model = HookedTransformer(cfg)

        # Full embedding should have 10 rows
        assert model.embed.W_E.shape[0] == 10

        # Excluding 1 token should give 9 rows
        W_E = get_embedding_weights(model, exclude_special_tokens=1)
        assert W_E.shape[0] == 9

    def test_exclude_zero(self):
        """Test with no tokens excluded."""
        from miscope.architectures import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            d_model=32,
            d_head=8,
            n_heads=2,
            n_layers=1,
            d_vocab=10,
            n_ctx=3,
            act_fn="relu",
        )
        model = HookedTransformer(cfg)

        W_E = get_embedding_weights(model, exclude_special_tokens=0)
        assert W_E.shape[0] == 10


# --- AnalyzerRegistry Tests ---


class TestAnalyzerRegistry:
    """Tests for AnalyzerRegistry."""

    def test_default_analyzers_registered(self):
        """Test that default analyzers are registered on import."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        assert AnalyzerRegistry.is_registered("dominant_frequencies")
        assert AnalyzerRegistry.is_registered("neuron_activations")
        assert AnalyzerRegistry.is_registered("neuron_freq_norm")

    def test_get_analyzer(self):
        """Test getting an analyzer by name."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        analyzer = AnalyzerRegistry.get("dominant_frequencies")
        assert analyzer.name == "dominant_frequencies"

    def test_get_unknown_analyzer_raises(self):
        """Test error when getting unknown analyzer."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        with pytest.raises(KeyError):
            AnalyzerRegistry.get("nonexistent_analyzer")

    def test_list_all(self):
        """Test listing all registered analyzers."""
        from miscope.analysis.analyzers import AnalyzerRegistry

        all_names = AnalyzerRegistry.list_all()
        assert "dominant_frequencies" in all_names
        assert "neuron_activations" in all_names
        assert "neuron_freq_norm" in all_names


# --- Integration Tests ---


class TestLibraryIntegration:
    """Integration tests for library functions working together."""

    def test_fourier_analysis_pipeline(self):
        """Test Fourier analysis workflow."""
        p = 11
        d_model = 32

        # Generate basis
        basis, names = get_fourier_basis(p)

        # Create fake embedding weights
        weights = torch.randn(p, d_model)

        # Project onto basis
        coefficients = project_onto_fourier_basis(weights, basis)

        # Find dominant frequencies
        dominant = get_dominant_frequency_indices(coefficients, threshold=0.5)

        # Should return valid indices
        assert all(0 <= idx < len(coefficients) for idx in dominant)

    def test_activation_analysis_pipeline(self):
        """Test activation analysis workflow."""
        p = 7
        d_mlp = 16

        # Create fake activations (batch = p^2)
        activations = torch.randn(p * p, d_mlp)

        # Reshape to grid
        grid = reshape_to_grid(activations, p)
        assert grid.shape == (d_mlp, p, p)

        # Apply Fourier transform
        basis, _ = get_fourier_basis(p)
        fourier_acts = compute_2d_fourier_transform(grid, basis)

        # Compute variance fractions
        fractions = compute_frequency_variance_fractions(fourier_acts, p)
        assert fractions.shape == (p // 2, d_mlp)
