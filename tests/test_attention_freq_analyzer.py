"""Tests for REQ_026: Attention Head Frequency Specialization Analyzer."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers.attention_freq import AttentionFreqAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library import (
    get_fourier_basis,
)
from miscope.analysis.protocols import ActivationContext

# ── Mock cache ──────────────────────────────────────────────────────────


class MockCache(dict):
    """Minimal mock for TransformerLens ActivationCache."""

    def __getitem__(self, key):
        return super().__getitem__(key)


# ── Analyzer Protocol ───────────────────────────────────────────────────


class TestAttentionFreqAnalyzerProtocol:
    """Tests that AttentionFreqAnalyzer conforms to the Analyzer protocol."""

    def test_has_name(self):
        analyzer = AttentionFreqAnalyzer()
        assert analyzer.name == "attention_freq"

    def test_has_description(self):
        analyzer = AttentionFreqAnalyzer()
        assert isinstance(analyzer.description, str)
        assert len(analyzer.description) > 0

    def test_has_analyze_method(self):
        analyzer = AttentionFreqAnalyzer()
        assert callable(analyzer.analyze)

    def test_has_summary_methods(self):
        analyzer = AttentionFreqAnalyzer()
        assert callable(analyzer.get_summary_keys)
        assert callable(analyzer.compute_summary)

    def test_conforms_to_analyzer_protocol(self):
        from miscope.analysis.protocols import Analyzer

        analyzer = AttentionFreqAnalyzer()
        assert isinstance(analyzer, Analyzer)


# ── Analyzer Output ─────────────────────────────────────────────────────


class TestAttentionFreqAnalyzerOutput:
    """Tests for analyze() output shape and values."""

    @pytest.fixture
    def p(self):
        return 7

    @pytest.fixture
    def n_heads(self):
        return 4

    @pytest.fixture
    def n_pos(self):
        return 3

    @pytest.fixture
    def analyzer_result(self, p, n_heads, n_pos):
        """Run analyzer on synthetic attention patterns."""
        np.random.seed(42)
        batch = p * p

        # Simulate softmax attention patterns
        raw = np.random.rand(batch, n_heads, n_pos, n_pos).astype(np.float32)
        # Normalize along from_pos dimension (softmax)
        attn = raw / raw.sum(axis=-1, keepdims=True)
        attn_tensor = torch.tensor(attn)

        cache = MockCache({("pattern", 0): attn_tensor})
        probe = torch.zeros(batch, 3)

        fourier_basis, _ = get_fourier_basis(p)
        context = {"fourier_basis": fourier_basis}

        analyzer = AttentionFreqAnalyzer()
        return analyzer.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(None, cache, None),
                probe=probe,
                analysis_params=context,
            )
        )

    def test_returns_dict(self, analyzer_result):
        assert isinstance(analyzer_result, dict)

    def test_has_freq_matrix_key(self, analyzer_result):
        assert "freq_matrix" in analyzer_result

    def test_output_is_numpy(self, analyzer_result):
        assert isinstance(analyzer_result["freq_matrix"], np.ndarray)

    def test_output_shape(self, analyzer_result, p, n_heads):
        n_freq = p // 2
        assert analyzer_result["freq_matrix"].shape == (n_freq, n_heads)

    def test_values_non_negative(self, analyzer_result):
        assert np.all(analyzer_result["freq_matrix"] >= 0)

    def test_values_at_most_one(self, analyzer_result):
        assert np.all(analyzer_result["freq_matrix"] <= 1.0 + 1e-6)

    def test_fractions_sum_at_most_one(self, analyzer_result):
        """Variance fractions should sum to <= 1 per head (DC removed)."""
        col_sums = analyzer_result["freq_matrix"].sum(axis=0)
        assert np.all(col_sums <= 1.0 + 1e-6)


# ── Summary Statistics ──────────────────────────────────────────────────


class TestAttentionFreqSummary:
    """Tests for get_summary_keys() and compute_summary()."""

    def test_summary_keys(self):
        analyzer = AttentionFreqAnalyzer()
        keys = analyzer.get_summary_keys()
        assert "dominant_freq_per_head" in keys
        assert "max_frac_per_head" in keys
        assert "mean_specialization" in keys

    def test_summary_keys_match_output(self):
        analyzer = AttentionFreqAnalyzer()
        keys = set(analyzer.get_summary_keys())
        result = {"freq_matrix": np.random.rand(5, 4).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert keys == set(summary.keys())

    def test_dominant_freq_shape(self):
        analyzer = AttentionFreqAnalyzer()
        n_freq, n_heads = 5, 4
        result = {"freq_matrix": np.random.rand(n_freq, n_heads).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert summary["dominant_freq_per_head"].shape == (n_heads,)

    def test_max_frac_shape(self):
        analyzer = AttentionFreqAnalyzer()
        n_freq, n_heads = 5, 4
        result = {"freq_matrix": np.random.rand(n_freq, n_heads).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert summary["max_frac_per_head"].shape == (n_heads,)

    def test_max_frac_in_range(self):
        analyzer = AttentionFreqAnalyzer()
        result = {"freq_matrix": np.random.rand(5, 4).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert np.all(summary["max_frac_per_head"] >= 0)
        assert np.all(summary["max_frac_per_head"] <= 1.0)

    def test_mean_specialization_is_scalar(self):
        analyzer = AttentionFreqAnalyzer()
        result = {"freq_matrix": np.random.rand(5, 4).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert isinstance(summary["mean_specialization"], float)

    def test_mean_specialization_in_range(self):
        analyzer = AttentionFreqAnalyzer()
        result = {"freq_matrix": np.random.rand(5, 4).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert 0 <= summary["mean_specialization"] <= 1.0

    def test_dominant_freq_valid_range(self):
        analyzer = AttentionFreqAnalyzer()
        n_freq, n_heads = 5, 4
        result = {"freq_matrix": np.random.rand(n_freq, n_heads).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert np.all(summary["dominant_freq_per_head"] >= 0)
        assert np.all(summary["dominant_freq_per_head"] < n_freq)

    def test_known_input_single_dominant(self):
        """When one frequency dominates, dominant_freq should match."""
        analyzer = AttentionFreqAnalyzer()
        n_freq, n_heads = 5, 4
        freq_matrix = np.zeros((n_freq, n_heads), dtype=np.float32)
        # Head 0 → freq 2, Head 1 → freq 0, Head 2 → freq 4, Head 3 → freq 1
        freq_matrix[2, 0] = 0.95
        freq_matrix[0, 1] = 0.95
        freq_matrix[4, 2] = 0.95
        freq_matrix[1, 3] = 0.95

        result = {"freq_matrix": freq_matrix}
        summary = analyzer.compute_summary(result, {})

        np.testing.assert_array_equal(summary["dominant_freq_per_head"], [2, 0, 4, 1])

    def test_known_input_all_zero(self):
        """All-zero matrix: counts should be zero, mean should be zero."""
        analyzer = AttentionFreqAnalyzer()
        result = {"freq_matrix": np.zeros((5, 4), dtype=np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert summary["mean_specialization"] == 0.0


# ── Position Pair Parameter ─────────────────────────────────────────────


class TestAttentionFreqPositionPair:
    """Tests that the position pair parameter works correctly."""

    def test_default_position_pair(self):
        analyzer = AttentionFreqAnalyzer()
        assert analyzer.to_position == 2
        assert analyzer.from_position == 0

    def test_custom_position_pair(self):
        analyzer = AttentionFreqAnalyzer(to_position=1, from_position=0)
        assert analyzer.to_position == 1
        assert analyzer.from_position == 0

    def test_different_position_pairs_different_output(self):
        """Different position pairs should produce different freq matrices."""
        np.random.seed(42)
        p, n_heads, n_pos = 7, 4, 3
        batch = p * p

        raw = np.random.rand(batch, n_heads, n_pos, n_pos).astype(np.float32)
        attn = raw / raw.sum(axis=-1, keepdims=True)
        attn_tensor = torch.tensor(attn)

        cache = MockCache({("pattern", 0): attn_tensor})
        probe = torch.zeros(batch, 3)
        fourier_basis, _ = get_fourier_basis(p)
        context = {"fourier_basis": fourier_basis}

        analyzer_a = AttentionFreqAnalyzer(to_position=2, from_position=0)
        analyzer_b = AttentionFreqAnalyzer(to_position=2, from_position=1)

        result_a = analyzer_a.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(None, cache, None),
                probe=probe,
                analysis_params=context,
            )
        )
        result_b = analyzer_b.analyze(
            ActivationContext(
                bundle=TransformerLensBundle(None, cache, None),
                probe=probe,
                analysis_params=context,
            )
        )

        assert not np.array_equal(result_a["freq_matrix"], result_b["freq_matrix"])
