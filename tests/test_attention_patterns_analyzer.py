"""Tests for REQ_025: Attention Patterns Analyzer."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers.attention_patterns import AttentionPatternsAnalyzer
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.library.activations import extract_attention_patterns

# ── Library function tests ──────────────────────────────────────────────


class MockCache(dict):
    """Minimal mock for TransformerLens ActivationCache."""

    def __getitem__(self, key):
        return super().__getitem__(key)


class TestExtractAttentionPatterns:
    """Tests for extract_attention_patterns library function."""

    def test_returns_correct_shape(self):
        """Returns tensor of shape (batch, n_heads, seq_to, seq_from)."""
        batch, n_heads, seq_len = 49, 4, 3
        patterns = torch.rand(batch, n_heads, seq_len, seq_len)
        cache = MockCache({("pattern", 0): patterns})

        result = extract_attention_patterns(cache, layer=0)

        assert result.shape == (batch, n_heads, seq_len, seq_len)

    def test_layer_parameter(self):
        """Layer parameter selects the correct layer."""
        patterns_0 = torch.zeros(9, 2, 3, 3)
        patterns_1 = torch.ones(9, 2, 3, 3)
        cache = MockCache({("pattern", 0): patterns_0, ("pattern", 1): patterns_1})

        result = extract_attention_patterns(cache, layer=1)

        assert torch.all(result == 1.0)

    def test_default_layer_is_zero(self):
        """Default layer is 0."""
        patterns = torch.rand(9, 4, 3, 3)
        cache = MockCache({("pattern", 0): patterns})

        result = extract_attention_patterns(cache)

        assert torch.equal(result, patterns)


# ── Analyzer tests ──────────────────────────────────────────────────────


class TestAttentionPatternsAnalyzer:
    """Tests for AttentionPatternsAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return AttentionPatternsAnalyzer()

    @pytest.fixture
    def mock_inputs(self):
        """Create mock inputs for a p=7 model with 4 heads, 3 positions."""
        p = 7
        n_heads = 4
        seq_len = 3
        batch = p * p

        # Probe: (p^2, 3) — standard modular arithmetic format
        probe = torch.zeros(batch, seq_len, dtype=torch.long)

        # Attention patterns: (batch, n_heads, seq_to, seq_from)
        # Values should be valid softmax outputs (rows sum to 1)
        raw = torch.rand(batch, n_heads, seq_len, seq_len)
        patterns = raw / raw.sum(dim=-1, keepdim=True)

        cache = MockCache({("pattern", 0): patterns})
        context = {"params": {"prime": p}}

        return probe, cache, context, p, n_heads, seq_len

    def test_has_name(self, analyzer):
        """Analyzer has correct name attribute."""
        assert analyzer.name == "attention_patterns"

    def test_has_description(self, analyzer):
        """Analyzer has description attribute."""
        assert hasattr(analyzer, "description")
        assert len(analyzer.description) > 0

    def test_analyze_returns_dict(self, analyzer, mock_inputs):
        """analyze() returns a dict."""
        probe, cache, context, *_ = mock_inputs
        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        assert isinstance(result, dict)

    def test_analyze_produces_patterns_key(self, analyzer, mock_inputs):
        """analyze() result contains 'patterns' key."""
        probe, cache, context, *_ = mock_inputs
        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        assert "patterns" in result

    def test_output_shape(self, analyzer, mock_inputs):
        """Output shape is (n_heads, n_pos, n_pos, p, p)."""
        probe, cache, context, p, n_heads, seq_len = mock_inputs
        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        patterns = result["patterns"]

        assert patterns.shape == (n_heads, seq_len, seq_len, p, p)

    def test_output_is_numpy(self, analyzer, mock_inputs):
        """Output is numpy array, not torch tensor."""
        probe, cache, context, *_ = mock_inputs
        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        assert isinstance(result["patterns"], np.ndarray)

    def test_values_in_valid_range(self, analyzer, mock_inputs):
        """Attention values are in [0, 1] (softmax outputs)."""
        probe, cache, context, *_ = mock_inputs
        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        patterns = result["patterns"]

        assert patterns.min() >= 0.0
        assert patterns.max() <= 1.0

    def test_grid_reshape_correctness(self, analyzer):
        """Verify the batch-to-grid reshape is correct.

        For a known input where pattern[a*p+b] has a specific value,
        the reshaped result should have that value at position [a, b].
        """
        p = 5
        n_heads = 1
        seq_len = 3
        batch = p * p

        probe = torch.zeros(batch, seq_len, dtype=torch.long)

        # Create a pattern where attention is a function of input index
        patterns = torch.zeros(batch, n_heads, seq_len, seq_len)
        for a in range(p):
            for b in range(p):
                idx = a * p + b
                # Set a distinguishable value
                patterns[idx, 0, 2, 0] = float(a + b) / (2 * p)

        cache = MockCache({("pattern", 0): patterns})
        context = {"params": {"prime": p}}

        result = analyzer.analyze(TransformerLensBundle(None, cache, None), probe, context)
        reshaped = result["patterns"]

        # Check that reshaped[head=0, to=2, from=0, a, b] == (a + b) / (2 * p)
        for a in range(p):
            for b in range(p):
                expected = float(a + b) / (2 * p)
                actual = reshaped[0, 2, 0, a, b]
                assert abs(actual - expected) < 1e-6, (
                    f"Mismatch at ({a}, {b}): expected {expected}, got {actual}"
                )
