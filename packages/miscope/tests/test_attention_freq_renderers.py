"""Tests for REQ_026: Attention Head Frequency Specialization Renderers."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import plotly.graph_objects as go
import pytest


class TestRenderAttentionFreqHeatmap:
    """Tests for render_attention_freq_heatmap renderer."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch attention frequency data.

        Shape: (n_freq=5, n_heads=4)
        """
        np.random.seed(42)
        freq_matrix = np.random.rand(5, 4).astype(np.float32)
        # Normalize columns to sum to 1
        freq_matrix = freq_matrix / freq_matrix.sum(axis=0, keepdims=True)
        return {"freq_matrix": freq_matrix}

    def test_returns_figure(self, epoch_data):
        from miscope.visualization import render_attention_freq_heatmap

        fig = render_attention_freq_heatmap(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, epoch_data):
        from miscope.visualization import render_attention_freq_heatmap

        fig = render_attention_freq_heatmap(epoch_data, epoch=100)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_default_title_has_epoch(self, epoch_data):
        from miscope.visualization import render_attention_freq_heatmap

        fig = render_attention_freq_heatmap(epoch_data, epoch=100)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "100" in title_text

    def test_custom_title(self, epoch_data):
        from miscope.visualization import render_attention_freq_heatmap

        fig = render_attention_freq_heatmap(epoch_data, epoch=100, title="Custom")
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "Custom"

    def test_head_labels_on_x_axis(self, epoch_data):
        from miscope.visualization import render_attention_freq_heatmap

        fig = render_attention_freq_heatmap(epoch_data, epoch=100)
        ticktext = fig.layout.xaxis.ticktext
        assert "Head 0" in ticktext
        assert "Head 3" in ticktext


class TestRenderAttentionSpecializationTrajectory:
    """Tests for render_attention_specialization_trajectory renderer."""

    @pytest.fixture
    def summary_data(self):
        """Create cross-epoch summary data."""
        np.random.seed(42)
        n_epochs, n_heads = 10, 4
        epochs = np.array([100 * i for i in range(n_epochs)])
        max_frac = np.random.rand(n_epochs, n_heads).astype(np.float64)
        return {
            "epochs": epochs,
            "max_frac_per_head": max_frac,
        }

    def test_returns_figure(self, summary_data):
        from miscope.visualization import render_attention_specialization_trajectory

        fig = render_attention_specialization_trajectory(summary_data, current_epoch=500)
        assert isinstance(fig, go.Figure)

    def test_has_traces_per_head(self, summary_data):
        from miscope.visualization import render_attention_specialization_trajectory

        fig = render_attention_specialization_trajectory(summary_data, current_epoch=500)
        n_heads = summary_data["max_frac_per_head"].shape[1]
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == n_heads

    def test_has_epoch_indicator(self, summary_data):
        from miscope.visualization import render_attention_specialization_trajectory

        fig = render_attention_specialization_trajectory(summary_data, current_epoch=500)
        # Vertical line shows up in layout shapes
        vlines = [s for s in (fig.layout.shapes or []) if getattr(s, "x0", None) == 500]
        assert len(vlines) > 0

    def test_custom_title(self, summary_data):
        from miscope.visualization import render_attention_specialization_trajectory

        fig = render_attention_specialization_trajectory(
            summary_data, current_epoch=500, title="My Title"
        )
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "My Title"

    def test_y_axis_range(self, summary_data):
        from miscope.visualization import render_attention_specialization_trajectory

        fig = render_attention_specialization_trajectory(summary_data, current_epoch=500)
        y_range = fig.layout.yaxis.range
        assert list(y_range) == [0, 1]


class TestRenderAttentionDominantFrequencies:
    """Tests for render_attention_dominant_frequencies renderer."""

    @pytest.fixture
    def summary_data(self):
        """Create cross-epoch summary data."""
        np.random.seed(42)
        n_epochs, n_heads = 10, 4
        epochs = np.array([100 * i for i in range(n_epochs)])
        dominant = np.random.randint(0, 5, size=(n_epochs, n_heads)).astype(np.float64)
        return {
            "epochs": epochs,
            "dominant_freq_per_head": dominant,
        }

    def test_returns_figure(self, summary_data):
        from miscope.visualization import render_attention_dominant_frequencies

        fig = render_attention_dominant_frequencies(summary_data)
        assert isinstance(fig, go.Figure)

    def test_has_traces_per_head(self, summary_data):
        from miscope.visualization import render_attention_dominant_frequencies

        fig = render_attention_dominant_frequencies(summary_data)
        n_heads = summary_data["dominant_freq_per_head"].shape[1]
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == n_heads

    def test_with_epoch_indicator(self, summary_data):
        from miscope.visualization import render_attention_dominant_frequencies

        fig = render_attention_dominant_frequencies(summary_data, current_epoch=500)
        vlines = [s for s in (fig.layout.shapes or []) if getattr(s, "x0", None) == 500]
        assert len(vlines) > 0

    def test_without_epoch_indicator(self, summary_data):
        from miscope.visualization import render_attention_dominant_frequencies

        fig = render_attention_dominant_frequencies(summary_data)
        shapes = fig.layout.shapes or []
        assert len(shapes) == 0

    def test_custom_title(self, summary_data):
        from miscope.visualization import render_attention_dominant_frequencies

        fig = render_attention_dominant_frequencies(summary_data, title="Custom")
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "Custom"
