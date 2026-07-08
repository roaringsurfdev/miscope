"""Tests for REQ_025: Attention Patterns Renderers."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import plotly.graph_objects as go
import pytest


class TestRenderAttentionHeads:
    """Tests for render_attention_heads renderer."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch attention patterns data.

        Shape: (n_heads=4, n_pos=3, n_pos=3, p=7, p=7)
        """
        np.random.seed(42)
        n_heads, n_pos, p = 4, 3, 7
        # Simulate softmax outputs (values in [0, 1])
        patterns = np.random.rand(n_heads, n_pos, n_pos, p, p)
        return {"patterns": patterns}

    def test_returns_figure(self, epoch_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_default_position_pair(self, epoch_data):
        """Default renders = attending to a (to=2, from=0)."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)
        # Title should mention = and a
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "=" in title_text
        assert "a" in title_text

    def test_custom_position_pair(self, epoch_data):
        """Can specify different position pair."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(epoch_data, epoch=100, to_position=1, from_position=0)
        assert isinstance(fig, go.Figure)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "b" in title_text

    def test_custom_title(self, epoch_data):
        """Custom title overrides auto-generated one."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(epoch_data, epoch=100, title="My Custom Title")
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "My Custom Title"

    def test_has_traces_for_each_head(self, epoch_data):
        """Figure has one heatmap trace per head."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(epoch_data, epoch=100)
        n_heads = epoch_data["patterns"].shape[0]
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == n_heads

    def test_custom_position_labels(self, epoch_data):
        """Custom position labels used in title."""
        from miscope.visualization import render_attention_heads

        fig = render_attention_heads(
            epoch_data,
            epoch=100,
            to_position=2,
            from_position=1,
            position_labels=["x", "y", "z"],
        )
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "z" in title_text
        assert "y" in title_text


class TestRenderAttentionSingleHead:
    """Tests for render_attention_single_head renderer."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch attention patterns data."""
        np.random.seed(42)
        n_heads, n_pos, p = 4, 3, 7
        patterns = np.random.rand(n_heads, n_pos, n_pos, p, p)
        return {"patterns": patterns}

    def test_returns_figure(self, epoch_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_attention_single_head

        fig = render_attention_single_head(epoch_data, epoch=100, head_idx=0)
        assert isinstance(fig, go.Figure)

    def test_single_heatmap_trace(self, epoch_data):
        """Figure has exactly one heatmap trace."""
        from miscope.visualization import render_attention_single_head

        fig = render_attention_single_head(epoch_data, epoch=100, head_idx=0)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_head_idx_out_of_range(self, epoch_data):
        """Invalid head index raises IndexError."""
        from miscope.visualization import render_attention_single_head

        with pytest.raises(IndexError):
            render_attention_single_head(epoch_data, epoch=100, head_idx=10)

    def test_custom_title(self, epoch_data):
        """Custom title overrides auto-generated one."""
        from miscope.visualization import render_attention_single_head

        fig = render_attention_single_head(epoch_data, epoch=100, head_idx=0, title="Custom")
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "Custom"

    def test_different_heads_different_data(self, epoch_data):
        """Different head_idx values produce different heatmap data."""
        from miscope.visualization import render_attention_single_head

        fig0 = render_attention_single_head(epoch_data, epoch=100, head_idx=0)
        fig1 = render_attention_single_head(epoch_data, epoch=100, head_idx=1)

        z0 = np.array(fig0.data[0].z)
        z1 = np.array(fig1.data[0].z)

        assert not np.array_equal(z0, z1)

    def test_position_pair_selects_data(self, epoch_data):
        """Different position pairs produce different data."""
        from miscope.visualization import render_attention_single_head

        fig_a = render_attention_single_head(
            epoch_data, epoch=100, head_idx=0, to_position=2, from_position=0
        )
        fig_b = render_attention_single_head(
            epoch_data, epoch=100, head_idx=0, to_position=2, from_position=1
        )

        z_a = np.array(fig_a.data[0].z)
        z_b = np.array(fig_b.data[0].z)

        assert not np.array_equal(z_a, z_b)
