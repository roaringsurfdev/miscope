"""Tests for REQ_004, REQ_005, REQ_006: Visualization Renderers."""
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import plotly.graph_objects as go
import pytest


class TestDominantFrequenciesRenderer:
    """Tests for REQ_004: Dominant Frequencies Visualization."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch dominant frequencies data."""
        return {
            "coefficients": np.array([0.5, 1.2, 0.3, 1.5, 0.2, 0.8, 0.1]),
        }

    @pytest.fixture
    def stacked_artifact(self):
        """Create multi-epoch stacked artifact for cross-epoch renderers."""
        return {
            "epochs": np.array([0, 50, 100]),
            "coefficients": np.array(
                [
                    [0.5, 1.2, 0.3, 1.5, 0.2, 0.8, 0.1],
                    [0.6, 1.5, 0.4, 2.0, 0.3, 1.0, 0.2],
                    [0.7, 1.8, 0.5, 2.5, 0.4, 1.2, 0.3],
                ]
            ),
        }

    def test_render_returns_figure(self, epoch_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)

    def test_render_with_threshold(self, epoch_data):
        """Threshold parameter affects visualization."""
        from miscope.visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(epoch_data, epoch=0, threshold=1.0)

        assert isinstance(fig, go.Figure)
        # Figure has horizontal line for threshold
        assert any(
            shape.get("type") == "line"
            for shape in fig.to_dict().get("layout", {}).get("shapes", [])
        )

    def test_render_without_highlight(self, epoch_data):
        """Can disable dominance highlighting."""
        from miscope.visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(epoch_data, epoch=0, highlight_dominant=False)

        assert isinstance(fig, go.Figure)

    def test_render_custom_title(self, epoch_data):
        """Can set custom title."""
        from miscope.visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(epoch_data, epoch=0, title="Custom Title")

        assert fig.layout.title.text == "Custom Title"

    def test_render_epoch_in_default_title(self, epoch_data):
        """Default title includes epoch number."""
        from miscope.visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(epoch_data, epoch=100)

        assert "100" in fig.layout.title.text

    def test_get_dominant_indices(self, epoch_data):
        """get_dominant_indices returns indices above threshold."""
        from miscope.visualization import get_dominant_indices

        indices = get_dominant_indices(epoch_data["coefficients"], threshold=1.0)

        assert 1 in indices  # 1.2 > 1.0
        assert 3 in indices  # 1.5 > 1.0
        assert 0 not in indices  # 0.5 < 1.0

    def test_get_fourier_basis_names(self):
        """Fourier basis names are correctly generated."""
        from miscope.visualization import get_fourier_basis_names

        names = get_fourier_basis_names(7)

        assert names[0] == "Const"
        assert names[1] == "cos 1"
        assert names[2] == "sin 1"
        assert names[3] == "cos 2"

    def test_render_over_time(self, stacked_artifact):
        """render_dominant_frequencies_over_time works with stacked data."""
        from miscope.visualization import render_dominant_frequencies_over_time

        fig = render_dominant_frequencies_over_time(stacked_artifact, top_k=3)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 traces for top 3 components


class TestNeuronActivationsRenderer:
    """Tests for REQ_005: Neuron Activation Heatmaps."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch neuron activations data."""
        # (d_mlp, p, p) — single epoch
        np.random.seed(42)
        return {
            "activations": np.random.randn(10, 7, 7),  # 10 neurons, p=7
        }

    @pytest.fixture
    def stacked_artifact(self):
        """Create multi-epoch stacked artifact for cross-epoch renderers."""
        np.random.seed(42)
        return {
            "epochs": np.array([0, 50, 100]),
            "activations": np.random.randn(3, 10, 7, 7),
        }

    def test_render_single_heatmap(self, epoch_data):
        """Single neuron heatmap renders correctly."""
        from miscope.visualization import render_neuron_heatmap

        fig = render_neuron_heatmap(epoch_data, epoch=0, neuron_idx=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_has_correct_shape(self, epoch_data):
        """Heatmap data has correct dimensions."""
        from miscope.visualization import render_neuron_heatmap

        fig = render_neuron_heatmap(epoch_data, epoch=0, neuron_idx=0)

        # Heatmap z data should be (p, p)
        z_data = fig.data[0].z
        assert z_data.shape == (7, 7)

    def test_heatmap_invalid_neuron(self, epoch_data):
        """Invalid neuron index raises error."""
        from miscope.visualization import render_neuron_heatmap

        with pytest.raises(IndexError):
            render_neuron_heatmap(epoch_data, epoch=0, neuron_idx=100)

    def test_render_neuron_grid(self, epoch_data):
        """Grid of neurons renders correctly."""
        from miscope.visualization import render_neuron_grid

        fig = render_neuron_grid(epoch_data, epoch=0, neuron_indices=[0, 1, 2, 3, 4])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5  # 5 neurons

    def test_grid_custom_cols(self, epoch_data):
        """Grid respects column count."""
        from miscope.visualization import render_neuron_grid

        fig = render_neuron_grid(epoch_data, epoch=0, neuron_indices=[0, 1, 2], cols=2)

        assert isinstance(fig, go.Figure)

    def test_render_neuron_across_epochs(self, stacked_artifact):
        """Single neuron across epochs renders correctly."""
        from miscope.visualization import render_neuron_across_epochs

        fig = render_neuron_across_epochs(stacked_artifact, neuron_idx=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 epochs

    def test_get_most_active_neurons(self, epoch_data):
        """get_most_active_neurons returns list of indices."""
        from miscope.visualization import get_most_active_neurons

        neurons = get_most_active_neurons(epoch_data, top_k=5)

        assert len(neurons) == 5
        assert all(isinstance(n, int) for n in neurons)
        assert all(0 <= n < 10 for n in neurons)


class TestFreqClustersRenderer:
    """Tests for REQ_006: Neuron Frequency Clusters."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch frequency clusters data."""
        np.random.seed(42)
        # (n_freq, d_mlp) — single epoch
        return {
            "norm_matrix": np.random.rand(8, 20),  # 8 freqs, 20 neurons
        }

    @pytest.fixture
    def stacked_artifact(self):
        """Create multi-epoch stacked artifact for cross-epoch renderers."""
        np.random.seed(42)
        return {
            "epochs": np.array([0, 50, 100]),
            "norm_matrix": np.random.rand(3, 8, 20),
        }

    def test_render_freq_clusters(self, epoch_data):
        """Frequency clusters heatmap renders correctly."""
        from miscope.visualization import render_freq_clusters

        fig = render_freq_clusters(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_has_correct_shape(self, epoch_data):
        """Heatmap has correct dimensions."""
        from miscope.visualization import render_freq_clusters

        fig = render_freq_clusters(epoch_data, epoch=0)

        z_data = fig.data[0].z
        assert z_data.shape == (8, 20)  # (n_freq, d_mlp)

    def test_sparse_labels(self, epoch_data):
        """Sparse labels are applied correctly."""
        from miscope.visualization import render_freq_clusters

        fig = render_freq_clusters(epoch_data, epoch=0, sparse_labels=True, label_interval=2)

        # Check y-axis has sparse tick values
        y_tickvals = fig.layout.yaxis.tickvals
        assert y_tickvals is not None
        assert len(y_tickvals) < 8  # Fewer than all frequencies

    def test_full_labels(self, epoch_data):
        """Full labels can be shown."""
        from miscope.visualization import render_freq_clusters

        fig = render_freq_clusters(epoch_data, epoch=0, sparse_labels=False)

        y_tickvals = fig.layout.yaxis.tickvals
        assert len(y_tickvals) == 8  # All frequencies

    def test_colorbar_optional(self, epoch_data):
        """Colorbar can be hidden."""
        from miscope.visualization import render_freq_clusters

        fig = render_freq_clusters(epoch_data, epoch=0, show_colorbar=False)

        assert fig.data[0].showscale is False

    def test_render_comparison(self, stacked_artifact):
        """Comparison view renders multiple epochs."""
        from miscope.visualization import render_freq_clusters_comparison

        fig = render_freq_clusters_comparison(stacked_artifact, epoch_indices=[0, 1, 2])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 epochs

    def test_get_specialized_neurons(self, epoch_data):
        """get_specialized_neurons returns correct neurons."""
        from miscope.visualization import get_specialized_neurons

        # Set one neuron to be clearly specialized for frequency 1
        epoch_data["norm_matrix"][0, 5] = 0.95

        neurons = get_specialized_neurons(epoch_data, frequency=1, threshold=0.85)

        assert 5 in neurons

    def test_get_neuron_specialization(self, epoch_data):
        """get_neuron_specialization returns dominant frequency."""
        from miscope.visualization import get_neuron_specialization

        # Set neuron 3 to be specialized for frequency 5
        epoch_data["norm_matrix"][:, 3] = 0.1
        epoch_data["norm_matrix"][4, 3] = 0.9

        freq, frac = get_neuron_specialization(epoch_data, neuron_idx=3)

        assert freq == 5  # 1-indexed (index 4 = freq 5)
        assert frac == pytest.approx(0.9)


class TestVisualizationIntegration:
    """Integration tests using per-epoch artifact structure (REQ_021f)."""

    @pytest.fixture
    def realistic_artifacts(self, tmp_path):
        """Create per-epoch artifacts matching pipeline output format."""

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        p = 17
        n_components = 2 * p - 1
        d_mlp = 32
        epochs = [0, 25, 50, 75, 100]

        # Create per-epoch files for each analyzer
        df_dir = artifacts_dir / "dominant_frequencies"
        df_dir.mkdir()
        for epoch in epochs:
            np.savez_compressed(
                df_dir / f"epoch_{epoch:05d}.npz",
                coefficients=np.random.rand(n_components),
            )

        na_dir = artifacts_dir / "neuron_activations"
        na_dir.mkdir()
        for epoch in epochs:
            np.savez_compressed(
                na_dir / f"epoch_{epoch:05d}.npz",
                activations=np.random.randn(d_mlp, p, p),
            )

        fn_dir = artifacts_dir / "neuron_freq_norm"
        fn_dir.mkdir()
        for epoch in epochs:
            np.savez_compressed(
                fn_dir / f"epoch_{epoch:05d}.npz",
                norm_matrix=np.random.rand(p // 2, d_mlp),
            )

        return artifacts_dir

    def test_load_epoch_and_render_dominant_frequencies(self, realistic_artifacts):
        """Can load single epoch and render dominant frequencies."""
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.visualization import render_dominant_frequencies

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("dominant_frequencies", 0)
        fig = render_dominant_frequencies(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)

    def test_load_epoch_and_render_neuron_heatmap(self, realistic_artifacts):
        """Can load single epoch and render neuron heatmap."""
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.visualization import render_neuron_heatmap

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("neuron_activations", 0)
        fig = render_neuron_heatmap(epoch_data, epoch=0, neuron_idx=0)

        assert isinstance(fig, go.Figure)

    def test_load_epoch_and_render_freq_clusters(self, realistic_artifacts):
        """Can load single epoch and render frequency clusters."""
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.visualization import render_freq_clusters

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("neuron_freq_norm", 0)
        fig = render_freq_clusters(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)

    def test_all_renderers_no_exceptions(self, realistic_artifacts):
        """All renderers work without exceptions on realistic data."""
        from miscope.analysis.artifact_loader import ArtifactLoader
        from miscope.visualization import (
            get_dominant_indices,
            get_most_active_neurons,
            get_neuron_specialization,
            get_specialized_neurons,
            render_dominant_frequencies,
            render_dominant_frequencies_over_time,
            render_freq_clusters,
            render_freq_clusters_comparison,
            render_neuron_across_epochs,
            render_neuron_grid,
            render_neuron_heatmap,
        )

        loader = ArtifactLoader(str(realistic_artifacts))

        # Per-epoch renderers (single-epoch data)
        df_data = loader.load_epoch("dominant_frequencies", 0)
        render_dominant_frequencies(df_data, epoch=0)
        get_dominant_indices(df_data["coefficients"], threshold=0.5)

        na_data = loader.load_epoch("neuron_activations", 0)
        render_neuron_heatmap(na_data, epoch=0, neuron_idx=0)
        render_neuron_grid(na_data, epoch=0, neuron_indices=[0, 1, 2])
        get_most_active_neurons(na_data, top_k=5)

        fc_data = loader.load_epoch("neuron_freq_norm", 0)
        render_freq_clusters(fc_data, epoch=0)
        get_specialized_neurons(fc_data, frequency=1, threshold=0.5)
        get_neuron_specialization(fc_data, neuron_idx=0)

        # Cross-epoch renderers (stacked data)
        df_stacked = loader.load("dominant_frequencies")
        render_dominant_frequencies_over_time(df_stacked)

        na_stacked = loader.load("neuron_activations")
        render_neuron_across_epochs(na_stacked, neuron_idx=0)

        fc_stacked = loader.load("neuron_freq_norm")
        render_freq_clusters_comparison(fc_stacked, epoch_indices=[0, 1])
