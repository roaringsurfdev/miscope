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
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_dominant_frequencies

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("dominant_frequencies", 0)
        fig = render_dominant_frequencies(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)

    def test_load_epoch_and_render_neuron_heatmap(self, realistic_artifacts):
        """Can load single epoch and render neuron heatmap."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_neuron_heatmap

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("neuron_activations", 0)
        fig = render_neuron_heatmap(epoch_data, epoch=0, neuron_idx=0)

        assert isinstance(fig, go.Figure)

    def test_load_epoch_and_render_freq_clusters(self, realistic_artifacts):
        """Can load single epoch and render frequency clusters."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_freq_clusters

        loader = ArtifactLoader(str(realistic_artifacts))
        epoch_data = loader.load_epoch("neuron_freq_norm", 0)
        fig = render_freq_clusters(epoch_data, epoch=0)

        assert isinstance(fig, go.Figure)

    def test_all_renderers_no_exceptions(self, realistic_artifacts):
        """All renderers work without exceptions on realistic data."""
        from miscope.analysis import ArtifactLoader
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


class TestCoarsenessTrajectoryRenderer:
    """Tests for REQ_024: Coarseness Trajectory Visualization."""

    @pytest.fixture
    def summary_data(self):
        """Create summary data matching ArtifactLoader.load_summary output."""
        n_epochs = 5
        return {
            "epochs": np.array([0, 100, 500, 1000, 5000]),
            "mean_coarseness": np.array([0.2, 0.25, 0.35, 0.55, 0.72]),
            "std_coarseness": np.array([0.1, 0.12, 0.15, 0.18, 0.12]),
            "p25_coarseness": np.array([0.12, 0.15, 0.22, 0.40, 0.63]),
            "p75_coarseness": np.array([0.28, 0.35, 0.48, 0.70, 0.82]),
            "blob_count": np.array([0, 0, 2, 15, 120], dtype=np.float64),
            "coarseness_hist": np.random.rand(n_epochs, 20),
        }

    def test_render_returns_figure(self, summary_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500)
        assert isinstance(fig, go.Figure)

    def test_render_custom_title(self, summary_data):
        """Can set custom title."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_render_has_mean_trace(self, summary_data):
        """Figure contains the mean coarseness trace."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500)
        trace_names = [t.name for t in fig.data]
        assert "Mean" in trace_names

    def test_render_has_percentile_band(self, summary_data):
        """Figure contains the p25-p75 band."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500)
        # Band uses 3 traces: lower bound (no name), upper bound (p25-p75), mean
        assert len(fig.data) >= 3

    def test_render_has_epoch_indicator(self, summary_data):
        """Figure contains vertical line at current epoch."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500)
        # Check for vertical line shapes
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        vlines = [s for s in shapes if s.get("type") == "line" and s.get("x0") == 500]
        assert len(vlines) > 0

    def test_render_has_blob_threshold(self, summary_data):
        """Figure contains horizontal reference line for blob threshold."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500, blob_threshold=0.7)
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        hlines = [s for s in shapes if s.get("type") == "line" and s.get("y0") == 0.7]
        assert len(hlines) > 0

    def test_y_axis_range(self, summary_data):
        """Y-axis is fixed to [0, 1] range."""
        from miscope.visualization import render_coarseness_trajectory

        fig = render_coarseness_trajectory(summary_data, current_epoch=500)
        assert list(fig.layout.yaxis.range) == [0, 1]


class TestCoarsenessDistributionRenderer:
    """Tests for REQ_024: Coarseness Distribution Visualization."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch coarseness data."""
        np.random.seed(42)
        return {
            "coarseness": np.random.rand(512),
        }

    def test_render_returns_figure(self, epoch_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_render_custom_title(self, epoch_data):
        """Can set custom title."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(epoch_data, epoch=100, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_render_epoch_in_default_title(self, epoch_data):
        """Default title includes epoch number."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(epoch_data, epoch=100)
        assert "100" in fig.layout.title.text

    def test_render_has_threshold_lines(self, epoch_data):
        """Figure has vertical threshold reference lines."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(
            epoch_data, epoch=100, blob_threshold=0.7, plaid_threshold=0.5
        )
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        vlines = [s for s in shapes if s.get("type") == "line"]
        assert len(vlines) >= 2

    def test_render_has_histogram_traces(self, epoch_data):
        """Figure has histogram traces for the three regions."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(epoch_data, epoch=100)
        # Should have up to 3 histogram traces (plaid, transitional, blob)
        histogram_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(histogram_traces) >= 1

    def test_render_stacked_barmode(self, epoch_data):
        """Histogram uses stacked bar mode."""
        from miscope.visualization import render_coarseness_distribution

        fig = render_coarseness_distribution(epoch_data, epoch=100)
        assert fig.layout.barmode == "stack"

    def test_blob_count_in_title(self):
        """Default title includes blob count."""
        from miscope.visualization import render_coarseness_distribution

        # 3 neurons above 0.7
        data = {"coarseness": np.array([0.1, 0.3, 0.5, 0.75, 0.8, 0.9])}
        fig = render_coarseness_distribution(data, epoch=100, blob_threshold=0.7)
        assert "3/6" in fig.layout.title.text


class TestBlobCountTrajectoryRenderer:
    """Tests for REQ_024: Blob Count Trajectory."""

    @pytest.fixture
    def summary_data(self):
        """Create summary data."""
        return {
            "epochs": np.array([0, 100, 500]),
            "blob_count": np.array([0, 5, 50], dtype=np.float64),
        }

    def test_render_returns_figure(self, summary_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_blob_count_trajectory

        fig = render_blob_count_trajectory(summary_data)
        assert isinstance(fig, go.Figure)

    def test_render_with_epoch_indicator(self, summary_data):
        """Optional epoch indicator is added when specified."""
        from miscope.visualization import render_blob_count_trajectory

        fig = render_blob_count_trajectory(summary_data, current_epoch=100)
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        vlines = [s for s in shapes if s.get("type") == "line" and s.get("x0") == 100]
        assert len(vlines) > 0

    def test_render_without_epoch_indicator(self, summary_data):
        """No indicator when current_epoch is None."""
        from miscope.visualization import render_blob_count_trajectory

        fig = render_blob_count_trajectory(summary_data)
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        assert len(shapes) == 0


class TestCoarsenessByNeuronRenderer:
    """Tests for REQ_024: Coarseness by Neuron Index."""

    @pytest.fixture
    def epoch_data(self):
        """Create single-epoch coarseness data."""
        np.random.seed(42)
        return {
            "coarseness": np.random.rand(64),
        }

    def test_render_returns_figure(self, epoch_data):
        """Renderer returns a Plotly Figure."""
        from miscope.visualization import render_coarseness_by_neuron

        fig = render_coarseness_by_neuron(epoch_data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_render_has_bar_trace(self, epoch_data):
        """Figure contains a bar trace."""
        from miscope.visualization import render_coarseness_by_neuron

        fig = render_coarseness_by_neuron(epoch_data, epoch=100)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1

    def test_bar_count_matches_neurons(self, epoch_data):
        """Number of bars matches number of neurons."""
        from miscope.visualization import render_coarseness_by_neuron

        fig = render_coarseness_by_neuron(epoch_data, epoch=100)
        bar_trace = fig.data[0]
        assert len(bar_trace.x) == 64

    def test_render_epoch_in_default_title(self, epoch_data):
        """Default title includes epoch number."""
        from miscope.visualization import render_coarseness_by_neuron

        fig = render_coarseness_by_neuron(epoch_data, epoch=100)
        assert "100" in fig.layout.title.text

    def test_render_has_threshold_lines(self, epoch_data):
        """Figure has horizontal threshold reference lines."""
        from miscope.visualization import render_coarseness_by_neuron

        fig = render_coarseness_by_neuron(epoch_data, epoch=100)
        shapes = fig.to_dict().get("layout", {}).get("shapes", [])
        hlines = [s for s in shapes if s.get("type") == "line"]
        assert len(hlines) >= 2


class TestCoarsenessVisualizationIntegration:
    """Integration tests for coarseness renderers with ArtifactLoader."""

    @pytest.fixture
    def coarseness_artifacts(self, tmp_path):
        """Create coarseness artifacts matching pipeline output format."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        d_mlp = 32
        epochs = [0, 50, 100]

        # Per-epoch coarseness files
        c_dir = artifacts_dir / "coarseness"
        c_dir.mkdir()
        for epoch in epochs:
            coarseness = np.random.rand(d_mlp).astype(np.float32)
            np.savez_compressed(
                c_dir / f"epoch_{epoch:05d}.npz",
                coarseness=coarseness,
            )

        # Summary file
        n = len(epochs)
        np.savez_compressed(
            c_dir / "summary.npz",
            epochs=np.array(epochs),
            mean_coarseness=np.random.rand(n),
            std_coarseness=np.random.rand(n) * 0.1,
            median_coarseness=np.random.rand(n),
            p25_coarseness=np.random.rand(n) * 0.5,
            p75_coarseness=0.5 + np.random.rand(n) * 0.5,
            blob_count=np.array([0, 2, 10], dtype=np.float64),
            coarseness_hist=np.random.rand(n, 20),
        )

        return artifacts_dir

    def test_load_epoch_and_render_distribution(self, coarseness_artifacts):
        """Can load single epoch and render coarseness distribution."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_coarseness_distribution

        loader = ArtifactLoader(str(coarseness_artifacts))
        epoch_data = loader.load_epoch("coarseness", 0)
        fig = render_coarseness_distribution(epoch_data, epoch=0)
        assert isinstance(fig, go.Figure)

    def test_load_summary_and_render_trajectory(self, coarseness_artifacts):
        """Can load summary and render coarseness trajectory."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_coarseness_trajectory

        loader = ArtifactLoader(str(coarseness_artifacts))
        summary = loader.load_summary("coarseness")
        fig = render_coarseness_trajectory(summary, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_load_summary_and_render_blob_count(self, coarseness_artifacts):
        """Can load summary and render blob count trajectory."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_blob_count_trajectory

        loader = ArtifactLoader(str(coarseness_artifacts))
        summary = loader.load_summary("coarseness")
        fig = render_blob_count_trajectory(summary, current_epoch=50)
        assert isinstance(fig, go.Figure)

    def test_load_epoch_and_render_by_neuron(self, coarseness_artifacts):
        """Can load single epoch and render coarseness by neuron."""
        from miscope.analysis import ArtifactLoader
        from miscope.visualization import render_coarseness_by_neuron

        loader = ArtifactLoader(str(coarseness_artifacts))
        epoch_data = loader.load_epoch("coarseness", 0)
        fig = render_coarseness_by_neuron(epoch_data, epoch=0)
        assert isinstance(fig, go.Figure)
