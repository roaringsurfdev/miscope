"""Tests for REQ_027: Neuron Frequency Specialization Summary Statistics."""

# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer

# ── Analyzer Summary Methods ─────────────────────────────────────────


class TestNeuronFreqClustersSummaryKeys:
    """Tests for get_summary_keys() method."""

    def test_returns_list(self):
        analyzer = NeuronFreqClustersAnalyzer()
        keys = analyzer.get_summary_keys()
        assert isinstance(keys, list)

    def test_expected_keys_present(self):
        analyzer = NeuronFreqClustersAnalyzer()
        keys = analyzer.get_summary_keys()
        expected = [
            "specialized_count_per_freq",
            "specialized_count_low",
            "specialized_count_mid",
            "specialized_count_high",
            "specialized_count_total",
            "mean_max_frac",
            "median_max_frac",
        ]
        assert keys == expected

    def test_keys_match_compute_summary_output(self):
        analyzer = NeuronFreqClustersAnalyzer()
        keys = set(analyzer.get_summary_keys())
        result = {"norm_matrix": np.random.rand(5, 16).astype(np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert keys == set(summary.keys())


class TestNeuronFreqClustersSummaryOutput:
    """Tests for compute_summary() output shapes and values."""

    @pytest.fixture
    def n_freq(self):
        return 5

    @pytest.fixture
    def d_mlp(self):
        return 64

    @pytest.fixture
    def norm_matrix(self, n_freq, d_mlp):
        np.random.seed(42)
        return np.random.rand(n_freq, d_mlp).astype(np.float32)

    @pytest.fixture
    def summary(self, norm_matrix):
        analyzer = NeuronFreqClustersAnalyzer()
        return analyzer.compute_summary({"norm_matrix": norm_matrix}, {})

    def test_specialized_count_per_freq_shape(self, summary, n_freq):
        assert summary["specialized_count_per_freq"].shape == (n_freq,)

    def test_specialized_count_per_freq_non_negative(self, summary):
        assert np.all(summary["specialized_count_per_freq"] >= 0)

    def test_scalar_counts_are_floats(self, summary):
        assert isinstance(summary["specialized_count_low"], float)
        assert isinstance(summary["specialized_count_mid"], float)
        assert isinstance(summary["specialized_count_high"], float)
        assert isinstance(summary["specialized_count_total"], float)

    def test_range_counts_sum_to_total(self, summary):
        range_sum = (
            summary["specialized_count_low"]
            + summary["specialized_count_mid"]
            + summary["specialized_count_high"]
        )
        assert range_sum == summary["specialized_count_total"]

    def test_total_at_most_d_mlp(self, summary, d_mlp):
        assert summary["specialized_count_total"] <= d_mlp

    def test_mean_max_frac_is_float(self, summary):
        assert isinstance(summary["mean_max_frac"], float)

    def test_median_max_frac_is_float(self, summary):
        assert isinstance(summary["median_max_frac"], float)

    def test_mean_max_frac_in_range(self, summary):
        assert 0 <= summary["mean_max_frac"] <= 1.0

    def test_median_max_frac_in_range(self, summary):
        assert 0 <= summary["median_max_frac"] <= 1.0


class TestNeuronFreqClustersKnownInputs:
    """Tests with known inputs for compute_summary()."""

    def test_all_zero_matrix(self):
        """All-zero matrix: no neurons specialized, metrics all zero."""
        analyzer = NeuronFreqClustersAnalyzer()
        result = {"norm_matrix": np.zeros((5, 16), dtype=np.float32)}
        summary = analyzer.compute_summary(result, {})
        assert summary["specialized_count_total"] == 0.0
        assert summary["mean_max_frac"] == 0.0
        assert summary["median_max_frac"] == 0.0

    def test_all_specialized_in_one_freq(self):
        """All neurons specialized in freq 0 (low range)."""
        analyzer = NeuronFreqClustersAnalyzer(specialization_threshold=0.9)
        n_freq, d_mlp = 6, 8
        norm_matrix = np.zeros((n_freq, d_mlp), dtype=np.float32)
        norm_matrix[0, :] = 0.95  # All neurons → freq 0
        result = {"norm_matrix": norm_matrix}
        summary = analyzer.compute_summary(result, {})

        assert summary["specialized_count_total"] == d_mlp
        assert summary["specialized_count_low"] == d_mlp
        assert summary["specialized_count_mid"] == 0.0
        assert summary["specialized_count_high"] == 0.0
        np.testing.assert_array_equal(
            summary["specialized_count_per_freq"],
            [d_mlp, 0, 0, 0, 0, 0],
        )

    def test_neurons_split_across_ranges(self):
        """Neurons spread across low, mid, high ranges."""
        analyzer = NeuronFreqClustersAnalyzer(specialization_threshold=0.9)
        n_freq = 9  # low=[0,1,2], mid=[3,4,5], high=[6,7,8]
        d_mlp = 9
        norm_matrix = np.zeros((n_freq, d_mlp), dtype=np.float32)
        # 3 neurons in low (freq 0,1,2), 3 in mid (3,4,5), 3 in high (6,7,8)
        for i in range(9):
            norm_matrix[i, i] = 0.95
        result = {"norm_matrix": norm_matrix}
        summary = analyzer.compute_summary(result, {})

        assert summary["specialized_count_total"] == 9.0
        assert summary["specialized_count_low"] == 3.0
        assert summary["specialized_count_mid"] == 3.0
        assert summary["specialized_count_high"] == 3.0

    def test_threshold_boundary(self):
        """Neuron at exactly threshold is counted as specialized."""
        analyzer = NeuronFreqClustersAnalyzer(specialization_threshold=0.9)
        norm_matrix = np.zeros((5, 4), dtype=np.float32)
        norm_matrix[2, 0] = 0.9  # Exactly at threshold
        norm_matrix[2, 1] = 0.89  # Below threshold
        result = {"norm_matrix": norm_matrix}
        summary = analyzer.compute_summary(result, {})

        assert summary["specialized_count_total"] == 1.0

    def test_custom_threshold(self):
        """Custom threshold changes count."""
        norm_matrix = np.zeros((5, 4), dtype=np.float32)
        norm_matrix[2, 0] = 0.8
        norm_matrix[2, 1] = 0.7
        norm_matrix[2, 2] = 0.6
        result = {"norm_matrix": norm_matrix}

        analyzer_high = NeuronFreqClustersAnalyzer(specialization_threshold=0.9)
        summary_high = analyzer_high.compute_summary(result, {})

        analyzer_low = NeuronFreqClustersAnalyzer(specialization_threshold=0.5)
        summary_low = analyzer_low.compute_summary(result, {})

        assert summary_high["specialized_count_total"] == 0.0
        assert summary_low["specialized_count_total"] == 3.0

    def test_per_freq_counts_sum_to_total(self):
        """Per-frequency counts should sum to total."""
        np.random.seed(123)
        analyzer = NeuronFreqClustersAnalyzer(specialization_threshold=0.5)
        norm_matrix = np.random.rand(5, 32).astype(np.float32)
        result = {"norm_matrix": norm_matrix}
        summary = analyzer.compute_summary(result, {})

        assert summary["specialized_count_per_freq"].sum() == summary["specialized_count_total"]


# ── Specialization Trajectory Renderer ────────────────────────────────


class TestRenderSpecializationTrajectory:
    """Tests for render_specialization_trajectory renderer."""

    @pytest.fixture
    def summary_data(self):
        """Create cross-epoch summary data for neuron specialization."""
        n_epochs = 10
        epochs = np.array([100 * i for i in range(n_epochs)])
        # Simulate increasing specialization over training
        total = np.linspace(0, 200, n_epochs)
        low = np.linspace(0, 80, n_epochs)
        mid = np.linspace(0, 60, n_epochs)
        high = np.linspace(0, 60, n_epochs)
        return {
            "epochs": epochs,
            "specialized_count_total": total,
            "specialized_count_low": low,
            "specialized_count_mid": mid,
            "specialized_count_high": high,
        }

    def test_returns_figure(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(summary_data, current_epoch=500)
        assert isinstance(fig, go.Figure)

    def test_has_four_traces(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(summary_data, current_epoch=500)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 4  # total, low, mid, high

    def test_trace_names(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(summary_data, current_epoch=500)
        names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "Total" in names
        assert "Low freq" in names
        assert "Mid freq" in names
        assert "High freq" in names

    def test_has_epoch_indicator(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(summary_data, current_epoch=500)
        vlines = [s for s in (fig.layout.shapes or []) if getattr(s, "x0", None) == 500]
        assert len(vlines) > 0

    def test_custom_title(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(
            summary_data, current_epoch=500, title="Custom Title"
        )
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "Custom Title"

    def test_default_title(self, summary_data):
        from miscope.visualization import render_specialization_trajectory

        fig = render_specialization_trajectory(summary_data, current_epoch=500)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "Specialization" in title_text


# ── Specialization by Frequency Renderer ──────────────────────────────


class TestRenderSpecializationByFrequency:
    """Tests for render_specialization_by_frequency renderer."""

    @pytest.fixture
    def summary_data(self):
        """Create cross-epoch summary data with per-freq counts."""
        np.random.seed(42)
        n_epochs, n_freq = 10, 5
        epochs = np.array([100 * i for i in range(n_epochs)])
        per_freq = np.random.randint(0, 50, size=(n_epochs, n_freq)).astype(np.float64)
        return {
            "epochs": epochs,
            "specialized_count_per_freq": per_freq,
        }

    def test_returns_figure(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_with_epoch_indicator(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data, current_epoch=500)
        vlines = [s for s in (fig.layout.shapes or []) if getattr(s, "x0", None) == 500]
        assert len(vlines) > 0

    def test_without_epoch_indicator(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data)
        shapes = fig.layout.shapes or []
        assert len(shapes) == 0

    def test_custom_title(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data, title="Custom")
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert title_text == "Custom"

    def test_default_title(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data)
        title_text = fig.layout.title.text if fig.layout.title else ""
        assert "Specialized" in title_text

    def test_freq_labels_on_y_axis(self, summary_data):
        from miscope.visualization import render_specialization_by_frequency

        fig = render_specialization_by_frequency(summary_data)
        n_freq = summary_data["specialized_count_per_freq"].shape[1]
        ticktext = fig.layout.yaxis.ticktext
        assert len(ticktext) == n_freq
        assert ticktext[0] == "1"
        assert ticktext[-1] == str(n_freq)
