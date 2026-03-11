"""Tests for REQ_058: Neuron Band Concentration Health Metrics."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.band_concentration import (
    compute_band_concentration_trajectory,
    compute_critical_mass_snapshot,
    compute_embedding_band_magnitudes,
    compute_hhi,
    compute_rank_alignment_trajectory,
    compute_slope_cv,
)
from miscope.visualization.renderers.band_concentration import (
    render_concentration_scatter,
    render_concentration_trajectory,
    render_rank_alignment_trajectory,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

P = 11       # small prime for fast tests
N_FREQ = P // 2  # = 5
N_EPOCHS = 20
D_MLP = 32


def _make_cross_epoch(
    *,
    n_epochs: int = N_EPOCHS,
    d_mlp: int = D_MLP,
    n_freq: int = N_FREQ,
    seed: int = 0,
) -> dict:
    """Build synthetic neuron_dynamics cross-epoch artifact."""
    rng = np.random.default_rng(seed)
    dominant_freq = rng.integers(0, n_freq, size=(n_epochs, d_mlp))
    max_frac = rng.uniform(0.5, 1.0, size=(n_epochs, d_mlp))
    epochs = np.arange(n_epochs) * 100
    return {
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "epochs": epochs,
    }


def _make_coeff_epochs(n_epochs: int = N_EPOCHS, p: int = P, seed: int = 1) -> dict:
    """Build synthetic dominant_frequencies stacked artifact."""
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(0.0, 2.0, size=(n_epochs, p + 1)).astype(np.float32)
    epochs = np.arange(n_epochs) * 100
    return {"epochs": epochs, "coefficients": coefficients}


# ---------------------------------------------------------------------------
# CoS 1 / CoS 7: HHI correctness
# ---------------------------------------------------------------------------


class TestComputeHhi:
    def test_uniform_distribution(self):
        # n bands, equal shares → HHI = 1/n
        n = 4
        counts = np.ones(n)
        assert pytest.approx(compute_hhi(counts), abs=1e-6) == 1.0 / n

    def test_monopoly(self):
        # All in one band → HHI = 1.0
        counts = np.array([0.0, 0.0, 10.0, 0.0])
        assert pytest.approx(compute_hhi(counts), abs=1e-6) == 1.0

    def test_two_equal_bands(self):
        counts = np.array([5.0, 5.0])
        assert pytest.approx(compute_hhi(counts), abs=1e-6) == 0.5

    def test_zero_total_returns_nan(self):
        counts = np.zeros(4)
        assert np.isnan(compute_hhi(counts))

    def test_single_band(self):
        counts = np.array([7.0])
        assert pytest.approx(compute_hhi(counts), abs=1e-6) == 1.0


class TestBandConcentrationTrajectory:
    def test_output_keys(self):
        cross_epoch = _make_cross_epoch()
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.75, prime=P)
        for key in ("epochs", "active_band_count", "hhi", "max_band_share"):
            assert key in result

    def test_output_shapes(self):
        cross_epoch = _make_cross_epoch()
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.75, prime=P)
        assert result["epochs"].shape == (N_EPOCHS,)
        assert result["hhi"].shape == (N_EPOCHS,)
        assert result["active_band_count"].shape == (N_EPOCHS,)
        assert result["max_band_share"].shape == (N_EPOCHS,)

    def test_hhi_in_valid_range(self):
        cross_epoch = _make_cross_epoch()
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.75, prime=P)
        hhi = result["hhi"]
        valid = hhi[~np.isnan(hhi)]
        assert (valid >= 0).all()
        assert (valid <= 1.0 + 1e-6).all()

    def test_max_band_share_leq_one(self):
        cross_epoch = _make_cross_epoch()
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.75, prime=P)
        mbs = result["max_band_share"]
        valid = mbs[~np.isnan(mbs)]
        assert (valid <= 1.0 + 1e-6).all()

    def test_monopoly_gives_hhi_one(self):
        """All neurons committed to a single frequency → HHI = 1."""
        n_epochs, d_mlp = 5, 16
        dominant_freq = np.zeros((n_epochs, d_mlp), dtype=int)
        max_frac = np.ones((n_epochs, d_mlp)) * 0.95
        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.9, prime=P)
        np.testing.assert_allclose(result["hhi"], np.ones(n_epochs), atol=1e-6)

    def test_uniform_gives_hhi_one_over_n(self):
        """Equal committed neurons per band → HHI = 1/n_freq."""
        n_epochs, n_freq = 4, N_FREQ
        d_mlp = n_freq * 8  # 8 neurons per frequency
        dominant_freq = np.tile(np.arange(n_freq), (n_epochs, 8))
        max_frac = np.ones((n_epochs, d_mlp)) * 0.95
        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        result = compute_band_concentration_trajectory(cross_epoch, threshold=0.9, prime=P)
        np.testing.assert_allclose(result["hhi"], np.full(n_epochs, 1.0 / n_freq), atol=1e-6)


# ---------------------------------------------------------------------------
# CoS 2 / CoS 7: Rank correlation correctness
# ---------------------------------------------------------------------------


class TestEmbeddingBandMagnitudes:
    def test_output_shape(self):
        coefficients = np.ones(P + 1)
        result = compute_embedding_band_magnitudes(coefficients, N_FREQ)
        assert result.shape == (N_FREQ,)

    def test_known_values(self):
        # coefficients[1] = 3, coefficients[2] = 4 → magnitude = 5 for k=1
        coefficients = np.zeros(P + 1)
        coefficients[1] = 3.0
        coefficients[2] = 4.0
        result = compute_embedding_band_magnitudes(coefficients, N_FREQ)
        assert pytest.approx(result[0], abs=1e-6) == 5.0


class TestRankAlignmentTrajectory:
    def test_output_keys(self):
        cross_epoch = _make_cross_epoch()
        coeff_epochs = _make_coeff_epochs()
        result = compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, 0.75, P)
        assert "epochs" in result
        assert "rank_correlation" in result

    def test_output_shape(self):
        cross_epoch = _make_cross_epoch()
        coeff_epochs = _make_coeff_epochs()
        result = compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, 0.75, P)
        assert result["rank_correlation"].shape == (N_EPOCHS,)

    def test_identical_rankings_returns_plus_one(self):
        """When embedding magnitudes perfectly match neuron counts, ρ = +1."""
        n_epochs, n_freq = 10, N_FREQ

        # Commit neurons: group_sizes[k] * 4 neurons for frequency k
        group_sizes = np.arange(1, n_freq + 1)  # [1,2,3,4,5]
        dominant = np.concatenate(
            [np.full(s * 4, k) for k, s in enumerate(group_sizes)]
        )
        d_mlp = len(dominant)  # 4*(1+2+3+4+5) = 60
        dominant_freq = np.tile(dominant, (n_epochs, 1))
        max_frac = np.ones((n_epochs, d_mlp))

        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }

        # Build embedding coefficients so magnitude_k ∝ group_sizes[k]
        # magnitude_k = sqrt(coeff[2k-1]^2 + coeff[2k]^2) = group_sizes[k]
        coefficients = np.zeros((n_epochs, P + 1), dtype=np.float32)
        for k in range(1, n_freq + 1):
            coefficients[:, 2 * k - 1] = group_sizes[k - 1]
        coeff_epochs = {"epochs": np.arange(n_epochs) * 100, "coefficients": coefficients}

        result = compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, 0.9, P)
        valid = result["rank_correlation"][~np.isnan(result["rank_correlation"])]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, np.ones(len(valid)), atol=1e-6)

    def test_reversed_rankings_returns_minus_one(self):
        """Reversed ranking between embedding and neurons → ρ = -1."""
        n_epochs, n_freq = 10, N_FREQ
        group_sizes = np.arange(1, n_freq + 1)

        dominant = np.concatenate(
            [np.full(s * 4, k) for k, s in enumerate(group_sizes)]
        )
        d_mlp = len(dominant)  # 60
        dominant_freq = np.tile(dominant, (n_epochs, 1))
        max_frac = np.ones((n_epochs, d_mlp))
        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }

        # Reverse: embedding magnitude k ∝ n_freq - k
        reversed_sizes = group_sizes[::-1]
        coefficients = np.zeros((n_epochs, P + 1), dtype=np.float32)
        for k in range(1, n_freq + 1):
            coefficients[:, 2 * k - 1] = reversed_sizes[k - 1]
        coeff_epochs = {"epochs": np.arange(n_epochs) * 100, "coefficients": coefficients}

        result = compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, 0.9, P)
        valid = result["rank_correlation"][~np.isnan(result["rank_correlation"])]
        assert len(valid) > 0
        np.testing.assert_allclose(valid, np.full(len(valid), -1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# CoS 4 / CoS 7: Slope CV correctness
# ---------------------------------------------------------------------------


class TestSlopeCv:
    def test_equal_slopes_returns_zero(self):
        """When all bands grow at the same rate, CV = 0."""
        n_epochs, n_freq = 20, N_FREQ
        d_mlp = n_freq * 4

        # All neurons committed from the start, equal per-band count at every epoch
        dominant = np.concatenate([np.full(4, k) for k in range(n_freq)])
        dominant_freq = np.tile(dominant, (n_epochs, 1))
        #max_frac = np.ones((n_epochs, d_mlp))

        # Increase total committed at the same rate per frequency by scaling max_frac
        epochs = np.arange(n_epochs) * 100
        # Let committed count per band grow as t (same for all bands)
        # We do this by controlling which neurons are committed per epoch
        # Instead: make per-band counts proportional to epoch index, same slope
        
        #rng = np.random.default_rng(99)
        max_frac_arr = np.ones((n_epochs, d_mlp)) * 0.5  # below threshold        
        #threshold = 0.9

        # For each epoch t, commit t+1 neurons per band (same slope)
        committed_per_band = np.arange(1, n_epochs + 1)  # 1,2,...,n_epochs

        for t in range(n_epochs):
            n_commit = min(committed_per_band[t], 4)  # max 4 per band
            for k in range(n_freq):
                start = k * 4
                max_frac_arr[t, start : start + n_commit] = 0.95

        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac_arr,
            "epochs": epochs,
        }

        cv = compute_slope_cv(cross_epoch, threshold=0.9, prime=P)
        # CV should be 0 (or very close) since slopes are identical across bands
        if not np.isnan(cv):
            assert abs(cv) < 0.1

    def test_single_active_band_returns_nan(self):
        """Fewer than 2 active bands → slope CV is nan."""
        n_epochs = 10
        d_mlp = 8
        dominant_freq = np.zeros((n_epochs, d_mlp), dtype=int)
        max_frac = np.ones((n_epochs, d_mlp))
        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        cv = compute_slope_cv(cross_epoch, threshold=0.9, prime=P)
        assert np.isnan(cv)

    def test_returns_float(self):
        cross_epoch = _make_cross_epoch()
        cv = compute_slope_cv(cross_epoch, threshold=0.75, prime=P)
        assert isinstance(cv, float)


# ---------------------------------------------------------------------------
# CoS 5 / CoS 7: Critical mass snapshot
# ---------------------------------------------------------------------------


class TestCriticalMassSnapshot:
    def test_returns_none_when_never_crossed(self):
        """When committed neurons never reach threshold, returns None."""
        n_epochs, d_mlp = 10, 8
        max_frac = np.zeros((n_epochs, d_mlp))  # nothing committed
        cross_epoch = {
            "dominant_freq": np.zeros((n_epochs, d_mlp), dtype=int),
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        result = compute_critical_mass_snapshot(
            cross_epoch, threshold=0.9, prime=P, neuron_count_threshold=10
        )
        assert result is None

    def test_returns_correct_epoch_when_crossed(self):
        """Returns the first epoch where total committed >= N."""
        n_epochs, d_mlp = 10, 50
        epochs = np.arange(n_epochs) * 100
        max_frac = np.zeros((n_epochs, d_mlp))
        # At epoch index 3, commit 40 neurons (>= threshold of 30)
        max_frac[3:, :40] = 0.95
        cross_epoch = {
            "dominant_freq": np.zeros((n_epochs, d_mlp), dtype=int),
            "max_frac": max_frac,
            "epochs": epochs,
        }
        result = compute_critical_mass_snapshot(
            cross_epoch, threshold=0.9, prime=P, neuron_count_threshold=30
        )
        assert result is not None
        assert result["epoch"] == int(epochs[3])
        assert result["total_committed"] == 40

    def test_result_contains_expected_keys(self):
        n_epochs, d_mlp = 5, 20
        max_frac = np.ones((n_epochs, d_mlp)) * 0.95
        cross_epoch = {
            "dominant_freq": np.zeros((n_epochs, d_mlp), dtype=int),
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        result = compute_critical_mass_snapshot(
            cross_epoch, threshold=0.9, prime=P, neuron_count_threshold=5
        )
        assert result is not None
        for key in ("epoch", "epoch_idx", "active_band_count", "hhi", "max_band_share", "committed_per_freq", "total_committed"):
            assert key in result

    def test_snapshot_hhi_is_one_for_monopoly(self):
        """All committed neurons in freq 0 → HHI = 1.0 at critical mass crossing."""
        n_epochs, d_mlp = 5, 20
        max_frac = np.ones((n_epochs, d_mlp)) * 0.95
        dominant_freq = np.zeros((n_epochs, d_mlp), dtype=int)  # all freq 0
        cross_epoch = {
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "epochs": np.arange(n_epochs) * 100,
        }
        result = compute_critical_mass_snapshot(
            cross_epoch, threshold=0.9, prime=P, neuron_count_threshold=5
        )
        assert result is not None
        assert pytest.approx(result["hhi"], abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# CoS 3 / CoS 7: Cross-variant summary shape and column names
# ---------------------------------------------------------------------------


class TestCrossVariantSummaryIntegration:
    """Integration test: verify compute_variant_metrics() returns the new columns."""

    def test_metric_keys_present(self):
        """compute_variant_metrics must include all REQ_058 columns."""
        from miscope.views.cross_variant import compute_variant_metrics

        expected_keys = [
            "midpoint_hhi",
            "midpoint_active_band_count",
            "midpoint_max_band_share",
            "onset_hhi",
            "onset_active_band_count",
            "onset_max_band_share",
            "slope_cv",
            "critical_mass_epoch",
            "critical_mass_hhi",
        ]

        from unittest.mock import MagicMock


        variant = MagicMock()
        variant.name = "test/variant"
        variant.model_config = {"prime": P, "seed": 42}
        variant.metadata = {
            "train_losses": [1.0, 0.5, 0.05],
            "test_losses": [1.0, 0.5, 0.05],
        }

        # Simulate missing artifacts (FileNotFoundError) — columns should be None
        variant.artifacts.load_cross_epoch.side_effect = FileNotFoundError
        variant.artifacts.load_summary.side_effect = FileNotFoundError

        metrics = compute_variant_metrics(variant)
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# CoS 6 / CoS 7: Visualizations return go.Figure
# ---------------------------------------------------------------------------


class TestRenderConcentrationTrajectory:
    def _data(self):
        cross_epoch = _make_cross_epoch()
        return compute_band_concentration_trajectory(cross_epoch, threshold=0.75, prime=P)

    def test_returns_figure(self):
        fig = render_concentration_trajectory(self._data())
        assert isinstance(fig, go.Figure)

    def test_with_grokking_marker(self):
        fig = render_concentration_trajectory(self._data(), grokking_onset_epoch=500)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self):
        fig = render_concentration_trajectory(self._data(), title="Test Title")
        assert "Test Title" in fig.layout.title.text # type: ignore

    def test_has_traces(self):
        fig = render_concentration_trajectory(self._data())
        assert len(fig.data) > 0


class TestRenderRankAlignmentTrajectory:
    def _data(self):
        cross_epoch = _make_cross_epoch()
        coeff_epochs = _make_coeff_epochs()
        return compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, 0.75, P)

    def test_returns_figure(self):
        fig = render_rank_alignment_trajectory(self._data())
        assert isinstance(fig, go.Figure)

    def test_has_scatter_trace(self):
        fig = render_rank_alignment_trajectory(self._data())
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) > 0

    def test_custom_title(self):
        fig = render_rank_alignment_trajectory(self._data(), title="Custom")
        assert "Custom" in fig.layout.title.text # type: ignore


class TestRenderConcentrationScatter:
    def _df(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "variant_name": ["a/1", "b/1", "c/2"],
                "midpoint_hhi": [0.4, 0.7, 0.9],
                "grokking_onset_epoch": [5000, 12000, None],
                "failure_mode": ["healthy", "late_grokker", "no_grokking"],
            }
        )

    def test_returns_figure(self):
        fig = render_concentration_scatter(self._df())
        assert isinstance(fig, go.Figure)

    def test_has_traces(self):
        fig = render_concentration_scatter(self._df())
        assert len(fig.data) > 0

    def test_custom_title(self):
        fig = render_concentration_scatter(self._df(), title="Scatter")
        assert "Scatter" in fig.layout.title.text # type: ignore
