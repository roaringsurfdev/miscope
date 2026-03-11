"""Tests for REQ_056: Frequency Specialization Sequencing and Threshold Analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from miscope.visualization.renderers.neuron_freq_clusters import (
    render_neuron_freq_trajectory,
    render_per_band_specialization,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cross_epoch_data(
    n_epochs: int = 5,
    d_mlp: int = 12,
    n_freq: int = 8,
    assignments: list[list[int]] | None = None,
    max_frac_value: float = 0.9,
) -> dict[str, np.ndarray]:
    """Build a minimal cross-epoch neuron_dynamics dict.

    assignments[t] is the dominant frequency for each neuron at epoch t.
    If None, neurons are spread evenly across frequencies 0..n_freq-1.
    """
    if assignments is None:
        base = [i % n_freq for i in range(d_mlp)]
        assignments = [base] * n_epochs

    dominant_freq = np.array(assignments, dtype=np.int32)  # (n_epochs, d_mlp)
    max_frac = np.full((n_epochs, d_mlp), max_frac_value, dtype=np.float32)
    epochs = np.arange(0, n_epochs * 100, 100)
    threshold = np.array([3.0 / n_freq])

    return {
        "epochs": epochs,
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# render_per_band_specialization
# ---------------------------------------------------------------------------


class TestRenderPerBandSpecialization:
    def test_returns_figure(self):
        data = _make_cross_epoch_data()
        fig = render_per_band_specialization(data, prime=17)
        assert isinstance(fig, go.Figure)

    def test_only_active_bands_shown(self):
        # Neurons split across freqs 0 and 2 only — freq 1 should not appear.
        assignments = [[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2]] * 5
        data = _make_cross_epoch_data(assignments=assignments)
        fig = render_per_band_specialization(data, prime=17)
        trace_names = {t.name for t in fig.data}
        assert "Freq 1" in trace_names  # freq 0 → "Freq 1" (1-indexed)
        assert "Freq 3" in trace_names  # freq 2 → "Freq 3"
        assert "Freq 2" not in trace_names  # freq 1 absent

    def test_threshold_filters_uncommitted_neurons(self):
        # Low max_frac — neurons do not meet a high threshold.
        data = _make_cross_epoch_data(max_frac_value=0.3)
        fig_low = render_per_band_specialization(data, prime=17, threshold=0.25)
        fig_high = render_per_band_specialization(data, prime=17, threshold=0.5)
        # At low threshold, band 0 (all neurons) should have count > 0.
        assert len(fig_low.data) > 0
        # At high threshold, no neurons committed — no traces.
        assert len(fig_high.data) == 0

    def test_uses_stored_threshold_when_none(self):
        # stored threshold is 3/8 = 0.375; max_frac=0.5 should be above it.
        data = _make_cross_epoch_data(max_frac_value=0.5, n_freq=8)
        fig = render_per_band_specialization(data, prime=17)
        assert len(fig.data) > 0

    def test_neuron_counts_correct(self):
        # 4 neurons on freq 0, 8 neurons on freq 3 at all epochs.
        d_mlp = 12
        assignments = [[0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3]] * 5
        data = _make_cross_epoch_data(assignments=assignments, d_mlp=d_mlp)
        fig = render_per_band_specialization(data, prime=17)
        trace_by_name = {t.name: t for t in fig.data}
        assert np.all(np.array(trace_by_name["Freq 1"].y) == 4)
        assert np.all(np.array(trace_by_name["Freq 4"].y) == 8)

    def test_threshold_in_title(self):
        data = _make_cross_epoch_data()
        fig = render_per_band_specialization(data, prime=17, threshold=0.8)
        assert "80%" in fig.layout.title.text

    def test_custom_title(self):
        data = _make_cross_epoch_data()
        fig = render_per_band_specialization(data, prime=17, title="My Title")
        assert fig.layout.title.text == "My Title"

    def test_band_appears_then_disappears(self):
        # freq 1 appears for first 3 epochs, then neurons switch to freq 2.
        assignments = [
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],  # epoch 0: 4 on freq 1
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],  # epoch 1
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],  # epoch 2
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # epoch 3: freq 1 gone
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # epoch 4
        ]
        data = _make_cross_epoch_data(assignments=assignments)
        fig = render_per_band_specialization(data, prime=17)
        trace_by_name = {t.name: t for t in fig.data}
        freq2_counts = np.array(trace_by_name["Freq 2"].y)
        # Should be 4 for first 3 epochs, then 0
        assert freq2_counts[0] == 4
        assert freq2_counts[3] == 0


# ---------------------------------------------------------------------------
# render_neuron_freq_trajectory — threshold override
# ---------------------------------------------------------------------------


class TestNeuronFreqTrajectoryThreshold:
    def test_threshold_override_masks_more_neurons(self):
        # With high threshold, more neurons should be masked (NaN).
        data = _make_cross_epoch_data(max_frac_value=0.5)
        # At stored threshold (3/8=0.375), max_frac=0.5 → all committed.
        fig_low = render_neuron_freq_trajectory(data, prime=17, threshold=0.3)
        # At high threshold, max_frac=0.5 < 0.8 → all masked.
        fig_high = render_neuron_freq_trajectory(data, prime=17, threshold=0.8)
        # High threshold figure should have mostly NaN in the heatmap.
        z_low = np.array(fig_low.data[0].z)
        z_high = np.array(fig_high.data[0].z)
        assert np.sum(~np.isnan(z_low)) > np.sum(~np.isnan(z_high))

    def test_no_threshold_uses_stored(self):
        # When threshold kwarg is omitted, stored threshold should be used.
        data = _make_cross_epoch_data(max_frac_value=0.9)
        fig = render_neuron_freq_trajectory(data, prime=17)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# neuron_dynamics.raw dataview (integration smoke test)
# ---------------------------------------------------------------------------


class TestNeuronDynamicsRawDataview:
    def test_dataview_registered(self):
        from miscope.views.dataview_catalog import _dataview_catalog

        names = _dataview_catalog.names()
        assert "neuron_dynamics.raw" in names

    def test_schema_fields(self):
        from miscope.views.dataview_catalog import _dataview_catalog

        dv_def = _dataview_catalog.get("neuron_dynamics.raw")
        field_names = dv_def.schema.field_names()
        assert "epochs" in field_names
        assert "dominant_freq" in field_names
        assert "max_frac" in field_names
        assert "stored_threshold" in field_names
