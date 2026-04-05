"""Tests for REQ_084: Transient Frequency Analyzer and renderers."""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.analyzers.transient_frequency import (
    TransientFrequencyAnalyzer,
    _compute_committed_counts,
    _pack_ragged,
    load_peak_members,
)
from miscope.visualization.renderers.transient_frequency import (
    render_transient_committed_counts,
    render_transient_pc1_cohesion,
    render_transient_peak_scatter,
)

# ── Fixtures ───────────────────────────────────────────────────────────


def _make_neuron_dynamics(
    n_epochs: int,
    n_freq: int,
    d_mlp: int,
    dominant_freq: np.ndarray | None = None,
    max_frac: np.ndarray | None = None,
    epochs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build a minimal neuron_dynamics artifact dict."""
    rng = np.random.default_rng(0)
    if dominant_freq is None:
        dominant_freq = rng.integers(0, n_freq, size=(n_epochs, d_mlp)).astype(np.int32)
    if max_frac is None:
        max_frac = rng.uniform(0.5, 1.0, size=(n_epochs, d_mlp)).astype(np.float32)
    if epochs is None:
        epochs = np.arange(n_epochs, dtype=np.int32) * 100
    return {
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "epochs": epochs,
        "switch_counts": np.zeros(d_mlp, dtype=np.int32),
        "commitment_epochs": np.zeros(d_mlp),
        "threshold": np.array([0.3]),
    }


@pytest.fixture
def artifacts_with_transient():
    """Temp artifacts dir with a neuron_dynamics cross_epoch.npz.

    Setup:
    - 4 epochs, 5 frequencies, 20 neurons
    - Freq 0 (transient): peaks at epoch 1 with 6 neurons (>5% of 20), absent at final
    - Freq 1 (persistent): present at final with 10 neurons (>10% of 20)
    - Freq 2 (too small): peak of 1 neuron — should NOT qualify
    - Remaining neurons scattered across freq 3, 4
    """
    n_epochs, n_freq, d_mlp = 4, 5, 20
    epochs = np.array([0, 100, 200, 300], dtype=np.int32)

    # dominant_freq: (n_epochs, d_mlp)
    dominant_freq = np.zeros((n_epochs, d_mlp), dtype=np.int32)
    # Epoch 0: freq 0 has 6 neurons (indices 0-5), freq 1 has 10 (6-15), rest on 3
    dominant_freq[0, :6] = 0
    dominant_freq[0, 6:16] = 1
    dominant_freq[0, 16:] = 3

    # Epoch 1 (peak for freq 0): same as epoch 0
    dominant_freq[1] = dominant_freq[0].copy()

    # Epoch 2: freq 0 neurons start moving to freq 1 and 4
    dominant_freq[2, :3] = 1  # 3 of the freq-0 neurons move to freq 1
    dominant_freq[2, 3:6] = 4  # 3 move to freq 4
    dominant_freq[2, 6:16] = 1
    dominant_freq[2, 16:] = 3

    # Epoch 3 (final): freq 0 is gone; freq 1 has 13 neurons (>10%), others scattered
    dominant_freq[3, :3] = 1
    dominant_freq[3, 3:6] = 4
    dominant_freq[3, 6:16] = 1
    dominant_freq[3, 16:] = 3

    # All neurons well-specialized throughout
    max_frac = np.full((n_epochs, d_mlp), 0.85, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        nd_dir = os.path.join(tmpdir, "neuron_dynamics")
        os.makedirs(nd_dir)
        nd_data = _make_neuron_dynamics(
            n_epochs,
            n_freq,
            d_mlp,
            dominant_freq=dominant_freq,
            max_frac=max_frac,
            epochs=epochs,
        )
        np.savez_compressed(os.path.join(nd_dir, "cross_epoch.npz"), **nd_data)  # type: ignore
        yield tmpdir, n_epochs, n_freq, d_mlp, epochs, dominant_freq, max_frac


# ── Unit tests: internal helpers ───────────────────────────────────────


class TestComputeCommittedCounts:
    def test_shape(self):
        dom = np.array([[0, 1, 2], [1, 1, 2]], dtype=np.int32)
        frac = np.full((2, 3), 0.9, dtype=np.float32)
        result = _compute_committed_counts(dom, frac, n_freq=3, neuron_threshold=0.7)
        assert result.shape == (2, 3)

    def test_counts_correct(self):
        """Counts committed neurons per frequency per epoch."""
        dom = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
        frac = np.full((2, 3), 0.9, dtype=np.float32)
        result = _compute_committed_counts(dom, frac, n_freq=2, neuron_threshold=0.7)
        np.testing.assert_array_equal(result[0], [2, 1])
        np.testing.assert_array_equal(result[1], [1, 2])

    def test_below_threshold_excluded(self):
        """Neurons below neuron_threshold are not counted."""
        dom = np.array([[0, 1]], dtype=np.int32)
        frac = np.array([[0.9, 0.3]], dtype=np.float32)
        result = _compute_committed_counts(dom, frac, n_freq=2, neuron_threshold=0.7)
        np.testing.assert_array_equal(result[0], [1, 0])


class TestPackRagged:
    def test_empty(self):
        flat, offsets = _pack_ragged([])
        assert len(flat) == 0
        assert len(offsets) == 1
        assert offsets[0] == 0

    def test_round_trip(self):
        arrays = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([10, 11], dtype=np.int32),
            np.array([99], dtype=np.int32),
        ]
        flat, offsets = _pack_ragged(arrays)
        assert len(flat) == 6
        assert len(offsets) == 4
        for i, arr in enumerate(arrays):
            recovered = flat[offsets[i] : offsets[i + 1]]
            np.testing.assert_array_equal(recovered, arr)


class TestLoadPeakMembers:
    def test_retrieves_correct_slice(self):
        arrays = [np.array([5, 6], dtype=np.int32), np.array([10], dtype=np.int32)]
        flat, offsets = _pack_ragged(arrays)
        artifact = {"peak_members_flat": flat, "peak_members_offsets": offsets}
        np.testing.assert_array_equal(load_peak_members(artifact, 0), [5, 6])
        np.testing.assert_array_equal(load_peak_members(artifact, 1), [10])


# ── Analyzer integration tests ─────────────────────────────────────────


class TestTransientFrequencyAnalyzer:
    def test_registration(self):
        assert "transient_frequency" in AnalyzerRegistry._cross_epoch_analyzers

    def test_output_keys(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        required_keys = [
            "ever_qualified_freqs",
            "is_final",
            "peak_epoch",
            "peak_count",
            "homeless_count",
            "committed_counts",
            "peak_members_flat",
            "peak_members_offsets",
            "epochs",
            "_neuron_threshold",
            "_transient_canonical_threshold",
            "_final_canonical_threshold",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_detects_transient_freq(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        ever_q = result["ever_qualified_freqs"].tolist()
        is_final = result["is_final"].tolist()

        # Freq 0 should be ever-qualified but not final
        assert 0 in ever_q
        idx = ever_q.index(0)
        assert not is_final[idx]

    def test_persistent_freq_marked_final(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        ever_q = result["ever_qualified_freqs"].tolist()
        is_final = result["is_final"].tolist()

        # Freq 1 ends with 13 neurons (>10% of 20) — should be final
        assert 1 in ever_q
        idx = ever_q.index(1)
        assert is_final[idx]

    def test_too_small_freq_excluded(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        ever_q = result["ever_qualified_freqs"].tolist()
        # Freq 2 never had enough neurons to qualify
        assert 2 not in ever_q

    def test_committed_counts_shape(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        n_ever_q = len(result["ever_qualified_freqs"])
        assert result["committed_counts"].shape == (n_epochs, n_ever_q)

    def test_homeless_count_for_transient(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        ever_q = result["ever_qualified_freqs"].tolist()
        is_final = result["is_final"]
        homeless = result["homeless_count"]

        # Freq 0's peak members: neurons 0-5 (6 neurons at epoch 1)
        # At final epoch, all 6 are still well-specialized (max_frac=0.85) but on
        # different frequencies — so homeless count = 0 (they ARE committed, just
        # to other freqs). Verify: homeless only counts uncommitted neurons.
        idx = ever_q.index(0)
        assert not is_final[idx]
        assert int(homeless[idx]) == 0  # All 6 neurons remain specialized (frac=0.85)

    def test_ragged_members_decodable(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        for i in range(len(result["ever_qualified_freqs"])):
            members = load_peak_members(result, i)
            assert members.dtype == np.int32
            assert len(members) > 0
            assert members.max() < d_mlp

    def test_peak_epoch_within_range(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        for ep in result["peak_epoch"]:
            assert int(ep) in epochs.tolist()

    def test_metadata_stored(self, artifacts_with_transient):
        tmpdir, n_epochs, n_freq, d_mlp, epochs, dom, frac = artifacts_with_transient
        analyzer = TransientFrequencyAnalyzer()
        result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        assert float(result["_neuron_threshold"]) == pytest.approx(0.70)
        assert float(result["_transient_canonical_threshold"]) == pytest.approx(0.05)
        assert float(result["_final_canonical_threshold"]) == pytest.approx(0.10)

    def test_empty_result_when_no_qualified_freqs(self):
        """Returns valid empty artifact when no frequency crosses the threshold."""
        n_epochs, n_freq, d_mlp = 3, 10, 20
        epochs = np.array([0, 100, 200], dtype=np.int32)
        dom = np.zeros((n_epochs, d_mlp), dtype=np.int32)
        frac = np.full((n_epochs, d_mlp), 0.1, dtype=np.float32)  # below 0.7 threshold

        with tempfile.TemporaryDirectory() as tmpdir:
            nd_dir = os.path.join(tmpdir, "neuron_dynamics")
            os.makedirs(nd_dir)
            nd_data = _make_neuron_dynamics(
                n_epochs,
                n_freq,
                d_mlp,
                dominant_freq=dom,
                max_frac=frac,
                epochs=epochs,
            )
            np.savez_compressed(os.path.join(nd_dir, "cross_epoch.npz"), **nd_data) # type: ignore

            analyzer = TransientFrequencyAnalyzer()
            result = analyzer.analyze_across_epochs(tmpdir, epochs.tolist(), context={})

        assert len(result["ever_qualified_freqs"]) == 0
        assert result["committed_counts"].shape == (n_epochs, 0)
        assert len(result["peak_members_flat"]) == 0
        assert len(result["peak_members_offsets"]) == 1


# ── Renderer tests ─────────────────────────────────────────────────────


@pytest.fixture
def sample_transient_artifact():
    """Minimal transient_frequency artifact for renderer tests."""
    n_epochs, _d_mlp = 5, 20
    epochs = np.arange(n_epochs, dtype=np.int32) * 100

    ever_qualified = np.array([0, 1], dtype=np.int32)
    is_final = np.array([False, True])
    peak_epoch = np.array([100, 400], dtype=np.int32)
    peak_count = np.array([6, 10], dtype=np.int32)
    homeless_count = np.array([2, 0], dtype=np.int32)

    committed_counts = np.zeros((n_epochs, 2), dtype=np.int32)
    committed_counts[:, 0] = [2, 6, 4, 1, 0]  # transient rises and falls
    committed_counts[:, 1] = [2, 4, 6, 8, 10]  # persistent grows

    members0 = np.arange(6, dtype=np.int32)
    members1 = np.arange(6, 16, dtype=np.int32)
    flat, offsets = _pack_ragged([members0, members1])

    return {
        "ever_qualified_freqs": ever_qualified,
        "is_final": is_final,
        "peak_epoch": peak_epoch,
        "peak_count": peak_count,
        "homeless_count": homeless_count,
        "committed_counts": committed_counts,
        "peak_members_flat": flat,
        "peak_members_offsets": offsets,
        "epochs": epochs,
        "_neuron_threshold": np.array(0.70),
        "_transient_canonical_threshold": np.array(0.05),
        "_final_canonical_threshold": np.array(0.10),
    }


@pytest.fixture
def sample_w_in_by_epoch(sample_transient_artifact):
    """W_in snapshots keyed by epoch, matching transient artifact epochs."""
    rng = np.random.default_rng(42)
    d_model, d_mlp = 64, 20
    artifact = sample_transient_artifact
    epochs = artifact["epochs"].tolist()
    return {ep: rng.standard_normal((d_model, d_mlp)).astype(np.float32) for ep in epochs}


class TestRenderTransientCommittedCounts:
    def test_returns_figure(self, sample_transient_artifact):
        fig = render_transient_committed_counts(sample_transient_artifact, epoch=None)
        assert isinstance(fig, go.Figure)

    def test_has_traces(self, sample_transient_artifact):
        fig = render_transient_committed_counts(sample_transient_artifact, epoch=None)
        assert len(fig.data) >= 1 # type: ignore

    def test_show_persistent_false(self, sample_transient_artifact):
        fig_all = render_transient_committed_counts(
            sample_transient_artifact, epoch=None, show_persistent=True
        )
        fig_transient_only = render_transient_committed_counts(
            sample_transient_artifact, epoch=None, show_persistent=False
        )
        assert len(fig_transient_only.data) < len(fig_all.data) # type: ignore


class TestRenderTransientPeakScatter:
    def test_returns_figure(self, sample_transient_artifact, sample_w_in_by_epoch):
        data = {"transient": sample_transient_artifact, "w_in_by_epoch": sample_w_in_by_epoch}
        fig = render_transient_peak_scatter(data["transient"], data["w_in_by_epoch"], epoch=None)
        assert isinstance(fig, go.Figure)

    def test_explicit_freq(self, sample_transient_artifact, sample_w_in_by_epoch):
        fig = render_transient_peak_scatter(
            sample_transient_artifact, sample_w_in_by_epoch, epoch=None, freq=0
        )
        assert isinstance(fig, go.Figure)

    def test_non_transient_freq_returns_empty(
        self, sample_transient_artifact, sample_w_in_by_epoch
    ):
        # freq=1 is persistent (is_final=True) — should return empty figure
        fig = render_transient_peak_scatter(
            sample_transient_artifact, sample_w_in_by_epoch, epoch=None, freq=1
        )
        assert isinstance(fig, go.Figure)
        # Empty figure has an annotation, not data traces
        assert len(fig.data) == 0 # type: ignore


class TestRenderTransientPc1Cohesion:
    def test_returns_figure(self, sample_transient_artifact, sample_w_in_by_epoch):
        fig = render_transient_pc1_cohesion(
            sample_transient_artifact, sample_w_in_by_epoch, epoch=None
        )
        assert isinstance(fig, go.Figure)

    def test_has_vline_at_peak(self, sample_transient_artifact, sample_w_in_by_epoch):
        fig = render_transient_pc1_cohesion(
            sample_transient_artifact, sample_w_in_by_epoch, epoch=None, freq=0
        )
        assert isinstance(fig, go.Figure)
        # Peak epoch vline present as shape
        assert len(fig.layout.shapes) > 0 or any(hasattr(trace, "x0") for trace in fig.data)

    def test_explicit_freq(self, sample_transient_artifact, sample_w_in_by_epoch):
        fig = render_transient_pc1_cohesion(
            sample_transient_artifact, sample_w_in_by_epoch, epoch=None, freq=0
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1 # type: ignore
