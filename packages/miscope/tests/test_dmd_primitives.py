"""Tests for REQ_117 DMD primitives and the activation_dmd analyzer:
windowed DMD, eigenvalue tracking, residual-driven regime detection,
per-regime DMD recursive pass, and the analyzer that composes them.

The standard `compute_dmd` primitive (REQ_051) is tested in
test_centroid_dmd.py and is composed by the windowed and per-regime
extensions tested here.
"""

import os
import tempfile

import numpy as np
import pytest

from miscope.analysis.analyzers.activation_dmd import ActivationDMD
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.library.dmd import (
    compute_per_regime_dmd,
    compute_windowed_dmd,
    detect_regime_boundaries,
    track_eigenvalues_across_windows,
)
from miscope.analysis.protocols import CrossEpochAnalyzer

_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def _make_trajectory(n_steps: int, state_dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_steps, state_dim))


def _make_two_regime_trajectory(
    n_steps_per_regime: int = 20,
    state_dim: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Trajectory that switches dynamics halfway: stable A1 → stable A2.

    Designed to produce a clear residual spike at the regime boundary
    when DMD is fit naively across the whole sequence — but per-window
    DMD inside each half should fit cleanly.
    """
    rng = np.random.default_rng(seed)
    A1 = rng.normal(size=(state_dim, state_dim)) * 0.3
    A2 = rng.normal(size=(state_dim, state_dim)) * 0.3
    x = rng.normal(size=state_dim)
    traj = []
    for _ in range(n_steps_per_regime):
        traj.append(x)
        x = A1 @ x
    for _ in range(n_steps_per_regime):
        traj.append(x)
        x = A2 @ x
    return np.array(traj)


# ── compute_windowed_dmd ─────────────────────────────────────────────


class TestComputeWindowedDmd:
    def test_returns_required_keys(self):
        traj = _make_trajectory(20, 6)
        result = compute_windowed_dmd(traj, window_size=5)
        for key in [
            "window_starts",
            "window_ends",
            "n_modes_per_window",
            "max_modes",
            "eigenvalues",
            "modes",
            "amplitudes",
            "residual_norms",
            "residual_norm_mean",
            "residual_norm_max",
        ]:
            assert key in result, f"missing key: {key}"

    def test_n_windows_default_stride(self):
        n_steps, window_size = 20, 5
        result = compute_windowed_dmd(_make_trajectory(n_steps, 6), window_size=window_size)
        # n_windows = n_steps - window_size + 1 with stride=1
        assert len(result["window_starts"]) == n_steps - window_size + 1

    def test_n_windows_with_stride(self):
        n_steps, window_size, stride = 20, 5, 3
        result = compute_windowed_dmd(
            _make_trajectory(n_steps, 6), window_size=window_size, stride=stride
        )
        # window starts: 0, 3, 6, 9, 12, 15 (16 would need step at 20)
        expected_starts = np.arange(0, n_steps - window_size + 1, stride)
        np.testing.assert_array_equal(result["window_starts"], expected_starts)

    def test_window_ends_match_starts_plus_size(self):
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=5)
        np.testing.assert_array_equal(result["window_ends"], result["window_starts"] + 5)

    def test_eigenvalues_shape(self):
        traj = _make_trajectory(20, 6)
        result = compute_windowed_dmd(traj, window_size=5)
        n_windows = len(result["window_starts"])
        max_modes = int(result["max_modes"])
        assert result["eigenvalues"].shape == (n_windows, max_modes)

    def test_modes_shape(self):
        state_dim = 6
        traj = _make_trajectory(20, state_dim)
        result = compute_windowed_dmd(traj, window_size=5)
        n_windows = len(result["window_starts"])
        max_modes = int(result["max_modes"])
        assert result["modes"].shape == (n_windows, state_dim, max_modes)

    def test_residual_norms_shape(self):
        window_size = 5
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=window_size)
        n_windows = len(result["window_starts"])
        assert result["residual_norms"].shape == (n_windows, window_size - 1)

    def test_padding_with_nan_for_short_modes(self):
        """Windows that retain fewer than max_modes should have NaN
        in the unused mode slots."""
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=5)
        max_modes = int(result["max_modes"])
        for i, k in enumerate(result["n_modes_per_window"]):
            if int(k) < max_modes:
                assert np.isnan(result["eigenvalues"][i, int(k) :]).all()

    def test_n_modes_per_window_positive(self):
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=5)
        assert (result["n_modes_per_window"] >= 1).all()

    def test_n_modes_bounded_by_window_pairs(self):
        """A window of size W produces W-1 snapshot pairs; max modes <= W-1."""
        window_size = 5
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=window_size)
        assert (result["n_modes_per_window"] <= window_size - 1).all()

    def test_residual_norms_non_negative(self):
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=5)
        assert (result["residual_norms"] >= 0).all()

    def test_residual_mean_max_consistent(self):
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=5)
        assert (result["residual_norm_mean"] <= result["residual_norm_max"] + 1e-12).all()

    def test_two_regime_residual_spike(self):
        """Across a regime boundary, mean residual should spike."""
        traj = _make_two_regime_trajectory(n_steps_per_regime=15, state_dim=4, seed=1)
        result = compute_windowed_dmd(traj, window_size=5, stride=1)
        # Windows fully inside regime 1: starts 0..10. Boundary at step 15.
        # Windows that straddle (start in [11, 14]) should show elevated residuals
        # vs. windows fully inside a regime. Use a relaxed check: max
        # straddling residual should exceed the median in-regime residual.
        in_regime_starts = (result["window_starts"] <= 10) | (result["window_starts"] >= 16)
        straddling_starts = ~in_regime_starts
        in_regime_median = np.median(result["residual_norm_mean"][in_regime_starts])
        straddling_max = result["residual_norm_mean"][straddling_starts].max()
        assert straddling_max > in_regime_median

    def test_window_size_minimum_two(self):
        result = compute_windowed_dmd(_make_trajectory(20, 6), window_size=2)
        assert result["residual_norms"].shape[1] == 1

    def test_window_size_too_large_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            compute_windowed_dmd(_make_trajectory(10, 4), window_size=20)

    def test_window_size_below_two_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            compute_windowed_dmd(_make_trajectory(10, 4), window_size=1)

    def test_stride_below_one_raises(self):
        with pytest.raises(ValueError, match="stride"):
            compute_windowed_dmd(_make_trajectory(10, 4), window_size=3, stride=0)


# ── track_eigenvalues_across_windows ────────────────────────────────


def _padded_eigenvalues(per_window_eigs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ragged list of per-window eigenvalue arrays into a NaN-padded
    (n_windows, max_modes) array, returning also the n_modes_per_window."""
    n_modes = np.array([len(e) for e in per_window_eigs], dtype=np.int64)
    max_modes = int(n_modes.max()) if len(n_modes) > 0 else 0
    eigs = np.full((len(per_window_eigs), max_modes), np.nan + 0j, dtype=np.complex128)
    for i, e in enumerate(per_window_eigs):
        eigs[i, : len(e)] = e
    return eigs, n_modes


class TestTrackEigenvaluesAcrossWindows:
    def test_identical_eigenvalues_get_same_track_id(self):
        eigs = [
            np.array([0.5 + 0.1j, 0.9 + 0.0j]),
            np.array([0.5 + 0.1j, 0.9 + 0.0j]),
            np.array([0.5 + 0.1j, 0.9 + 0.0j]),
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        # Each column should be a constant track ID
        assert result["track_ids"][0, 0] == result["track_ids"][1, 0] == result["track_ids"][2, 0]
        assert result["track_ids"][0, 1] == result["track_ids"][1, 1] == result["track_ids"][2, 1]
        assert int(result["n_tracks"]) == 2

    def test_drifting_eigenvalues_track_through(self):
        """Slowly drifting eigenvalues should get a stable track ID."""
        eigs = [
            np.array([0.50 + 0.10j, 0.90 + 0.00j]),
            np.array([0.55 + 0.12j, 0.91 + 0.02j]),
            np.array([0.60 + 0.14j, 0.92 + 0.04j]),
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        ids = result["track_ids"]
        # Slot 0 follows the first eigenvalue; slot 1 follows the second.
        assert ids[0, 0] == ids[1, 0] == ids[2, 0]
        assert ids[0, 1] == ids[1, 1] == ids[2, 1]
        assert ids[0, 0] != ids[0, 1]
        assert int(result["n_tracks"]) == 2

    def test_swapped_order_still_tracks_correctly(self):
        """If row order swaps but values stay close, tracks should follow."""
        eigs = [
            np.array([0.50 + 0.10j, 0.90 + 0.00j]),
            np.array([0.91 + 0.02j, 0.55 + 0.12j]),  # swapped
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        # Track ID at window 1 slot 0 should match window 0 slot 1
        assert result["track_ids"][1, 0] == result["track_ids"][0, 1]
        assert result["track_ids"][1, 1] == result["track_ids"][0, 0]

    def test_new_mode_appears_gets_fresh_id(self):
        eigs = [
            np.array([0.5 + 0.1j]),
            np.array([0.5 + 0.1j, 0.9 + 0.0j]),
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        # First slot keeps track 0; new mode gets a fresh ID
        assert result["track_ids"][0, 0] == 0
        assert result["track_ids"][1, 0] == 0
        assert result["track_ids"][1, 1] == 1
        assert int(result["n_tracks"]) == 2

    def test_disappearing_mode_id_not_reused(self):
        eigs = [
            np.array([0.5 + 0.1j, 0.9 + 0.0j]),
            np.array([0.5 + 0.1j]),  # second mode disappeared
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        # Padded slot in window 1 stays sentinel
        assert result["track_ids"][1, 1] == -1
        # Track 0 persists; track 1 is unused but counted
        assert int(result["n_tracks"]) == 2

    def test_padding_slots_sentinel(self):
        eigs = [
            np.array([0.5 + 0.1j, 0.9 + 0.0j, 0.7 + 0.3j]),
            np.array([0.5 + 0.1j]),
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        # All padded slots in window 1 (slots 1 and 2) should be -1
        assert (result["track_ids"][1, 1:] == -1).all()

    def test_empty_input(self):
        eigs = np.empty((0, 0), dtype=np.complex128)
        n_modes = np.empty(0, dtype=np.int64)
        result = track_eigenvalues_across_windows(eigs, n_modes)
        assert result["track_ids"].shape == (0, 0)
        assert int(result["n_tracks"]) == 0

    def test_all_unique_track_ids_are_contiguous(self):
        """Issued track IDs should be 0, 1, 2, ... up to n_tracks-1."""
        eigs = [
            np.array([0.1 + 0j, 0.2 + 0j]),
            np.array([0.1 + 0j, 0.2 + 0j, 0.3 + 0j]),  # new mode
            np.array([0.1 + 0j, 0.3 + 0j, 0.4 + 0j]),  # another new mode, one drops
        ]
        e, k = _padded_eigenvalues(eigs)
        result = track_eigenvalues_across_windows(e, k)
        observed = set(int(t) for t in result["track_ids"].flatten() if int(t) != -1)
        assert observed == set(range(int(result["n_tracks"])))

    def test_compose_with_windowed_dmd(self):
        """End-to-end: windowed DMD output feeds tracking primitive cleanly."""
        traj = _make_trajectory(20, 6, seed=7)
        dmd = compute_windowed_dmd(traj, window_size=5)
        result = track_eigenvalues_across_windows(dmd["eigenvalues"], dmd["n_modes_per_window"])
        n_windows = len(dmd["window_starts"])
        max_modes = int(dmd["max_modes"])
        assert result["track_ids"].shape == (n_windows, max_modes)
        assert int(result["n_tracks"]) >= max_modes  # at least one track per slot


# ── detect_regime_boundaries ────────────────────────────────────────


class TestDetectRegimeBoundaries:
    def test_returns_required_keys(self):
        result = detect_regime_boundaries(np.array([0.1, 0.2, 0.3, 0.4]))
        for key in [
            "segment_starts",
            "segment_ends",
            "boundary_indices",
            "threshold_used",
            "min_prominence_used",
            "peak_prominences",
        ]:
            assert key in result

    def test_no_spike_one_segment(self):
        """Flat low signal: one segment covering everything."""
        signal = np.ones(10) * 0.1
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.1)
        assert len(result["segment_starts"]) == 1
        assert result["segment_starts"][0] == 0
        assert result["segment_ends"][0] == 10
        assert len(result["boundary_indices"]) == 0

    def test_single_spike_two_segments(self):
        """One clear spike → two segments with the peak as the boundary."""
        signal = np.array([0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=1.0)
        assert len(result["segment_starts"]) == 2
        assert result["boundary_indices"][0] == 3

    def test_multiple_spikes_multiple_segments(self):
        signal = np.array([0.1, 5.0, 0.1, 0.1, 5.0, 0.1, 0.1, 5.0, 0.1])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=1.0)
        assert len(result["segment_starts"]) == 4
        np.testing.assert_array_equal(
            result["boundary_indices"], np.array([1, 4, 7], dtype=np.int64)
        )

    def test_signal_starts_above_threshold_detects_peak(self):
        """Problem A regression: a signal that starts high and decays without
        re-crossing threshold must still produce a boundary at the peak."""
        signal = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.5)
        # Peak at index 0 should be detected (rising-edge logic would have missed it)
        assert 0 in result["boundary_indices"]

    def test_signal_ends_above_threshold_detects_peak(self):
        """Mirror of the start-high case: peak at the last index."""
        signal = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.5)
        assert (len(signal) - 1) in result["boundary_indices"]

    def test_contiguous_spike_run_one_boundary(self):
        """A flat plateau above threshold has one peak (find_peaks picks an
        interior point of the plateau)."""
        signal = np.array([0.1, 0.1, 5.0, 5.0, 5.0, 0.1, 0.1])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=1.0)
        assert len(result["segment_starts"]) == 2
        # The exact peak position inside a flat plateau is implementation-
        # defined; we only require it's somewhere on the plateau.
        assert int(result["boundary_indices"][0]) in (2, 3, 4)

    def test_segments_partition_the_input(self):
        """Segments must cover [0, n) exactly with no overlaps or gaps."""
        signal = np.array([0.1, 5.0, 0.1, 0.1, 5.0, 0.1, 5.0, 0.1])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=1.0)
        starts = result["segment_starts"]
        ends = result["segment_ends"]
        assert starts[0] == 0
        assert ends[-1] == len(signal)
        np.testing.assert_array_equal(starts[1:], ends[:-1])

    def test_default_threshold_is_robust(self):
        """Default threshold (median + 3 MAD) ignores small fluctuations."""
        rng = np.random.default_rng(0)
        signal = np.abs(rng.normal(loc=1.0, scale=0.05, size=50))
        signal[25] = 10.0  # inject a clear spike
        result = detect_regime_boundaries(signal, threshold=None, min_prominence=None)
        assert 25 in result["boundary_indices"]
        assert len(result["boundary_indices"]) == 1

    def test_caller_supplied_threshold_overrides(self):
        signal = np.array([0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1])
        result_high = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.05)
        result_low = detect_regime_boundaries(signal, threshold=0.3, min_prominence=0.05)
        assert len(result_low["boundary_indices"]) > len(result_high["boundary_indices"])

    def test_threshold_used_reported(self):
        result = detect_regime_boundaries(
            np.array([0.1, 1.0, 0.1]), threshold=0.5, min_prominence=0.1
        )
        assert float(result["threshold_used"]) == pytest.approx(0.5)
        assert float(result["min_prominence_used"]) == pytest.approx(0.1)

    def test_peak_prominences_returned_per_boundary(self):
        signal = np.array([0.1, 0.1, 3.0, 0.1, 0.1, 8.0, 0.1, 0.1])
        result = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.5)
        assert result["peak_prominences"].shape == result["boundary_indices"].shape
        # The bigger spike should have higher prominence
        idx_big = list(result["boundary_indices"]).index(5)
        idx_small = list(result["boundary_indices"]).index(2)
        assert result["peak_prominences"][idx_big] > result["peak_prominences"][idx_small]

    def test_min_prominence_filters_low_prominence_peaks(self):
        """A small bump on top of a high baseline should be filtered when
        min_prominence is large enough, even if it exceeds the height
        threshold."""
        signal = np.array([5.0, 5.0, 5.1, 5.0, 5.0, 8.0, 5.0, 5.0])
        # Both 5.1 and 8.0 are above threshold=1.0, but 5.1's prominence is
        # only ~0.1 while 8.0's is ~3.0
        result_strict = detect_regime_boundaries(signal, threshold=1.0, min_prominence=1.0)
        result_loose = detect_regime_boundaries(signal, threshold=1.0, min_prominence=0.05)
        assert len(result_strict["boundary_indices"]) == 1  # only the big peak
        assert len(result_loose["boundary_indices"]) == 2  # both

    def test_min_segment_length_merges_short(self):
        """A short segment between peaks should merge with neighbor."""
        signal = np.array([0.1, 0.1, 0.1, 5.0, 0.1, 5.0, 0.1, 0.1, 0.1])
        result = detect_regime_boundaries(
            signal, threshold=1.0, min_prominence=1.0, min_segment_length=3
        )
        for s, e in zip(result["segment_starts"], result["segment_ends"]):
            assert (e - s) >= 3

    def test_min_segment_length_keeps_prominences_aligned(self):
        """After segment merging, the surviving boundaries' prominences must
        still be aligned 1-to-1 with the boundaries."""
        signal = np.array([0.1, 0.1, 0.1, 5.0, 0.1, 5.0, 0.1, 0.1, 0.1])
        result = detect_regime_boundaries(
            signal, threshold=1.0, min_prominence=1.0, min_segment_length=3
        )
        assert result["peak_prominences"].shape == result["boundary_indices"].shape

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            detect_regime_boundaries(np.array([]))

    def test_min_segment_length_zero_raises(self):
        with pytest.raises(ValueError, match="min_segment_length"):
            detect_regime_boundaries(np.array([0.1, 0.2]), min_segment_length=0)

    def test_compose_with_windowed_dmd(self):
        """End-to-end: windowed-DMD residual feeds regime detection cleanly.

        Uses an explicit threshold halfway between the median and max of the
        residual signal — this isolates the plumbing test from the default
        threshold heuristic (which has its own dedicated test).
        """
        traj = _make_two_regime_trajectory(n_steps_per_regime=15, state_dim=4, seed=2)
        dmd = compute_windowed_dmd(traj, window_size=5)
        signal = dmd["residual_norm_mean"]
        threshold = float(np.median(signal)) + 0.5 * (
            float(signal.max()) - float(np.median(signal))
        )
        regimes = detect_regime_boundaries(signal, threshold=threshold)
        assert len(regimes["boundary_indices"]) >= 1


# ── compute_per_regime_dmd ──────────────────────────────────────────


class TestComputePerRegimeDmd:
    def test_returns_required_keys(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 10], dtype=np.int64)
        ends = np.array([10, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        for key in [
            "segment_starts",
            "segment_ends",
            "n_modes_per_segment",
            "max_modes",
            "eigenvalues",
            "modes",
            "amplitudes",
            "residual_norms",
            "residual_norm_mean",
            "residual_norm_max",
        ]:
            assert key in result

    def test_n_segments_matches_input(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 5, 12], dtype=np.int64)
        ends = np.array([5, 12, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        assert len(result["segment_starts"]) == 3
        assert len(result["n_modes_per_segment"]) == 3

    def test_eigenvalues_shape(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 10], dtype=np.int64)
        ends = np.array([10, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        n_segments = 2
        max_modes = int(result["max_modes"])
        assert result["eigenvalues"].shape == (n_segments, max_modes)

    def test_modes_shape(self):
        state_dim = 6
        traj = _make_trajectory(20, state_dim)
        starts = np.array([0, 10], dtype=np.int64)
        ends = np.array([10, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        max_modes = int(result["max_modes"])
        assert result["modes"].shape == (2, state_dim, max_modes)

    def test_short_segment_skipped(self):
        """A segment of length 1 produces all-NaN slot, n_modes=0."""
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 5, 6], dtype=np.int64)  # second segment is length 1
        ends = np.array([5, 6, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        assert int(result["n_modes_per_segment"][1]) == 0
        assert np.isnan(result["eigenvalues"][1]).all()
        assert np.isnan(result["residual_norm_mean"][1])

    def test_per_regime_residual_below_global(self):
        """Per-regime DMD on a two-regime trajectory should produce lower
        residuals than a single global DMD — that's the whole point.

        Construct a dramatic regime change: slow-decay followed by a magnitude
        jump and fast-decay. Global DMD sees the discontinuity as a residual
        spike; per-regime DMD splits the trajectory and fits each half cleanly.
        """
        from miscope.analysis.library.dmd import compute_dmd

        rng = np.random.default_rng(0)
        state_dim = 4
        A1 = np.eye(state_dim) * 0.9
        A2 = np.eye(state_dim) * 0.3

        x = rng.normal(size=state_dim)
        traj1 = []
        for _ in range(20):
            traj1.append(x.copy())
            x = A1 @ x
        x = rng.normal(size=state_dim) * 5.0  # magnitude jump at boundary
        traj2 = []
        for _ in range(20):
            traj2.append(x.copy())
            x = A2 @ x

        traj = np.array(traj1 + traj2)
        global_dmd = compute_dmd(traj, energy_threshold=0.99)
        global_mean_residual = float(global_dmd["residual_norms"].mean())

        starts = np.array([0, 20], dtype=np.int64)
        ends = np.array([20, 40], dtype=np.int64)
        per_regime = compute_per_regime_dmd(traj, starts, ends, energy_threshold=0.99)
        per_regime_mean_residual = float(np.nanmean(per_regime["residual_norm_mean"]))

        assert per_regime_mean_residual < global_mean_residual

    def test_padding_with_nan_for_short_modes(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 8], dtype=np.int64)
        ends = np.array([8, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        max_modes = int(result["max_modes"])
        for i, k in enumerate(result["n_modes_per_segment"]):
            if int(k) < max_modes:
                assert np.isnan(result["eigenvalues"][i, int(k) :]).all()

    def test_residual_norm_mean_max_consistent(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 10], dtype=np.int64)
        ends = np.array([10, 20], dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        assert (result["residual_norm_mean"] <= result["residual_norm_max"] + 1e-12).all()

    def test_empty_segments(self):
        traj = _make_trajectory(20, 6)
        starts = np.empty(0, dtype=np.int64)
        ends = np.empty(0, dtype=np.int64)
        result = compute_per_regime_dmd(traj, starts, ends)
        assert len(result["segment_starts"]) == 0

    def test_invalid_segment_raises(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 25], dtype=np.int64)  # past end
        ends = np.array([10, 30], dtype=np.int64)
        with pytest.raises(ValueError, match="invalid segment"):
            compute_per_regime_dmd(traj, starts, ends)

    def test_mismatched_shapes_raises(self):
        traj = _make_trajectory(20, 6)
        starts = np.array([0, 10], dtype=np.int64)
        ends = np.array([5, 10, 20], dtype=np.int64)
        with pytest.raises(ValueError, match="must have the same shape"):
            compute_per_regime_dmd(traj, starts, ends)

    def test_compose_with_regime_detection(self):
        """End-to-end: detect boundaries, run per-regime DMD on the segments."""
        traj = _make_two_regime_trajectory(n_steps_per_regime=15, state_dim=4, seed=4)
        dmd = compute_windowed_dmd(traj, window_size=5)
        signal = dmd["residual_norm_mean"]
        threshold = float(np.median(signal)) + 0.5 * (
            float(signal.max()) - float(np.median(signal))
        )
        regimes = detect_regime_boundaries(signal, threshold=threshold)
        # Translate window-space segments back to step-space using window_starts
        # (a real analyzer would do this; the test just sanity-checks the chain).
        step_starts = dmd["window_starts"][regimes["segment_starts"]]
        step_ends = np.concatenate(
            [
                dmd["window_starts"][regimes["segment_starts"][1:]],
                np.array([len(traj)], dtype=np.int64),
            ]
        )
        per_regime = compute_per_regime_dmd(traj, step_starts, step_ends)
        assert len(per_regime["segment_starts"]) == len(regimes["segment_starts"])


# ── ActivationDMD analyzer ──────────────────────────────────────────


def _make_global_pca_artifact(
    n_epochs: int = 30,
    n_classes: int = 7,
    n_components: int = 3,
    seed: int = 0,
) -> dict:
    """Synthetic global_centroid_pca cross_epoch artifact."""
    rng = np.random.default_rng(seed)
    data: dict = {"epochs": np.arange(n_epochs) * 100}
    for site in _SITES:
        data[f"{site}__projections"] = rng.normal(size=(n_epochs, n_classes, n_components))
        data[f"{site}__basis"] = rng.normal(size=(16, n_components))
        data[f"{site}__mean"] = rng.normal(size=(16,))
        data[f"{site}__explained_variance_ratio"] = np.array([0.70, 0.20, 0.09])
    return data


@pytest.fixture
def artifacts_with_global_pca():
    """Temp artifacts dir with global_centroid_pca cross_epoch.npz and
    repr_geometry per-epoch stubs (the requires-check looks for those)."""
    n_epochs, n_classes, n_components = 30, 7, 3
    pca_data = _make_global_pca_artifact(n_epochs, n_classes, n_components)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        pca_dir = os.path.join(artifacts_dir, "global_centroid_pca")
        rg_dir = os.path.join(artifacts_dir, "repr_geometry")
        os.makedirs(pca_dir)
        os.makedirs(rg_dir)
        for epoch in pca_data["epochs"]:
            p = os.path.join(rg_dir, f"epoch_{int(epoch):05d}.npz")
            np.savez_compressed(p, dummy=np.array([0]))
        np.savez_compressed(  # type: ignore[arg-type]
            os.path.join(pca_dir, "cross_epoch.npz"), **pca_data
        )
        yield artifacts_dir, list(pca_data["epochs"].astype(int)), n_epochs, n_classes, n_components


class TestActivationDMDProtocol:
    def test_conforms_to_cross_epoch_protocol(self):
        assert isinstance(ActivationDMD(), CrossEpochAnalyzer)

    def test_name(self):
        assert ActivationDMD().name == "activation_dmd"

    def test_requires(self):
        assert ActivationDMD().requires == ["repr_geometry"]

    def test_registered_in_registry(self):
        assert "activation_dmd" in AnalyzerRegistry._cross_epoch_analyzers

    def test_distinct_from_centroid_dmd(self):
        """Parallel construction: both analyzers exist independently."""
        assert "centroid_dmd" in AnalyzerRegistry._cross_epoch_analyzers
        assert "activation_dmd" in AnalyzerRegistry._cross_epoch_analyzers
        assert (
            AnalyzerRegistry._cross_epoch_analyzers["centroid_dmd"]
            is not AnalyzerRegistry._cross_epoch_analyzers["activation_dmd"]
        )


class TestActivationDMDOutput:
    def test_returns_dict(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        assert isinstance(result, dict)

    def test_contains_epochs(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        np.testing.assert_array_equal(result["epochs"], epochs)

    def test_contains_namespaced_keys_per_site(self, artifacts_with_global_pca):
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            for stage_key in [
                f"{site}__trajectory",
                f"{site}__n_classes",
                f"{site}__n_components",
                f"{site}__windowed__eigenvalues",
                f"{site}__windowed__residual_norm_mean",
                f"{site}__tracks__track_ids",
                f"{site}__tracks__n_tracks",
                f"{site}__regimes__segment_starts",
                f"{site}__regimes__segment_ends",
                f"{site}__regimes__threshold_used",
                f"{site}__per_regime__eigenvalues",
                f"{site}__per_regime__residual_norm_mean",
            ]:
                assert stage_key in result, f"missing: {stage_key}"

    def test_trajectory_shape(self, artifacts_with_global_pca):
        artifacts_dir, epochs, n_epochs, n_classes, n_components = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        state_dim = n_classes * n_components
        for site in _SITES:
            assert result[f"{site}__trajectory"].shape == (n_epochs, state_dim)

    def test_windowed_dmd_shapes(self, artifacts_with_global_pca):
        artifacts_dir, epochs, n_epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        # Default window_size = 10, stride = 1 -> n_windows = n_epochs - 10 + 1 = 21
        expected_n_windows = n_epochs - 10 + 1
        for site in _SITES:
            assert len(result[f"{site}__windowed__window_starts"]) == expected_n_windows
            assert result[f"{site}__windowed__residual_norm_mean"].shape == (expected_n_windows,)

    def test_short_trajectory_scales_window(self, artifacts_with_global_pca, tmp_path):
        """If n_epochs < 10, window_size is scaled down so the analyzer
        still runs (rather than raising)."""
        short_data = _make_global_pca_artifact(n_epochs=5)
        artifacts_dir = str(tmp_path / "artifacts")
        pca_dir = os.path.join(artifacts_dir, "global_centroid_pca")
        rg_dir = os.path.join(artifacts_dir, "repr_geometry")
        os.makedirs(pca_dir)
        os.makedirs(rg_dir)
        for epoch in short_data["epochs"]:
            p = os.path.join(rg_dir, f"epoch_{int(epoch):05d}.npz")
            np.savez_compressed(p, dummy=np.array([0]))
        np.savez_compressed(  # type: ignore[arg-type]
            os.path.join(pca_dir, "cross_epoch.npz"), **short_data
        )

        epochs_list = list(short_data["epochs"].astype(int))
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs_list, {})
        for site in _SITES:
            assert len(result[f"{site}__windowed__window_starts"]) == 1

    def test_regime_segments_partition_window_space(self, artifacts_with_global_pca):
        """Detected regime segments must partition [0, n_windows) exactly."""
        artifacts_dir, epochs, n_epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        n_windows = n_epochs - 10 + 1
        for site in _SITES:
            starts = result[f"{site}__regimes__segment_starts"]
            ends = result[f"{site}__regimes__segment_ends"]
            assert starts[0] == 0
            assert ends[-1] == n_windows
            np.testing.assert_array_equal(starts[1:], ends[:-1])

    def test_per_regime_dmd_segments_match_regimes(self, artifacts_with_global_pca):
        """Per-regime DMD must run on the same number of segments as
        regime detection produced."""
        artifacts_dir, epochs, *_ = artifacts_with_global_pca
        result = ActivationDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for site in _SITES:
            n_regimes = len(result[f"{site}__regimes__segment_starts"])
            n_per_regime = len(result[f"{site}__per_regime__segment_starts"])
            assert n_regimes == n_per_regime

    def test_missing_global_pca_raises(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        rg_dir = os.path.join(artifacts_dir, "repr_geometry")
        os.makedirs(rg_dir)
        np.savez_compressed(os.path.join(rg_dir, "epoch_00000.npz"), dummy=np.array([0]))
        with pytest.raises(FileNotFoundError, match="global_centroid_pca"):
            ActivationDMD().analyze_across_epochs(artifacts_dir, [0], {})
