"""Dynamic Mode Decomposition (DMD) library functions.

Originally REQ_051 (standard exact DMD on discrete-time state trajectories).
Extended under REQ_117 with windowed DMD, eigenvalue tracking across windows,
residual-driven regime detection, and per-regime DMD as a recursive pass.

The standard DMD primitive (`compute_dmd`) remains the workhorse; the
windowed and per-regime extensions compose it over time slices. Designed
for centroid class trajectories in global PCA space, but applicable to any
real-valued time series with consistent state dimensionality.

Reference: Tu et al. (2014), "On Dynamic Mode Decomposition: Theory and
Applications." Journal of Computational Dynamics.

Key outputs:
- DMD eigenvalues: oscillatory structure of centroid dynamics
- DMD modes: spatial patterns associated with each dynamical mode
- Residual norms: per-step linear prediction error
- Windowed eigenvalue trajectories: how modes evolve across training (REQ_117)
- Regime boundaries: residual-spike-driven segmentation (REQ_117)
"""

import numpy as np
from scipy.signal import find_peaks


def compute_dmd(
    trajectory: np.ndarray,
    energy_threshold: float = 0.99,
) -> dict[str, np.ndarray]:
    """Standard exact DMD on a discrete-time state trajectory.

    Constructs consecutive snapshot pairs from the trajectory, computes
    the reduced DMD operator via SVD, and extracts eigenvalues, modes,
    and per-step linear prediction residuals.

    Truncation strategy (flagged for review): energy-based, retaining
    the smallest rank r such that the top-r singular values of the
    snapshot matrix account for >= energy_threshold of total energy
    (sum of squared singular values). This is more conservative than
    a spectral gap criterion — appropriate here since we're studying
    the residual, which is sensitive to rank choice.

    Args:
        trajectory: State trajectory of shape (n_steps, state_dim).
            Rows are consecutive time steps; columns are state variables.
            Must have at least 2 steps (n_steps >= 2).
        energy_threshold: Cumulative energy fraction for SVD rank truncation.
            Default 0.99. Higher retains more modes; lower is more compressed.

    Returns:
        Dict with:
            "eigenvalues":      (n_modes,) complex128 — DMD eigenvalues λ_i
            "modes":            (state_dim, n_modes) complex128 — DMD modes Φ_i
            "amplitudes":       (n_modes,) complex128 — initial amplitudes α_i
            "residual_norms":   (n_steps-1,) float64 — per-step prediction error
            "singular_values":  (min(state_dim, n_steps-1),) float64 — SVD spectrum
            "n_modes":          scalar int64 — number of retained modes
    """
    n_steps, state_dim = trajectory.shape
    # n_pairs = n_steps - 1

    X = trajectory[:-1].T  # (state_dim, n_pairs) — "from" snapshots
    Xp = trajectory[1:].T  # (state_dim, n_pairs) — "to" snapshots

    U, s, Vt = np.linalg.svd(X, full_matrices=False)  # U: (d,k), s: (k,), Vt: (k,n)

    r = _truncation_rank(s, energy_threshold)

    U_r = U[:, :r]  # (state_dim, r)
    s_r = s[:r]  # (r,)
    Vt_r = Vt[:r, :]  # (r, n_pairs)

    # Reduced DMD operator: Ã = U_r^T X' V_r Σ_r^{-1}
    A_tilde = (U_r.T @ Xp) @ Vt_r.T @ np.diag(1.0 / s_r)  # (r, r)

    eigenvalues, W = np.linalg.eig(A_tilde)  # eigenvalues: (r,), W: (r, r)

    # Exact DMD modes: Φ = X' V_r Σ_r^{-1} W
    Phi = (Xp @ Vt_r.T) @ np.diag(1.0 / s_r) @ W  # (state_dim, r)

    # Amplitudes: project initial state onto DMD modes
    amplitudes, _, _, _ = np.linalg.lstsq(Phi, trajectory[0], rcond=None)

    residual_norms = _compute_residual_norms(trajectory, Phi, eigenvalues)

    return {
        "eigenvalues": eigenvalues,
        "modes": Phi,
        "amplitudes": amplitudes,
        "residual_norms": residual_norms,
        "singular_values": s,
        "n_modes": np.array(r, dtype=np.int64),
    }


def dmd_reconstruct(
    eigenvalues: np.ndarray,
    modes: np.ndarray,
    amplitudes: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Reconstruct a DMD trajectory from modes and eigenvalues.

    Evaluates x̂_t = Re(Σ_i α_i λ_i^t φ_i) for t = 0 ... n_steps-1.

    Args:
        eigenvalues: (n_modes,) complex — DMD eigenvalues
        modes:       (state_dim, n_modes) complex — DMD modes
        amplitudes:  (n_modes,) complex — initial amplitudes
        n_steps:     Number of time steps to reconstruct

    Returns:
        Reconstructed trajectory of shape (n_steps, state_dim), real-valued.
    """
    # Power series: lambda^t for t = 0..n_steps-1
    t_idx = np.arange(n_steps)
    # eigenvalues[:, None] ** t_idx[None, :] → (n_modes, n_steps)
    lambda_powers = eigenvalues[:, np.newaxis] ** t_idx[np.newaxis, :]  # (n_modes, n_steps)
    # Weighted modes: α_i * φ_i * λ_i^t → sum over modes → (state_dim, n_steps)
    reconstruction = modes @ (amplitudes[:, np.newaxis] * lambda_powers)
    return reconstruction.real.T  # (n_steps, state_dim)


# --- Windowed DMD (REQ_117) ---


def compute_windowed_dmd(
    trajectory: np.ndarray,
    window_size: int,
    stride: int = 1,
    energy_threshold: float = 0.99,
) -> dict[str, np.ndarray]:
    """Sliding-window DMD on a discrete-time state trajectory.

    Runs `compute_dmd` on each window slice. Per-window outputs are padded
    along the modes axis to a common `max_modes` with NaN, so the result is
    a regular ndarray suitable for npz storage.

    Args:
        trajectory: (n_steps, state_dim) real-valued state trajectory.
        window_size: Steps per window. Must be >= 2 and <= n_steps.
        stride: Step between window starts. Default 1.
        energy_threshold: SVD truncation threshold passed to `compute_dmd`.

    Returns:
        Dict with:
            "window_starts":     (n_windows,) int64 — first time index per window
            "window_ends":       (n_windows,) int64 — exclusive end index per window
            "n_modes_per_window":(n_windows,) int64 — modes retained per window
            "max_modes":         scalar int64 — max(n_modes_per_window)
            "eigenvalues":       (n_windows, max_modes) complex128, NaN-padded
            "modes":             (n_windows, state_dim, max_modes) complex128, NaN-padded
            "amplitudes":        (n_windows, max_modes) complex128, NaN-padded
            "residual_norms":    (n_windows, window_size - 1) float64 — per-step within window
            "residual_norm_mean":(n_windows,) float64 — mean over within-window residuals
            "residual_norm_max": (n_windows,) float64 — max over within-window residuals

    Raises:
        ValueError: If window_size < 2 or window_size > n_steps or stride < 1.
    """
    n_steps, state_dim = trajectory.shape
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2 (got {window_size})")
    if window_size > n_steps:
        raise ValueError(f"window_size ({window_size}) must be <= n_steps ({n_steps})")
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")

    window_starts = np.arange(0, n_steps - window_size + 1, stride, dtype=np.int64)
    n_windows = len(window_starts)
    window_ends = window_starts + window_size

    per_window: list[dict[str, np.ndarray]] = []
    for start in window_starts:
        slice_ = trajectory[start : start + window_size]
        per_window.append(compute_dmd(slice_, energy_threshold=energy_threshold))

    n_modes_per_window = np.array([int(w["n_modes"]) for w in per_window], dtype=np.int64)
    max_modes = int(n_modes_per_window.max())

    eigenvalues = np.full((n_windows, max_modes), np.nan + 0j, dtype=np.complex128)
    modes = np.full((n_windows, state_dim, max_modes), np.nan + 0j, dtype=np.complex128)
    amplitudes = np.full((n_windows, max_modes), np.nan + 0j, dtype=np.complex128)
    residual_norms = np.full((n_windows, window_size - 1), np.nan, dtype=np.float64)

    for i, w in enumerate(per_window):
        k = int(w["n_modes"])
        eigenvalues[i, :k] = w["eigenvalues"]
        modes[i, :, :k] = w["modes"]
        amplitudes[i, :k] = w["amplitudes"]
        residual_norms[i] = w["residual_norms"]

    return {
        "window_starts": window_starts,
        "window_ends": window_ends,
        "n_modes_per_window": n_modes_per_window,
        "max_modes": np.array(max_modes, dtype=np.int64),
        "eigenvalues": eigenvalues,
        "modes": modes,
        "amplitudes": amplitudes,
        "residual_norms": residual_norms,
        "residual_norm_mean": residual_norms.mean(axis=1),
        "residual_norm_max": residual_norms.max(axis=1),
    }


def track_eigenvalues_across_windows(
    eigenvalues: np.ndarray,
    n_modes_per_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Greedy nearest-neighbor matching of eigenvalues across adjacent windows.

    Assigns a stable global track ID to each eigenvalue slot so that
    consumers can plot per-mode trajectories of |λ| and arg(λ) across
    training. New IDs are issued when modes appear (current window has
    more modes than previous); previous IDs are dropped when modes
    disappear.

    Greedy NN with exclusion: at each window pair, the (current, previous)
    eigenvalue pair with the smallest distance gets matched first; both
    are marked used; repeat until one side is exhausted. Hungarian
    matching is documented as a possible upgrade in REQ_117 if the
    greedy result proves unstable on real data.

    Args:
        eigenvalues: (n_windows, max_modes) complex, NaN-padded for slots
            beyond `n_modes_per_window[w]`.
        n_modes_per_window: (n_windows,) int — actual mode count per window.

    Returns:
        Dict with:
            "track_ids":  (n_windows, max_modes) int64 — global track ID per
                          slot. Sentinel -1 for padded slots.
            "n_tracks":   scalar int64 — total distinct tracks issued.
    """
    n_windows, max_modes = eigenvalues.shape
    track_ids = np.full((n_windows, max_modes), -1, dtype=np.int64)

    if n_windows == 0:
        return {"track_ids": track_ids, "n_tracks": np.array(0, dtype=np.int64)}

    k0 = int(n_modes_per_window[0])
    track_ids[0, :k0] = np.arange(k0, dtype=np.int64)
    next_track_id = k0

    for w in range(1, n_windows):
        k_prev = int(n_modes_per_window[w - 1])
        k_curr = int(n_modes_per_window[w])
        if k_prev == 0 or k_curr == 0:
            track_ids[w, :k_curr] = np.arange(
                next_track_id, next_track_id + k_curr, dtype=np.int64
            )
            next_track_id += k_curr
            continue

        prev_eigs = eigenvalues[w - 1, :k_prev]
        curr_eigs = eigenvalues[w, :k_curr]
        prev_tracks = track_ids[w - 1, :k_prev]

        cost = np.abs(curr_eigs[:, np.newaxis] - prev_eigs[np.newaxis, :])

        used_prev = np.zeros(k_prev, dtype=bool)
        new_track_ids = np.full(k_curr, -1, dtype=np.int64)

        # Process pairs in order of ascending distance.
        flat_idx = np.argsort(cost, axis=None)
        for idx in flat_idx:
            i, j = np.unravel_index(idx, cost.shape)
            if new_track_ids[i] == -1 and not used_prev[j]:
                new_track_ids[i] = prev_tracks[j]
                used_prev[j] = True
                if (new_track_ids != -1).all() or used_prev.all():
                    break

        for i in range(k_curr):
            if new_track_ids[i] == -1:
                new_track_ids[i] = next_track_id
                next_track_id += 1

        track_ids[w, :k_curr] = new_track_ids

    return {
        "track_ids": track_ids,
        "n_tracks": np.array(next_track_id, dtype=np.int64),
    }


def detect_regime_boundaries(
    residual_norms: np.ndarray,
    threshold: float | None = None,
    min_prominence: float | None = None,
    min_segment_length: int = 1,
) -> dict[str, np.ndarray]:
    """Identify regime boundaries as prominent peaks in the residual signal.

    A boundary is a local maximum that exceeds `threshold` (height filter)
    and rises at least `min_prominence` above its neighborhood (prominence
    filter). Segments are the runs between consecutive peaks.

    Peak detection (vs. the previous rising-edge approach) handles signals
    that start above threshold and decay without re-crossing — common at
    `resid_post` for fast-grokking variants where the initial reorganization
    *is* the first regime, not a baseline.

    The default `threshold` is `median + 3·MAD`. The default `min_prominence`
    is `1 MAD` — gentle enough to preserve real signal, restrictive enough
    to filter trivial blips.

    Args:
        residual_norms: (n,) float — typically `residual_norm_mean` from
            `compute_windowed_dmd`. Must have length >= 1.
        threshold: Passed as `height` to `scipy.signal.find_peaks`.
            If None, derived from the data.
        min_prominence: Passed as `prominence` to `scipy.signal.find_peaks`.
            If None, derived from the data.
        min_segment_length: Segments shorter than this are merged with their
            longer adjacent neighbor. Default 1 (no merging).

    Returns:
        Dict with:
            "segment_starts":     (n_segments,) int64 — first window per segment
            "segment_ends":       (n_segments,) int64 — exclusive end window
            "boundary_indices":   (n_segments - 1,) int64 — window index of
                                  each detected peak
            "threshold_used":     scalar float64
            "min_prominence_used":scalar float64
            "peak_prominences":   (n_boundaries,) float64 — prominence per peak,
                                  for use as a quality signal in plots

    Raises:
        ValueError: If `residual_norms` is empty or `min_segment_length < 1`.
    """
    n = len(residual_norms)
    if n == 0:
        raise ValueError("residual_norms must be non-empty")
    if min_segment_length < 1:
        raise ValueError(f"min_segment_length must be >= 1 (got {min_segment_length})")

    median = float(np.median(residual_norms))
    mad = float(np.median(np.abs(residual_norms - median)))

    threshold_used = float(threshold) if threshold is not None else median + 3.0 * mad
    if min_prominence is None:
        # Floor at a tiny positive number so a constant signal doesn't accept
        # arbitrary numerical noise as peaks.
        min_prominence_used = max(mad, 1e-12)
    else:
        min_prominence_used = float(min_prominence)

    boundary_indices, peak_prominences = _find_residual_peaks(
        residual_norms,
        height=threshold_used,
        prominence=min_prominence_used,
    )

    segment_starts, segment_ends = _segments_from_boundaries(boundary_indices, n)

    if min_segment_length > 1:
        # Drop the corresponding prominences for any boundaries removed by
        # the merge step — easier to recompute from the new boundaries.
        segment_starts, segment_ends, new_boundaries = _merge_short_segments(
            segment_starts, segment_ends, min_segment_length
        )
        kept_mask = np.isin(boundary_indices, new_boundaries)
        boundary_indices = new_boundaries
        peak_prominences = peak_prominences[kept_mask]

    return {
        "segment_starts": segment_starts,
        "segment_ends": segment_ends,
        "boundary_indices": boundary_indices,
        "threshold_used": np.array(threshold_used, dtype=np.float64),
        "min_prominence_used": np.array(min_prominence_used, dtype=np.float64),
        "peak_prominences": peak_prominences,
    }


def _find_residual_peaks(
    residual_norms: np.ndarray,
    height: float,
    prominence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap scipy.signal.find_peaks with edge-padding so peaks at index 0 or n-1
    are detectable. Returns (peak indices in the original signal, prominences).
    """
    pad_value = float(residual_norms.min()) - 1.0
    padded = np.concatenate(([pad_value], residual_norms, [pad_value]))
    peaks, props = find_peaks(padded, height=height, prominence=prominence)
    # Translate padded indices back to original signal indices.
    boundary_indices = (peaks - 1).astype(np.int64)
    # Defensive clip: shouldn't fire, but keeps invariants if scipy returns
    # an unexpected boundary.
    valid = (boundary_indices >= 0) & (boundary_indices < len(residual_norms))
    return boundary_indices[valid], props["prominences"][valid].astype(np.float64)


def compute_per_regime_dmd(
    trajectory: np.ndarray,
    segment_starts: np.ndarray,
    segment_ends: np.ndarray,
    energy_threshold: float = 0.99,
) -> dict[str, np.ndarray]:
    """Run standard DMD inside each regime segment.

    The recursive second pass: once `detect_regime_boundaries` has identified
    where dynamics change, this fits a clean linear DMD operator inside each
    segment. Segments with fewer than 2 steps produce empty (all-NaN) slots
    since DMD needs at least one snapshot pair.

    Per-segment outputs are padded along the modes axis to a common
    `max_modes` and along the time axis to the longest segment's residual
    length, so the result is a regular ndarray suitable for npz storage.

    Args:
        trajectory: (n_steps, state_dim).
        segment_starts: (n_segments,) int — first time index per segment.
        segment_ends: (n_segments,) int — exclusive end index per segment.
        energy_threshold: SVD truncation passed to `compute_dmd`.

    Returns:
        Dict with:
            "segment_starts":     (n_segments,) int64 — copy of input
            "segment_ends":       (n_segments,) int64 — copy of input
            "n_modes_per_segment":(n_segments,) int64 — modes retained
                                  (0 for segments with length < 2)
            "max_modes":          scalar int64
            "eigenvalues":        (n_segments, max_modes) complex128, NaN-padded
            "modes":              (n_segments, state_dim, max_modes) complex128, NaN-padded
            "amplitudes":         (n_segments, max_modes) complex128, NaN-padded
            "residual_norms":     (n_segments, max_pair_count) float64, NaN-padded
            "residual_norm_mean": (n_segments,) float64 — per-segment mean
                                  (NaN for segments of length < 2)
            "residual_norm_max":  (n_segments,) float64 — per-segment max
                                  (NaN for segments of length < 2)

    Raises:
        ValueError: If `segment_starts` and `segment_ends` have mismatched
            lengths or contain invalid indices.
    """
    if segment_starts.shape != segment_ends.shape:
        raise ValueError(
            f"segment_starts {segment_starts.shape} and segment_ends "
            f"{segment_ends.shape} must have the same shape"
        )
    n_segments = len(segment_starts)
    n_steps, state_dim = trajectory.shape

    if n_segments == 0:
        return {
            "segment_starts": np.empty(0, dtype=np.int64),
            "segment_ends": np.empty(0, dtype=np.int64),
            "n_modes_per_segment": np.empty(0, dtype=np.int64),
            "max_modes": np.array(0, dtype=np.int64),
            "eigenvalues": np.empty((0, 0), dtype=np.complex128),
            "modes": np.empty((0, state_dim, 0), dtype=np.complex128),
            "amplitudes": np.empty((0, 0), dtype=np.complex128),
            "residual_norms": np.empty((0, 0), dtype=np.float64),
            "residual_norm_mean": np.empty(0, dtype=np.float64),
            "residual_norm_max": np.empty(0, dtype=np.float64),
        }

    # First pass: compute DMD per segment (skipping degenerate ones).
    per_segment: list[dict[str, np.ndarray] | None] = []
    for s, e in zip(segment_starts, segment_ends):
        if e <= s or e > n_steps or s < 0:
            raise ValueError(
                f"invalid segment [{s}, {e}) on trajectory with {n_steps} steps"
            )
        if (e - s) < 2:
            per_segment.append(None)
        else:
            per_segment.append(
                compute_dmd(trajectory[s:e], energy_threshold=energy_threshold)
            )

    n_modes_per_segment = np.array(
        [int(d["n_modes"]) if d is not None else 0 for d in per_segment],
        dtype=np.int64,
    )
    max_modes = int(n_modes_per_segment.max()) if n_modes_per_segment.size else 0
    max_pair_count = int((segment_ends - segment_starts - 1).clip(min=0).max())

    eigenvalues = np.full((n_segments, max_modes), np.nan + 0j, dtype=np.complex128)
    modes = np.full((n_segments, state_dim, max_modes), np.nan + 0j, dtype=np.complex128)
    amplitudes = np.full((n_segments, max_modes), np.nan + 0j, dtype=np.complex128)
    residual_norms = np.full((n_segments, max_pair_count), np.nan, dtype=np.float64)
    residual_norm_mean = np.full(n_segments, np.nan, dtype=np.float64)
    residual_norm_max = np.full(n_segments, np.nan, dtype=np.float64)

    for i, d in enumerate(per_segment):
        if d is None:
            continue
        k = int(d["n_modes"])
        eigenvalues[i, :k] = d["eigenvalues"]
        modes[i, :, :k] = d["modes"]
        amplitudes[i, :k] = d["amplitudes"]
        seg_residuals = d["residual_norms"]
        residual_norms[i, : len(seg_residuals)] = seg_residuals
        residual_norm_mean[i] = float(seg_residuals.mean())
        residual_norm_max[i] = float(seg_residuals.max())

    return {
        "segment_starts": segment_starts.astype(np.int64),
        "segment_ends": segment_ends.astype(np.int64),
        "n_modes_per_segment": n_modes_per_segment,
        "max_modes": np.array(max_modes, dtype=np.int64),
        "eigenvalues": eigenvalues,
        "modes": modes,
        "amplitudes": amplitudes,
        "residual_norms": residual_norms,
        "residual_norm_mean": residual_norm_mean,
        "residual_norm_max": residual_norm_max,
    }


# --- Private helpers ---


def _segments_from_boundaries(
    boundary_indices: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build segment [start, end) ranges from boundary indices over [0, n)."""
    starts = np.concatenate(([0], boundary_indices)).astype(np.int64)
    ends = np.concatenate((boundary_indices, [n])).astype(np.int64)
    return starts, ends


def _merge_short_segments(
    starts: np.ndarray, ends: np.ndarray, min_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge segments shorter than min_length with the longer adjacent neighbor."""
    starts_list = list(starts)
    ends_list = list(ends)
    changed = True
    while changed and len(starts_list) > 1:
        changed = False
        for i in range(len(starts_list)):
            length = ends_list[i] - starts_list[i]
            if length >= min_length:
                continue
            left_len = ends_list[i - 1] - starts_list[i - 1] if i > 0 else -1
            right_len = (
                ends_list[i + 1] - starts_list[i + 1] if i < len(starts_list) - 1 else -1
            )
            if right_len < 0 or (left_len >= right_len and left_len > 0):
                # Merge into left
                ends_list[i - 1] = ends_list[i]
                del starts_list[i]
                del ends_list[i]
            else:
                # Merge into right
                starts_list[i + 1] = starts_list[i]
                del starts_list[i]
                del ends_list[i]
            changed = True
            break
    new_starts = np.array(starts_list, dtype=np.int64)
    new_ends = np.array(ends_list, dtype=np.int64)
    new_boundaries = new_starts[1:].copy()
    return new_starts, new_ends, new_boundaries





def _truncation_rank(singular_values: np.ndarray, energy_threshold: float) -> int:
    """Determine SVD truncation rank by cumulative energy fraction."""
    energy = singular_values**2
    total = energy.sum()
    if total < 1e-12:
        return 1
    cumulative = np.cumsum(energy) / total
    passing = np.where(cumulative >= energy_threshold)[0]
    return int(passing[0]) + 1 if len(passing) > 0 else len(singular_values)


def _compute_residual_norms(
    trajectory: np.ndarray,
    modes: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Per-step DMD prediction residuals, vectorized over all time steps.

    For each step t: residual_t = ||x_{t+1} - Re(Φ diag(λ) Φ† x_t)||

    Args:
        trajectory: (n_steps, state_dim) real-valued state trajectory
        modes:      (state_dim, n_modes) complex DMD modes
        eigenvalues:(n_modes,) complex DMD eigenvalues

    Returns:
        residual_norms: (n_steps-1,) float64
    """
    Phi_pinv = np.linalg.pinv(modes)  # (n_modes, state_dim)

    # Project all "from" states into DMD modal coordinates
    Z = Phi_pinv @ trajectory[:-1].T  # (n_modes, n_pairs)
    # Advance one step by multiplying each mode coordinate by its eigenvalue
    Z_next = eigenvalues[:, np.newaxis] * Z  # (n_modes, n_pairs)
    # Reconstruct predicted "to" states (real part)
    X_pred = (modes @ Z_next).real  # (state_dim, n_pairs)

    # Residual norms along state dimension
    diff = trajectory[1:].T - X_pred  # (state_dim, n_pairs)
    return np.linalg.norm(diff, axis=0)  # (n_pairs,)
