"""REQ_058: Neuron Band Concentration metrics.

Pure metric functions operating on numpy arrays. All functions accept
artifact data directly and are usable from notebooks without variant objects.

Public API:
    compute_band_concentration_trajectory(cross_epoch, threshold, prime)
    compute_embedding_band_magnitudes(coefficients, n_freq)
    compute_rank_alignment_trajectory(cross_epoch, coeff_epochs, threshold, prime)
    compute_slope_cv(cross_epoch, threshold, prime, grokking_onset_epoch)
    compute_critical_mass_snapshot(cross_epoch, threshold, prime, neuron_count_threshold)
    compute_band_concentration_at_epoch(cross_epoch, epoch_idx, threshold, prime)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Core concentration metrics
# ---------------------------------------------------------------------------


def compute_hhi(counts: np.ndarray) -> float:
    """Compute normalized Herfindahl-Hirschman Index for a distribution of counts.

    HHI = Σ(share_k²), where share_k = count_k / total.
    Ranges from 1/n (uniform) to 1.0 (monopoly).

    Args:
        counts: 1D array of non-negative counts across bands.

    Returns:
        HHI scalar, or nan if total is zero.
    """
    total = counts.sum()
    if total == 0:
        return float("nan")
    shares = counts / total
    return float((shares**2).sum())


def compute_band_concentration_trajectory(
    cross_epoch: dict,
    threshold: float,
    prime: int,
) -> dict:
    """Per-epoch band concentration metrics from neuron_dynamics cross-epoch data.

    Args:
        cross_epoch: neuron_dynamics artifact with keys:
            dominant_freq (n_epochs, d_mlp) — per-neuron dominant frequency index
            max_frac      (n_epochs, d_mlp) — per-neuron max Fourier fraction
            epochs        (n_epochs,)       — epoch numbers
        threshold: Specialization threshold for commitment (e.g. 0.75 or 0.9).
        prime: The prime p; n_freq = prime // 2.

    Returns:
        Dict with arrays of shape (n_epochs,):
            epochs, active_band_count, hhi, max_band_share
    """
    n_freq = prime // 2
    epochs = cross_epoch["epochs"]
    dominant_freq = cross_epoch["dominant_freq"]  # (n_epochs, d_mlp)
    max_frac = cross_epoch["max_frac"]            # (n_epochs, d_mlp)

    committed = max_frac >= threshold  # (n_epochs, d_mlp)

    per_freq = np.stack(
        [(committed & (dominant_freq == k)).sum(axis=1) for k in range(n_freq)],
        axis=1,
    ).astype(float)  # (n_epochs, n_freq)

    totals = per_freq.sum(axis=1)  # (n_epochs,)
    shares = np.where(totals[:, None] > 0, per_freq / totals[:, None], 0.0)

    hhi = (shares**2).sum(axis=1)
    hhi = np.where(totals > 0, hhi, np.nan)

    active_band_count = (per_freq > 0).sum(axis=1).astype(float)
    max_band_share = shares.max(axis=1)
    max_band_share = np.where(totals > 0, max_band_share, np.nan)

    return {
        "epochs": epochs,
        "active_band_count": active_band_count,
        "hhi": hhi,
        "max_band_share": max_band_share,
    }


def compute_band_concentration_at_epoch(
    cross_epoch: dict,
    epoch_idx: int,
    threshold: float,
    prime: int,
) -> dict:
    """Band concentration metrics at a single epoch index.

    Args:
        cross_epoch: neuron_dynamics cross-epoch artifact.
        epoch_idx: Index into the epoch dimension.
        threshold: Specialization threshold.
        prime: The prime p.

    Returns:
        Dict with scalar fields: active_band_count, hhi, max_band_share,
        committed_per_freq (array of shape n_freq).
    """
    n_freq = prime // 2
    dominant_freq = cross_epoch["dominant_freq"][epoch_idx]  # (d_mlp,)
    max_frac = cross_epoch["max_frac"][epoch_idx]            # (d_mlp,)
    committed = max_frac >= threshold

    per_freq = np.array(
        [(committed & (dominant_freq == k)).sum() for k in range(n_freq)],
        dtype=float,
    )

    return {
        "active_band_count": int((per_freq > 0).sum()),
        "hhi": compute_hhi(per_freq),
        "max_band_share": float(per_freq.max() / per_freq.sum()) if per_freq.sum() > 0 else float("nan"),
        "committed_per_freq": per_freq,
    }


# ---------------------------------------------------------------------------
# Embedding-neuron rank alignment (CoS 2)
# ---------------------------------------------------------------------------


def compute_embedding_band_magnitudes(
    coefficients: np.ndarray,
    n_freq: int,
) -> np.ndarray:
    """Extract per-band embedding Fourier magnitudes from a coefficients vector.

    Combines sin and cos components for each frequency pair:
        magnitude_k = sqrt(coeff[2k-1]^2 + coeff[2k]^2)

    Args:
        coefficients: Shape (p+1,) — output of DominantFrequenciesAnalyzer.
        n_freq: Number of frequency pairs = prime // 2.

    Returns:
        Shape (n_freq,) — per-band magnitude.
    """
    mags = np.zeros(n_freq, dtype=float)
    for k in range(1, n_freq + 1):
        sin_idx = 2 * k - 1
        cos_idx = 2 * k
        mags[k - 1] = np.sqrt(coefficients[sin_idx] ** 2 + coefficients[cos_idx] ** 2)
    return mags


def compute_rank_alignment_trajectory(
    cross_epoch: dict,
    coeff_epochs: dict,
    threshold: float,
    prime: int,
) -> dict:
    """Per-epoch Spearman rank correlation between embedding magnitudes and neuron counts.

    Only bands with at least one committed neuron are included in the correlation.
    Returns nan for epochs with fewer than 2 active bands.

    Args:
        cross_epoch: neuron_dynamics cross-epoch artifact.
        coeff_epochs: dominant_frequencies stacked artifact with keys:
            epochs        (n_epochs,)
            coefficients  (n_epochs, p+1)
        threshold: Specialization threshold.
        prime: The prime p.

    Returns:
        Dict with arrays of shape (n_epochs,): epochs, rank_correlation.
        Epochs are aligned to those present in both artifacts.
    """
    n_freq = prime // 2
    nd_epochs = cross_epoch["epochs"]
    df_epochs = coeff_epochs["epochs"]
    coeff_stack = coeff_epochs["coefficients"]  # (n_df_epochs, p+1)

    dominant_freq = cross_epoch["dominant_freq"]
    max_frac = cross_epoch["max_frac"]

    # Build index for epoch alignment
    df_epoch_to_idx = {int(e): i for i, e in enumerate(df_epochs)}

    rank_correlation = np.full(len(nd_epochs), np.nan)

    for t, epoch in enumerate(nd_epochs):
        df_idx = df_epoch_to_idx.get(int(epoch))
        if df_idx is None:
            continue

        committed = max_frac[t] >= threshold
        per_freq = np.array(
            [(committed & (dominant_freq[t] == k)).sum() for k in range(n_freq)],
            dtype=float,
        )

        active_mask = per_freq > 0
        if active_mask.sum() < 2:
            continue

        emb_mags = compute_embedding_band_magnitudes(coeff_stack[df_idx], n_freq)

        corr, _ = spearmanr(emb_mags[active_mask], per_freq[active_mask])
        rank_correlation[t] = float(corr)

    return {
        "epochs": nd_epochs,
        "rank_correlation": rank_correlation,
    }


# ---------------------------------------------------------------------------
# Band growth slope CV (CoS 4)
# ---------------------------------------------------------------------------


def compute_slope_cv(
    cross_epoch: dict,
    threshold: float,
    prime: int,
    grokking_onset_epoch: int | None = None,
) -> float:
    """Coefficient of variation of per-band committed neuron growth slopes.

    Estimates linear growth slope for each active band in the pre-grokking
    window (or full training if grokking_onset_epoch is None).

    Low CV = balanced competition across bands.
    High CV = runaway growth in one or more bands.

    Args:
        cross_epoch: neuron_dynamics cross-epoch artifact.
        threshold: Specialization threshold.
        prime: The prime p.
        grokking_onset_epoch: If provided, restrict to epochs before this value.

    Returns:
        CV scalar, or nan if fewer than 2 active bands with nonzero slope.
    """
    n_freq = prime // 2
    epochs = cross_epoch["epochs"].astype(float)
    dominant_freq = cross_epoch["dominant_freq"]
    max_frac = cross_epoch["max_frac"]

    # Restrict to pre-grokking window
    if grokking_onset_epoch is not None:
        mask = epochs <= grokking_onset_epoch
    else:
        mask = np.ones(len(epochs), dtype=bool)

    if mask.sum() < 2:
        return float("nan")

    t = epochs[mask]
    committed = max_frac[mask] >= threshold

    per_freq = np.stack(
        [(committed & (dominant_freq[mask] == k)).sum(axis=1) for k in range(n_freq)],
        axis=1,
    ).astype(float)  # (n_window, n_freq)

    # Only include bands that have at least one committed neuron
    active_bands = (per_freq > 0).any(axis=0)
    if active_bands.sum() < 2:
        return float("nan")

    slopes = []
    t_centered = t - t.mean()
    denom = (t_centered**2).sum()
    if denom == 0:
        return float("nan")

    for k in range(n_freq):
        if not active_bands[k]:
            continue
        slope = (t_centered * per_freq[:, k]).sum() / denom
        slopes.append(slope)

    slopes = np.array(slopes)
    mean_slope = slopes.mean()
    if mean_slope == 0:
        return float("nan")

    return float(slopes.std() / abs(mean_slope))


# ---------------------------------------------------------------------------
# Critical mass snapshot (CoS 5)
# ---------------------------------------------------------------------------


def compute_critical_mass_snapshot(
    cross_epoch: dict,
    threshold: float,
    prime: int,
    neuron_count_threshold: int = 100,
) -> dict | None:
    """Find the first epoch where total committed neurons reaches neuron_count_threshold.

    Args:
        cross_epoch: neuron_dynamics cross-epoch artifact.
        threshold: Specialization threshold.
        prime: The prime p.
        neuron_count_threshold: Minimum committed neuron count to trigger snapshot.

    Returns:
        None if threshold is never crossed. Otherwise a dict with:
            epoch, epoch_idx, active_band_count, hhi, max_band_share,
            committed_per_freq, total_committed.
    """
    max_frac = cross_epoch["max_frac"]  # (n_epochs, d_mlp)
    epochs = cross_epoch["epochs"]

    committed_counts = (max_frac >= threshold).sum(axis=1)  # (n_epochs,)
    crossing_indices = np.where(committed_counts >= neuron_count_threshold)[0]

    if len(crossing_indices) == 0:
        return None

    epoch_idx = int(crossing_indices[0])
    snapshot = compute_band_concentration_at_epoch(cross_epoch, epoch_idx, threshold, prime)
    snapshot["epoch"] = int(epochs[epoch_idx])
    snapshot["epoch_idx"] = epoch_idx
    snapshot["total_committed"] = int(committed_counts[epoch_idx])

    return snapshot
