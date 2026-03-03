"""REQ_042: Neuron dynamics cross-epoch analyzer.

Consumes neuron_freq_norm per-epoch artifacts and produces
per-neuron frequency trajectory metrics: dominant frequency over time,
switch counts, and commitment epochs.
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader


class NeuronDynamicsAnalyzer:
    """Cross-epoch analyzer for neuron frequency dynamics.

    Precomputes per-neuron dominant frequency trajectories, frequency
    switch counts, and commitment epochs so the dashboard can render
    neuron dynamics visualizations without loading all per-epoch data.
    """

    name = "neuron_dynamics"
    requires = ["neuron_freq_norm"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute neuron frequency dynamics across all epochs."""
        loader = ArtifactLoader(artifacts_dir)
        stacked = loader.load_epochs("neuron_freq_norm", epochs)

        norm_matrix = stacked["norm_matrix"]  # (n_epochs, n_freq, d_mlp)
        n_epochs, n_freq, d_mlp = norm_matrix.shape

        # Uncommitted threshold: 3× uniform baseline
        threshold = 3.0 / n_freq

        # Per-neuron dominant frequency and max frac at each epoch
        dominant_freq = np.argmax(norm_matrix, axis=1)  # (n_epochs, d_mlp)
        max_frac = np.max(norm_matrix, axis=1)  # (n_epochs, d_mlp)

        # Switch counts: times each neuron changes dominant frequency
        switch_counts = _compute_switch_counts(dominant_freq, max_frac, threshold)

        # Commitment epochs: when each neuron locks into its final frequency
        commitment_epochs = _compute_commitment_epochs(
            dominant_freq, max_frac, np.array(epochs), threshold
        )

        return {
            "epochs": np.array(epochs),
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "switch_counts": switch_counts,
            "commitment_epochs": commitment_epochs,
            "threshold": np.array([threshold]),
        }


def _compute_switch_counts(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Count how many times each neuron changes its dominant frequency.

    Only counts transitions between committed states (above threshold).
    """
    n_epochs, d_mlp = dominant_freq.shape
    switch_counts = np.zeros(d_mlp, dtype=np.int32)
    last_freq = np.full(d_mlp, -1, dtype=np.int32)

    for t in range(n_epochs):
        committed = max_frac[t] >= threshold
        changed = committed & (last_freq >= 0) & (dominant_freq[t] != last_freq)
        switch_counts[changed] += 1
        last_freq[committed] = dominant_freq[t, committed]

    return switch_counts


def _compute_commitment_epochs(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    epochs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Find the epoch at which each neuron commits to its final frequency.

    A neuron is "committed" when it holds the same dominant frequency
    (above threshold) from some epoch through to the end of training.
    """
    n_epochs, d_mlp = dominant_freq.shape
    commitment_epochs = np.full(d_mlp, np.nan)
    final_freq = dominant_freq[-1]

    for n in range(d_mlp):
        if max_frac[-1, n] < threshold:
            continue

        # Walk backward to find earliest stable point
        stable_from = n_epochs - 1
        for t in range(n_epochs - 2, -1, -1):
            if max_frac[t, n] >= threshold and dominant_freq[t, n] == final_freq[n]:
                stable_from = t
            else:
                break

        commitment_epochs[n] = epochs[stable_from]

    return commitment_epochs
