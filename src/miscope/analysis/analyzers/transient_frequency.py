"""REQ_084: Transient Frequency Analyzer.

Cross-epoch analyzer that identifies frequencies which ever cross the specialized-neuron
threshold during training but are absent from the final learned set.  For each such
transient frequency, records the peak epoch, peak committed-neuron count, the neuron
cohort at peak, and the number of those neurons that fail to commit to any frequency by
the final epoch (homeless neurons).

Artifact keys produced (stored as cross_epoch.npz):
    ever_qualified_freqs       int32  (n_transient,)          0-indexed frequencies
    is_final                   bool   (n_transient,)          in final learned set?
    peak_epoch                 int32  (n_transient,)          epoch of max committed count
    peak_count                 int32  (n_transient,)          committed count at peak
    homeless_count             int32  (n_transient,)          uncommitted at final epoch
    committed_counts           int32  (n_epochs, n_transient) count trajectory per freq
    peak_members_flat          int32  (total_members,)        ragged member storage
    peak_members_offsets       int32  (n_transient + 1,)      offset into flat array
    epochs                     int32  (n_epochs,)

Threshold parameters stored in artifact metadata attrs:
    neuron_threshold            float  per-neuron max_frac gate (default 0.70)
    transient_canonical_threshold  float  fraction of d_mlp for transient detection (default 0.05)
    final_canonical_threshold   float  fraction of d_mlp for final-set classification (default 0.10)
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader

NEURON_THRESHOLD: float = 0.70
TRANSIENT_CANONICAL_THRESHOLD: float = 0.05
FINAL_CANONICAL_THRESHOLD: float = 0.10


class TransientFrequencyAnalyzer:
    """Identifies transient frequency groups and their neuron fate.

    A frequency is "ever-qualified" if at any epoch its committed neuron count
    (neurons with max_frac >= neuron_threshold whose dominant freq is this freq)
    reaches transient_canonical_threshold * d_mlp.

    A frequency is "final" if it is ever-qualified AND its committed count at the
    final epoch reaches final_canonical_threshold * d_mlp.

    Transient frequencies are ever-qualified but not final.
    """

    name = "transient_frequency"
    requires = ["neuron_dynamics"]
    architecture_support = ["transformer", "mlp"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute transient frequency metrics from neuron_dynamics artifact."""
        loader = ArtifactLoader(artifacts_dir)
        nd = loader.load_cross_epoch("neuron_dynamics")

        dominant_freq = nd["dominant_freq"]  # (n_epochs, d_mlp)
        max_frac = nd["max_frac"]  # (n_epochs, d_mlp)
        nd_epochs = nd["epochs"]  # (n_epochs,)

        n_epochs, d_mlp = dominant_freq.shape
        n_freq = int(dominant_freq.max()) + 1

        committed_counts_full = _compute_committed_counts(
            dominant_freq, max_frac, n_freq, NEURON_THRESHOLD
        )  # (n_epochs, n_freq)

        transient_threshold_count = TRANSIENT_CANONICAL_THRESHOLD * d_mlp
        final_threshold_count = FINAL_CANONICAL_THRESHOLD * d_mlp

        peak_counts_per_freq = committed_counts_full.max(axis=0)  # (n_freq,)
        ever_qualified = np.where(peak_counts_per_freq >= transient_threshold_count)[0]

        if len(ever_qualified) == 0:
            return _empty_result(nd_epochs)

        final_counts_per_freq = committed_counts_full[-1]
        final_set = set(int(f) for f in np.where(final_counts_per_freq >= final_threshold_count)[0])

        # Build per-transient arrays
        is_final_arr = np.array([f in final_set for f in ever_qualified], dtype=bool)
        peak_epoch_indices = np.argmax(committed_counts_full[:, ever_qualified], axis=0)
        peak_epoch_arr = nd_epochs[peak_epoch_indices].astype(np.int32)
        peak_count_arr = peak_counts_per_freq[ever_qualified].astype(np.int32)
        committed_counts_out = committed_counts_full[:, ever_qualified].astype(np.int32)

        final_max_frac = max_frac[-1]  # (d_mlp,)

        members_list, homeless_counts = _build_peak_members(
            ever_qualified,
            peak_epoch_indices,
            dominant_freq,
            max_frac,
            final_max_frac,
            NEURON_THRESHOLD,
        )

        flat, offsets = _pack_ragged(members_list)

        return {
            "ever_qualified_freqs": ever_qualified.astype(np.int32),
            "is_final": is_final_arr,
            "peak_epoch": peak_epoch_arr,
            "peak_count": peak_count_arr,
            "homeless_count": np.array(homeless_counts, dtype=np.int32),
            "committed_counts": committed_counts_out,
            "peak_members_flat": flat,
            "peak_members_offsets": offsets,
            "epochs": nd_epochs.astype(np.int32),
            # Threshold metadata stored as 0-d arrays (npz-serializable)
            "_neuron_threshold": np.array(NEURON_THRESHOLD),
            "_transient_canonical_threshold": np.array(TRANSIENT_CANONICAL_THRESHOLD),
            "_final_canonical_threshold": np.array(FINAL_CANONICAL_THRESHOLD),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_committed_counts(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    n_freq: int,
    neuron_threshold: float,
) -> np.ndarray:
    """Committed neuron count per frequency per epoch.

    Args:
        dominant_freq: (n_epochs, d_mlp) 0-indexed argmax
        max_frac:      (n_epochs, d_mlp) per-neuron max frac_explained
        n_freq:        number of frequency channels
        neuron_threshold: per-neuron quality gate

    Returns:
        (n_epochs, n_freq) int32 committed count matrix
    """
    n_epochs, d_mlp = dominant_freq.shape
    counts = np.zeros((n_epochs, n_freq), dtype=np.int32)
    specialized = max_frac >= neuron_threshold

    for ep_idx in range(n_epochs):
        spec_neurons = np.where(specialized[ep_idx])[0]
        if len(spec_neurons) > 0:
            counts[ep_idx] = np.bincount(dominant_freq[ep_idx, spec_neurons], minlength=n_freq)

    return counts


def _build_peak_members(
    ever_qualified: np.ndarray,
    peak_epoch_indices: np.ndarray,
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    final_max_frac: np.ndarray,
    neuron_threshold: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Build peak member lists and homeless counts for each ever-qualified frequency.

    Members are neurons committed to the frequency at its peak epoch.
    Homeless count is how many of those members have max_frac < neuron_threshold
    at the final epoch.

    Returns:
        (members_list, homeless_counts) parallel lists, one entry per frequency
    """
    members_list = []
    homeless_counts = []

    for i, freq in enumerate(ever_qualified):
        ep_idx = int(peak_epoch_indices[i])
        specialized_at_peak = max_frac[ep_idx] >= neuron_threshold
        members = np.where((dominant_freq[ep_idx] == freq) & specialized_at_peak)[0]
        members_list.append(members.astype(np.int32))
        n_homeless = int((final_max_frac[members] < neuron_threshold).sum())
        homeless_counts.append(n_homeless)

    return members_list, homeless_counts


def _pack_ragged(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length arrays into flat storage with offset index.

    Args:
        arrays: list of 1-D int32 arrays

    Returns:
        (flat, offsets) where flat is the concatenated data and offsets has
        length len(arrays)+1; the i-th array is flat[offsets[i]:offsets[i+1]]
    """
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    for i, arr in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(arr)
    flat = np.concatenate(arrays) if arrays else np.array([], dtype=np.int32)
    return flat, offsets


def load_peak_members(artifact: dict, group_idx: int) -> np.ndarray:
    """Return neuron indices in the peak-epoch cohort for one ever-qualified frequency.

    Args:
        artifact: dict loaded from transient_frequency cross_epoch artifact
        group_idx: index into ever_qualified_freqs

    Returns:
        1-D int32 array of neuron indices
    """
    flat = artifact["peak_members_flat"]
    offsets = artifact["peak_members_offsets"]
    return flat[offsets[group_idx] : offsets[group_idx + 1]]


def _empty_result(epochs: np.ndarray) -> dict[str, Any]:
    n = len(epochs)
    return {
        "ever_qualified_freqs": np.array([], dtype=np.int32),
        "is_final": np.array([], dtype=bool),
        "peak_epoch": np.array([], dtype=np.int32),
        "peak_count": np.array([], dtype=np.int32),
        "homeless_count": np.array([], dtype=np.int32),
        "committed_counts": np.empty((n, 0), dtype=np.int32),
        "peak_members_flat": np.array([], dtype=np.int32),
        "peak_members_offsets": np.array([0], dtype=np.int32),
        "epochs": epochs.astype(np.int32),
        "_neuron_threshold": np.array(NEURON_THRESHOLD),
        "_transient_canonical_threshold": np.array(TRANSIENT_CANONICAL_THRESHOLD),
        "_final_canonical_threshold": np.array(FINAL_CANONICAL_THRESHOLD),
    }
