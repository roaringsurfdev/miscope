"""Transient-frequency conformed accessor (REQ_141 bucket-2).

The cross-epoch ``transient_frequency`` analyzer is gone; its outputs are now the
``committed_counts`` / ``transient_frequencies`` / ``transient_peak_members``
derived tables over the conformed ``neuron_frequency_attribution``. This module is
the single place that reassembles the legacy artifact-shaped dict from those
tables, so the existing renderers (committed-counts line chart, peak-cohort
scatter) consume an identical structure — only the source changed.
"""

from __future__ import annotations

import numpy as np

from miscope.warehouse.reader import read_table

# Thresholds carried over from the retired transient_frequency analyzer, so the
# reassembled dict matches the legacy artifact the renderers read (the per-neuron
# max_frac gate is shown in the committed-counts subtitle).
NEURON_THRESHOLD = 0.70
TRANSIENT_DETECTION_FRACTION = 0.05
FINAL_CANONICAL_FRACTION = 0.10

_SUMMARY_TABLE = "transient_frequencies"
_COMMITTED_TABLE = "committed_counts"
_MEMBERS_TABLE = "transient_peak_members"
_ATTRIBUTION_TABLE = "neuron_frequency_attribution"


def load_transient_dict(variant: object) -> dict:
    """Reassemble the legacy ``transient_frequency`` artifact dict from the warehouse.

    Self-healing (mirrors :func:`miscope.analysis.neuron_frequency.load`): if the
    derived tables are not materialized yet, materialize the warehouse once and
    retry. If the inputs are genuinely absent (the variant was never analyzed with
    the attribution analyzer), ``FileNotFoundError`` propagates — the same signal
    the old ``load_cross_epoch`` raised, so callers' absence handling is unchanged.
    """
    try:
        return _assemble(variant)
    except FileNotFoundError:
        variant.warehouse.materialize()  # type: ignore[attr-defined]
        return _assemble(variant)


def _assemble(variant: object) -> dict:
    """Read the three transient derived tables + attribution epochs into the dict."""
    summary = read_table(variant, _SUMMARY_TABLE).df.sort_values("frequency")
    committed = read_table(variant, _COMMITTED_TABLE).df
    members = read_table(variant, _MEMBERS_TABLE).df
    epochs = np.array(sorted(read_table(variant, _ATTRIBUTION_TABLE).df["epoch"].unique()))

    ever = summary["frequency"].to_numpy().astype(np.int32)
    committed_matrix = _committed_matrix(committed, epochs, ever)
    flat, offsets = _pack_members(members, ever)

    return {
        "ever_qualified_freqs": ever,
        "is_final": summary["is_final"].to_numpy().astype(bool),
        "peak_epoch": summary["peak_epoch"].to_numpy().astype(np.int32),
        "peak_count": summary["peak_count"].to_numpy().astype(np.int32),
        "homeless_count": summary["homeless_count"].to_numpy().astype(np.int32),
        "committed_counts": committed_matrix,
        "peak_members_flat": flat,
        "peak_members_offsets": offsets,
        "epochs": epochs.astype(np.int32),
        "_neuron_threshold": np.array(NEURON_THRESHOLD),
        "_transient_canonical_threshold": np.array(TRANSIENT_DETECTION_FRACTION),
        "_final_canonical_threshold": np.array(FINAL_CANONICAL_FRACTION),
    }


def load_peak_members(artifact: dict, group_idx: int) -> np.ndarray:
    """Neuron indices in the peak-epoch cohort for one ever-qualified frequency."""
    flat = artifact["peak_members_flat"]
    offsets = artifact["peak_members_offsets"]
    return flat[offsets[group_idx] : offsets[group_idx + 1]]


def _committed_matrix(committed: object, epochs: np.ndarray, ever: np.ndarray) -> np.ndarray:
    """Dense ``(n_epochs, n_transient)`` committed-count trajectory for ever-qualified freqs."""
    if len(ever) == 0:
        return np.empty((len(epochs), 0), dtype=np.int32)
    pivot = committed.pivot_table(  # type: ignore[attr-defined]
        index="epoch", columns="frequency", values="committed_counts", fill_value=0
    )
    pivot = pivot.reindex(index=epochs, columns=ever, fill_value=0)
    return pivot.to_numpy().astype(np.int32)


def _pack_members(members: object, ever: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the per-frequency peak cohorts to (flat, offsets), in ever-qualified order."""
    by_freq = {
        int(freq): group["member_neuron"].to_numpy().astype(np.int32)
        for freq, group in members.sort_values(["frequency", "member_neuron"]).groupby(  # type: ignore[attr-defined]
            "frequency"
        )
    }
    offsets = np.zeros(len(ever) + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    for i, freq in enumerate(ever):
        arr = by_freq.get(int(freq), np.array([], dtype=np.int32))
        chunks.append(arr)
        offsets[i + 1] = offsets[i] + len(arr)
    flat = np.concatenate(chunks) if chunks else np.array([], dtype=np.int32)
    return flat, offsets
