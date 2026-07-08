"""Conformed ``(epoch, neuron) -> dominant-frequency`` dimension (REQ_110D).

Three engines — :mod:`miscope.analysis.variant_analysis_summary`,
:mod:`miscope.views.cross_variant`, and the now-deleted ``variant_summary`` —
each used to re-load ``neuron_dynamics.npz`` and recompute this dimension with
their own ``+1`` conversions and specialization thresholds. That duplication is
the failure mode REQ_110D collapses: the warehouse ``neuron_frequency_attribution``
table (110-A) *is* this dimension, so it is read **once** here and the
0-indexed → 1-indexed frequency convention is applied in exactly one place.

The conformed dimension is exposed as 2-D ``(n_epochs, d_mlp)`` arrays (the same
shape the ``.npz`` previously provided, so consumers change their *source*, not
their loop bodies) plus per-epoch counting primitives. Scan *policy* (which
threshold, which count gate, how to walk epochs) stays with each consumer; only
the dimension and the counting primitives are shared.

``dominant_freq`` returned here is **1-indexed**. Consumers must not re-apply the
``+1``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from miscope.warehouse.reader import list_signatures, read_table

# "A neuron is specialized to its dominant frequency." 0.7 is the value used
# throughout the summary engine; cross_variant's first-mover scan historically
# used 0.75 — kept distinct (passed explicitly) so this collapse changes no
# threshold behavior. The 0.7/0.75 discrepancy is flagged, not silently unified.
DEFAULT_SPECIALIZATION_THRESHOLD: float = 0.7

ATTRIBUTION_TABLE = "neuron_frequency_attribution"
_NEURON_DYNAMICS_TABLE = "neuron_dynamics"
_NEURON_SIGNATURE = "by__variant_neuron"
_SCALAR_SIGNATURE = "by__variant"


@dataclass(frozen=True)
class NeuronFrequencyAttribution:
    """The conformed dimension as ``(n_epochs, d_mlp)`` arrays, **1-indexed**.

    ``dominant_freq`` / ``frac_explained`` are the per-(epoch, neuron) dominant
    frequency and its fraction explained — sourced from the warehouse, not the
    ``.npz``. ``commitment_epochs`` / ``switch_counts`` / ``threshold`` carry the
    per-neuron and scalar ``neuron_dynamics`` aux fields (``None`` when absent).
    """

    epochs: np.ndarray  # (n_epochs,) ascending
    neurons: np.ndarray  # (d_mlp,) ascending
    dominant_freq: np.ndarray  # (n_epochs, d_mlp) int, 1-indexed
    frac_explained: np.ndarray  # (n_epochs, d_mlp) float
    commitment_epochs: np.ndarray | None  # (d_mlp,) artifact commitment epochs
    switch_counts: np.ndarray | None  # (d_mlp,)
    threshold: float | None  # artifact "uncommitted floor" threshold

    @classmethod
    def from_legacy_dict(cls, nd: dict) -> NeuronFrequencyAttribution:
        """Build from a legacy ``neuron_dynamics`` npz-style dict (0-indexed freqs).

        The inverse of :meth:`as_legacy_arrays`. Useful for tests and any caller
        holding the raw arrays rather than a warehouse handle; the ``+1`` convention
        is applied here, as everywhere.
        """
        dom = np.asarray(nd["dominant_freq"])
        frac = np.asarray(nd["max_frac"], dtype=float)
        epochs = np.asarray(nd["epochs"])
        thr = nd.get("threshold")
        threshold = (
            float(np.asarray(thr).ravel()[0]) if thr is not None and np.asarray(thr).size else None
        )
        return cls(
            epochs=epochs,
            neurons=np.arange(dom.shape[1]),
            dominant_freq=dom.astype(int) + 1,
            frac_explained=frac,
            commitment_epochs=(
                np.asarray(nd["commitment_epochs"], dtype=float)
                if "commitment_epochs" in nd
                else None
            ),
            switch_counts=(np.asarray(nd["switch_counts"]) if "switch_counts" in nd else None),
            threshold=threshold,
        )

    @property
    def d_mlp(self) -> int:
        return self.dominant_freq.shape[1]

    @property
    def n_epochs(self) -> int:
        return self.dominant_freq.shape[0]

    def epoch_index(self, epoch: int) -> int:
        """Index of the nearest stored epoch at or after ``epoch`` (clamped)."""
        idx = int(np.searchsorted(self.epochs, epoch))
        return min(idx, self.n_epochs - 1)

    def specialized_mask(self, epoch_idx: int, *, threshold: float) -> np.ndarray:
        """Boolean ``(d_mlp,)`` mask of neurons specialized at ``epoch_idx``."""
        return self.frac_explained[epoch_idx] >= threshold

    def frequency_counts(self, epoch_idx: int, *, threshold: float) -> dict[int, int]:
        """Map ``frequency (1-indexed) -> #specialized neurons`` at ``epoch_idx``."""
        freqs = self.dominant_freq[epoch_idx][self.specialized_mask(epoch_idx, threshold=threshold)]
        return dict(Counter(int(f) for f in freqs))

    def specialized_frequencies(self, epoch_idx: int, *, threshold: float) -> list[int]:
        """Sorted unique 1-indexed frequencies with at least one specialized neuron."""
        return sorted(self.frequency_counts(epoch_idx, threshold=threshold))

    def committed_frequencies(
        self, epoch_idx: int, *, threshold: float, population_floor: float
    ) -> list[int]:
        """Frequencies whose specialized-neuron count >= ``population_floor * d_mlp``."""
        floor = population_floor * self.d_mlp
        counts = self.frequency_counts(epoch_idx, threshold=threshold)
        return sorted(f for f, c in counts.items() if c >= floor)

    def specialized_count(self, epoch_idx: int, *, threshold: float) -> int:
        """Total number of specialized neurons at ``epoch_idx``."""
        return int(self.specialized_mask(epoch_idx, threshold=threshold).sum())

    def final_specialized_frequencies(self, *, threshold: float) -> set[int]:
        """1-indexed frequencies still specialized at the final epoch (survival checks)."""
        return set(self.specialized_frequencies(self.n_epochs - 1, threshold=threshold))

    def as_legacy_arrays(self) -> dict[str, np.ndarray]:
        """The legacy ``neuron_dynamics`` npz array dict, sourced from the warehouse.

        For consumers written against the ``.npz`` arrays (band concentration, the
        view loaders) — ``dominant_freq`` is handed back **0-indexed** (the npz
        convention these consumers expect), so they leave the npz behind without a
        behavior change. Aux arrays are included when available.
        """
        arrays: dict[str, np.ndarray] = {
            "epochs": self.epochs,
            "dominant_freq": self.dominant_freq - 1,  # restore 0-indexed for legacy consumers
            "max_frac": self.frac_explained,
        }
        if self.commitment_epochs is not None:
            arrays["commitment_epochs"] = self.commitment_epochs
        if self.switch_counts is not None:
            arrays["switch_counts"] = self.switch_counts
        if self.threshold is not None:
            arrays["threshold"] = np.array([self.threshold])
        return arrays

    def recompute_commitment_epochs(self, *, threshold: float) -> np.ndarray:
        """Per-neuron commitment epoch recomputed at ``threshold`` (NaN if uncommitted).

        A neuron's commitment epoch is the earliest epoch from which it stays
        specialized to its *final* dominant frequency continuously through the
        end. Only neurons specialized at the final epoch get a value.
        """
        commitment = np.full(self.d_mlp, np.nan)
        final_freq = self.dominant_freq[-1]
        for n in range(self.d_mlp):
            if self.frac_explained[-1, n] < threshold:
                continue
            stable_from = self.n_epochs - 1
            for t in range(self.n_epochs - 2, -1, -1):
                if (
                    self.frac_explained[t, n] >= threshold
                    and self.dominant_freq[t, n] == final_freq[n]
                ):
                    stable_from = t
                else:
                    break
            commitment[n] = self.epochs[stable_from]
        return commitment


def load(variant: object) -> NeuronFrequencyAttribution:
    """Load the conformed dimension for ``variant`` from the warehouse (self-healing).

    Reads ``neuron_frequency_attribution`` (the dominant rows) plus the
    ``neuron_dynamics`` aux signatures. If the columnar warehouse has not been
    materialized yet, materializes it once and retries — so a summary run after a
    fresh analysis pass is self-sufficient.
    """
    frame = _read_attribution(variant)
    aux = _read_aux(variant)
    epochs = np.array(sorted(frame["epoch"].unique()))
    neurons = np.array(sorted(frame["neuron"].unique()))
    # Pivot the long dominant rows to (n_epochs, d_mlp); +1 is applied here only.
    freq_wide = frame.pivot(index="epoch", columns="neuron", values="frequency")
    frac_wide = frame.pivot(index="epoch", columns="neuron", values="frac_explained")
    freq_wide = freq_wide.reindex(index=epochs, columns=neurons)
    frac_wide = frac_wide.reindex(index=epochs, columns=neurons)
    return NeuronFrequencyAttribution(
        epochs=epochs,
        neurons=neurons,
        dominant_freq=freq_wide.to_numpy().astype(int) + 1,
        frac_explained=frac_wide.to_numpy().astype(float),
        commitment_epochs=aux.commitment_epochs,
        switch_counts=aux.switch_counts,
        threshold=aux.threshold,
    )


def from_legacy_dict(nd: dict) -> NeuronFrequencyAttribution:
    """Module-level alias for :meth:`NeuronFrequencyAttribution.from_legacy_dict`."""
    return NeuronFrequencyAttribution.from_legacy_dict(nd)


def classify_band(freq: int, prime: int) -> str:
    """Classify a 1-indexed frequency into low / mid / high band relative to ``prime``.

    The single definition of the band split (previously duplicated across the
    summary engine and ``cross_variant``).
    """
    if freq <= prime // 4:
        return "low"
    if freq > 3 * prime // 8:
        return "high"
    return "mid"


def _read_attribution(variant: object):
    try:
        return read_table(variant, ATTRIBUTION_TABLE).df
    except FileNotFoundError:
        variant.warehouse.materialize()  # type: ignore[attr-defined]
        return read_table(variant, ATTRIBUTION_TABLE).df


@dataclass(frozen=True)
class _NeuronAux:
    """The optional ``neuron_dynamics`` aux signatures, tolerant of absence."""

    commitment_epochs: np.ndarray | None = None
    switch_counts: np.ndarray | None = None
    threshold: float | None = None


def _read_aux(variant: object) -> _NeuronAux:
    """Per-neuron + scalar ``neuron_dynamics`` aux fields, tolerant of absence."""
    commitment_epochs: np.ndarray | None = None
    switch_counts: np.ndarray | None = None
    threshold: float | None = None
    sigs = list_signatures(variant, _NEURON_DYNAMICS_TABLE)
    if _NEURON_SIGNATURE in sigs:
        ndf = read_table(variant, _NEURON_DYNAMICS_TABLE, _NEURON_SIGNATURE).df
        ndf = ndf.sort_values("neuron")
        if "commitment_epochs" in ndf:
            commitment_epochs = ndf["commitment_epochs"].to_numpy().astype(float)
        if "switch_counts" in ndf:
            switch_counts = ndf["switch_counts"].to_numpy()
    if _SCALAR_SIGNATURE in sigs:
        sdf = read_table(variant, _NEURON_DYNAMICS_TABLE, _SCALAR_SIGNATURE).df
        if "threshold" in sdf and len(sdf):
            threshold = float(sdf["threshold"].iloc[0])
    return _NeuronAux(commitment_epochs, switch_counts, threshold)
