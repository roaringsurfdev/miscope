"""REQ_127 Phase B: composition objects for views that source from
more than one analyzer artifact.

Anti-designed on purpose: each composition case is a plain dataclass with
a ``from_variant`` constructor and an accessor that produces exactly what
its renderer consumes. There is no base class, registry, or protocol — if
a second case proves the shape worth lifting into a shared abstraction,
that is a future-REQ decision driven by observed need, not anticipated
need. (The experimental ``dataview_universal`` / ``dataview_catalog``
modules predate this and are not the foundation here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miscope.families.variant import Variant


@dataclass
class RepresentationGeometryTimeseries:
    """Per-site representation-geometry summary for the ``geometry.timeseries`` panel.

    REQ_126 PR 3 defused the per-site ``{site}_fourier_alignment`` field out
    of the ``repr_geometry`` analyzer and into the dedicated
    ``centroid_fourier_alignment`` analyzer. The timeseries renderer still
    expects a single flat summary dict carrying both the geometric measures
    (circularity, radius, dimensionality, …) and the Fourier-alignment
    series. This object composes the two sources so the renderer stays pure
    plotting.

    The Fourier-alignment values are stacked from the dedicated analyzer's
    per-epoch artifacts (it does not always emit a ``summary.npz``) and are
    treated as authoritative — they overwrite any stale field that an
    older pre-defusion ``repr_geometry`` summary may still carry.
    """

    summary: dict[str, Any]

    @classmethod
    def from_variant(cls, variant: Variant) -> RepresentationGeometryTimeseries:
        """Load + compose the repr_geometry summary with Fourier alignment."""
        summary = dict(variant.artifacts.load_summary("repr_geometry"))
        cls._merge_fourier_alignment(variant, summary)
        return cls(summary=summary)

    @staticmethod
    def _merge_fourier_alignment(variant: Variant, summary: dict[str, Any]) -> None:
        """Stack the dedicated analyzer's per-site scalars into the summary.

        Degrades silently when the analyzer hasn't been run (no artifacts):
        the panel then behaves as it did before this composition — the
        Fourier trace simply doesn't appear.
        """
        try:
            stacked = variant.artifacts.load_epochs("centroid_fourier_alignment")
        except FileNotFoundError:
            return
        epochs = summary.get("epochs")
        for key, arr in stacked.items():
            if not key.endswith("_fourier_alignment"):
                continue
            # Only merge when epoch axes line up; mismatched runs are skipped
            # rather than silently misaligned.
            if epochs is None or len(arr) == len(epochs):
                summary[key] = arr

    def as_summary_dict(self) -> dict[str, Any]:
        """Return the flat summary dict the timeseries renderer consumes."""
        return self.summary
