"""REQ_155: Cross-variant circuit-spectra overlay (data prep).

Unlike the per-variant ``circuits.spectra.*`` ViewCatalog views (which bind a
single variant), comparing a circuit's dynamics *across* variants needs a
multi-variant entry point — same shape as :mod:`miscope.views.cross_variant`. This
module prepares one metric's head-mean trajectory per variant, aligned on
event-relative time (epoch − grok), as a thin payload the
``render_circuit_spectra_cross_variant`` renderer overlays.

All processing lives here (REQ_155 plot-only line): the per-variant warehouse read,
the head-mean reduction, the grok lookup, and the event-relative shift. The
renderer only draws the prepared series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from miscope.views.universal import _circuit_metric_names, _grok_epoch

if TYPE_CHECKING:
    from miscope.families.variant import Variant

# Stable per-variant palette (cycled by selection order). Kept here, not in the
# renderer, so colour is part of the prepared payload (plot-only renderer).
_PALETTE = ("steelblue", "tomato", "orange", "mediumseagreen", "mediumpurple", "slategray")


def load_circuit_spectra_cross_variant(
    variants: list[Variant],
    *,
    site: str = "full_ov",
    metric: str = "copying_score",
    align_on_grok: bool = True,
) -> dict[str, Any]:
    """One circuit metric's head-mean trajectory per variant, event-relative.

    Args:
        variants: Variants to overlay. Any lacking ``circuit_spectra`` artifacts or
            the requested ``site`` are skipped (not an error — the overlay shows
            whatever has the data).
        site: Circuit composition site (``full_ov``, ``full_qk``, …).
        metric: Spectral metric (``copying_score``, ``effective_rank``,
            ``operator_norm``).
        align_on_grok: When True, shift each x-axis by the variant's grok epoch so
            curves share an event-relative origin; otherwise plot raw epochs.

    Returns:
        Payload for ``render_circuit_spectra_cross_variant``: ``series`` (one
        ``{label, x, y, color}`` per included variant), ``site``, ``metric``, and an
        ``x_title``.
    """
    if metric not in _circuit_metric_names():
        raise ValueError(f"Unknown metric {metric!r}; expected one of {_circuit_metric_names()}.")

    series: list[dict[str, Any]] = []
    field = f"{site}_{metric}"
    for variant in variants:
        prepared = _variant_series(variant, site, field, align_on_grok)
        if prepared is None:
            continue
        prepared["color"] = _PALETTE[len(series) % len(_PALETTE)]
        series.append(prepared)

    return {
        "series": series,
        "site": site,
        "metric": metric,
        "x_title": "epochs since grok" if align_on_grok else "epoch",
    }


def _variant_series(
    variant: Variant, site: str, field: str, align_on_grok: bool
) -> dict[str, Any] | None:
    """Prepare one variant's head-mean trajectory, or None if it lacks the data."""
    declared = {s.name for s in variant.family.circuit_spectra_sites}
    if site not in declared or "circuit_spectra" not in variant.artifacts.get_available_analyzers():
        return None
    art = variant.artifacts.load_epochs("circuit_spectra", fields=[field])
    head_mean = np.nanmean(art[field], axis=1)  # (n_epochs,) — head-mean reduction here
    epochs = np.asarray(art["epochs"], dtype=float)
    x = epochs
    if align_on_grok:
        grok = _grok_epoch(variant)
        x = epochs - grok if grok is not None else epochs
    return {"label": variant.name, "x": x.tolist(), "y": head_mean.tolist()}
