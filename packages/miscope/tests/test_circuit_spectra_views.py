"""REQ_155: circuit-spectra trajectory view + renderer tests.

The renderer is plot-only, so it is tested on synthetic prepared data. A guarded
integration test exercises the full View Catalog path on a baseline variant when
its ``circuit_spectra`` artifacts are present (skipped in CI, where per-variant
data is absent).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from miscope.visualization.renderers.circuit_spectra import render_circuit_spectra_trajectory

SITES = ("full_ov", "full_qk")
METRICS = ("copying_score", "effective_rank", "operator_norm")


def _synthetic_data(n_heads: int = 4) -> dict:
    rng = np.random.default_rng(0)
    epochs = np.arange(0, 1000, 100)
    series = {(s, m): rng.random((len(epochs), n_heads)) for s in SITES for m in METRICS}
    return {
        "epochs": epochs,
        "series": series,
        "sites": list(SITES),
        "metrics": list(METRICS),
        "grok_epoch": 400,
    }


def test_renderer_one_trace_per_head():
    fig = render_circuit_spectra_trajectory(
        _synthetic_data(n_heads=4), epoch=500, site="full_ov", metric="copying_score"
    )
    assert isinstance(fig, go.Figure)
    assert sum(1 for t in fig.data if t.type == "scatter") == 4


def test_renderer_draws_grok_and_cursor_lines():
    fig = render_circuit_spectra_trajectory(
        _synthetic_data(), epoch=500, site="full_qk", metric="effective_rank"
    )
    # grok vline + epoch cursor vline → two layout shapes.
    assert len(fig.layout.shapes) >= 2


def test_renderer_no_cursor_when_epoch_none():
    fig = render_circuit_spectra_trajectory(
        _synthetic_data(), epoch=None, site="full_ov", metric="operator_norm"
    )
    # Only the grok line (no epoch cursor).
    assert len(fig.layout.shapes) == 1


def test_renderer_unknown_site_or_metric_raises():
    with pytest.raises(KeyError):
        render_circuit_spectra_trajectory(_synthetic_data(), site="nope", metric="copying_score")


def test_view_catalog_path_on_baseline_if_present():
    """Full load_data → renderer path on a real baseline (skipped if data absent)."""
    from miscope import load_family

    try:
        fam = load_family("modulo_addition_1layer")
        variant = next((v for v in fam.variants if v.name == "p113_seed999_dseed598"), None)
    except Exception:  # pragma: no cover - environment-dependent
        pytest.skip("modulo_addition_1layer family unavailable")
    if variant is None or "circuit_spectra" not in variant.artifacts.get_available_analyzers():
        pytest.skip("circuit_spectra artifacts absent for baseline")

    fig = (
        variant.at(None)
        .view("circuits.spectra.trajectory")
        .figure(site="full_qk", metric="effective_rank")
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
