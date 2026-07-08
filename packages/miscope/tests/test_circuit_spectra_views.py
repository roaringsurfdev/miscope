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

from miscope.visualization.renderers.circuit_spectra import (
    render_circuit_matrix_heatmap,
    render_circuit_spectra_cross_variant,
    render_circuit_spectra_ranking,
    render_circuit_spectra_trajectory,
)

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


# --- Ranking bar renderer (per-head metric at one epoch, sorted) ---


def _ranking_data(n_heads: int = 4) -> dict:
    values = {
        ("full_ov", "copying_score"): np.array([0.3, 0.1, 0.8, 0.2][:n_heads]),
        ("full_qk", "operator_norm"): np.arange(n_heads, dtype=float),
    }
    return {"epoch": 5000, "values": values, "sites": list(SITES), "metrics": list(METRICS)}


def test_ranking_sorted_descending_keeps_head_identity():
    fig = render_circuit_spectra_ranking(_ranking_data(), site="full_ov", metric="copying_score")
    bar = fig.data[0]
    assert bar.type == "bar"
    # Sorted descending by value; head 2 (0.8) leads, labels keep the true head index.
    assert list(bar.x) == ["head 2", "head 0", "head 3", "head 1"]
    assert list(bar.y) == [0.8, 0.3, 0.2, 0.1]


def test_ranking_unknown_site_metric_raises():
    with pytest.raises(KeyError):
        render_circuit_spectra_ranking(_ranking_data(), site="nope", metric="copying_score")


# --- Circuit-matrix heatmap renderer ---


def _matrix_data(n_heads: int = 4, p: int = 7) -> dict:
    rng = np.random.default_rng(1)
    return {
        "epoch": 5000,
        "matrices_by_site": {
            "full_ov": rng.standard_normal((n_heads, p, p)),
            "direct_path": rng.standard_normal((1, p, p)),
        },
        "sites": ["direct_path", "full_ov"],
    }


def test_matrix_heatmap_picks_head_and_site():
    fig = render_circuit_matrix_heatmap(_matrix_data(), site="full_ov", head=2)
    assert fig.data[0].type == "heatmap"
    assert np.asarray(fig.data[0].z).shape == (7, 7)


def test_matrix_heatmap_clamps_head_for_headless_circuit():
    # direct_path has one head; an out-of-range head index clamps rather than raising.
    fig = render_circuit_matrix_heatmap(_matrix_data(), site="direct_path", head=3)
    assert np.asarray(fig.data[0].z).shape == (7, 7)


def test_matrix_heatmap_unknown_site_raises():
    with pytest.raises(KeyError):
        render_circuit_matrix_heatmap(_matrix_data(), site="nope")


# --- Cross-variant overlay renderer ---


def _cross_variant_data() -> dict:
    return {
        "series": [
            {
                "label": "p113",
                "x": [-100.0, 0.0, 100.0],
                "y": [0.2, 0.5, 0.7],
                "color": "steelblue",
            },
            {"label": "p109", "x": [-50.0, 0.0, 50.0], "y": [0.3, 0.6, 0.8], "color": "tomato"},
        ],
        "site": "full_ov",
        "metric": "copying_score",
        "x_title": "epochs since grok",
    }


def test_cross_variant_one_trace_per_variant_plus_event_line():
    fig = render_circuit_spectra_cross_variant(_cross_variant_data())
    assert sum(1 for t in fig.data if t.type == "scatter") == 2
    assert len(fig.layout.shapes) == 1  # the event (x=0) vline
    assert fig.layout.xaxis.title.text == "epochs since grok"


# --- Guarded integration: new views + cross-variant prep on baselines ---


def _baseline(name: str):
    from miscope import load_family

    try:
        fam = load_family("modulo_addition_1layer")
    except Exception:  # pragma: no cover - environment-dependent
        pytest.skip("modulo_addition_1layer family unavailable")
    variant = next((v for v in fam.variants if v.name == name), None)
    if variant is None or "circuit_spectra" not in variant.artifacts.get_available_analyzers():
        pytest.skip(f"circuit_spectra artifacts absent for {name}")
    return variant


def test_ranking_and_matrix_views_on_baseline_if_present():
    variant = _baseline("p113_seed999_dseed598")
    epoch = variant.artifacts.get_epochs("circuit_spectra")[
        len(variant.artifacts.get_epochs("circuit_spectra")) // 2
    ]

    bar = (
        variant.at(epoch)
        .view("circuits.spectra.ranking")
        .figure(site="full_ov", metric="copying_score")
    )
    assert bar.data[0].type == "bar"
    assert len(bar.data[0].x) >= 1

    heat = variant.at(epoch).view("circuits.spectra.matrix").figure(site="full_ov", head=0)
    assert heat.data[0].type == "heatmap"
    # Square in token space (p x p).
    z = np.asarray(heat.data[0].z)
    assert z.ndim == 2 and z.shape[0] == z.shape[1]


def test_cross_variant_prep_aligns_on_grok_if_present():
    from miscope.views.circuit_spectra import load_circuit_spectra_cross_variant

    variants = [_baseline(n) for n in ("p113_seed999_dseed598", "p109_seed485_dseed598")]
    data = load_circuit_spectra_cross_variant(
        variants, site="full_ov", metric="copying_score", align_on_grok=True
    )
    assert data["x_title"] == "epochs since grok"
    assert len(data["series"]) == 2
    for s in data["series"]:
        assert len(s["x"]) == len(s["y"]) >= 1
        assert s["color"] is not None


def test_cross_variant_prep_rejects_unknown_metric():
    from miscope.views.circuit_spectra import load_circuit_spectra_cross_variant

    with pytest.raises(ValueError):
        load_circuit_spectra_cross_variant([], site="full_ov", metric="nope")
