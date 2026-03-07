"""Tests for REQ_057: Cross-Variant Grokking Health Comparison."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from miscope.views.cross_variant import (
    ClassificationRules,
    classify_failure_mode,
    compute_variant_metrics,
    load_family_comparison,
)
from miscope.visualization.renderers.cross_variant import (
    render_loss_curve_overlay,
    render_metrics_table,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_variant(
    test_losses: list[float],
    train_losses: list[float] | None = None,
    prime: int = 113,
    seed: int = 999,
    nd_data: dict | None = None,
    rg_data: dict | None = None,
    variant_name: str | None = None,
) -> MagicMock:
    """Build a mock Variant with realistic metadata and optional artifact data."""
    if train_losses is None:
        train_losses = test_losses[:]

    variant = MagicMock()
    variant.name = variant_name or f"mock_p{prime}_seed{seed}"
    variant.model_config = {"prime": prime, "seed": seed}
    variant.metadata = {
        "test_losses": test_losses,
        "train_losses": train_losses,
    }

    def mock_load_cross_epoch(name):
        if name == "neuron_dynamics" and nd_data is not None:
            return nd_data
        raise FileNotFoundError(f"No cross-epoch for {name}")

    def mock_load_summary(name):
        if name == "repr_geometry" and rg_data is not None:
            return rg_data
        raise FileNotFoundError(f"No summary for {name}")

    variant.artifacts.load_cross_epoch.side_effect = mock_load_cross_epoch
    variant.artifacts.load_summary.side_effect = mock_load_summary
    return variant


def _make_nd_data(
    n_epochs: int = 5,
    d_mlp: int = 12,
    n_freq: int = 8,
    commitment_epoch_range: tuple[int, int] = (100, 400),
) -> dict:
    """Build minimal neuron_dynamics cross-epoch data."""
    epochs = np.arange(0, n_epochs * 100, 100)
    dominant_freq = np.zeros((n_epochs, d_mlp), dtype=np.int32)
    for i in range(d_mlp):
        dominant_freq[:, i] = i % n_freq
    max_frac = np.full((n_epochs, d_mlp), 0.9, dtype=np.float32)
    switch_counts = np.zeros(d_mlp, dtype=np.int32)
    commitment_epochs = np.linspace(
        commitment_epoch_range[0], commitment_epoch_range[1], d_mlp
    )
    threshold = np.array([3.0 / n_freq])
    return {
        "epochs": epochs,
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "switch_counts": switch_counts,
        "commitment_epochs": commitment_epochs,
        "threshold": threshold,
    }


def _make_rg_data(n_epochs: int = 5, circularity: float = 0.85, fisher: float = 12.0) -> dict:
    """Build minimal repr_geometry summary data."""
    return {
        "epochs": np.arange(n_epochs),
        "resid_post_circularity": np.linspace(0.1, circularity, n_epochs),
        "resid_post_fisher_mean": np.linspace(1.0, fisher, n_epochs),
    }


# ---------------------------------------------------------------------------
# classify_failure_mode
# ---------------------------------------------------------------------------


class TestClassifyFailureMode:
    def test_healthy_variant(self):
        metrics = {
            "grokking_onset_epoch": 5000,
            "final_test_loss": 0.0001,
            "frequency_band_count": 3,
        }
        mode, reasons = classify_failure_mode(metrics)
        assert mode == "healthy"
        assert len(reasons) > 0

    def test_no_grokking(self):
        metrics = {
            "grokking_onset_epoch": None,
            "final_test_loss": 1.5,
            "frequency_band_count": 1,
        }
        mode, reasons = classify_failure_mode(metrics)
        assert mode == "no_grokking"

    def test_degraded_high_final_loss(self):
        metrics = {
            "grokking_onset_epoch": 20000,
            "final_test_loss": 0.02,
            "frequency_band_count": 1,
        }
        mode, reasons = classify_failure_mode(metrics)
        assert mode == "degraded"
        assert any("final_test_loss" in r for r in reasons)

    def test_late_grokker(self):
        metrics = {
            "grokking_onset_epoch": 20000,
            "final_test_loss": 0.001,
            "frequency_band_count": 3,
        }
        rules = ClassificationRules(late_grokking_epoch=15000)
        mode, reasons = classify_failure_mode(metrics, rules)
        assert mode == "late_grokker"
        assert any("grokking_onset" in r for r in reasons)

    def test_custom_rules_applied(self):
        metrics = {
            "grokking_onset_epoch": 8000,
            "final_test_loss": 0.0001,
            "frequency_band_count": 3,
        }
        strict_rules = ClassificationRules(late_grokking_epoch=5000)
        mode, _ = classify_failure_mode(metrics, strict_rules)
        assert mode == "late_grokker"

        lenient_rules = ClassificationRules(late_grokking_epoch=15000)
        mode, _ = classify_failure_mode(metrics, lenient_rules)
        assert mode == "healthy"


# ---------------------------------------------------------------------------
# compute_variant_metrics
# ---------------------------------------------------------------------------


class TestComputeVariantMetrics:
    def test_metadata_fields_populated(self):
        losses = [2.0, 1.5, 0.5, 0.05, 0.001]
        variant = _make_variant(test_losses=losses, prime=113, seed=999)
        metrics = compute_variant_metrics(variant)
        assert metrics["prime"] == 113
        assert metrics["seed"] == 999
        assert metrics["num_epochs"] == 5
        assert abs(metrics["final_test_loss"] - 0.001) < 1e-6

    def test_grokking_onset_detected(self):
        losses = [2.0, 1.5, 0.5, 0.05, 0.001]
        variant = _make_variant(test_losses=losses)
        metrics = compute_variant_metrics(variant)
        # 0.05 < 0.1 at index 3
        assert metrics["grokking_onset_epoch"] == 3

    def test_no_grokking_returns_none(self):
        losses = [2.0, 1.5, 1.2, 1.1, 1.0]
        variant = _make_variant(test_losses=losses)
        metrics = compute_variant_metrics(variant)
        assert metrics["grokking_onset_epoch"] is None
        assert metrics["failure_mode"] == "no_grokking"

    def test_neuron_dynamics_metrics_loaded(self):
        losses = [2.0, 1.5, 0.05, 0.001, 0.0001]
        nd = _make_nd_data(n_freq=4)
        variant = _make_variant(test_losses=losses, nd_data=nd)
        metrics = compute_variant_metrics(variant)
        assert metrics["frequency_band_count"] is not None
        assert metrics["frequency_band_count"] >= 1
        assert metrics["competition_window_start"] is not None
        assert metrics["competition_window_end"] is not None
        assert metrics["competition_window_duration"] >= 0

    def test_missing_neuron_dynamics_returns_none(self):
        losses = [2.0, 0.05, 0.001]
        variant = _make_variant(test_losses=losses, nd_data=None)
        metrics = compute_variant_metrics(variant)
        assert metrics["frequency_band_count"] is None
        assert metrics["competition_window_duration"] is None

    def test_repr_geometry_metrics_loaded(self):
        losses = [2.0, 0.05, 0.001]
        rg = _make_rg_data(circularity=0.9, fisher=15.0)
        variant = _make_variant(test_losses=losses, rg_data=rg)
        metrics = compute_variant_metrics(variant)
        assert metrics["final_circularity"] is not None
        assert abs(metrics["final_circularity"] - 0.9) < 0.01
        assert metrics["final_fisher_discriminant"] is not None

    def test_failure_mode_and_reasons_present(self):
        losses = [2.0, 1.0, 0.05, 0.001]
        variant = _make_variant(test_losses=losses)
        metrics = compute_variant_metrics(variant)
        assert "failure_mode" in metrics
        assert "failure_mode_reasons" in metrics
        assert isinstance(metrics["failure_mode_reasons"], list)


# ---------------------------------------------------------------------------
# load_family_comparison
# ---------------------------------------------------------------------------


class TestLoadFamilyComparison:
    def _make_family(self, variants: list) -> MagicMock:
        family = MagicMock()
        family.list_variants.return_value = variants
        return family

    def test_returns_dataframe(self):
        v1 = _make_variant([2.0, 0.05, 0.001], prime=113, seed=999)
        v2 = _make_variant([2.0, 2.0, 1.5], prime=101, seed=999)
        family = self._make_family([v1, v2])
        df = load_family_comparison(family)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_healthy_variant_sorted_first(self):
        healthy = _make_variant([2.0, 0.05, 0.001], prime=113, seed=999)
        no_grok = _make_variant([2.0, 2.0, 1.5], prime=101, seed=999)
        family = self._make_family([no_grok, healthy])
        df = load_family_comparison(family)
        # Healthy variant (early grokking) should be first
        assert df.iloc[0]["failure_mode"] == "healthy"

    def test_failure_mode_column_present(self):
        v = _make_variant([2.0, 0.05, 0.001])
        family = self._make_family([v])
        df = load_family_comparison(family)
        assert "failure_mode" in df.columns

    def test_none_metrics_appear_as_nan(self):
        v = _make_variant([2.0, 0.05, 0.001], nd_data=None, rg_data=None)
        family = self._make_family([v])
        df = load_family_comparison(family)
        assert pd.isna(df.iloc[0]["frequency_band_count"])
        assert pd.isna(df.iloc[0]["final_circularity"])


# ---------------------------------------------------------------------------
# render_metrics_table
# ---------------------------------------------------------------------------


class TestRenderMetricsTable:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "variant_name": "healthy_variant",
                    "prime": 113,
                    "seed": 999,
                    "grokking_onset_epoch": 5000,
                    "final_test_loss": 0.0001,
                    "frequency_band_count": 3,
                    "competition_window_duration": 3000,
                    "final_circularity": 0.9,
                    "final_fisher_discriminant": 12.0,
                    "failure_mode": "healthy",
                },
                {
                    "variant_name": "degraded_variant",
                    "prime": 101,
                    "seed": 999,
                    "grokking_onset_epoch": 18000,
                    "final_test_loss": 0.02,
                    "frequency_band_count": 1,
                    "competition_window_duration": 1000,
                    "final_circularity": 0.4,
                    "final_fisher_discriminant": 6.0,
                    "failure_mode": "degraded",
                },
            ]
        )

    def test_returns_figure(self):
        df = self._make_df()
        fig = render_metrics_table(df)
        assert isinstance(fig, go.Figure)

    def test_has_table_trace(self):
        df = self._make_df()
        fig = render_metrics_table(df)
        assert any(isinstance(t, go.Table) for t in fig.data)

    def test_sort_by_column(self):
        df = self._make_df()
        fig_asc = render_metrics_table(df, sort_by="final_test_loss", ascending=True)
        fig_desc = render_metrics_table(df, sort_by="final_test_loss", ascending=False)
        # Just verify both produce valid figures
        assert isinstance(fig_asc, go.Figure)
        assert isinstance(fig_desc, go.Figure)


# ---------------------------------------------------------------------------
# render_loss_curve_overlay
# ---------------------------------------------------------------------------


class TestRenderLossCurveOverlay:
    def _make_variant_with_losses(self, test_losses: list, prime: int = 113, seed: int = 999):
        return _make_variant(test_losses=test_losses, prime=prime, seed=seed)

    def test_returns_figure(self):
        v1 = self._make_variant_with_losses([2.0, 0.5, 0.05, 0.001])
        v2 = self._make_variant_with_losses([2.0, 1.5, 0.5, 0.05], prime=101)
        fig = render_loss_curve_overlay([v1, v2])
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_variant(self):
        v1 = self._make_variant_with_losses([2.0, 0.5, 0.05, 0.001])
        v2 = self._make_variant_with_losses([2.0, 1.5, 0.5, 0.05], prime=101)
        fig = render_loss_curve_overlay([v1, v2])
        assert len(fig.data) == 2

    def test_align_by_grokking_skips_no_grokkers(self):
        grokker = self._make_variant_with_losses([2.0, 0.5, 0.05, 0.001])
        no_grok = self._make_variant_with_losses([2.0, 1.5, 1.2, 1.1])
        fig = render_loss_curve_overlay([grokker, no_grok], align_by_grokking=True)
        # Only the grokking variant should appear
        assert len(fig.data) == 1

    def test_show_train_doubles_traces(self):
        v = self._make_variant_with_losses([2.0, 0.5, 0.05, 0.001])
        fig = render_loss_curve_overlay([v], show_train=True)
        assert len(fig.data) == 2  # test + train

    def test_failure_mode_coloring_accepted(self):
        v = self._make_variant_with_losses([2.0, 0.5, 0.05, 0.001])
        fig = render_loss_curve_overlay(
            [v], failure_modes={v.name: "healthy"}
        )
        assert isinstance(fig, go.Figure)
