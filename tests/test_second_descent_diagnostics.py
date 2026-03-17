"""Tests for REQ_065: Second Descent Diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from miscope.views.cross_variant import (
    ClassificationRules,
    _compute_first_mover_metrics,
    _compute_test_loss_trajectory,
    classify_failure_mode,
    compute_variant_metrics,
    load_family_comparison,
)

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

_NEW_FIELDS = [
    "peak_test_loss",
    "peak_test_loss_epoch",
    "second_descent_onset_epoch",
    "second_descent_survived",
    "post_descent_recovery",
    "descent_onset_frequency_bands",
    "descent_onset_has_low_band",
    "descent_onset_band_count",
    "first_mover_frequency",
    "first_mover_epoch",
    "first_mover_band",
    "first_mover_survived",
]


def _make_variant(
    test_losses: list[float],
    prime: int = 113,
    seed: int = 999,
    nd_data: dict | None = None,
) -> MagicMock:
    variant = MagicMock()
    variant.name = f"mock_p{prime}_seed{seed}"
    variant.model_config = {"prime": prime, "seed": seed}
    variant.metadata = {"test_losses": test_losses, "train_losses": test_losses[:]}

    def mock_load_cross_epoch(name):
        if name == "neuron_dynamics" and nd_data is not None:
            return nd_data
        raise FileNotFoundError(f"No cross-epoch for {name}")

    def mock_load_summary(name):
        raise FileNotFoundError(f"No summary for {name}")

    variant.artifacts.load_cross_epoch.side_effect = mock_load_cross_epoch
    variant.artifacts.load_summary.side_effect = mock_load_summary
    return variant


def _make_nd_data(
    n_epochs: int,
    d_mlp: int,
    freq_per_neuron: list[int],
    max_frac_val: float = 0.9,
    epochs: list[int] | None = None,
) -> dict:
    """Build neuron_dynamics data with specified frequency assignments per neuron."""
    if epochs is None:
        epochs = list(range(n_epochs))
    epochs_arr = np.array(epochs, dtype=np.int64)
    dominant_freq = np.zeros((n_epochs, d_mlp), dtype=np.int32)
    for neuron_idx, freq in enumerate(freq_per_neuron):
        dominant_freq[:, neuron_idx] = freq
    max_frac = np.full((n_epochs, d_mlp), max_frac_val, dtype=np.float32)
    commitment_epochs = np.full(d_mlp, float("nan"))
    threshold = np.array([0.75])
    return {
        "epochs": epochs_arr,
        "dominant_freq": dominant_freq,
        "max_frac": max_frac,
        "switch_counts": np.zeros(d_mlp, dtype=np.int32),
        "commitment_epochs": commitment_epochs,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Unit: _compute_test_loss_trajectory
# ---------------------------------------------------------------------------


class TestSecondDescentOnset:
    def test_onset_is_none_when_never_descends_80_percent(self):
        """Test loss rises then descends only 50% — onset should be None."""
        # Peak=2.0 at idx 1; max descent = (2.0-1.0)/2.0 = 50% < 80%
        losses = [1.0, 2.0, 1.8, 1.5, 1.0]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["second_descent_onset_epoch"] is None

    def test_onset_detected_at_correct_epoch(self):
        """Second descent onset is the first epoch where descent_fraction >= 0.8."""
        # Peak=2.0 at idx 0; descent_fraction at idx 3 = (2.0-0.4)/2.0 = 0.8
        losses = [2.0, 1.5, 0.8, 0.4, 0.2]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["second_descent_onset_epoch"] == 3

    def test_peak_epoch_recorded(self):
        losses = [1.0, 3.0, 2.0, 1.0, 0.5]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["peak_test_loss"] == pytest.approx(3.0)
        assert metrics["peak_test_loss_epoch"] == 1

    def test_peak_at_first_epoch(self):
        """Peak at epoch 0 — onset search begins from start."""
        losses = [4.0, 3.0, 1.5, 0.8, 0.4]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["peak_test_loss_epoch"] == 0
        # descent at idx 3 = (4.0-0.8)/4.0 = 0.8 >= 0.8 → first qualifying epoch
        assert metrics["second_descent_onset_epoch"] == 3


class TestPostDescentRecovery:
    def test_recovery_true_for_descend_then_climb(self):
        """Loss descends 80%+ from peak, then climbs back above recovery_threshold."""
        # Peak=2.0 at idx 1.
        # At idx 3: descent=(2.0-0.2)/2.0=0.9 >= 0.8 → onset=3.
        # Post-onset: [0.2, 1.5]. min=0.2, recovery_ceiling=0.2+0.2*2.0=0.6. max=1.5>0.6.
        losses = [1.0, 2.0, 0.8, 0.2, 1.5]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["second_descent_onset_epoch"] is not None
        assert metrics["post_descent_recovery"] is True

    def test_recovery_false_for_clean_descent(self):
        """Loss descends cleanly — no recovery."""
        # Peak=2.0 at idx 0; descends cleanly to 0.1.
        # At idx 4: descent=(2.0-0.1)/2.0=0.95 >= 0.8 → onset=4.
        # Post-onset: [0.1]. min=0.1, recovery_ceiling=0.1+0.2*2.0=0.5. max=0.1 < 0.5.
        losses = [2.0, 1.2, 0.6, 0.2, 0.1]
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["second_descent_onset_epoch"] is not None
        assert metrics["post_descent_recovery"] is False

    def test_recovery_none_when_no_onset(self):
        """post_descent_recovery is None when second_descent_onset_epoch is None."""
        losses = [2.0, 1.8, 1.5, 1.2, 1.0]  # Never descends 80%
        metrics: dict = {}
        _compute_test_loss_trajectory(metrics, losses, ClassificationRules())
        assert metrics["second_descent_onset_epoch"] is None
        assert (
            "post_descent_recovery" not in metrics or metrics.get("post_descent_recovery") is None
        )


# ---------------------------------------------------------------------------
# Unit: _compute_first_mover_metrics
# ---------------------------------------------------------------------------


class TestFirstMoverMetrics:
    def _make_metrics(self):
        return {
            "first_mover_frequency": None,
            "first_mover_epoch": None,
            "first_mover_band": None,
            "first_mover_survived": None,
        }

    def test_first_mover_frequency_detected(self):
        """First frequency to reach 20% of d_mlp neurons is identified."""
        # d_mlp=10, threshold=0.2*10=2 neurons. Assign 3 neurons to freq=5.
        prime = 113
        d_mlp = 10
        # freq_per_neuron: 3 neurons → freq 5, rest → freq 20
        freq_per_neuron = [5, 5, 5, 20, 20, 20, 20, 20, 20, 20]
        nd = _make_nd_data(
            n_epochs=3, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100, 200]
        )
        metrics = self._make_metrics()
        rules = ClassificationRules()
        _compute_first_mover_metrics(metrics, nd, prime, rules)
        assert metrics["first_mover_frequency"] == 5

    def test_first_mover_epoch_recorded(self):
        """first_mover_epoch is the epoch at which the threshold was first crossed."""
        prime = 113
        d_mlp = 10
        freq_per_neuron = [5, 5, 5, 20, 20, 20, 20, 20, 20, 20]
        nd = _make_nd_data(
            n_epochs=3, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100, 200]
        )
        metrics = self._make_metrics()
        _compute_first_mover_metrics(metrics, nd, prime, ClassificationRules())
        assert metrics["first_mover_epoch"] == 0

    def test_first_mover_band_low(self):
        """freq=5 with prime=113: 5 <= 113//4=28 → low band."""
        prime = 113
        d_mlp = 10
        freq_per_neuron = [5, 5, 5, 20, 20, 20, 20, 20, 20, 20]
        nd = _make_nd_data(
            n_epochs=2, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100]
        )
        metrics = self._make_metrics()
        _compute_first_mover_metrics(metrics, nd, prime, ClassificationRules())
        assert metrics["first_mover_band"] == "low"

    def test_first_mover_survived_true(self):
        """first_mover frequency still active in final epoch → survived=True."""
        prime = 113
        d_mlp = 10
        freq_per_neuron = [5, 5, 5, 20, 20, 20, 20, 20, 20, 20]
        nd = _make_nd_data(
            n_epochs=2, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100]
        )
        metrics = self._make_metrics()
        _compute_first_mover_metrics(metrics, nd, prime, ClassificationRules())
        assert metrics["first_mover_survived"] is True

    def test_first_mover_survived_false_when_lost(self):
        """First-mover frequency present at epoch 0 but absent at final epoch."""
        prime = 113
        d_mlp = 10
        n_epochs = 2
        epochs_arr = np.array([0, 100], dtype=np.int64)
        dominant_freq = np.zeros((n_epochs, d_mlp), dtype=np.int32)
        max_frac = np.full((n_epochs, d_mlp), 0.9, dtype=np.float32)
        # Epoch 0: 3 neurons → freq 5 (first mover)
        dominant_freq[0, :3] = 5
        dominant_freq[0, 3:] = 20
        # Epoch 1 (final): all neurons → freq 20 (freq 5 lost)
        dominant_freq[1, :] = 20
        nd = {
            "epochs": epochs_arr,
            "dominant_freq": dominant_freq,
            "max_frac": max_frac,
            "switch_counts": np.zeros(d_mlp, dtype=np.int32),
            "commitment_epochs": np.full(d_mlp, float("nan")),
            "threshold": np.array([0.75]),
        }
        metrics = self._make_metrics()
        _compute_first_mover_metrics(metrics, nd, prime, ClassificationRules())
        assert metrics["first_mover_frequency"] == 5
        assert metrics["first_mover_survived"] is False

    def test_no_first_mover_when_below_threshold(self):
        """No frequency reaches threshold — all first_mover fields remain None."""
        prime = 113
        d_mlp = 10
        # Only 1 neuron per frequency — threshold=2, never reached
        freq_per_neuron = list(range(d_mlp))  # all different frequencies
        nd = _make_nd_data(
            n_epochs=2, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100]
        )
        metrics = self._make_metrics()
        _compute_first_mover_metrics(metrics, nd, prime, ClassificationRules())
        assert metrics["first_mover_frequency"] is None


# ---------------------------------------------------------------------------
# Unit: classify_failure_mode — degraded_recovery
# ---------------------------------------------------------------------------


class TestClassifyFailureModeREQ065:
    def _base_metrics(self, **overrides) -> dict:
        base = {
            "grokking_onset_epoch": 5000,
            "final_test_loss": 0.05,
            "frequency_band_count": 2,
            "second_descent_onset_epoch": None,
            "post_descent_recovery": None,
        }
        base.update(overrides)
        return base

    def test_degraded_recovery_returned(self):
        """Returns degraded_recovery when descent onset set and recovery occurred."""
        metrics = self._base_metrics(
            second_descent_onset_epoch=8000,
            post_descent_recovery=True,
            final_test_loss=0.05,  # > grokking_threshold=0.1? No, 0.05 < 0.1
        )
        # Need final_test_loss > grokking_threshold for degraded_recovery
        metrics["final_test_loss"] = 0.15
        mode, reasons = classify_failure_mode(metrics)
        assert mode == "degraded_recovery"
        assert any("second_descent_onset" in r for r in reasons)

    def test_degraded_recovery_priority_over_degraded(self):
        """degraded_recovery takes priority over degraded when both conditions fire."""
        metrics = self._base_metrics(
            second_descent_onset_epoch=8000,
            post_descent_recovery=True,
            final_test_loss=0.5,  # high → would trigger degraded
            frequency_band_count=1,  # low → would add to degraded reasons
        )
        mode, _ = classify_failure_mode(metrics)
        assert mode == "degraded_recovery"

    def test_no_degraded_recovery_when_recovery_false(self):
        """post_descent_recovery=False does not trigger degraded_recovery."""
        metrics = self._base_metrics(
            second_descent_onset_epoch=8000,
            post_descent_recovery=False,
            final_test_loss=0.15,
        )
        mode, _ = classify_failure_mode(metrics)
        assert mode != "degraded_recovery"

    def test_no_degraded_recovery_when_onset_none(self):
        """No onset → degraded_recovery not triggered even if recovery=True."""
        metrics = self._base_metrics(
            second_descent_onset_epoch=None,
            post_descent_recovery=True,
            final_test_loss=0.15,
        )
        mode, _ = classify_failure_mode(metrics)
        assert mode != "degraded_recovery"

    def test_no_degraded_recovery_when_survived(self):
        """degraded_recovery not triggered when final_loss <= grokking_threshold."""
        metrics = self._base_metrics(
            second_descent_onset_epoch=8000,
            post_descent_recovery=True,
            final_test_loss=0.001,  # <= grokking_threshold
        )
        mode, _ = classify_failure_mode(metrics)
        assert mode != "degraded_recovery"

    def test_classification_order_no_grokking_first(self):
        """no_grokking takes priority over everything."""
        metrics = self._base_metrics(
            grokking_onset_epoch=None,  # no grokking
            second_descent_onset_epoch=8000,
            post_descent_recovery=True,
            final_test_loss=0.5,
        )
        mode, _ = classify_failure_mode(metrics)
        assert mode == "no_grokking"


# ---------------------------------------------------------------------------
# Unit: compute_variant_metrics — new fields present
# ---------------------------------------------------------------------------


class TestComputeVariantMetricsREQ065:
    def test_new_fields_always_present(self):
        """All REQ_065 fields appear in metrics even without neuron_dynamics artifact."""
        losses = [2.0, 1.5, 0.8, 0.4, 0.05]
        variant = _make_variant(test_losses=losses, nd_data=None)
        metrics = compute_variant_metrics(variant)
        for field in _NEW_FIELDS:
            assert field in metrics, f"Missing field: {field}"

    def test_second_descent_onset_populated_from_loss_series(self):
        """second_descent_onset_epoch is derived from test_losses in metadata."""
        # Peak=2.0 at idx 0; at idx 4: (2.0-0.4)/2.0=0.8 → onset_epoch=4
        losses = [2.0, 1.5, 0.9, 0.6, 0.4]
        variant = _make_variant(test_losses=losses)
        metrics = compute_variant_metrics(variant)
        assert metrics["second_descent_onset_epoch"] == 4

    def test_first_mover_populated_from_nd_artifact(self):
        """first_mover_frequency uses neuron_dynamics artifact when available."""
        losses = [2.0, 1.0, 0.5, 0.05, 0.001]
        prime = 113
        d_mlp = 20
        freq_per_neuron = [5] * 5 + [20] * 15  # 5 neurons on freq=5 → threshold=4
        nd = _make_nd_data(
            n_epochs=3, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100, 200]
        )
        variant = _make_variant(test_losses=losses, prime=prime, nd_data=nd)
        metrics = compute_variant_metrics(variant)
        assert metrics["first_mover_frequency"] == 5
        assert metrics["first_mover_band"] in ("low", "mid", "high")

    def test_portfolio_fields_none_without_nd(self):
        """descent_onset_* fields are None when neuron_dynamics artifact absent."""
        losses = [2.0, 1.5, 0.8, 0.4, 0.4]
        variant = _make_variant(test_losses=losses, nd_data=None)
        metrics = compute_variant_metrics(variant)
        assert metrics["descent_onset_frequency_bands"] is None
        assert metrics["descent_onset_has_low_band"] is None
        assert metrics["descent_onset_band_count"] is None


# ---------------------------------------------------------------------------
# Integration: load_family_comparison includes all new fields
# ---------------------------------------------------------------------------


class TestLoadFamilyComparisonREQ065:
    def _make_family(self, variants) -> MagicMock:
        family = MagicMock()
        family.list_variants.return_value = variants
        return family

    def test_all_new_fields_in_dataframe(self):
        """load_family_comparison DataFrame includes all REQ_065 fields."""
        losses = [2.0, 1.0, 0.5, 0.05, 0.001]
        v = _make_variant(test_losses=losses)
        family = self._make_family([v])
        df = load_family_comparison(family)
        for field in _NEW_FIELDS:
            assert field in df.columns, f"Missing column: {field}"

    def test_new_fields_nan_when_nd_missing(self):
        """Artifact-dependent fields are NaN in DataFrame when artifact absent."""
        losses = [2.0, 1.0, 0.5, 0.05, 0.001]
        v = _make_variant(test_losses=losses, nd_data=None)
        family = self._make_family([v])
        df = load_family_comparison(family)
        assert pd.isna(df.iloc[0]["descent_onset_frequency_bands"])
        assert pd.isna(df.iloc[0]["first_mover_frequency"])

    def test_runs_without_error_mixed_variants(self):
        """No error when mixing variants with and without neuron_dynamics."""
        d_mlp = 10
        freq_per_neuron = [5, 5, 5, 20, 20, 20, 20, 20, 20, 20]
        nd = _make_nd_data(
            n_epochs=3, d_mlp=d_mlp, freq_per_neuron=freq_per_neuron, epochs=[0, 100, 200]
        )
        v_with_nd = _make_variant([2.0, 1.0, 0.5, 0.05, 0.001], nd_data=nd)
        v_no_nd = _make_variant([2.0, 2.0, 1.5, 1.2, 1.0], nd_data=None)
        family = self._make_family([v_with_nd, v_no_nd])
        df = load_family_comparison(family)
        assert len(df) == 2
        assert "second_descent_onset_epoch" in df.columns
