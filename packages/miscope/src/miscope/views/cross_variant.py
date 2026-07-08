"""REQ_057: Cross-variant grokking health comparison.

Provides family-level metric extraction and failure mode classification.
Unlike the per-variant ViewCatalog/DataViewCatalog, these functions operate
across all variants in a family simultaneously.

Public API:
    compute_variant_metrics(variant) -> dict
    load_family_comparison(family) -> pd.DataFrame
    classify_failure_mode(metrics, rules) -> str
    ClassificationRules: adjustable thresholds for failure mode classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from miscope.analysis import neuron_frequency as nf
from miscope.analysis.band_concentration import (
    compute_band_concentration_at_epoch,
    compute_critical_mass_snapshot,
    compute_slope_cv,
)

if TYPE_CHECKING:
    from miscope.families.protocols import ModelFamily
    from miscope.families.variant import Variant


@dataclass
class ClassificationRules:
    """Adjustable thresholds for failure mode classification.

    All thresholds are explicit — classification output lists which rules fired.

    Attributes:
        grokking_threshold: test_loss value below which the model is considered
            to have grokked (default 0.1).
        early_grokking_epoch: grokking onset epoch below which a model is
            classified as "early_grokker" (default 9000).
        late_grokking_epoch: grokking onset epoch above which a model is
            classified as "late_grokker" (default 12000).
        degraded_test_loss: final test_loss above which a model with slow
            grokking is classified as "degraded" (default 1.0e-6).
        min_frequency_bands: minimum expected number of frequency bands for a
            healthy model. Models below this with slow grokking are flagged.
    Notes:
        - neuron specialization health metric
            (do neuron specialization counts rise together or out of balance)
        - neuron specialization diversity metric
            (is there enough frequency mix (low/mid/high)?)
            (this probably boils down to: is there a low-frequency band?)
    """

    # TODO: Correct logic around grokking_threshold. Needs to use threshold diff in second descent
    #   Can add threshold for measuring grokking success by difference between final train and test losses (within 1.0e-5)
    # grokking_threshold: float = 0.1
    early_grokking_epoch: int = 9000
    late_grokking_epoch: int = 12000
    degraded_test_loss: float = 1.0e-6
    min_frequency_bands: int = 2
    # REQ_065: second descent diagnostics
    second_descent_onset_diff_threshold: float = 0.8
    recovery_threshold: float = 0.2
    first_mover_neuron_threshold: float = (
        0.2  # TODO: Revisit this threshold. Probably need to be higher
    )


def compute_variant_metrics(
    variant: Variant,
    rules: ClassificationRules | None = None,
) -> dict[str, Any]:
    """Compute summary metrics for a single variant.

    Loads from metadata (cheap) and optionally from cross-epoch artifacts
    (neuron_dynamics, repr_geometry summary). Missing artifacts produce None
    values for the corresponding metrics rather than raising errors.

    Args:
        variant: The variant to analyze.
        rules: Classification rules used to determine grokking onset threshold.
            If None, uses default ClassificationRules().

    Returns:
        Dict with keys: variant_name, prime, seed, grokking_onset_epoch,
        final_test_loss, num_epochs, frequency_band_count,
        competition_window_start, competition_window_end,
        competition_window_duration, final_circularity,
        final_fisher_discriminant, failure_mode, failure_mode_reasons.
    """
    if rules is None:
        rules = ClassificationRules()

    # Domain Parameters
    prime = int(variant.model_config.get("prime", 0))
    seed = variant.model_config.get("seed")
    data_seed = variant.model_config.get("data_seed")

    # Training Metadata
    meta = variant.metadata
    train_losses = list(meta["train_losses"])
    test_losses = list(meta["test_losses"])
    num_epochs = len(test_losses)
    final_train_loss = float(train_losses[-1])
    final_test_loss = float(test_losses[-1])

    metrics: dict[str, Any] = {
        "variant_name": variant.name,
        "prime": prime,
        "seed": seed,
        "data_seed": data_seed,
        # "grokking_onset_epoch": None,
        "final_train_loss": final_train_loss,
        "final_test_loss": final_test_loss,
        "num_epochs": num_epochs,
        # REQ_065: second descent diagnostics — test loss trajectory
        "peak_test_loss": None,
        "peak_test_loss_epoch": None,
        "second_descent_onset_epoch": None,
        "second_descent_completion_epoch": None,
        "second_descent_survived": None,
        "post_descent_test_loss_increase": None,
        "frequency_band_count": None,
        # Competition Window metrics
        "competition_window_start": None,
        "competition_window_end": None,
        "competition_window_duration": None,
        "final_circularity": None,
        "final_fisher_discriminant": None,
        # REQ_065: frequency portfolio at second descent onset
        "second_descent_onset_frequency_bands": None,
        "second_descent_onset_has_low_band": None,
        "second_descent_onset_band_count": None,
        # REQ_065: first-mover frequency
        "first_mover_frequency": None,
        "first_mover_epoch": None,
        "first_mover_band": None,
        "first_mover_survived": None,
        # REQ_058: band concentration health metrics
        "midpoint_hhi": None,
        "midpoint_active_band_count": None,
        "midpoint_max_band_share": None,
        "onset_hhi": None,
        "onset_active_band_count": None,
        "onset_max_band_share": None,
        "slope_cv": None,
        "critical_mass_epoch": None,
        "critical_mass_hhi": None,
    }

    # REQ_110D: the conformed (epoch, neuron) -> dominant-frequency dimension is
    # loaded once from the warehouse and shared across the neuron-frequency
    # consumers (no more per-helper neuron_dynamics.npz reads).
    try:
        attribution = nf.load(variant)
    except FileNotFoundError:
        attribution = None

    _load_second_descent_metrics(metrics, test_losses, prime, rules, attribution)
    _load_neuron_dynamics_metrics(metrics, prime, attribution)
    _load_repr_geometry_metrics(variant, metrics)
    _load_band_concentration_metrics(metrics, prime, num_epochs, attribution)

    failure_mode, reasons = classify_failure_mode(metrics, rules)
    metrics["failure_mode"] = failure_mode
    metrics["failure_mode_reasons"] = reasons

    return metrics


def _load_neuron_dynamics_metrics(
    metrics: dict[str, Any],
    prime: int,
    attribution: nf.NeuronFrequencyAttribution | None,
) -> None:
    """Populate neuron_dynamics-derived metrics in place. No-op on missing artifact.

    Uses the artifact "uncommitted floor" threshold (≈3/(p//2)) to count *active*
    frequencies at the final epoch, and the artifact's stored commitment epochs for
    the competition window (distinct from the 0.7-threshold recompute the summary
    engine uses).
    """
    if attribution is None:
        return

    threshold = attribution.threshold if attribution.threshold is not None else 3.0 / (prime // 2)
    final_idx = attribution.n_epochs - 1
    active_freqs = attribution.specialized_frequencies(final_idx, threshold=threshold)
    metrics["frequency_band_count"] = len(active_freqs)

    commitment_epochs = attribution.commitment_epochs  # artifact values (low threshold)
    if commitment_epochs is not None:
        committed_epoch_values = commitment_epochs[~np.isnan(commitment_epochs)]
        if len(committed_epoch_values) > 0:
            metrics["competition_window_start"] = int(committed_epoch_values.min())
            metrics["competition_window_end"] = int(committed_epoch_values.max())
            metrics["competition_window_duration"] = (
                metrics["competition_window_end"] - metrics["competition_window_start"]
            )


def _load_repr_geometry_metrics(variant: Variant, metrics: dict[str, Any]) -> None:
    """Populate repr_geometry-derived metrics in place. No-op on missing artifact."""
    try:
        rg = variant.artifacts.load_summary("repr_geometry")
    except FileNotFoundError:
        return

    circularity = rg.get("resid_post_circularity")
    fisher = rg.get("resid_post_fisher_mean")

    if circularity is not None and len(circularity) > 0:
        metrics["final_circularity"] = float(circularity[-1])
    if fisher is not None and len(fisher) > 0:
        metrics["final_fisher_discriminant"] = float(fisher[-1])


_DEFAULT_CONCENTRATION_THRESHOLD = 0.7
_DEFAULT_CRITICAL_MASS_N = 100


def _load_band_concentration_metrics(
    metrics: dict[str, Any],
    prime: int,
    # grokking_onset_epoch: int | None,
    num_epochs: int,
    attribution: nf.NeuronFrequencyAttribution | None,
    threshold: float = _DEFAULT_CONCENTRATION_THRESHOLD,
    neuron_count_threshold: int = _DEFAULT_CRITICAL_MASS_N,
) -> None:
    """Populate band concentration metrics in place. No-op on missing artifact.

    The band-concentration functions consume the legacy ``neuron_dynamics`` array
    dict (0-indexed ``dominant_freq``); it is reconstructed from the conformed
    dimension via :meth:`NeuronFrequencyAttribution.as_legacy_arrays`, so those
    pure functions are unchanged while the source moves to the warehouse.
    """
    if attribution is None:
        return
    nd = attribution.as_legacy_arrays()

    epochs = nd["epochs"]
    n_epochs = len(epochs)

    midpoint_epoch_idx = n_epochs // 2
    midpoint = compute_band_concentration_at_epoch(nd, midpoint_epoch_idx, threshold, prime)
    metrics["midpoint_hhi"] = midpoint["hhi"]
    metrics["midpoint_active_band_count"] = midpoint["active_band_count"]
    metrics["midpoint_max_band_share"] = midpoint["max_band_share"]

    grokking_onset_epoch = metrics["second_descent_onset_epoch"]
    if grokking_onset_epoch is not None:
        onset_idx = int(np.searchsorted(epochs, grokking_onset_epoch))
        onset_idx = min(onset_idx, n_epochs - 1)
        onset = compute_band_concentration_at_epoch(nd, onset_idx, threshold, prime)
        metrics["onset_hhi"] = onset["hhi"]
        metrics["onset_active_band_count"] = onset["active_band_count"]
        metrics["onset_max_band_share"] = onset["max_band_share"]

    metrics["slope_cv"] = compute_slope_cv(nd, threshold, prime, grokking_onset_epoch)

    snapshot = compute_critical_mass_snapshot(nd, threshold, prime, neuron_count_threshold)
    if snapshot is not None:
        metrics["critical_mass_epoch"] = snapshot["epoch"]
        metrics["critical_mass_hhi"] = snapshot["hhi"]


def _classify_frequency_band(freq: int, prime: int) -> str:
    """Classify a frequency into low / mid / high band (single definition in nf)."""
    return nf.classify_band(freq, prime)


def _load_second_descent_metrics(
    metrics: dict[str, Any],
    test_losses: list[float],
    prime: int,
    rules: ClassificationRules,
    attribution: nf.NeuronFrequencyAttribution | None,
) -> None:
    """Populate second descent and first-mover metrics in place."""
    _compute_test_loss_trajectory(metrics, test_losses, rules)
    if attribution is None:
        return
    _compute_first_mover_metrics(metrics, attribution, prime, rules)
    _compute_descent_onset_portfolio(metrics, attribution, prime, rules)


def _compute_test_loss_trajectory(
    metrics: dict[str, Any],
    test_losses: list[float],
    rules: ClassificationRules,
) -> None:
    """Compute peak, second descent onset, and recovery from test loss series."""
    if not test_losses:
        return

    peak_loss = max(test_losses)
    peak_epoch = test_losses.index(peak_loss)
    metrics["peak_test_loss"] = peak_loss
    metrics["peak_test_loss_epoch"] = peak_epoch

    if peak_loss <= 0:
        return

    # Second descent onset: first epoch after peak where descent_fraction >= threshold
    onset_epoch = None
    for i in range(peak_epoch, len(test_losses)):
        descent_fraction = (peak_loss - test_losses[i]) / peak_loss
        if descent_fraction >= rules.second_descent_onset_diff_threshold:
            onset_epoch = i
            break

    metrics["second_descent_onset_epoch"] = onset_epoch

    if onset_epoch is None:
        return

    final_loss = test_losses[-1]
    metrics["second_descent_survived"] = final_loss <= rules.degraded_test_loss

    # Recovery: test loss climbs back by more than recovery_threshold * peak after onset
    min_post_onset = min(test_losses[onset_epoch:])
    recovery_floor = min_post_onset
    recovery_ceiling = recovery_floor + rules.recovery_threshold * peak_loss
    post_onset_max = max(test_losses[onset_epoch:])
    metrics["post_descent_test_loss_increase"] = post_onset_max > recovery_ceiling


def _compute_first_mover_metrics(
    metrics: dict[str, Any],
    attribution: nf.NeuronFrequencyAttribution,
    prime: int,
    rules: ClassificationRules,
) -> None:
    """Compute first-mover frequency metrics from the conformed dimension.

    REQ_110D correctness fix: ``first_mover_frequency`` is now 1-indexed, like
    every other frequency the platform reports (it was previously stored 0-indexed
    here, inconsistent with the descent-onset portfolio in this same module). The
    0.75 specialization threshold and the population-count gate are unchanged, so
    the *selection* (which frequency, which epoch) is identical — only the reported
    value shifts +1, and the band classification it feeds is now correct.
    """
    threshold = rules.first_mover_neuron_threshold * attribution.d_mlp
    final_freqs = attribution.final_specialized_frequencies(threshold=0.75)

    for epoch_idx in range(attribution.n_epochs):
        freq_counts = attribution.frequency_counts(epoch_idx, threshold=0.75)
        for freq, count in freq_counts.items():
            if count >= threshold:
                metrics["first_mover_frequency"] = freq
                metrics["first_mover_epoch"] = int(attribution.epochs[epoch_idx])
                metrics["first_mover_band"] = _classify_frequency_band(freq, prime)
                metrics["first_mover_survived"] = freq in final_freqs
                return


def _compute_descent_onset_portfolio(
    metrics: dict[str, Any],
    attribution: nf.NeuronFrequencyAttribution,
    prime: int,
    rules: ClassificationRules,
) -> None:
    """Compute frequency portfolio at second descent onset epoch."""
    onset_epoch = metrics.get("second_descent_onset_epoch")
    if onset_epoch is None:
        return

    onset_idx = attribution.epoch_index(onset_epoch)
    active_freqs = attribution.specialized_frequencies(onset_idx, threshold=0.7)

    if not active_freqs:
        metrics["second_descent_onset_committed_frequencies"] = []
        metrics["second_descent_onset_frequency_bands"] = []
        metrics["second_descent_onset_has_low_band"] = False
        metrics["second_descent_onset_band_count"] = 0
        return

    bands = [_classify_frequency_band(f, prime) for f in active_freqs]
    metrics["second_descent_onset_committed_frequencies"] = list(active_freqs)
    metrics["second_descent_onset_frequency_bands"] = bands
    metrics["second_descent_onset_has_low_band"] = "low" in bands
    metrics["second_descent_onset_band_count"] = len(set(bands))


def classify_failure_mode(
    metrics: dict[str, Any],
    rules: ClassificationRules | None = None,
) -> tuple[str, list[str]]:
    """Classify a variant's failure mode from its metrics.

    Returns a (category, reasons) tuple. Reasons list the specific rules that
    fired — classification is fully auditable.

    Categories (in priority order):
        no_grokking: never crossed the grokking threshold during training
        degraded_recovery: entered second descent, test loss climbed back
        degraded: high final test loss, never properly descended
        late_grokker: grokked but significantly past the expected window
        healthy: grokked on time, final loss acceptable

    Args:
        metrics: Dict as returned by compute_variant_metrics.
        rules: ClassificationRules instance. If None, uses defaults.

    Returns:
        Tuple of (category_str, reasons_list).
    """
    if rules is None:
        rules = ClassificationRules()

    reasons: list[str] = []
    onset = metrics.get("second_descent_onset_epoch")
    final_loss = metrics.get("final_test_loss")
    band_count = metrics.get("frequency_band_count")
    second_descent_onset = metrics.get("second_descent_onset_epoch")
    post_descent_recovery = metrics.get("post_descent_test_loss_increase")

    # 1. no_grokking: never crossed threshold
    if onset is None:
        reasons.append(f"never reached test_loss < {rules.degraded_test_loss}")
        return "no_grokking", reasons

    # 2. degraded_recovery: entered second descent, test loss climbed back
    if (
        second_descent_onset is not None
        and post_descent_recovery is True
        and (final_loss is None or final_loss > rules.degraded_test_loss)
    ):
        reasons.append(
            f"second_descent_onset={second_descent_onset}, post-descent recovery detected"
        )
        if final_loss is not None:
            reasons.append(f"final_test_loss={final_loss:.4f} > {rules.degraded_test_loss}")
        return "degraded_recovery", reasons

    # 3. degraded: high final loss, never properly descended
    if final_loss is not None and final_loss > rules.degraded_test_loss:
        reasons.append(f"final_test_loss={final_loss:.4f} > {rules.degraded_test_loss}")
        if band_count is not None and band_count < rules.min_frequency_bands:
            reasons.append(f"frequency_band_count={band_count} < {rules.min_frequency_bands}")
        return "degraded", reasons

    # 4. late_grokker: grokked but past expected window
    if onset > rules.late_grokking_epoch:
        reasons.append(f"grokking_onset={onset} > {rules.late_grokking_epoch}")
        return "late_grokker", reasons

    # 5. healthy
    reasons.append("grokking onset on time, final loss acceptable")
    return "healthy", reasons


def load_family_comparison(
    family: ModelFamily,
    rules: ClassificationRules | None = None,
) -> pd.DataFrame:
    """Compute summary metrics for all variants in a family.

    Iterates over all variants, calls compute_variant_metrics for each,
    and returns results as a sorted DataFrame. Missing artifact metrics
    appear as NaN in the DataFrame.

    Args:
        family: The loaded family to compare.
        rules: ClassificationRules for failure mode classification.

    Returns:
        DataFrame with one row per variant, sorted by grokking_onset_epoch
        (None/no-grokking variants last), then by final_test_loss.
    """
    if rules is None:
        rules = ClassificationRules()

    rows = []
    for variant in family.variants:
        metrics = compute_variant_metrics(variant, rules)
        metrics.pop("failure_mode_reasons", None)
        rows.append(metrics)

    df = pd.DataFrame(rows)

    # Sort: healthy variants by grokking onset, degraded/no-grokking last.
    df["_sort_onset"] = df["second_descent_onset_epoch"].fillna(float("inf"))
    df = df.sort_values(["_sort_onset", "final_test_loss"]).drop(columns=["_sort_onset"])
    df = df.reset_index(drop=True)

    return df
