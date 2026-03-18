"""REQ_074: Variant Outcome Registry.

Computes a per-variant summary JSON and aggregates them into a family-level
variant_registry.json.  All heavy lifting (second descent detection, failure
mode classification) delegates to cross_variant.compute_variant_metrics so
logic stays in one place.

Public API:
    CANONICAL_SPECIALIZATION_THRESHOLD: float
    extract_learned_frequencies(variant, canonical_threshold) -> (list[int], float) | (None, None)
    compute_variant_summary(variant, canonical_threshold, rules) -> dict
    write_variant_summary(variant, canonical_threshold, rules) -> Path
    build_variant_registry(results_dir, family_name) -> Path
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from miscope.views.cross_variant import ClassificationRules, compute_variant_metrics

if TYPE_CHECKING:
    from miscope.families.variant import Variant


CANONICAL_SPECIALIZATION_THRESHOLD: float = 0.10


def extract_learned_frequencies(
    variant: Variant,
    canonical_threshold: float = CANONICAL_SPECIALIZATION_THRESHOLD,
) -> tuple[list[int], float] | tuple[None, None]:
    """Return frequencies with neuron specialization above threshold at the final epoch.

    Counts committed neurons per frequency at the final checkpoint using the
    neuron_dynamics cross-epoch artifact.  A frequency qualifies if its
    committed neuron count / d_mlp >= canonical_threshold.

    Args:
        variant: The variant to inspect.
        canonical_threshold: Fraction of d_mlp neurons required to qualify.

    Returns:
        (learned_frequencies, canonical_threshold) if artifact is present,
        (None, None) if neuron_dynamics artifact is absent.
    """
    try:
        nd = variant.artifacts.load_cross_epoch("neuron_dynamics")
    except FileNotFoundError:
        return None, None

    dominant_freq = nd["dominant_freq"]  # (n_epochs, d_mlp)
    max_frac = nd["max_frac"]           # (n_epochs, d_mlp)
    d_mlp = dominant_freq.shape[1]
    threshold_count = canonical_threshold * d_mlp

    prime = int(variant.model_config.get("prime", 0))
    n_freqs = prime // 2 if prime > 0 else dominant_freq.shape[1]

    committed_final = max_frac[-1] >= (
        float(nd["threshold"][0]) if nd["threshold"].size > 0 else 3.0 / n_freqs
    )
    final_dominant = dominant_freq[-1]

    freq_counts: dict[int, int] = {}
    for neuron_idx in range(d_mlp):
        if committed_final[neuron_idx]:
            freq = int(final_dominant[neuron_idx])
            freq_counts[freq] = freq_counts.get(freq, 0) + 1

    learned = sorted(f for f, cnt in freq_counts.items() if cnt >= threshold_count)
    return learned, canonical_threshold


def _extract_max_resid_post_circularity(variant: Variant) -> float | None:
    """Return maximum resid_post circularity across all checkpoints, or None."""
    try:
        rg = variant.artifacts.load_summary("repr_geometry")
    except FileNotFoundError:
        return None

    circularity = rg.get("resid_post_circularity")
    if circularity is None or len(circularity) == 0:
        return None
    return float(np.max(circularity))


def _committed_frequencies_at_epoch(
    nd: dict,
    epoch: int,
    canonical_threshold: float,
    d_mlp: int,
) -> list[int]:
    """Return frequencies committed above threshold at a specific epoch."""
    epochs = nd["epochs"]
    dominant_freq = nd["dominant_freq"]
    max_frac = nd["max_frac"]

    epoch_idx = int(np.searchsorted(epochs, epoch))
    epoch_idx = min(epoch_idx, len(epochs) - 1)

    threshold_count = canonical_threshold * d_mlp
    n_freqs = dominant_freq.shape[1]

    committed_mask = max_frac[epoch_idx] >= (
        float(nd["threshold"][0]) if nd["threshold"].size > 0 else 3.0 / n_freqs
    )
    freq_at_epoch = dominant_freq[epoch_idx]

    freq_counts: dict[int, int] = {}
    for neuron_idx in range(n_freqs):
        if committed_mask[neuron_idx]:
            freq = int(freq_at_epoch[neuron_idx])
            freq_counts[freq] = freq_counts.get(freq, 0) + 1

    return sorted(f for f, cnt in freq_counts.items() if cnt >= threshold_count)


def _compute_handshake_fields(
    variant: Variant,
    learned_frequencies: list[int] | None,
    second_descent_onset_epoch: int | None,
    canonical_threshold: float,
) -> dict[str, Any]:
    """Compute committed_frequencies_at_onset and handshake_failures fields."""
    result: dict[str, Any] = {
        "committed_frequencies_at_onset": None,
        "handshake_failures": None,
        "handshake_succeeded": None,
    }

    if second_descent_onset_epoch is None or learned_frequencies is None:
        return result

    try:
        nd = variant.artifacts.load_cross_epoch("neuron_dynamics")
    except FileNotFoundError:
        return result

    d_mlp = nd["dominant_freq"].shape[1]
    committed_at_onset = _committed_frequencies_at_epoch(
        nd, second_descent_onset_epoch, canonical_threshold, d_mlp
    )

    learned_set = set(learned_frequencies)
    failures = [f for f in committed_at_onset if f not in learned_set]

    result["committed_frequencies_at_onset"] = committed_at_onset
    result["handshake_failures"] = failures
    result["handshake_succeeded"] = len(failures) == 0

    return result


def compute_variant_summary(
    variant: Variant,
    canonical_threshold: float = CANONICAL_SPECIALIZATION_THRESHOLD,
    rules: ClassificationRules | None = None,
) -> dict[str, Any]:
    """Compute the full variant summary dict.

    Delegates to compute_variant_metrics for second descent detection and
    failure mode classification.  Augments with learned_frequencies,
    handshake analysis, and max_resid_post_circularity.

    Window-dependent handshake fields are populated when the neuron_dynamics
    artifact is present and second descent was detected; otherwise None.

    Args:
        variant: The variant to summarize.
        canonical_threshold: Neuron fraction threshold for frequency qualification.
        rules: Classification rules for failure mode. Defaults to ClassificationRules().

    Returns:
        Dict suitable for JSON serialization as variant_summary.json.
    """
    if rules is None:
        rules = ClassificationRules()

    base = compute_variant_metrics(variant, rules)

    learned_frequencies, _ = extract_learned_frequencies(variant, canonical_threshold)
    learned_frequency_count = len(learned_frequencies) if learned_frequencies is not None else None

    handshake = _compute_handshake_fields(
        variant,
        learned_frequencies,
        base.get("second_descent_onset_epoch"),
        canonical_threshold,
    )

    max_circularity = _extract_max_resid_post_circularity(variant)

    prime = int(variant.model_config.get("prime", 0))
    model_seed = variant.model_config.get("seed")
    data_seed = variant.model_config.get("data_seed")

    summary: dict[str, Any] = {
        # Identity
        "prime": prime,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "family": variant.family.name,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        # Learned frequencies
        "learned_frequencies": learned_frequencies,
        "learned_frequency_count": learned_frequency_count,
        "canonical_specialization_threshold": canonical_threshold,
        # Second descent window
        "second_descent_onset_epoch": base.get("second_descent_onset_epoch"),
        "second_descent_completion_epoch": base.get("second_descent_completion_epoch"),
        "second_descent_survived": base.get("second_descent_survived"),
        # Handshake analysis
        "committed_frequencies_at_onset": handshake["committed_frequencies_at_onset"],
        "handshake_failures": handshake["handshake_failures"],
        "handshake_succeeded": handshake["handshake_succeeded"],
        # Outcome metrics
        "failure_mode": base.get("failure_mode"),
        "max_resid_post_circularity": max_circularity,
        "final_specialized_frequency_count": learned_frequency_count,
        "peak_test_loss_epoch": base.get("peak_test_loss_epoch"),
        "final_test_loss": base.get("final_test_loss"),
    }

    return summary


def write_variant_summary(
    variant: Variant,
    canonical_threshold: float = CANONICAL_SPECIALIZATION_THRESHOLD,
    rules: ClassificationRules | None = None,
) -> Path:
    """Compute and write variant_summary.json to the variant's results directory.

    Overwrites any existing file.  Calling this again with new artifacts or a
    different threshold will always produce a fresh result.

    Args:
        variant: The variant to summarize.
        canonical_threshold: Neuron fraction threshold.
        rules: Classification rules. Defaults to ClassificationRules().

    Returns:
        Path to the written variant_summary.json file.
    """
    summary = compute_variant_summary(variant, canonical_threshold, rules)
    output_path = variant.variant_dir / "variant_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path


def build_variant_registry(results_dir: Path | str, family_name: str) -> Path:
    """Aggregate all existing variant_summary.json files into variant_registry.json.

    Scans all subdirectories of results_dir/family_name for variant_summary.json
    files and assembles them into a single registry array, adding a variant_id
    field ("{prime}_{model_seed}_{data_seed}") to each entry.

    Reads from already-written summaries only — does not re-run any artifact
    analysis.  If a variant's summary is updated, re-running this call refreshes
    its entry while leaving other entries unchanged.

    Args:
        results_dir: Root results directory.
        family_name: Name of the model family subdirectory.

    Returns:
        Path to the written variant_registry.json file.
    """
    family_dir = Path(results_dir) / family_name
    registry: list[dict[str, Any]] = []

    for summary_path in sorted(family_dir.glob("*/variant_summary.json")):
        entry = json.loads(summary_path.read_text())
        prime = entry.get("prime", "?")
        model_seed = entry.get("model_seed", "?")
        data_seed = entry.get("data_seed", "?")
        entry["variant_id"] = f"{prime}_{model_seed}_{data_seed}"
        registry.append(entry)

    output_path = family_dir / "variant_registry.json"
    output_path.write_text(json.dumps(registry, indent=2))
    return output_path
