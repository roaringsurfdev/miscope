"""Assemble the variant summary dict from conformed warehouse tables (REQ_144 Stage 3).

The 814-line imperative ``VariantAnalysisSummary`` is replaced by this thin reader:
the *stable* outcome fields come from the seven cluster derived tables (byte-parity,
Stage 2a), the *window* fields from the three window tables (Stage 2b), and the two
classifications stay as small auditable Python functions over the assembled row
(fork (b)). No analyzer summary npz is reopened and no warehouse path is composed —
every input is a queryable conformed fact (CoS #3), so the god-object's job becomes
"join the tables and add the two classifiers."

This module produces the same ``variant_summary.json`` dict the engine did, so its
consumers are untouched while the *source* moves onto the query surface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

import miscope.query
from miscope.views.cross_variant import ClassificationRules, classify_failure_mode

if TYPE_CHECKING:
    from miscope.families.variant import Variant

# Thresholds for the performance classification (carried from the engine verbatim).
_SECOND_DESCENT_ONSET_DIFF_THRESHOLD = 0.8
_SUCCESSFUL_TEST_LOSS_THRESHOLD = 1.0e-5
_REBOUND_TEST_LOSS_THRESHOLD = 0.2
_LATE_SECOND_DESCENT_EPOCH = 12000

# The seven stable-outcome cluster tables (Stage 2a), joined to one wide row.
_STABLE_TABLES = (
    "loss_outcomes",
    "neuron_threshold_outcomes",
    "dimensionality_outcomes",
    "competition_geometry_outcomes",
    "learned_frequencies_outcome",
    "onset_portfolio_outcome",
    "transient_outcome",
)

# Columns that are keys/provenance, not summary values — dropped when reading a row.
_NON_VALUE_COLUMNS = frozenset(
    {
        "variant_id",
        "prime",
        "seed",
        "data_seed",
        "run_set",
        "epoch",
        "window",
        "site",
        "head",
        "group",
        "frequency",
        "neuron",
        "row_id",
        "pc_index",
        "group_type",
        "operation_type",
        "analyzer_version",
        "conditions_satisfied",
        "trust_tier",
    }
)

_WINDOW_NAMES = ("first_descent", "plateau", "cascade", "second_descent", "final")

# window_metrics column -> the engine's (start/end-suffixed) summary key stem.
_METRIC_KEYS = {
    "train_loss": "train_loss",
    "test_loss": "test_loss",
    "pr_w_in": "resid_post_pr_w_in",
    "pr_w_out": "resid_post_pr_w_out",
    "circularity": "resid_post_circularity",
    "fisher_mean": "resid_post_fisher_mean",
}


def assemble_variant_summary(variant: Variant) -> dict[str, Any]:
    """Build the variant summary dict from the warehouse (engine replacement)."""
    with miscope.query.open(family=variant.family) as con:
        stable = _read_stable(con, variant.name)
        windows = _reassemble_windows(con, variant.name)

    summary: dict[str, Any] = {
        "prime": variant.params["prime"],
        "model_seed": variant.params["seed"],
        "data_seed": variant.params["data_seed"],
        "family": variant.family.name,
        "computed_at": datetime.now(UTC).isoformat(),
    }
    summary.update(stable)
    summary.update(windows)
    _apply_classifications(summary)
    return summary


def _read_stable(con: Any, variant_id: str) -> dict[str, Any]:
    """Merge the seven stable cluster rows into one flat ``{summary_key: value}`` dict."""
    values: dict[str, Any] = {}
    for table in _STABLE_TABLES:
        frame = con.df(f"SELECT * FROM {table} WHERE variant_id = '{variant_id}'")
        if frame.empty:
            continue
        row = frame.iloc[0]
        for col in frame.columns:
            if col in _NON_VALUE_COLUMNS:
                continue
            values[col] = _scalarize(row[col])
    return values


def _reassemble_windows(con: Any, variant_id: str) -> dict[str, Any]:
    """Rebuild the five ``*_window`` dicts from the window_* tables."""
    ranges = con.df(
        f'SELECT "window", start_epoch, end_epoch FROM window_ranges '
        f"WHERE variant_id = '{variant_id}'"
    )
    metrics = con.df(
        f'SELECT "window", boundary, {", ".join(_METRIC_KEYS)} FROM window_metrics '
        f"WHERE variant_id = '{variant_id}'"
    )
    freqs = con.df(
        f'SELECT "window", frequency, boundary, role, band FROM window_frequencies '
        f"WHERE variant_id = '{variant_id}'"
    )
    bounds = {
        r["window"]: (int(r["start_epoch"]), int(r["end_epoch"])) for _, r in ranges.iterrows()
    }
    return {
        f"{name}_window": _one_window(name, bounds.get(name), metrics, freqs)
        for name in _WINDOW_NAMES
    }


def _one_window(
    name: str, bound: tuple[int, int] | None, metrics: Any, freqs: Any
) -> dict[str, Any]:
    """Assemble one window dict, or the engine's skip sentinel for a degenerate span."""
    if bound is None:
        return {"start_epoch": 0, "end_epoch": 0}
    start_epoch, end_epoch = bound
    window: dict[str, Any] = {"start_epoch": start_epoch, "end_epoch": end_epoch}
    if start_epoch >= end_epoch:
        window["skipped"] = True
        window["skip_reason"] = "start_epoch >= end_epoch"
        return window

    for boundary in ("start", "end"):
        row = metrics[(metrics["window"] == name) & (metrics["boundary"] == boundary)]
        if not row.empty:
            for col, stem in _METRIC_KEYS.items():
                window[f"{stem}_{boundary}"] = _scalarize(row.iloc[0][col])
        window[f"learned_frequencies_{boundary}"] = _freq_list(freqs, name, boundary, "specialized")
        window[f"committed_frequencies_{boundary}"] = _freq_list(freqs, name, boundary, "committed")
        window[f"learned_frequency_bands_{boundary}"] = _band_set(freqs, name, boundary)
    window["frequency_gains"] = _freq_list(freqs, name, "window", "gain")
    window["frequency_losses"] = _freq_list(freqs, name, "window", "loss")
    return window


def _freq_list(freqs: Any, name: str, boundary: str, role: str) -> list[int]:
    """Sorted 1-indexed frequencies for one (window, boundary, role) slice."""
    sel = freqs[
        (freqs["window"] == name) & (freqs["boundary"] == boundary) & (freqs["role"] == role)
    ]
    return sorted(int(f) for f in sel["frequency"])


def _band_set(freqs: Any, name: str, boundary: str) -> list[str]:
    """Unique bands of the specialized frequencies at a boundary (engine's list(set(...)))."""
    sel = freqs[
        (freqs["window"] == name)
        & (freqs["boundary"] == boundary)
        & (freqs["role"] == "specialized")
    ]
    return list({str(b) for b in sel["band"]})


def _apply_classifications(summary: dict[str, Any]) -> None:
    """Add failure_mode/reasons and performance_classification over the assembled row (fork b)."""
    failure_mode, reasons = classify_failure_mode(
        {
            "second_descent_onset_epoch": summary.get("second_descent_onset_epoch"),
            "final_test_loss": summary.get("test_loss_final"),
            "post_descent_test_loss_increase": summary.get("post_descent_test_loss_increase"),
            "frequency_band_count": None,
        },
        ClassificationRules(),
    )
    summary["failure_mode"] = failure_mode
    summary["failure_mode_reasons"] = reasons
    summary["performance_classification"] = _classify_performance(summary)


def _classify_performance(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """The engine's performance classification — a pure function of the outcomes row."""
    reasons: list[str] = []
    onset = summary["second_descent_onset_epoch"]
    final_test_loss = summary["test_loss_final"]
    min_test_loss = summary["test_loss_min"]
    rebound = (final_test_loss - min_test_loss) >= _REBOUND_TEST_LOSS_THRESHOLD

    if onset is None or onset == 0:
        reasons.append(
            f"test loss never dropped more than {_SECOND_DESCENT_ONSET_DIFF_THRESHOLD} between epochs"
        )
        return "no_second_descent", reasons
    if rebound and (final_test_loss is None or final_test_loss > _SUCCESSFUL_TEST_LOSS_THRESHOLD):
        reasons.append(f"second_descent_onset={onset}, post-descent recovery detected")
        if final_test_loss is not None:
            reasons.append(
                f"final_test_loss={final_test_loss:.6f} > {_SUCCESSFUL_TEST_LOSS_THRESHOLD}"
            )
        return "degraded_rebound", reasons
    if final_test_loss is not None and final_test_loss > _SUCCESSFUL_TEST_LOSS_THRESHOLD:
        reasons.append(f"final_test_loss={final_test_loss:.6f} > {_SUCCESSFUL_TEST_LOSS_THRESHOLD}")
        return "degraded", reasons
    if onset > _LATE_SECOND_DESCENT_EPOCH:
        reasons.append(f"second_descent_onset={onset} > {_LATE_SECOND_DESCENT_EPOCH}")
        return "late_grokker", reasons
    reasons.append("grokking onset on time, final loss acceptable")
    return "healthy", reasons


def _scalarize(value: Any) -> Any:
    """Coerce a pandas/numpy cell to a JSON-friendly Python scalar or list."""
    if value is None:
        return None
    if isinstance(value, (list, np.ndarray)):
        return [_scalarize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
