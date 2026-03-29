"""Tests for REQ_080: Artifact freshness reporting.

CoS coverage:
- Unit: PerEpochFreshness / CrossEpochFreshness status labels and is_fresh logic.
- Unit: FreshnessReport.any_stale and format() output.
- Unit: _scan_epoch_files correctly parses epoch_*.npz filenames.
- Unit: _read_covered_epoch_count returns correct count or -1 on missing key.
- Integration: check_freshness on a synthetic fixture directory produces correct
  PerEpochFreshness and CrossEpochFreshness entries.
- Unit: cross_epoch_is_stale returns True when dependency epochs > covered count.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from miscope.analysis.freshness import (
    CrossEpochFreshness,
    FreshnessReport,
    PerEpochFreshness,
    _read_covered_epoch_count,
    _scan_epoch_files,
    check_freshness,
    cross_epoch_is_stale,
)

# ---------------------------------------------------------------------------
# PerEpochFreshness
# ---------------------------------------------------------------------------


def test_per_epoch_fresh():
    fe = PerEpochFreshness("attn_freq", 10, 10, [])
    assert fe.is_fresh
    assert fe.status_label == "fresh"


def test_per_epoch_absent():
    fe = PerEpochFreshness("attn_freq", 10, 0, [1, 2, 3])
    assert not fe.is_fresh
    assert fe.status_label == "absent"


def test_per_epoch_incomplete():
    fe = PerEpochFreshness("attn_freq", 10, 7, [8, 9, 10])
    assert not fe.is_fresh
    assert "3 missing" in fe.status_label


# ---------------------------------------------------------------------------
# CrossEpochFreshness
# ---------------------------------------------------------------------------


def test_cross_epoch_fresh():
    ce = CrossEpochFreshness("neuron_dynamics", True, 10, 10)
    assert ce.is_fresh
    assert ce.status_label == "fresh"


def test_cross_epoch_absent():
    ce = CrossEpochFreshness("neuron_dynamics", False, 10, 0)
    assert not ce.is_fresh
    assert ce.status_label == "absent"


def test_cross_epoch_stale_gap():
    ce = CrossEpochFreshness("neuron_dynamics", True, 10, 7)
    assert not ce.is_fresh
    assert "3 new epoch(s)" in ce.status_label


def test_cross_epoch_stale_no_metadata():
    ce = CrossEpochFreshness("neuron_dynamics", True, 10, -1)
    assert not ce.is_fresh
    assert "no epoch metadata" in ce.status_label


def test_cross_epoch_covered_equals_available():
    ce = CrossEpochFreshness("neuron_dynamics", True, 5, 5)
    assert ce.is_fresh


def test_cross_epoch_covered_exceeds_available():
    # More epochs in artifact than available checkpoints — treat as fresh.
    ce = CrossEpochFreshness("neuron_dynamics", True, 5, 7)
    assert ce.is_fresh


# ---------------------------------------------------------------------------
# FreshnessReport
# ---------------------------------------------------------------------------


def _make_report(per_fresh=True, cross_fresh=True, summary_stale=False) -> FreshnessReport:
    pe = PerEpochFreshness("a", 5, 5 if per_fresh else 3, [] if per_fresh else [4, 5])
    ce = CrossEpochFreshness("b", True, 5, 5 if cross_fresh else 3)
    return FreshnessReport(
        variant_name="test_variant",
        checked_at="2026-01-01T00:00:00Z",
        total_checkpoints=5,
        per_epoch=[pe],
        cross_epoch=[ce],
        summary_stale=summary_stale,
    )


def test_report_all_fresh():
    report = _make_report()
    assert not report.any_stale


def test_report_stale_per_epoch():
    report = _make_report(per_fresh=False)
    assert report.any_stale


def test_report_stale_cross_epoch():
    report = _make_report(cross_fresh=False)
    assert report.any_stale


def test_report_stale_summary():
    report = _make_report(summary_stale=True)
    assert report.any_stale


def test_report_format_contains_variant_name():
    report = _make_report()
    text = report.format()
    assert "test_variant" in text


def test_report_format_fresh_message():
    report = _make_report()
    assert "All artifacts are fresh" in report.format()


def test_report_format_no_fresh_message_when_stale():
    report = _make_report(per_fresh=False)
    assert "All artifacts are fresh" not in report.format()


def test_report_format_checkmark_for_fresh():
    report = _make_report()
    assert "✓" in report.format()


def test_report_format_cross_for_stale():
    report = _make_report(per_fresh=False)
    assert "✗" in report.format()


# ---------------------------------------------------------------------------
# _scan_epoch_files
# ---------------------------------------------------------------------------


def test_scan_epoch_files(tmp_path):
    for epoch in [0, 100, 200, 500]:
        (tmp_path / f"epoch_{epoch}.npz").touch()
    # Also add a non-matching file that should be ignored.
    (tmp_path / "cross_epoch.npz").touch()
    (tmp_path / "epoch_bad.npz").touch()

    epochs = _scan_epoch_files(tmp_path)
    assert epochs == [0, 100, 200, 500]


def test_scan_epoch_files_empty(tmp_path):
    assert _scan_epoch_files(tmp_path) == []


# ---------------------------------------------------------------------------
# _read_covered_epoch_count
# ---------------------------------------------------------------------------


def test_read_covered_epoch_count(tmp_path):
    path = tmp_path / "cross_epoch.npz"
    np.savez(path, epochs=np.arange(7, dtype=np.int32), data=np.zeros(3))
    assert _read_covered_epoch_count(path) == 7


def test_read_covered_epoch_count_missing_key(tmp_path):
    path = tmp_path / "cross_epoch.npz"
    np.savez(path, data=np.zeros(3))
    assert _read_covered_epoch_count(path) == -1


def test_read_covered_epoch_count_missing_file(tmp_path):
    path = tmp_path / "nonexistent.npz"
    assert _read_covered_epoch_count(path) == -1


# ---------------------------------------------------------------------------
# check_freshness (integration on synthetic fixture)
# ---------------------------------------------------------------------------


def _make_variant(tmp_path: Path, checkpoints: list[int]) -> MagicMock:
    """Build a minimal Variant mock pointing at tmp_path."""
    variant = MagicMock()
    variant.name = "test_variant"
    variant.artifacts_dir = str(tmp_path / "artifacts")
    variant.variant_dir = tmp_path
    variant.get_available_checkpoints.return_value = checkpoints
    return variant


def _write_per_epoch(artifacts_dir: Path, name: str, epochs: list[int]) -> None:
    d = artifacts_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        (d / f"epoch_{e}.npz").touch()


def _write_cross_epoch(artifacts_dir: Path, name: str, n_epochs: int | None) -> None:
    d = artifacts_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "cross_epoch.npz"
    if n_epochs is None:
        np.savez(path, data=np.zeros(3))  # no epochs key
    else:
        np.savez(path, epochs=np.arange(n_epochs, dtype=np.int32), data=np.zeros(3))


def test_check_freshness_fully_fresh(tmp_path):
    checkpoints = [0, 100, 200]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "attn_freq", checkpoints)
    _write_cross_epoch(artifacts_dir, "neuron_dynamics", len(checkpoints))

    # summary must be newer than artifacts — write it last
    summary = tmp_path / "variant_summary.json"
    summary.write_text(json.dumps({}))

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant)

    assert report.total_checkpoints == 3
    pe = next((fe for fe in report.per_epoch if fe.analyzer_name == "attn_freq"), None)
    assert pe is not None
    assert pe.is_fresh

    ce = next((ce for ce in report.cross_epoch if ce.analyzer_name == "neuron_dynamics"), None)
    assert ce is not None
    assert ce.is_fresh

    assert not report.summary_stale
    assert not report.any_stale


def test_check_freshness_missing_per_epoch(tmp_path):
    checkpoints = [0, 100, 200, 300]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "attn_freq", [0, 100])  # missing 200, 300

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant, per_epoch_names=["attn_freq"])

    pe = report.per_epoch[0]
    assert not pe.is_fresh
    assert set(pe.missing_epochs) == {200, 300}


def test_check_freshness_stale_cross_epoch(tmp_path):
    checkpoints = [0, 100, 200, 300, 400]
    artifacts_dir = tmp_path / "artifacts"
    _write_cross_epoch(artifacts_dir, "neuron_dynamics", 3)  # only covered 3 of 5

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant, cross_epoch_names=["neuron_dynamics"])

    ce = report.cross_epoch[0]
    assert not ce.is_fresh
    assert "2 new epoch(s)" in ce.status_label


def test_check_freshness_summary_stale(tmp_path):
    import os

    checkpoints = [0, 100]
    artifacts_dir = tmp_path / "artifacts"

    d = artifacts_dir / "attn_freq"
    d.mkdir(parents=True, exist_ok=True)
    artifact = d / "epoch_0.npz"
    artifact.touch()

    # Set summary mtime to 10 seconds before artifact mtime so it's clearly older.
    summary = tmp_path / "variant_summary.json"
    summary.write_text(json.dumps({}))
    artifact_mtime = artifact.stat().st_mtime
    os.utime(summary, (artifact_mtime - 10, artifact_mtime - 10))

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant)
    assert report.summary_stale


def test_check_freshness_auto_discovery_skips_cross_epoch_only(tmp_path):
    """Auto-discovery should not list cross-epoch-only dirs as per-epoch analyzers."""
    checkpoints = [0, 100]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "attn_freq", checkpoints)
    _write_cross_epoch(artifacts_dir, "neuron_dynamics", len(checkpoints))

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant)  # no explicit names → auto-discover

    per_names = [fe.analyzer_name for fe in report.per_epoch]
    assert "attn_freq" in per_names
    assert "neuron_dynamics" not in per_names


# ---------------------------------------------------------------------------
# cross_epoch_is_stale
# ---------------------------------------------------------------------------


def test_cross_epoch_is_stale_when_more_dep_epochs(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    ce_path = artifacts_dir / "neuron_dynamics" / "cross_epoch.npz"
    ce_path.parent.mkdir(parents=True)
    np.savez(ce_path, epochs=np.arange(5, dtype=np.int32))

    # Dependency has 8 per-epoch artifacts
    dep_dir = artifacts_dir / "attn_freq"
    dep_dir.mkdir()
    for i in range(8):
        (dep_dir / f"epoch_{i * 100}.npz").touch()

    assert cross_epoch_is_stale(ce_path, ["attn_freq"], artifacts_dir, list(range(8)))


def test_cross_epoch_is_stale_false_when_up_to_date(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    ce_path = artifacts_dir / "neuron_dynamics" / "cross_epoch.npz"
    ce_path.parent.mkdir(parents=True)
    np.savez(ce_path, epochs=np.arange(5, dtype=np.int32))

    dep_dir = artifacts_dir / "attn_freq"
    dep_dir.mkdir()
    for i in range(5):
        (dep_dir / f"epoch_{i * 100}.npz").touch()

    assert not cross_epoch_is_stale(ce_path, ["attn_freq"], artifacts_dir, list(range(5)))


def test_cross_epoch_is_stale_true_when_no_epochs_key(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    ce_path = artifacts_dir / "neuron_dynamics" / "cross_epoch.npz"
    ce_path.parent.mkdir(parents=True)
    np.savez(ce_path, data=np.zeros(3))  # no epochs key

    assert cross_epoch_is_stale(ce_path, [], artifacts_dir, [])


def test_cross_epoch_is_stale_false_when_no_deps(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    ce_path = artifacts_dir / "neuron_dynamics" / "cross_epoch.npz"
    ce_path.parent.mkdir(parents=True)
    np.savez(ce_path, epochs=np.arange(5, dtype=np.int32))

    # No dependency dirs at all
    assert not cross_epoch_is_stale(ce_path, ["missing_dep"], artifacts_dir, [])
