"""Tests for REQ_080: Artifact freshness reporting.

CoS coverage:
- Unit: PerEpochFreshness / CrossEpochFreshness status labels and is_fresh logic.
- Unit: FreshnessReport.any_stale and format() output.
- Unit: _scan_epoch_files correctly parses epoch_*.npz filenames.
- Unit: _read_covered_epoch_count returns correct count or -1 on missing key.
- Integration: check_freshness on a synthetic fixture directory produces correct
  PerEpochFreshness and CrossEpochFreshness entries.

After REQ_119, ``check_freshness`` is a wrapper over ``plan_analysis``;
the per-pipeline staleness helper ``cross_epoch_is_stale`` was removed
and its decision logic now lives in ``miscope.analysis.planner``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from miscope.analysis.artifact_loader import read_signature_manifest, write_signature_manifest
from miscope.analysis.freshness import (
    CrossEpochFreshness,
    FreshnessReport,
    PerEpochFreshness,
    _read_covered_epoch_count,
    _scan_epoch_files,
    check_freshness,
)
from miscope.analysis.planner import plan_analysis


class _PrimaryStub:
    """Per-epoch model-driven analyzer stub (no registered Spec → version 1)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, inputs, context):  # pragma: no cover - never executed
        return {}


class _CrossEpochStub:
    def __init__(self, name: str, requires: tuple[str, ...] = ()) -> None:
        self.name = name
        self.requires = requires

    def analyze_across_epochs(self, *a, **k):  # pragma: no cover - never executed
        return {}


def _stamp_fresh(variant, analyzers, checkpoints=None) -> None:
    """Stamp signature manifests for ``analyzers`` so their artifacts read fresh.

    Mirrors the pipeline: build a plan, then write each planned node's post-run
    signatures to its manifest. After this, a re-plan over the same disk state is a
    no-op — the REQ_145 freshness contract.
    """
    plan = plan_analysis(variant, analyzers, checkpoints=checkpoints)
    for name, sigs in plan.signatures.items():
        merged = read_signature_manifest(variant.artifacts_dir, name, "")
        merged.update(sigs)
        write_signature_manifest(variant.artifacts_dir, name, "", merged)


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


def test_cross_epoch_stale_with_reason():
    """REQ_145: staleness is the planner's reason, not an epoch-count gap."""
    ce = CrossEpochFreshness(
        "neuron_dynamics", True, 10, 7, plan_reason="stale: upstream x changed"
    )
    assert not ce.is_fresh
    assert "upstream x changed" in ce.status_label


def test_cross_epoch_fresh_ignores_count():
    """A present artifact with no planner reason is fresh regardless of covered count."""
    ce = CrossEpochFreshness("neuron_dynamics", True, 10, 7)
    assert ce.is_fresh
    assert ce.status_label == "fresh"


# ---------------------------------------------------------------------------
# FreshnessReport
# ---------------------------------------------------------------------------


def _make_report(per_fresh=True, cross_fresh=True, summary_stale=False) -> FreshnessReport:
    pe = PerEpochFreshness("a", 5, 5 if per_fresh else 3, [] if per_fresh else [4, 5])
    ce = CrossEpochFreshness(
        "b", True, 5, 5 if cross_fresh else 3, plan_reason=None if cross_fresh else "stale"
    )
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
    variant.summary_path = tmp_path / "variant_summary.json"
    variant.get_available_checkpoints.return_value = checkpoints
    variant.checkpoint_fingerprint.side_effect = lambda e: f"ckpt-{e}"
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
    _write_cross_epoch(artifacts_dir, "cross_demo", len(checkpoints))

    variant = _make_variant(tmp_path, checkpoints)
    # Stamp signatures so the artifacts read fresh under the REQ_145 predicate.
    _stamp_fresh(variant, [_PrimaryStub("attn_freq"), _CrossEpochStub("cross_demo")])

    # summary must be newer than artifacts — write it last
    summary = tmp_path / "variant_summary.json"
    summary.write_text(json.dumps({}))

    report = check_freshness(variant)

    assert report.total_checkpoints == 3
    pe = next((fe for fe in report.per_epoch if fe.analyzer_name == "attn_freq"), None)
    assert pe is not None
    assert pe.is_fresh

    ce = next((ce for ce in report.cross_epoch if ce.analyzer_name == "cross_demo"), None)
    assert ce is not None
    assert ce.is_fresh

    assert not report.summary_stale
    assert not report.any_stale


def test_check_freshness_missing_per_epoch(tmp_path):
    checkpoints = [0, 100, 200, 300]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "attn_freq", [0, 100])  # done 0, 100

    variant = _make_variant(tmp_path, checkpoints)
    # 0, 100 stamped fresh; 200, 300 are genuinely new checkpoints.
    _stamp_fresh(variant, [_PrimaryStub("attn_freq")], checkpoints=[0, 100])

    report = check_freshness(variant, per_epoch_names=["attn_freq"])

    pe = report.per_epoch[0]
    assert not pe.is_fresh
    assert set(pe.missing_epochs) == {200, 300}


def test_intragroup_manifold_present_but_unstamped_is_stale(tmp_path):
    """REQ_145 "free" fixture: the real ``intragroup_manifold`` analyzer.

    Its stored cross-epoch artifacts on non-pinned variants are stale/wrong-shape yet
    coverage-complete, so the old count-based predicate reported them fresh and they
    errored on read. They predate signatures (no manifest), so the new predicate reads
    them stale -> recompute — the turn-1 regression on a real, in-tree analyzer, no
    artificial version bump. The upstream ``neuron_group_pca`` is present so the item
    is stale (not blocked)."""
    checkpoints = [0, 100, 200]
    artifacts_dir = tmp_path / "artifacts"
    _write_cross_epoch(artifacts_dir, "neuron_group_pca", 3)  # upstream satisfied
    _write_cross_epoch(artifacts_dir, "intragroup_manifold", 3)  # present, full, unstamped

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant, cross_epoch_names=["intragroup_manifold"])

    ce = report.cross_epoch[0]
    assert ce.analyzer_name == "intragroup_manifold"
    assert not ce.is_fresh
    assert "stale" in ce.status_label


def test_check_freshness_present_but_unstamped_is_stale(tmp_path):
    """REQ_145 turn-1 regression: a present, coverage-complete cross-epoch artifact
    with no signature manifest (a legacy artifact predating REQ_145) reads stale —
    the case the old count-based predicate reported as fresh."""
    checkpoints = [0, 100, 200, 300, 400]
    artifacts_dir = tmp_path / "artifacts"
    _write_cross_epoch(artifacts_dir, "cross_demo", 5)  # full coverage, but no manifest

    variant = _make_variant(tmp_path, checkpoints)
    report = check_freshness(variant, cross_epoch_names=["cross_demo"])

    ce = report.cross_epoch[0]
    assert not ce.is_fresh
    assert "stale" in ce.status_label


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


# cross_epoch_is_stale removed by REQ_119; its decision logic now lives in
# miscope.analysis.planner._plan_cross_epoch_item and is exercised through
# tests/test_planner.py.


# ---------------------------------------------------------------------------
# check_freshness with analyzers= (registered-but-never-run surfaces as absent)
# ---------------------------------------------------------------------------


class _PrimarySpec:
    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, inputs, context):  # protocol stub
        pass


class _SecondarySpec:
    def __init__(self, name: str, depends_on: str) -> None:
        self.name = name
        self.depends_on = depends_on

    def analyze(self, artifact, context):  # protocol stub
        pass


class _CrossEpochSpec:
    def __init__(self, name: str, requires: tuple[str, ...] = ()) -> None:
        self.name = name
        self.requires = requires

    def analyze_across_epochs(self, *args, **kwargs):  # protocol stub
        pass


def test_check_freshness_with_analyzers_surfaces_unrun_secondary(tmp_path):
    """A registered secondary analyzer that has never run must appear in the
    report as 'absent' — the bug REQ_119 was meant to expose.

    Disk-only auto-discovery silently omits never-run analyzers because their
    directory doesn't exist. Passing ``analyzers=`` to check_freshness makes
    the registered set the source of truth alongside disk discovery.
    """
    checkpoints = [0, 100, 200]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "parameter_snapshot", checkpoints)

    variant = _make_variant(tmp_path, checkpoints)
    analyzers = [
        _PrimarySpec("parameter_snapshot"),
        _SecondarySpec("neuron_grouping", depends_on="parameter_snapshot"),
    ]
    report = check_freshness(variant, analyzers=analyzers)

    per_epoch_names = {fe.analyzer_name for fe in report.per_epoch}
    assert "neuron_grouping" in per_epoch_names

    ng = next(fe for fe in report.per_epoch if fe.analyzer_name == "neuron_grouping")
    assert ng.artifact_epoch_count == 0
    assert ng.status_label == "absent"


def test_check_freshness_with_analyzers_keeps_disk_leftovers(tmp_path):
    """Disk artifacts from removed/unregistered analyzers stay visible —
    the analyzers= mode unions registered + on-disk names."""
    checkpoints = [0, 100]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "coarseness", [0])  # leftover, not in registry

    variant = _make_variant(tmp_path, checkpoints)
    analyzers = [_PrimarySpec("attention_freq")]  # registry doesn't include coarseness
    report = check_freshness(variant, analyzers=analyzers)

    per_epoch_names = {fe.analyzer_name for fe in report.per_epoch}
    assert "attention_freq" in per_epoch_names  # registered, missing
    assert "coarseness" in per_epoch_names  # leftover, on disk


def test_check_freshness_with_analyzers_classifies_cross_epoch(tmp_path):
    """A registered cross-epoch analyzer that has never run must appear in
    the cross-epoch list as 'absent', not silently dropped."""
    checkpoints = [0, 100]
    artifacts_dir = tmp_path / "artifacts"
    _write_per_epoch(artifacts_dir, "parameter_snapshot", checkpoints)

    variant = _make_variant(tmp_path, checkpoints)
    analyzers = [
        _PrimarySpec("parameter_snapshot"),
        _CrossEpochSpec("activation_dmd", requires=("parameter_snapshot",)),
    ]
    report = check_freshness(variant, analyzers=analyzers)

    cross_names = {ce.analyzer_name for ce in report.cross_epoch}
    assert "activation_dmd" in cross_names
    ce = next(ce for ce in report.cross_epoch if ce.analyzer_name == "activation_dmd")
    assert ce.status_label == "absent"


def test_freshness_does_not_double_list_cross_epoch_analyzers(tmp_path):
    """Regression: post-REQ_121, every analyzer satisfies the same protocol,
    so the old ``hasattr(.., 'analyze_across_epochs')`` discriminator put
    every registered analyzer into the per-epoch bucket. Disk discovery
    then added cross-epoch analyzers to the cross-epoch bucket too, so
    the report listed each cross-epoch analyzer twice (fresh in
    per-epoch, stale in cross-epoch). Classify by the Spec's effective
    category instead.

    A registered cross-epoch analyzer with a real cross_epoch.npz on
    disk must appear only in the cross-epoch section.
    """
    from miscope.analysis.registry import AnalyzerRegistry, register_analyzer
    from miscope.analysis.spec import AnalyzerSpec

    saved_specs = dict(_specs_snapshot())
    saved_factories = dict(_factories_snapshot())
    try:
        AnalyzerRegistry.clear()

        # Register a real unified cross-epoch Spec (output_scope="cross_epoch").
        ce_spec = AnalyzerSpec(name="ce_test", output_scope="cross_epoch", inputs=())

        @register_analyzer(ce_spec)
        class _CETest:
            name = "ce_test"

            def analyze(self, inputs, context):  # noqa: ARG002
                return {}

        checkpoints = [0, 100]
        artifacts_dir = tmp_path / "artifacts"
        _write_cross_epoch(artifacts_dir, "ce_test", n_epochs=1)  # stale (1 < 2)

        variant = _make_variant(tmp_path, checkpoints)
        report = check_freshness(variant, analyzers=[_CETest()])

        per_epoch_names = {fe.analyzer_name for fe in report.per_epoch}
        cross_names = {ce.analyzer_name for ce in report.cross_epoch}
        assert "ce_test" in cross_names
        assert "ce_test" not in per_epoch_names, (
            "cross-epoch analyzer leaked into per-epoch freshness list"
        )
    finally:
        AnalyzerRegistry.clear()
        from miscope.analysis import registry as reg_mod

        reg_mod._specs.update(saved_specs)
        reg_mod._factories.update(saved_factories)


def _specs_snapshot():
    from miscope.analysis import registry as reg_mod

    return dict(reg_mod._specs)


def _factories_snapshot():
    from miscope.analysis import registry as reg_mod

    return dict(reg_mod._factories)
