"""Tests for REQ_119: Analysis Planner — plan/execute separation.

CoS coverage:
- Unit: PlanItem and Plan dataclasses (is_empty, format, to_dict).
- Unit: plan_analysis decision-tree scenarios (force, missing, stale, blocked).
- Parity: planner output matches the historical pipeline work-queue shape.
- Freshness: plan_analysis-driven freshness report matches pre-REQ output
  on the same fixture state (covered by test_freshness.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from miscope.analysis.planner import (
    Plan,
    PlanItem,
    plan_analysis,
    scan_epoch_files,
)

# ---------------------------------------------------------------------------
# Synthetic analyzers — minimal protocol stand-ins (no real work)
# ---------------------------------------------------------------------------


class _PrimaryStub:
    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, ctx: Any) -> dict[str, np.ndarray]:
        return {}


class _SecondaryStub:
    def __init__(self, name: str, depends_on: str) -> None:
        self.name = name
        self.depends_on = depends_on

    def analyze(self, artifact: Any, context: Any) -> dict[str, np.ndarray]:
        return {}


class _CrossEpochStub:
    def __init__(self, name: str, requires: tuple[str, ...] = ()) -> None:
        self.name = name
        self.requires = requires

    def analyze_across_epochs(
        self, artifacts_dir: str, epochs: list[int], context: Any
    ) -> dict[str, np.ndarray]:
        return {}


def _make_variant(tmp_path: Path, checkpoints: list[int], name: str = "test") -> MagicMock:
    """Return a minimal Variant-shaped mock pointing at ``tmp_path``."""
    variant = MagicMock()
    variant.name = name
    variant.artifacts_dir = str(tmp_path / "artifacts")
    variant.get_available_checkpoints.return_value = list(checkpoints)
    return variant


def _write_epochs(artifacts_dir: Path, name: str, epochs: list[int]) -> None:
    d = artifacts_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        np.savez(d / f"epoch_{e:05d}.npz", data=np.zeros(1))


def _write_cross_epoch(artifacts_dir: Path, name: str, n_epochs: int | None) -> None:
    d = artifacts_dir / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "cross_epoch.npz"
    if n_epochs is None:
        np.savez(path, data=np.zeros(3))  # no epochs key
    else:
        np.savez(path, epochs=np.arange(n_epochs, dtype=np.int32), data=np.zeros(3))


# ---------------------------------------------------------------------------
# PlanItem / Plan dataclasses
# ---------------------------------------------------------------------------


def test_plan_is_empty_true():
    plan = Plan(variant_name="v")
    assert plan.is_empty
    assert "No analysis work" in plan.format()


def test_plan_is_empty_false_with_per_epoch():
    plan = Plan(
        variant_name="v",
        per_epoch=[PlanItem(analyzer_name="a", epochs=(0, 1))],
    )
    assert not plan.is_empty


def test_plan_is_empty_false_with_cross_epoch_blocked():
    plan = Plan(
        variant_name="v",
        cross_epoch=[PlanItem(analyzer_name="ce", blocked_by=("prim",), requires=("prim",))],
    )
    assert not plan.is_empty


def test_plan_format_three_sections():
    plan = Plan(
        variant_name="v",
        available_checkpoints=(0, 1, 2),
        target_epochs=(0, 1, 2),
        per_epoch=[PlanItem(analyzer_name="prim", epochs=(2,))],
        secondary=[PlanItem(analyzer_name="sec", epochs=(2,), depends_on="prim")],
        cross_epoch=[PlanItem(analyzer_name="ce", reason="stale", requires=("prim",))],
    )
    text = plan.format()
    assert "Per-epoch analyzers" in text
    assert "Secondary analyzers" in text
    assert "Cross-epoch analyzers" in text
    assert "prim" in text
    assert "sec" in text
    assert "ce" in text


def test_plan_to_dict_serializable():
    plan = Plan(
        variant_name="v",
        available_checkpoints=(0, 1),
        target_epochs=(0, 1),
        per_epoch=[PlanItem(analyzer_name="a", epochs=(1,))],
    )
    d = plan.to_dict()
    import json

    json.dumps(d)  # raises if not serializable
    assert d["variant_name"] == "v"
    assert d["per_epoch"][0]["analyzer_name"] == "a"
    assert d["per_epoch"][0]["epochs"] == [1]


# ---------------------------------------------------------------------------
# plan_analysis: per-epoch primary analyzers
# ---------------------------------------------------------------------------


def test_plan_all_computed_empty_plan(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert plan.per_epoch == []
    assert plan.is_empty


def test_plan_some_epochs_missing(tmp_path):
    checkpoints = [0, 100, 200, 300]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", [0, 100])

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert len(plan.per_epoch) == 1
    item = plan.per_epoch[0]
    assert item.analyzer_name == "prim"
    assert list(item.epochs) == [200, 300]


def test_plan_no_artifacts_yet(tmp_path):
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert len(plan.per_epoch) == 1
    assert list(plan.per_epoch[0].epochs) == [0, 100]


def test_plan_force_includes_all(tmp_path):
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)  # all computed

    plan = plan_analysis(variant, [_PrimaryStub("prim")], force=True)
    assert len(plan.per_epoch) == 1
    assert list(plan.per_epoch[0].epochs) == [0, 100]


def test_plan_checkpoints_filter(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)

    plan = plan_analysis(variant, [_PrimaryStub("prim")], checkpoints=[0, 200, 999])
    # 999 not in available — filtered out
    assert list(plan.per_epoch[0].epochs) == [0, 200]
    assert plan.target_epochs == (0, 200)


# ---------------------------------------------------------------------------
# plan_analysis: secondary analyzers
# ---------------------------------------------------------------------------


def test_plan_secondary_targets_dependency_epochs(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    # Dependency has only 2 epochs computed
    _write_epochs(Path(variant.artifacts_dir), "prim", [0, 100])

    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert len(plan.secondary) == 1
    item = plan.secondary[0]
    assert item.depends_on == "prim"
    # Secondary follows dependency, not checkpoints
    assert list(item.epochs) == [0, 100]


def test_plan_secondary_blocked_when_dep_empty(tmp_path):
    variant = _make_variant(tmp_path, [0, 100])

    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert len(plan.secondary) == 1
    item = plan.secondary[0]
    assert item.blocked_by == ("prim",)
    assert item.epochs == ()


def test_plan_secondary_resumes_only_missing(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "sec", [0])  # secondary done for epoch 0

    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert list(plan.secondary[0].epochs) == [100, 200]


# ---------------------------------------------------------------------------
# plan_analysis: cross-epoch analyzers
# ---------------------------------------------------------------------------


def test_plan_cross_epoch_missing_artifact(tmp_path):
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert len(plan.cross_epoch) == 1
    item = plan.cross_epoch[0]
    assert item.reason == "missing"
    assert item.blocked_by == ()


def test_plan_cross_epoch_stale_when_dep_grows(tmp_path):
    """Cross-epoch artifact built on fewer epochs than dependency now has."""
    checkpoints = [0, 100, 200, 300]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    _write_cross_epoch(Path(variant.artifacts_dir), "ce", n_epochs=2)

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].reason == "stale"


def test_plan_cross_epoch_fresh_when_caught_up(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    _write_cross_epoch(Path(variant.artifacts_dir), "ce", n_epochs=3)

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert plan.cross_epoch == []


def test_plan_cross_epoch_blocked_when_dep_empty(tmp_path):
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    # 'prim' has no artifacts

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert len(plan.cross_epoch) == 1
    item = plan.cross_epoch[0]
    assert item.blocked_by == ("prim",)
    assert item.epochs == ()


def test_plan_cross_epoch_force_rebuilds(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    _write_cross_epoch(Path(variant.artifacts_dir), "ce", n_epochs=3)  # fresh

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))], force=True)
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].reason == "forced"


def test_plan_cross_epoch_stale_no_metadata(tmp_path):
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    _write_cross_epoch(Path(variant.artifacts_dir), "ce", n_epochs=None)  # no 'epochs' key

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].reason == "stale"


def test_plan_cross_epoch_dep_satisfied_by_cross_artifact(tmp_path):
    """Cross-to-cross: a required cross-epoch analyzer is satisfied by its own
    cross_epoch.npz even without per-epoch files."""
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    _write_cross_epoch(Path(variant.artifacts_dir), "prim_cross", n_epochs=2)

    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim_cross",))])
    # 'ce' artifact missing → in plan; dep is satisfied (not blocked)
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].blocked_by == ()
    assert plan.cross_epoch[0].reason == "missing"


# ---------------------------------------------------------------------------
# Mixed-phase planning
# ---------------------------------------------------------------------------


def test_plan_classifies_mixed_analyzers_with_forward_projection(tmp_path):
    """When primary is in the same plan as a dependent, the Plan reflects
    post-execution state: secondary and cross-epoch are NOT blocked."""
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)

    plan = plan_analysis(
        variant,
        [
            _PrimaryStub("prim"),
            _SecondaryStub("sec", depends_on="prim"),
            _CrossEpochStub("ce", requires=("prim",)),
        ],
    )
    # prim missing all
    assert len(plan.per_epoch) == 1
    # sec sees projected post-primary state — planned, not blocked
    assert len(plan.secondary) == 1
    assert plan.secondary[0].blocked_by == ()
    assert list(plan.secondary[0].epochs) == checkpoints
    # ce sees projected post-primary state — planned, not blocked
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].blocked_by == ()
    assert plan.cross_epoch[0].reason == "missing"


def test_plan_secondary_blocked_when_primary_absent_from_plan(tmp_path):
    """Secondary is blocked when its dep is neither on disk nor in the plan."""
    variant = _make_variant(tmp_path, [0, 100])
    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert plan.secondary[0].blocked_by == ("prim",)


def test_plan_cross_epoch_blocked_when_required_absent_from_plan(tmp_path):
    """Cross-epoch is blocked when its required dep is neither on disk nor in the plan."""
    variant = _make_variant(tmp_path, [0, 100])
    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert plan.cross_epoch[0].blocked_by == ("prim",)


# ---------------------------------------------------------------------------
# Parity: planner output mirrors the historical pipeline work-queue shape.
#
# Snapshot recorded from the pre-REQ behavior; planner must produce the
# same per-analyzer missing-epoch lists for the same fixture state.
# ---------------------------------------------------------------------------


def test_parity_with_pre_req_work_queue(tmp_path):
    """Snapshot: per_epoch[i].epochs equals the pre-REQ work_queue's
    missing-epoch lists for the same disk state."""
    checkpoints = [0, 100, 200, 300, 400]
    variant = _make_variant(tmp_path, checkpoints)
    artifacts_dir = Path(variant.artifacts_dir)
    _write_epochs(artifacts_dir, "prim_a", [0, 100, 200])  # missing 300, 400
    _write_epochs(artifacts_dir, "prim_b", checkpoints)  # all done

    plan = plan_analysis(
        variant,
        [_PrimaryStub("prim_a"), _PrimaryStub("prim_b")],
    )
    # Pre-REQ work_queue: [(prim_a, [300, 400])] (prim_b absent because nothing missing)
    by_name = {item.analyzer_name: list(item.epochs) for item in plan.per_epoch}
    assert by_name == {"prim_a": [300, 400]}


@pytest.mark.parametrize("force", [True, False])
def test_parity_force_flag(tmp_path, force):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)

    plan = plan_analysis(variant, [_PrimaryStub("prim")], force=force)
    if force:
        assert list(plan.per_epoch[0].epochs) == checkpoints
    else:
        assert plan.per_epoch == []


# ---------------------------------------------------------------------------
# Disk-state primitive: scan_epoch_files (used by both planner and freshness)
# ---------------------------------------------------------------------------


def test_scan_epoch_files_round_trip(tmp_path):
    for e in [0, 100, 200]:
        np.savez(tmp_path / f"epoch_{e:05d}.npz", data=np.zeros(1))
    assert scan_epoch_files(tmp_path) == [0, 100, 200]


def test_scan_epoch_files_missing_dir(tmp_path):
    assert scan_epoch_files(tmp_path / "nope") == []
