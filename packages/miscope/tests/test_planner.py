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

from miscope.analysis.artifact_loader import read_signature_manifest, write_signature_manifest
from miscope.analysis.planner import (
    Plan,
    PlanItem,
    plan_analysis,
    scan_epoch_files,
)
from miscope.analysis.signature import CROSS_EPOCH_KEY

# ---------------------------------------------------------------------------
# Synthetic analyzers — minimal protocol stand-ins (no real work)
# ---------------------------------------------------------------------------


class _PrimaryStub:
    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, inputs, context) -> dict[str, np.ndarray]:
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
    variant.checkpoint_fingerprint.side_effect = lambda e: f"ckpt-{e}"
    return variant


def _write_epochs(artifacts_dir: Path, name: str, epochs: list[int]) -> None:
    d = artifacts_dir / name
    d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        np.savez(d / f"epoch_{e:05d}.npz", data=np.zeros(1))


def _apply_plan(variant: MagicMock, analyzers: list[Any], checkpoints: list[int] | None = None):
    """Simulate a pipeline run: plan, write the planned artifacts, stamp their
    signatures (REQ_145). After this, a re-plan over the same disk state is a no-op.

    Returns the plan that was applied. Mirrors ``AnalysisPipeline``: only planned
    (non-blocked) nodes are written + stamped.
    """
    artifacts_dir = Path(variant.artifacts_dir)
    plan = plan_analysis(variant, analyzers, checkpoints=checkpoints)
    for item in plan.per_epoch:
        if item.blocked_by or not item.epochs:
            continue
        _write_epochs(artifacts_dir, item.analyzer_name, list(item.epochs))
        _stamp(variant, plan, item.analyzer_name, [str(e) for e in item.epochs])
    for item in plan.cross_epoch:
        if item.blocked_by:
            continue
        _write_cross_epoch(artifacts_dir, item.analyzer_name, len(item.epochs))
        _stamp(variant, plan, item.analyzer_name, [CROSS_EPOCH_KEY])
    return plan


def _stamp(variant: MagicMock, plan: Plan, name: str, keys: list[str]) -> None:
    """Write the planner's post-run signatures for ``name`` to its manifest."""
    sigs = plan.signatures.get(name, {})
    merged = read_signature_manifest(variant.artifacts_dir, name, "")
    for key in keys:
        if key in sigs:
            merged[key] = sigs[key]
    write_signature_manifest(variant.artifacts_dir, name, "", merged)


def _per_epoch(plan: Plan, name: str) -> PlanItem:
    """Return the per-epoch PlanItem for ``name`` (REQ_133: one merged list)."""
    return next(it for it in plan.per_epoch if it.analyzer_name == name)


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


def test_plan_format_two_scopes():
    """REQ_133: a per-epoch artifact-derived analyzer (former secondary) shows
    up in the single Per-epoch section; there is no Secondary section."""
    plan = Plan(
        variant_name="v",
        available_checkpoints=(0, 1, 2),
        target_epochs=(0, 1, 2),
        per_epoch=[
            PlanItem(analyzer_name="prim", epochs=(2,)),
            PlanItem(analyzer_name="sec", epochs=(2,), depends_on="prim", requires=("prim",)),
        ],
        cross_epoch=[PlanItem(analyzer_name="ce", reason="stale", requires=("prim",))],
    )
    text = plan.format()
    assert "Per-epoch analyzers" in text
    assert "Secondary analyzers" not in text
    assert "Cross-epoch analyzers" in text
    assert "prim" in text
    assert "sec" in text
    assert "depends_on=prim" in text
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
    """REQ_145: after a run stamps signatures, a re-plan over the same state is a no-op."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _apply_plan(variant, [_PrimaryStub("prim")])

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert plan.per_epoch == []
    assert plan.is_empty


def test_plan_some_epochs_missing(tmp_path):
    """REQ_145: per-(analyzer,epoch) granularity — only the new checkpoints replan."""
    checkpoints = [0, 100, 200, 300]
    variant = _make_variant(tmp_path, checkpoints)
    # 0, 100 analyzed + stamped; 200, 300 are new checkpoints.
    _apply_plan(variant, [_PrimaryStub("prim")], checkpoints=[0, 100])

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert len(plan.per_epoch) == 1
    item = plan.per_epoch[0]
    assert item.analyzer_name == "prim"
    assert list(item.epochs) == [200, 300]
    assert item.reason == "stale: new epoch"


def test_plan_present_but_unstamped_recomputes(tmp_path):
    """REQ_145 turn-1 regression: artifacts present on disk but never stamped
    (legacy, predating signatures) are stale and fully recomputed."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)  # no manifest

    plan = plan_analysis(variant, [_PrimaryStub("prim")])
    assert list(plan.per_epoch[0].epochs) == checkpoints
    assert plan.per_epoch[0].reason == "missing"


def test_plan_code_version_bump_recomputes(tmp_path):
    """REQ_145 invalidation: a producer version bump restages every covered epoch."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)

    class _V1(_PrimaryStub):
        version = 1

    class _V2(_PrimaryStub):
        version = 2

    _apply_plan(variant, [_V1("prim")])
    assert plan_analysis(variant, [_V1("prim")]).is_empty  # stable at v1
    plan = plan_analysis(variant, [_V2("prim")])
    assert list(plan.per_epoch[0].epochs) == checkpoints
    assert plan.per_epoch[0].reason == "stale: code v1->v2"


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
    assert len(plan.per_epoch) == 1
    item = _per_epoch(plan, "sec")
    assert item.depends_on == "prim"
    # Artifact-derived per-epoch follows its upstream's epochs, not checkpoints
    assert list(item.epochs) == [0, 100]


def test_plan_secondary_blocked_when_dep_empty(tmp_path):
    variant = _make_variant(tmp_path, [0, 100])

    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert len(plan.per_epoch) == 1
    item = _per_epoch(plan, "sec")
    assert item.blocked_by == ("prim",)
    assert item.epochs == ()


def test_plan_secondary_resumes_only_missing(tmp_path):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _apply_plan(variant, [_PrimaryStub("prim")])  # prim all done + stamped
    # secondary done + stamped for epoch 0 only (an artifact-derived analyzer
    # follows its upstream's epochs, so restrict by stamping one key).
    sec_plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    _write_epochs(Path(variant.artifacts_dir), "sec", [0])
    _stamp(variant, sec_plan, "sec", ["0"])

    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert list(_per_epoch(plan, "sec").epochs) == [100, 200]


def test_plan_per_epoch_chain_ordered_and_unblocked(tmp_path):
    """REQ_130/REQ_133: an artifact-derived per-epoch analyzer depending on
    another is topologically ordered after it within the single per-epoch list
    (not left blocked by input order). fourier_frequency_quality →
    neuron_grouping is the first such per-epoch chain."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    # Pass the dependent BEFORE its dependency to exercise the ordering.
    analyzers = [
        _SecondaryStub("sec_b", depends_on="sec_a"),
        _SecondaryStub("sec_a", depends_on="prim"),
        _PrimaryStub("prim"),
    ]
    plan = plan_analysis(variant, analyzers)

    order = [it.analyzer_name for it in plan.per_epoch]
    assert order.index("sec_a") < order.index("sec_b")
    sec_b = _per_epoch(plan, "sec_b")
    assert sec_b.blocked_by == ()  # not blocked: sec_a is projected complete first
    assert list(sec_b.epochs) == checkpoints


def test_plan_cyclic_dependency_raises(tmp_path):
    """A dependency cycle is a configuration error, surfaced clearly."""
    variant = _make_variant(tmp_path, [0])
    analyzers = [
        _SecondaryStub("sec_a", depends_on="sec_b"),
        _SecondaryStub("sec_b", depends_on="sec_a"),
    ]
    with pytest.raises(ValueError, match="cyclic analyzer dependency"):
        plan_analysis(variant, analyzers)


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
    analyzers = [_PrimaryStub("prim"), _CrossEpochStub("ce", requires=("prim",))]
    _apply_plan(variant, analyzers)  # prim + ce computed and stamped

    plan = plan_analysis(variant, analyzers)
    assert plan.cross_epoch == []
    assert plan.per_epoch == []


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
    # prim + sec both per-epoch now (REQ_133); both missing all
    assert len(plan.per_epoch) == 2
    # sec sees projected post-primary state — planned, not blocked
    sec = _per_epoch(plan, "sec")
    assert sec.blocked_by == ()
    assert list(sec.epochs) == checkpoints
    # prim is ordered before its dependent sec
    order = [it.analyzer_name for it in plan.per_epoch]
    assert order.index("prim") < order.index("sec")
    # ce sees projected post-primary state — planned, not blocked
    assert len(plan.cross_epoch) == 1
    assert plan.cross_epoch[0].blocked_by == ()
    assert plan.cross_epoch[0].reason == "missing"


def test_plan_secondary_blocked_when_primary_absent_from_plan(tmp_path):
    """Secondary is blocked when its dep is neither on disk nor in the plan."""
    variant = _make_variant(tmp_path, [0, 100])
    plan = plan_analysis(variant, [_SecondaryStub("sec", depends_on="prim")])
    assert _per_epoch(plan, "sec").blocked_by == ("prim",)


def test_plan_cross_epoch_blocked_when_required_absent_from_plan(tmp_path):
    """Cross-epoch is blocked when its required dep is neither on disk nor in the plan."""
    variant = _make_variant(tmp_path, [0, 100])
    plan = plan_analysis(variant, [_CrossEpochStub("ce", requires=("prim",))])
    assert plan.cross_epoch[0].blocked_by == ("prim",)


# ---------------------------------------------------------------------------
# REQ_133: cross-epoch → cross-epoch DAG ordering and transitive staleness
# ---------------------------------------------------------------------------


def test_cross_to_cross_satisfiable_in_one_pass(tmp_path):
    """CoS 2/3: on a clean variant (no cross_epoch.npz yet), a cross-epoch
    analyzer depending on another cross-epoch analyzer is planned — not
    blocked — because the upstream's projected coverage is seeded after it is
    planned in the same pass. Mirrors neuron_group_pca → intragroup_manifold."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _write_epochs(Path(variant.artifacts_dir), "prim", checkpoints)
    # Pass the dependent first to prove ordering is by DAG, not input order.
    analyzers = [
        _CrossEpochStub("ce_leaf", requires=("ce_root",)),
        _CrossEpochStub("ce_root", requires=("prim",)),
        _PrimaryStub("prim"),
    ]
    plan = plan_analysis(variant, analyzers)

    names = {it.analyzer_name for it in plan.cross_epoch}
    assert names == {"ce_root", "ce_leaf"}
    for item in plan.cross_epoch:
        assert item.blocked_by == (), f"{item.analyzer_name} should not be blocked"
        assert item.reason == "missing"
    order = [it.analyzer_name for it in plan.cross_epoch]
    assert order.index("ce_root") < order.index("ce_leaf")


def test_cross_to_cross_blocked_propagates_when_root_absent(tmp_path):
    """A genuinely unsatisfiable cross→cross chain stays blocked at every hop:
    seeding only treats a *planned* (not blocked) upstream as future-complete."""
    checkpoints = [0, 100]
    variant = _make_variant(tmp_path, checkpoints)
    # 'prim' has no artifacts and is not in the plan → ce_root blocked → ce_leaf blocked.
    analyzers = [
        _CrossEpochStub("ce_root", requires=("prim",)),
        _CrossEpochStub("ce_leaf", requires=("ce_root",)),
    ]
    plan = plan_analysis(variant, analyzers)

    by_name = {it.analyzer_name: it for it in plan.cross_epoch}
    assert by_name["ce_root"].blocked_by == ("prim",)
    assert by_name["ce_leaf"].blocked_by == ("ce_root",)


def test_transitive_staleness_two_hop_per_epoch(tmp_path):
    """CoS 4: a 2-hop per-epoch chain (prim → mid → leaf). When the root has a
    newer epoch than the mid, both the mid and the leaf are replanned for it —
    the leaf only because the mid's projected coverage (post-regeneration)
    transitively includes the new epoch."""
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    analyzers = [
        _PrimaryStub("prim"),
        _SecondaryStub("mid", depends_on="prim"),
        _SecondaryStub("leaf", depends_on="mid"),
    ]
    # Whole chain computed + stamped for [0, 100]; 200 is a new checkpoint.
    _apply_plan(variant, analyzers, checkpoints=[0, 100])

    plan = plan_analysis(variant, analyzers)
    assert list(_per_epoch(plan, "mid").epochs) == [200]
    assert list(_per_epoch(plan, "leaf").epochs) == [200]


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
    # prim_a stamped for [0,100,200] (missing 300,400); prim_b stamped for all.
    _apply_plan(variant, [_PrimaryStub("prim_a")], checkpoints=[0, 100, 200])
    _apply_plan(variant, [_PrimaryStub("prim_b")])

    plan = plan_analysis(
        variant,
        [_PrimaryStub("prim_a"), _PrimaryStub("prim_b")],
    )
    # Only prim_a's two new checkpoints replan; prim_b is signature-fresh.
    by_name = {item.analyzer_name: list(item.epochs) for item in plan.per_epoch}
    assert by_name == {"prim_a": [300, 400]}


@pytest.mark.parametrize("force", [True, False])
def test_parity_force_flag(tmp_path, force):
    checkpoints = [0, 100, 200]
    variant = _make_variant(tmp_path, checkpoints)
    _apply_plan(variant, [_PrimaryStub("prim")])  # computed + stamped fresh

    plan = plan_analysis(variant, [_PrimaryStub("prim")], force=force)
    if force:
        assert list(plan.per_epoch[0].epochs) == checkpoints
        assert plan.per_epoch[0].reason == "forced"
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
