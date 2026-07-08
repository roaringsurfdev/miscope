"""Recipe liveness/GC + inventory re-resolution freshness (REQ_138, Phase 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from miscope.analysis.artifact_loader import analyzer_dir, iter_recipe_dirs
from miscope.analysis.freshness import check_reference_freshness
from miscope.analysis.parameters import (
    LiteralBinding,
    Parameterization,
    ParameterSpec,
    Reducer,
    ReferenceBinding,
)
from miscope.analysis.spec import AnalyzerSpec
from miscope.warehouse import live_recipe_signatures, orphaned_recipe_dirs, record_run_set


class _FakeVariant:
    def __init__(self, root: Path):
        self.name = "p5_seed1_dseed2"
        self.variant_dir = root / "variants" / self.name
        self.artifacts_dir = self.variant_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _write_recipe_artifact(variant, analyzer, sig):
    d = Path(analyzer_dir(str(variant.artifacts_dir), analyzer, sig))
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "cross_epoch", epochs=np.array([0, 100]))


def _specs():
    from miscope.analysis.inputs import ArtifactInput

    dmd = AnalyzerSpec(
        name="dmd",
        output_scope="cross_epoch",
        inputs=(ArtifactInput("grouping"),),
        parameters=(
            ParameterSpec(
                "reference_epoch",
                "int64",
                "analyzer",
                ReferenceBinding("reference_epoch", "grouping", Reducer("max_epoch")),
            ),
        ),
    )
    return {"dmd": dmd}


# ---------------------------------------------------------------------------
# Liveness + orphan detection
# ---------------------------------------------------------------------------


def test_iter_recipe_dirs_finds_only_recipe_segments(tmp_path):
    variant = _FakeVariant(tmp_path)
    _write_recipe_artifact(variant, "dmd", "sigA")
    # A plain default artifact (no recipe segment) must not be reported.
    (variant.artifacts_dir / "dmd").mkdir(exist_ok=True)
    np.savez_compressed(variant.artifacts_dir / "dmd" / "cross_epoch", epochs=np.array([0]))

    found = list(iter_recipe_dirs(str(variant.artifacts_dir)))
    assert [(a, s) for a, s, _ in found] == [("dmd", "sigA")]


def test_orphan_when_no_run_set_references_signature(tmp_path):
    variant = _FakeVariant(tmp_path)
    _write_recipe_artifact(variant, "dmd", "live_sig")
    _write_recipe_artifact(variant, "dmd", "orphan_sig")

    pz = Parameterization(bindings=(LiteralBinding("reference_epoch", 20000, analyzer="dmd"),))
    record_run_set(variant, pz, {"dmd": "live_sig"}, {"dmd": {"reference_epoch": 20000}}, _specs())

    assert live_recipe_signatures(variant) == {"live_sig"}
    orphans = orphaned_recipe_dirs(variant)
    assert [(o.analyzer, o.recipe_signature) for o in orphans] == [("dmd", "orphan_sig")]


def test_no_orphans_when_all_recipes_live(tmp_path):
    variant = _FakeVariant(tmp_path)
    _write_recipe_artifact(variant, "dmd", "s1")
    pz = Parameterization(bindings=(LiteralBinding("reference_epoch", 1, analyzer="dmd"),))
    record_run_set(variant, pz, {"dmd": "s1"}, {"dmd": {"reference_epoch": 1}}, _specs())
    assert orphaned_recipe_dirs(variant) == []


# ---------------------------------------------------------------------------
# Inventory re-resolution staleness
# ---------------------------------------------------------------------------


def test_reference_freshness_stale_when_inventory_moves(tmp_path):
    variant = _FakeVariant(tmp_path)
    # grouping inventory now extends to 50000 (training extended)...
    g = variant.artifacts_dir / "grouping"
    g.mkdir(parents=True, exist_ok=True)
    for e in (0, 100, 50000):
        np.savez_compressed(g / f"epoch_{e:05d}", x=np.array([0]))
    # ...but the default-plane dmd artifact recorded reference_epoch=100 (the old max).
    d = variant.artifacts_dir / "dmd"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "cross_epoch", reference_epoch=np.array(100, dtype=np.int64))

    (res,) = check_reference_freshness(variant, specs=list(_specs().values()))
    assert res.analyzer_name == "dmd"
    assert res.stored_value == 100
    assert res.resolved_value == 50000
    assert res.is_stale


def test_reference_freshness_fresh_when_inventory_unchanged(tmp_path):
    variant = _FakeVariant(tmp_path)
    g = variant.artifacts_dir / "grouping"
    g.mkdir(parents=True, exist_ok=True)
    for e in (0, 100):
        np.savez_compressed(g / f"epoch_{e:05d}", x=np.array([0]))
    d = variant.artifacts_dir / "dmd"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "cross_epoch", reference_epoch=np.array(100, dtype=np.int64))

    (res,) = check_reference_freshness(variant, specs=list(_specs().values()))
    assert not res.is_stale
