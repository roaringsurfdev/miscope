"""Consumer API: variant.parameterize() recipe-scoped handle (REQ_138, Phase 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from miscope.analysis.artifact_loader import analyzer_dir
from miscope.analysis.parameters import LiteralBinding
from miscope.families.parameterized_variant import ParameterizedVariant


class _FakeFamily:
    name = "modulo_addition_1layer"


class _FakeVariant:
    """Minimal Variant surface the parameterize handle + loader need."""

    def __init__(self, root: Path):
        self.name = "p5_seed1_dseed2"
        self.variant_dir = root / "v"
        self.artifacts_dir = self.variant_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.family = _FakeFamily()
        self.params = {"prime": 5}

    @property
    def artifacts(self):
        from miscope.analysis.artifact_loader import ArtifactLoader

        return ArtifactLoader(str(self.artifacts_dir))

    def parameterize(self, *, label=None, local=None, **bindings):
        from miscope.analysis.parameters import Binding, Parameterization

        bound = []
        for name, value in bindings.items():
            bound.append(value if isinstance(value, Binding) else LiteralBinding(name, value))
        for analyzer, params in (local or {}).items():
            for name, value in params.items():
                bound.append(LiteralBinding(name, value, analyzer=analyzer))
        return ParameterizedVariant(self, Parameterization(bindings=tuple(bound), label=label))


def _write_cross_epoch(variant, analyzer, recipe_sig, value):
    d = Path(analyzer_dir(str(variant.artifacts_dir), analyzer, recipe_sig))
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "cross_epoch", reference_epoch=np.array(value, dtype=np.int64))


def test_empty_parameterization_reads_default_plane(tmp_path):
    variant = _FakeVariant(tmp_path)
    _write_cross_epoch(variant, "parameter_dmd", "", 100)  # default plane
    handle = variant.parameterize()  # no bindings -> empty
    assert handle.recipe_map == {}
    got = handle.artifacts.load_cross_epoch("parameter_dmd", fields=["reference_epoch"])
    assert int(got["reference_epoch"]) == 100


def test_parameterized_handle_reads_recipe_plane(tmp_path):
    variant = _FakeVariant(tmp_path)
    # Pin parameter_dmd.reference_epoch=20000 -> a non-empty recipe for parameter_dmd.
    handle = variant.parameterize(local={"parameter_dmd": {"reference_epoch": 20000}})
    sig = handle.recipe_map["parameter_dmd"]
    assert sig  # non-empty recipe signature
    # Default plane and recipe plane hold different recorded values.
    _write_cross_epoch(variant, "parameter_dmd", "", 100)
    _write_cross_epoch(variant, "parameter_dmd", sig, 20000)

    default_val = variant.artifacts.load_cross_epoch("parameter_dmd", fields=["reference_epoch"])
    recipe_val = handle.artifacts.load_cross_epoch("parameter_dmd", fields=["reference_epoch"])
    assert int(default_val["reference_epoch"]) == 100
    assert int(recipe_val["reference_epoch"]) == 20000


def test_handle_delegates_unknown_attributes_to_variant(tmp_path):
    variant = _FakeVariant(tmp_path)
    handle = variant.parameterize(local={"parameter_dmd": {"reference_epoch": 1}})
    assert handle.name == variant.name
    assert handle.params == variant.params
