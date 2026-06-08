"""Tests for the variant summary surface (REQ_144).

The 814-line ``VariantAnalysisSummary`` engine was retired in REQ_144 Stage 3: the
summary is now assembled from the conformed warehouse tables by
``variant_summary_assembler.assemble_variant_summary`` (validated value-identical
to the old engine on the baselines), and this module is the thin
``write_variant_summary`` + ``build_variant_registry`` surface over it. These tests
cover the *roll-up* responsibilities (write one file; aggregate across variants);
the assembler's field-level correctness is covered by the cluster/window derived
tables and the assembler's own baseline parity.
"""

from __future__ import annotations

import json

from miscope.analysis import variant_analysis_summary as vas
from miscope.warehouse import paths


def _make_family(tmp_path):
    """A minimal 1-layer family with two empty variant directories."""
    from miscope.families.base_model_family import BaseModelFamily

    config = {
        "name": "modulo_addition_1layer",
        "display_name": "Modulo Addition (1 Layer)",
        "description": "Test",
        "architecture": {},
        "domain_parameters": {
            "prime": {"type": "int"},
            "seed": {"type": "int"},
            "data_seed": {"type": "int"},
        },
        "analyzers": [],
        "visualizations": [],
        "analysis_dataset": {"type": "modulo_addition_grid"},
        "variant_pattern": "p{prime}_seed{seed}_dseed{data_seed}",
    }
    family = BaseModelFamily(config, data_root=tmp_path)
    family.variants_dir.mkdir(parents=True)
    for prime, mseed in [(113, 485), (113, 999)]:
        (family.variants_dir / f"p{prime}_seed{mseed}_dseed598").mkdir(parents=True)
    return family


def _mark_outcomes(variant) -> None:
    """Create the sentinel outcomes table dir so the variant counts as analyzed."""
    paths.table_dir(variant, "loss_outcomes").mkdir(parents=True, exist_ok=True)


def test_write_variant_summary_writes_assembled_dict(tmp_path, monkeypatch):
    family = _make_family(tmp_path)
    variant = family.variants[0]
    monkeypatch.setattr(
        vas, "assemble_variant_summary", lambda v: {"prime": v.params["prime"], "ok": True}
    )

    out = vas.write_variant_summary(variant)

    assert out == variant.variant_dir / "variant_summary.json"
    written = json.loads(out.read_text())
    assert written == {"prime": 113, "ok": True}


def test_build_variant_registry_one_entry_per_variant(tmp_path, monkeypatch):
    family = _make_family(tmp_path)
    for v in family.variants:
        _mark_outcomes(v)
    # The assembler is mocked (its correctness is tested on baselines); the registry's
    # job is to roll the assembled dicts up with variant_id + domain-parameter columns.
    monkeypatch.setattr(
        vas,
        "assemble_variant_summary",
        lambda v: {"prime": v.params["prime"], "failure_mode": "healthy"},
    )

    registry_path = vas.build_variant_registry(family)

    assert registry_path.exists()
    registry = json.loads(registry_path.read_text())
    assert len(registry) == 2

    # variant_id is the family-owned directory handle (REQ_107), and the declared
    # domain parameters are attached as columns for filtering.
    by_id = {e["variant_id"]: e for e in registry}
    assert set(by_id) == {"p113_seed485_dseed598", "p113_seed999_dseed598"}
    assert by_id["p113_seed485_dseed598"]["prime"] == 113
    assert by_id["p113_seed485_dseed598"]["seed"] == 485
    assert by_id["p113_seed485_dseed598"]["data_seed"] == 598
    # Entries are ordered by variant_id for a stable file.
    assert [e["variant_id"] for e in registry] == sorted(by_id)


def test_build_variant_registry_skips_unmaterialized_variants(tmp_path, monkeypatch):
    family = _make_family(tmp_path)
    # Only the first variant has its outcomes materialized; the other is skipped.
    _mark_outcomes(family.variants[0])
    monkeypatch.setattr(vas, "assemble_variant_summary", lambda v: {"prime": v.params["prime"]})

    registry = json.loads(vas.build_variant_registry(family).read_text())

    assert len(registry) == 1
    assert registry[0]["variant_id"] == family.variants[0].name
