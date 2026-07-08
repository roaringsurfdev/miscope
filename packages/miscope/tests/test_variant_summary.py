"""Tests for the variant summary surface (REQ_144).

The 814-line ``VariantAnalysisSummary`` engine was retired in REQ_144 Stage 3: the
summary is now assembled from the conformed warehouse tables by
``variant_summary_assembler.assemble_variant_summary`` (validated value-identical
to the old engine on the baselines), and this module is the thin
``write_variant_summary`` + ``assemble_variant_registry`` surface over it. Stage 4c
(fork a) turned the registry into a pure cross-variant *view* over the
``variant_outcomes`` derived table (no ``variant_registry.json`` file). These tests
cover the roll-up responsibilities (write one file; project + classify across
variants); the assembler's field-level correctness is covered by the cluster/window
derived tables and the assembler's own baseline parity.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd

from miscope.analysis import variant_analysis_summary as vas


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


def _outcomes_row(variant_id: str, seed: int) -> dict:
    """One variant_outcomes row with the fields the classifiers read."""
    return {
        "variant_id": variant_id,
        "prime": 113,
        "seed": seed,
        "data_seed": 598,
        "run_set": "default",
        "learned_frequencies": [9, 33, 38, 55],
        "second_descent_onset_epoch": 9503,
        "test_loss_final": 1.0e-7,
        "test_loss_min": 1.0e-7,
    }


@contextmanager
def _fake_query(frame: pd.DataFrame):
    """A miscope.query.open stand-in yielding a connection whose df() returns frame."""

    class _Con:
        def df(self, _query: str) -> pd.DataFrame:
            return frame

    yield _Con()


def _patch_warehouse(monkeypatch, frame: pd.DataFrame, *, table_present: bool = True) -> None:
    import miscope.query
    from miscope.warehouse import paths

    monkeypatch.setattr(
        paths,
        "list_family_tables",
        lambda family: ["variant_outcomes"] if table_present else [],
    )
    monkeypatch.setattr(miscope.query, "open", lambda **_: _fake_query(frame))


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


def test_assemble_variant_registry_projects_outcomes(tmp_path, monkeypatch):
    family = _make_family(tmp_path)
    frame = pd.DataFrame(
        [
            _outcomes_row("p113_seed999_dseed598", 999),
            _outcomes_row("p113_seed485_dseed598", 485),
        ]
    )
    _patch_warehouse(monkeypatch, frame)

    registry = vas.assemble_variant_registry(family)

    # One entry per outcomes row, ordered by variant_id for a stable result.
    assert [e["variant_id"] for e in registry] == [
        "p113_seed485_dseed598",
        "p113_seed999_dseed598",
    ]
    entry = registry[0]
    # Key columns survive; seed is mapped to the consumer-facing model_seed; family stamped.
    assert entry["prime"] == 113
    assert entry["seed"] == 485
    assert entry["model_seed"] == 485
    assert entry["data_seed"] == 598
    assert entry["family"] == "modulo_addition_1layer"
    # run_set is a warehouse plane selector, not a summary field.
    assert "run_set" not in entry
    # The two Python classifications (fork b) are computed over the row.
    assert entry["failure_mode"] == "healthy"
    assert entry["performance_classification"][0] == "healthy"


def test_assemble_variant_registry_missing_outcomes_raises(tmp_path, monkeypatch):
    family = _make_family(tmp_path)
    _patch_warehouse(monkeypatch, pd.DataFrame(), table_present=False)

    import pytest

    with pytest.raises(FileNotFoundError, match="variant_outcomes"):
        vas.assemble_variant_registry(family)
