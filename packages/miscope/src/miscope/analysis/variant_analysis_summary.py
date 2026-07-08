"""Variant summary writing + cross-variant registry view (REQ_144).

The 814-line imperative ``VariantAnalysisSummary`` engine is gone. Its output — the
per-variant ``variant_summary.json`` — is now produced by
:func:`~miscope.analysis.variant_summary_assembler.assemble_variant_summary`, which
reads the conformed warehouse tables (the seven Stage-2a outcome clusters + the
Stage-2b window tables) and adds the two Python classifiers. This module is the thin
surface over that warehouse:

- :func:`write_variant_summary` writes one variant's ``variant_summary.json`` (the
  engine's old job, now a table read).
- :func:`assemble_variant_registry` is the family's variant registry — REQ_144
  Stage 4 (fork a) makes it a pure cross-variant *view* over the ``variant_outcomes``
  derived table (no ``variant_registry.json`` file), each row carrying the two
  Python classifications (fork b). Only the *stable* outcome layer feeds it; the
  provisional window layer never leaks in (fork e one-way DAG).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from miscope.analysis.derived_tables import VARIANT_OUTCOMES
from miscope.analysis.variant_summary_assembler import (
    _scalarize,
    assemble_variant_summary,
    classify_outcomes,
)

if TYPE_CHECKING:
    from pathlib import Path

    from miscope.families.protocols import ModelFamily
    from miscope.families.variant import Variant

logger = logging.getLogger(__name__)

# Key/provenance columns of variant_outcomes that are not registry-entry values.
# ``variant_id`` + the domain params (prime/seed/data_seed) are kept; ``run_set`` is
# the warehouse plane selector, not a summary field.
_REGISTRY_DROP_COLUMNS = frozenset({"run_set"})


def write_variant_summary(variant: Variant) -> Path:
    """Assemble the variant summary from the warehouse and write variant_summary.json."""
    summary = assemble_variant_summary(variant)
    output_path = variant.variant_dir / "variant_summary.json"
    output_path.write_text(json.dumps(summary, default=int, indent=2))
    return output_path


def assemble_variant_registry(family: ModelFamily) -> list[dict[str, Any]]:
    """The family's variant registry: a cross-variant view over ``variant_outcomes``.

    REQ_144 Stage 4 (fork a): the registry is no longer a written file — it is the
    cross-variant projection of the ``variant_outcomes`` derived table (one wide row
    per analyzed variant), each entry carrying ``model_seed``/``family`` and the two
    Python classifications (fork b). Pure stable layer: the provisional window tables
    never feed it. Variants whose outcomes are not materialized simply do not appear
    (their glob matches no file), mirroring the old skip.

    Raises:
        FileNotFoundError: If no variant has materialized ``variant_outcomes`` yet —
            preserving the missing-registry contract the consumers catch.
    """
    import miscope.query
    from miscope.warehouse import paths

    if VARIANT_OUTCOMES not in paths.list_family_tables(family):
        raise FileNotFoundError(
            f"No variant_outcomes materialized for family {family.name!r}. "
            "Run analysis (which materializes the warehouse) first."
        )
    with miscope.query.open(family=family) as con:
        frame = con.df(f"SELECT * FROM {VARIANT_OUTCOMES}")
    entries = [_registry_entry(row, family) for _, row in frame.iterrows()]
    entries.sort(key=lambda entry: entry["variant_id"])
    return entries


def _registry_entry(row: Any, family: ModelFamily) -> dict[str, Any]:
    """One registry entry from a ``variant_outcomes`` row + the two classifications.

    Scalarizes each cell the same way the per-variant assembler does (so the
    classifiers see identical Python scalars), maps ``seed`` to the consumer-facing
    ``model_seed``, stamps the family name, and appends the classification keys.
    """
    entry = {col: _scalarize(row[col]) for col in row.index if col not in _REGISTRY_DROP_COLUMNS}
    entry["model_seed"] = entry.get("seed")
    entry["family"] = family.name
    entry.update(classify_outcomes(entry))
    return entry
