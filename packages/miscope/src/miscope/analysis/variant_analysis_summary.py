"""Variant summary writing + cross-variant registry roll-up (REQ_144).

The 814-line imperative ``VariantAnalysisSummary`` engine is gone. Its output — the
per-variant ``variant_summary.json`` — is now produced by
:func:`~miscope.analysis.variant_summary_assembler.assemble_variant_summary`, which
reads the conformed warehouse tables (the seven Stage-2a outcome clusters + the
Stage-2b window tables) and adds the two Python classifiers. This module is the thin
surface over that assembler:

- :func:`write_variant_summary` writes one variant's ``variant_summary.json`` (the
  engine's old job, now a table read).
- :func:`build_variant_registry` rolls every analyzed variant's summary into the
  family's ``variant_registry.json`` — a cross-variant aggregate the per-variant
  derived-table executor cannot produce (it never merges across variants), so it
  stays an explicit roll-up here. REQ_144 Stage 4 reframes the registry as a pure
  query view over ``variant_outcomes``; until then it is materialized to the file.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from miscope.analysis.variant_summary_assembler import assemble_variant_summary
from miscope.warehouse import paths

if TYPE_CHECKING:
    from pathlib import Path

    from miscope.families.protocols import ModelFamily
    from miscope.families.variant import Variant

logger = logging.getLogger(__name__)

# A variant is "summarizable" once its stable outcome clusters are materialized;
# loss_outcomes is the keystone cluster, present iff the warehouse pass ran.
_OUTCOMES_SENTINEL_TABLE = "loss_outcomes"


def write_variant_summary(variant: Variant) -> Path:
    """Assemble the variant summary from the warehouse and write variant_summary.json."""
    summary = assemble_variant_summary(variant)
    output_path = variant.variant_dir / "variant_summary.json"
    output_path.write_text(json.dumps(summary, default=int, indent=2))
    return output_path


def build_variant_registry(family: ModelFamily) -> Path:
    """Roll every analyzed variant's summary into the family's variant_registry.json.

    Each entry is the variant's assembled summary plus its ``variant_id`` (the
    family-owned directory handle) and declared ``domain_parameters`` columns.
    Variants whose warehouse outcomes have not been materialized are skipped
    (mirroring the old skip of variants without a summary). Entries are ordered by
    ``variant_id`` for a stable file.
    """
    registry = [entry for variant in family.variants if (entry := _registry_entry(variant))]
    registry.sort(key=lambda entry: entry["variant_id"])

    output_path = family.family_dir / "variant_registry.json"
    output_path.write_text(json.dumps(registry, default=int, indent=2))
    return output_path


def _registry_entry(variant: Variant) -> dict[str, Any] | None:
    """One registry entry from the warehouse, or ``None`` if absent/unassemblable.

    Per-variant isolation (REQ_140 quarantine discipline): a variant whose outcomes
    aren't materialized, or whose assembly fails (e.g. partially-materialized stale
    tables), is skipped — never fatal to the cross-variant roll-up.
    """
    if not paths.table_dir(variant, _OUTCOMES_SENTINEL_TABLE).is_dir():
        return None
    try:
        entry = assemble_variant_summary(variant)
    except Exception as exc:  # noqa: BLE001 — quarantine one bad variant, keep the roll-up going
        logger.warning(
            "variant registry: skipping %s (%s: %s)", variant.name, type(exc).__name__, exc
        )
        return None
    entry["variant_id"] = variant.name
    for key, value in variant.params.items():
        entry.setdefault(key, value)
    return entry
