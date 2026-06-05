"""Schema-stability gate — the guard that keeps published bundles immutable (REQ_110E).

A published bundle is frozen: once a reader cites ``data-ring-geometry-v1.0``, the
columns behind that tag never change. Evolution happens by minting a *new* bundle
version, never by mutating an old one. This module enforces that rule mechanically
by comparing a freshly built bundle against its committed baseline manifest and
classifying the difference per table:

- **identical** — same columns, same types. A rebuild at the same version is fine.
- **additive** — only new columns appended; every baseline column survives with
  its type. Allowed, *but only under a new bundle version* — the published tag
  stays immutable.
- **breaking** — a baseline column was removed (a rename reads as remove + add) or
  changed type. Never allowed silently; minting requires a new version, and the
  caller must acknowledge the break.

The gate reads declared schemas off the manifests (which the writer captured from
the materialized Parquet), so it needs no live data — a committed manifest history
is a sufficient baseline in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from miscope.publish.manifest import BundleManifest


class SchemaChange(str, Enum):
    """The classification of one bundle's schema delta against its baseline."""

    IDENTICAL = "identical"
    ADDITIVE = "additive"
    BREAKING = "breaking"


class SchemaBreakError(Exception):
    """Raised when a bundle would break a published schema or evolve without a version bump."""


@dataclass(frozen=True)
class TableDelta:
    """The per-table schema difference between a baseline and a rebuild."""

    table: str
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    retyped: tuple[tuple[str, str, str], ...] = ()  # (column, baseline_type, new_type)

    @property
    def change(self) -> SchemaChange:
        if self.removed or self.retyped:
            return SchemaChange.BREAKING
        if self.added:
            return SchemaChange.ADDITIVE
        return SchemaChange.IDENTICAL

    def describe(self) -> str:
        """A one-line human summary of the delta (for gate error messages / logs)."""
        parts: list[str] = []
        if self.removed:
            parts.append(f"removed {list(self.removed)}")
        if self.retyped:
            parts.append("retyped " + ", ".join(f"{c}: {a}->{b}" for c, a, b in self.retyped))
        if self.added:
            parts.append(f"added {list(self.added)}")
        return f"{self.table}: " + ("; ".join(parts) if parts else "no change")


@dataclass(frozen=True)
class GateReport:
    """The full gate verdict across every table compared (plus tables new to this bundle)."""

    deltas: tuple[TableDelta, ...] = ()
    new_tables: tuple[str, ...] = ()  # tables present now, absent from the baseline
    dropped_tables: tuple[str, ...] = field(default=())  # in baseline, absent now (breaking)

    @property
    def change(self) -> SchemaChange:
        """The bundle-level classification — the most severe per-table change."""
        if self.dropped_tables or any(d.change is SchemaChange.BREAKING for d in self.deltas):
            return SchemaChange.BREAKING
        if self.new_tables or any(d.change is SchemaChange.ADDITIVE for d in self.deltas):
            return SchemaChange.ADDITIVE
        return SchemaChange.IDENTICAL

    def breaking_reasons(self) -> list[str]:
        """Human-readable reasons the bundle is breaking (empty if it is not)."""
        reasons = [d.describe() for d in self.deltas if d.change is SchemaChange.BREAKING]
        if self.dropped_tables:
            reasons.append(f"dropped tables {list(self.dropped_tables)}")
        return reasons


def diff_table(table: str, baseline: dict[str, str], new: dict[str, str]) -> TableDelta:
    """Classify one table's column-schema delta (baseline -> new)."""
    added = tuple(c for c in new if c not in baseline)
    removed = tuple(c for c in baseline if c not in new)
    retyped = tuple(
        (c, baseline[c], new[c]) for c in baseline if c in new and baseline[c] != new[c]
    )
    return TableDelta(table=table, added=added, removed=removed, retyped=retyped)


def compare(baseline: BundleManifest, candidate: BundleManifest) -> GateReport:
    """Compare a candidate bundle against its baseline manifest, table by table."""
    base_tables = {t.name: t for t in baseline.tables}
    cand_tables = {t.name: t for t in candidate.tables}
    deltas = tuple(
        diff_table(name, base_tables[name].schema, cand.schema)
        for name, cand in cand_tables.items()
        if name in base_tables
    )
    new_tables = tuple(name for name in cand_tables if name not in base_tables)
    dropped = tuple(name for name in base_tables if name not in cand_tables)
    return GateReport(deltas=deltas, new_tables=new_tables, dropped_tables=dropped)


def enforce(baseline: BundleManifest, candidate: BundleManifest) -> GateReport:
    """Run the gate; raise :class:`SchemaBreakError` if the candidate is not publishable.

    Publishable means: a breaking change is refused outright, and any non-identical
    change (additive or new/dropped tables) must carry a new ``bundle_version`` so
    the published tag stays immutable.
    """
    report = compare(baseline, candidate)
    if report.change is SchemaChange.BREAKING:
        reasons = "; ".join(report.breaking_reasons())
        raise SchemaBreakError(
            f"Bundle '{candidate.bundle_name}' breaks the published schema of "
            f"'{baseline.bundle_version}': {reasons}. A breaking change requires a new "
            f"bundle minted under a new version — published tags are immutable."
        )
    if (
        report.change is SchemaChange.ADDITIVE
        and candidate.bundle_version == baseline.bundle_version
    ):
        raise SchemaBreakError(
            f"Bundle '{candidate.bundle_name}' changes the schema (additive) but reuses "
            f"version '{baseline.bundle_version}'. Bump bundle_version to publish — a "
            f"published tag is immutable."
        )
    return report
