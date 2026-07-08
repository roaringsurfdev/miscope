"""Derived tables — declarative, registered queries over warehouse tables (REQ_141).

A :class:`DerivedTableSpec` is a first-class output producer alongside an analyzer:
it declares an output schema (REQ_107 :class:`~miscope.analysis.output_schema.OutputField`s
with kind + coords), a ``version``, the warehouse tables it reads, and the query
that produces it — and it carries the same audit trail an analyzer does. The
difference is the *source*: an analyzer maps an on-disk artifact; a derived table
runs a query over already-materialized warehouse tables.

It satisfies the :class:`~miscope.analysis.spec.SchemaProducer` protocol (``name``,
``version``, ``outputs``), so the registry enumerates and reverse-looks-up its
fields exactly as it does an analyzer's — see :mod:`miscope.registry`.

The declaration is **pure data**: the query is SQL text over warehouse *table
names* (not path literals), and the input tables are names. It carries no warehouse
dependency, so the registry can enumerate derived tables without importing the
warehouse. The executor that runs a spec and materializes it lives in
:mod:`miscope.warehouse.derived` and is imported only at materialize time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from miscope.analysis.output_schema import OutputField


@dataclass(frozen=True)
class DerivedTableSpec:
    """Declarative metadata for one derived table (REQ_141).

    Attributes:
        name: Unique table name — its identity on the query surface, mirroring an
            analyzer's semantic-table name. ``miscope.query.open(...)`` exposes a
            view of this name.
        query: The query that produces the table, as SQL text over the warehouse's
            registered table names (``input_tables``). Authored as text so the
            provenance is the literal computation, auditable in the registry.
        input_tables: The warehouse tables this query reads — the DAG edges that
            make the derived table a node downstream of its inputs (freshness).
        outputs: Declared output fields (REQ_107). Names match the columns the
            query yields — the honesty contract, identical to an analyzer's.
        version: Output-schema version. Bumped when a field is added/removed or a
            dtype changes; consumers declare a minimum compatible version.
        materialized: Cost-driven storage property (REQ_141). ``True`` persists the
            result to a Parquet table read back as bytes (expensive aggregations,
            paid once off the interactive path); ``False`` is a query-time view
            (cheap, always live, no storage).
    """

    name: str
    query: str
    input_tables: tuple[str, ...]
    outputs: tuple[OutputField, ...] = field(default_factory=tuple)
    version: int = 1
    materialized: bool = True

    def output_names(self) -> tuple[str, ...]:
        """Names of every declared output field, in declaration order."""
        return tuple(f.name for f in self.outputs)

    def output_field(self, name: str) -> OutputField:
        """Look up a declared output field by name."""
        for f in self.outputs:
            if f.name == name:
                return f
        raise KeyError(
            f"Derived table '{self.name}' declares no output field '{name}'. "
            f"Declared: {list(self.output_names())}"
        )


# ---------------------------------------------------------------------------
# Registry — mirrors miscope.analysis.registry.AnalyzerRegistry
# ---------------------------------------------------------------------------

# Module-level backing state. Derived tables self-register at import time so
# ``build_index`` collects them via the same import-side-effect pattern the
# analyzer and DataView surfaces use.
_derived: dict[str, DerivedTableSpec] = {}


def register_derived_table(spec: DerivedTableSpec) -> DerivedTableSpec:
    """Register a derived table under its name; returns the spec for module binding.

    Usage (declaration sites bind the result so the spec is importable)::

        NEURON_FREQUENCY = register_derived_table(DerivedTableSpec(name=..., ...))
    """
    _derived[spec.name] = spec
    return spec


class DerivedTableRegistry:
    """Registry of derived-table Specs (REQ_141)."""

    @classmethod
    def get(cls, name: str) -> DerivedTableSpec:
        if name not in _derived:
            raise KeyError(f"No derived table '{name}'. Registered: {sorted(_derived)}")
        return _derived[name]

    @classmethod
    def has(cls, name: str) -> bool:
        return name in _derived

    @classmethod
    def list_specs(cls) -> list[DerivedTableSpec]:
        return list(_derived.values())

    @classmethod
    def list_names(cls) -> list[str]:
        return sorted(_derived)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered derived tables. Mainly for testing."""
        _derived.clear()
