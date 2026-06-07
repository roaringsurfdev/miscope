"""Registry index construction, validation, and drift detection (REQ_107).

The registry is the codebase's ``INFORMATION_SCHEMA``: it enumerates every
analyzer output field and every DataView field, with each field's ``kind``,
keying ``coords``, and one-line description, plus the producer/consumer edges
between them. It reads from the existing registration surfaces — the
:class:`~miscope.analysis.registry.AnalyzerRegistry` (Specs, now carrying
``outputs``) and the DataView catalog — so there is no parallel manifest to
drift from.

``build_index`` is pure (no disk IO). ``validate`` enforces the load-time
contract: every analyzer declares ≥1 output, every output dtype is real, and
every DataView source resolves against its producer's declared schema and
version (drift detection). The public API in :mod:`miscope.registry` wraps these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from miscope.analysis.derived_table import DerivedTableRegistry, DerivedTableSpec
from miscope.analysis.inputs import ArtifactInput
from miscope.analysis.output_schema import OutputField
from miscope.analysis.parameters import ParameterSpec, ReferenceBinding
from miscope.analysis.registry import AnalyzerRegistry
from miscope.analysis.spec import AnalyzerSpec, SchemaProducer
from miscope.views.dataview_catalog import (
    DataViewDefinition,
    DataViewField,
    DataViewSource,
    _dataview_catalog,
)


@dataclass(frozen=True)
class FieldInfo:
    """Reverse-lookup result for :func:`miscope.registry.field`.

    Attributes:
        name: The field name queried.
        producers: ``(analyzer_name, OutputField)`` pairs that declare this field.
        derived_producers: ``(derived_table_name, OutputField)`` pairs — derived
            tables that declare this field (REQ_141). Kept distinct from analyzer
            producers so ``field()`` names the derived table as the producer.
        dataview_fields: ``(dataview_name, DataViewField)`` pairs that expose it.
        dataview_consumers: DataView names that declare a source consuming this
            field (field-level edge).
        analyzer_consumers: Analyzer names that consume a producer's artifact
            (artifact-level edge — analyzers depend on whole artifacts, not fields).
    """

    name: str
    producers: tuple[tuple[str, OutputField], ...]
    dataview_fields: tuple[tuple[str, DataViewField], ...]
    dataview_consumers: tuple[str, ...]
    analyzer_consumers: tuple[str, ...]
    derived_producers: tuple[tuple[str, OutputField], ...] = ()

    def __repr__(self) -> str:
        if not self.producers and not self.dataview_fields and not self.derived_producers:
            return f"FieldInfo(name={self.name!r}, <not found>)"
        lines = [f"field {self.name!r}"]
        for an, f in self.producers:
            lines.append(
                f"  produced by analyzer {an!r}: kind={f.kind.value}, "
                f"coords={list(f.coord_names)}, dtype={f.dtype}"
            )
            lines.append(f"      {f.description}")
        for dt, f in self.derived_producers:
            lines.append(
                f"  produced by derived table {dt!r}: kind={f.kind.value}, "
                f"coords={list(f.coord_names)}, dtype={f.dtype}"
            )
            lines.append(f"      {f.description}")
        for dv, f in self.dataview_fields:
            lines.append(f"  exposed by dataview {dv!r}: kind={f.kind.value if f.kind else '?'}")
        if self.dataview_consumers:
            lines.append(f"  consumed by dataviews: {list(self.dataview_consumers)}")
        if self.analyzer_consumers:
            lines.append(
                f"  downstream analyzers (artifact-level): {list(self.analyzer_consumers)}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class RegistryIndex:
    """The built registry: analyzer Specs + DataView definitions + derived tables."""

    analyzers: tuple[AnalyzerSpec, ...]
    dataviews: tuple[DataViewDefinition, ...]
    derived: tuple[DerivedTableSpec, ...] = ()

    def schema_producers(self) -> tuple[SchemaProducer, ...]:
        """Every output producer (analyzers + derived tables) as ``SchemaProducer``.

        The shared seam (REQ_141) over which field enumeration and reverse-lookup
        iterate uniformly, without unifying the two concrete spec types.
        """
        return (*self.analyzers, *self.derived)

    def variant_key_columns(self, family: object) -> tuple[str, ...]:
        """The family-owned expansion of the ``variant`` coord (REQ_107).

        Returns ``("variant_id", *domain_parameters)`` — the opaque composed
        handle for cross-family joins plus the family's declared parameter
        columns for in-family filters (``WHERE prime > 100``). This replaces the
        hardcoded ``{prime}_{model_seed}_{data_seed}`` key.
        """
        params = tuple(getattr(family, "domain_parameters", {}).keys())
        return ("variant_id", *params)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index() -> RegistryIndex:
    """Build the registry from the analyzer and DataView registration surfaces.

    Importing this triggers registration of the bundled analyzers and DataViews
    via their package import side effects, so the index sees the full set.
    """
    # Import for registration side effects (idempotent). The derived-table
    # declarations are pure data (no warehouse dependency), so importing them keeps
    # the registry off the warehouse's load path.
    import miscope.analysis.analyzers  # noqa: F401
    import miscope.analysis.derived_tables  # noqa: F401
    import miscope.analysis.derived_tables_windows  # noqa: F401 — isolated window layer (REQ_144)
    import miscope.views.dataview_universal  # noqa: F401

    analyzers = tuple(sorted(AnalyzerRegistry.list_specs(), key=lambda s: s.name))
    dataviews = tuple(_dataview_catalog.get(name) for name in _dataview_catalog.names())
    derived = tuple(sorted(DerivedTableRegistry.list_specs(), key=lambda d: d.name))
    return RegistryIndex(analyzers=analyzers, dataviews=dataviews, derived=derived)


# ---------------------------------------------------------------------------
# Validation + drift detection
# ---------------------------------------------------------------------------


def _valid_dtype(dtype: str) -> bool:
    """Whether a declared dtype string is a real numpy dtype (``str`` allowed)."""
    if dtype == "str":
        return True
    try:
        np.dtype(dtype)
        return True
    except TypeError:
        return False


def validate(index: RegistryIndex) -> None:
    """Enforce the registry contract; raise ``RegistryError`` on any violation.

    1. Every analyzer declares at least one output field.
    2. Every declared dtype is a real dtype.
    3. Every declared generation parameter is well-formed (REQ_138): valid dtype,
       a known scope, and a default binding whose ``ReferenceBinding.source_analyzer``
       is a registered analyzer (a reference binding is a DAG edge, so its source
       must exist). This is the load-time half of the parameter discipline; the
       runtime half (reading an undeclared parameter raises) lives in the scoped
       parameters mapping handed to ``analyze()``.
    4. Drift: every DataView source resolves against its producer's declared
       schema (analyzer registered, version ≥ min_version, every consumed field
       declared).
    """
    problems: list[str] = []
    specs_by_name = {s.name: s for s in index.analyzers}

    for spec in index.analyzers:
        problems.extend(_check_output_schema(f"analyzer {spec.name!r}", spec.outputs))
        for p in spec.parameters:
            problems.extend(_check_parameter(spec.name, p, specs_by_name))

    # Derived tables (REQ_141) carry the same output-schema contract as analyzers,
    # plus must declare the warehouse tables they query.
    for dt in index.derived:
        problems.extend(_check_output_schema(f"derived table {dt.name!r}", dt.outputs))
        if not dt.input_tables:
            problems.append(
                f"derived table {dt.name!r} declares no input_tables "
                f"(REQ_141: a derived table is a query over warehouse tables)"
            )

    for dv in index.dataviews:
        for src in dv.sources:
            problems.extend(_check_source(dv, src, specs_by_name))

    if problems:
        raise RegistryError("registry validation failed:\n  - " + "\n  - ".join(problems))


def _check_output_schema(label: str, outputs: tuple[OutputField, ...]) -> list[str]:
    """Output-schema contract shared by analyzers and derived tables (REQ_107/141).

    Every producer declares at least one output field, and every declared dtype is
    a real dtype. ``label`` carries the producer kind + name for the message.
    """
    out: list[str] = []
    if not outputs:
        out.append(
            f"{label} declares no output fields "
            f"(REQ_107: every producer must declare its output schema)"
        )
    for f in outputs:
        if not _valid_dtype(f.dtype):
            out.append(f"{label} field {f.name!r} has unknown dtype {f.dtype!r}")
    return out


def _check_parameter(
    analyzer_name: str,
    param: ParameterSpec,
    specs_by_name: dict[str, AnalyzerSpec],
) -> list[str]:
    """Well-formedness checks for one declared generation parameter (REQ_138)."""
    out: list[str] = []
    if not _valid_dtype(param.dtype):
        out.append(
            f"analyzer {analyzer_name!r} parameter {param.name!r} has unknown dtype {param.dtype!r}"
        )
    if param.scope not in ("run", "analyzer"):
        out.append(
            f"analyzer {analyzer_name!r} parameter {param.name!r} has unknown "
            f"scope {param.scope!r} (expected 'run' or 'analyzer')"
        )
    default = param.default
    if isinstance(default, ReferenceBinding) and default.source_analyzer not in specs_by_name:
        out.append(
            f"analyzer {analyzer_name!r} parameter {param.name!r} defaults to a "
            f"reference into {default.source_analyzer!r} which is not registered"
        )
    return out


def _check_source(
    dv: DataViewDefinition,
    src: DataViewSource,
    specs_by_name: dict[str, AnalyzerSpec],
) -> list[str]:
    """Drift check for one DataView source against its producer's schema."""
    producer = specs_by_name.get(src.analyzer_name)
    if producer is None:
        return [
            f"dataview {dv.name!r} sources analyzer {src.analyzer_name!r} which is not registered"
        ]
    out: list[str] = []
    if producer.version < src.min_version:
        out.append(
            f"dataview {dv.name!r} requires {src.analyzer_name!r} "
            f">= v{src.min_version} but it is v{producer.version}"
        )
    declared = set(producer.output_names())
    for fname in src.fields:
        if fname not in declared:
            out.append(
                f"dataview {dv.name!r} consumes field {fname!r} from "
                f"{src.analyzer_name!r} which does not declare it "
                f"(declared: {sorted(declared)})"
            )
    return out


# ---------------------------------------------------------------------------
# Reverse lookups (used by miscope.registry.field)
# ---------------------------------------------------------------------------


def lookup_field(index: RegistryIndex, name: str) -> FieldInfo:
    """Reverse lookup: producers, exposing DataViews, and consumers of a field."""
    producers = tuple((s.name, f) for s in index.analyzers for f in s.outputs if f.name == name)
    derived_producers = tuple(
        (d.name, f) for d in index.derived for f in d.outputs if f.name == name
    )
    dv_fields = tuple(
        (dv.name, f) for dv in index.dataviews for f in dv.schema.fields if f.name == name
    )
    producer_names = {an for an, _ in producers} | {dt for dt, _ in derived_producers}
    dv_consumers = tuple(
        dv.name
        for dv in index.dataviews
        for src in dv.sources
        if name in src.fields and src.analyzer_name in producer_names
    )
    analyzer_consumers = tuple(
        s.name
        for s in index.analyzers
        for i in s.inputs
        if isinstance(i, ArtifactInput) and i.analyzer_name in producer_names
    )
    return FieldInfo(
        name=name,
        producers=producers,
        derived_producers=derived_producers,
        dataview_fields=dv_fields,
        dataview_consumers=dv_consumers,
        analyzer_consumers=tuple(dict.fromkeys(analyzer_consumers)),
    )


class RegistryError(RuntimeError):
    """Raised when the registry fails its load-time validation / drift checks."""
