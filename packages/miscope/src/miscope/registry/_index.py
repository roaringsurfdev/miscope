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

from miscope.analysis.inputs import ArtifactInput
from miscope.analysis.output_schema import OutputField
from miscope.analysis.parameters import ParameterSpec, ReferenceBinding
from miscope.analysis.registry import AnalyzerRegistry
from miscope.analysis.spec import AnalyzerSpec
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

    def __repr__(self) -> str:
        if not self.producers and not self.dataview_fields:
            return f"FieldInfo(name={self.name!r}, <not found>)"
        lines = [f"field {self.name!r}"]
        for an, f in self.producers:
            lines.append(
                f"  produced by analyzer {an!r}: kind={f.kind.value}, "
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
    """The built registry: analyzer Specs + DataView definitions, indexed."""

    analyzers: tuple[AnalyzerSpec, ...]
    dataviews: tuple[DataViewDefinition, ...]

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
    # Import for registration side effects (idempotent).
    import miscope.analysis.analyzers  # noqa: F401
    import miscope.views.dataview_universal  # noqa: F401

    analyzers = tuple(sorted(AnalyzerRegistry.list_specs(), key=lambda s: s.name))
    dataviews = tuple(_dataview_catalog.get(name) for name in _dataview_catalog.names())
    return RegistryIndex(analyzers=analyzers, dataviews=dataviews)


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
        if not spec.outputs:
            problems.append(
                f"analyzer {spec.name!r} declares no output fields "
                f"(REQ_107: every analyzer must declare its output schema)"
            )
        for f in spec.outputs:
            if not _valid_dtype(f.dtype):
                problems.append(
                    f"analyzer {spec.name!r} field {f.name!r} has unknown dtype {f.dtype!r}"
                )
        for p in spec.parameters:
            problems.extend(_check_parameter(spec.name, p, specs_by_name))

    for dv in index.dataviews:
        for src in dv.sources:
            problems.extend(_check_source(dv, src, specs_by_name))

    if problems:
        raise RegistryError("registry validation failed:\n  - " + "\n  - ".join(problems))


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
    dv_fields = tuple(
        (dv.name, f) for dv in index.dataviews for f in dv.schema.fields if f.name == name
    )
    producer_names = {an for an, _ in producers}
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
        dataview_fields=dv_fields,
        dataview_consumers=dv_consumers,
        analyzer_consumers=tuple(dict.fromkeys(analyzer_consumers)),
    )


class RegistryError(RuntimeError):
    """Raised when the registry fails its load-time validation / drift checks."""
