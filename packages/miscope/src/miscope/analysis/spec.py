"""Analyzer Spec — declarative metadata for analyzers (REQ_120 + REQ_121).

A ``Spec`` is the small declarative object the Planner consults to know
*what an analyzer needs* without inspecting its ``.analyze()`` body.
Specs live next to the analyzers they describe (each module exports a
module-level ``SPEC``) and are registered through the ``@register_analyzer``
decorator in :mod:`miscope.analysis.registry`.

A Spec declares its input materializations structurally via ``inputs`` and
its output scope via ``output_scope``. Capability requirements — upstream
artifacts, whether model weights or an activation cache are needed — are
*derived* from those declarations, not authored separately. The internal
execution phase (the primary/secondary/cross-epoch grouping) is likewise
derived by the Planner via ``derive_category``; it is not part of the Spec's
public surface (REQ_132). The Pipeline materializes whatever the Spec asks
for and hands the analyzer a uniform
:class:`miscope.analysis.inputs.ResolvedInputs` value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from miscope.analysis.inputs import (
    InputSpec,
    derive_needs_activation_cache,
    derive_needs_model_weights,
    derive_required_artifacts,
)
from miscope.analysis.output_schema import OutputField
from miscope.analysis.parameters import ParameterSpec

OutputScope = Literal["per_epoch", "cross_epoch"]


@runtime_checkable
class SchemaProducer(Protocol):
    """The narrow contract shared by :class:`AnalyzerSpec` and ``DerivedTableSpec``.

    REQ_141 introduces derived tables as a second kind of output producer. Both
    declare a ``name``, a schema ``version``, and an ``outputs`` tuple of
    :class:`~miscope.analysis.output_schema.OutputField` — that is the *only* thing
    they share (analyzers also have inputs/parameters/scope; derived tables have a
    query and input tables). This protocol is the seam that lets the registry's
    enumeration and reverse-lookup (``field()``, ``search()``, the output-schema
    half of ``validate()``) treat both kinds uniformly **without** collapsing the
    two concrete spec types into one lossy record.

    Members are declared read-only so frozen dataclasses (both concrete specs are
    ``@dataclass(frozen=True)``) satisfy the protocol.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def outputs(self) -> tuple[OutputField, ...]: ...


@dataclass(frozen=True)
class AnalyzerSpec:
    """Declarative metadata describing an analyzer's needs.

    Author ``inputs`` and ``output_scope``; everything else is derived::

        AnalyzerSpec(
            name="x",
            output_scope="per_epoch",
            inputs=(ModelInput(needs_cache=False),),
            required_hooks=(),
        )

    Attributes:
        name: Unique identifier. Mirrors the analyzer's ``name`` property.
        output_scope: "per_epoch" (one artifact per epoch) or "cross_epoch"
            (one artifact across all epochs). Default "per_epoch".
        inputs: Declared input materializations.
        required_hooks: Canonical hook names the analyzer reads.
        produces_summary: Whether the analyzer implements the
            ``get_summary_keys`` / ``compute_summary`` surface (REQ_022).
        outputs: Declared output fields (REQ_107). Each :class:`OutputField`
            states a field's name, dtype, ``kind`` (columnar|tensor), keying
            ``coords``, and a one-line description. This is the write-routing +
            join-key + discoverability declaration REQ_110 consumes. The names
            match the analyzer's ``analyze()`` return-dict keys (and the on-disk
            artifact keys) — the honesty contract.
        version: Output-schema version (REQ_107). Bumped when a field is added,
            removed, or its dtype changes; consumers declare the minimum version
            they are compatible with, and drift detection compares the two.
        parameters: Declared generation parameters (REQ_138). Each
            :class:`~miscope.analysis.parameters.ParameterSpec` states a parameter's
            name, dtype, scope, and default *binding*. An analyzer reads its declared
            parameters from ``inputs.parameters`` (scoped to exactly these names);
            reading an undeclared parameter raises. These are the bindings that, when
            non-default, form an artifact's storage recipe — the parameterization
            coordinate REQ_110 threads through.
    """

    name: str
    output_scope: OutputScope = "per_epoch"
    inputs: tuple[InputSpec, ...] = ()
    required_hooks: tuple[str, ...] = ()
    produces_summary: bool = False
    outputs: tuple[OutputField, ...] = ()
    version: int = 1
    parameters: tuple[ParameterSpec, ...] = ()

    # ----- Derived properties ----------------------------------------------

    @property
    def requires(self) -> tuple[str, ...]:
        """Names of upstream analyzers — derived from ``inputs``."""
        return derive_required_artifacts(self.inputs)

    @property
    def requires_model_weights(self) -> bool:
        return derive_needs_model_weights(self.inputs)

    @property
    def requires_activation_cache(self) -> bool:
        return derive_needs_activation_cache(self.inputs)

    # ----- Output schema helpers (REQ_107) ---------------------------------

    def output_names(self) -> tuple[str, ...]:
        """Names of every declared output field, in declaration order."""
        return tuple(f.name for f in self.outputs)

    def output_field(self, name: str) -> OutputField:
        """Look up a declared output field by name."""
        for f in self.outputs:
            if f.name == name:
                return f
        raise KeyError(
            f"Analyzer '{self.name}' declares no output field '{name}'. "
            f"Declared: {list(self.output_names())}"
        )

    # ----- Parameter helpers (REQ_138) -------------------------------------

    def parameter_names(self) -> tuple[str, ...]:
        """Names of every declared generation parameter, in declaration order."""
        return tuple(p.name for p in self.parameters)

    def parameter(self, name: str) -> ParameterSpec:
        """Look up a declared generation parameter by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(
            f"Analyzer '{self.name}' declares no parameter '{name}'. "
            f"Declared: {list(self.parameter_names())}"
        )
