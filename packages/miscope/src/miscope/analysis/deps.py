"""Scoped, lazy dependency accessor for analyzers (REQ_128).

An analyzer declares its upstream dependencies as ``ArtifactInput`` entries on
its Spec. The pipeline constructs one :class:`DepsAccessor` per analyzer,
scoped to exactly those declared names, and hands it to ``.analyze()`` on
``ResolvedInputs.deps``. The analyzer loads what it needs, when it needs it, in
a releasable scope — instead of receiving every dependency eagerly
materialized for the lifetime of the inputs object.

Four verbs, one honest memory shape each (REQ_128 design lock):

    load_epoch(name, epoch, *, fields)   one epoch's dict at a given epoch
    stream(name, *, fields)              iterator of (epoch, dict), one resident
    load_stack(name, *, fields)          stacked (n_epochs, ...) arrays
    load_cross_epoch(name, *, fields)    an upstream's single cross_epoch.npz

The first three read **per-epoch** upstreams; ``load_cross_epoch`` reads an
upstream that is itself a **cross-epoch** analyzer. ``fields`` is required on
every verb (use the :data:`ALL` sentinel to request everything) and is
validated against the upstream artifact's actual on-disk field set.

The accessor is a thin wrapper over :class:`ArtifactLoader` — the storage
primitive still owns all path composition and file access (PROJECT.md
constraint 3). The accessor adds scoping, the verb surface, and the ``fields``
contract; it does not open files itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from miscope.analysis.artifact_loader import ArtifactLoader


class _AllFields:
    """Sentinel requesting every field of an artifact, legibly (`fields=ALL`)."""

    _instance: _AllFields | None = None

    def __new__(cls) -> _AllFields:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALL"


ALL = _AllFields()

FieldSpec = list[str] | _AllFields


class UndeclaredDependencyError(ValueError):
    """Raised when an analyzer accesses an upstream it did not declare."""


class ArtifactLayoutError(ValueError):
    """Raised when a verb's storage layout doesn't match the upstream's."""


def _resolve_fields(fields: FieldSpec | str | None) -> list[str] | None:
    """Normalize the required ``fields`` argument to the loader's contract.

    ``ALL`` maps to ``None`` (the loader's "all fields"); a list passes
    through. ``None`` and bare strings are rejected so the required-fields
    discipline can't erode into a silent whole-artifact load. (The parameter
    type is widened beyond ``FieldSpec`` so these runtime guards are honest
    about what analyzer authors may actually pass.)
    """
    if isinstance(fields, _AllFields):
        return None
    if fields is None:
        raise ValueError("fields is required; pass a list of field names or the ALL sentinel.")
    if isinstance(fields, str):
        raise ValueError(
            f"fields must be a list of names or the ALL sentinel, not the string {fields!r}."
        )
    return list(fields)


class DepsAccessor:
    """Per-analyzer, scoped, lazy accessor over upstream artifacts.

    Scoped to the set of upstream names the analyzer declared on its Spec;
    any verb call with an undeclared name raises
    :class:`UndeclaredDependencyError`.
    """

    def __init__(self, loader: ArtifactLoader, allowed: frozenset[str]):
        self._loader = loader
        self._allowed = allowed

    # ----- verbs ------------------------------------------------------------

    def load_epoch(self, name: str, epoch: int, *, fields: FieldSpec) -> dict[str, np.ndarray]:
        """One epoch's dict at ``epoch`` (per-epoch upstream)."""
        self._require_declared(name)
        self._require_per_epoch(name)
        return self._loader.load_epoch(name, epoch, fields=_resolve_fields(fields))

    def stream(
        self, name: str, *, fields: FieldSpec, epochs: list[int] | None = None
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        """Iterate ``(epoch, dict)`` over a per-epoch upstream, one resident.

        Validation and scoping happen eagerly (at call time); iteration loads
        one epoch at a time. ``epochs`` restricts iteration to a specific set
        (e.g. the analysis run's ``inputs.epochs``); default is every available
        epoch of the upstream.
        """
        self._require_declared(name)
        self._require_per_epoch(name)
        resolved = _resolve_fields(fields)
        return self._iter_epochs(name, resolved, epochs)

    def load_stack(
        self, name: str, *, fields: FieldSpec, epochs: list[int] | None = None
    ) -> dict[str, np.ndarray]:
        """Stacked ``(n_epochs, ...)`` arrays for a per-epoch upstream.

        Materializes the whole stack — named so the cost is legible at the
        call site. ``epochs`` restricts the stack to a specific set (e.g. the
        run's ``inputs.epochs``); default is every available epoch. Prefer
        :meth:`stream` when an epoch-by-epoch reduction suffices.
        """
        self._require_declared(name)
        self._require_per_epoch(name)
        return self._loader.load_epochs(name, epochs=epochs, fields=_resolve_fields(fields))

    def load_cross_epoch(self, name: str, *, fields: FieldSpec) -> dict[str, np.ndarray]:
        """The single ``cross_epoch.npz`` of a cross-epoch upstream.

        A *per-epoch* upstream (epoch files but no ``cross_epoch.npz``) raises
        :class:`ArtifactLayoutError`. A wholly-absent upstream falls through to
        ``ArtifactLoader.load_cross_epoch``, which raises ``FileNotFoundError``.
        """
        self._require_declared(name)
        if not self._loader.has_cross_epoch(name) and self._loader.get_epochs(name):
            raise ArtifactLayoutError(
                f"'{name}' has per-epoch artifacts, not a cross_epoch.npz. "
                f"Use stream()/load_stack()."
            )
        return self._loader.load_cross_epoch(name, fields=_resolve_fields(fields))

    def epochs(self, name: str) -> list[int]:
        """Sorted epochs available for a declared upstream (metadata, no load).

        Used by analyzers that must pick a reference epoch (e.g. the latest
        available upstream checkpoint) before loading.
        """
        self._require_declared(name)
        return self._loader.get_epochs(name)

    # ----- internals --------------------------------------------------------

    def _iter_epochs(
        self, name: str, resolved: list[str] | None, epochs: list[int] | None = None
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        for epoch in sorted(epochs) if epochs is not None else self._loader.get_epochs(name):
            yield epoch, self._loader.load_epoch(name, epoch, fields=resolved)

    def _require_declared(self, name: str) -> None:
        if name not in self._allowed:
            raise UndeclaredDependencyError(
                f"Analyzer accessed undeclared dependency '{name}'. Declared: "
                f"{sorted(self._allowed)}. Add ArtifactInput('{name}') to the "
                f"analyzer's Spec.inputs."
            )

    def _require_per_epoch(self, name: str) -> None:
        if not self._loader.get_epochs(name):
            raise ArtifactLayoutError(
                f"'{name}' has no per-epoch artifacts. Use load_cross_epoch() if it "
                f"is a cross-epoch upstream."
            )
