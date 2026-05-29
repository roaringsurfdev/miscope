"""In-memory test doubles for the REQ_128 DepsAccessor.

Analyzer unit tests exercise analyzer *logic*; the real DepsAccessor and the
ArtifactLoader selective-loading/validation path are covered by
``test_deps_accessor.py``. :class:`FakeDeps` serves declared upstream artifacts
from in-memory dicts, honoring the fields contract (``ALL`` -> everything; a
list -> subset; unknown field -> ``ValueError``) without touching disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader
from miscope.analysis.deps import ALL, DepsAccessor, FieldSpec
from miscope.analysis.inputs import ResolvedInputs


def _select(data: dict[str, Any], fields: FieldSpec) -> dict[str, Any]:
    if fields is ALL:
        return dict(data)
    if not isinstance(fields, list):
        raise ValueError("fields must be a list of field names or the ALL sentinel.")
    missing = [f for f in fields if f not in data]
    if missing:
        raise ValueError(f"missing field(s) {missing}; available: {sorted(data)}.")
    return {f: data[f] for f in fields}


class FakeDeps(DepsAccessor):
    """A DepsAccessor that serves artifacts from memory (no loader, no disk)."""

    def __init__(
        self,
        *,
        epoch: int | None = 0,
        epoch_artifacts: dict[str, dict[str, Any]] | None = None,
        series: dict[str, dict[int, dict[str, Any]]] | None = None,
        cross_epoch: dict[str, dict[str, Any]] | None = None,
    ):
        # Deliberately does not call super().__init__ — there is no loader.
        self._pe: dict[str, dict[int, dict[str, Any]]] = {}
        for name, artifact in (epoch_artifacts or {}).items():
            self._pe[name] = {epoch: artifact}  # type: ignore[dict-item]
        for name, by_epoch in (series or {}).items():
            self._pe.setdefault(name, {}).update(by_epoch)
        self._ce: dict[str, dict[str, Any]] = dict(cross_epoch or {})

    def load_epoch(self, name: str, epoch: int, *, fields: FieldSpec) -> dict[str, Any]:
        return _select(self._pe[name][epoch], fields)

    def stream(self, name: str, *, fields: FieldSpec) -> Iterator[tuple[int, dict[str, Any]]]:
        for ep in sorted(self._pe[name]):
            yield ep, _select(self._pe[name][ep], fields)

    def load_stack(self, name: str, *, fields: FieldSpec) -> dict[str, Any]:
        epochs = sorted(self._pe[name])
        selected = [_select(self._pe[name][ep], fields) for ep in epochs]
        stacked: dict[str, Any] = {"epochs": np.array(epochs)}
        for key in selected[0] if selected else []:
            stacked[key] = np.stack([s[key] for s in selected], axis=0)
        return stacked

    def load_cross_epoch(self, name: str, *, fields: FieldSpec) -> dict[str, Any]:
        return _select(self._ce[name], fields)

    def epochs(self, name: str) -> list[int]:
        return sorted(self._pe.get(name, {}))


def deps_inputs(
    epoch_artifacts: dict[str, dict[str, Any]] | None = None,
    *,
    epoch: int | None = 0,
    series: dict[str, dict[int, dict[str, Any]]] | None = None,
    cross_epoch: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> ResolvedInputs:
    """Build a ResolvedInputs whose ``deps`` serves the given artifacts in memory.

    ``epoch_artifacts`` places one dict per upstream at ``epoch`` (the common
    per-epoch case); ``series`` provides ``{name: {epoch: dict}}`` for
    stream/load_stack; ``cross_epoch`` provides ``{name: dict}`` for
    ``load_cross_epoch``.
    """
    deps = FakeDeps(
        epoch=epoch,
        epoch_artifacts=epoch_artifacts,
        series=series,
        cross_epoch=cross_epoch,
    )
    return ResolvedInputs(deps=deps, epoch=epoch, **kwargs)


class PermissiveDeps(DepsAccessor):
    """A real-loader DepsAccessor with scope enforcement disabled.

    For integration-style cross-epoch tests that drive an analyzer against a
    real on-disk store: the loader, fields validation, and layout checks are
    real (high fidelity), but any upstream name is allowed. (Scope enforcement
    is unit-tested in ``test_deps_accessor.py``.)
    """

    def _require_declared(self, name: str) -> None:  # noqa: ARG002 — allow any name
        return


def store_inputs(
    artifacts_dir: str,
    *,
    epochs: tuple[int, ...] | None = None,
    epoch: int | None = None,
    **kwargs: Any,
) -> ResolvedInputs:
    """ResolvedInputs with a REAL (permissive) DepsAccessor over an on-disk store.

    Drop-in for the legacy ``ResolvedInputs(artifacts_dir=..., epochs=...)``:
    builds a real (permissive) ``deps`` over the on-disk store.
    """
    deps = PermissiveDeps(ArtifactLoader(str(artifacts_dir)), frozenset())
    return ResolvedInputs(deps=deps, epochs=epochs, epoch=epoch, **kwargs)
