"""Researcher-facing parameterization handle over a Variant (REQ_138).

``variant.parameterize(...)`` returns a :class:`ParameterizedVariant` — a thin
proxy that reads its analyzer artifacts through a **recipe-scoped** loader. Views
are universal instruments (architectural invariant 1): the proxy does not change a
view's shape, it only redirects where each view reads its bytes. Because every view
reaches artifacts through ``variant.artifacts`` / ``variant.at(...).view(...)``, a
parameterized read is transparent — the same ``.view(name).figure()`` surface,
pointed at the parameterization's recipe plane.

The empty parameterization resolves every analyzer to today's path, so
``variant.parameterize()`` (no bindings) is byte-for-byte ``variant``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miscope.analysis.artifact_loader import ArtifactLoader
    from miscope.analysis.parameters import Parameterization
    from miscope.families.variant import Variant
    from miscope.views.catalog import BoundView, EpochContext
    from miscope.views.dataview_catalog import BoundDataView


class ParameterizedVariant:
    """A Variant bound to a run set; artifact reads are recipe-scoped.

    Delegates all attribute access to the wrapped variant except the artifact
    accessors and the view entry points, which are redirected through the
    parameterization's per-analyzer recipe map.
    """

    def __init__(self, variant: Variant, parameterization: Parameterization) -> None:
        self._variant = variant
        self._parameterization = parameterization
        self._recipe_map = self._build_recipe_map()

    @property
    def parameterization(self) -> Parameterization:
        return self._parameterization

    @property
    def recipe_map(self) -> dict[str, str]:
        """Per-analyzer recipe signatures realized by this parameterization."""
        return dict(self._recipe_map)

    def _build_recipe_map(self) -> dict[str, str]:
        if self._parameterization.is_empty:
            return {}
        from miscope.analysis.recipe import project_recipe
        from miscope.analysis.registry import AnalyzerRegistry

        specs = {s.name: s for s in AnalyzerRegistry.list_specs()}
        out: dict[str, str] = {}
        for name in specs:
            sig = project_recipe(name, self._parameterization, specs).signature()
            if sig:
                out[name] = sig
        return out

    # --- Recipe-scoped artifact surface -----------------------------------

    @property
    def artifacts(self) -> ArtifactLoader:
        from miscope.analysis.artifact_loader import ArtifactLoader

        return ArtifactLoader(str(self._variant.artifacts_dir), recipe_map=self._recipe_map)

    def get_artifact_loader(self) -> ArtifactLoader:
        return self.artifacts

    # --- View entry points (bind through this proxy) -----------------------

    def at(self, epoch: int | None) -> EpochContext:
        from miscope.views.catalog import EpochContext

        return EpochContext(variant=self, epoch=epoch)  # type: ignore[arg-type]

    def view(self, name: str, **kwargs: Any) -> BoundView:
        return self.at(epoch=None).view(name, **kwargs)

    def dataview(self, name: str) -> BoundDataView:
        return self.at(epoch=None).dataview(name)

    # --- Everything else is the underlying variant -------------------------

    def __getattr__(self, item: str) -> Any:
        # Only reached for attributes not defined on the proxy.
        return getattr(self._variant, item)

    def __repr__(self) -> str:
        label = self._parameterization.label or "unlabeled"
        return f"ParameterizedVariant({self._variant.name!r}, run_set={label!r})"
