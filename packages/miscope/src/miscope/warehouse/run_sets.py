"""Run-set registry — the third warehouse index relation (REQ_138).

Alongside ``_catalog`` (110-A columnar) and ``_tensor_catalog`` (110-B), the
``_run_sets`` relation maps a **run set** (a researcher-facing parameterization
handle) to its bindings, the resolved values those bindings took, the per-analyzer
recipe signatures it realized, and provenance. It is the queryable answer to
"which run sets exist for this variant / used binding X?" and the liveness source
for recipe GC (REQ_138 Phase 5).

Decision (OQ #6): persist **named binding columns** (so the warehouse can filter on
``ref_epoch`` / ``probe`` directly) **plus** an opaque ``recipe_signature`` per
realized recipe. Decision (OQ #4): a row records the *resolved value* of each
binding for provenance, while addressing stays by recipe signature.

One row per ``(run_set, analyzer)`` realized recipe: the run set's identity and
label repeat across its analyzers, and each row carries that analyzer's recipe
signature plus the resolved binding values in its recipe. The bindings are stored
as a JSON column (``bindings_json``) — heterogeneous per run set — with a few
common named columns promoted for direct filtering when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from miscope.analysis.parameters import LiteralBinding, Parameterization, binding_key
from miscope.analysis.recipe import Recipe, project_recipe
from miscope.warehouse import paths

if TYPE_CHECKING:
    from miscope.analysis.spec import AnalyzerSpec


@dataclass(frozen=True)
class RunSetRecord:
    """One ``(run_set, analyzer)`` realized-recipe row in the registry."""

    run_set: str  # stable id (recipe-derived) — the comparison/query unit
    label: str | None  # human-facing label, if the analyst supplied one
    variant_id: str
    analyzer: str
    recipe_signature: str  # this analyzer's recipe address (empty -> default plane)
    bindings_json: str  # the run set's bindings + resolved values (provenance)
    created_at: str


def run_set_id(parameterization: Parameterization) -> str:
    """Stable identifier for a run set: its full bindings' recipe signature.

    Recipe-derived (not the label) so the same bindings always address the same run
    set, and a label is free-text provenance. The empty parameterization is the
    default plane.
    """
    if parameterization.is_empty:
        return paths.DEFAULT_RUN_SET
    return Recipe(bindings=tuple(parameterization.bindings)).signature()


def record_run_set(
    variant: object,
    parameterization: Parameterization,
    recipe_map: dict[str, str],
    resolved: dict[str, dict[str, Any]],
    specs_by_name: dict[str, AnalyzerSpec],
) -> None:
    """Persist a run set's realized recipes to the variant's registry (idempotent).

    Args:
        variant: The variant whose registry is updated.
        parameterization: The run set executed.
        recipe_map: ``analyzer -> recipe signature`` realized this run (non-empty
            entries only — the parameterized analyzers).
        resolved: ``analyzer -> {param_name: resolved_value}`` for provenance.
        specs_by_name: Registered specs, for projecting each analyzer's bindings.
    """
    if parameterization.is_empty:
        return  # the default plane needs no registry row
    rs_id = run_set_id(parameterization)
    created = datetime.now(UTC).isoformat()
    rows = [
        RunSetRecord(
            run_set=rs_id,
            label=parameterization.label,
            variant_id=variant.name,  # type: ignore[attr-defined]
            analyzer=analyzer,
            recipe_signature=sig,
            bindings_json=_bindings_json(
                analyzer, parameterization, specs_by_name, resolved.get(analyzer, {})
            ),
            created_at=created,
        )
        for analyzer, sig in sorted(recipe_map.items())
    ]
    if rows:
        _append_run_sets(variant, rows)


def live_recipe_signatures(variant: object) -> set[str]:
    """Recipe signatures referenced by some run set in the registry (REQ_138 GC).

    A recipe directory on disk whose signature is *not* in this set is orphaned —
    no run set vouches for it (e.g. an upstream re-ran and a derived value moved).
    The empty/default plane carries no signature and is never an orphan.
    """
    df = read_run_sets(variant)
    if df.empty:
        return set()
    return {s for s in df["recipe_signature"].tolist() if s}


@dataclass(frozen=True)
class OrphanRecipe:
    """An on-disk recipe directory not referenced by any run set."""

    analyzer: str
    recipe_signature: str
    path: str


def orphaned_recipe_dirs(variant: object) -> list[OrphanRecipe]:
    """On-disk recipe directories with no live run set (candidates for pruning)."""
    from miscope.analysis.artifact_loader import iter_recipe_dirs

    live = live_recipe_signatures(variant)
    artifacts_dir = str(variant.artifacts_dir)  # type: ignore[attr-defined]
    return [
        OrphanRecipe(analyzer=analyzer, recipe_signature=sig, path=path)
        for analyzer, sig, path in iter_recipe_dirs(artifacts_dir)
        if sig not in live
    ]


def read_run_sets(variant: object) -> pd.DataFrame:
    """The variant's run-set registry as a DataFrame (empty frame if none)."""
    path = paths.run_sets_parquet_path(variant)  # type: ignore[arg-type]
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "run_set",
                "label",
                "variant_id",
                "analyzer",
                "recipe_signature",
                "bindings_json",
                "created_at",
            ]
        )
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _bindings_json(
    analyzer: str,
    parameterization: Parameterization,
    specs_by_name: dict[str, AnalyzerSpec],
    resolved: dict[str, Any],
) -> str:
    """JSON of the recipe bindings for one analyzer, with their resolved values."""
    recipe = project_recipe(analyzer, parameterization, specs_by_name)
    entries = []
    for b in recipe.bindings:
        scope_analyzer, name = binding_key(b)
        entry: dict[str, Any] = {
            "analyzer": scope_analyzer,
            "name": name,
            "resolved": _jsonable(resolved.get(name)),
        }
        if isinstance(b, LiteralBinding):
            entry["literal"] = _jsonable(b.value)
        else:
            entry["reference"] = {"source": b.source_analyzer, "selector": repr(b.selector)}
        entries.append(entry)
    return json.dumps(entries, sort_keys=True)


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-serializable form of a resolved value (ints/floats/str)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _append_run_sets(variant: object, rows: list[RunSetRecord]) -> None:
    """Write/merge the run-set Parquet, deduped on (run_set, analyzer)."""
    from dataclasses import asdict

    path = paths.run_sets_parquet_path(variant)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([asdict(r) for r in rows])
    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["run_set", "analyzer"], keep="last", ignore_index=True
        )
    else:
        combined = new
    combined.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
