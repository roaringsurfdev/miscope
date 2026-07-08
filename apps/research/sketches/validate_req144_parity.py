"""REQ_144 parity check — the variant-summary surfaces agree (read-only).

The 814-line ``VariantAnalysisSummary`` engine is gone; stable-layer byte-parity
against the live engine was gated per-cluster at Stages 2a/3. This check guards the
*surfaces that replaced it* from drifting apart: for each baseline the stable
outcome fields must be value-identical across

  1. ``assemble_variant_summary(v)`` — the per-variant reader (``open_variant``),
  2. the ``variant_registry`` entry — the cross-variant view (``open(family)`` glob),
  3. the materialized ``variant_outcomes`` row read straight from the table,

and the two Python classifications must match between (1) and (2). The cross-variant
glob and per-variant scan are different DuckDB paths (nullable-column promotion,
dtype coercion), so their agreement is the real regression surface. Writes nothing.

The window layer is parity-relaxed (fork e): its presence is reported, not asserted.

Run: ``uv run python apps/research/sketches/validate_req144_parity.py``
"""

from __future__ import annotations

from typing import Any

import miscope
import miscope.query
import miscope.registry as reg
from miscope.analysis.derived_tables import _VARIANT_OUTCOME_COLUMNS
from miscope.analysis.variant_analysis_summary import assemble_variant_registry
from miscope.analysis.variant_summary_assembler import _scalarize, assemble_variant_summary

BASELINES = [(113, 999, 598), (109, 485, 598), (101, 999, 598)]

# The intrinsically-meaningful stable fields are exactly the variant_outcomes columns.
_STABLE_FIELDS = [column for _, column, _ in _VARIANT_OUTCOME_COLUMNS]


def _equalish(a: Any, b: Any) -> bool:
    """Value equality with a REQ_126 float tolerance; lists compared element-wise."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equalish(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(a)))
    return a == b


def check(prime: int, seed: int, dseed: int) -> None:
    fam = miscope.load_family("modulo_addition_1layer")
    v = fam.get_variant(prime=prime, seed=seed, data_seed=dseed)

    assembled = assemble_variant_summary(v)
    entry = next(e for e in assemble_variant_registry(fam) if e["variant_id"] == v.name)
    with miscope.query.open(family=fam) as con:
        table = con.df(f"SELECT * FROM variant_outcomes WHERE variant_id = '{v.name}'").iloc[0]

    for field in _STABLE_FIELDS:
        a, e, t = assembled.get(field), entry.get(field), _scalarize(table[field])
        assert _equalish(a, e), f"{v.name}: {field} assembler={a!r} != registry={e!r}"
        assert _equalish(e, t), f"{v.name}: {field} registry={e!r} != table={t!r}"

    for key in ("failure_mode", "failure_mode_reasons", "performance_classification"):
        assert _equalish(assembled[key], entry[key]), f"{v.name}: {key} differs"

    n_windows = sum(1 for k in assembled if k.endswith("_window"))
    print(
        f"  p{prime}/s{seed}/ds{dseed}: OK — {len(_STABLE_FIELDS)} stable fields agree across "
        f"assembler/registry/table; class={entry['performance_classification'][0]}; "
        f"{n_windows} window dicts present (parity-relaxed)"
    )


if __name__ == "__main__":
    reg.load()  # register the derived tables before querying
    print("REQ_144 parity (read-only) on the three baselines:")
    for p, s, d in BASELINES:
        check(p, s, d)
    print("All baselines parity-clean.")
