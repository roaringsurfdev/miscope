"""REQ_145 incremental-refresh validation against the three baselines.

Two modes:

``preview`` (default, **read-only**, fast) — for each baseline, print the
signature-aware plan. On not-yet-stamped baselines this lists the one-time rebuild
scope (everything stale because the artifacts predate signatures). After a real
signature-stamped run it should print "no work" — the no-change no-op.

``selectivity`` (**read-only**, fast) — prove the predicate is selective on the real
family DAG without running analysis or touching disk: stamp each analyzer's
*current* would-signatures into an in-memory manifest overlay, confirm the plan is
then empty (a no-change re-run is a no-op), bump ONE producer's version, and confirm
exactly that producer and its transitive dependents restage while nothing else does.

The heavy acceptance bar — byte-parity of a signature-aware rebuild vs. a ``force``
rebuild, and the measured selectivity wall-clock — is a real ~40-min×3 run; drive it
with ``scripts/run_analysis.py`` (FORCE=True then FORCE=False) and compare, or the
``--warehouse-parity`` mode below for the warehouse half only.

Run: ``uv run python apps/research/sketches/validate_req145_refresh.py [preview|selectivity]``
"""

from __future__ import annotations

import sys
from dataclasses import replace

import miscope
from miscope.analysis import plan_analysis
from miscope.analysis.registry import AnalyzerRegistry
from miscope.analysis.signature import SigRecord

BASELINES = [(113, 999, 598), (109, 485, 598), (101, 999, 598)]


def _variant(prime: int, seed: int, dseed: int):
    fam = miscope.load_family("modulo_addition_1layer")
    return fam, fam.get_variant(prime=prime, seed=seed, data_seed=dseed)


def preview() -> None:
    """Read-only: print the signature-aware plan for each baseline."""
    print("REQ_145 plan preview (read-only) on the three baselines:\n")
    for prime, seed, dseed in BASELINES:
        fam, v = _variant(prime, seed, dseed)
        specs = AnalyzerRegistry.list_for_family(fam)
        plan = plan_analysis(v, specs)
        per = sum(len(it.epochs) for it in plan.per_epoch)
        print(
            f"p{prime}/s{seed}/ds{dseed}: {len(plan.per_epoch)} per-epoch analyzers "
            f"({per} epoch-units), {len(plan.cross_epoch)} cross-epoch "
            f"{'— NO WORK (no-op)' if plan.is_empty else 'stale'}"
        )


def _stamped_overlay(v, specs) -> dict[str, dict[str, SigRecord]]:
    """Build the would-signature manifest a real run of ``specs`` would stamp.

    Returns ``{analyzer -> {epoch_key -> SigRecord}}`` from a single plan pass —
    the planner already computes every planned node's post-run signature.
    """
    plan = plan_analysis(v, specs, force=True)  # force → signatures for every node
    overlay: dict[str, dict[str, SigRecord]] = {}
    for name, sigs in plan.signatures.items():
        overlay[name] = {k: SigRecord.from_json(r) for k, r in sigs.items()}  # type: ignore[misc]
    return overlay


def _plan_with_overlay(v, specs, overlay):
    """Plan with ``read_signature_manifest`` patched to serve the in-memory overlay."""
    import miscope.analysis.planner as planner_mod

    def fake_read(artifacts_dir, analyzer, recipe_sig=""):  # noqa: ARG001
        return {k: r.to_json() for k, r in overlay.get(analyzer, {}).items()}

    original = planner_mod.read_signature_manifest
    planner_mod.read_signature_manifest = fake_read  # type: ignore[assignment]
    try:
        return plan_analysis(v, specs)
    finally:
        planner_mod.read_signature_manifest = original  # type: ignore[assignment]


def selectivity() -> None:
    """Read-only: prove no-op + single-analyzer selectivity on the real DAG."""
    print("REQ_145 selectivity (read-only, in-memory overlay) on the three baselines:\n")
    for prime, seed, dseed in BASELINES:
        fam, v = _variant(prime, seed, dseed)
        specs = list(AnalyzerRegistry.list_for_family(fam))
        overlay = _stamped_overlay(v, specs)

        # 1. With current signatures stamped, a re-plan is a no-op.
        noop = _plan_with_overlay(v, specs, overlay)
        assert noop.is_empty, f"expected no-op after stamping, got {noop.format()}"

        # 2. Bump one producer's version; it + its transitive dependents restage,
        #    and nothing else. Target the documented multi-hop chain root when present
        #    (activation_frequency_norm -> neuron_frequency_attribution ->
        #    neuron_dynamics) so forward propagation is visible.
        names = {s.name for s in specs}
        root = "activation_frequency_norm"
        target = root if root in names else next(s.name for s in specs if s.requires)
        bumped = [replace(s, version=s.version + 1) if s.name == target else s for s in specs]
        plan = _plan_with_overlay(v, bumped, overlay)
        restaged = {it.analyzer_name for it in (*plan.per_epoch, *plan.cross_epoch)}
        expected = _transitive_dependents(specs, target) | {target}
        assert restaged == expected, (
            f"selectivity mismatch for {target!r}: restaged {sorted(restaged)} "
            f"!= expected {sorted(expected)}"
        )
        print(
            f"  p{prime}/s{seed}/ds{dseed}: no-op clean; bumping {target!r} "
            f"restages exactly {sorted(restaged)}"
        )
    print("\nSelectivity OK on all baselines.")


def _transitive_dependents(specs, target: str) -> set[str]:
    """Names reachable from ``target`` via the reverse of the ``requires`` DAG."""
    children: dict[str, set[str]] = {}
    for s in specs:
        for up in s.requires:
            children.setdefault(up, set()).add(s.name)
    out: set[str] = set()
    frontier = [target]
    while frontier:
        cur = frontier.pop()
        for dep in children.get(cur, set()):
            if dep not in out:
                out.add(dep)
                frontier.append(dep)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        preview()
    elif mode == "selectivity":
        selectivity()
    else:
        print(f"unknown mode {mode!r}; use 'preview' or 'selectivity'")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
