# %% Neuron Fourier Analysis
# Runs the NeuronFourierAnalyzer across all variants in a family.
# The pipeline skips already-completed epochs per analyzer, so re-running
# is safe and only computes missing data.
#
# Usage: Run all cells, or run from the command line:
#   python scripts/run_analysis.py

# %% imports
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

# Importing analyzers package triggers @register_analyzer decorators
# under analyzers/__init__.py — needed before AnalyzerRegistry queries.
import miscope.analysis.analyzers  # noqa: E402, F401
from miscope import load_family  # noqa: E402
from miscope.analysis import AnalysisPipeline, plan_analysis  # noqa: E402
from miscope.analysis.registry import AnalyzerRegistry  # noqa: E402
from miscope.analysis.variant_analysis_summary import (  # noqa: E402
    build_variant_registry,
    write_variant_summary,
)
from miscope.warehouse import (  # noqa: E402
    materialize_variant_columnar,
    materialize_variant_derived,
)

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
# REQ_145: signature-aware by default — only stale (changed code/recipe/checkpoint
# /upstream) work runs. Set FORCE=True for the explicit "rebuild everything" override.
FORCE = False
COOLING_NEEDED = False
COOLING_PERIOD = 1 * 20  # timer to allow machine to cool between runs

# %% analyzer selection
# REQ_120: pick Specs from the Registry by name. Defaults to every analyzer
# the family declares; set ANALYZER_NAMES to restrict to a subset.
ANALYZER_NAMES: list[str] | None = None
# Example: ANALYZER_NAMES = ["intragroup_manifold"]

# %% discover variants
family = load_family(FAMILY_NAME)
variants = family.variants
print(f"Family: {FAMILY_NAME}")
print(f"Variants: {len(variants)}")
for v in variants:
    print(f"  {v.name} [{v.state.value}]")

# %% run analysis
results = []
exclude_list = []
include_list = ["p113_seed999_dseed598"]
for i, variant in enumerate(variants):
    print(f"\n{'=' * 60}")
    print(f"[{i + 1}/{len(variants)}] {variant.name}")
    print(f"{'=' * 60}")

    if not variant._has_checkpoints():
        print("  SKIPPED: No checkpoints")
        results.append((variant.name, "skipped", 0))
        continue

    if (variant.name in exclude_list) or (
        len(include_list) > 0 and variant.name not in include_list
    ):
        print("  SKIPPED: In exclude list")
        results.append((variant.name, "skipped", 0))
        continue

    start = time.time()

    def progress_callback(pct: float, desc: str) -> None:
        print(f"  [{pct:5.1%}] {desc}", end="\r")

    try:
        # REQ_120: enumerate Specs from the Registry — no hand-coded
        # register(...) block, no per-category dispatch. The pipeline
        # instantiates Spec-only items from the Registry at execute time.
        if ANALYZER_NAMES is None:
            specs = AnalyzerRegistry.list_for_family(variant.family)
        else:
            specs = [AnalyzerRegistry.get_spec(n) for n in ANALYZER_NAMES]

        plan = plan_analysis(variant, specs, force=FORCE)
        print(plan.format())

        pipeline = AnalysisPipeline(variant)
        pipeline.run(force=FORCE, progress_callback=progress_callback, plan=plan)

        # REQ_145 parity with the dashboard path: analyze -> materialize -> summarize
        # as one signature-aware flow. Surgical by default (force propagates the
        # "rebuild everything" override) — only stale tables/summaries are rebuilt.
        print("\n  Materializing warehouse (columnar + derived)...")
        materialize_variant_columnar(variant, force=FORCE)
        materialize_variant_derived(variant, force=FORCE)
        write_variant_summary(variant)

        elapsed = time.time() - start
        print(f"\n  DONE in {elapsed:.1f}s")
        results.append((variant.name, "success", elapsed))
        if COOLING_NEEDED:
            print("  COOLING OFF: Entering cooling off period.")
            time.sleep(COOLING_PERIOD)

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.1f}s: {e}")
        results.append((variant.name, "failed", elapsed))

# %% compile cross-variant registry (parity with the dashboard's post-run step)
build_variant_registry(family)

# %% summary
print(f"\n{'=' * 60}")
print("Summary")
print(f"{'=' * 60}")
total_time = sum(r[2] for r in results)
for name, status, elapsed in results:
    print(f"  {status:>8s}  {elapsed:6.1f}s  {name}")
print(f"\nTotal: {total_time:.0f}s ({total_time / 60:.1f} min)")
