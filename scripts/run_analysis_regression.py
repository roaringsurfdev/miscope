# %% Neuron Fourier Analysis
# Runs the NeuronFourierAnalyzer across all variants in a family.
# The pipeline skips already-completed epochs per analyzer, so re-running
# is safe and only computes missing data.
#
# Usage: Run all cells, or run from the command line:
#   python scripts/run_analysis_regression.py

# %% imports
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

# Importing analyzers package triggers @register_analyzer decorators.
import miscope.analysis.analyzers  # noqa: E402, F401
from miscope import load_family  # noqa: E402
from miscope.analysis import AnalysisPipeline, plan_analysis  # noqa: E402
from miscope.analysis.registry import AnalyzerRegistry  # noqa: E402

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
FORCE = True  # Re-run even if artifacts exist (needed for new summary keys)
COOLING_NEEDED = False
COOLING_PERIOD = 1 * 20  # timer to allow machine to cool between runs

# Regression-specific analyzer selection. LandscapeFlatnessAnalyzer and
# FourierNucleationAnalyzer were excluded from regression in the legacy
# script (the former is stochastic; the latter is initialization-only).
EXCLUDE_FROM_REGRESSION = {"landscape_flatness", "fourier_nucleation"}

# %% discover variants
family = load_family(FAMILY_NAME)
variants = family.list_variants()
print(f"Family: {FAMILY_NAME}")
print(f"Variants: {len(variants)}")
for v in variants:
    print(f"  {v.name} [{v.state.value}]")

# %% run analysis
results = []
exclude_list = []
# include_list = ['modulo_addition_1layer_p113_seed999_dseed598']
include_list = ["modulo_addition_1layer_p109_seed485_dseed598"]
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
        # REQ_120: enumerate Specs from the Registry, filtered against the
        # regression-specific exclude list. The pipeline instantiates the
        # corresponding analyzers from the Registry at execute time.
        specs = [
            s
            for s in AnalyzerRegistry.list_for_family(variant.family)
            if s.name not in EXCLUDE_FROM_REGRESSION
        ]
        plan = plan_analysis(variant, specs, force=FORCE)
        print(plan.format())

        pipeline = AnalysisPipeline(variant)
        pipeline.run(force=FORCE, progress_callback=progress_callback, plan=plan)
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

# %% summary
print(f"\n{'=' * 60}")
print("Summary")
print(f"{'=' * 60}")
total_time = sum(r[2] for r in results)
for name, status, elapsed in results:
    print(f"  {status:>8s}  {elapsed:6.1f}s  {name}")
print(f"\nTotal: {total_time:.0f}s ({total_time / 60:.1f} min)")
