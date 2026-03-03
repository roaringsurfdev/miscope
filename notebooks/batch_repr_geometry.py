# %% Batch Repr Geometry Analysis
# Runs the RepresentationalGeometryAnalyzer across all variants in a family.
# The pipeline skips already-completed epochs per analyzer, so re-running
# is safe and only computes missing data (e.g., new summary keys from REQ_045).
#
# Usage: Run all cells, or run from the command line:
#   python notebooks/batch_repr_geometry.py

# %% imports
import sys
import os
import time

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family
from miscope.analysis import AnalysisPipeline
from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
FORCE = True  # Re-run even if artifacts exist (needed for new summary keys)

# %% discover variants
family = load_family(FAMILY_NAME)
variants = family.list_variants()
print(f"Family: {FAMILY_NAME}")
print(f"Variants: {len(variants)}")
for v in variants:
    print(f"  {v.name} [{v.state.value}]")

# %% run analysis
cooling_off_period = 60 * 4 # timer to allow machine to cool between runs
results = []
exclude_list = ["modulo_addition_1layer_p101_seed999"]
include_list = [
    "modulo_addition_1layer_p113_seed999",
    "modulo_addition_1layer_p59_seed485"
    ]
for i, variant in enumerate(variants):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(variants)}] {variant.name}")
    print(f"{'='*60}")

    if not variant._has_checkpoints():
        print("  SKIPPED: No checkpoints")
        results.append((variant.name, "skipped", 0))
        continue

    if variant.name not in include_list:
        print("  SKIPPED: Not in include list")
        results.append((variant.name, "skipped", 0))
        continue

    start = time.time()

    def progress_callback(pct: float, desc: str) -> None:
        print(f"  [{pct:5.1%}] {desc}", end="\r")

    try:
        pipeline = AnalysisPipeline(variant)
        pipeline.register(RepresentationalGeometryAnalyzer())
        pipeline.run(force=FORCE, progress_callback=progress_callback)
        elapsed = time.time() - start
        print(f"\n  DONE in {elapsed:.1f}s")
        results.append((variant.name, "success", elapsed))
        print("  COOLING OFF: Entering cooling off period.")
        time.sleep(cooling_off_period)

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  FAILED after {elapsed:.1f}s: {e}")
        results.append((variant.name, "failed", elapsed))

# %% summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
total_time = sum(r[2] for r in results)
for name, status, elapsed in results:
    print(f"  {status:>8s}  {elapsed:6.1f}s  {name}")
print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)")
