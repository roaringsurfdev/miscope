# %% Neuron Fourier Analysis
# Runs the NeuronFourierAnalyzer across all variants in a family.
# The pipeline skips already-completed epochs per analyzer, so re-running
# is safe and only computes missing data.
#
# Usage: Run all cells, or run from the command line:
#   python notebooks/run_analysis.py

# %% imports
import sys
import os
import time

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family
from miscope.analysis import AnalysisPipeline
#from miscope.analysis.analyzers.gradient_site import GradientSiteAnalyzer
from miscope.analysis.analyzers.input_trace import InputTraceAnalyzer
from miscope.analysis.analyzers.input_trace_graduation import InputTraceGraduationAnalyzer
#from miscope.analysis.analyzers.attention_fourier import AttentionFourierAnalyzer
#from miscope.analysis.analyzers.parameter_snapshot import ParameterSnapshotAnalyzer
#from miscope.analysis.analyzers.neuron_fourier import NeuronFourierAnalyzer
#from miscope.analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
#from miscope.analysis.analyzers.fourier_frequency_quality import FourierFrequencyQualityAnalyzer

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
FORCE = True  # Re-run even if artifacts exist (needed for new summary keys)
COOLING_NEEDED = False
COOLING_PERIOD = 4 * 60 # timer to allow machine to cool between runs

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
include_list = []
for i, variant in enumerate(variants):
    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(variants)}] {variant.name}")
    print(f"{'='*60}")

    if not variant._has_checkpoints():
        print("  SKIPPED: No checkpoints")
        results.append((variant.name, "skipped", 0))
        continue

    if variant.name in exclude_list:
        print("  SKIPPED: In exclude list")
        results.append((variant.name, "skipped", 0))
        continue

    start = time.time()

    def progress_callback(pct: float, desc: str) -> None:
        print(f"  [{pct:5.1%}] {desc}", end="\r")

    try:
        pipeline = AnalysisPipeline(variant)
        pipeline.register(InputTraceAnalyzer())
        pipeline.register_cross_epoch(InputTraceGraduationAnalyzer())
        #pipeline.register_secondary(FourierFrequencyQualityAnalyzer())
        pipeline.run(force=FORCE, progress_callback=progress_callback)
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
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
total_time = sum(r[2] for r in results)
for name, status, elapsed in results:
    print(f"  {status:>8s}  {elapsed:6.1f}s  {name}")
print(f"\nTotal: {total_time:.0f}s ({total_time/60:.1f} min)")

