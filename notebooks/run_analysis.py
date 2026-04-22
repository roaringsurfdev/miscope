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
from miscope.analysis.analyzers import (
    AttentionFourierAnalyzer,
    AttentionFreqAnalyzer,
    AttentionPatternsAnalyzer,
    CentroidDMD,
    DominantFrequenciesAnalyzer,
    EffectiveDimensionalityAnalyzer,
    FourierFrequencyQualityAnalyzer,
    FourierNucleationAnalyzer,
    GlobalCentroidPCA,
    InputTraceAnalyzer,
    InputTraceGraduationAnalyzer,
    IntraGroupManifoldAnalyzer,
    LandscapeFlatnessAnalyzer,
    NeuronActivationsAnalyzer,
    NeuronDynamicsAnalyzer,
    NeuronFourierAnalyzer,
    NeuronFreqClustersAnalyzer,
    NeuronGroupPCAAnalyzer,
    ParameterSnapshotAnalyzer,
    ParameterTrajectoryPCA,
    RepresentationalGeometryAnalyzer,
    TransientFrequencyAnalyzer,
    FreqGroupWeightGeometryAnalyzer,
)

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
FORCE = True  # Re-run even if artifacts exist (needed for new summary keys)
COOLING_NEEDED = False
COOLING_PERIOD = 1 * 20 # timer to allow machine to cool between runs

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

    if (variant.name in exclude_list) or (len(include_list) > 0 and variant.name not in include_list):
        print("  SKIPPED: In exclude list")
        results.append((variant.name, "skipped", 0))
        continue

    start = time.time()

    def progress_callback(pct: float, desc: str) -> None:
        print(f"  [{pct:5.1%}] {desc}", end="\r")

    try:
        pipeline = AnalysisPipeline(variant)
        # pipeline.register(AttentionFreqAnalyzer())
        # pipeline.register(AttentionPatternsAnalyzer())
        # pipeline.register(DominantFrequenciesAnalyzer())
        # pipeline.register(InputTraceAnalyzer())
        # pipeline.register(NeuronActivationsAnalyzer())
        # pipeline.register(NeuronFreqClustersAnalyzer())
        # pipeline.register(ParameterSnapshotAnalyzer())
        # pipeline.register(EffectiveDimensionalityAnalyzer())
        # pipeline.register(LandscapeFlatnessAnalyzer())
        # pipeline.register(RepresentationalGeometryAnalyzer())
        # pipeline.register(AttentionFourierAnalyzer())
        # pipeline.register(FourierNucleationAnalyzer())
        # pipeline.register_secondary(FourierFrequencyQualityAnalyzer())
        # pipeline.register_secondary(NeuronFourierAnalyzer())
        # pipeline.register_cross_epoch(InputTraceGraduationAnalyzer())
        # pipeline.register_cross_epoch(NeuronDynamicsAnalyzer())
        #pipeline.register_cross_epoch(NeuronGroupPCAAnalyzer())
        # pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        # pipeline.register_cross_epoch(GlobalCentroidPCA())
        # pipeline.register_cross_epoch(CentroidDMD())
        # pipeline.register_cross_epoch(TransientFrequencyAnalyzer())
        #pipeline.register_cross_epoch(FreqGroupWeightGeometryAnalyzer())
        pipeline.register_cross_epoch(IntraGroupManifoldAnalyzer())
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

