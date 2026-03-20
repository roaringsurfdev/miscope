# %% imports
#import sys
#import os

#parent_dir = os.path.dirname(os.path.dirname(__file__))
#sys.path.append(parent_dir)
from pathlib import Path

from typing import Any

import numpy as np

from miscope import load_family, EpochContext
from miscope.families import Variant
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES
from miscope.analysis.variant_analysis_summary import VariantAnalysisSummary
from miscope.analysis.variant_summary import write_variant_summary, build_variant_registry

# %% Test VariantAnalysisSummary
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999, data_seed=598)

analysis_summary = VariantAnalysisSummary(variant)
analysis_summary.analyze()

# %% Create files for all variants
for variant in family.list_variants():
    #if variant.name in ["modulo_addition_1layer_p109_seed485_dseed598"]:
    print(f"creating summary file for {variant.name}")
    analysis_summary = VariantAnalysisSummary(variant)
    analysis_summary.analyze()

results_dir = Path(__file__).parents[1] / "results"
registry_path = build_variant_registry(results_dir, "modulo_addition_1layer")
print(f"Registry written to {registry_path}")

# %%
