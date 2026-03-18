# %% imports
#import sys
#import os

#import numpy as np

#parent_dir = os.path.dirname(os.path.dirname(__file__))
#sys.path.append(parent_dir)

from miscope import load_family, EpochContext
from miscope.analysis.variant_summary import write_variant_summary
# %% load model and list variants
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999, data_seed=598)
for v in family.list_variants():
    #print(f"Creating variant summary for {v.name}")
    path1 = write_variant_summary(v, 0.10)

# %% build registry
from pathlib import Path
from miscope.analysis.variant_summary import build_variant_registry

results_dir = Path(__file__).parents[1] / "results"
registry_path = build_variant_registry(results_dir, "modulo_addition_1layer")
print(f"Registry written to {registry_path}")

# %%
