# %% imports
from miscope import catalog, load_family, get_config

# %% load variant
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=101, seed=485)

#%% plot centroid pca over epochs
variant.view("centroid_pca_variance").show()
variant.view("explained_variance").show()
# %% 
import os

from miscope.analysis.pipeline import AnalysisPipeline
from miscope.analysis.protocols import AnalysisRunConfig
from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer

cfg = get_config()
registry = FamilyRegistry(str(cfg.model_families_dir), str(cfg.results_dir))
#%%
from miscope.analysis.pipeline import AnalysisPipeline
from miscope.analysis.protocols import AnalysisRunConfig
from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer
for variant in family.list_variants():
    print(f"Regenerating summary for {variant.name}...")
    pipeline = AnalysisPipeline(variant, AnalysisRunConfig(analyzers=["repr_geometry"]))
    pipeline.register(RepresentationalGeometryAnalyzer())
    pipeline.run(force=True)
#%%
variants_remaining = [
]
for variant in family.list_variants():
    if variant.name in variants_remaining:
        print(f"Testing summary for {variant.name}...")
        pipeline = AnalysisPipeline(variant, AnalysisRunConfig(analyzers=["repr_geometry"]))
        pipeline.register(RepresentationalGeometryAnalyzer())
        pipeline.run(force=True)

# %%
