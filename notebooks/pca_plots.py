# %% imports
from miscope import catalog, load_family

# %% load variant
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=127, seed=485)

#%% plot centroid pca over epochs
variant.view("centroid_pca_variance").show()
variant.view("trajectory_pca_variance").show()

# %%
