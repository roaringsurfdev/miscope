# %% imports
from miscope import catalog, load_family

# %% Inspect all available views
print(f"{len(catalog.names())} views registered:")
for name in catalog.names():
    print(f"  {name}")

# %% Load family, define views, set epochs
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=107, seed=999)
views = []
epochs = []

# %% [markdown]
# ## Circularity Crossover Analysis
#
# Detect the epochs where Attention Circularity rises above and falls below
# MLP/Resid Circularity — marking the start and end of the "handoff" period
# where Attention takes over maintaining circular structure during grokking.

# %% Detect circularity crossover epochs
from miscope.analysis.library import find_circularity_crossovers
from miscope.visualization.renderers.loss_curves import render_loss_curves_with_indicator

summary = variant.artifacts.load_summary("repr_geometry")
crossovers = find_circularity_crossovers(summary, reference_sites=["resid_post"])

print("All crossover events (sorted by epoch):")
for evt in crossovers["events"]:
    print(f"  epoch={evt['epoch']:>7}  {evt['direction']:<4}  [{evt['site']}]")
print()
print("Per-site breakdown:")
for site, site_events in crossovers["per_site"].items():
    if site_events:
        summary_str = ", ".join(f"{e['direction']}@{e['epoch']}" for e in site_events)
        print(f"  {site}: {summary_str}")
    else:
        print(f"  {site}: (no crossovers)")

# %% Plot all crossover epochs on loss curve
_DIRECTION_COLORS = {"rise": "purple", "fall": "darkorange"}
meta = variant.metadata
event_epochs = [
    (evt["epoch"], f"{evt['direction']} ({evt['site']})", _DIRECTION_COLORS[evt["direction"]])
    for evt in crossovers["events"]
]

fig = render_loss_curves_with_indicator(
    meta["train_losses"],
    meta["test_losses"],
    current_epoch=0,
    checkpoint_epochs=meta.get("checkpoint_epochs"),
    event_epochs=event_epochs,
    title=f"Loss Curve — All Circularity Crossovers ({variant.params})",
)
fig.show()
# %%
