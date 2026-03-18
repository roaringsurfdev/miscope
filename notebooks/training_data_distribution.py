# %% imports
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope import load_family

# %% Configuration
# data_seed=598 = known good (grokks cleanly for p=113)
# data_seeds=42, 999 = known bad (higher test loss, rebounding behavior)
PRIME = 113
MODEL_SEED = 999
GOOD_DATA_SEED = 598
BAD_DATA_SEEDS = [42, 999]
ALL_DATA_SEEDS = [GOOD_DATA_SEED] + BAD_DATA_SEEDS

family = load_family("modulo_addition_1layer")
p = PRIME

COLORS = {GOOD_DATA_SEED: 'steelblue'}
for ds in BAD_DATA_SEEDS:
    COLORS[ds] = 'tomato'


def load_split(data_seed):
    v = family.get_variant(prime=PRIME, seed=MODEL_SEED, data_seed=data_seed)
    td, tl, _, _, _, _ = v.generate_training_dataset()
    return td, tl


splits = {ds: load_split(ds) for ds in ALL_DATA_SEEDS}

# %% --- VIEW 1: Training count per residue — good vs bad seeds ---
# The simplest question: are some residues getting fewer training pairs in the bad seeds?
# Expected variance: Binomial(p, 0.3) -> std ~4.9 for p=113.
# Anything outside ~3 sigma (~15 below mean) would be notable.

fig_counts = make_subplots(
    rows=1, cols=len(ALL_DATA_SEEDS),
    subplot_titles=[f"data_seed={ds}" for ds in ALL_DATA_SEEDS],
    shared_yaxes=True
)
expected = p * 0.3
for col_idx, ds in enumerate(ALL_DATA_SEEDS):
    _, tl = splits[ds]
    counts = pd.Series(tl.numpy()).value_counts().sort_index().reindex(range(p), fill_value=0)
    fig_counts.add_trace(
        go.Bar(x=counts.index.tolist(), y=counts.values.tolist(),
               marker_color=COLORS[ds], showlegend=False),
        row=1, col=col_idx + 1
    )
    fig_counts.add_hline(y=expected, line_dash='dash', line_color='black',
                         row=1, col=col_idx + 1)

fig_counts.update_layout(
    title=f"Training pairs per residue — p={p} (blue=good, red=bad)",
    height=350
)
fig_counts.show()

# %% --- VIEW 2: Fourier imbalance of training a-values ---
# The model learns (a+b) mod p via key Fourier frequencies.
# For each frequency k, the gradient signal quality depends on how uniformly
# training a-values are distributed around the k-th frequency circle.
#
# Fourier imbalance at frequency k: |Σ_a e^(2πika/p)| / n_train
# Near 0 = uniform coverage (good). Large = gappy/biased coverage (bad).
#
# If the bad seeds consistently show high imbalance at the same frequencies as
# the model's key frequencies, that's the mechanism.

def fourier_imbalance(a_values, p):
    """DFT magnitude of the training indicator at each frequency, normalized by n."""
    indicator = np.zeros(p)
    for a in a_values:
        indicator[int(a)] += 1
    dft = np.fft.rfft(indicator)
    return np.abs(dft[1:p // 2 + 1]) / len(a_values)


fig_fourier = make_subplots(
    rows=1, cols=len(ALL_DATA_SEEDS),
    subplot_titles=[f"data_seed={ds}" for ds in ALL_DATA_SEEDS],
    shared_yaxes=True
)
imbalances = {}
for col_idx, ds in enumerate(ALL_DATA_SEEDS):
    td, _ = splits[ds]
    imb = fourier_imbalance(td[:, 0].numpy(), p)
    imbalances[ds] = imb
    fig_fourier.add_trace(
        go.Bar(x=list(range(1, len(imb) + 1)), y=imb.tolist(),
               marker_color=COLORS[ds], showlegend=False),
        row=1, col=col_idx + 1
    )

fig_fourier.update_layout(
    title=f"Fourier imbalance of training a-values per frequency — p={p} (lower = better)",
    height=350
)
fig_fourier.show()

# %% --- VIEW 3: Imbalance difference (bad − good) ---
# Highlights which specific frequencies are worse in the bad seeds.
# If a peak here coincides with the known key frequencies for p=113,
# that's a direct link between data geometry and learning failure.
#
# Key frequencies for p=113 can be identified from the model weights (Fourier analysis
# of W_E). Without that, look for consistent peaks across both bad seeds.

fig_diff = go.Figure()
freqs = list(range(1, len(imbalances[GOOD_DATA_SEED]) + 1))
for ds in BAD_DATA_SEEDS:
    diff = (imbalances[ds] - imbalances[GOOD_DATA_SEED]).tolist()
    fig_diff.add_trace(go.Bar(
        x=freqs, y=diff,
        name=f"seed {ds} − {GOOD_DATA_SEED}",
        marker_color=COLORS[ds],
        opacity=0.7
    ))
fig_diff.add_hline(y=0, line_color='black', line_width=1)
fig_diff.update_layout(
    title=f"Fourier imbalance difference (bad − good) — p={p}",
    xaxis_title="Frequency k",
    yaxis_title="Imbalance difference",
    barmode='overlay'
)
fig_diff.show()

# %% --- VIEW 4: Per-residue max Fourier imbalance ---
# Break down the imbalance per residue (per anti-diagonal).
# Each residue's training pairs lie on a+b≡r (mod p); parameterize by a.
# A residue with high max imbalance gets poor gradient signal for key frequencies.
# Look for residues that are consistently worse in bad seeds.

def per_residue_max_imbalance(td, tl, p):
    max_imb = np.zeros(p)
    for r in range(p):
        mask = tl == r
        if mask.sum() < 2:
            max_imb[r] = 1.0
            continue
        a_vals = td[mask, 0].numpy()
        max_imb[r] = fourier_imbalance(a_vals, p).max()
    return max_imb


fig_residue = go.Figure()
for ds in ALL_DATA_SEEDS:
    td, tl = splits[ds]
    max_imb = per_residue_max_imbalance(td, tl, p)
    fig_residue.add_trace(go.Scatter(
        x=list(range(p)), y=max_imb.tolist(),
        mode='markers',
        name=f"data_seed={ds}",
        marker=dict(color=COLORS[ds], size=5, opacity=0.7)
    ))
fig_residue.update_layout(
    title=f"Max Fourier imbalance per residue — p={p}",
    xaxis_title="Residue",
    yaxis_title="Max imbalance across frequencies",
    height=400
)
fig_residue.show()

# %% --- VIEW 5: Training set overlap and unique-pair residue distribution ---
# How different are the training sets? Then: where do the differences land?
# Pairs unique to the good seed = coverage that bad seeds are missing.
# If these are non-uniformly distributed across residues, some residues lose
# more coverage in bad seeds than others.

good_set = set(zip(
    splits[GOOD_DATA_SEED][0][:, 0].numpy().astype(int),
    splits[GOOD_DATA_SEED][0][:, 1].numpy().astype(int)
))
n_train = len(good_set)

print(f"Training set size: {n_train} pairs ({100 * n_train / p**2:.1f}% of {p}²={p**2})")
print()
for ds in BAD_DATA_SEEDS:
    bad_set = set(zip(
        splits[ds][0][:, 0].numpy().astype(int),
        splits[ds][0][:, 1].numpy().astype(int)
    ))
    overlap = len(good_set & bad_set)
    print(f"data_seed={ds} vs {GOOD_DATA_SEED}: "
          f"overlap={overlap}/{n_train} ({100*overlap/n_train:.1f}%), "
          f"unique to good={len(good_set - bad_set)}, "
          f"unique to bad={len(bad_set - good_set)}")

# %% --- VIEW 6: Residue distribution of pairs unique to the good seed ---
# For each bad seed, which residues are losing coverage relative to the good seed?
# Uniform distribution = random sampling noise. Skewed distribution = structural bias.

fig_unique = go.Figure()
for ds in BAD_DATA_SEEDS:
    bad_set = set(zip(
        splits[ds][0][:, 0].numpy().astype(int),
        splits[ds][0][:, 1].numpy().astype(int)
    ))
    good_only = [(a + b) % p for a, b in (good_set - bad_set)]
    counts = pd.Series(good_only).value_counts().sort_index().reindex(range(p), fill_value=0)
    fig_unique.add_trace(go.Bar(
        x=counts.index.tolist(), y=counts.values.tolist(),
        name=f"Missing from seed {ds}",
        marker_color=COLORS[ds],
        opacity=0.7
    ))
fig_unique.update_layout(
    title=f"Residues losing coverage in bad seeds (pairs present in seed={GOOD_DATA_SEED} but absent) — p={p}",
    xaxis_title="Residue",
    yaxis_title="Pairs missing from bad seed",
    barmode='overlay'
)
fig_unique.show()

# %%
