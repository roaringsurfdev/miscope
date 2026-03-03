# %% imports
import sys
import os

import numpy as np
import plotly.graph_objects as go

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family
from miscope.analysis.artifact_loader import ArtifactLoader

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
PRIME = 59
SEED = 485

# Threshold: neurons with max frac_explained below this are "uncommitted"
# Uniform baseline for p=101 is 1/50 = 0.02, so 3x that ≈ 0.06
UNCOMMITTED_THRESHOLD = 0.06

family = load_family(FAMILY_NAME)
variant = family.get_variant(prime=PRIME, seed=SEED)
loader = ArtifactLoader(variant.artifacts_dir)

# %% load all epochs of neuron_freq_norm
stacked = loader.load_epochs("neuron_freq_norm")
epochs = stacked["epochs"]  # (n_epochs,)
norm_matrix = stacked["norm_matrix"]  # (n_epochs, n_freq, d_mlp)
n_epochs, n_freq, d_mlp = norm_matrix.shape
print(f"Loaded: {n_epochs} epochs, {n_freq} frequencies, {d_mlp} neurons")
print(f"Epoch range: {int(epochs[0])} - {int(epochs[-1])}")

# %% compute per-neuron dominant frequency and frac_explained at each epoch
# dominant_freq: (n_epochs, d_mlp) — the frequency index with max frac_explained
# max_frac: (n_epochs, d_mlp) — the max frac_explained value
dominant_freq = np.argmax(norm_matrix, axis=1)  # (n_epochs, d_mlp)
max_frac = np.max(norm_matrix, axis=1)  # (n_epochs, d_mlp)

# Mask uncommitted neurons (set to -1 so they render differently)
dominant_freq_masked = dominant_freq.astype(float)
dominant_freq_masked[max_frac < UNCOMMITTED_THRESHOLD] = np.nan

print(f"At final epoch: {np.sum(max_frac[-1] >= UNCOMMITTED_THRESHOLD)} committed neurons")
print(f"At epoch 0: {np.sum(max_frac[0] >= UNCOMMITTED_THRESHOLD)} committed neurons")

# %% render: dominant frequency over time heatmap
def render_neuron_freq_trajectory(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    epochs: np.ndarray,
    threshold: float = 0.06,
    title: str | None = None,
    height: int = 600,
    width: int = 1100,
) -> go.Figure:
    """Neuron dominant frequency over training.

    Y-axis: neuron index, X-axis: epoch, color: dominant frequency.
    Uncommitted neurons (below threshold) shown as grey.
    """
    n_epochs, d_mlp = dominant_freq.shape
    n_freq = int(np.nanmax(dominant_freq)) + 1

    # Build display matrix: dominant freq for committed, NaN for uncommitted
    display = dominant_freq.astype(float).copy()
    display[max_frac < threshold] = np.nan

    # Transpose so neurons are on y-axis: (d_mlp, n_epochs)
    display_T = display.T

    # Custom colorscale: use HSL hue rotation for frequency discrimination
    # We want visually distinct colors for different frequencies
    import colorsys
    n_colors = n_freq
    colorscale_values = []
    for i in range(n_colors):
        hue = i / n_colors
        r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
        colorscale_values.append(
            [i / (n_colors - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
        )

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=display_T,
            x=epochs,
            y=list(range(d_mlp)),
            colorscale=colorscale_values,
            zmin=0,
            zmax=n_freq - 1,
            colorbar=dict(
                title=dict(text="Dominant<br>Freq", side="right"),
                thickness=15,
                len=0.9,
            ),
            hovertemplate=(
                "Epoch %{x}<br>"
                "Neuron %{y}<br>"
                "Dominant Freq: %{z:.0f}<br>"
                "<extra></extra>"
            ),
            # NaN renders as transparent (shows background)
        )
    )

    if title is None:
        title = f"Neuron Dominant Frequency Over Training (p={PRIME}, seed={SEED})"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Neuron Index",
        height=height,
        width=width,
        template="plotly_white",
        margin=dict(l=60, r=80, t=50, b=50),
        # Grey background so NaN (uncommitted) neurons show as grey
        plot_bgcolor="rgb(220,220,220)",
    )

    return fig


fig = render_neuron_freq_trajectory(
    dominant_freq, max_frac, epochs, threshold=UNCOMMITTED_THRESHOLD
)
fig.show()

# %% frequency stability analysis: how often does each neuron switch its dominant freq?
def compute_freq_switches(dominant_freq: np.ndarray, max_frac: np.ndarray, threshold: float = 0.06):
    """Count how many times each neuron changes its dominant frequency.

    Only counts transitions between committed states (above threshold).
    Returns switch_count per neuron, and a matrix of transition epochs.
    """
    n_epochs, d_mlp = dominant_freq.shape
    switch_counts = np.zeros(d_mlp, dtype=int)
    # Track last committed frequency per neuron
    last_freq = np.full(d_mlp, -1, dtype=int)

    switch_epochs = [[] for _ in range(d_mlp)]

    for t in range(n_epochs):
        for n in range(d_mlp):
            if max_frac[t, n] >= threshold:
                current = dominant_freq[t, n]
                if last_freq[n] >= 0 and current != last_freq[n]:
                    switch_counts[n] += 1
                    switch_epochs[n].append(int(epochs[t]))
                last_freq[n] = current

    return switch_counts, switch_epochs


switch_counts, switch_epochs = compute_freq_switches(
    dominant_freq, max_frac, threshold=UNCOMMITTED_THRESHOLD
)
print(f"Neurons that never switch: {np.sum(switch_counts == 0)}")
print(f"Neurons that switch 1+ times: {np.sum(switch_counts >= 1)}")
print(f"Neurons that switch 3+ times: {np.sum(switch_counts >= 3)}")
print(f"Max switches: {np.max(switch_counts)}")

# Show top thrashers
top_thrashers = np.argsort(switch_counts)[::-1][:10]
for n in top_thrashers:
    if switch_counts[n] == 0:
        break
    final_freq = dominant_freq[-1, n] + 1  # 1-indexed
    print(f"  Neuron {n}: {switch_counts[n]} switches, final freq={final_freq}, "
          f"final frac={max_frac[-1, n]:.4f}")

# %% histogram of switch counts
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=switch_counts,
    nbinsx=max(int(np.max(switch_counts)) + 1, 10),
    marker_color="steelblue",
))
fig_hist.update_layout(
    title=f"Frequency Switch Count Distribution (p={PRIME}, seed={SEED})",
    xaxis_title="Number of Dominant Frequency Switches",
    yaxis_title="Neuron Count",
    template="plotly_white",
    height=350,
    width=600,
)
fig_hist.show()

# %% commitment timeline: when do neurons commit and stay?
def compute_commitment_epoch(
    dominant_freq: np.ndarray,
    max_frac: np.ndarray,
    epochs: np.ndarray,
    threshold: float = 0.06,
    stability_window: int = 5,
):
    """Find the epoch at which each neuron commits to its final frequency.

    A neuron is "committed" when it holds the same dominant frequency
    for stability_window consecutive epochs (above threshold) through
    to the end of training.
    """
    n_epochs, d_mlp = dominant_freq.shape
    commitment_epochs = np.full(d_mlp, np.nan)
    final_freq = dominant_freq[-1]

    for n in range(d_mlp):
        if max_frac[-1, n] < threshold:
            continue  # Never commits

        # Walk backward from end to find earliest stable point
        stable_from = n_epochs - 1
        for t in range(n_epochs - 2, -1, -1):
            if max_frac[t, n] >= threshold and dominant_freq[t, n] == final_freq[n]:
                stable_from = t
            else:
                break

        commitment_epochs[n] = epochs[stable_from]

    return commitment_epochs


commitment_epochs = compute_commitment_epoch(
    dominant_freq, max_frac, epochs, threshold=UNCOMMITTED_THRESHOLD
)

committed_mask = ~np.isnan(commitment_epochs)
print(f"Neurons that commit: {np.sum(committed_mask)} / {d_mlp}")
if np.sum(committed_mask) > 0:
    print(f"Earliest commitment: epoch {int(np.nanmin(commitment_epochs))}")
    print(f"Median commitment: epoch {int(np.nanmedian(commitment_epochs[committed_mask]))}")
    print(f"Latest commitment: epoch {int(np.nanmax(commitment_epochs))}")

# Histogram of commitment epochs
fig_commit = go.Figure()
fig_commit.add_trace(go.Histogram(
    x=commitment_epochs[committed_mask],
    nbinsx=30,
    marker_color="seagreen",
))
fig_commit.update_layout(
    title=f"When Neurons Commit to Final Frequency (p={PRIME}, seed={SEED})",
    xaxis_title="Commitment Epoch",
    yaxis_title="Neuron Count",
    template="plotly_white",
    height=350,
    width=600,
)
fig_commit.show()

# %% sorted neuron view: reorder neurons by their final dominant frequency
# This makes the structure much more visible
sort_order = np.lexsort((
    -max_frac[-1],  # secondary: higher frac first within frequency
    dominant_freq[-1],  # primary: group by frequency
))

fig_sorted = render_neuron_freq_trajectory(
    dominant_freq[:, sort_order],
    max_frac[:, sort_order],
    epochs,
    threshold=UNCOMMITTED_THRESHOLD,
    title=f"Neuron Freq Trajectory (sorted by final freq) — p={PRIME}, seed={SEED}",
)
fig_sorted.show()

# %%
