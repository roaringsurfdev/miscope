# %% He et al. (2026) Neuron Fourier Analysis — Proof of Concept
# Reproduces key results from He, Wang, Chen & Yang (arXiv:2602.16849):
#   1. Per-neuron Fourier heatmaps (§3.1, Fig. 2)
#   2. IPR trajectory (§3.3, Fig. 5d)
#   3. Phase alignment tracking (§3.2, Fig. 5c)
#   4. Lottery ticket prediction from initialization (§6.1, Corollary 6.1)
#   5. Healthy (p=113) vs. anomalous (p=101) comparison
#
# Requires: neuron_fourier artifacts computed for both variants.
# Run notebooks/run_analysis.py first if artifacts are missing.

# %% imports
import sys
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family                          # also registers view catalog
from miscope.analysis.artifact_loader import ArtifactLoader

# %% configuration
FAMILY_NAME = "modulo_addition_1layer"
HEALTHY = dict(prime=113, seed=999)
ANOMALOUS = dict(prime=101, seed=999)

family = load_family(FAMILY_NAME)

# %% --- helpers ---

def compute_ipr(magnitudes: np.ndarray) -> np.ndarray:
    """IPR per neuron: Σ_k α_k^4 / (Σ_k α_k^2)^2.

    magnitudes: (..., K) — per-neuron magnitude vectors.
    Returns (...,) IPR values in [1/K, 1.0].
    """
    num = np.sum(magnitudes ** 4, axis=-1)
    denom = np.sum(magnitudes ** 2, axis=-1) ** 2
    return np.where(denom > 0, num / denom, 0.0)


def dominant_freq_idx(magnitudes: np.ndarray) -> np.ndarray:
    """Index of dominant frequency per neuron. magnitudes: (..., M, K) → (..., M)."""
    return np.argmax(magnitudes, axis=-1)


def phase_misalignment(phi_mk: np.ndarray, psi_mk: np.ndarray) -> np.ndarray:
    """D̃^k_m = |wrap(2φ - ψ)| ∈ [0, π]. Measures distance from alignment.

    phi_mk, psi_mk: any broadcastable shape ending in K.
    Returns same shape without the K axis (squeezed last axis if K=1).
    """
    diff = 2 * phi_mk - psi_mk
    wrapped = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.abs(wrapped)


def alignment_progress_series(
    phi_mk: np.ndarray,
    psi_mk: np.ndarray,
    alpha_mk: np.ndarray,
) -> np.ndarray:
    """Population phase alignment progress per epoch (He et al. Fig 5c).

    Uses the dominant-frequency phase for each neuron at each epoch.
    Returns mean |sin(2φ_m - ψ_m)| across neurons — lower is better aligned.

    alpha_mk, phi_mk, psi_mk: (n_epochs, M, K).
    Returns: (n_epochs,).
    """
    n_epochs, M, _ = alpha_mk.shape
    progress = np.zeros(n_epochs)
    m_idx = np.arange(M)
    for t in range(n_epochs):
        dom_k = dominant_freq_idx(alpha_mk[t])          # (M,)
        phi_dom = phi_mk[t, m_idx, dom_k]               # (M,)
        psi_dom = psi_mk[t, m_idx, dom_k]               # (M,)
        progress[t] = np.mean(np.abs(np.sin(2 * phi_dom - psi_dom)))
    return progress


def load_fourier(family, prime: int, seed: int) -> dict:
    """Load all neuron_fourier epochs for a variant. Returns stacked dict."""
    variant = family.get_variant(prime=prime, seed=seed)
    loader = ArtifactLoader(variant.artifacts_dir)
    stacked = loader.load_epochs("neuron_fourier")
    # freq_indices is constant across epochs — keep only the first copy
    stacked["freq_indices"] = stacked["freq_indices"][0]
    return stacked


# %% load healthy variant
data_h = load_fourier(family, **HEALTHY)
epochs_h = data_h["epochs"]                   # (n_epochs,)
alpha_h  = data_h["alpha_mk"]                 # (n_epochs, M, K)
phi_h    = data_h["phi_mk"]
beta_h   = data_h["beta_mk"]
psi_h    = data_h["psi_mk"]
freq_idx = data_h["freq_indices"]             # (K,)

n_epochs_h, M, K = alpha_h.shape
p = HEALTHY["prime"]

print(f"Healthy: p={p}, seed={HEALTHY['seed']}")
print(f"  {n_epochs_h} epochs  |  {M} neurons  |  {K} frequencies")
print(f"  Epoch range: {int(epochs_h[0])} – {int(epochs_h[-1])}")


# %% --- Section 1: Frequency Heatmaps (He et al. Fig. 2) ---
# View catalog renders alpha_mk and beta_mk for a given epoch.

variant_h = family.get_variant(**HEALTHY)
final_epoch_h = int(epochs_h[-1])

fig_input  = variant_h.at(final_epoch_h).view("neuron_fourier_heatmap").figure()
fig_output = variant_h.at(final_epoch_h).view("neuron_fourier_heatmap_output").figure()

fig_input.show()
fig_output.show()


# %% --- Section 2: IPR Trajectory (He et al. Fig. 5d) ---

ipr_input_h  = compute_ipr(alpha_h)             # (n_epochs, M)
ipr_output_h = compute_ipr(beta_h)              # (n_epochs, M)

mean_ipr_input_h  = ipr_input_h.mean(axis=1)   # (n_epochs,)
mean_ipr_output_h = ipr_output_h.mean(axis=1)

fig_ipr = go.Figure()
fig_ipr.add_trace(go.Scatter(
    x=epochs_h, y=mean_ipr_input_h,
    mode="lines", name="Input (α)",
    line=dict(color="steelblue"),
))
fig_ipr.add_trace(go.Scatter(
    x=epochs_h, y=mean_ipr_output_h,
    mode="lines", name="Output (β)",
    line=dict(color="coral"),
))
fig_ipr.update_layout(
    title=f"Mean IPR over Training — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="Epoch",
    yaxis_title="Mean IPR  (higher = more specialized)",
    template="plotly_white",
    height=400,
)
fig_ipr.show()

early_idx = min(3, n_epochs_h - 1)
for label, series in [("Input α", mean_ipr_input_h), ("Output β", mean_ipr_output_h)]:
    print(f"{label}:  early={series[early_idx]:.4f}  final={series[-1]:.4f}"
          f"  ({series[-1] / series[early_idx]:.1f}x)")


# %% --- Section 3: Phase Alignment (He et al. Fig. 5c) ---

# Population progress over training
align_progress_h = alignment_progress_series(phi_h, psi_h, alpha_h)

fig_progress = go.Figure()
fig_progress.add_trace(go.Scatter(
    x=epochs_h, y=align_progress_h,
    mode="lines", name=f"p={p}",
    line=dict(color="steelblue"),
))
fig_progress.update_layout(
    title=f"Phase Alignment Progress — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="Epoch",
    yaxis_title="Mean |sin(2φ − ψ)|  (lower = better aligned)",
    template="plotly_white",
    height=400,
)
fig_progress.show()

# Scatter: ψ_m vs. 2φ_m at final epoch (dominant frequency per neuron)
m_idx = np.arange(M)
dom_k_final = dominant_freq_idx(alpha_h[-1])          # (M,)
phi_final = phi_h[-1, m_idx, dom_k_final]             # (M,)
psi_final = psi_h[-1, m_idx, dom_k_final]             # (M,)

phi_line = np.linspace(-np.pi, np.pi, 200)
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=phi_line, y=2 * phi_line,
    mode="lines", name="ψ = 2φ (ideal)",
    line=dict(color="grey", dash="dash"),
))
fig_scatter.add_trace(go.Scatter(
    x=phi_final, y=psi_final,
    mode="markers", name="Neurons",
    marker=dict(color="steelblue", size=4, opacity=0.6),
))
fig_scatter.update_layout(
    title=f"Phase Alignment at Final Epoch — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="Input phase φ_m",
    yaxis_title="Output phase ψ_m",
    template="plotly_white",
    height=460,
)
fig_scatter.show()


# %% --- Section 4: Lottery Ticket Prediction (He et al. Corollary 6.1) ---
# Predict winning frequency from initialization via smallest phase misalignment.

D_init = phase_misalignment(phi_h[0], psi_h[0])      # (M, K)
predicted_k = np.argmin(D_init, axis=1)               # (M,) — predicted winner index
observed_k  = dominant_freq_idx(alpha_h[-1])          # (M,) — actual winner at final epoch

accuracy = np.mean(predicted_k == observed_k)
n_correct = int(accuracy * M)
print(f"\nLottery Ticket Prediction Accuracy: {accuracy:.1%}  ({n_correct}/{M} neurons)")

wrong_mask = predicted_k != observed_k
n_wrong = int(wrong_mask.sum())
if n_wrong > 0:
    freq_error = np.abs(freq_idx[predicted_k[wrong_mask]] - freq_idx[observed_k[wrong_mask]])
    print(f"  Mispredicted: {n_wrong} neurons")
    print(f"  Median |Δk| for mispredicted: {np.median(freq_error):.1f}")
    print(f"  Max |Δk|: {int(freq_error.max())}")

# Distribution of initial misalignment at predicted winner vs. next-best competitor
winner_misalign  = D_init[m_idx, predicted_k]
sorted_D = np.sort(D_init, axis=1)
margin = sorted_D[:, 1] - sorted_D[:, 0]              # gap to runner-up

fig_margin = go.Figure()
fig_margin.add_trace(go.Histogram(
    x=margin,
    nbinsx=40,
    marker_color="steelblue",
    name="Margin",
))
fig_margin.update_layout(
    title=f"Lottery Ticket Initialization Margin — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="D̃(2nd best) − D̃(winner)  (radians)",
    yaxis_title="Neuron count",
    template="plotly_white",
    height=360,
)
fig_margin.show()

print(f"\n  Mean winner misalignment at init: {winner_misalign.mean():.3f} rad")
print(f"  Mean margin to runner-up:          {margin.mean():.3f} rad")


# %% --- Section 5: Healthy vs. Anomalous Comparison ---

data_a = load_fourier(family, **ANOMALOUS)
epochs_a = data_a["epochs"]
alpha_a  = data_a["alpha_mk"]
phi_a    = data_a["phi_mk"]
psi_a    = data_a["psi_mk"]
p_a = ANOMALOUS["prime"]

print(f"\nAnomalous: p={p_a}, seed={ANOMALOUS['seed']}")
print(f"  {len(epochs_a)} epochs  |  {alpha_a.shape[1]} neurons  |  {alpha_a.shape[2]} frequencies")

ipr_input_a   = compute_ipr(alpha_a).mean(axis=1)
align_progress_a = alignment_progress_series(phi_a, psi_a, alpha_a)

# Side-by-side: IPR
fig_ipr_compare = go.Figure()
fig_ipr_compare.add_trace(go.Scatter(
    x=epochs_h, y=mean_ipr_input_h,
    mode="lines", name=f"p={p} (healthy)",
    line=dict(color="steelblue"),
))
fig_ipr_compare.add_trace(go.Scatter(
    x=epochs_a, y=ipr_input_a,
    mode="lines", name=f"p={p_a} (anomalous)",
    line=dict(color="tomato"),
))
fig_ipr_compare.update_layout(
    title="IPR Comparison: Healthy vs. Anomalous",
    xaxis_title="Epoch",
    yaxis_title="Mean Input IPR",
    template="plotly_white",
    height=400,
)
fig_ipr_compare.show()

# Side-by-side: phase alignment
fig_align_compare = go.Figure()
fig_align_compare.add_trace(go.Scatter(
    x=epochs_h, y=align_progress_h,
    mode="lines", name=f"p={p} (healthy)",
    line=dict(color="steelblue"),
))
fig_align_compare.add_trace(go.Scatter(
    x=epochs_a, y=align_progress_a,
    mode="lines", name=f"p={p_a} (anomalous)",
    line=dict(color="tomato"),
))
fig_align_compare.update_layout(
    title="Phase Alignment Comparison: Healthy vs. Anomalous",
    xaxis_title="Epoch",
    yaxis_title="Mean |sin(2φ − ψ)|",
    template="plotly_white",
    height=400,
)
fig_align_compare.show()

# Heatmap for anomalous at final epoch
variant_a = family.get_variant(**ANOMALOUS)
final_epoch_a = int(epochs_a[-1])
variant_a.at(final_epoch_a).view("neuron_fourier_heatmap").show()


# %% --- Section 6: Margin vs. Switching — Phase and Magnitude Predictors ---
# Tests whether initialization conditions predict which neurons thrash.
# Compares two predictors: He et al.'s phase margin vs. magnitude ratio.
# Also addresses Section 3's dilution issue via conditioned alignment.

# -- 6a: Switch counts from neuron_freq_norm (activation-space) --

loader_h = ArtifactLoader(variant_h.artifacts_dir)
norm_stacked = loader_h.load_epochs("neuron_freq_norm")
norm_matrix = norm_stacked["norm_matrix"]       # (n_epochs, n_freq, d_mlp)
norm_epochs = norm_stacked["epochs"]

SWITCH_THRESHOLD = 0.06   # 3× uniform baseline; matches neuron_freq_trajectory.py

dominant_acts = np.argmax(norm_matrix, axis=1)  # (n_epochs, d_mlp)
max_frac_acts = np.max(norm_matrix, axis=1)     # (n_epochs, d_mlp)


def count_switches(dominant_freq: np.ndarray, max_frac: np.ndarray, threshold: float) -> np.ndarray:
    """Per-neuron switch count between committed states."""
    n_epochs, d_mlp = dominant_freq.shape
    counts = np.zeros(d_mlp, dtype=int)
    last = np.full(d_mlp, -1, dtype=int)
    for t in range(n_epochs):
        committed = max_frac[t] >= threshold
        switched = committed & (last >= 0) & (dominant_freq[t] != last)
        counts += switched.astype(int)
        last = np.where(committed, dominant_freq[t], last)
    return counts


switch_counts = count_switches(dominant_acts, max_frac_acts, SWITCH_THRESHOLD)
print(f"Switch counts — p={p}, seed={HEALTHY['seed']}")
print(f"  Never switched: {np.sum(switch_counts == 0)}/{M}")
print(f"  1+ switches:    {np.sum(switch_counts >= 1)}/{M}")
print(f"  Max switches:   {switch_counts.max()}")
print(f"  Mean:           {switch_counts.mean():.2f}")


# -- 6b: Magnitude margin at initialization --

alpha_0 = alpha_h[0]                                    # (M, K) — magnitudes at epoch 0
sorted_alpha = np.sort(alpha_0, axis=1)[:, ::-1]       # descending
mag_ratio = sorted_alpha[:, 0] / (sorted_alpha[:, 1] + 1e-8)   # top / 2nd
mag_winner_k = np.argmax(alpha_0, axis=1)              # predicted winner by magnitude

# Compare prediction accuracy: phase margin vs. magnitude ratio
mag_accuracy = np.mean(mag_winner_k == observed_k)
print(f"\nInitialization predictor accuracy (→ final dominant frequency):")
print(f"  Phase margin (He et al.):    {accuracy:.1%}  ({int(accuracy * M)}/{M})")
print(f"  Magnitude ratio:             {mag_accuracy:.1%}  ({int(mag_accuracy * M)}/{M})")
print(f"  Chance baseline (1/K={K}):  {1/K:.1%}")


# -- 6c: Scatter plots --

rng = np.random.default_rng(42)
jitter = rng.uniform(-0.2, 0.2, M)
ipr_final = ipr_alpha_h[-1]                             # (M,) per-neuron IPR at final epoch

# Phase margin vs. switches
fig_phase_sw = go.Figure()
fig_phase_sw.add_trace(go.Scatter(
    x=margin,
    y=switch_counts + jitter,
    mode="markers",
    marker=dict(color=ipr_final, colorscale="Viridis", size=4, opacity=0.6,
                colorbar=dict(title="Final IPR")),
    text=[f"Neuron {m}" for m in range(M)],
    hovertemplate="Neuron %{text}<br>Phase margin: %{x:.3f} rad<br>Switches: %{customdata}<extra></extra>",
    customdata=switch_counts,
))
fig_phase_sw.update_layout(
    title=f"Phase Margin at Init vs. Frequency Switches — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="D̃(2nd best) − D̃(winner) at epoch 0  (radians)",
    yaxis_title="Switch count  (jittered)",
    template="plotly_white",
    height=440,
)
fig_phase_sw.show()

# Magnitude ratio vs. switches
fig_mag_sw = go.Figure()
fig_mag_sw.add_trace(go.Scatter(
    x=np.log10(mag_ratio),
    y=switch_counts + jitter,
    mode="markers",
    marker=dict(color=ipr_final, colorscale="Viridis", size=4, opacity=0.6,
                colorbar=dict(title="Final IPR")),
    text=[f"Neuron {m}" for m in range(M)],
    hovertemplate="Neuron %{text}<br>log₁₀(ratio): %{x:.3f}<br>Switches: %{customdata}<extra></extra>",
    customdata=switch_counts,
))
fig_mag_sw.update_layout(
    title=f"Magnitude Ratio at Init vs. Frequency Switches — p={p}, seed={HEALTHY['seed']}",
    xaxis_title="log₁₀(α_top / α_2nd)  at epoch 0",
    yaxis_title="Switch count  (jittered)",
    template="plotly_white",
    height=440,
)
fig_mag_sw.show()


# -- 6d: Conditioned alignment progress --
# Section 3's population metric is diluted by uncommitted neurons (IPR near random).
# Conditioning on IPR > threshold isolates neurons that have actually committed.

IPR_THRESHOLD = 0.25
print(f"\nConditioned phase alignment (IPR > {IPR_THRESHOLD}, dominant-frequency neurons only):")
print(f"  {'Epoch':>8}  {'Committed':>10}  {'Unconditioned':>14}  {'Conditioned':>12}")

sample_indices = np.linspace(0, n_epochs_h - 1, min(8, n_epochs_h), dtype=int)
for t in sample_indices:
    mask = ipr_alpha_h[t] > IPR_THRESHOLD
    dom_k_t = dominant_freq_idx(alpha_h[t])
    phi_t = phi_h[t, m_idx, dom_k_t]
    psi_t = psi_h[t, m_idx, dom_k_t]
    uncond = np.mean(np.abs(np.sin(2 * phi_t - psi_t)))
    cond = np.mean(np.abs(np.sin(2 * phi_t[mask] - psi_t[mask]))) if mask.sum() > 0 else float("nan")
    print(f"  {int(epochs_h[t]):>8}  {mask.sum():>10}  {uncond:>14.4f}  {cond:>12.4f}")

# %%
