# %% imports
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope import load_family
from miscope.analysis.library import get_fourier_basis

# %% Configuration
# All three dseed variants share model_seed=999, so their epoch 0 weights are identical.
# Any difference in the epoch 0 gradient is purely due to which training pairs are present.
PRIME = 59
MODEL_SEED = 485
GOOD_DATA_SEED = 598
BAD_DATA_SEEDS = [42, 999]
ALL_DATA_SEEDS = [GOOD_DATA_SEED] + BAD_DATA_SEEDS

COLORS = {598: 'steelblue', 42: 'tomato', 999: 'orange'}

family = load_family("modulo_addition_1layer")
fourier_basis, fourier_labels = get_fourier_basis(PRIME)  # shape (p+1, p)

# Frequency index helpers: basis row 0=const, 1=sin1, 2=cos1, 3=sin2, 4=cos2, ...
# For frequency k (1-indexed): sin row = 2k-1, cos row = 2k
N_FREQS = PRIME // 2  # 56 for p=113


def fourier_gradient_energy(model, train_data, train_labels, p):
    """
    Compute per-frequency gradient energy in W_in at the current model state.

    Returns array of shape (N_FREQS,): RMS gradient energy per Fourier frequency,
    summed over neurons. High value at frequency k means the first gradient step
    pushes W_in strongly in the k-th Fourier direction.

    Approach:
      1. Forward pass → cross-entropy loss on training set
      2. Backward → grad_W_in  (d_mlp × d_model)
      3. Project through W_E: grad_R = W_E[:p] @ grad_W_in.T  (p × d_mlp)
         This is the gradient of the loss w.r.t. each neuron's response to each token.
      4. Project onto Fourier basis: fourier_grad = F @ grad_R  (p+1 × d_mlp)
      5. For each frequency k, combine sin/cos rows → RMS over neurons
    """
    model.zero_grad()
    logits = model(train_data)[:, -1, :p]
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()

    # TransformerLens W_in shape: (d_model, d_mlp) = (128, 512)
    grad_W_in = model.blocks[0].mlp.W_in.grad  # (d_model, d_mlp)
    W_E = model.embed.W_E.detach()             # (d_vocab, d_model)

    # Project gradient through embedding: (p × d_mlp)
    # W_E[:p] is (p, d_model), grad_W_in is (d_model, d_mlp) → result (p, d_mlp)
    grad_R = W_E[:p] @ grad_W_in

    # Project onto Fourier basis: (p+1 × d_mlp)
    F = fourier_basis.to(grad_R.device)
    fourier_grad = F @ grad_R  # (p+1, d_mlp)

    # Per-frequency energy: combine sin + cos rows, RMS over neurons
    freq_energy = np.zeros(N_FREQS)
    fg = fourier_grad.detach().cpu().numpy()
    for k in range(1, N_FREQS + 1):
        sin_row = fg[2 * k - 1]  # shape (d_mlp,)
        cos_row = fg[2 * k]
        freq_energy[k - 1] = np.sqrt(np.mean(sin_row ** 2 + cos_row ** 2))

    return freq_energy


# %% --- EPOCH 0: Pure data-seed effect on initial gradient ---
# Load epoch 0 weights once (all dseed variants have identical epoch 0 weights).
# Run the gradient with each seed's training data to see which frequencies
# the first step pushes toward.

base_variant = family.get_variant(prime=PRIME, seed=MODEL_SEED, data_seed=GOOD_DATA_SEED)
model_epoch0 = base_variant.load_model_at_checkpoint(0)
model_epoch0.eval()

epoch0_gradients = {}
for ds in ALL_DATA_SEEDS:
    v = family.get_variant(prime=PRIME, seed=MODEL_SEED, data_seed=ds)
    td, tl, _, _, _, _ = v.generate_training_dataset()
    td = td.to(next(model_epoch0.parameters()).device)
    tl = tl.to(next(model_epoch0.parameters()).device)
    epoch0_gradients[ds] = fourier_gradient_energy(model_epoch0, td, tl, PRIME)
    model_epoch0.zero_grad()

freqs = list(range(1, N_FREQS + 1))

fig_epoch0 = go.Figure()
for ds in ALL_DATA_SEEDS:
    fig_epoch0.add_trace(go.Scatter(
        x=freqs,
        y=epoch0_gradients[ds].tolist(),
        mode='lines+markers',
        name=f"dseed={ds}",
        line=dict(color=COLORS[ds]),
        marker=dict(size=4)
    ))

fig_epoch0.update_layout(
    title=f"Epoch 0 gradient energy per Fourier frequency — p={PRIME}, model_seed={MODEL_SEED}",
    xaxis_title="Frequency k",
    yaxis_title="RMS gradient energy (W_in projected via W_E)",
    height=400
)
fig_epoch0.show()

# %% --- EPOCH 0: Difference (bad - good) ---
# Which frequencies does the bad seed push harder (positive) or softer (negative)?
# If peaks here land on p=113's known key frequencies {9, 33, 38, 55},
# that's the initial gradient bias driving divergent frequency selection.

fig_diff = go.Figure()
for ds in BAD_DATA_SEEDS:
    diff = (epoch0_gradients[ds] - epoch0_gradients[GOOD_DATA_SEED]).tolist()
    fig_diff.add_trace(go.Bar(
        x=freqs, y=diff,
        name=f"dseed={ds} − {GOOD_DATA_SEED}",
        marker_color=COLORS[ds],
        opacity=0.7
    ))
fig_diff.add_hline(y=0, line_color='black', line_width=1)

# Mark known key frequencies for p=113 (from 2026-03-08 findings: {9, 33, 38, 55})
for kf in [35, 41, 43, 44]:
    fig_diff.add_vline(x=kf, line_dash='dash', line_color='gray', opacity=0.5,
                       annotation_text=str(kf), annotation_position='top')

fig_diff.update_layout(
    title=f"Epoch 0 gradient difference (bad − good) — p={PRIME}, model_seed={MODEL_SEED}",
    xaxis_title="Frequency k",
    yaxis_title="Gradient energy difference",
    barmode='overlay',
    height=400
)
fig_diff.show()

# %% --- EARLY EPOCH SWEEP: When does divergence appear? ---
# Sweep epochs 0 → 1000 (every 100) loading the actual weights for each variant.
# At epoch 0, weights are identical — divergence comes from data alone.
# After epoch 0, weights diverge — divergence reflects accumulated history + data.
# The epoch where the curves separate is when the data seed effect locks in.

SWEEP_EPOCHS = list(range(0, 1100, 100))

sweep_gradients = {ds: {} for ds in ALL_DATA_SEEDS}

for ds in ALL_DATA_SEEDS:
    v = family.get_variant(prime=PRIME, seed=MODEL_SEED, data_seed=ds)
    td, tl, _, _, _, _ = v.generate_training_dataset()
    device = next(model_epoch0.parameters()).device
    td = td.to(device)
    tl = tl.to(device)
    for epoch in SWEEP_EPOCHS:
        m = v.load_model_at_checkpoint(epoch)
        m.eval()
        sweep_gradients[ds][epoch] = fourier_gradient_energy(m, td, tl, PRIME)
        m.zero_grad()
        del m

# %% --- SWEEP PLOT: Gradient energy at key frequencies over early epochs ---
# For each key frequency, plot how gradient energy evolves across data seeds.
# Divergence between seeds at a given epoch = that's when the data seed
# interaction with the weights creates different optimization pressure.

KEY_FREQS = [35, 41, 43, 44]

fig_sweep = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"Frequency {k}" for k in KEY_FREQS],
    shared_xaxes=True, shared_yaxes=False
)

for idx, kf in enumerate(KEY_FREQS):
    row, col = divmod(idx, 2)
    for ds in ALL_DATA_SEEDS:
        energies = [sweep_gradients[ds][e][kf - 1] for e in SWEEP_EPOCHS]
        fig_sweep.add_trace(
            go.Scatter(
                x=SWEEP_EPOCHS, y=energies,
                mode='lines+markers',
                name=f"dseed={ds}" if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=COLORS[ds]),
                marker=dict(size=5)
            ),
            row=row + 1, col=col + 1
        )

fig_sweep.update_layout(
    title=f"Gradient energy at key frequencies (epochs 0–1000) — p={PRIME}, model_seed={MODEL_SEED}",
    height=500
)
fig_sweep.update_xaxes(title_text="Epoch")
fig_sweep.show()

# %% --- SWEEP PLOT: Total gradient divergence across all frequencies ---
# Summarize: at each epoch, how different is the gradient profile between seeds?
# Metric: L2 distance in frequency-gradient space between good and each bad seed.
# Epoch where distance grows rapidly = lock-in point.

fig_divergence = go.Figure()
for ds in BAD_DATA_SEEDS:
    distances = []
    for epoch in SWEEP_EPOCHS:
        diff = sweep_gradients[ds][epoch] - sweep_gradients[GOOD_DATA_SEED][epoch]
        distances.append(float(np.linalg.norm(diff)))
    fig_divergence.add_trace(go.Scatter(
        x=SWEEP_EPOCHS, y=distances,
        mode='lines+markers',
        name=f"dseed={ds} vs {GOOD_DATA_SEED}",
        line=dict(color=COLORS[ds]),
        marker=dict(size=6)
    ))

fig_divergence.update_layout(
    title=f"Gradient profile divergence from good seed (epochs 0–1000) — p={PRIME}, model_seed={MODEL_SEED}",
    xaxis_title="Epoch",
    yaxis_title="L2 distance in frequency-gradient space",
    height=350
)
fig_divergence.show()


# %% --- MULTI-SITE ANALYSIS ---
# Do all three sites (embedding, attention, MLP) show the same frequency preference
# at epoch 0? If the data seed amplifies different frequencies at different sites,
# that's an initialization-level site conflict — a structural disagreement baked in
# before any learning happens.
#
# Projection approach (same logic for all sites):
#   - W_E[:p] projects each site's gradient from d_model into token space (p, *)
#   - Fourier basis F then projects into frequency space
#   - Per-frequency energy = RMS over all output dims (neurons / heads / model dims)
#
# Attention combines Q, K, V as three input-reading matrices (W_O omitted —
# it reads from head output space, not from the residual stream input).


def fourier_gradient_by_site(model, train_data, train_labels, p, F, n_freqs):
    """Compute per-frequency gradient energy at embedding, attention, and MLP sites.

    Returns dict with keys 'embedding', 'attention', 'mlp', each an ndarray
    of shape (n_freqs,). All three sites are projected through W_E[:p] into
    token space before Fourier decomposition, making spectra directly comparable.
    """
    model.zero_grad()
    logits = model(train_data)[:, -1, :p]
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()

    W_E = model.embed.W_E.detach()  # (d_vocab, d_model)
    F_dev = F.to(W_E.device)

    def _freq_energy(fourier_projected):
        """RMS per-frequency energy from a (p+1, d_out) Fourier-projected tensor."""
        fg = fourier_projected.detach().cpu().numpy()
        energy = np.zeros(n_freqs)
        for k in range(1, n_freqs + 1):
            sin_row = fg[2 * k - 1]
            cos_row = fg[2 * k]
            energy[k - 1] = np.sqrt(np.mean(sin_row ** 2 + cos_row ** 2))
        return energy

    # --- Embedding site ---
    # grad_W_E[:p] is (p, d_model) — already in token space; project directly.
    grad_W_E = model.embed.W_E.grad[:p]        # (p, d_model)
    emb_energy = _freq_energy(F_dev @ grad_W_E)  # (p+1, d_model) → energy

    # --- Attention site (Q, K, V) ---
    # Each W is (n_heads, d_model, d_head); project each head's d_model dim through W_E.
    attn_sq = np.zeros(n_freqs)
    n_contribution = 0
    for W_name in ("W_Q", "W_K", "W_V"):
        grad_W = getattr(model.blocks[0].attn, W_name).grad  # (n_heads, d_model, d_head)
        for h in range(grad_W.shape[0]):
            projected = W_E[:p] @ grad_W[h]      # (p, d_head)
            fg = (F_dev @ projected).detach().cpu().numpy()  # (p+1, d_head)
            for k in range(1, n_freqs + 1):
                attn_sq[k - 1] += np.mean(fg[2 * k - 1] ** 2 + fg[2 * k] ** 2)
            n_contribution += 1
    attn_energy = np.sqrt(attn_sq / n_contribution)

    # --- MLP site ---
    grad_W_in = model.blocks[0].mlp.W_in.grad   # (d_model, d_mlp)
    grad_R = W_E[:p] @ grad_W_in                 # (p, d_mlp)
    mlp_energy = _freq_energy(F_dev @ grad_R)

    return {"embedding": emb_energy, "attention": attn_energy, "mlp": mlp_energy}


# %% --- EPOCH 0 MULTI-SITE: Same weights, different data → do sites agree? ---
# Run multi-site gradient for each data seed using the shared epoch-0 model.
# Within each seed: do the three site spectra peak at the same frequencies?
# Across seeds: does the data seed shift all sites together, or differentially?

SITE_COLORS = {"embedding": "royalblue", "attention": "darkorange", "mlp": "seagreen"}
SITE_DASH   = {"embedding": "solid", "attention": "dash", "mlp": "dot"}

epoch0_by_site = {}
for ds in ALL_DATA_SEEDS:
    v = family.get_variant(prime=PRIME, seed=MODEL_SEED, data_seed=ds)
    td, tl, _, _, _, _ = v.generate_training_dataset()
    td = td.to(next(model_epoch0.parameters()).device)
    tl = tl.to(next(model_epoch0.parameters()).device)
    epoch0_by_site[ds] = fourier_gradient_by_site(
        model_epoch0, td, tl, PRIME, fourier_basis, N_FREQS
    )
    model_epoch0.zero_grad()

freqs = list(range(1, N_FREQS + 1))

# One subplot per data seed — three site traces each
fig_multisite = make_subplots(
    rows=len(ALL_DATA_SEEDS), cols=1,
    subplot_titles=[f"dseed={ds}" for ds in ALL_DATA_SEEDS],
    shared_xaxes=True,
)
for row_idx, ds in enumerate(ALL_DATA_SEEDS):
    for site, energy in epoch0_by_site[ds].items():
        fig_multisite.add_trace(
            go.Scatter(
                x=freqs, y=energy.tolist(),
                mode="lines",
                name=site,
                showlegend=(row_idx == 0),
                line=dict(color=SITE_COLORS[site], dash=SITE_DASH[site], width=1.5),
            ),
            row=row_idx + 1, col=1,
        )
    # Mark known key frequencies
    for kf in [35, 41, 43, 44]:
        fig_multisite.add_vline(
            x=kf, line_dash="dash", line_color="gray", opacity=0.4, row=row_idx + 1, col=1,
        )

fig_multisite.update_xaxes(title_text="Frequency k", row=len(ALL_DATA_SEEDS), col=1)
fig_multisite.update_layout(
    title=f"Epoch 0 multi-site gradient energy — p={PRIME}, model_seed={MODEL_SEED}",
    height=200 * len(ALL_DATA_SEEDS) + 100,
)
fig_multisite.show()


# %% --- SITE DIFFERENCE: Does data seed shift sites uniformly or differentially? ---
# For each bad seed, compute (bad - good) per site.
# If all three sites shift together → data seed applies uniform pressure.
# If sites diverge → data seed creates site-level conflict at initialization.

fig_site_diff = make_subplots(
    rows=len(BAD_DATA_SEEDS), cols=1,
    subplot_titles=[f"dseed={ds} − {GOOD_DATA_SEED}" for ds in BAD_DATA_SEEDS],
    shared_xaxes=True,
)
for row_idx, ds in enumerate(BAD_DATA_SEEDS):
    for site in ("embedding", "attention", "mlp"):
        diff = (epoch0_by_site[ds][site] - epoch0_by_site[GOOD_DATA_SEED][site]).tolist()
        fig_site_diff.add_trace(
            go.Bar(
                x=freqs, y=diff,
                name=site,
                showlegend=(row_idx == 0),
                marker_color=SITE_COLORS[site],
                opacity=0.65,
            ),
            row=row_idx + 1, col=1,
        )
    for kf in [35, 41, 43, 44]:
        fig_site_diff.add_vline(
            x=kf, line_dash="dash", line_color="gray", opacity=0.4, row=row_idx + 1, col=1,
        )
    fig_site_diff.add_hline(y=0, line_color="black", line_width=1, row=row_idx + 1, col=1)

fig_site_diff.update_xaxes(title_text="Frequency k", row=len(BAD_DATA_SEEDS), col=1)
fig_site_diff.update_layout(
    title=f"Epoch 0 site-level gradient shift (bad − good dseed) — p={PRIME}, model_seed={MODEL_SEED}",
    barmode="overlay",
    height=250 * len(BAD_DATA_SEEDS) + 100,
)
fig_site_diff.show()

# %%


# %% --- FULL-TRAINING MULTI-SITE SWEEP ---
# Extend the epoch-0 analysis across all five training windows.
# Window boundaries come from variant_summary.json — no hardcoding.
# At each checkpoint, we compute site-level gradient energy using each
# variant's own training data (captures the accumulated effect of that seed).
#
# Visualization: per-site gradient energy heatmap over training time.
# X = epoch, Y = frequency, Color = energy, panels = sites.
# This shows: when does each frequency's gradient energy peak at each site?
# When do the three sites converge onto the same frequencies?

import json

WINDOW_NAMES = [
    "first_descent_window",
    "plateau_window",
    "second_descent_window",
    "final_window",
]


def load_variant_summary(prime, seed, data_seed):
    v = family.get_variant(prime=prime, seed=seed, data_seed=data_seed)
    path = v.variant_dir / "variant_summary.json"
    with open(path) as f:
        return json.load(f)


def sample_window_epochs(summary, n_interior=2):
    """Return a sorted list of sample epochs spanning all windows.

    Includes each window's start/end boundary plus n_interior evenly-spaced
    interior points. Deduplicates and sorts.
    """
    epochs = set()
    for name in WINDOW_NAMES:
        w = summary.get(name)
        if w is None:
            continue
        start, end = w["start_epoch"], w["end_epoch"]
        epochs.add(start)
        epochs.add(end)
        if n_interior > 0 and end > start:
            step = (end - start) / (n_interior + 1)
            for i in range(1, n_interior + 1):
                epochs.add(round(start + i * step))
    return sorted(epochs)


def window_boundaries(summary):
    """Return list of (name, start_epoch, end_epoch) for vertical line markers."""
    bounds = []
    for name in WINDOW_NAMES:
        w = summary.get(name)
        if w:
            bounds.append((name, w["start_epoch"], w["end_epoch"]))
    return bounds


def snap_to_available(requested_epochs, available_epochs):
    """Map each requested epoch to the nearest available checkpoint epoch."""
    available = sorted(available_epochs)
    snapped = []
    for ep in requested_epochs:
        nearest = min(available, key=lambda a: abs(a - ep))
        snapped.append(nearest)
    return sorted(set(snapped))


def run_multisite_sweep(prime, seed, data_seed, n_interior=2):
    """Load checkpoints at window-sampled epochs and compute site-level gradient energy.

    Returns:
        epochs: list of int
        energies: dict[epoch -> dict[site -> ndarray(n_freqs)]]
    """
    v = family.get_variant(prime=prime, seed=seed, data_seed=data_seed)
    td, tl, _, _, _, _ = v.generate_training_dataset()
    summary = load_variant_summary(prime, seed, data_seed)
    requested = sample_window_epochs(summary, n_interior=n_interior)
    epochs = snap_to_available(requested, v.get_available_checkpoints())
    print(f"  p={prime}/s={seed}/ds={data_seed}: {len(epochs)} checkpoints — {epochs}")

    energies = {}
    for epoch in epochs:
        m = v.load_model_at_checkpoint(epoch)
        device = next(m.parameters()).device
        td_dev = td.to(device)
        tl_dev = tl.to(device)
        energies[epoch] = fourier_gradient_by_site(
            m, td_dev, tl_dev, prime, fourier_basis, N_FREQS
        )
        m.zero_grad()
        del m
    return epochs, energies


# %% --- RUN SWEEP FOR CANON MODEL + COMPARISON DATA SEEDS ---
print("Running full-training multi-site sweep...")
sweep_results = {}
for ds in ALL_DATA_SEEDS:
    print(f"  data_seed={ds}")
    sweep_results[ds] = run_multisite_sweep(PRIME, MODEL_SEED, ds, n_interior=2)


# %% --- PLOT: Site-level gradient energy heatmap over training ---
# One figure per data seed. Three heatmap panels: embedding / attention / MLP.
# Color = gradient energy at that (epoch, frequency) cell.
# Vertical lines mark window boundaries.

SITES = ["embedding", "attention", "mlp"]
SITE_TITLES = {"embedding": "Embedding", "attention": "Attention (Q+K+V)", "mlp": "MLP"}


def _row_normalize(vec):
    """Normalize a vector by its L2 norm; return zeros if norm is negligible."""
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-30 else np.zeros_like(vec)


def plot_site_heatmaps(prime, seed, data_seed, epochs, energies, summary):
    bounds = window_boundaries(summary)
    # Row-normalize each epoch's energy vector so color shows relative frequency
    # preference regardless of gradient magnitude (which collapses 7 orders of
    # magnitude between epoch 0 and post-grokking).
    z_matrices = {}
    for site in SITES:
        matrix = []
        for ep in epochs:
            matrix.append(_row_normalize(energies[ep][site]).tolist())
        # shape: (n_epochs, n_freqs) — rows=epochs, cols=freqs
        z_matrices[site] = matrix

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[SITE_TITLES[s] for s in SITES],
        shared_yaxes=True,
    )
    for col_idx, site in enumerate(SITES):
        fig.add_trace(
            go.Heatmap(
                z=z_matrices[site],
                x=list(range(1, N_FREQS + 1)),
                y=epochs,
                colorscale="Viridis",
                showscale=(col_idx == 2),
                colorbar=dict(title="Energy", len=0.8) if col_idx == 2 else None,
            ),
            row=1, col=col_idx + 1,
        )
        # Mark key frequencies
        for kf in [9, 33, 38, 55]:
            fig.add_vline(
                x=kf, line_dash="dash", line_color="white", opacity=0.5,
                row=1, col=col_idx + 1,
            )
    # Mark window boundaries on the y-axis (epoch axis)
    for name, start, end in bounds:
        for ep_mark in [start, end]:
            fig.add_hline(
                y=ep_mark, line_dash="dot", line_color="red", opacity=0.4,
            )

    fig.update_yaxes(title_text="Epoch", col=1)
    fig.update_xaxes(title_text="Frequency k")
    fig.update_layout(
        title=f"Site-level gradient energy across training — p={prime}/s={seed}/ds={data_seed}",
        height=500,
        width=1100,
    )
    fig.show()
    fig.write_image(f"exports/site_level_gradient_energy_p{prime}s{seed}ds{data_seed}.png", format="png")


for ds in ALL_DATA_SEEDS:
    epochs, energies = sweep_results[ds]
    summary = load_variant_summary(PRIME, MODEL_SEED, ds)
    plot_site_heatmaps(PRIME, MODEL_SEED, ds, epochs, energies, summary)


# %% --- PLOT: Site convergence over training ---
# For each epoch and data seed: how much do the three site spectra agree?
# Metric: pairwise cosine similarity between site energy vectors.
# High similarity = sites pointing at the same frequencies.
# Low similarity = site conflict.
# This directly tests whether site conflict resolves during second descent.

from numpy.linalg import norm


def cosine_sim(a, b):
    """Cosine similarity between gradient direction vectors.

    Normalizes both inputs before dotting so magnitude collapse at low loss
    (post-grokking) doesn't corrupt the result. Returns NaN if either vector
    is effectively zero (no gradient signal at all).
    """
    na, nb = norm(a), norm(b)
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float(np.dot(a / na, b / nb))


SITE_PAIRS = [
    ("embedding", "attention"),
    ("embedding", "mlp"),
    ("attention", "mlp"),
]
PAIR_COLORS = {
    ("embedding", "attention"): "royalblue",
    ("embedding", "mlp"): "darkorange",
    ("attention", "mlp"): "seagreen",
}

fig_conv = make_subplots(
    rows=len(ALL_DATA_SEEDS), cols=1,
    subplot_titles=[f"dseed={ds}" for ds in ALL_DATA_SEEDS],
    shared_xaxes=True,
)
for row_idx, ds in enumerate(ALL_DATA_SEEDS):
    epochs, energies = sweep_results[ds]
    summary = load_variant_summary(PRIME, MODEL_SEED, ds)
    bounds = window_boundaries(summary)
    for pair in SITE_PAIRS:
        sims = [cosine_sim(energies[ep][pair[0]], energies[ep][pair[1]]) for ep in epochs]
        # NaN → None so Plotly renders a gap rather than a misleading zero
        sims = [s if not np.isnan(s) else None for s in sims]
        fig_conv.add_trace(
            go.Scatter(
                x=epochs, y=sims,
                mode="lines+markers",
                name=f"{pair[0]} ↔ {pair[1]}",
                showlegend=(row_idx == 0),
                line=dict(color=PAIR_COLORS[pair], width=1.5),
                marker=dict(size=4),
            ),
            row=row_idx + 1, col=1,
        )
    for _, start, end in bounds:
        for ep_mark in [start, end]:
            fig_conv.add_vline(
                x=ep_mark, line_dash="dot", line_color="gray", opacity=0.5,
                row=row_idx + 1, col=1,
            )

fig_conv.update_xaxes(title_text="Epoch", row=len(ALL_DATA_SEEDS), col=1)
fig_conv.update_yaxes(title_text="Cosine similarity", range=[0, 1])
fig_conv.update_layout(
    title=f"Site convergence over training — p={PRIME}/s={MODEL_SEED}",
    height=220 * len(ALL_DATA_SEEDS) + 100,
)
fig_conv.show()
fig_conv.write_image(f"exports/convergence_p{PRIME}s{MODEL_SEED}.png", format="png")

# %%
