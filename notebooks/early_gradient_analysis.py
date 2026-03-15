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
PRIME = 113
MODEL_SEED = 999
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
for kf in [9, 33, 38, 55]:
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

KEY_FREQS = [9, 33, 38, 55]

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
