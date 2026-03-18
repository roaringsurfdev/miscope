# %% imports
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope import load_family
from miscope.analysis.library import get_fourier_basis

# %% Configuration
# All primes that have both model seeds trained against dseed=598.
# p=89 excluded — only seed=999 available.
#
# Goal: hold data constant (dseed=598) and vary only the initialization.
# This isolates the model-seed effect on the epoch-0 frequency gradient profile.
#
# Methodological note: the projection W_E[:p] @ grad_W_in depends on W_E,
# which itself varies with model seed. The profiles therefore reflect both
# "which frequency direction does the gradient push?" AND "how does the current
# W_E project that gradient onto the Fourier basis?" Both are initialization
# effects — but they're not separable here. If profiles look similar despite
# different W_E, data dominates. If they diverge, initialization is steering
# from step one.

PRIMES = [59, 97, 101, 103, 107, 109, 113, 127]
MODEL_SEEDS = [485, 999]
REFERENCE_DSEED = 598
MSEED_COLORS = {485: "mediumseagreen", 999: "mediumpurple"}

family = load_family("modulo_addition_1layer")


# %% Helper: per-frequency gradient energy at epoch 0
def fourier_gradient_energy(model, train_data, train_labels, p: int) -> np.ndarray:
    """
    Compute per-frequency gradient energy in W_in at the current model state.

    Projects the gradient through W_E onto the Fourier basis, then computes
    RMS energy per frequency across neurons.

    Returns array of shape (p // 2,): one value per frequency (1-indexed).
    """
    n_freqs = p // 2
    model.zero_grad()
    logits = model(train_data)[:, -1, :p]
    loss = torch.nn.functional.cross_entropy(logits, train_labels)
    loss.backward()

    grad_W_in = model.blocks[0].mlp.W_in.grad          # (d_model, d_mlp)
    W_E = model.embed.W_E.detach()                      # (d_vocab, d_model)

    grad_R = W_E[:p] @ grad_W_in                        # (p, d_mlp)
    fourier_basis, _ = get_fourier_basis(p)
    F = fourier_basis.to(grad_R.device)
    fourier_grad = F @ grad_R                            # (p+1, d_mlp)
    fg = fourier_grad.detach().cpu().numpy()

    freq_energy = np.zeros(n_freqs)
    for k in range(1, n_freqs + 1):
        sin_row = fg[2 * k - 1]
        cos_row = fg[2 * k]
        freq_energy[k - 1] = np.sqrt(np.mean(sin_row ** 2 + cos_row ** 2))

    return freq_energy


# %% Compute epoch-0 gradient profiles for all prime × mseed combinations
print("Computing epoch-0 gradient profiles...")
profiles: dict[tuple[int, int], np.ndarray] = {}

for prime in PRIMES:
    for mseed in MODEL_SEEDS:
        v = family.get_variant(prime=prime, seed=mseed, data_seed=REFERENCE_DSEED)
        model = v.load_model_at_checkpoint(0)
        model.eval()
        td, tl, _, _, _, _ = v.generate_training_dataset()
        device = next(model.parameters()).device
        td, tl = td.to(device), tl.to(device)
        profiles[(prime, mseed)] = fourier_gradient_energy(model, td, tl, prime)
        model.zero_grad()
        del model
        print(f"  p={prime}/mseed={mseed}: done ({prime // 2} frequencies)")

print("Done.")


# %% Plot 1: Overlaid profiles — both model seeds per prime
# One subplot per prime, two lines (one per mseed).
# If the lines track closely: data-dominated gradient at epoch 0.
# If they diverge: initialization is already steering frequency selection.

n_rows, n_cols = 2, 4
fig_overlay = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"p={p}" for p in PRIMES],
    shared_xaxes=False,
)

for pidx, prime in enumerate(PRIMES):
    row, col = divmod(pidx, n_cols)
    freqs = list(range(1, prime // 2 + 1))
    for mseed in MODEL_SEEDS:
        fig_overlay.add_trace(
            go.Scatter(
                x=freqs,
                y=profiles[(prime, mseed)].tolist(),
                mode="lines",
                name=f"mseed={mseed}",
                line=dict(color=MSEED_COLORS[mseed]),
                showlegend=(pidx == 0),
            ),
            row=row + 1, col=col + 1,
        )

fig_overlay.update_layout(
    title=f"Epoch-0 gradient energy per frequency — mseed=485 vs mseed=999 (dseed={REFERENCE_DSEED})",
    height=600,
)
fig_overlay.update_xaxes(title_text="Frequency k")
fig_overlay.update_yaxes(title_text="RMS gradient energy")
fig_overlay.show()
fig_overlay.write_image("Plot1_OverlaidProfiles.png", format="png")


# %% Plot 2: Difference profiles — (mseed=999) − (mseed=485)
# Positive: mseed=999 pushes harder on that frequency at step 0.
# Negative: mseed=485 pushes harder.
# Flat/near-zero across all primes would mean data dominates initialization.

fig_diff = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"p={p}" for p in PRIMES],
    shared_xaxes=False,
)

for pidx, prime in enumerate(PRIMES):
    row, col = divmod(pidx, n_cols)
    freqs = list(range(1, prime // 2 + 1))
    diff = (profiles[(prime, 999)] - profiles[(prime, 485)]).tolist()
    fig_diff.add_trace(
        go.Bar(
            x=freqs,
            y=diff,
            name=f"p={prime}",
            marker_color=[
                "mediumpurple" if d >= 0 else "mediumseagreen" for d in diff
            ],
            showlegend=False,
        ),
        row=row + 1, col=col + 1,
    )
    fig_diff.add_hline(y=0, line_color="black", line_width=0.8, row=row + 1, col=col + 1)

fig_diff.update_layout(
    title=f"Epoch-0 gradient difference (mseed=999 − mseed=485) per prime  "
          f"[purple=999 pushes harder · green=485 pushes harder]",
    height=600,
)
fig_diff.update_xaxes(title_text="Frequency k")
fig_diff.update_yaxes(title_text="Energy difference")
fig_diff.show()
fig_diff.write_image("Plot2_DifferenceProfiles.png", format="png")


# %% Plot 3: Normalized profiles — divide each by its own mean
# Removes scale differences between primes/seeds and focuses on shape.
# If normalized shapes match: initialization scales the gradient but doesn't
# redirect it. If shapes diverge: initialization is choosing different frequencies.

fig_norm = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=[f"p={p}" for p in PRIMES],
    shared_xaxes=False,
)

for pidx, prime in enumerate(PRIMES):
    row, col = divmod(pidx, n_cols)
    freqs = list(range(1, prime // 2 + 1))
    for mseed in MODEL_SEEDS:
        raw = profiles[(prime, mseed)]
        normalized = (raw / (raw.mean() + 1e-12)).tolist()
        fig_norm.add_trace(
            go.Scatter(
                x=freqs,
                y=normalized,
                mode="lines",
                name=f"mseed={mseed}",
                line=dict(color=MSEED_COLORS[mseed]),
                showlegend=(pidx == 0),
            ),
            row=row + 1, col=col + 1,
        )

fig_norm.update_layout(
    title="Epoch-0 gradient energy (normalized by mean) — shape comparison across model seeds",
    height=600,
)
fig_norm.update_xaxes(title_text="Frequency k")
fig_norm.update_yaxes(title_text="Energy / mean energy")
fig_norm.show()
fig_norm.write_image("Plot3_NormalizedProfiles.png", format="png")

# %%
