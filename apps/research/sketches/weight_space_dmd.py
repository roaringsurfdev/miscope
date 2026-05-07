# %% imports
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope import load_family
from miscope.analysis.library.dmd import compute_dmd
from miscope.analysis.library import get_fourier_basis

# %% Configuration
# Focus: first-descent window (epochs 0–1500, 100-epoch intervals = 16 snapshots).
# Primary group: p113/seed999 — mixed outcomes across data seeds.
# Comparison group: p109/seed485 — all three data seeds grokk (stability reference).
#
# Key questions:
#   1. Is first descent low-rank or high-rank? (singular value spectrum)
#   2. Do the data seeds show different DMD structure even in this early window?
#   3. Which Fourier frequencies do the dominant W_in and W_E modes activate?

FAMILY_NAME = "modulo_addition_1layer"
WEIGHT_KEYS = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]

FIRST_DESCENT_MAX_EPOCH = 1500

GROUPS = [
    (113, 999, [598, 42, 999]),   # mixed outcomes
    (109, 485, [598, 42, 999]),   # all grokk — stability reference
]

COLORS = {42: "tomato", 598: "steelblue", 999: "orange"}

family = load_family(FAMILY_NAME)


# %% Helper: load first-descent weight trajectory as a stacked array
def load_first_descent_trajectory(
    variant,
    max_epoch: int = FIRST_DESCENT_MAX_EPOCH,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """Load parameter_snapshot epochs up to max_epoch.

    Returns:
        epochs: list of included checkpoint epochs
        weights: dict mapping matrix name -> (n_epochs, *shape) stacked array
    """
    all_epochs = variant.artifacts.get_epochs("parameter_snapshot")
    epochs = [e for e in all_epochs if e <= max_epoch]
    stacked: dict[str, list[np.ndarray]] = {k: [] for k in WEIGHT_KEYS}
    for epoch in epochs:
        snap = variant.artifacts.load_epoch("parameter_snapshot", epoch)
        for k in WEIGHT_KEYS:
            stacked[k].append(snap[k])
    return epochs, {k: np.stack(v) for k, v in stacked.items()}


# %% Helper: Fourier projection of a weight-space mode vector
def fourier_project_W_in_mode(
    mode_vec: np.ndarray,
    W_E_epoch0: np.ndarray,
    fourier_basis: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Project a W_in DMD mode onto the Fourier basis via W_E.

    Mirrors the gradient analysis: mode_R = W_E[:p] @ mode_W_in gives the
    mode's effect on each neuron's response to each token. Projecting through
    the Fourier basis then gives per-frequency energy in this mode.

    Args:
        mode_vec: Real (or magnitude) part of DMD mode, shape (d_model * d_mlp,)
        W_E_epoch0: Embedding matrix at epoch 0, shape (vocab, d_model)
        fourier_basis: (p, p) Fourier basis matrix
        prime: p

    Returns:
        freq_energy: (n_freqs,) RMS energy per Fourier frequency
    """
    d_model, d_mlp = W_E_epoch0.shape[1], mode_vec.shape[0] // W_E_epoch0.shape[1]
    mode_W_in = mode_vec.reshape(d_model, d_mlp)        # (d_model, d_mlp)
    mode_R = W_E_epoch0[:prime] @ mode_W_in             # (p, d_mlp)
    F = fourier_basis[:prime, :prime]
    fourier_mode = F @ mode_R                           # (p, d_mlp)

    n_freqs = prime // 2
    freq_energy = np.zeros(n_freqs)
    for k in range(1, n_freqs + 1):
        sin_row = fourier_mode[2 * k - 1]
        cos_row = fourier_mode[2 * k] if 2 * k < prime else np.zeros_like(sin_row)
        freq_energy[k - 1] = np.sqrt(np.mean(sin_row ** 2 + cos_row ** 2))
    return freq_energy


def fourier_project_W_E_mode(
    mode_vec: np.ndarray,
    fourier_basis: np.ndarray,
    prime: int,
    d_vocab: int,
    d_model: int,
) -> np.ndarray:
    """Project a W_E DMD mode onto the Fourier basis directly.

    Args:
        mode_vec: Real (or magnitude) part of DMD mode, shape (d_vocab * d_model,)
        fourier_basis: (p, p) Fourier basis matrix
        prime: p
        d_vocab: vocabulary size (p+1)
        d_model: model dimension

    Returns:
        freq_energy: (n_freqs,) RMS energy per Fourier frequency
    """
    mode_W_E = mode_vec.reshape(d_vocab, d_model)       # (p+1, d_model)
    F = fourier_basis[:prime, :prime]
    fourier_mode = F @ mode_W_E[:prime]                 # (p, d_model)

    n_freqs = prime // 2
    freq_energy = np.zeros(n_freqs)
    for k in range(1, n_freqs + 1):
        sin_row = fourier_mode[2 * k - 1]
        cos_row = fourier_mode[2 * k] if 2 * k < prime else np.zeros_like(sin_row)
        freq_energy[k - 1] = np.sqrt(np.mean(sin_row ** 2 + cos_row ** 2))
    return freq_energy


# %% Load trajectories
print("Loading first-descent trajectories...")
trajectories: dict[tuple, tuple[list[int], dict[str, np.ndarray]]] = {}

for prime, mseed, dseeds in GROUPS:
    for ds in dseeds:
        v = family.get_variant(prime=prime, seed=mseed, data_seed=ds)
        epochs, weights = load_first_descent_trajectory(v)
        trajectories[(prime, mseed, ds)] = (epochs, weights)
        print(f"  p{prime}/seed{mseed}/dseed{ds}: {len(epochs)} epochs "
              f"({epochs[0]}–{epochs[-1]})")

print("Done.")


# %% Run DMD per variant, per matrix (W_in and W_E)
# Full-weight DMD is also computed for the singular value spectrum.
dmd_results: dict[tuple, dict[str, dict]] = {}  # (prime, mseed, ds) -> {matrix -> dmd_output}

MATRICES_FOR_DMD = ["W_in", "W_E", "full"]

for prime, mseed, dseeds in GROUPS:
    for ds in dseeds:
        epochs, weights = trajectories[(prime, mseed, ds)]
        dmd_results[(prime, mseed, ds)] = {}

        for matrix_name in MATRICES_FOR_DMD:
            if matrix_name == "full":
                # Concatenate all matrices into one flat state vector
                flat = np.concatenate(
                    [weights[k].reshape(len(epochs), -1) for k in WEIGHT_KEYS],
                    axis=1,
                )
            else:
                flat = weights[matrix_name].reshape(len(epochs), -1)

            result = compute_dmd(flat.astype(np.float64))
            dmd_results[(prime, mseed, ds)][matrix_name] = result


# %% Plot 1: Singular value spectra — W_in, W_E, and full weight vector
# Is first descent low-rank (sharp dropoff) or high-rank (flat spectrum)?
# Compare across data seeds within each group.

for prime, mseed, dseeds in GROUPS:
    fig_sv = make_subplots(
        rows=1, cols=3,
        subplot_titles=["W_in", "W_E", "Full weight vector"],
    )
    for matrix_idx, matrix_name in enumerate(MATRICES_FOR_DMD):
        for ds in dseeds:
            sv = dmd_results[(prime, mseed, ds)][matrix_name]["singular_values"]
            # Normalize by total energy for comparability
            sv_norm = sv ** 2 / (sv ** 2).sum()
            fig_sv.add_trace(
                go.Scatter(
                    x=list(range(1, len(sv_norm) + 1)),
                    y=sv_norm.tolist(),
                    mode="lines+markers",
                    name=f"dseed{ds}",
                    line=dict(color=COLORS[ds]),
                    marker=dict(size=5),
                    showlegend=(matrix_idx == 0),
                ),
                row=1, col=matrix_idx + 1,
            )

    fig_sv.update_layout(
        title=f"Singular value spectra — first descent (0–{FIRST_DESCENT_MAX_EPOCH}) "
              f"— p{prime}/seed{mseed}",
        height=400,
        xaxis_title="Rank",
        yaxis_title="Fractional energy",
    )
    fig_sv.show()


# %% Plot 2: DMD eigenvalue spectra — complex plane
# Growing modes (|λ| > 1): active reorganization.
# Decaying modes (|λ| < 1): convergence to structure.
# Oscillatory modes (|λ| ≈ 1): sustained dynamics.
# Sized by mode amplitude |α_i|.

for prime, mseed, dseeds in GROUPS:
    fig_eig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["W_in", "W_E"],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )
    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 200)
    for col in [1, 2]:
        fig_eig.add_trace(
            go.Scatter(
                x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
                mode="lines", line=dict(color="lightgray", dash="dot"),
                showlegend=False,
            ),
            row=1, col=col,
        )

    for matrix_idx, matrix_name in enumerate(["W_in", "W_E"]):
        for ds in dseeds:
            result = dmd_results[(prime, mseed, ds)][matrix_name]
            eigs = result["eigenvalues"]
            amps = np.abs(result["amplitudes"])
            # Scale marker size by amplitude (relative to max)
            sizes = 5 + 20 * (amps / amps.max())
            fig_eig.add_trace(
                go.Scatter(
                    x=eigs.real.tolist(),
                    y=eigs.imag.tolist(),
                    mode="markers",
                    name=f"dseed{ds}",
                    marker=dict(color=COLORS[ds], size=sizes.tolist(), opacity=0.7),
                    showlegend=(matrix_idx == 0),
                ),
                row=1, col=matrix_idx + 1,
            )

    fig_eig.update_layout(
        title=f"DMD eigenvalue spectra — first descent — p{prime}/seed{mseed}",
        height=450,
    )
    fig_eig.update_xaxes(title_text="Re(λ)", range=[-1.5, 1.5])
    fig_eig.update_yaxes(title_text="Im(λ)", range=[-1.5, 1.5], scaleanchor="x")
    fig_eig.show()


# %% Plot 3: Residual norms over the first-descent window
# High residual = dynamics are NOT well-approximated by linear step-to-step transitions.
# Spikes mark moments of rapid nonlinear change.
# Comparing across data seeds: does the fork appear in the residual signal?

for prime, mseed, dseeds in GROUPS:
    fig_res = make_subplots(
        rows=1, cols=2,
        subplot_titles=["W_in residuals", "W_E residuals"],
    )
    for matrix_idx, matrix_name in enumerate(["W_in", "W_E"]):
        for ds in dseeds:
            epochs, _ = trajectories[(prime, mseed, ds)]
            result = dmd_results[(prime, mseed, ds)][matrix_name]
            residuals = result["residual_norms"]
            # Residuals are between consecutive snapshot pairs: label at midpoint epoch
            step_epochs = [(epochs[i] + epochs[i + 1]) / 2 for i in range(len(residuals))]
            fig_res.add_trace(
                go.Scatter(
                    x=step_epochs,
                    y=residuals.tolist(),
                    mode="lines+markers",
                    name=f"dseed{ds}",
                    line=dict(color=COLORS[ds]),
                    marker=dict(size=5),
                    showlegend=(matrix_idx == 0),
                ),
                row=1, col=matrix_idx + 1,
            )

    fig_res.update_layout(
        title=f"DMD residual norms — first descent — p{prime}/seed{mseed}",
        height=400,
    )
    fig_res.update_xaxes(title_text="Epoch (midpoint)")
    fig_res.update_yaxes(title_text="Residual norm")
    fig_res.show()


# %% Plot 4: Fourier projection of dominant W_in modes
# For each variant, project the top-2 DMD modes (by amplitude) in W_in through
# the Fourier basis via W_E. Which frequencies does the dominant first-descent
# mode activate? Does this differ by data seed?

N_TOP_MODES = 2

for prime, mseed, dseeds in GROUPS:
    fourier_basis, _ = get_fourier_basis(prime)
    F_np = fourier_basis.numpy()
    n_freqs = prime // 2
    freqs = list(range(1, n_freqs + 1))

    fig_fourier_in = make_subplots(
        rows=N_TOP_MODES, cols=len(dseeds),
        subplot_titles=[
            f"dseed{ds} — mode {m + 1}"
            for m in range(N_TOP_MODES)
            for ds in dseeds
        ],
        shared_yaxes=True,
    )

    for ds_idx, ds in enumerate(dseeds):
        epochs, weights = trajectories[(prime, mseed, ds)]
        W_E_epoch0 = weights["W_E"][0].astype(np.float64)  # epoch 0 embedding
        result = dmd_results[(prime, mseed, ds)]["W_in"]
        modes = result["modes"]          # (state_dim, n_modes) complex
        amps = np.abs(result["amplitudes"])
        top_mode_indices = np.argsort(amps)[::-1][:N_TOP_MODES]

        for mode_rank, mode_idx in enumerate(top_mode_indices):
            mode_vec = np.abs(modes[:, mode_idx])  # magnitude of complex mode
            freq_energy = fourier_project_W_in_mode(
                mode_vec, W_E_epoch0, F_np, prime,
            )
            fig_fourier_in.add_trace(
                go.Bar(
                    x=freqs, y=freq_energy.tolist(),
                    marker_color=COLORS[ds],
                    name=f"dseed{ds} mode{mode_rank + 1}",
                    showlegend=False,
                ),
                row=mode_rank + 1, col=ds_idx + 1,
            )

    fig_fourier_in.update_layout(
        title=f"Fourier projection of dominant W_in DMD modes — first descent — p{prime}/seed{mseed}",
        height=200 * N_TOP_MODES + 100,
    )
    fig_fourier_in.update_xaxes(title_text="Frequency k")
    fig_fourier_in.update_yaxes(title_text="Energy")
    fig_fourier_in.show()


# %% Plot 5: Fourier projection of dominant W_E modes
# Same as Plot 4 but for W_E directly.

for prime, mseed, dseeds in GROUPS:
    fourier_basis, _ = get_fourier_basis(prime)
    F_np = fourier_basis.numpy()
    n_freqs = prime // 2
    freqs = list(range(1, n_freqs + 1))

    fig_fourier_E = make_subplots(
        rows=N_TOP_MODES, cols=len(dseeds),
        subplot_titles=[
            f"dseed{ds} — mode {m + 1}"
            for m in range(N_TOP_MODES)
            for ds in dseeds
        ],
        shared_yaxes=True,
    )

    for ds_idx, ds in enumerate(dseeds):
        epochs, weights = trajectories[(prime, mseed, ds)]
        result = dmd_results[(prime, mseed, ds)]["W_E"]
        modes = result["modes"]
        amps = np.abs(result["amplitudes"])
        top_mode_indices = np.argsort(amps)[::-1][:N_TOP_MODES]

        d_vocab, d_model = weights["W_E"].shape[1], weights["W_E"].shape[2]

        for mode_rank, mode_idx in enumerate(top_mode_indices):
            mode_vec = np.abs(modes[:, mode_idx])
            freq_energy = fourier_project_W_E_mode(
                mode_vec, F_np, prime, d_vocab, d_model,
            )
            fig_fourier_E.add_trace(
                go.Bar(
                    x=freqs, y=freq_energy.tolist(),
                    marker_color=COLORS[ds],
                    name=f"dseed{ds} mode{mode_rank + 1}",
                    showlegend=False,
                ),
                row=mode_rank + 1, col=ds_idx + 1,
            )

    fig_fourier_E.update_layout(
        title=f"Fourier projection of dominant W_E DMD modes — first descent — p{prime}/seed{mseed}",
        height=200 * N_TOP_MODES + 100,
    )
    fig_fourier_E.update_xaxes(title_text="Frequency k")
    fig_fourier_E.update_yaxes(title_text="Energy")
    fig_fourier_E.show()


# %% Plot 6: Token geometry in W_E — epoch 0 vs epoch 1500 vs dominant DMD mode
#
# The Fourier projection asks "which specific frequency does this mode prefer?"
# and found a flat spectrum. But flat ≠ unstructured. A mode building general
# circular/toroidal geometry — all tokens moving toward a ring arrangement —
# would look flat in the Fourier projection (no single-frequency preference)
# while still producing clear circular structure in token-space PCA.
#
# Three panels per data seed:
#   Left:   W_E at epoch 0 — initial token arrangement (random)
#   Middle: W_E at epoch 1500 — after first descent
#   Right:  Dominant DMD mode reshaped to (p, d_model) — the direction W_E moved
#
# Tokens colored by index 0..p-1 with a circular colormap. If the mode is
# building circular structure, tokens in the right panel will trace a ring.
# If it's noise, they'll scatter.

def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Reduce (n_points, d) to (n_points, 2) via SVD. Returns 2D coords."""
    centered = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T  # (n_points, 2)


def circular_colors(n: int) -> list[str]:
    """Generate n colors cycling through HSL hue for a circular colormap."""
    colors = []
    for i in range(n):
        hue = i / n
        # HSL to RGB: simple conversion for pure hue colors
        h6 = hue * 6
        x = 1 - abs(h6 % 2 - 1)
        if h6 < 1:   r, g, b = 1, x, 0
        elif h6 < 2: r, g, b = x, 1, 0
        elif h6 < 3: r, g, b = 0, 1, x
        elif h6 < 4: r, g, b = 0, x, 1
        elif h6 < 5: r, g, b = x, 0, 1
        else:        r, g, b = 1, 0, x
        colors.append(f"rgb({int(r*220)},{int(g*220)},{int(b*220)})")
    return colors


for prime, mseed, dseeds in GROUPS:
    # Rows = one per data seed, cols = epoch 0 | epoch 1500 | DMD mode.
    # Reading across a row: how did this seed's W_E evolve and what was the dominant mode?
    # Reading down a column: how do the seeds differ at the same stage?
    col_titles = ["W_E epoch 0", f"W_E epoch {FIRST_DESCENT_MAX_EPOCH}", "DMD mode 1"]
    row_titles = [f"dseed{ds}" for ds in dseeds]

    fig_geom = make_subplots(
        rows=len(dseeds), cols=3,
        subplot_titles=[
            f"dseed{ds} — {label}"
            for ds in dseeds
            for label in col_titles
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    token_colors = circular_colors(prime)

    for ds_idx, ds in enumerate(dseeds):
        epochs, weights = trajectories[(prime, mseed, ds)]
        d_vocab, d_model = weights["W_E"].shape[1], weights["W_E"].shape[2]

        W_E_start = weights["W_E"][0].astype(np.float64)[:prime]
        W_E_end   = weights["W_E"][-1].astype(np.float64)[:prime]

        result = dmd_results[(prime, mseed, ds)]["W_E"]
        modes = result["modes"]
        amps = np.abs(result["amplitudes"])
        top_idx = int(np.argsort(amps)[::-1][0])
        mode_vec = np.real(modes[:, top_idx]).reshape(d_vocab, d_model)[:prime]

        for col_idx, matrix in enumerate([W_E_start, W_E_end, mode_vec]):
            coords = pca_2d(matrix)
            fig_geom.add_trace(
                go.Scatter(
                    x=coords[:, 0].tolist(),
                    y=coords[:, 1].tolist(),
                    mode="markers",
                    marker=dict(color=token_colors, size=5, showscale=False),
                    text=[str(i) for i in range(prime)],
                    hovertemplate="token %{text}<extra></extra>",
                    showlegend=False,
                ),
                row=ds_idx + 1, col=col_idx + 1,
            )

    fig_geom.update_layout(
        title=f"Token geometry in W_E — epoch 0 vs {FIRST_DESCENT_MAX_EPOCH} vs dominant DMD mode "
              f"— p{prime}/seed{mseed}",
        height=300 * len(dseeds),
    )
    fig_geom.show()


# %% Plot 7: Token trajectory fan — W_E movement during first descent
#
# For each token, trace its W_E position across all first-descent checkpoints in PCA
# space. PCA basis is anchored to the epoch-1500 W_E so all trajectories are expressed
# in the "target frame" — the space where the final first-descent geometry is most visible.
#
# Key question: is first descent a radial fan (all tokens expanding outward at the same
# scale = confirms rank-1 weight-space result in activation space) or does angular
# differentiation appear (tokens sorting toward distinct angular positions = pre-organization)?
#
# Reading the plot:
#   Gray open circles = epoch 0 positions (all clustered near origin, random init)
#   Colored dots = epoch 1500 positions
#   Lines = trajectories; line color = token index via circular colormap
#
# Caveat: if PC1+PC2 variance fraction is low (< ~60%), the ring is distributing
# variance across 3+ components and a 2D projection misses structure. This is noted
# in each subplot title.

for prime, mseed, dseeds in GROUPS:
    token_colors = circular_colors(prime)

    # Pre-compute PCA projections per seed (variance needed for subplot titles)
    fan_data = {}
    for ds in dseeds:
        epochs, weights = trajectories[(prime, mseed, ds)]
        n_epochs = len(epochs)

        # Anchor PCA basis to the final checkpoint's W_E
        W_E_final = weights["W_E"][-1].astype(np.float64)[:prime]
        global_mean = W_E_final.mean(axis=0)
        centered_final = W_E_final - global_mean
        _, s_final, Vt_final = np.linalg.svd(centered_final, full_matrices=False)

        var_frac = float((s_final[:2] ** 2).sum() / (s_final ** 2).sum())

        traj = np.zeros((prime, n_epochs, 2))
        for ep_idx in range(n_epochs):
            W_E_ep = weights["W_E"][ep_idx].astype(np.float64)[:prime]
            traj[:, ep_idx, :] = (W_E_ep - global_mean) @ Vt_final[:2].T

        fan_data[ds] = {"traj": traj, "var_frac": var_frac, "epochs": epochs}

    subplot_titles = [
        f"dseed{ds} — PC1+PC2: {fan_data[ds]['var_frac']:.1%} of variance"
        for ds in dseeds
    ]

    fig_fan = make_subplots(
        rows=len(dseeds), cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )

    for ds_idx, ds in enumerate(dseeds):
        traj = fan_data[ds]["traj"]

        # One line trace per token (trajectory over first-descent epochs)
        for tok_idx in range(prime):
            fig_fan.add_trace(
                go.Scatter(
                    x=traj[tok_idx, :, 0].tolist(),
                    y=traj[tok_idx, :, 1].tolist(),
                    mode="lines",
                    line=dict(color=token_colors[tok_idx], width=1.5),
                    opacity=0.55,
                    showlegend=False,
                    hovertemplate=f"token {tok_idx}<extra></extra>",
                ),
                row=ds_idx + 1, col=1,
            )

        # Endpoint markers (colored by token, single batched trace)
        fig_fan.add_trace(
            go.Scatter(
                x=traj[:, -1, 0].tolist(),
                y=traj[:, -1, 1].tolist(),
                mode="markers",
                marker=dict(color=token_colors, size=6),
                showlegend=False,
                text=[str(t) for t in range(prime)],
                hovertemplate="token %{text} (final)<extra></extra>",
            ),
            row=ds_idx + 1, col=1,
        )

        # Start markers (all near origin at epoch 0 — gray open circles)
        fig_fan.add_trace(
            go.Scatter(
                x=traj[:, 0, 0].tolist(),
                y=traj[:, 0, 1].tolist(),
                mode="markers",
                marker=dict(color="lightgray", size=3, symbol="circle-open"),
                showlegend=False,
            ),
            row=ds_idx + 1, col=1,
        )

    fig_fan.update_layout(
        title=f"Token trajectory fan in W_E PCA space (epochs 0–{FIRST_DESCENT_MAX_EPOCH}) "
              f"— p{prime}/seed{mseed}",
        height=380 * len(dseeds),
    )
    fig_fan.update_xaxes(title_text="PC1")
    fig_fan.update_yaxes(title_text="PC2")
    fig_fan.show()

# %%
