"""REQ_066: Multi-Stream Frequency Specialization Trajectory.

Four-panel view comparing frequency specialization accumulation across
embedding dimensions, attention heads, MLP neurons, and effective
dimensionality over training.

Each panel shares the same x-axis (epochs) and per-frequency color scheme
so a single frequency can be tracked visually across all three specialization
streams. An epoch cursor (vertical line) marks the currently selected epoch.
"""

import colorsys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope.visualization.renderers.effective_dimensionality import _MATRIX_COLORS

# Which ED matrices to include in panel 4
_ED_MATRICES = ["W_E", "W_in", "W_out", "W_O"]


def _freq_color(k: int, n_freq: int) -> str:
    """Return a consistent RGB color string for frequency k (1-indexed)."""
    hue = (k - 1) / n_freq
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def _compute_mlp_band_counts(
    neuron_dynamics: dict[str, np.ndarray],
    prime: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute per-frequency committed neuron counts from neuron_dynamics."""
    epochs = neuron_dynamics["epochs"]
    dominant_freq = neuron_dynamics["dominant_freq"]  # (n_epochs, d_mlp)
    max_frac = neuron_dynamics["max_frac"]  # (n_epochs, d_mlp)

    n_freq = prime // 2
    committed = max_frac >= threshold
    band_counts = np.zeros((len(epochs), n_freq), dtype=np.int32)
    for k in range(n_freq):
        band_counts[:, k] = np.sum(committed & (dominant_freq == k), axis=1)

    active = [k for k in range(n_freq) if band_counts[:, k].max() > 0]
    return epochs, band_counts, active


def _compute_attn_aggregate(
    attn_fourier_epochs: dict[str, np.ndarray],
    attn_floor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Compute mean QK^T Fourier fraction across heads per frequency per epoch."""
    epochs = attn_fourier_epochs["epochs"]
    qk_freq_norms = attn_fourier_epochs["qk_freq_norms"]  # (n_epochs, n_heads, n_freq)
    mean_qk = qk_freq_norms.mean(axis=1)  # (n_epochs, n_freq)

    # Show frequencies that reach at least attn_floor aggregate commitment at any epoch
    active = [k for k in range(mean_qk.shape[1]) if mean_qk[:, k].max() > attn_floor]
    return epochs, mean_qk, active


def _compute_embedding_dim_counts(
    embedding_w_e: dict[str, np.ndarray],
    prime: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Count d_model dimensions specializing in each frequency per epoch.

    For each d_model dimension j and frequency k, computes the fraction of
    that dimension's cross-token variance explained by frequency k using the
    standard Fourier decomposition of the embedding matrix along the vocab axis.
    """
    epochs = embedding_w_e["epochs"]
    W_E = embedding_w_e["W_E"]  # (n_snap, p, d_model)
    n_snap, p, d_model = W_E.shape
    n_freq = prime // 2

    k_vals = np.arange(1, n_freq + 1)
    n_vals = np.arange(p)
    # Fourier basis: (n_freq, p)
    F_cos = np.cos(2 * np.pi * k_vals[:, np.newaxis] * n_vals[np.newaxis, :] / p)
    F_sin = np.sin(2 * np.pi * k_vals[:, np.newaxis] * n_vals[np.newaxis, :] / p)

    # Projections: (n_snap, d_model, n_freq)
    W_E_t = W_E.transpose(0, 2, 1)  # (n_snap, d_model, p)
    cos_proj = W_E_t @ F_cos.T  # (n_snap, d_model, n_freq)
    sin_proj = W_E_t @ F_sin.T

    # Power per freq with Parseval normalization: (n_snap, d_model, n_freq)
    power_per_freq = (2.0 / p) * (cos_proj**2 + sin_proj**2)

    # Total power per (epoch, dim) = sum over freqs + DC
    total_power = power_per_freq.sum(axis=2)  # (n_snap, d_model)
    dc_power = (W_E.sum(axis=1) ** 2) / p  # (n_snap, d_model)
    total_power += dc_power

    # Fraction of each dim's variance in each freq: (n_snap, n_freq, d_model)
    power_per_freq_t = power_per_freq.transpose(0, 2, 1)
    frac_power = power_per_freq_t / (total_power[:, np.newaxis, :] + 1e-10)

    # Count dims over threshold: (n_snap, n_freq)
    dim_counts = (frac_power > threshold).sum(axis=2).astype(np.int32)

    active = [k for k in range(n_freq) if dim_counts[:, k].max() > 0]
    return epochs, dim_counts, active


def render_multi_stream_specialization(
    data: dict,
    epoch: int | None,
    threshold_mlp: float = 0.5,
    threshold_embedding: float = 0.5,
    attn_floor: float = 0.02,
    title: str | None = None,
    height: int = 1400,
    width: int = 950,
) -> go.Figure:
    """Four-panel multi-stream frequency specialization trajectory.

    Panels (top to bottom):
      1. MLP: committed neuron count per frequency (threshold_mlp)
      2. Attention: mean QK^T Fourier fraction across heads per frequency
      3. Embedding: d_model dimension count per frequency (threshold_embedding)
      4. Effective Dimensionality: participation ratio for W_E, W_in, W_out, W_O

    All panels share an x-axis. A vertical cursor marks the selected epoch.
    Per-frequency colors are consistent across panels 1-3.

    Args:
        data: Dict with keys:
            neuron_dynamics   — cross_epoch dict from neuron_dynamics analyzer
            attn_fourier_epochs — stacked dict from attention_fourier analyzer
            embedding_w_e     — {"epochs", "W_E"} from parameter_snapshot (W_E only)
            eff_dim_summary   — summary dict from effective_dimensionality analyzer
            prime             — int, the modulus p
        epoch: Current epoch for vertical cursor. None omits the cursor.
        threshold_mlp: Neuron commitment threshold for panel 1.
        threshold_embedding: Dimension commitment threshold for panel 3.
        attn_floor: Minimum peak aggregate commitment for a frequency to appear
            in the attention panel. Filters noise at low values; raise to show
            only frequencies with meaningful head commitment (e.g., 0.10–0.20).
        title: Custom figure title.
        height: Total figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with four vertically stacked subplots.
    """
    prime = data["prime"]
    n_freq = prime // 2

    mlp_epochs, mlp_counts, mlp_active = _compute_mlp_band_counts(
        data["neuron_dynamics"], prime, threshold_mlp
    )
    attn_epochs, attn_mean_qk, attn_active = _compute_attn_aggregate(
        data["attn_fourier_epochs"], attn_floor=attn_floor
    )
    emb_epochs, emb_counts, emb_active = _compute_embedding_dim_counts(
        data["embedding_w_e"], prime, threshold_embedding
    )

    # all_active = sorted(set(mlp_active) | set(attn_active) | set(emb_active))

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"MLP — Committed Neurons per Frequency (threshold={int(threshold_mlp * 100)}%)",
            "Attention — Mean Head Commitment per Frequency",
            f"Embedding Dims — Specialized per Frequency (threshold={int(threshold_embedding * 100)}%)",
            "Effective Dimensionality (Participation Ratio)",
        ],
        row_heights=[0.28, 0.22, 0.22, 0.22],
    )

    # Track which frequencies have been added to the legend (show only once)
    legend_shown: set[int] = set()

    def _add_freq_trace(row: int, x, y, k_zero_indexed: int) -> None:
        k1 = k_zero_indexed + 1  # 1-indexed display label
        color = _freq_color(k1, n_freq)
        show = k_zero_indexed not in legend_shown
        if show:
            legend_shown.add(k_zero_indexed)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Freq {k1}",
                legendgroup=f"freq_{k_zero_indexed}",
                showlegend=show,
                line=dict(color=color, width=1.5),
                hovertemplate=f"Freq {k1}<br>Epoch %{{x}}<br>%{{y}}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    # Panel 1: MLP
    for k in mlp_active:
        _add_freq_trace(1, mlp_epochs, mlp_counts[:, k], k)

    # Panel 2: Attention (attn_active uses 0-indexed k relative to n_freq)
    for k in attn_active:
        _add_freq_trace(2, attn_epochs, attn_mean_qk[:, k], k)

    # Panel 3: Embedding
    for k in emb_active:
        _add_freq_trace(3, emb_epochs, emb_counts[:, k], k)

    # Panel 4: Effective Dimensionality
    eff_dim = data["eff_dim_summary"]
    ed_epochs = eff_dim["epochs"]
    for name in _ED_MATRICES:
        pr_key = f"pr_{name}"
        if pr_key not in eff_dim:
            continue
        pr_data = eff_dim[pr_key]
        if pr_data.ndim == 2:
            pr_values = pr_data.mean(axis=1)
            display_name = f"{name} (mean)"
        else:
            pr_values = pr_data
            display_name = name
        fig.add_trace(
            go.Scatter(
                x=ed_epochs,
                y=pr_values,
                mode="lines",
                name=display_name,
                legendgroup=f"ed_{name}",
                showlegend=True,
                line=dict(color=_MATRIX_COLORS.get(name, "gray"), width=2),
                hovertemplate=f"{display_name}<br>Epoch %{{x}}<br>PR: %{{y:.1f}}<extra></extra>",
            ),
            row=4,
            col=1,
        )

    # Epoch cursor across all panels
    if epoch is not None:
        fig.add_vline(
            x=epoch,
            line_dash="solid",
            line_color="rgba(180, 0, 0, 0.6)",
            line_width=1.5,
        )

    threshold_mlp_pct = int(threshold_mlp * 100)
    threshold_emb_pct = int(threshold_embedding * 100)
    if title is None:
        title = (
            f"Multi-Stream Specialization — p={prime} "
            f"(MLP thr={threshold_mlp_pct}%, Emb thr={threshold_emb_pct}%)"
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        width=width,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=10),
            tracegroupgap=8,
        ),
        margin=dict(l=60, r=160, t=60, b=50),
    )

    fig.update_yaxes(title_text="Neuron Count", row=1, col=1)
    fig.update_yaxes(title_text="Mean Commitment", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Dim Count", row=3, col=1)
    fig.update_yaxes(title_text="Participation Ratio", row=4, col=1)
    fig.update_xaxes(title_text="Epoch", row=4, col=1)

    return fig
