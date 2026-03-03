"""REQ_051: DMD visualization renderers.

Three views for centroid DMD analysis:
- render_dmd_eigenvalues:    Eigenvalue spectrum on the complex plane
- render_dmd_residual:       Per-step residual norm over training
- render_dmd_reconstruction: Actual vs DMD-reconstructed centroid trajectories
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope.analysis.library.dmd import dmd_reconstruct

_SITE_LABELS = {
    "resid_pre": "Post-Embed",
    "attn_out": "Attn Out",
    "mlp_out": "MLP Out",
    "resid_post": "Resid Post",
}

_ALL_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]

_SITE_COLORS = {
    "resid_pre": "rgba(31, 119, 180, 1.0)",
    "attn_out": "rgba(44, 160, 44, 1.0)",
    "mlp_out": "rgba(214, 39, 40, 1.0)",
    "resid_post": "rgba(148, 103, 189, 1.0)",
}


def render_dmd_eigenvalues(
    cross_epoch_data: dict[str, np.ndarray],
    site: str = "resid_post",
    height: int = 500,
) -> go.Figure:
    """DMD eigenvalue spectrum on the complex plane.

    Eigenvalues on the unit circle are purely oscillatory. Inside → decaying
    transients. Outside → growing instabilities. Marker size scales with mode
    amplitude magnitude |α_i|, identifying the dynamically dominant modes.

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("centroid_dmd").
        site: Activation site to display.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with unit circle and eigenvalue scatter.
    """
    eigenvalues = cross_epoch_data[f"{site}__eigenvalues"].astype(np.complex128)
    amplitudes = cross_epoch_data[f"{site}__amplitudes"]

    amp_magnitude = np.abs(amplitudes)
    max_amp = amp_magnitude.max() if amp_magnitude.max() > 0 else 1.0
    normalized_amp = amp_magnitude / max_amp

    theta = np.linspace(0, 2 * np.pi, 200)

    fig = go.Figure()

    # Unit circle
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="rgba(0,0,0,0.3)", width=1.5, dash="dash"),
            name="Unit circle",
            hoverinfo="skip",
        )
    )

    # Eigenvalue scatter
    fig.add_trace(
        go.Scatter(
            x=eigenvalues.real,
            y=eigenvalues.imag,
            mode="markers",
            marker=dict(
                size=6 + 14 * normalized_amp,
                color=amp_magnitude,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="|α|", x=1.02),
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            ),
            text=[
                f"λ={ev.real:.3f}+{ev.imag:.3f}j<br>|α|={a:.3f}"
                for ev, a in zip(eigenvalues, amp_magnitude)
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Eigenvalues",
        )
    )

    # Origin
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(size=4, color="black", symbol="cross"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    site_label = _SITE_LABELS.get(site, site)
    n_modes = int(cross_epoch_data.get(f"{site}__n_modes", len(eigenvalues)))
    fig.update_layout(
        title=f"DMD Eigenvalue Spectrum — {site_label} ({n_modes} modes)",
        xaxis_title="Re(λ)",
        yaxis_title="Im(λ)",
        xaxis=dict(scaleanchor="y", scaleratio=1, zeroline=True, zerolinewidth=1),
        yaxis=dict(zeroline=True, zerolinewidth=1),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=80, t=60, b=50),
    )

    return fig


def render_dmd_residual(
    cross_epoch_data: dict[str, np.ndarray],
    site: str | None = None,
    current_epoch: int | None = None,
    log_y: bool = False,
    height: int = 400,
) -> go.Figure:
    """DMD residual norm over training — the primary grokking onset signal.

    Each point is the step-ahead prediction error: how well the linear DMD
    operator predicts the next centroid state from the current one. Large
    residuals indicate where linear dynamics break down — the candidate
    grokking transition window.

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("centroid_dmd").
        site: Single site to display. None = overlay all four sites.
        current_epoch: Epoch for optional vertical indicator.
        log_y: Use log scale on the y-axis. Reveals mid-training structure
            that is otherwise swamped by the large early-training spike.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with residual norm time series.
    """
    epochs = cross_epoch_data["epochs"]
    # Residual at step t corresponds to the transition epoch[t] → epoch[t+1].
    # We label residuals by the "from" epoch.
    residual_epochs = epochs[:-1]

    sites = [site] if site else _ALL_SITES

    fig = go.Figure()

    for s in sites:
        key = f"{s}__residual_norms"
        if key not in cross_epoch_data:
            continue
        residuals = cross_epoch_data[key]
        color = _SITE_COLORS.get(s, "gray")
        label = _SITE_LABELS.get(s, s)

        fig.add_trace(
            go.Scatter(
                x=residual_epochs,
                y=residuals,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=3),
                hovertemplate=(f"{label}<br>Epoch %{{x}}<br>Residual: %{{y:.4f}}<extra></extra>"),
            )
        )

    if current_epoch is not None:
        fig.add_vline(
            x=current_epoch,
            line_dash="solid",
            line_color="red",
            line_width=1,
        )

    yaxis_cfg: dict = dict(title="Residual Norm")
    if log_y:
        yaxis_cfg["type"] = "log"

    fig.update_layout(
        title="DMD Residual Norm — Centroid Trajectory Prediction Error",
        xaxis_title="Epoch",
        yaxis=yaxis_cfg,
        template="plotly_white",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig


def render_dmd_reconstruction(
    cross_epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    height: int = 500,
) -> go.Figure:
    """Actual vs DMD-reconstructed centroid trajectories in global PCA space.

    Shows PC1 vs PC2 positions of class centroids at the given epoch, comparing
    the actual positions from global PCA to the DMD reconstruction. Divergence
    between actual and reconstructed indicates where the linear dynamics model
    breaks down.

    Args:
        cross_epoch_data: From ArtifactLoader.load_cross_epoch("centroid_dmd").
        epoch: Epoch number to highlight (nearest available epoch selected).
        site: Activation site to display.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 2×1 subplots: PC1-PC2 scatter + trajectory overlay.
    """
    epochs = cross_epoch_data["epochs"]
    epoch_arr = np.array(epochs)
    epoch_idx = int(np.argmin(np.abs(epoch_arr - epoch)))
    actual_epoch = int(epochs[epoch_idx])
    n_epochs = len(epochs)

    trajectory = cross_epoch_data[f"{site}__trajectory"]  # (n_epochs, state_dim)
    eigenvalues = cross_epoch_data[f"{site}__eigenvalues"]
    modes = cross_epoch_data[f"{site}__modes"]
    amplitudes = cross_epoch_data[f"{site}__amplitudes"]
    n_classes = int(cross_epoch_data[f"{site}__n_classes"])

    reconstruction = dmd_reconstruct(eigenvalues, modes, amplitudes, n_epochs)

    # Reshape both to (n_epochs, n_classes, n_components)
    state_dim = trajectory.shape[1]
    n_components = state_dim // n_classes
    actual = trajectory.reshape(n_epochs, n_classes, n_components)
    recon = reconstruction.reshape(n_epochs, n_classes, n_components)

    residues = np.arange(n_classes)
    labels = [str(r) for r in residues]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"PC1 vs PC2 — Epoch {actual_epoch}",
            "PC1 Trajectory (actual vs DMD)",
        ],
        horizontal_spacing=0.10,
    )

    # Left panel: centroid scatter at selected epoch
    for label_prefix, data, symbol, dash in [
        ("Actual", actual[epoch_idx], "circle", None),
        ("DMD", recon[epoch_idx], "x", None),
    ]:
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1] if n_components >= 2 else np.zeros(n_classes),
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=residues,
                    colorscale="HSV",
                    showscale=False,
                    symbol=symbol,
                    line=dict(width=1, color="rgba(0,0,0,0.3)"),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=6),
                name=label_prefix,
                hovertemplate=(
                    f"{label_prefix} — Residue %{{text}}<br>"
                    "PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    # Constrain scatter axes to actual data — DMD divergence doesn't corrupt scale.
    actual_at_epoch = actual[epoch_idx]  # (n_classes, n_components)
    _x = actual_at_epoch[:, 0]
    _y = actual_at_epoch[:, 1] if n_components >= 2 else np.zeros(n_classes)
    _xpad = max(float(np.abs(_x).max()) * 0.2, 0.1)
    _ypad = max(float(np.abs(_y).max()) * 0.2, 0.1)
    fig.update_xaxes(title_text="PC1", range=[_x.min() - _xpad, _x.max() + _xpad], row=1, col=1)
    fig.update_yaxes(title_text="PC2", range=[_y.min() - _ypad, _y.max() + _ypad], row=1, col=1)

    # Right panel: PC1 trajectory over time for a sample of classes
    sample_classes = _sample_classes(n_classes, max_classes=5)
    for cls_idx in sample_classes:
        pc1_actual = actual[:, cls_idx, 0]
        pc1_recon = recon[:, cls_idx, 0]
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=pc1_actual,
                mode="lines",
                name=f"cls {cls_idx} actual",
                legendgroup=f"cls{cls_idx}",
                line=dict(width=2),
                hovertemplate=f"Class {cls_idx} actual<br>Epoch %{{x}}<br>PC1: %{{y:.3f}}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=list(epochs),
                y=pc1_recon,
                mode="lines",
                name=f"cls {cls_idx} DMD",
                legendgroup=f"cls{cls_idx}",
                line=dict(width=2, dash="dash"),
                hovertemplate=f"Class {cls_idx} DMD<br>Epoch %{{x}}<br>PC1: %{{y:.3f}}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=2,
        )
    # Mark selected epoch on trajectory panel
    fig.add_vline(
        x=actual_epoch,
        line_dash="dot",
        line_color="red",
        line_width=1,
        row=1,  # type: ignore[reportArgumentType]
        col=2,  # type: ignore[reportArgumentType]
    )
    # Constrain trajectory axis to actual PC1 range across all epochs.
    _pc1_all = actual[:, :, 0].flatten()
    _pc1_pad = max(float(np.abs(_pc1_all).max()) * 0.2, 0.1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(
        title_text="PC1", range=[_pc1_all.min() - _pc1_pad, _pc1_all.max() + _pc1_pad], row=1, col=2
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=f"DMD Reconstruction — {site_label}",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=80, b=50),
        legend=dict(orientation="v", x=1.01),
    )

    return fig


def _sample_classes(n_classes: int, max_classes: int = 5) -> list[int]:
    """Select a representative sample of class indices for trajectory plots."""
    if n_classes <= max_classes:
        return list(range(n_classes))
    step = n_classes // max_classes
    return list(range(0, n_classes, step))[:max_classes]
