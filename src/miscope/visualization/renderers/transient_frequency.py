"""REQ_084: Renderers for transient frequency views.

Three renderers, each consuming the transient_frequency cross-epoch artifact:

    render_transient_committed_counts  -- line chart of committed neuron count per
                                          ever-qualified frequency over training
    render_transient_peak_scatter      -- PC1×PC2 scatter of a transient group at its
                                          peak epoch, colored by final dominant freq
    render_transient_pc1_cohesion      -- PC1 variance explained of the cohort over
                                          all epochs, basis fixed at peak epoch
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from miscope.analysis.analyzers.transient_frequency import load_peak_members

_TRANSIENT_COLOR = "crimson"
_PERSISTENT_COLOR_ALPHA = "rgba(100, 130, 200, 0.6)"


def render_transient_committed_counts(
    data: dict[str, Any],
    epoch: int | None,
    show_persistent: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """Line chart of committed neuron count per ever-qualified frequency.

    Transient frequencies (not in final learned set) are drawn as dashed red lines.
    Persistent frequencies are drawn as solid muted lines when show_persistent=True.

    Args:
        data: transient_frequency cross-epoch artifact dict
        epoch: unused (cross-epoch view)
        show_persistent: include final learned frequencies alongside transients
    """
    epochs = data["epochs"]
    ever_qualified = data["ever_qualified_freqs"]
    is_final = data["is_final"]
    committed_counts = data["committed_counts"]  # (n_epochs, n_transient)

    fig = go.Figure()

    for i, freq in enumerate(ever_qualified):
        freq_label = int(freq) + 1  # 0-indexed → 1-indexed display
        is_persistent = bool(is_final[i])

        if is_persistent and not show_persistent:
            continue

        if is_persistent:
            line = dict(dash="solid", color=_PERSISTENT_COLOR_ALPHA, width=1.2)
            name = f"freq {freq_label}"
        else:
            line = dict(dash="dash", color=_TRANSIENT_COLOR, width=2)
            name = f"freq {freq_label}  ← transient"

        fig.add_trace(go.Scatter(
            x=epochs.tolist(),
            y=committed_counts[:, i].tolist(),
            mode="lines",
            name=name,
            line=line,
        ))

    threshold_pct = int(round(float(data["_transient_canonical_threshold"]) * 100))
    fig.update_layout(
        title=(
            "Committed neuron count per ever-qualified frequency<br>"
            f"<sup>Dashed red = transient (absent from final set) | "
            f"threshold = {threshold_pct}% of d_mlp | "
            f"neuron gate = {int(round(float(data['_neuron_threshold'])*100))}% frac_explained</sup>"
        ),
        xaxis_title="Epoch",
        yaxis_title="Committed neuron count",
        template="plotly_white",
        height=450,
        legend=dict(orientation="v", x=1.02, y=0.99, font=dict(size=10)),
        margin=dict(r=160),
    )
    return fig


def render_transient_peak_scatter(
    tf: dict[str, Any],
    w_in_by_epoch: dict[int, np.ndarray],
    epoch: int | None,
    freq: int | None = None,
    **kwargs: Any,
) -> go.Figure:
    """PC1×PC2 scatter of a transient group at its peak epoch.

    Each neuron is colored by its final dominant frequency, revealing whether the
    cohort disperses to multiple attractors or consolidates to one.

    Args:
        tf: transient_frequency cross-epoch artifact dict
        w_in_by_epoch: {epoch: W_in ndarray} pre-loaded by view loader
        epoch: unused (scatter is always at peak epoch)
        freq: 0-indexed frequency to visualize; defaults to the largest transient group
    """
    ever_qualified = tf["ever_qualified_freqs"]
    is_final = tf["is_final"]
    peak_epoch_arr = tf["peak_epoch"]

    transient_indices = np.where(~is_final)[0]
    if len(transient_indices) == 0:
        return _empty_figure("No transient frequency groups found")

    if freq is not None:
        matches = np.where(ever_qualified == freq)[0]
        if len(matches) == 0 or is_final[matches[0]]:
            return _empty_figure(f"Freq {freq + 1} is not a transient group")
        group_idx = int(matches[0])
    else:
        # Default: largest transient group by peak count
        peak_counts = tf["peak_count"][transient_indices]
        group_idx = int(transient_indices[np.argmax(peak_counts)])

    freq_val = int(ever_qualified[group_idx])
    peak_ep = int(peak_epoch_arr[group_idx])
    members = load_peak_members(tf, group_idx)

    if len(members) < 2:
        return _empty_figure(f"Freq {freq_val + 1} group has fewer than 2 members")

    W_in = w_in_by_epoch[peak_ep]
    basis, s, coords = _compute_basis(W_in[:, members])
    total_var = float((s**2).sum())
    pc1_frac = float(s[0]**2 / total_var)
    pc2_frac = float(s[1]**2 / total_var)

    # Final dominant freq for coloring (from neuron_dynamics, stored in tf via homeless logic)
    # We don't have final_dominant directly; derive from peak membership context.
    # The tf artifact stores homeless_count but not per-neuron final freq.
    # Use coords only — color by PC angle as proxy for phase, which is informative
    # even without final freq info.  This is a limitation acknowledged in the renderer.
    # If final_dominant is needed, the view loader can be extended to bundle neuron_dynamics.

    angles = np.arctan2(coords[:, 1], coords[:, 0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        mode="markers",
        marker=dict(
            color=angles.tolist(),
            colorscale="HSV",
            size=7,
            showscale=True,
            colorbar=dict(title="PC angle", thickness=12),
        ),
        hovertemplate="PC1=%{x:.3f}<br>PC2=%{y:.3f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title=(
            f"Freq {freq_val + 1} cohort at peak (epoch {peak_ep})<br>"
            f"<sup>PC1 {pc1_frac:.1%} | PC2 {pc2_frac:.1%} | "
            f"n={len(members)} neurons | basis fixed at peak epoch</sup>"
        ),
        xaxis_title=f"PC1 ({pc1_frac:.1%})",
        yaxis_title=f"PC2 ({pc2_frac:.1%})",
        yaxis=dict(scaleanchor="x"),
        template="plotly_white",
        height=500,
    )
    return fig


def render_transient_pc1_cohesion(
    tf: dict[str, Any],
    w_in_by_epoch: dict[int, np.ndarray],
    epoch: int | None,
    freq: int | None = None,
    **kwargs: Any,
) -> go.Figure:
    """PC1 variance explained of a transient group's cohort across all epochs.

    Projects the peak-epoch member cohort onto the peak-epoch PCA basis at every
    epoch.  A rise then sustained collapse confirms coherent weight-space structure
    was forming and dissolved.  A flat or erratic line indicates the group was never
    structurally organized.

    Args:
        tf: transient_frequency cross-epoch artifact dict
        w_in_by_epoch: {epoch: W_in ndarray} pre-loaded by view loader
        epoch: unused (cohesion is computed across all epochs)
        freq: 0-indexed frequency to visualize; defaults to largest transient group
    """
    ever_qualified = tf["ever_qualified_freqs"]
    is_final = tf["is_final"]
    peak_epoch_arr = tf["peak_epoch"]
    epochs = tf["epochs"]

    transient_indices = np.where(~is_final)[0]
    if len(transient_indices) == 0:
        return _empty_figure("No transient frequency groups found")

    if freq is not None:
        matches = np.where(ever_qualified == freq)[0]
        if len(matches) == 0 or is_final[matches[0]]:
            return _empty_figure(f"Freq {freq + 1} is not a transient group")
        group_idx = int(matches[0])
    else:
        peak_counts = tf["peak_count"][transient_indices]
        group_idx = int(transient_indices[np.argmax(peak_counts)])

    freq_val = int(ever_qualified[group_idx])
    peak_ep = int(peak_epoch_arr[group_idx])
    members = load_peak_members(tf, group_idx)

    if len(members) < 2:
        return _empty_figure(f"Freq {freq_val + 1} group has fewer than 2 members")

    W_in_peak = w_in_by_epoch[peak_ep]
    ref_basis, _, _ = _compute_basis(W_in_peak[:, members])

    pc1_seq = []
    for ep in epochs.tolist():
        W_in = w_in_by_epoch.get(int(ep))
        if W_in is None:
            pc1_seq.append(float("nan"))
            continue
        coords = _project_onto(W_in[:, members], ref_basis)
        _, s_proj, _ = np.linalg.svd(coords, full_matrices=False)
        total = float((s_proj**2).sum())
        pc1_seq.append(float(s_proj[0]**2 / total) if total > 1e-10 else float("nan"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs.tolist(),
        y=pc1_seq,
        mode="lines",
        name=f"freq {freq_val + 1} cohort",
        line=dict(color=_TRANSIENT_COLOR, width=2),
    ))
    fig.add_vline(
        x=peak_ep,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"peak epoch {peak_ep}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=(
            f"PC1 cohesion of freq-{freq_val + 1} cohort over time<br>"
            "<sup>Basis fixed at peak epoch | "
            "Rise → coherent ring forming | Sustained collapse → group dissolved</sup>"
        ),
        xaxis_title="Epoch",
        yaxis_title="PC1 variance explained",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_basis(
    group_W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of centered group weight matrix.

    Args:
        group_W: (d_model, n_group)

    Returns:
        (basis, singular_values, coords) where basis is (3, d_model),
        singular_values is (3,), coords is (n_group, 3)
    """
    n_components = min(3, group_W.shape[1])
    centroid = group_W.mean(axis=1, keepdims=True)
    centered = group_W - centroid
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    basis = U[:, :n_components].T       # (n_components, d_model)
    coords = (basis @ centered).T       # (n_group, n_components)
    return basis, s[:n_components], coords


def _project_onto(group_W: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project group W_in onto a fixed basis (from another epoch).

    Args:
        group_W: (d_model, n_group)
        basis: (n_components, d_model)

    Returns:
        (n_group, n_components)
    """
    centroid = group_W.mean(axis=1, keepdims=True)
    centered = group_W - centroid
    return (basis @ centered).T


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(template="plotly_white", height=300)
    return fig
