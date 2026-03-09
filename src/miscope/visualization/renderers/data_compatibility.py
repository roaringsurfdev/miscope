"""REQ_064: Fourier Data Compatibility Visualizations.

Two renderers for how well each Fourier frequency is supported by the
training data split:

- render_data_compatibility_spectrum: Bar chart of compatibility score per
  frequency with component breakdown (phase uniformity, condition score).

- render_data_compatibility_overlap: Synthesizes REQ_063 init energy with
  REQ_064 compatibility to predict nucleation winners. Degrades gracefully
  when the nucleation artifact is absent.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_data_compatibility_spectrum(
    epoch_data: dict,
    epoch: int | None,
    **kwargs,
) -> go.Figure:
    """Bar chart of per-frequency data compatibility with component breakdown.

    Args:
        epoch_data: Dict from compute_data_compatibility() with keys:
            frequencies, compatibility_score, condition_score,
            phase_uniformity, prime, data_seed.
        epoch: Unused (epoch-independent view). Present for renderer protocol.
        **kwargs: Unused.

    Returns:
        Plotly Figure with compatibility bars and component overlay lines.
    """
    frequencies = epoch_data["frequencies"]
    compatibility = epoch_data["compatibility_score"]
    condition_score = epoch_data["condition_score"]
    phase_uniformity = epoch_data["phase_uniformity"]
    prime = int(epoch_data["prime"])
    data_seed = int(epoch_data["data_seed"])

    freq_labels = frequencies.tolist()
    bar_colors = _compatibility_colors(compatibility)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=freq_labels,
            y=compatibility.tolist(),
            marker_color=bar_colors,
            name="Compatibility",
            hovertemplate="k=%{x}<br>Compatibility: %{y:.3f}<extra></extra>",
            opacity=0.85,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_labels,
            y=condition_score.tolist(),
            mode="lines",
            name="Condition Score",
            line=dict(color="#50b0e0", width=1.5, dash="dot"),
            hovertemplate="k=%{x}<br>Condition Score: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_labels,
            y=phase_uniformity.tolist(),
            mode="lines",
            name="Phase Uniformity",
            line=dict(color="#a060e0", width=1.5, dash="dash"),
            hovertemplate="k=%{x}<br>Phase Uniformity: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=dict(
            text=f"Data Compatibility Spectrum — p={prime}, data_seed={data_seed}",
            x=0.5,
        ),
        xaxis_title="Frequency k",
        yaxis_title="Score [0, 1]",
        height=400,
        template="plotly_dark",
        plot_bgcolor="#0a1010",
        paper_bgcolor="#060a0a",
        font=dict(family="IBM Plex Mono, monospace", size=11),
        legend=dict(x=1.0, y=0.99, xanchor="left"),
        barmode="overlay",
    )

    return fig


def render_data_compatibility_overlap(
    epoch_data: dict,
    epoch: int | None,
    **kwargs,
) -> go.Figure:
    """Overlap view synthesizing init energy (REQ_063) with data compatibility.

    Predicts nucleation winners: frequencies where the model is biased at
    initialization AND the training data provides good coverage.

    Degrades gracefully when the nucleation artifact is absent — shows
    compatibility only.

    Args:
        epoch_data: Dict with keys:
            compatibility: output of compute_data_compatibility()
            nucleation: epoch-0 fourier_nucleation artifact (or None)
        epoch: Unused (epoch-independent view). Present for renderer protocol.
        **kwargs: Unused.

    Returns:
        Plotly Figure with overlap analysis.
    """
    compat = epoch_data["compatibility"]
    nucleation = epoch_data.get("nucleation")

    frequencies = compat["frequencies"]
    compatibility = compat["compatibility_score"]
    prime = int(compat["prime"])
    data_seed = int(compat["data_seed"])
    freq_labels = frequencies.tolist()

    has_nucleation = nucleation is not None
    if has_nucleation:
        agg_energy = nucleation["aggregate_energy"]  # (n_iters+1, n_freqs)
        init_energy = agg_energy[-1]  # Use sharpened final iteration
        nuc_freqs = nucleation["frequencies"]
        # Align frequencies — both should match, but guard against mismatch
        init_energy = _align_to_frequencies(init_energy, nuc_freqs, frequencies)
        overlap_score = init_energy * compatibility
    else:
        init_energy = np.zeros(len(frequencies), dtype=np.float32)
        overlap_score = np.zeros(len(frequencies), dtype=np.float32)

    top_quartile_mask = _top_quartile_mask(overlap_score) if has_nucleation else None

    # Color bars by overlap score when available, else by compatibility
    if has_nucleation:
        bar_colors = _overlap_colors(overlap_score, top_quartile_mask)
    else:
        bar_colors = _compatibility_colors(compatibility)

    fig = go.Figure()

    if has_nucleation:
        fig.add_trace(
            go.Bar(
                x=freq_labels,
                y=init_energy.tolist(),
                name="Init Energy (sharpened)",
                marker_color=bar_colors,
                opacity=0.75,
                hovertemplate="k=%{x}<br>Init Energy: %{y:.3f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=freq_labels,
            y=compatibility.tolist(),
            mode="lines+markers",
            name="Compatibility",
            line=dict(color="#1ea0a0", width=2),
            marker=dict(size=5),
            hovertemplate="k=%{x}<br>Compatibility: %{y:.3f}<extra></extra>",
        )
    )

    if has_nucleation:
        fig.add_trace(
            go.Scatter(
                x=freq_labels,
                y=overlap_score.tolist(),
                mode="lines",
                name="Overlap Score",
                line=dict(color="#ffc850", width=1.5, dash="dot"),
                hovertemplate="k=%{x}<br>Overlap: %{y:.3f}<extra></extra>",
            )
        )

        # Annotate top-quartile frequencies
        _add_winner_annotations(fig, frequencies, overlap_score, top_quartile_mask)

    subtitle = "" if has_nucleation else " (nucleation artifact not available)"
    fig.update_layout(
        title=dict(
            text=f"Nucleation Overlap — p={prime}, data_seed={data_seed}{subtitle}",
            x=0.5,
        ),
        xaxis_title="Frequency k",
        yaxis_title="Score [0, 1]",
        height=450,
        template="plotly_dark",
        plot_bgcolor="#0a1010",
        paper_bgcolor="#060a0a",
        font=dict(family="IBM Plex Mono, monospace", size=11),
        legend=dict(x=1.0, y=0.99, xanchor="left"),
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compatibility_colors(compatibility: np.ndarray) -> list[str]:
    """Map compatibility scores to colors: low=muted teal, high=warm gold."""
    colors = []
    for c in compatibility:
        r = int(30 + c * 225)
        g = int(80 + c * 120)
        b = int(80 + c * 0)
        colors.append(f"rgba({r}, {g}, {b}, 0.85)")
    return colors


def _overlap_colors(
    overlap_score: np.ndarray,
    top_quartile_mask: np.ndarray,
) -> list[str]:
    """Color bars: winners in gold, others in muted teal."""
    colors = []
    for i, score in enumerate(overlap_score):
        if top_quartile_mask[i]:
            colors.append("rgba(255, 200, 80, 0.90)")
        else:
            v = int(30 + score * 100)
            colors.append(f"rgba({v}, 160, 160, 0.45)")
    return colors


def _top_quartile_mask(scores: np.ndarray) -> np.ndarray:
    """Boolean mask for the top 25% of scores (at least 1 entry)."""
    if scores.max() == 0:
        return np.zeros(len(scores), dtype=bool)
    threshold = np.percentile(scores, 75)
    return scores >= threshold


def _align_to_frequencies(
    energy: np.ndarray,
    source_freqs: np.ndarray,
    target_freqs: np.ndarray,
) -> np.ndarray:
    """Reindex energy array from source to target frequency array."""
    if np.array_equal(source_freqs, target_freqs):
        return energy
    result = np.zeros(len(target_freqs), dtype=np.float32)
    for i, k in enumerate(target_freqs):
        src_idx = np.searchsorted(source_freqs, k)
        if src_idx < len(source_freqs) and source_freqs[src_idx] == k:
            result[i] = energy[src_idx]
    return result


def _add_winner_annotations(
    fig: go.Figure,
    frequencies: np.ndarray,
    overlap_score: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Add star annotations above top-quartile frequency bars."""
    annotations = list(fig.layout.annotations or [])
    for i, k in enumerate(frequencies):
        if mask[i]:
            annotations.append(
                dict(
                    x=int(k),
                    y=overlap_score[i] + 0.05,
                    text="★",
                    showarrow=False,
                    font=dict(size=10, color="#ffc850"),
                    xanchor="center",
                )
            )
    fig.update_layout(annotations=annotations)
