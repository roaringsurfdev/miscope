"""Residue class graduation renderers.

Two views into how modular arithmetic residue classes graduate (transition
from incorrect to stably-correct) during training:
- graduation_spread: box plot of graduation epoch distribution per residue class
- graduation_cohesion: heatmap of graduation epochs arranged by class and pair rank

Both views use the input_trace_graduation cross-epoch artifact and prime p to
reconstruct residue class membership from pair indices.
"""

import numpy as np
import plotly.graph_objects as go


def _build_class_epoch_map(
    graduation_epochs: np.ndarray,
    split: np.ndarray,
    prime: int,
) -> tuple[dict[int, np.ndarray], list[int]]:
    """Group test-pair graduation epochs by residue class, sorted by mean epoch.

    Returns:
        class_epochs: dict mapping residue class → graduation epoch array
            (graduated pairs only; -1 entries excluded)
        sorted_classes: residue classes ordered by mean graduation epoch
    """
    p = prime
    indices = np.arange(p * p)
    residue_class = (indices // p + indices % p) % p
    test_mask = ~split

    class_epochs: dict[int, np.ndarray] = {}
    for c in range(p):
        mask = test_mask & (residue_class == c) & (graduation_epochs >= 0)
        if mask.any():
            class_epochs[c] = graduation_epochs[mask]

    sorted_classes = sorted(
        class_epochs.keys(),
        key=lambda c: float(np.mean(class_epochs[c])),
    )
    return class_epochs, sorted_classes


def render_graduation_spread(
    data: dict,
    prime: int,
    **kwargs,
) -> go.Figure:
    """Box plot of graduation epoch distribution per residue class.

    Each box shows the spread of graduation epochs for test pairs within
    one residue class (a+b) % p. Classes are sorted by median graduation
    epoch. Tight boxes indicate the class graduates as a coordinated unit.

    Args:
        data: cross_epoch artifact from input_trace_graduation
        prime: modulus p used to infer residue class from pair index
    """
    graduation_epochs = data["graduation_epochs"]
    split = data["split"]

    class_epochs, sorted_classes = _build_class_epoch_map(graduation_epochs, split, prime)

    if not sorted_classes:
        fig = go.Figure()
        fig.add_annotation(
            text="No graduated test pairs",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray"),
        )
        return fig

    fig = go.Figure()
    for c in sorted_classes:
        fig.add_trace(
            go.Box(
                y=class_epochs[c].tolist(),
                name=str(c),
                marker_color="steelblue",
                line_color="steelblue",
                showlegend=False,
                boxpoints="outliers",
                hovertemplate=f"class {c}<br>%{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Residue class graduation spread (p={prime})<br>"
        "<sup>Pairs grouped by (a+b) % p, sorted by median graduation epoch</sup>",
        xaxis_title="Residue class (sorted by median graduation epoch)",
        yaxis_title="Graduation epoch",
        xaxis=dict(showticklabels=False),
        template="plotly_white",
        height=480,
        margin=dict(l=60, r=20, t=70, b=60),
    )
    return fig


def render_graduation_cohesion(
    data: dict,
    prime: int,
    **kwargs,
) -> go.Figure:
    """Heatmap of graduation epochs arranged by residue class and pair rank.

    Rows = residue classes (sorted by mean graduation epoch).
    Columns = pair rank within class (sorted by individual graduation epoch).
    Color = graduation epoch. Uniform row color means the class graduates
    cohesively; rainbow rows indicate scattered graduation.

    Args:
        data: cross_epoch artifact from input_trace_graduation
        prime: modulus p used to infer residue class from pair index
    """
    graduation_epochs = data["graduation_epochs"]
    split = data["split"]

    p = prime
    indices = np.arange(p * p)
    residue_class = (indices // p + indices % p) % p
    test_mask = ~split

    class_data: dict[int, np.ndarray] = {}
    for c in range(p):
        mask = test_mask & (residue_class == c)
        if mask.any():
            class_data[c] = graduation_epochs[mask]

    class_means = {
        c: float(np.mean(v[v >= 0])) if (v >= 0).any() else float("inf")
        for c, v in class_data.items()
    }
    sorted_classes = sorted(class_means.keys(), key=lambda c: class_means[c])

    max_class_size = max(len(v) for v in class_data.values()) if class_data else 0
    n_classes = len(sorted_classes)

    heatmap_z = np.full((n_classes, max_class_size), np.nan)
    for row_idx, c in enumerate(sorted_classes):
        pairs = np.sort(class_data[c])
        valid = np.where(pairs >= 0, pairs.astype(float), np.nan)
        heatmap_z[row_idx, : len(valid)] = valid

    fig = go.Figure(
        go.Heatmap(
            z=heatmap_z,
            colorscale="Viridis",
            colorbar=dict(title="Graduation epoch"),
            hovertemplate="class rank=%{y}<br>pair rank=%{x}<br>epoch=%{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Residue class cohesion heatmap (p={prime})<br>"
        "<sup>Rows = classes sorted by mean graduation epoch | "
        "Columns = pairs sorted by graduation epoch</sup>",
        xaxis_title="Pair rank within class",
        yaxis_title="Residue class rank",
        template="plotly_white",
        height=500,
        margin=dict(l=60, r=20, t=70, b=60),
    )
    return fig
