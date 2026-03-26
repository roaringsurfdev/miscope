"""Input trace renderers — REQ_075.

Three views for the per-input prediction trace:
  - accuracy_grid: p×p heatmap of correct/incorrect at a single epoch,
                   distinguishing train from test pairs
  - residue_class_accuracy_timeline: per-class accuracy across training
                                     (test split is the primary signal)
  - pair_graduation_heatmap: p×p map of graduation epochs

Pair indexing convention: probe order, index k → a = k // p, b = k % p.
Reshaped to grid: grid.reshape(p, p)[a, b] → grid.T[b, a] for (x=a, y=b) layout.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go


def render_accuracy_grid(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """p×p heatmap of correct/incorrect predictions at a single checkpoint.

    Color encoding:
      Test correct   = blue   — generalization achieved
      Test incorrect = light gray — not yet generalized
      Train correct  = dark gray — memorized (expected after ~epoch 200)
      Train incorrect = white  — unexpected after early training

    Anti-diagonal structure (a+b = constant) visible when model learns by
    residue class. All anti-diagonals sharing a color = class-level graduation.

    Args:
        data: Dict with 'epoch_data' (input_trace per-epoch artifact) and 'prime'
        epoch: Checkpoint epoch for the title
    """
    prime = int(data["prime"])
    epoch_data = data["epoch_data"]
    correct = epoch_data["correct"]  # (p²,) bool
    split = epoch_data["split"]  # (p²,) bool — True=train

    # 0=train_incorrect, 1=train_correct, 2=test_incorrect, 3=test_correct
    values = np.where(split, np.where(correct, 1, 0), np.where(correct, 3, 2))

    # Reshape: probe order k → a=k//p, b=k%p → grid[a,b]; transpose for x=a, y=b
    grid = values.reshape(prime, prime).T  # grid[b, a]

    colorscale = [
        [0.00, "white"],  # train incorrect
        [0.24, "white"],
        [0.25, "#888888"],  # train correct (memorized)
        [0.49, "#888888"],
        [0.50, "#f0f0f0"],  # test incorrect
        [0.74, "#f0f0f0"],
        [0.75, "#1f77b4"],  # test correct (generalized)
        [1.00, "#1f77b4"],
    ]

    epoch_label = epoch if epoch is not None else "?"
    test_mask = ~split
    train_mask = split
    test_acc = correct[test_mask].mean() if test_mask.any() else 0.0
    train_acc = correct[train_mask].mean() if train_mask.any() else 0.0

    fig = go.Figure(
        go.Heatmap(
            z=grid,
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate="a=%{x}, b=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            f"Prediction Accuracy — Epoch {epoch_label} "
            f"(test: {test_acc:.1%}, train: {train_acc:.1%})"
        ),
        xaxis_title="a",
        yaxis_title="b",
        template="plotly_white",
    )
    return fig


def render_residue_class_accuracy_timeline(
    data: dict[str, Any],
    epoch: int | None,
    split: str = "test",
    **kwargs: Any,
) -> go.Figure:
    """Line plot of per-residue-class accuracy across all checkpoints.

    One trace per residue class c (0 to p-1), colored circularly.
    Overall accuracy shown as a dashed black line.

    The test split is the primary signal for grokking: a staircase pattern
    where residue classes graduate in discrete steps is the predicted
    attractor signature of frequency-based learning.

    GCD blind spots: residue classes that are multiples of gcd(committed
    frequency, p) may graduate together — watch for grouped staircases
    corresponding to the model's committed frequency set.

    Args:
        data: Dict with 'summary' (input_trace summary.npz) and 'prime'
        epoch: Cursor epoch (shown as vertical line)
        split: 'test' (default) or 'train'
    """
    prime = int(data["prime"])
    summary = data["summary"]
    epochs_arr = summary["epochs"]

    split_key = "test" if split == "test" else "train"
    residue_accuracy = summary[f"{split_key}_residue_class_accuracy"]  # (n_epochs, p)
    overall_accuracy = summary[f"{split_key}_overall_accuracy"]  # (n_epochs,)

    colors = [f"hsl({int(360 * c / prime)}, 70%, 50%)" for c in range(prime)]

    fig = go.Figure()

    for c in range(prime):
        fig.add_trace(
            go.Scatter(
                x=epochs_arr,
                y=residue_accuracy[:, c],
                mode="lines",
                line=dict(color=colors[c], width=1),
                name=f"c={c}",
                showlegend=False,
                hovertemplate=f"c={c}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=epochs_arr,
            y=overall_accuracy,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="Overall",
            hovertemplate="Overall: %{y:.2f}<extra></extra>",
        )
    )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="red", width=1, dash="dot"))

    split_label = "Test" if split_key == "test" else "Train"
    fig.update_layout(
        title=f"{split_label} Residue Class Accuracy by Epoch",
        xaxis_title="Epoch",
        yaxis_title="Fraction Correct",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )
    return fig


def render_pair_graduation_heatmap(
    data: dict[str, Any],
    epoch: int | None,
    split: str = "test",
    **kwargs: Any,
) -> go.Figure:
    """p×p heatmap of graduation epochs for each pair.

    Cell color encodes the epoch at which pair (a, b) first achieved
    stable correctness. Never-graduated pairs get gray. Train pairs are
    shown in a muted overlay when split='test'.

    Anti-diagonal structure: if the model learns by residue class, pairs
    with the same sum (a+b = constant) should share similar graduation epochs,
    producing uniformly-colored anti-diagonals.

    Args:
        data: Dict with 'graduation' (input_trace_graduation cross_epoch) and 'prime'
        epoch: Unused (summary view; no cursor)
        split: 'test' (default) show test pairs, 'all' show all pairs
    """
    prime = int(data["prime"])
    graduation = data["graduation"]
    graduation_epochs = graduation["graduation_epochs"].astype(np.float32)  # (p²,)
    split_arr = graduation["split"]  # (p²,) True=train

    # For 'test' split: replace train pairs with a sentinel to mute them
    TRAIN_SENTINEL = -998.0
    NEVER_GRAD = -999.0
    # TEST_PAIR_NOT_SHOWN = np.nan

    grad_values = np.where(graduation_epochs < 0, NEVER_GRAD, graduation_epochs).astype(np.float32)

    if split == "test":
        grad_values = np.where(split_arr, TRAIN_SENTINEL, grad_values)

    # Reshape to grid: probe order k → a=k//p, b=k%p; transpose for x=a, y=b
    grid = grad_values.reshape(prime, prime).T  # grid[b, a]

    # Color scale: TRAIN_SENTINEL → light beige, NEVER_GRAD → gray,
    # graduated pairs → navy to yellow by epoch
    graduated_mask = graduation_epochs >= 0
    if split == "test":
        graduated_mask = graduated_mask & ~split_arr
    graduated_vals = graduation_epochs[graduated_mask]

    vmin = float(graduated_vals.min()) if len(graduated_vals) > 0 else 0
    vmax = float(graduated_vals.max()) if len(graduated_vals) > 0 else 1
    epoch_range = vmax - vmin if vmax > vmin else 1.0

    def _remap(v: np.ndarray) -> np.ndarray:
        """Remap raw graduation values to [0, 1] colorscale range."""
        return np.where(
            v == TRAIN_SENTINEL,
            0.1,
            np.where(
                v == NEVER_GRAD,
                0.2,
                0.3 + 0.7 * (v - vmin) / epoch_range,
            ),
        )

    grid_remapped = _remap(grid)

    colorscale = [
        [0.00, "#e8e0d0"],  # train pair (muted beige)
        [0.14, "#e8e0d0"],
        [0.15, "#aaaaaa"],  # never graduated (gray)
        [0.29, "#aaaaaa"],
        [0.30, "navy"],  # earliest graduation
        [1.00, "lightyellow"],  # latest graduation
    ]

    tick_vals = [0.1, 0.2, 0.3, 0.65, 1.0]
    tick_text = ["Train", "Never", str(int(vmin)), str(int((vmin + vmax) / 2)), str(int(vmax))]

    split_label = "Test" if split == "test" else "All"
    fig = go.Figure(
        go.Heatmap(
            z=grid_remapped,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="Graduation<br>Epoch",
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            hovertemplate="a=%{x}, b=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Pair Graduation Epochs ({split_label})",
        xaxis_title="a",
        yaxis_title="b",
        template="plotly_white",
    )
    return fig
