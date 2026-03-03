"""REQ_009: Loss curves with epoch indicator.

REQ_020: Checkpoint tooltips include epoch-to-index mapping.
"""

import plotly.graph_objects as go


def render_loss_curves_with_indicator(
    train_losses: list[float] | None,
    test_losses: list[float] | None,
    current_epoch: int,
    checkpoint_epochs: list[int] | None = None,
    event_epochs: list[tuple[int, str, str]] | None = None,
    log_scale: bool = True,
    title: str = "Training Progress",
) -> go.Figure:
    """Render train/test loss curves with vertical epoch indicator.

    Args:
        train_losses: List of training losses per epoch.
        test_losses: List of test losses per epoch.
        current_epoch: Current epoch for vertical line indicator.
        checkpoint_epochs: Optional list of checkpointed epochs to mark.
        event_epochs: Optional list of (epoch, label, color) tuples for named event lines.
        log_scale: Whether to use log scale for y-axis.
        title: Chart title.

    Returns:
        Plotly Figure with loss curves and indicator line.
    """
    fig = go.Figure()

    if train_losses is None or test_losses is None:
        # Empty state - show placeholder
        fig.add_annotation(
            text="No training data loaded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=300,
        )
        return fig

    epochs = list(range(len(train_losses)))

    # Train loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name="Train Loss",
            line=dict(color="blue", width=1.5),
            hovertemplate="Epoch: %{x}<br>Train Loss: %{y:.6f}<extra></extra>",
        )
    )

    # Test loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_losses,
            mode="lines",
            name="Test Loss",
            line=dict(color="orange", width=1.5),
            hovertemplate="Epoch: %{x}<br>Test Loss: %{y:.6f}<extra></extra>",
        )
    )

    # Vertical line indicator at current epoch
    if 0 <= current_epoch < len(train_losses):
        fig.add_vline(
            x=current_epoch,
            line_dash="solid",
            line_color="red",
            line_width=2,
            annotation_text=f"Epoch {current_epoch}",
            annotation_position="top right",
            annotation_font_color="red",
        )

    # Mark checkpoint epochs if provided (REQ_020: include index in tooltip)
    if checkpoint_epochs:
        valid_checkpoints = [e for e in checkpoint_epochs if e < len(train_losses)]
        if valid_checkpoints:
            checkpoint_train = [train_losses[e] for e in valid_checkpoints]
            # Build index mapping for tooltips
            checkpoint_indices = list(range(len(valid_checkpoints)))
            fig.add_trace(
                go.Scatter(
                    x=valid_checkpoints,
                    y=checkpoint_train,
                    mode="markers",
                    name="Checkpoints",
                    marker=dict(size=8, color="green", symbol="diamond"),
                    customdata=checkpoint_indices,
                    hovertemplate=(
                        "Checkpoint<br>"
                        "Epoch: %{x} (Index: %{customdata})<br>"
                        "Loss: %{y:.6f}<extra></extra>"
                    ),
                )
            )

    # Named event lines (e.g., circularity crossover epochs)
    if event_epochs:
        for epoch, label, color in event_epochs:
            if 0 <= epoch < len(train_losses or []):
                fig.add_vline(
                    x=epoch,
                    line_dash="dash",
                    line_color=color,
                    line_width=1.5,
                    annotation_text=label,
                    annotation_position="top left",
                    annotation_font_color=color,
                )

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log" if log_scale else "linear",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
