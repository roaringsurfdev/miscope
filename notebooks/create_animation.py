# %% imports
import io
import sys
import os
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family
from miscope import visualization
from miscope.loaded_family import Variant
#from miscope.visualization.export import export_animation

# %% create animation method

def _save_gif(frames: list[Image.Image], output_path: Path, fps: int) -> None:
    """Save a list of PIL Images as an animated GIF."""
    duration_ms = int(1000 / fps)
    # Convert RGBA to P mode for GIF (GIF doesn't support RGBA)
    rgb_frames = [frame.convert("RGB") for frame in frames]
    rgb_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=rgb_frames[1:],
        duration=duration_ms,
        loop=0,
    )

def _fig_to_pil(fig: go.Figure, width: int, height: int, scale: int) -> Image.Image:
    """Render a Plotly figure to a PIL Image (in memory, no temp files)."""
    png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

def export_animation(
    variant: Variant,
    view_name: str,
    output_path: str | Path,
    epochs: list[int] | None = None,
    fps: int = 5,
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    **view_kwargs,
) -> Path:
    """Create an animated GIF from a per-epoch renderer.

    Iterates over epochs, calls render_fn(epoch_data, epoch, **render_kwargs)
    for each, renders each frame to PNG, and stitches into an animated GIF.

    Args:
        render_fn: A per-epoch renderer function with signature
            (epoch_data: dict, epoch: int, **kwargs) -> go.Figure.
        artifacts_dir: Path to variant's artifacts directory.
        analyzer_name: Name of the analyzer whose artifacts to load.
        output_path: Destination file (.gif extension).
        epochs: Specific epochs to include. None = all available.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        scale: Resolution multiplier.
        **render_kwargs: Additional arguments passed to render_fn.

    Returns:
        Path to the written GIF file.

    Raises:
        FileNotFoundError: If no artifacts found for the analyzer.
        ValueError: If fewer than 2 epochs available.
    """
    if epochs is None:
        epochs = variant.get_available_checkpoints()
    if len(epochs) < 2:
        raise ValueError(f"Need at least 2 epochs for animation, got {len(epochs)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for epoch in epochs:
        ctx = variant.at(epoch)
        fig =ctx.view(view_name).figure(**view_kwargs)
        frame = _fig_to_pil(fig, width, height, scale)
        frames.append(frame)

    _save_gif(frames, output_path, fps)
    return output_path

# %% load model and list variants
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)
view_name = "centroid_pca"
view_kwargs = {"site": "attn_out"}
output_path = os.path.join("animations", f"{variant.name}_{view_name}_att_out.gif")

export_animation(variant, view_name=view_name, output_path=output_path, fps=2, **view_kwargs)

# %%
