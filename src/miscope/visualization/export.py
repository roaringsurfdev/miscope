"""REQ_033: Visualization Export (Static and Animated).

Provides utilities to export Plotly figures to static images (PNG, SVG, PDF),
standalone HTML, and animated GIFs. Works without a running dashboard —
designed for programmatic use from notebooks, scripts, or Claude.

Dependencies:
    - kaleido: Required for raster/vector image export (PNG, SVG, PDF)
    - Pillow: Required for GIF animation stitching

Usage:
    from miscope.visualization.export import export_figure, export_animation

    # Static export of any Plotly figure
    export_figure(fig, "output.png")

    # Animated GIF from a per-epoch renderer
    export_animation(
        render_fn=render_dominant_frequencies,
        artifacts_dir="results/variant/artifacts",
        analyzer_name="dominant_frequencies",
        output_path="dominant_freq.gif",
    )
"""

import io
from collections.abc import Callable
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image

from miscope.analysis.artifact_loader import ArtifactLoader

VALID_FORMATS = {"png", "svg", "pdf", "html"}


def _read_prime_from_config(config_path: Path) -> int:
    """Read the prime modulus from a variant's config.json."""
    import json

    with open(config_path) as f:
        config = json.load(f)
    return int(config["prime"])


def export_figure(
    fig: go.Figure,
    output_path: str | Path,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
) -> Path:
    """Export a Plotly figure to a static image or HTML file.

    Args:
        fig: Any Plotly Figure (from any renderer).
        output_path: Destination file path (extension optional if format given).
        format: "png", "svg", "pdf", or "html".
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution multiplier (2 = retina).

    Returns:
        Path to the written file.

    Raises:
        ValueError: If format is not supported.
    """
    format = format.lower()
    if format not in VALID_FORMATS:
        raise ValueError(f"Unsupported format '{format}'. Use one of: {sorted(VALID_FORMATS)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        fig.write_html(str(output_path))
    else:
        fig.write_image(
            str(output_path),
            format=format,
            width=width,
            height=height,
            scale=scale,
        )

    return output_path


def export_animation(
    render_fn: Callable,
    artifacts_dir: str | Path,
    analyzer_name: str,
    output_path: str | Path,
    epochs: list[int] | None = None,
    fps: int = 5,
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    **render_kwargs,
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
    loader = ArtifactLoader(str(artifacts_dir))

    if epochs is None:
        epochs = loader.get_epochs(analyzer_name)
    if len(epochs) < 2:
        raise ValueError(f"Need at least 2 epochs for animation, got {len(epochs)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for epoch in epochs:
        epoch_data = loader.load_epoch(analyzer_name, epoch)
        fig = render_fn(epoch_data, epoch, **render_kwargs)
        frame = _fig_to_pil(fig, width, height, scale)
        frames.append(frame)

    _save_gif(frames, output_path, fps)
    return output_path


def export_cross_epoch_animation(
    render_fn: Callable,
    snapshots: list[dict],
    epochs: list[int],
    output_path: str | Path,
    fps: int = 5,
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    **render_kwargs,
) -> Path:
    """Animate a cross-epoch renderer by sweeping the current_epoch highlight.

    The underlying data stays fixed; the epoch indicator moves across
    frames, showing progression through training. Renders one frame per
    epoch in the epoch list.

    Args:
        render_fn: A cross-epoch renderer that accepts current_epoch.
            Signature: (snapshots, epochs, current_epoch, **kwargs) -> go.Figure
        snapshots: Pre-loaded snapshot data (list of per-epoch dicts).
        epochs: Epoch list.
        output_path: Destination file (.gif extension).
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        scale: Resolution multiplier.
        **render_kwargs: Additional arguments passed to render_fn.

    Returns:
        Path to the written GIF file.

    Raises:
        ValueError: If fewer than 2 epochs provided.
    """
    if len(epochs) < 2:
        raise ValueError(f"Need at least 2 epochs for animation, got {len(epochs)}")

    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for current_epoch in epochs:
        fig = render_fn(snapshots, epochs, current_epoch, **render_kwargs)
        frame = _fig_to_pil(fig, width, height, scale)
        frames.append(frame)

    _save_gif(frames, output_path, fps)
    return output_path


# ---------------------------------------------------------------------------
# Visualization name → renderer mapping for convenience function
# ---------------------------------------------------------------------------

# Each entry: (analyzer_name, renderer_fn, data_pattern)
# data_pattern is one of: "per_epoch", "cross_epoch", "summary", "snapshot"
_VISUALIZATION_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Per-epoch renderers (epoch_data, epoch)
    "dominant_frequencies": ("dominant_frequencies", "render_dominant_frequencies", "per_epoch"),
    "neuron_heatmap": ("neuron_activations", "render_neuron_heatmap", "per_epoch"),
    "freq_clusters": ("neuron_freq_norm", "render_freq_clusters", "per_epoch"),
    "coarseness_distribution": ("coarseness", "render_coarseness_distribution", "per_epoch"),
    "coarseness_by_neuron": ("coarseness", "render_coarseness_by_neuron", "per_epoch"),
    "attention_heads": ("attention_patterns", "render_attention_heads", "per_epoch"),
    "attention_freq_heatmap": ("attention_freq", "render_attention_freq_heatmap", "per_epoch"),
    "singular_value_spectrum": (
        "effective_dimensionality",
        "render_singular_value_spectrum",
        "per_epoch",
    ),
    "perturbation_distribution": (
        "landscape_flatness",
        "render_perturbation_distribution",
        "per_epoch",
    ),
    # Summary-based cross-epoch renderers (summary_data, current_epoch)
    "coarseness_trajectory": ("coarseness", "render_coarseness_trajectory", "summary"),
    "blob_count_trajectory": ("coarseness", "render_blob_count_trajectory", "summary"),
    "specialization_trajectory": (
        "neuron_freq_norm",
        "render_specialization_trajectory",
        "summary",
    ),
    "specialization_by_frequency": (
        "neuron_freq_norm",
        "render_specialization_by_frequency",
        "summary",
    ),
    "dimensionality_trajectory": (
        "effective_dimensionality",
        "render_dimensionality_trajectory",
        "summary",
    ),
    "attention_specialization_trajectory": (
        "attention_freq",
        "render_attention_specialization_trajectory",
        "summary",
    ),
    "attention_dominant_frequencies": (
        "attention_freq",
        "render_attention_dominant_frequencies",
        "summary",
    ),
    "flatness_trajectory": ("landscape_flatness", "render_flatness_trajectory", "summary"),
    # Cross-epoch stacked renderers (artifact with "epochs" key)
    "dominant_frequencies_over_time": (
        "dominant_frequencies",
        "render_dominant_frequencies_over_time",
        "cross_epoch",
    ),
    # Cross-epoch precomputed renderers (REQ_038)
    "parameter_trajectory": (
        "parameter_trajectory",
        "render_parameter_trajectory",
        "cross_epoch_pca",
    ),
    "trajectory_3d": ("parameter_trajectory", "render_trajectory_3d", "cross_epoch_pca"),
    "trajectory_pc1_pc3": ("parameter_trajectory", "render_trajectory_pc1_pc3", "cross_epoch_pca"),
    "trajectory_pc2_pc3": ("parameter_trajectory", "render_trajectory_pc2_pc3", "cross_epoch_pca"),
    "explained_variance": (
        "parameter_trajectory",
        "render_explained_variance",
        "cross_epoch_pca_no_epoch",
    ),
    "parameter_velocity": (
        "parameter_trajectory",
        "render_parameter_velocity",
        "cross_epoch_velocity",
    ),
    "component_velocity": (
        "parameter_trajectory",
        "render_component_velocity",
        "cross_epoch_component_velocity",
    ),
    # REQ_042: Cross-epoch neuron dynamics
    "neuron_freq_trajectory": (
        "neuron_dynamics",
        "render_neuron_freq_trajectory",
        "cross_epoch_neuron_dynamics",
    ),
}


def _get_renderer(name: str) -> Callable:
    """Import and return a renderer function by name."""
    import miscope.visualization as viz

    return getattr(viz, name)


def export_variant_visualization(
    variant_dir: str | Path,
    visualization: str,
    epoch: int | None = None,
    output_dir: str | Path | None = None,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
    **kwargs,
) -> Path:
    """Export a named visualization for a variant.

    High-level convenience function that connects artifacts → renderer → file.
    Maps visualization names to the correct renderer and data loading pattern.

    Args:
        variant_dir: Path to the variant directory (contains artifacts/).
        visualization: Name like "dominant_frequencies", "parameter_trajectory", etc.
        epoch: For per-epoch visualizations, the epoch to render.
            None = use latest available epoch.
        output_dir: Where to save. None = variant_dir/exports/.
        format: Output format ("png", "svg", "pdf", "html").
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution multiplier.
        **kwargs: Additional arguments passed to the renderer.

    Returns:
        Path to the exported file.

    Raises:
        ValueError: If visualization name is not recognized.
        FileNotFoundError: If artifacts don't exist for the visualization.
    """
    if visualization not in _VISUALIZATION_REGISTRY:
        available = sorted(_VISUALIZATION_REGISTRY.keys())
        raise ValueError(f"Unknown visualization '{visualization}'. Available: {available}")

    variant_dir = Path(variant_dir)
    artifacts_dir = variant_dir / "artifacts"
    loader = ArtifactLoader(str(artifacts_dir))

    analyzer_name, renderer_name, data_pattern = _VISUALIZATION_REGISTRY[visualization]
    render_fn = _get_renderer(renderer_name)

    if output_dir is None:
        output_dir = variant_dir / "exports"
    output_dir = Path(output_dir)

    # Build the figure based on data pattern
    if data_pattern == "per_epoch":
        epochs = loader.get_epochs(analyzer_name)
        if not epochs:
            raise FileNotFoundError(f"No artifacts for '{analyzer_name}' in {artifacts_dir}")
        if epoch is None:
            epoch = epochs[-1]
        epoch_data = loader.load_epoch(analyzer_name, epoch)
        fig = render_fn(epoch_data, epoch, **kwargs)
        filename = f"{visualization}_epoch_{epoch:05d}"

    elif data_pattern == "summary":
        summary_data = loader.load_summary(analyzer_name)
        current_epoch = epoch
        if current_epoch is None:
            all_epochs = loader.get_epochs(analyzer_name)
            current_epoch = all_epochs[-1] if all_epochs else 0
        fig = render_fn(summary_data, current_epoch, **kwargs)
        filename = visualization

    elif data_pattern == "cross_epoch":
        artifact = loader.load_epochs(analyzer_name)
        fig = render_fn(artifact, **kwargs)
        filename = visualization

    elif data_pattern == "cross_epoch_pca":
        cross_epoch = loader.load_cross_epoch(analyzer_name)
        epochs_arr = cross_epoch["epochs"].tolist()
        current_epoch = epoch if epoch is not None else epochs_arr[-1]
        group = kwargs.pop("group", "all")
        pca_result = {
            "projections": cross_epoch[f"{group}__projections"],
            "explained_variance_ratio": cross_epoch[f"{group}__explained_variance_ratio"],
            "explained_variance": cross_epoch[f"{group}__explained_variance"],
        }
        fig = render_fn(pca_result, epochs_arr, current_epoch, **kwargs)
        filename = visualization

    elif data_pattern == "cross_epoch_pca_no_epoch":
        cross_epoch = loader.load_cross_epoch(analyzer_name)
        group = kwargs.pop("group", "all")
        pca_result = {
            "projections": cross_epoch[f"{group}__projections"],
            "explained_variance_ratio": cross_epoch[f"{group}__explained_variance_ratio"],
            "explained_variance": cross_epoch[f"{group}__explained_variance"],
        }
        fig = render_fn(pca_result, **kwargs)
        filename = visualization

    elif data_pattern == "cross_epoch_velocity":
        cross_epoch = loader.load_cross_epoch(analyzer_name)
        epochs_arr = cross_epoch["epochs"].tolist()
        current_epoch = epoch if epoch is not None else epochs_arr[-1]
        group = kwargs.pop("group", "all")
        velocity = cross_epoch[f"{group}__velocity"]
        fig = render_fn(velocity, epochs_arr, current_epoch, **kwargs)
        filename = visualization

    elif data_pattern == "cross_epoch_component_velocity":
        cross_epoch = loader.load_cross_epoch(analyzer_name)
        epochs_arr = cross_epoch["epochs"].tolist()
        current_epoch = epoch if epoch is not None else epochs_arr[-1]
        fig = render_fn(cross_epoch, epochs_arr, current_epoch, **kwargs)
        filename = visualization

    elif data_pattern in ("cross_epoch_neuron_dynamics", "cross_epoch_neuron_dynamics_sorted"):
        cross_epoch = loader.load_cross_epoch(analyzer_name)
        config_path = variant_dir / "config.json"
        prime = _read_prime_from_config(config_path)
        sorted_flag = data_pattern.endswith("_sorted")
        fig = render_fn(cross_epoch, prime, sorted_by_final=sorted_flag, **kwargs)
        filename = visualization

    else:
        raise ValueError(f"Unknown data pattern '{data_pattern}'")

    output_path = output_dir / f"{filename}.{format}"
    return export_figure(fig, output_path, format=format, width=width, height=height, scale=scale)


def get_available_visualizations() -> list[str]:
    """Return sorted list of visualization names for export_variant_visualization."""
    return sorted(_VISUALIZATION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fig_to_pil(fig: go.Figure, width: int, height: int, scale: int) -> Image.Image:
    """Render a Plotly figure to a PIL Image (in memory, no temp files)."""
    png_bytes = fig.to_image(format="png", width=width, height=height, scale=scale)
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


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


def export_neuron_animation(
    variant_dir: str | Path,
    neuron_idx: int,
    epochs: list[int] | None = None,
    output_path: str | Path | None = None,
    fps: int = 4,
    width: int = 600,
    height: int = 500,
    scale: int = 2,
) -> Path:
    """Export animated GIF of a single neuron's activation heatmap across epochs.

    Shows how a neuron's activation pattern evolves during training,
    useful for understanding whether frequency switches are smooth
    transitions or abrupt phase changes.

    Args:
        variant_dir: Path to the variant directory.
        neuron_idx: Which neuron to animate.
        epochs: Specific epochs to include. None = all available.
        output_path: Output file path. None = variant_dir/exports/neuron_{idx}.gif.
        fps: Frames per second for the GIF.
        width: Frame width in pixels.
        height: Frame height in pixels.
        scale: Resolution multiplier.

    Returns:
        Path to the exported GIF file.
    """
    from miscope.visualization.renderers.neuron_activations import render_neuron_heatmap

    variant_dir = Path(variant_dir)
    artifacts_dir = variant_dir / "artifacts"
    loader = ArtifactLoader(str(artifacts_dir))

    if epochs is None:
        epochs = loader.get_epochs("neuron_activations")

    frames: list[Image.Image] = []
    for epoch in epochs:
        epoch_data = loader.load_epoch("neuron_activations", epoch)
        fig = render_neuron_heatmap(epoch_data, epoch, neuron_idx)
        frames.append(_fig_to_pil(fig, width, height, scale))

    if output_path is None:
        output_dir = variant_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"neuron_{neuron_idx:03d}.gif"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gif(frames, output_path, fps)

    return output_path
