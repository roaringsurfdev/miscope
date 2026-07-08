# REQ_033: Visualization Export (Static and Animated)

**Status:** Draft
**Priority:** Medium (research workflow enhancement)
**Dependencies:** REQ_021f (Per-Epoch Artifacts), REQ_029 (Parameter Trajectory)
**Last Updated:** 2026-02-10

## Problem Statement

All visualizations in the workbench are currently only viewable through the Gradio dashboard. This creates two limitations:

1. **Sharing:** Researchers must take screenshots to capture visualizations for discussion, documentation, or presentation. Screenshots are lossy — they depend on window size, scroll position, and manual effort.

2. **Programmatic access:** Claude (or any script/notebook) cannot generate visualizations on demand from artifacts. When analyzing a variant collaboratively, the researcher must manually navigate the dashboard, capture images, and share file paths. If the visualization layer were directly accessible, either party could render exactly the visualization needed from the compiled artifacts.

The renderers already return Plotly `go.Figure` objects, which support export to static images (PNG, SVG, PDF) and to standalone HTML. The missing piece is a lightweight export utility that connects artifacts → renderers → files, plus an animation utility that stitches per-epoch renders into GIF or MP4.

## Design

### Static Export

A utility module at `visualization/export.py` that provides:

```python
def export_figure(
    fig: go.Figure,
    output_path: str | Path,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
    scale: int = 2,
) -> Path:
    """Export a Plotly figure to a static image file.

    Args:
        fig: Any Plotly Figure (from any renderer).
        output_path: Destination file path (extension optional if format given).
        format: "png", "svg", "pdf", or "html".
        width: Image width in pixels.
        height: Image height in pixels.
        scale: Resolution multiplier (2 = retina).

    Returns:
        Path to the written file.
    """
```

For HTML export, use `fig.write_html()` (no kaleido needed). For raster/vector, use `fig.write_image()` which requires `kaleido`.

### Animation Export

```python
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
    """Create an animated GIF or MP4 from a per-epoch renderer.

    Iterates over epochs, calls render_fn for each, and stitches
    frames into an animation.

    Args:
        render_fn: A per-epoch renderer function.
        artifacts_dir: Path to variant's artifacts directory.
        analyzer_name: Name of the analyzer whose artifacts to load.
        output_path: Destination file (extension determines format).
        epochs: Specific epochs to include. None = all available.
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        scale: Resolution multiplier.
        **render_kwargs: Additional arguments passed to render_fn.

    Returns:
        Path to the written animation file.
    """
```

For cross-epoch renderers (trajectory, specialization trajectory, etc.), the animation would sweep the `current_epoch` highlight across a fixed plot:

```python
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
    frames, showing progression through training.

    Args:
        render_fn: A cross-epoch renderer that accepts current_epoch.
        snapshots: Pre-loaded snapshot data.
        epochs: Epoch list.
        output_path: Destination file.
        fps: Frames per second.
        **render_kwargs: Additional arguments passed to render_fn.

    Returns:
        Path to the written animation file.
    """
```

### Animation Implementation

GIF stitching from Plotly figures:
1. Render each frame to PNG bytes via `fig.to_image()` (kaleido)
2. Load as PIL Image
3. Save as animated GIF via PIL or as MP4 via imageio/ffmpeg

Dependencies: `kaleido` (required), `Pillow` (for GIF), `imageio[ffmpeg]` (optional, for MP4).

### Convenience Functions

For common export patterns, provide high-level wrappers:

```python
def export_variant_visualization(
    variant_dir: str | Path,
    visualization: str,
    epoch: int | None = None,
    output_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Export a named visualization for a variant.

    Args:
        variant_dir: Path to the variant directory.
        visualization: Name like "dominant_frequencies", "parameter_trajectory", etc.
        epoch: For per-epoch visualizations. None = use latest.
        output_dir: Where to save. None = variant_dir/exports/.
        **kwargs: Passed to the underlying renderer.

    Returns:
        Path to the exported file.
    """
```

This is the function that makes it trivial for Claude or a notebook to generate a specific visualization:

```python
# Example: Claude generating a visualization during analysis discussion
export_variant_visualization(
    "results/modulo_addition_1layer/p101_seed999",
    "parameter_trajectory",
    output_dir="/tmp/analysis",
    components=["W_in", "W_out"],
)
```

### Dependency Addition

Add to `pyproject.toml`:
- `kaleido>=0.2.1` (required for image export)
- `Pillow>=10.0` (required for GIF animation)
- `imageio[ffmpeg]>=2.30` (optional, for MP4)

## Scope

**This requirement covers:**
1. `visualization/export.py` — static export and animation utilities
2. Dependency additions (kaleido, Pillow)
3. Tests for export functions
4. Support for both per-epoch and cross-epoch renderer patterns

**This requirement does not cover:**
- Dashboard UI for triggering exports (future enhancement)
- Batch export of all visualizations (could be a script built on these utilities)
- Video with audio or annotations
- Custom frame transitions or interpolation

## Conditions of Satisfaction

### Static Export
- [ ] `export_figure()` writes PNG files from any Plotly Figure
- [ ] `export_figure()` writes SVG files from any Plotly Figure
- [ ] `export_figure()` writes standalone HTML files (no kaleido needed)
- [ ] Width, height, and scale parameters control output resolution
- [ ] Returns the Path to the written file

### Animation Export
- [ ] `export_animation()` creates animated GIF from a per-epoch renderer
- [ ] Animation loads artifacts via ArtifactLoader (does not require pre-loaded data)
- [ ] `export_cross_epoch_animation()` creates animated GIF from a cross-epoch renderer by sweeping current_epoch
- [ ] FPS parameter controls animation speed
- [ ] Epochs parameter allows selecting a subset of epochs

### Convenience
- [ ] `export_variant_visualization()` maps visualization names to renderers
- [ ] Defaults to latest epoch for per-epoch visualizations when epoch is None
- [ ] Creates output directory if it doesn't exist
- [ ] Default output location is `{variant_dir}/exports/`

### Tests
- [ ] Static export produces files with expected extensions
- [ ] Animation produces a valid GIF with multiple frames
- [ ] Convenience function resolves visualization names correctly
- [ ] Error handling for missing artifacts, invalid visualization names

## Constraints

**Must have:**
- Works without a running dashboard (pure library/CLI usage)
- Consistent image quality across exports (explicit width/height/scale)
- Support for all existing renderers without renderer modifications

**Must avoid:**
- Modifying any existing renderer code (export wraps renderers, doesn't change them)
- Requiring ffmpeg for the core GIF functionality (ffmpeg only for optional MP4)
- Large memory footprint during animation (render and write frames incrementally)

**Flexible:**
- Exact convenience function API (the mapping from names to renderers can evolve)
- Whether MP4 support is included in this requirement or deferred
- Default resolution values
- Animation frame labeling (epoch number overlay is nice-to-have)

## Notes

**Staged approach:** This requirement covers the middle-tier export utilities. A future requirement could add dashboard UI for triggering exports (download buttons, animation preview). The utilities here are the foundation that makes both CLI/notebook export and future dashboard export possible.

**Claude workflow:** With these utilities, Claude can generate specific visualizations during analysis discussions by running a short Python command, rather than requiring the researcher to navigate the dashboard and capture screenshots. This significantly improves the collaborative analysis workflow.

**Renderer compatibility:** All existing renderers return `go.Figure` objects and accept standard parameters (epoch, current_epoch, etc.). The export utilities should work with any renderer without special-casing, but the convenience function will need a mapping from visualization names to renderer functions and their expected data sources.
