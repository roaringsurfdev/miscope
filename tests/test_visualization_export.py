"""Tests for REQ_033: Visualization Export (Static and Animated)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

from miscope.visualization.export import (
    export_animation,
    export_cross_epoch_animation,
    export_figure,
    export_variant_visualization,
    get_available_visualizations,
)


@pytest.fixture
def simple_figure():
    """Create a minimal Plotly figure for export tests."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="lines"))
    fig.update_layout(title="Test Figure", width=400, height=300)
    return fig


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ── export_figure tests ──────────────────────────────────────────────


class TestExportFigure:
    """Tests for static image/HTML export."""

    def test_png_export(self, simple_figure, output_dir):
        """export_figure() writes a PNG file."""
        path = export_figure(simple_figure, output_dir / "test.png")
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_svg_export(self, simple_figure, output_dir):
        """export_figure() writes an SVG file."""
        path = export_figure(simple_figure, output_dir / "test.svg", format="svg")
        assert path.exists()
        assert path.suffix == ".svg"
        content = path.read_text()
        assert "<svg" in content

    def test_html_export(self, simple_figure, output_dir):
        """export_figure() writes standalone HTML (no kaleido needed)."""
        path = export_figure(simple_figure, output_dir / "test.html", format="html")
        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        assert "plotly" in content.lower()

    def test_pdf_export(self, simple_figure, output_dir):
        """export_figure() writes a PDF file."""
        path = export_figure(simple_figure, output_dir / "test.pdf", format="pdf")
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.stat().st_size > 0

    def test_returns_path(self, simple_figure, output_dir):
        """export_figure() returns the Path to the written file."""
        path = export_figure(simple_figure, output_dir / "test.png")
        assert isinstance(path, Path)
        assert path == output_dir / "test.png"

    def test_auto_extension(self, simple_figure, output_dir):
        """Adds extension when output_path has none."""
        path = export_figure(simple_figure, output_dir / "test", format="svg")
        assert path.suffix == ".svg"
        assert path.exists()

    def test_creates_parent_directory(self, simple_figure, output_dir):
        """Creates parent directories if they don't exist."""
        nested = output_dir / "sub" / "dir" / "test.png"
        path = export_figure(simple_figure, nested)
        assert path.exists()

    def test_width_height_scale(self, simple_figure, output_dir):
        """Width, height, and scale parameters control output resolution."""
        small = export_figure(
            simple_figure, output_dir / "small.png", width=200, height=150, scale=1
        )
        large = export_figure(
            simple_figure, output_dir / "large.png", width=800, height=600, scale=2
        )
        # Larger image should produce a larger file
        assert large.stat().st_size > small.stat().st_size

    def test_invalid_format_raises(self, simple_figure, output_dir):
        """Raises ValueError for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_figure(simple_figure, output_dir / "test.bmp", format="bmp")

    def test_png_is_valid_image(self, simple_figure, output_dir):
        """Exported PNG can be opened as a valid image."""
        path = export_figure(simple_figure, output_dir / "test.png")
        img = Image.open(path)
        assert img.size[0] > 0
        assert img.size[1] > 0


# ── export_animation tests ──────────────────────────────────────────


class TestExportAnimation:
    """Tests for per-epoch animated GIF export."""

    @pytest.fixture
    def mock_artifacts_dir(self, output_dir):
        """Create a mock artifacts directory with fake epoch files."""
        analyzer_dir = output_dir / "artifacts" / "test_analyzer"
        analyzer_dir.mkdir(parents=True)
        for epoch in [100, 200, 300, 400, 500]:
            np.savez(analyzer_dir / f"epoch_{epoch:05d}.npz", values=np.random.rand(10))
        return output_dir / "artifacts"

    @staticmethod
    def _dummy_renderer(epoch_data, epoch, **kwargs):
        """A minimal per-epoch renderer for testing."""
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[1, 2, 3], y=epoch_data["values"][:3]))
        fig.update_layout(title=f"Epoch {epoch}", width=400, height=300)
        return fig

    def test_creates_gif(self, mock_artifacts_dir, output_dir):
        """export_animation() produces a GIF file."""
        path = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "anim.gif",
            width=400,
            height=300,
            scale=1,
        )
        assert path.exists()
        assert path.suffix == ".gif"

    def test_gif_has_multiple_frames(self, mock_artifacts_dir, output_dir):
        """Animated GIF contains multiple frames."""
        path = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "anim.gif",
            width=400,
            height=300,
            scale=1,
        )
        img = Image.open(path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        assert frame_count == 5  # 5 epochs

    def test_epoch_subset(self, mock_artifacts_dir, output_dir):
        """Epochs parameter selects a subset of epochs."""
        path = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "anim.gif",
            epochs=[100, 300, 500],
            width=400,
            height=300,
            scale=1,
        )
        img = Image.open(path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        assert frame_count == 3

    def test_fps_controls_duration(self, mock_artifacts_dir, output_dir):
        """FPS parameter affects frame duration in the GIF."""
        path_slow = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "slow.gif",
            fps=2,
            width=400,
            height=300,
            scale=1,
        )
        path_fast = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "fast.gif",
            fps=10,
            width=400,
            height=300,
            scale=1,
        )
        slow = Image.open(path_slow)
        fast = Image.open(path_fast)
        # PIL reports duration in ms; slower FPS = longer duration per frame
        assert slow.info["duration"] > fast.info["duration"]

    def test_too_few_epochs_raises(self, output_dir):
        """Raises ValueError if fewer than 2 epochs."""
        analyzer_dir = output_dir / "artifacts" / "test_analyzer"
        analyzer_dir.mkdir(parents=True)
        np.savez(analyzer_dir / "epoch_00001.npz", values=np.array([1.0]))
        with pytest.raises(ValueError, match="at least 2 epochs"):
            export_animation(
                render_fn=self._dummy_renderer,
                artifacts_dir=output_dir / "artifacts",
                analyzer_name="test_analyzer",
                output_path=output_dir / "fail.gif",
            )

    def test_auto_gif_extension(self, mock_artifacts_dir, output_dir):
        """Adds .gif extension when not provided."""
        path = export_animation(
            render_fn=self._dummy_renderer,
            artifacts_dir=mock_artifacts_dir,
            analyzer_name="test_analyzer",
            output_path=output_dir / "anim",
            width=400,
            height=300,
            scale=1,
        )
        assert path.suffix == ".gif"


# ── export_cross_epoch_animation tests ───────────────────────────────


class TestExportCrossEpochAnimation:
    """Tests for cross-epoch animated GIF export."""

    @staticmethod
    def _dummy_cross_renderer(snapshots, epochs, current_epoch, **kwargs):
        """A minimal cross-epoch renderer for testing."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=[1] * len(epochs), mode="lines"))
        fig.add_vline(x=current_epoch, line_color="red")
        fig.update_layout(title=f"Current: {current_epoch}", width=400, height=300)
        return fig

    def test_creates_gif(self, output_dir):
        """export_cross_epoch_animation() produces a GIF file."""
        snapshots = [{"w": np.random.rand(5)} for _ in range(4)]
        epochs = [100, 200, 300, 400]
        path = export_cross_epoch_animation(
            render_fn=self._dummy_cross_renderer,
            snapshots=snapshots,
            epochs=epochs,
            output_path=output_dir / "cross.gif",
            width=400,
            height=300,
            scale=1,
        )
        assert path.exists()
        assert path.suffix == ".gif"

    def test_frames_match_epochs(self, output_dir):
        """One frame per epoch in the animation."""
        snapshots = [{"w": np.random.rand(5)} for _ in range(6)]
        epochs = [10, 20, 30, 40, 50, 60]
        path = export_cross_epoch_animation(
            render_fn=self._dummy_cross_renderer,
            snapshots=snapshots,
            epochs=epochs,
            output_path=output_dir / "cross.gif",
            width=400,
            height=300,
            scale=1,
        )
        img = Image.open(path)
        frame_count = 0
        try:
            while True:
                frame_count += 1
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        assert frame_count == 6

    def test_too_few_epochs_raises(self, output_dir):
        """Raises ValueError if fewer than 2 epochs."""
        with pytest.raises(ValueError, match="at least 2 epochs"):
            export_cross_epoch_animation(
                render_fn=self._dummy_cross_renderer,
                snapshots=[{"w": np.array([1.0])}],
                epochs=[100],
                output_path=output_dir / "fail.gif",
            )

    def test_render_kwargs_passed(self, output_dir):
        """Additional kwargs are forwarded to the renderer."""
        received_kwargs = {}

        def capturing_renderer(snapshots, epochs, current_epoch, **kwargs):
            received_kwargs.update(kwargs)
            fig = go.Figure()
            fig.update_layout(width=400, height=300)
            return fig

        snapshots = [{"w": np.random.rand(5)} for _ in range(3)]
        export_cross_epoch_animation(
            render_fn=capturing_renderer,
            snapshots=snapshots,
            epochs=[1, 2, 3],
            output_path=output_dir / "test.gif",
            width=400,
            height=300,
            scale=1,
            components=["W_in"],
        )
        assert received_kwargs["components"] == ["W_in"]


# ── export_variant_visualization tests ───────────────────────────────


class TestExportVariantVisualization:
    """Tests for the convenience function."""

    @pytest.fixture
    def mock_variant_dir(self, output_dir):
        """Create a mock variant directory with artifacts."""
        variant = output_dir / "variant"
        artifacts = variant / "artifacts"

        # Create dominant_frequencies artifacts
        df_dir = artifacts / "dominant_frequencies"
        df_dir.mkdir(parents=True)
        for epoch in [100, 200, 300]:
            np.savez(df_dir / f"epoch_{epoch:05d}.npz", coefficients=np.random.rand(50))

        # Create coarseness summary
        coarse_dir = artifacts / "coarseness"
        coarse_dir.mkdir(parents=True)
        for epoch in [100, 200, 300]:
            np.savez(coarse_dir / f"epoch_{epoch:05d}.npz", coarseness_scores=np.random.rand(32))
        np.savez(
            coarse_dir / "summary.npz",
            epochs=np.array([100, 200, 300]),
            mean_coarseness=np.random.rand(3),
            blob_count=np.array([5, 10, 15]),
        )

        return variant

    def test_unknown_visualization_raises(self, output_dir):
        """Raises ValueError for unrecognized visualization name."""
        with pytest.raises(ValueError, match="Unknown visualization"):
            export_variant_visualization(output_dir, "nonexistent_viz")

    def test_get_available_visualizations(self):
        """get_available_visualizations() returns sorted list of names."""
        available = get_available_visualizations()
        assert isinstance(available, list)
        assert len(available) > 0
        assert available == sorted(available)
        assert "dominant_frequencies" in available
        assert "parameter_trajectory" in available

    def test_registry_has_all_expected_entries(self):
        """Registry covers both per-epoch and cross-epoch visualizations."""
        available = get_available_visualizations()
        # Per-epoch
        assert "dominant_frequencies" in available
        assert "activations.mlp.neuron_frequency_clusters" in available
        assert "loss_landscape.perturbation_distribution" in available
        # Summary-based
        assert "coarseness_trajectory" in available
        assert "loss_landscape.flatness_trajectory" in available
        # Snapshot-based
        assert "parameter_trajectory" in available
        assert "parameters.pca.3d_scatter" in available

    def test_per_epoch_export(self, mock_variant_dir, output_dir):
        """Exports a per-epoch visualization (dominant_frequencies)."""
        # Mock the renderer to avoid needing real data format
        mock_fig = go.Figure()
        mock_fig.add_trace(go.Bar(x=[1], y=[1]))
        mock_fig.update_layout(width=400, height=300)

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda epoch_data, epoch, **kw: mock_fig
            path = export_variant_visualization(
                mock_variant_dir,
                "dominant_frequencies",
                epoch=200,
                output_dir=output_dir / "exports",
                width=400,
                height=300,
                scale=1,
            )
        assert path.exists()
        assert "epoch_00200" in path.name

    def test_uses_latest_epoch_when_none(self, mock_variant_dir, output_dir):
        """Defaults to latest epoch when epoch is None."""
        mock_fig = go.Figure()
        mock_fig.update_layout(width=400, height=300)
        captured_epoch = {}

        def capturing_renderer(epoch_data, epoch, **kw):
            captured_epoch["epoch"] = epoch
            return mock_fig

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = capturing_renderer
            export_variant_visualization(
                mock_variant_dir,
                "dominant_frequencies",
                epoch=None,
                output_dir=output_dir / "exports",
                width=400,
                height=300,
                scale=1,
            )
        assert captured_epoch["epoch"] == 300  # Latest of 100, 200, 300

    def test_default_output_dir(self, mock_variant_dir):
        """Default output location is variant_dir/exports/."""
        mock_fig = go.Figure()
        mock_fig.update_layout(width=400, height=300)

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda epoch_data, epoch, **kw: mock_fig
            path = export_variant_visualization(
                mock_variant_dir,
                "dominant_frequencies",
                epoch=100,
                width=400,
                height=300,
                scale=1,
            )
        assert "exports" in str(path.parent)
        assert path.exists()

    def test_creates_output_directory(self, mock_variant_dir, output_dir):
        """Creates output directory if it doesn't exist."""
        mock_fig = go.Figure()
        mock_fig.update_layout(width=400, height=300)
        nested = output_dir / "deep" / "nested" / "dir"

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda epoch_data, epoch, **kw: mock_fig
            path = export_variant_visualization(
                mock_variant_dir,
                "dominant_frequencies",
                epoch=100,
                output_dir=nested,
                width=400,
                height=300,
                scale=1,
            )
        assert path.exists()

    def test_summary_based_export(self, mock_variant_dir, output_dir):
        """Exports a summary-based visualization."""
        mock_fig = go.Figure()
        mock_fig.update_layout(width=400, height=300)

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda summary_data, current_epoch, **kw: mock_fig
            path = export_variant_visualization(
                mock_variant_dir,
                "coarseness_trajectory",
                output_dir=output_dir / "exports",
                width=400,
                height=300,
                scale=1,
            )
        assert path.exists()

    def test_missing_artifacts_raises(self, output_dir):
        """Raises FileNotFoundError for missing artifacts."""
        empty_variant = output_dir / "empty_variant"
        (empty_variant / "artifacts" / "dominant_frequencies").mkdir(parents=True)

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda *a, **kw: go.Figure()
            with pytest.raises(FileNotFoundError):
                export_variant_visualization(
                    empty_variant,
                    "dominant_frequencies",
                    width=400,
                    height=300,
                    scale=1,
                )

    def test_format_parameter(self, mock_variant_dir, output_dir):
        """Respects the format parameter for output."""
        mock_fig = go.Figure()
        mock_fig.update_layout(width=400, height=300)

        with patch("miscope.visualization.export._get_renderer") as mock_get:
            mock_get.return_value = lambda epoch_data, epoch, **kw: mock_fig
            path = export_variant_visualization(
                mock_variant_dir,
                "dominant_frequencies",
                epoch=100,
                output_dir=output_dir / "exports",
                format="html",
            )
        assert path.suffix == ".html"
        assert path.exists()
