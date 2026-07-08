"""Tests for REQ_003_004: Artifact Loader (updated for REQ_021f per-epoch storage
and REQ_022 summary statistics)."""

import os
import tempfile

import numpy as np
import pytest

from miscope.analysis.artifact_loader import ArtifactLoader


class TestArtifactLoaderInstantiation:
    """Tests for ArtifactLoader instantiation."""

    def test_instantiate_with_directory(self):
        """Can instantiate with artifacts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ArtifactLoader(tmpdir)
            assert loader is not None
            assert loader.artifacts_dir == tmpdir

    def test_instantiate_with_nonexistent_directory(self):
        """Can instantiate even if directory doesn't exist."""
        # This should not raise - directory is checked at load time
        loader = ArtifactLoader("/nonexistent/path")
        assert loader is not None


class TestArtifactLoaderWithEmptyDirectory:
    """Tests with no artifacts."""

    @pytest.fixture
    def empty_artifacts_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_get_available_analyzers_empty(self, empty_artifacts_dir):
        """Returns empty list when no artifacts exist."""
        loader = ArtifactLoader(empty_artifacts_dir)
        assert loader.get_available_analyzers() == []

    def test_load_nonexistent_raises(self, empty_artifacts_dir):
        """Raises FileNotFoundError for missing artifact."""
        loader = ArtifactLoader(empty_artifacts_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestArtifactLoaderPerEpoch:
    """Tests with per-epoch artifact files (REQ_021f format)."""

    @pytest.fixture
    def artifacts_with_epochs(self):
        """Create per-epoch artifact files matching pipeline output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = tmpdir

            # Create dominant_frequencies analyzer directory with per-epoch files
            analyzer_dir = os.path.join(artifacts_dir, "dominant_frequencies")
            os.makedirs(analyzer_dir)

            epochs = [0, 25, 49]
            for epoch in epochs:
                coefficients = np.random.rand(33).astype(np.float32)  # p=17 → 2p-1=33
                np.savez_compressed(
                    os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz"),
                    coefficients=coefficients,
                )

            yield artifacts_dir

    def test_load_epoch(self, artifacts_with_epochs):
        """Can load a single epoch's data."""
        loader = ArtifactLoader(artifacts_with_epochs)
        epoch_data = loader.load_epoch("dominant_frequencies", 0)

        assert isinstance(epoch_data, dict)
        assert "coefficients" in epoch_data
        assert "epochs" not in epoch_data  # Single-epoch — no epochs key
        assert epoch_data["coefficients"].shape == (33,)

    def test_load_epoch_not_found(self, artifacts_with_epochs):
        """Raises FileNotFoundError for missing epoch."""
        loader = ArtifactLoader(artifacts_with_epochs)
        with pytest.raises(FileNotFoundError):
            loader.load_epoch("dominant_frequencies", 999)

    def test_load_all_epochs_stacked(self, artifacts_with_epochs):
        """load() returns stacked data with epochs key."""
        loader = ArtifactLoader(artifacts_with_epochs)
        artifact = loader.load("dominant_frequencies")

        assert isinstance(artifact, dict)
        assert "epochs" in artifact
        assert "coefficients" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 25, 49])
        assert artifact["coefficients"].shape == (3, 33)

    def test_load_epochs_subset(self, artifacts_with_epochs):
        """load_epochs() with specific epoch list."""
        loader = ArtifactLoader(artifacts_with_epochs)
        artifact = loader.load_epochs("dominant_frequencies", epochs=[0, 49])

        np.testing.assert_array_equal(artifact["epochs"], [0, 49])
        assert artifact["coefficients"].shape == (2, 33)

    def test_get_available_analyzers(self, artifacts_with_epochs):
        """Lists available analyzers from directory structure."""
        loader = ArtifactLoader(artifacts_with_epochs)
        available = loader.get_available_analyzers()

        assert "dominant_frequencies" in available

    def test_get_epochs(self, artifacts_with_epochs):
        """Gets epochs for analyzer from filesystem."""
        loader = ArtifactLoader(artifacts_with_epochs)
        epochs = loader.get_epochs("dominant_frequencies")

        assert epochs == [0, 25, 49]


class TestArtifactLoaderIndependence:
    """Tests verifying loader works without pipeline (per-epoch format)."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def artifacts_dir_only(self, temp_dir):
        """Create per-epoch artifacts manually without using pipeline."""
        artifacts_dir = os.path.join(temp_dir, "artifacts")
        os.makedirs(artifacts_dir)

        # Create per-epoch files
        analyzer_dir = os.path.join(artifacts_dir, "test_analyzer")
        os.makedirs(analyzer_dir)

        np.random.seed(42)
        for epoch in [0, 100, 200]:
            data = np.random.randn(10)
            np.savez_compressed(
                os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz"),
                data=data,
            )

        return artifacts_dir

    def test_load_without_pipeline(self, artifacts_dir_only):
        """Can load stacked artifacts without any pipeline involvement."""
        loader = ArtifactLoader(artifacts_dir_only)
        artifact = loader.load("test_analyzer")

        assert "epochs" in artifact
        assert "data" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 100, 200])

    def test_load_epoch_without_pipeline(self, artifacts_dir_only):
        """Can load single epoch without pipeline."""
        loader = ArtifactLoader(artifacts_dir_only)
        epoch_data = loader.load_epoch("test_analyzer", 100)

        assert "data" in epoch_data
        assert epoch_data["data"].shape == (10,)


class TestArtifactLoaderSummary:
    """Tests for REQ_022: Summary statistics loading."""

    @pytest.fixture
    def artifacts_with_summary(self):
        """Create artifacts directory with summary.npz file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer_dir = os.path.join(tmpdir, "test_analyzer")
            os.makedirs(analyzer_dir)

            # Create per-epoch files
            for epoch in [0, 100, 200]:
                np.savez_compressed(
                    os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz"),
                    data=np.random.rand(10).astype(np.float32),
                )

            # Create summary file
            epochs = np.array([0, 100, 200])
            mean_val = np.array([0.5, 0.6, 0.7])
            max_val = np.array([0.9, 0.95, 0.99])
            np.savez_compressed(
                os.path.join(analyzer_dir, "summary.npz"),
                epochs=epochs,
                mean_val=mean_val,
                max_val=max_val,
            )

            yield tmpdir

    def test_load_summary(self, artifacts_with_summary):
        """Can load summary statistics."""
        loader = ArtifactLoader(artifacts_with_summary)
        summary = loader.load_summary("test_analyzer")

        assert isinstance(summary, dict)
        assert "epochs" in summary
        assert "mean_val" in summary
        assert "max_val" in summary
        np.testing.assert_array_equal(summary["epochs"], [0, 100, 200])
        assert summary["mean_val"].shape == (3,)

    def test_has_summary_true(self, artifacts_with_summary):
        """has_summary returns True when summary exists."""
        loader = ArtifactLoader(artifacts_with_summary)
        assert loader.has_summary("test_analyzer") is True

    def test_has_summary_false(self, artifacts_with_summary):
        """has_summary returns False when no summary exists."""
        loader = ArtifactLoader(artifacts_with_summary)
        assert loader.has_summary("nonexistent") is False

    def test_has_summary_no_summary_file(self):
        """has_summary returns False for analyzer dir without summary.npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer_dir = os.path.join(tmpdir, "test_analyzer")
            os.makedirs(analyzer_dir)
            np.savez_compressed(
                os.path.join(analyzer_dir, "epoch_00000.npz"),
                data=np.ones(5),
            )

            loader = ArtifactLoader(tmpdir)
            assert loader.has_summary("test_analyzer") is False

    def test_load_summary_not_found(self):
        """load_summary raises FileNotFoundError when no summary exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ArtifactLoader(tmpdir)
            with pytest.raises(FileNotFoundError) as exc_info:
                loader.load_summary("nonexistent")
            assert "nonexistent" in str(exc_info.value)
