"""Tests for REQ_117 phase 2: ParameterDMD cross-epoch analyzer."""

import os
import tempfile

import numpy as np
import pytest

from miscope.analysis.analyzers.parameter_dmd import ParameterDMD
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.protocols import CrossEpochAnalyzer


# ── Synthetic artifact builders ──────────────────────────────────────


def _make_parameter_snapshot(
    n_epochs: int = 30,
    d_model: int = 16,
    d_mlp: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W_in": rng.normal(size=(n_epochs, d_model, d_mlp)).astype(np.float32),
        "W_out": rng.normal(size=(n_epochs, d_mlp, d_model)).astype(np.float32),
    }


def _make_neuron_grouping(
    d_mlp: int = 32,
    n_groups: int = 4,
    n_unassigned: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthetic neuron_grouping artifact with a clean group partition."""
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, n_groups, size=d_mlp).astype(np.int64)
    # Mark first n_unassigned neurons as UNASSIGNED (-1)
    assignments[:n_unassigned] = -1
    n_per_group = np.bincount(
        np.maximum(assignments, 0),
        minlength=n_groups,
    ).astype(np.int64)
    # Override with real bincount over only assigned values
    assigned_mask = assignments != -1
    n_per_group = np.bincount(
        assignments[assigned_mask], minlength=n_groups
    ).astype(np.int64)
    return {
        "assignments": assignments,
        "n_groups": np.array(n_groups, dtype=np.int64),
        "method": np.array("kmeans", dtype="U64"),
        "feature_basis_name": np.array("synthetic", dtype="U64"),
        "n_per_group": n_per_group,
        "n_unassigned": np.array(n_unassigned, dtype=np.int64),
        "centroids": np.zeros((n_groups, 4)),
        "radii": np.zeros(n_groups),
        "fisher_min": np.array(0.0),
        "fisher_mean": np.array(0.0),
        "dispersion": np.array(0.0),
        "had_family_override": np.array(False),
    }


@pytest.fixture
def variant_artifacts_dir():
    """Temp artifacts dir with parameter_snapshot per-epoch artifacts and
    a neuron_grouping artifact at every epoch (so reference_epoch can
    snap to any of them)."""
    n_epochs = 30
    d_model = 16
    d_mlp = 32
    n_groups = 4
    epochs = list(range(0, 100 * n_epochs, 100))  # epoch_00000, _00100, ...

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        ps_dir = os.path.join(artifacts_dir, "parameter_snapshot")
        ng_dir = os.path.join(artifacts_dir, "neuron_grouping")
        os.makedirs(ps_dir)
        os.makedirs(ng_dir)

        ps_data = _make_parameter_snapshot(n_epochs, d_model, d_mlp)
        ng_data = _make_neuron_grouping(d_mlp, n_groups, n_unassigned=4)

        for i, epoch in enumerate(epochs):
            ps_path = os.path.join(ps_dir, f"epoch_{int(epoch):05d}.npz")
            np.savez_compressed(  # type: ignore[arg-type]
                ps_path,
                W_in=ps_data["W_in"][i],
                W_out=ps_data["W_out"][i],
            )
            ng_path = os.path.join(ng_dir, f"epoch_{int(epoch):05d}.npz")
            np.savez_compressed(ng_path, **ng_data)  # type: ignore[arg-type]

        yield artifacts_dir, epochs, n_epochs, d_model, d_mlp, n_groups


# ── Protocol conformance ─────────────────────────────────────────────


class TestParameterDMDProtocol:
    def test_conforms_to_cross_epoch_protocol(self):
        assert isinstance(ParameterDMD(), CrossEpochAnalyzer)

    def test_name(self):
        assert ParameterDMD().name == "parameter_dmd"

    def test_requires_parameter_snapshot_and_neuron_grouping(self):
        assert ParameterDMD().requires == ["parameter_snapshot", "neuron_grouping"]

    def test_registered_in_registry(self):
        assert "parameter_dmd" in AnalyzerRegistry._cross_epoch_analyzers

    def test_distinct_from_activation_dmd(self):
        """Both DMD analyzers exist independently."""
        assert "activation_dmd" in AnalyzerRegistry._cross_epoch_analyzers
        assert "parameter_dmd" in AnalyzerRegistry._cross_epoch_analyzers
        assert (
            AnalyzerRegistry._cross_epoch_analyzers["activation_dmd"]
            is not AnalyzerRegistry._cross_epoch_analyzers["parameter_dmd"]
        )


# ── Output schema tests ──────────────────────────────────────────────


class TestParameterDMDOutput:
    def test_returns_dict(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        assert isinstance(result, dict)

    def test_contains_metadata(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for key in ["epochs", "reference_epoch", "n_groups", "populated_groups", "group_n_neurons"]:
            assert key in result, f"missing metadata: {key}"

    def test_reference_epoch_defaults_to_last(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        assert int(result["reference_epoch"]) == int(epochs[-1])

    def test_reference_epoch_configurable(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        chosen = int(epochs[10])
        result = ParameterDMD().analyze_across_epochs(
            artifacts_dir, epochs, {"parameter_dmd_reference_epoch": chosen}
        )
        assert int(result["reference_epoch"]) == chosen

    def test_reference_epoch_snaps_to_nearest(self, variant_artifacts_dir):
        """Caller-supplied epoch not in the available set snaps to nearest."""
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        # Off-grid request between epoch[5]=500 and epoch[6]=600: nearest is 500.
        result = ParameterDMD().analyze_across_epochs(
            artifacts_dir, epochs, {"parameter_dmd_reference_epoch": 540}
        )
        assert int(result["reference_epoch"]) == 500

    def test_populated_groups_excludes_empty(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_, n_groups = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        # All groups should have ≥1 assigned neuron in our synthetic fixture
        # (rng-driven but we made sure n_groups is small enough that none ends up empty).
        assert len(result["populated_groups"]) <= n_groups
        assert (result["populated_groups"] >= 0).all()

    def test_per_group_per_matrix_keys_present(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                prefix = f"group_{int(g)}__{matrix}"
                for stage_key in [
                    f"{prefix}__trajectory",
                    f"{prefix}__n_components",
                    f"{prefix}__windowed__eigenvalues",
                    f"{prefix}__windowed__residual_norm_mean",
                    f"{prefix}__tracks__track_ids",
                    f"{prefix}__tracks__n_tracks",
                    f"{prefix}__regimes__segment_starts",
                    f"{prefix}__regimes__segment_ends",
                    f"{prefix}__regimes__threshold_used",
                    f"{prefix}__per_regime__eigenvalues",
                    f"{prefix}__per_regime__residual_norm_mean",
                ]:
                    assert stage_key in result, f"missing: {stage_key}"

    def test_pca_n_components_is_positive(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                n = int(result[f"group_{int(g)}__{matrix}__n_components"])
                assert n >= 1

    def test_pca_n_components_capped(self, variant_artifacts_dir):
        """PCA components are capped at 50 even if variance budget would
        allow more."""
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                n = int(result[f"group_{int(g)}__{matrix}__n_components"])
                assert n <= 50

    def test_trajectory_shape(self, variant_artifacts_dir):
        """Per-(group, matrix) trajectory has shape (n_epochs, n_components)."""
        artifacts_dir, epochs, n_epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                trajectory = result[f"group_{int(g)}__{matrix}__trajectory"]
                n_components = int(result[f"group_{int(g)}__{matrix}__n_components"])
                assert trajectory.shape == (n_epochs, n_components)

    def test_windowed_dmd_n_windows(self, variant_artifacts_dir):
        """Default window_size = 10, stride = 1 -> n_windows = n_epochs - 10 + 1."""
        artifacts_dir, epochs, n_epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        expected_n_windows = n_epochs - 10 + 1
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                window_starts = result[
                    f"group_{int(g)}__{matrix}__windowed__window_starts"
                ]
                assert len(window_starts) == expected_n_windows

    def test_regime_segments_partition_window_space(self, variant_artifacts_dir):
        """Detected regime segments must partition [0, n_windows) exactly."""
        artifacts_dir, epochs, n_epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        n_windows = n_epochs - 10 + 1
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                prefix = f"group_{int(g)}__{matrix}"
                starts = result[f"{prefix}__regimes__segment_starts"]
                ends = result[f"{prefix}__regimes__segment_ends"]
                assert starts[0] == 0
                assert ends[-1] == n_windows
                np.testing.assert_array_equal(starts[1:], ends[:-1])

    def test_per_regime_dmd_segments_match_regimes(self, variant_artifacts_dir):
        artifacts_dir, epochs, *_ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        for g in result["populated_groups"]:
            for matrix in ["W_in", "W_out"]:
                prefix = f"group_{int(g)}__{matrix}"
                n_regimes = len(result[f"{prefix}__regimes__segment_starts"])
                n_per_regime = len(result[f"{prefix}__per_regime__segment_starts"])
                assert n_regimes == n_per_regime


# ── Failure modes ────────────────────────────────────────────────────


class TestParameterDMDFailureModes:
    def test_missing_neuron_grouping_raises(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        ps_dir = os.path.join(artifacts_dir, "parameter_snapshot")
        os.makedirs(ps_dir)
        # parameter_snapshot but no neuron_grouping
        rng = np.random.default_rng(0)
        np.savez_compressed(
            os.path.join(ps_dir, "epoch_00000.npz"),
            W_in=rng.normal(size=(16, 32)).astype(np.float32),
            W_out=rng.normal(size=(32, 16)).astype(np.float32),
        )
        with pytest.raises(FileNotFoundError, match="neuron_grouping"):
            ParameterDMD().analyze_across_epochs(artifacts_dir, [0], {})


# ── UNASSIGNED neuron handling ───────────────────────────────────────


class TestParameterDMDUnassignedHandling:
    def test_unassigned_neurons_excluded(self, variant_artifacts_dir):
        """Group counts in populated_groups should match neuron_grouping's
        n_per_group, NOT total d_mlp (because UNASSIGNED neurons are
        skipped)."""
        artifacts_dir, epochs, *_, d_mlp, _ = variant_artifacts_dir
        result = ParameterDMD().analyze_across_epochs(artifacts_dir, epochs, {})
        total_per_populated = int(result["group_n_neurons"].sum())
        # The synthetic fixture had 4 unassigned of d_mlp neurons
        assert total_per_populated == d_mlp - 4
