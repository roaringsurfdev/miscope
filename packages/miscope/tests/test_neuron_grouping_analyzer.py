"""Tests for REQ_118 NeuronGrouping per-epoch SecondaryAnalyzer."""

import numpy as np
import pytest

from miscope.analysis.analyzers.neuron_grouping import (
    NeuronGrouping,
    unpack_assignment,
    unpack_summary,
)
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.protocols import SecondaryAnalyzer
from miscope.core.grouping import UNASSIGNED, GroupAssignment


def _make_parameter_snapshot(
    d_model: int = 16,
    d_mlp: int = 32,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthetic parameter_snapshot artifact with the required shapes."""
    rng = np.random.default_rng(seed)
    return {
        "W_in": rng.normal(size=(d_model, d_mlp)).astype(np.float32),
        "W_out": rng.normal(size=(d_mlp, d_model)).astype(np.float32),
        "W_E": rng.normal(size=(8, d_model)).astype(np.float32),
    }


# ── Protocol conformance ─────────────────────────────────────────────


class TestNeuronGroupingProtocol:
    def test_conforms_to_secondary_protocol(self):
        assert isinstance(NeuronGrouping(), SecondaryAnalyzer)

    def test_name(self):
        assert NeuronGrouping().name == "neuron_grouping"

    def test_depends_on_parameter_snapshot(self):
        assert NeuronGrouping().depends_on == "parameter_snapshot"

    def test_registered_in_registry(self):
        assert "neuron_grouping" in AnalyzerRegistry._secondary_analyzers


# ── Universal path (no family override) ──────────────────────────────


class TestNeuronGroupingUniversalPath:
    def test_returns_dict(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        for key in [
            "assignments",
            "n_groups",
            "method",
            "feature_basis_name",
            "centroids",
            "radii",
            "n_per_group",
            "n_unassigned",
            "fisher_min",
            "fisher_mean",
            "dispersion",
            "had_family_override",
        ]:
            assert key in result, f"missing: {key}"

    def test_assignments_shape_matches_d_mlp(self):
        artifact = _make_parameter_snapshot(d_mlp=32)
        result = NeuronGrouping().analyze(artifact, context={})
        assert result["assignments"].shape == (32,)

    def test_method_is_kmeans_when_no_override(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert str(result["method"]) == "kmeans"

    def test_feature_basis_name_is_weight_signature(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert str(result["feature_basis_name"]) == "weight_signature"

    def test_no_confidence_field_for_kmeans(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert "confidence" not in result

    def test_had_family_override_false(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert bool(result["had_family_override"]) is False

    def test_n_groups_from_context(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_n_groups": 4})
        assert int(result["n_groups"]) == 4

    def test_n_groups_default_when_unset(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={})
        assert int(result["n_groups"]) == 8  # documented default

    def test_centroids_shape_matches_n_groups_x_2dmodel(self):
        d_model, d_mlp = 16, 32
        artifact = _make_parameter_snapshot(d_model=d_model, d_mlp=d_mlp)
        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_n_groups": 4})
        # Universal-path features = 2 * d_model wide
        assert result["centroids"].shape == (4, 2 * d_model)

    def test_unsupported_feature_source_raises(self):
        artifact = _make_parameter_snapshot()
        with pytest.raises(NotImplementedError, match="feature_source"):
            NeuronGrouping().analyze(
                artifact,
                context={"neuron_grouping_feature_source": "activation"},
            )


# ── Family-override path ─────────────────────────────────────────────


class TestNeuronGroupingFamilyOverride:
    def test_override_takes_precedence(self):
        """When the family supplies an override, the universal path is bypassed."""
        artifact = _make_parameter_snapshot(d_mlp=20)

        def _override(_artifact, _context):
            features = np.eye(20, 4)  # one neuron per basis component, then duplicates
            assignment = GroupAssignment(
                assignments=np.array([0, 1, 2, 3] * 5, dtype=np.int64),
                n_groups=4,
                method="custom_test",
                feature_basis_name="custom_basis",
                confidence=np.ones(20),
            )
            return assignment, features

        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_override": _override})
        assert str(result["method"]) == "custom_test"
        assert str(result["feature_basis_name"]) == "custom_basis"
        assert int(result["n_groups"]) == 4
        assert bool(result["had_family_override"]) is True

    def test_override_confidence_propagates(self):
        artifact = _make_parameter_snapshot(d_mlp=4)

        def _override(_artifact, _context):
            features = np.eye(4)
            assignment = GroupAssignment(
                assignments=np.array([0, 1, 2, 3], dtype=np.int64),
                n_groups=4,
                method="custom_test",
                feature_basis_name="custom_basis",
                confidence=np.array([0.9, 0.8, 0.7, 0.6]),
            )
            return assignment, features

        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_override": _override})
        assert "confidence" in result
        np.testing.assert_allclose(result["confidence"], [0.9, 0.8, 0.7, 0.6])

    def test_override_with_unassigned_neurons(self):
        artifact = _make_parameter_snapshot(d_mlp=4)

        def _override(_artifact, _context):
            features = np.eye(4)
            assignment = GroupAssignment(
                assignments=np.array([0, 1, UNASSIGNED, UNASSIGNED], dtype=np.int64),
                n_groups=2,
                method="custom_test",
                feature_basis_name="custom_basis",
                confidence=np.array([0.9, 0.9, 0.4, 0.3]),
            )
            return assignment, features

        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_override": _override})
        assert int(result["n_unassigned"]) == 2


# ── Round-trip via unpack helpers ────────────────────────────────────


class TestUnpackHelpers:
    def test_unpack_assignment_round_trip(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_n_groups": 4})
        assignment = unpack_assignment(result)
        assert isinstance(assignment, GroupAssignment)
        np.testing.assert_array_equal(assignment.assignments, result["assignments"])
        assert assignment.n_groups == int(result["n_groups"])
        assert assignment.method == str(result["method"])
        assert assignment.feature_basis_name == str(result["feature_basis_name"])
        assert assignment.confidence is None

    def test_unpack_summary_round_trip(self):
        artifact = _make_parameter_snapshot()
        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_n_groups": 4})
        summary = unpack_summary(result)
        np.testing.assert_array_equal(summary.centroids, result["centroids"])
        np.testing.assert_array_equal(summary.radii, result["radii"])
        np.testing.assert_array_equal(summary.n_per_group, result["n_per_group"])
        assert summary.n_unassigned == int(result["n_unassigned"])

    def test_unpack_assignment_with_confidence(self):
        artifact = _make_parameter_snapshot(d_mlp=4)

        def _override(_artifact, _context):
            features = np.eye(4)
            return (
                GroupAssignment(
                    assignments=np.array([0, 1, 2, 3], dtype=np.int64),
                    n_groups=4,
                    method="custom",
                    feature_basis_name="custom",
                    confidence=np.array([0.9, 0.8, 0.7, 0.6]),
                ),
                features,
            )

        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_override": _override})
        assignment = unpack_assignment(result)
        assert assignment.confidence is not None
        np.testing.assert_allclose(assignment.confidence, [0.9, 0.8, 0.7, 0.6])


# ── Modadd family Fourier override (real family path) ───────────────


class TestModaddFourierOverride:
    """Exercises the modadd family's Fourier-based grouping override
    end-to-end through the analyzer."""

    def _setup_modadd_artifact_and_context(self, prime: int = 7, seed: int = 42):
        """Build a real modadd parameter_snapshot artifact + context."""
        from pathlib import Path

        from miscope.analysis.library import extract_parameter_snapshot
        from miscope.families.implementations.modulo_addition_1layer import (
            ModuloAddition1LayerFamily,
        )

        # Use the real family.json from the repo's model_families/ tree
        # so the family is configured the way the running pipeline configures
        # it (including the new neuron_grouping in secondary_analyzers).
        repo_root = Path(__file__).resolve().parents[3]
        family_json = repo_root / "model_families" / "modulo_addition_1layer" / "family.json"
        family = ModuloAddition1LayerFamily.from_json(family_json)
        params = {"prime": prime, "seed": seed}
        model = family.create_model(params)
        device = model.cfg.device
        context = family.prepare_analysis_context(params, device)

        # Use the same extraction the real parameter_snapshot analyzer uses,
        # so weight shapes match what the pipeline writes to disk.
        artifact = extract_parameter_snapshot(model)
        return artifact, context, model

    def test_override_present_in_context(self):
        artifact, context, _ = self._setup_modadd_artifact_and_context()
        assert "neuron_grouping_override" in context
        assert callable(context["neuron_grouping_override"])

    def test_override_produces_argmax_by_basis_assignment(self):
        artifact, context, _ = self._setup_modadd_artifact_and_context()
        result = NeuronGrouping().analyze(artifact, context)
        assert str(result["method"]) == "argmax_by_basis"
        assert str(result["feature_basis_name"]) == "fourier_w_in"

    def test_override_n_groups_equals_K_pairs(self):
        """For prime p, K = (p-1)/2 frequency pairs."""
        prime = 7
        artifact, context, _ = self._setup_modadd_artifact_and_context(prime=prime)
        result = NeuronGrouping().analyze(artifact, context)
        assert int(result["n_groups"]) == (prime - 1) // 2

    def test_override_assignment_shape_matches_d_mlp(self):
        artifact, context, model = self._setup_modadd_artifact_and_context()
        result = NeuronGrouping().analyze(artifact, context)
        assert result["assignments"].shape == (model.cfg.d_mlp,)

    def test_override_records_family_override_flag(self):
        artifact, context, _ = self._setup_modadd_artifact_and_context()
        result = NeuronGrouping().analyze(artifact, context)
        assert bool(result["had_family_override"]) is True

    def test_override_emits_confidence(self):
        artifact, context, _ = self._setup_modadd_artifact_and_context()
        result = NeuronGrouping().analyze(artifact, context)
        assert "confidence" in result
        # Untrained model: most neurons will have low confidence, but the
        # confidence values themselves should be in [0, 1].
        confidence = result["confidence"]
        assert (confidence >= 0).all()
        assert (confidence <= 1.0 + 1e-9).all()

    def test_override_threshold_configurable_via_context(self):
        """A very strict threshold should drive most neurons to UNASSIGNED."""
        from miscope.core.grouping import UNASSIGNED

        artifact, context, _ = self._setup_modadd_artifact_and_context()
        # With threshold=0.99, almost no untrained neuron will qualify.
        context_strict = {
            **context,
            "neuron_grouping_confidence_threshold": 0.99,
        }
        result = NeuronGrouping().analyze(artifact, context_strict)
        n_unassigned = int(result["n_unassigned"])
        n_total = int(result["assignments"].shape[0])
        assert n_unassigned > n_total // 2  # majority unassigned

        # And inspecting assignments directly: many should be sentinel.
        assignments = result["assignments"]
        assert int((assignments == UNASSIGNED).sum()) == n_unassigned

    def test_override_lenient_threshold_assigns_more(self):
        """Compared to a strict threshold, a lenient one assigns more neurons."""
        artifact, context, _ = self._setup_modadd_artifact_and_context()
        context_strict = {**context, "neuron_grouping_confidence_threshold": 0.9}
        context_lenient = {**context, "neuron_grouping_confidence_threshold": 0.1}
        n_assigned_strict = int(
            NeuronGrouping().analyze(artifact, context_strict)["assignments"].shape[0]
        ) - int(NeuronGrouping().analyze(artifact, context_strict)["n_unassigned"])
        n_assigned_lenient = int(
            NeuronGrouping().analyze(artifact, context_lenient)["assignments"].shape[0]
        ) - int(NeuronGrouping().analyze(artifact, context_lenient)["n_unassigned"])
        assert n_assigned_lenient >= n_assigned_strict
        artifact = _make_parameter_snapshot(d_mlp=4)

        def _override(_artifact, _context):
            features = np.eye(4)
            return (
                GroupAssignment(
                    assignments=np.array([0, 1, 2, 3], dtype=np.int64),
                    n_groups=4,
                    method="custom",
                    feature_basis_name="custom",
                    confidence=np.array([0.9, 0.8, 0.7, 0.6]),
                ),
                features,
            )

        result = NeuronGrouping().analyze(artifact, context={"neuron_grouping_override": _override})
        assignment = unpack_assignment(result)
        assert assignment.confidence is not None
        np.testing.assert_allclose(assignment.confidence, [0.9, 0.8, 0.7, 0.6])
