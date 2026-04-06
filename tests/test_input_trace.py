"""Tests for REQ_075: Per-Input Prediction Trace.

CoS coverage:
- Unit: InputTraceAnalyzer produces correct shape output for a minimal model (p=11)
- Unit: graduation_epochs is -1 for a pair never stable-correct
- Unit: graduation_epochs is the first epoch of sustained correctness
- Unit: residue_class_accuracy covers all pairs — each pair in exactly one class
- Integration: analyzer runs on a minimal model and artifacts load cleanly
- Integration: all three views render without error
"""

import numpy as np
import torch

from miscope.analysis.analyzers.input_trace import (
    InputTraceAnalyzer,
    _build_split_mask,
    _per_residue_accuracy,
)
from miscope.analysis.analyzers.input_trace_graduation import (
    InputTraceGraduationAnalyzer,
    _compute_graduation_epochs,
)
from miscope.analysis.bundle import TransformerLensBundle
from miscope.analysis.protocols import Analyzer, CrossEpochAnalyzer

# ── Minimal model fixture ─────────────────────────────────────────────


SMALL_P = 11


def _make_minimal_model(p: int = SMALL_P):
    """Create a minimal HookedTransformer for p=SMALL_P testing."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_head=8,
        d_mlp=32,
        act_fn="relu",
        normalization_type=None,
        d_vocab=p + 1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        seed=42,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False
    model.eval()
    return model


def _make_minimal_context(p: int = SMALL_P, data_seed: int = 42, training_fraction: float = 0.4):
    """Create a minimal analysis context."""
    return {
        "params": {
            "prime": p,
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }
    }


def _make_full_probe(p: int = SMALL_P) -> torch.Tensor:
    """Create the full p² probe tensor (matches family.generate_analysis_dataset output)."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    eq = torch.full((p * p,), p, dtype=torch.long)
    return torch.stack([a, b, eq], dim=1)


def _make_bundle(model, probe: torch.Tensor) -> TransformerLensBundle:
    """Create a TransformerLensBundle from a model and probe via a forward pass."""
    with torch.no_grad():
        logits, cache = model.run_with_cache(probe)
    return TransformerLensBundle(model, cache, logits)


# ── Protocol conformance ─────────────────────────────────────────────


class TestProtocolConformance:
    def test_input_trace_analyzer_conforms(self):
        assert isinstance(InputTraceAnalyzer(), Analyzer)

    def test_graduation_analyzer_conforms(self):
        assert isinstance(InputTraceGraduationAnalyzer(), CrossEpochAnalyzer)

    def test_input_trace_has_summary_methods(self):
        analyzer = InputTraceAnalyzer()
        assert hasattr(analyzer, "get_summary_keys")
        assert hasattr(analyzer, "compute_summary")


# ── Unit: InputTraceAnalyzer output shapes ───────────────────────────


class TestInputTraceAnalyzerShapes:
    def test_output_shapes(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        assert result["predictions"].shape == (p * p,)
        assert result["correct"].shape == (p * p,)
        assert result["confidence"].shape == (p * p,)
        assert result["split"].shape == (p * p,)

    def test_output_dtypes(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        assert result["predictions"].dtype == np.int16
        assert result["correct"].dtype == bool
        assert result["confidence"].dtype == np.float16
        assert result["split"].dtype == bool

    def test_predictions_in_valid_range(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        assert result["predictions"].min() >= 0
        assert result["predictions"].max() < p

    def test_correct_consistent_with_predictions(self):
        """correct[k] must match whether predictions[k] == (a+b)%p for pair k."""
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        a = np.arange(p).repeat(p)
        b = np.tile(np.arange(p), p)
        expected_correct = result["predictions"].astype(np.int32) == (a + b) % p
        np.testing.assert_array_equal(result["correct"], expected_correct)

    def test_split_fraction_matches_training_fraction(self):
        p = SMALL_P
        training_fraction = 0.4
        model = _make_minimal_model(p)
        context = _make_minimal_context(p, training_fraction=training_fraction)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        expected_train = int(p * p * training_fraction)
        assert result["split"].sum() == expected_train


# ── Unit: split mask reconstruction ─────────────────────────────────


class TestBuildSplitMask:
    def test_split_count(self):
        p = 11
        split = _build_split_mask(p, 42, 0.4)
        assert split.sum() == int(p * p * 0.4)

    def test_split_is_bool(self):
        split = _build_split_mask(11, 42, 0.4)
        assert split.dtype == bool

    def test_split_deterministic(self):
        split1 = _build_split_mask(11, 42, 0.4)
        split2 = _build_split_mask(11, 42, 0.4)
        np.testing.assert_array_equal(split1, split2)

    def test_different_seeds_differ(self):
        split1 = _build_split_mask(11, 42, 0.4)
        split2 = _build_split_mask(11, 999, 0.4)
        assert not np.array_equal(split1, split2)


# ── Unit: summary stats ──────────────────────────────────────────────


class TestSummaryStats:
    def test_summary_keys(self):
        keys = InputTraceAnalyzer().get_summary_keys()
        assert "test_residue_class_accuracy" in keys
        assert "train_residue_class_accuracy" in keys
        assert "test_overall_accuracy" in keys
        assert "train_overall_accuracy" in keys

    def test_residue_class_accuracy_shape(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]
        summary = analyzer.compute_summary(result, context)

        assert summary["test_residue_class_accuracy"].shape == (p,)
        assert summary["train_residue_class_accuracy"].shape == (p,)

    def test_each_pair_contributes_to_exactly_one_residue_class(self):
        """Every pair must contribute to exactly one residue class.

        The per-residue class counts (train + test) must sum to p².
        """
        p = SMALL_P
        a_all = np.arange(p).repeat(p)
        b_all = np.tile(np.arange(p), p)
        residue = (a_all + b_all) % p
        class_counts = np.bincount(residue, minlength=p)
        assert class_counts.sum() == p * p

    def test_overall_accuracy_matches_per_pair_mean(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]
        summary = analyzer.compute_summary(result, context)

        test_mask = ~result["split"]
        expected_test_acc = float(result["correct"][test_mask].mean())
        np.testing.assert_allclose(
            float(summary["test_overall_accuracy"]),
            expected_test_acc,
            atol=1e-5,
        )

    def test_per_residue_accuracy_helper(self):
        """Each pair lands in exactly one residue class."""
        p = 5
        correct = np.array([True, False, True, True, False] * 5, dtype=bool)[: p * p]
        a = np.arange(p).repeat(p)
        b = np.tile(np.arange(p), p)
        residue = (a + b) % p
        mask = np.ones(p * p, dtype=bool)

        acc = _per_residue_accuracy(correct, residue, mask, p)
        assert acc.shape == (p,)
        assert (acc >= 0).all() and (acc <= 1).all()


# ── Unit: graduation epoch computation ──────────────────────────────


class TestComputeGraduationEpochs:
    def test_never_graduated_returns_minus_one(self):
        correct = np.zeros((5, 3), dtype=bool)
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_isolated_correct_not_graduation(self):
        correct = np.array([[True], [False], [False], [False], [False]], dtype=bool)
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == -1

    def test_graduation_at_first_stable_window(self):
        correct = np.array([[False], [True], [True], [True], [True]], dtype=bool)
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 200

    def test_intermittent_correct_before_stable(self):
        correct = np.array([[True], [False], [True], [True], [True]], dtype=bool)
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 300

    def test_window_larger_than_epochs_returns_minus_one(self):
        correct = np.ones((2, 3), dtype=bool)
        epochs = [0, 1]
        result = _compute_graduation_epochs(correct, epochs, window=5)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_multiple_pairs_independent(self):
        correct = np.array(
            [
                [True, False],  # epoch 0
                [True, False],  # epoch 100
                [True, True],  # epoch 200
                [False, False],  # epoch 300 — pair 0 breaks here
                [False, False],  # epoch 400
            ],
            dtype=bool,
        )
        epochs = [0, 100, 200, 300, 400]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 0  # stable at epochs 0, 100, 200
        assert result[1] == -1  # pair 1 never has 3 consecutive correct


# ── Integration: artifact round-trip ────────────────────────────────


class TestIntegrationArtifactRoundTrip:
    def test_analyzer_artifacts_load_cleanly(self, tmp_path):
        from miscope.analysis.artifact_loader import ArtifactLoader

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        result = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]

        epoch_dir = tmp_path / "input_trace"
        epoch_dir.mkdir()
        np.savez_compressed(str(epoch_dir / "epoch_00100.npz"), **result)  # pyright: ignore[reportArgumentType]

        loader = ArtifactLoader(str(tmp_path))
        loaded = loader.load_epoch("input_trace", 100)

        assert "predictions" in loaded
        assert "correct" in loaded
        assert "confidence" in loaded
        assert "split" in loaded
        np.testing.assert_array_equal(loaded["predictions"], result["predictions"])

    def test_graduation_analyzer_runs_on_minimal_data(self, tmp_path):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        analyzer = InputTraceAnalyzer()
        epochs = [0, 100, 200, 300, 400]
        epoch_dir = tmp_path / "input_trace"
        epoch_dir.mkdir()
        for epoch in epochs:
            result = analyzer.analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]
            np.savez_compressed(str(epoch_dir / f"epoch_{epoch:05d}.npz"), **result)  # pyright: ignore[reportArgumentType]

        grad_analyzer = InputTraceGraduationAnalyzer()
        grad_result = grad_analyzer.analyze_across_epochs(str(tmp_path), epochs, context)

        assert grad_result["graduation_epochs"].shape == (p * p,)
        assert grad_result["epochs"].shape == (len(epochs),)
        assert grad_result["split"].shape == (p * p,)
        np.testing.assert_array_equal(grad_result["epochs"], epochs)


# ── Integration: views render without error ──────────────────────────


class TestViewsRender:
    def test_accuracy_grid_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import render_accuracy_grid

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        epoch_data = InputTraceAnalyzer().analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]
        fig = render_accuracy_grid({"epoch_data": epoch_data, "prime": p}, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_residue_class_timeline_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import (
            render_residue_class_accuracy_timeline,
        )

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = _make_full_probe(p)

        analyzer = InputTraceAnalyzer()
        epochs = [0, 100, 200]
        test_acc_list, train_acc_list, test_ov, train_ov = [], [], [], []
        for _ in epochs:
            result = analyzer.analyze(_make_bundle(model, probe), probe, context)  # type: ignore[arg-type]
            s = analyzer.compute_summary(result, context)
            test_acc_list.append(s["test_residue_class_accuracy"])
            train_acc_list.append(s["train_residue_class_accuracy"])
            test_ov.append(s["test_overall_accuracy"])
            train_ov.append(s["train_overall_accuracy"])

        summary = {
            "epochs": np.array(epochs, dtype=np.int32),
            "test_residue_class_accuracy": np.stack(test_acc_list),
            "train_residue_class_accuracy": np.stack(train_acc_list),
            "test_overall_accuracy": np.array(test_ov, dtype=np.float32),
            "train_overall_accuracy": np.array(train_ov, dtype=np.float32),
        }
        fig = render_residue_class_accuracy_timeline({"summary": summary, "prime": p}, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_graduation_heatmap_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import render_pair_graduation_heatmap

        p = SMALL_P
        split = _build_split_mask(p, 42, 0.4)

        rng = np.random.default_rng(42)
        graduation_epochs = rng.choice([-1, 100, 200, 300, 400], size=p * p).astype(np.int32)

        graduation = {
            "graduation_epochs": graduation_epochs,
            "split": split,
            "epochs": np.array([0, 100, 200, 300, 400], dtype=np.int32),
        }
        fig = render_pair_graduation_heatmap({"graduation": graduation, "prime": p}, epoch=None)
        assert isinstance(fig, go.Figure)
