"""Tests for TwoLayerMLP, MLPActivationBundle, and TwoLayerMLPFamily."""

from __future__ import annotations

import pytest
import torch

from miscope.analysis.protocols import ActivationBundle
from miscope.families.implementations.two_layer_mlp import (
    MLPActivationBundle,
    TwoLayerMLP,
    TwoLayerMLPFamily,
    load_two_layer_mlp_family,
)


P = 13  # Small prime for fast tests


@pytest.fixture
def model() -> TwoLayerMLP:
    return TwoLayerMLP(vocab_size=P, d_hidden=64, seed=42)


@pytest.fixture
def probe() -> torch.Tensor:
    """One-hot encoded (a, b) pairs for all P² inputs."""
    a_vals = torch.arange(P).repeat_interleave(P)
    b_vals = torch.arange(P).repeat(P)
    one_hot_a = torch.zeros(P * P, P)
    one_hot_b = torch.zeros(P * P, P)
    one_hot_a.scatter_(1, a_vals.unsqueeze(1), 1.0)
    one_hot_b.scatter_(1, b_vals.unsqueeze(1), 1.0)
    return torch.cat([one_hot_a, one_hot_b], dim=1)


@pytest.fixture
def bundle(model, probe) -> MLPActivationBundle:
    captured: dict = {}

    def hook(m, inp, output):
        captured["hidden"] = output

    h = model.relu.register_forward_hook(hook)
    with torch.inference_mode():
        logits = model(probe)
    h.remove()
    return MLPActivationBundle(model, captured["hidden"], logits)


@pytest.fixture
def family() -> TwoLayerMLPFamily:
    return load_two_layer_mlp_family("model_families")


@pytest.fixture
def params() -> dict:
    return {"prime": P, "seed": 42, "data_seed": 598}


# ── TwoLayerMLP ──────────────────────────────────────────────────────────────


class TestTwoLayerMLP:
    def test_forward_output_shape(self, model, probe):
        with torch.inference_mode():
            out = model(probe)
        assert out.shape == (P * P, P)

    def test_seedable_initialization(self):
        m1 = TwoLayerMLP(vocab_size=P, d_hidden=64, seed=7)
        m2 = TwoLayerMLP(vocab_size=P, d_hidden=64, seed=7)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = TwoLayerMLP(vocab_size=P, d_hidden=64, seed=1)
        m2 = TwoLayerMLP(vocab_size=P, d_hidden=64, seed=2)
        differs = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert differs

    def test_weight_shapes(self, model):
        assert model.W_in.weight.shape == (64, 2 * P)
        assert model.W_out.weight.shape == (P, 64)

    def test_no_bias(self, model):
        assert model.W_in.bias is None
        assert model.W_out.bias is None


# ── MLPActivationBundle ───────────────────────────────────────────────────────


class TestMLPActivationBundle:
    def test_implements_activation_bundle_protocol(self, bundle):
        assert isinstance(bundle, ActivationBundle)

    def test_mlp_post_layer0(self, bundle, probe):
        hidden = bundle.mlp_post(0, -1)
        assert hidden.shape == (P * P, 64)

    def test_mlp_post_wrong_layer_raises(self, bundle):
        with pytest.raises(ValueError, match="layer 0"):
            bundle.mlp_post(1, -1)

    def test_weight_w_in(self, bundle):
        w = bundle.weight("W_in")
        assert w.shape == (64, 2 * P)

    def test_weight_w_out(self, bundle):
        w = bundle.weight("W_out")
        assert w.shape == (P, 64)

    def test_weight_transformer_names_raise_key_error(self, bundle):
        for name in ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_U"):
            with pytest.raises(KeyError):
                bundle.weight(name)

    def test_attention_pattern_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.attention_pattern(0)

    def test_residual_stream_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.residual_stream(0, -1, "pre")

    def test_logits_shape(self, bundle, probe):
        logits = bundle.logits(-1)
        assert logits.shape == (P * P, P)

    def test_logits_position_ignored(self, bundle):
        """logits() should return the same tensor regardless of position arg."""
        assert torch.equal(bundle.logits(0), bundle.logits(-1))
        assert torch.equal(bundle.logits(0), bundle.logits(99))


# ── TwoLayerMLPFamily ────────────────────────────────────────────────────────


class TestTwoLayerMLPFamily:
    def test_name(self, family):
        assert family.name == "modulo_addition_2layer_mlp"

    def test_create_model_returns_two_layer_mlp(self, family, params):
        model = family.create_model(params)
        assert isinstance(model, TwoLayerMLP)
        assert model.vocab_size == P

    def test_generate_analysis_dataset_shape(self, family, params):
        dataset = family.generate_analysis_dataset(params)
        assert dataset.shape == (P * P, 2 * P)
        assert dataset.dtype == torch.float32

    def test_generate_analysis_dataset_is_one_hot(self, family, params):
        dataset = family.generate_analysis_dataset(params)
        # Each row sums to 2.0 (one 1 in a-half, one 1 in b-half)
        assert torch.allclose(dataset.sum(dim=1), torch.full((P * P,), 2.0))

    def test_generate_training_dataset_shapes(self, family, params):
        train_d, train_l, test_d, test_l, tr_idx, te_idx = family.generate_training_dataset(params)
        total = P * P
        n_train = len(tr_idx)
        n_test = len(te_idx)
        assert n_train + n_test == total
        assert train_d.shape == (n_train, 2 * P)
        assert test_d.shape == (n_test, 2 * P)

    def test_generate_training_dataset_reproducible(self, family, params):
        r1 = family.generate_training_dataset(params)
        r2 = family.generate_training_dataset(params)
        assert torch.equal(r1[4], r2[4])  # train_indices

    def test_run_forward_pass_returns_mlp_bundle(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        assert isinstance(bundle, MLPActivationBundle)

    def test_run_forward_pass_bundle_shapes(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        assert bundle.mlp_post(0, -1).shape == (P * P, 512)
        assert bundle.logits(-1).shape == (P * P, P)

    def test_prepare_analysis_context_has_fourier_basis(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert "fourier_basis" in ctx
        assert "params" in ctx
        assert "loss_fn" in ctx
        assert ctx["fourier_basis"].shape == (P, P)

    def test_prepare_analysis_context_loss_fn(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        ctx = family.prepare_analysis_context(params, "cpu")
        loss = ctx["loss_fn"](model, probe)
        assert isinstance(loss, float)
        assert loss > 0

    def test_make_probe(self, family, params):
        probe = family.make_probe(params, [[3, 5], [0, 12]])
        assert probe.shape == (2, 2 * P)
        # Row 0: a=3 → one-hot[3]=1, b=5 → one-hot[P+5]=1
        assert probe[0, 3] == 1.0
        assert probe[0, P + 5] == 1.0
        assert probe[0].sum() == 2.0

    def test_get_variant_directory_name(self, family, params):
        name = family.get_variant_directory_name(params)
        assert name == f"modulo_addition_2layer_mlp_p{P}_seed42_dseed598"

    def test_analyzers_excludes_transformer_specific(self, family):
        transformer_only = {"attention_freq", "attention_fourier", "attention_patterns", "fourier_nucleation"}
        assert not transformer_only.intersection(set(family.analyzers))

    def test_analyzers_includes_neuron_activations(self, family):
        assert "neuron_activations" in family.analyzers

    def test_analyzers_includes_parameter_snapshot(self, family):
        assert "parameter_snapshot" in family.analyzers

    def test_secondary_analyzers_includes_neuron_fourier(self, family):
        assert "neuron_fourier" in family.secondary_analyzers

    def test_cross_epoch_analyzers_includes_neuron_group_pca(self, family):
        assert "neuron_group_pca" in family.cross_epoch_analyzers
